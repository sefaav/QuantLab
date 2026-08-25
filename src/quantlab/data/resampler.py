"""Resample canonical OHLCV data without creating finer bars."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    PRICE_COLUMNS,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.data.base import ensure_canonical_schema
from quantlab.data.calendar import (
    FREQUENCY_TIMEDELTA,
    is_247,
    session_labels,
    weekly_bucket_start,
)
from quantlab.exceptions import DataValidationError


def _sum_known_volume(values: pd.Series) -> float:
    """Sum volume while preserving an all-missing bucket as missing."""
    return float(values.sum(min_count=1))


_AGG: dict[str, str | Callable[[pd.Series], float]] = {
    OPEN: "first",
    HIGH: "max",
    LOW: "min",
    CLOSE: "last",
    ADJUSTED_CLOSE: "last",
    VOLUME: _sum_known_volume,
}

# Left-labelled buckets match the timestamp convention used by the data sources:
# Monday starts a weekly bar and the first calendar day starts a monthly bar.
_FREQ_ALIAS = {
    "1h": "h",
    "1H": "h",
    "1d": "D",
    "1D": "D",
    "1w": "W-MON",
    "1W": "W-MON",
    "1mo": "MS",
    "1M": "MS",
}

# _FREQ_ALIAS's values are for `.resample(rule, label="left", closed="left")`,
# not for `.to_period()` -- the two disagree about what a "W-MON"-anchored
# week even is. `.resample("W-MON", label="left", closed="left")` genuinely
# bins Monday..Sunday (the explicit label/closed override the ambiguity).
# `Period(freq="W-MON")` does not: it means "week *ending* on Monday", i.e.
# Tuesday..Monday -- so grouping by `.to_period("W-MON")` would silently
# split a real Monday..Sunday week into a lone Monday plus a Tuesday..Friday
# remainder the following calendar week. pandas.Period has no anchored
# "month start" frequency distinct from a calendar month either -- 'MS' (a
# resample/date_range-only alias) must become 'M' for to_period(). The
# calendar-aware weekly path below never reaches `_PERIOD_ALIAS` at all
# (see `_resample_by_session`); this mapping is only used for the
# UTC-period path (`calendar` omitted or "24/7") and for monthly targets.
_PERIOD_ALIAS = {**_FREQ_ALIAS, "1mo": "M", "1M": "M"}

#: `resample_ohlcv`'s own aliases for a weekly target.
_WEEKLY_KEYS = frozenset({"1w", "1W"})


def _resample_by_session(
    indexed: pd.DataFrame, target: str, calendar: str
) -> pd.DataFrame:
    """Group by each bar's real trading session instead of a raw UTC period.

    Mirrors :func:`~quantlab.portfolio.rebalancing.rebalance_dates`'s
    calendar-aware grouping technique: each row is first mapped to its own
    session's real label date, then that label is grouped into the target
    period -- so a session that straddles UTC midnight (e.g. XASX under
    daylight saving) is never split across two output bars just because its
    bars happen to fall on two different raw UTC calendar days.

    A weekly target additionally never groups by a fixed Monday-Sunday
    period (``Period(freq="W-SUN")``): a calendar whose trading week isn't
    Western (e.g. XSAU, Sunday-Thursday) has its Sunday session glued to
    the *previous* ISO week instead of the Monday-Thursday sessions it
    actually trades alongside, splitting one real trading week across two
    output bars. :func:`~quantlab.data.calendar.weekly_bucket_start`
    already resolves "which trading week does this date belong to, and
    what's its own label date" using the calendar's own structural
    weekdays (the same weekly-boundary logic `bar_bucket_end` uses for
    cache/gap checks via its sibling `weekly_bucket_settlement`), so
    grouping by it directly -- rather than by a pandas ``Period`` -- is
    correct for every calendar, Western or not.
    """
    session_dates = session_labels(calendar, pd.Series(indexed.index))
    if target in _WEEKLY_KEYS:

        def label_week_start(value: Any) -> pd.Timestamp:
            return weekly_bucket_start(pd.Timestamp(value), calendar=calendar)

        week_starts = session_dates.map(label_week_start)
        aggregated = indexed.groupby(week_starts.to_numpy()).agg(_AGG)
    else:
        periods = pd.DatetimeIndex(session_dates.to_numpy()).to_period(
            _PERIOD_ALIAS[target]
        )
        aggregated = indexed.groupby(periods.to_numpy()).agg(_AGG)
        aggregated.index = pd.DatetimeIndex(
            [period.start_time for period in aggregated.index]
        )
    aggregated.index.name = TIMESTAMP
    return aggregated.sort_index()


def _snap_to_nominal_frequency(observed: pd.Timedelta) -> pd.Timedelta:
    """Classify a raw observed gap as the finest nominal cadence it fits.

    A raw gap between two adjacent bars is not directly comparable to
    :data:`~quantlab.data.calendar.FREQUENCY_TIMEDELTA`'s fixed nominal
    values: a real trading day's next bar can land 1-4 raw calendar days
    later (a weekend, or a weekend plus a holiday), and a real calendar
    month is 28-31 raw days, not a fixed 30. Comparing a literal gap
    directly against a nominal target would misclassify already-daily data
    with a weekend gap (e.g. Friday to Monday, 3 raw days) as coarser than
    daily, or already-monthly data whose two bars are a genuine 31-day month
    apart as coarser than monthly -- rejecting a resample that should be a
    same-frequency no-op. Each band's upper bound comfortably covers its
    cadence's known calendar variability while staying well short of the
    next cadence's own nominal step, so this only ever *widens* what the
    literal minimum would have accepted, never narrows it.
    """
    if observed <= FREQUENCY_TIMEDELTA["1h"]:
        return FREQUENCY_TIMEDELTA["1h"]
    if observed <= pd.Timedelta(days=5):
        return FREQUENCY_TIMEDELTA["1d"]
    if observed <= pd.Timedelta(days=10):
        return FREQUENCY_TIMEDELTA["1w"]
    return FREQUENCY_TIMEDELTA["1mo"]


def _observed_source_step(data: pd.DataFrame) -> pd.Timedelta:
    """Infer the nominal source frequency from the finest observed spacing."""
    candidates: list[pd.Timedelta] = []
    for _, group in data.groupby(SYMBOL, sort=False):
        timestamps = pd.DatetimeIndex(group[TIMESTAMP]).sort_values().unique()
        if len(timestamps) < 2:
            continue
        deltas = pd.Series(timestamps).diff().dropna()
        positive = deltas[deltas > pd.Timedelta(0)]
        if not positive.empty:
            candidates.append(pd.Timedelta(positive.min()))
    if not candidates:
        raise DataValidationError(
            "Cannot infer the source frequency from fewer than two distinct "
            "timestamps for any symbol; pass source_frequency explicitly."
        )
    return _snap_to_nominal_frequency(min(candidates))


def resample_ohlcv(
    data: pd.DataFrame,
    frequency: str,
    *,
    source_frequency: str | None = None,
    calendar: str | None = None,
) -> pd.DataFrame:
    """Aggregate a canonical long OHLCV frame to the same or a coarser frequency.

    Args:
        data: Canonical long OHLCV frame.
        frequency: Supported target frequency.
        source_frequency: Optional declared input frequency. When omitted, the
            finest positive timestamp spacing is inferred per symbol.
        calendar: Optional instrument calendar. Omitted (the default) or
            ``"24/7"`` groups by raw UTC period boundaries, exactly as
            before. Any other calendar groups by each bar's real trading
            session instead (see :func:`~quantlab.data.calendar.
            session_labels`) -- a raw UTC boundary would otherwise split one
            real session's bars across two output bars for a calendar whose
            local session crosses UTC midnight (e.g. XASX, UTC+10/+11).

    Raises:
        DataValidationError: If the schema, frequency, uniqueness, or
            coarsening contract is violated.
    """
    target = str(frequency)
    if target not in _FREQ_ALIAS:
        raise DataValidationError(
            f"Unsupported resampling frequency {frequency!r}; expected one of "
            f"{sorted(_FREQ_ALIAS)}."
        )

    canonical = ensure_canonical_schema(data)
    if canonical.empty:
        return canonical

    duplicate_mask = canonical.duplicated(subset=[TIMESTAMP, SYMBOL], keep=False)
    if duplicate_mask.any():
        raise DataValidationError(
            "Cannot resample data with duplicate (timestamp, symbol) rows; "
            "clean the input first."
        )

    if source_frequency is None:
        source_step = _observed_source_step(canonical)
    else:
        source_key = str(source_frequency)
        if source_key not in FREQUENCY_TIMEDELTA:
            raise DataValidationError(
                f"Unsupported source_frequency {source_frequency!r}; expected one "
                f"of {sorted(FREQUENCY_TIMEDELTA)}."
            )
        source_step = FREQUENCY_TIMEDELTA[source_key]

    target_step = FREQUENCY_TIMEDELTA[target]
    if target_step < source_step:
        raise DataValidationError(
            f"Cannot resample {source_step} source bars to the finer target "
            f"frequency {frequency!r}."
        )

    session_calendar = (
        calendar if calendar is not None and not is_247(calendar) else None
    )
    out_frames: list[pd.DataFrame] = []
    price_columns = [column for column in PRICE_COLUMNS if column in canonical]
    for symbol, group in canonical.groupby(SYMBOL, sort=True):
        indexed = group.set_index(TIMESTAMP).sort_index()
        if session_calendar is not None:
            resampled = _resample_by_session(indexed, target, session_calendar)
        else:
            # This mapping is valid pandas usage; pandas-stubs does not model
            # the mixed string/callable aggregation overload precisely.
            resampled = indexed.resample(
                _FREQ_ALIAS[target], label="left", closed="left"
            ).agg(_AGG)  # type: ignore[arg-type]
        resampled = resampled.dropna(subset=price_columns, how="all")
        resampled[SYMBOL] = symbol
        out_frames.append(resampled.reset_index())

    if not out_frames:
        return canonical.iloc[0:0].copy()
    return ensure_canonical_schema(pd.concat(out_frames, ignore_index=True))
