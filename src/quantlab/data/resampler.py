"""Resample canonical OHLCV data without creating finer bars."""

from __future__ import annotations

from collections.abc import Callable

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
from quantlab.data.calendar import FREQUENCY_TIMEDELTA
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


def _observed_source_step(data: pd.DataFrame) -> pd.Timedelta:
    """Infer the finest positive spacing observed within any symbol."""
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
    return min(candidates)


def resample_ohlcv(
    data: pd.DataFrame,
    frequency: str,
    *,
    source_frequency: str | None = None,
) -> pd.DataFrame:
    """Aggregate a canonical long OHLCV frame to the same or a coarser frequency.

    Args:
        data: Canonical long OHLCV frame.
        frequency: Supported target frequency.
        source_frequency: Optional declared input frequency. When omitted, the
            finest positive timestamp spacing is inferred per symbol.

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

    out_frames: list[pd.DataFrame] = []
    price_columns = [column for column in PRICE_COLUMNS if column in canonical]
    for symbol, group in canonical.groupby(SYMBOL, sort=True):
        indexed = group.set_index(TIMESTAMP).sort_index()
        # This mapping is valid pandas usage; pandas-stubs does not model the
        # mixed string/callable aggregation overload precisely.
        resampled = indexed.resample(
            _FREQ_ALIAS[target], label="left", closed="left"
        ).agg(_AGG)  # type: ignore[arg-type]
        resampled = resampled.dropna(subset=price_columns, how="all")
        resampled[SYMBOL] = symbol
        out_frames.append(resampled.reset_index())

    if not out_frames:
        return canonical.iloc[0:0].copy()
    return ensure_canonical_schema(pd.concat(out_frames, ignore_index=True))
