"""Validation and quality reporting for canonical long OHLCV data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OHLCV_COLUMNS,
    OPEN,
    PRICE_COLUMNS,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.data.calendar import DAILY_FREQUENCIES as _DAILY_FREQUENCIES
from quantlab.data.calendar import FREQUENCY_TIMEDELTA as _FREQUENCY_TIMEDELTA
from quantlab.data.calendar import business_day_offset as _business_day_offset
from quantlab.data.calendar import (
    first_trading_day_on_or_after as _first_trading_day_on_or_after,
)
from quantlab.data.calendar import has_session_break as _has_session_break
from quantlab.data.calendar import holidays_between as _holidays_between
from quantlab.data.calendar import is_247 as _is_247
from quantlab.data.calendar import (
    last_trading_day_on_or_before as _last_trading_day_on_or_before,
)
from quantlab.data.calendar import session_labels as _session_labels
from quantlab.data.calendar import session_weekmask as _session_weekmask
from quantlab.data.calendar import sessions as _sessions
from quantlab.exceptions import DataValidationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

#: Allowed ratio between declared and observed bar spacing.
_FREQUENCY_MISMATCH_TOLERANCE = 1.2

#: Equity data allows a minority of calendar-related spacing exceptions.
_FREQUENCY_MATCH_MINIMUM_FRACTION = 0.70

#: Continuous markets should have almost no spacing exceptions.
_FREQUENCY_MATCH_MINIMUM_FRACTION_247 = 0.90


@dataclass(frozen=True, slots=True)
class MissingPeriod:
    """An abnormal gap attributed to one symbol."""

    symbol: str
    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable representation."""
        return {
            "symbol": self.symbol,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass
class DataQualityReport:
    """Structured source-quality and post-cleaning summary."""

    row_count: int = 0
    raw_row_count: int | None = None
    clean_row_count: int | None = None
    duplicate_count: int = 0
    missing_value_count: dict[str, int] = field(default_factory=dict)
    invalid_price_count: int = 0
    missing_periods: list[MissingPeriod] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Real rows discarded because they fell on a verified closure for their
    #: own symbol's calendar (a data anomaly, not a gap).
    closure_discarded_count: int = 0
    #: Synthetic flat bars inserted for a verified closure with no real row.
    closure_inserted_count: int = 0
    #: Whether the bundled synthetic CSV fallback (``use_bundled_demo_data``)
    #: was actually triggered for at least one instrument -- distinct from
    #: the config merely *enabling* it, which by itself says nothing about
    #: whether local files were actually missing this run.
    bundled_demo_data_used: bool = False

    @property
    def removed_row_count(self) -> int | None:
        """Return the net raw-to-clean row-count delta when both stages are known.

        A net figure, not a count of any single effect: cleaning can both
        remove rows (duplicates, invalid prices, dropped gaps) and add rows
        (verified-closure bars, forward-filled gaps), so this can be zero or
        even negative (more rows added than removed) without that meaning
        "nothing happened." For what specifically happened, use the
        dedicated counts instead: ``duplicate_count``, ``invalid_price_
        count``, ``missing_value_count``, ``closure_inserted_count``,
        ``closure_discarded_count``.
        """
        if self.raw_row_count is None or self.clean_row_count is None:
            return None
        return self.raw_row_count - self.clean_row_count

    @property
    def is_clean(self) -> bool:
        """True when no hard problems and no warnings were found."""
        return (
            self.duplicate_count == 0
            and self.invalid_price_count == 0
            and sum(self.missing_value_count.values()) == 0
            and not self.warnings
        )

    def summary(self) -> str:
        """Return a short human-readable summary."""
        rows = str(self.row_count)
        if self.raw_row_count is not None and self.clean_row_count is not None:
            rows = f"{self.clean_row_count}/{self.raw_row_count} clean/raw"
        lines = [
            f"rows={rows}",
            f"duplicates={self.duplicate_count}",
            f"invalid_prices={self.invalid_price_count}",
            f"missing_values={sum(self.missing_value_count.values())}",
            f"gaps={len(self.missing_periods)}",
            f"warnings={len(self.warnings)}",
        ]
        return "DataQualityReport(" + ", ".join(lines) + ")"

    def to_dict(self) -> dict[str, Any]:
        """Return the representation persisted in metadata and reports."""
        return {
            "row_count": self.row_count,
            "raw_row_count": self.raw_row_count,
            "clean_row_count": self.clean_row_count,
            "removed_row_count": self.removed_row_count,
            "duplicate_count": self.duplicate_count,
            "missing_value_count": dict(self.missing_value_count),
            "invalid_price_count": self.invalid_price_count,
            "missing_periods_count": len(self.missing_periods),
            "missing_periods": [period.to_dict() for period in self.missing_periods],
            "warnings": list(self.warnings),
            "is_clean": self.is_clean,
            "closure_discarded_count": self.closure_discarded_count,
            "closure_inserted_count": self.closure_inserted_count,
            "bundled_demo_data_used": self.bundled_demo_data_used,
        }


class DataValidator:
    """Validate canonical OHLCV data and produce a quality report.

    Args:
        expected_frequency: QuantLab frequency such as ``"1h"`` or ``"1d"``.
            If omitted, spacing is inferred from observed timestamps.
        max_gap_periods: A run of missing expected bars longer than this many
            periods is flagged as an abnormal gap.
        min_coverage_rows: Minimum rows per symbol before a short-coverage
            warning is raised.
        symbol_calendars: Each symbol's own calendar name (``"24/7"`` or any
            ``pandas_market_calendars`` name). Required for every symbol this
            validator will see — gap detection resolves each symbol's own
            calendar rather than assuming one market for the whole dataset.
            Sub-daily bars on a market that is *not* 24/7 legitimately jump
            overnight and across weekends; gap detection tolerates that
            explicitly instead of flagging every session boundary.
    """

    def __init__(
        self,
        expected_frequency: str | None = None,
        *,
        max_gap_periods: int = 5,
        min_coverage_rows: int = 30,
        symbol_calendars: Mapping[str, str],
    ) -> None:
        if (
            expected_frequency is not None
            and expected_frequency not in _FREQUENCY_TIMEDELTA
        ):
            raise ValueError(
                f"Unknown expected_frequency {expected_frequency!r}. "
                f"Supported: {sorted(_FREQUENCY_TIMEDELTA)}."
            )
        for name, value in (
            ("max_gap_periods", max_gap_periods),
            ("min_coverage_rows", min_coverage_rows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(symbol_calendars, Mapping):
            raise TypeError("symbol_calendars must be a mapping of symbol -> calendar.")
        self.expected_frequency = expected_frequency
        self.max_gap_periods = max_gap_periods
        self.min_coverage_rows = min_coverage_rows
        self.symbol_calendars = symbol_calendars

    @staticmethod
    def _prepare_input(data: pd.DataFrame) -> pd.DataFrame:
        """Return a safe canonical copy or raise an actionable schema error."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        missing_columns = [column for column in OHLCV_COLUMNS if column not in data]
        if missing_columns:
            raise DataValidationError(
                f"Market data is missing canonical columns: {missing_columns}."
            )

        out = data.copy()
        timestamps = pd.to_datetime(out[TIMESTAMP], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise DataValidationError(
                "Market-data timestamps must be present and parseable."
            )
        out[TIMESTAMP] = timestamps.dt.tz_localize(None)

        symbols = out[SYMBOL]
        string_symbols = symbols.map(lambda value: isinstance(value, str)).astype(bool)
        invalid_symbol = symbols.isna() | ~string_symbols
        if invalid_symbol.any():
            raise DataValidationError(
                "Market-data symbols must be non-missing strings."
            )
        stripped_symbols = symbols.astype(str).str.strip()
        if stripped_symbols.eq("").any():
            raise DataValidationError("Market-data symbols must not be empty.")
        out[SYMBOL] = stripped_symbols

        numeric_columns = (OPEN, HIGH, LOW, CLOSE, ADJUSTED_CLOSE, VOLUME)
        for column in numeric_columns:
            original = out[column]
            numeric = pd.to_numeric(original, errors="coerce")
            malformed = original.notna() & numeric.isna()
            if malformed.any():
                raise DataValidationError(
                    f"Column {column!r} contains {int(malformed.sum())} "
                    "non-numeric value(s)."
                )
            out[column] = numeric.astype(float)
        return out

    def validate(
        self,
        data: pd.DataFrame,
        *,
        start: date | None = None,
        end: date | None = None,
        strict: bool = False,
        expected_symbols: Sequence[str] | None = None,
    ) -> DataQualityReport:
        """Validate ``data`` and return a :class:`DataQualityReport`.

        Args:
            data: Canonical long OHLCV frame.
            start: Optional requested start date (dates before it are flagged).
            end: Optional requested end date (dates after it are flagged).
            strict: If True, raise on hard invariant violations.
            expected_symbols: Symbols that must remain present after cleaning.

        Returns:
            A populated :class:`DataQualityReport`.

        Raises:
            DataValidationError: In strict mode when a hard invariant fails.
        """
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")
        for name, value in (("start", start), ("end", end)):
            if value is not None and not isinstance(value, date):
                raise TypeError(f"{name} must be a date or None.")
        if start is not None and end is not None and start > end:
            raise ValueError("start must be on or before end.")
        if isinstance(expected_symbols, (str, bytes)):
            raise TypeError("expected_symbols must be a sequence of symbols.")
        prepared = self._prepare_input(data)
        report = DataQualityReport(
            row_count=len(prepared),
            raw_row_count=len(prepared),
            clean_row_count=len(prepared),
        )
        if prepared.empty:
            report.warnings.append("Dataset is empty.")
            if strict:
                raise DataValidationError("Dataset is empty; nothing to validate.")
            return report

        self._check_duplicates(prepared, report, strict)
        self._check_missing_values(prepared, report)
        self._check_prices(prepared, report, strict)
        self._check_ohlc_consistency(prepared, report, strict)
        self._check_sorted(prepared, report)
        self._check_date_range(prepared, report, start, end)
        self._check_expected_symbols(prepared, report, expected_symbols, strict)

        for symbol, group in prepared.groupby(SYMBOL, sort=True):
            self._check_symbol_coverage(str(symbol), group, report, start, end)

        logger.info("Validation complete: %s", report.summary())
        return report

    def check_pre_clean_defects(
        self, data: pd.DataFrame, *, strict: bool = False
    ) -> DataQualityReport:
        """Record defects that deterministic cleaning may remove.

        The loader merges duplicate, missing-value and invalid-price findings
        into the final report. Missing counts are merged by maximum per column
        so a no-op cleaning policy cannot double-count them.
        """
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")
        prepared = self._prepare_input(data)
        report = DataQualityReport(
            row_count=len(prepared),
            raw_row_count=len(prepared),
            clean_row_count=len(prepared),
        )
        if prepared.empty:
            return report
        self._check_duplicates(prepared, report, strict)
        self._check_missing_values(prepared, report)
        count = self._invalid_price_count(prepared)
        report.invalid_price_count = count
        if count:
            if strict:
                raise DataValidationError(
                    f"Found {count} non-positive or non-finite price values. "
                    "Prices must be finite and strictly positive."
                )
            report.warnings.append(
                f"Found {count} non-positive or non-finite price values."
            )
        return report

    def _check_duplicates(
        self, data: pd.DataFrame, report: DataQualityReport, strict: bool
    ) -> None:
        dup_mask = data.duplicated(subset=[TIMESTAMP, SYMBOL], keep=False)
        count = int(dup_mask.sum())
        report.duplicate_count = count
        if not count:
            return
        example = data.loc[dup_mask, SYMBOL].iloc[0]
        if strict:
            raise DataValidationError(
                f"Found {count} duplicate (timestamp, symbol) rows "
                f"(e.g. symbol {example}). "
                f"Run the data cleaning pipeline or inspect the source file."
            )
        report.warnings.append(
            f"Found {count} duplicate (timestamp, symbol) rows (e.g. symbol {example})."
        )

    def _check_missing_values(
        self, data: pd.DataFrame, report: DataQualityReport
    ) -> None:
        na = data[list(OHLCV_COLUMNS)].isna().sum()
        report.missing_value_count = {
            str(col): int(n) for col, n in na.items() if int(n) > 0
        }
        if report.missing_value_count:
            report.warnings.append(
                f"Missing values present: {report.missing_value_count}."
            )

    @staticmethod
    def _invalid_price_count(data: pd.DataFrame) -> int:
        cols = [c for c in PRICE_COLUMNS if c in data.columns]
        if not cols:
            return 0
        values = data[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        invalid = (values <= 0) | np.isinf(values)
        return int(np.nansum(invalid))

    def _check_prices(
        self, data: pd.DataFrame, report: DataQualityReport, strict: bool
    ) -> None:
        # Prices must be finite and strictly positive.
        count = self._invalid_price_count(data)
        report.invalid_price_count += count
        if count:
            if strict:
                raise DataValidationError(
                    f"Found {count} non-positive or non-finite price values. "
                    "Prices must be finite and strictly positive."
                )
            report.warnings.append(
                f"Found {count} non-positive or non-finite price values."
            )
        if VOLUME in data.columns:
            volume = pd.to_numeric(data[VOLUME], errors="coerce").to_numpy(dtype=float)
            invalid_volume = int(((volume < 0) | np.isinf(volume)).sum())
            if invalid_volume:
                report.warnings.append(
                    f"{invalid_volume} negative or non-finite volume values."
                )
                if strict:
                    raise DataValidationError(
                        f"Found {invalid_volume} negative or non-finite volume "
                        "values; volumes must be finite and non-negative."
                    )

    def _check_ohlc_consistency(
        self, data: pd.DataFrame, report: DataQualityReport, strict: bool
    ) -> None:
        highs_ok = (data[HIGH] >= data[[OPEN, CLOSE, LOW]].max(axis=1)).fillna(True)
        lows_ok = (data[LOW] <= data[[OPEN, CLOSE, HIGH]].min(axis=1)).fillna(True)
        bad = int((~(highs_ok & lows_ok)).sum())
        if bad:
            report.invalid_price_count += bad
            report.warnings.append(f"{bad} OHLC-inconsistent rows (high/low bounds).")
            if strict:
                raise DataValidationError(
                    f"Found {bad} rows where high < max(open,close,low) or "
                    f"low > min(open,close,high). Inspect or clean the source."
                )

    def _check_sorted(self, data: pd.DataFrame, report: DataQualityReport) -> None:
        for symbol, group in data.groupby(SYMBOL, sort=False):
            ts = group[TIMESTAMP]
            if not ts.is_monotonic_increasing:
                report.warnings.append(f"Timestamps not sorted for {symbol}.")

    def _check_date_range(
        self,
        data: pd.DataFrame,
        report: DataQualityReport,
        start: date | None,
        end: date | None,
    ) -> None:
        ts = pd.to_datetime(data[TIMESTAMP])
        if start is not None:
            before = int((ts < pd.Timestamp(start)).sum())
            if before:
                report.warnings.append(f"{before} rows before requested start {start}.")
        if end is not None:
            # The requested end date includes its full calendar day.
            after = int((ts >= pd.Timestamp(end) + pd.Timedelta(days=1)).sum())
            if after:
                report.warnings.append(f"{after} rows after requested end {end}.")

    def _check_expected_symbols(
        self,
        data: pd.DataFrame,
        report: DataQualityReport,
        expected_symbols: Sequence[str] | None,
        strict: bool,
    ) -> None:
        """Flag requested symbols with zero usable rows."""
        if not expected_symbols:
            return
        present = set(data[SYMBOL].unique())
        missing = sorted(set(expected_symbols) - present)
        if missing:
            message = (
                f"Requested symbol(s) {missing} have zero rows after "
                "loading/cleaning — the backtest would silently run on a "
                "smaller universe than configured."
            )
            report.warnings.append(message)
            if strict:
                raise DataValidationError(message)

    @staticmethod
    def _session_break(
        calendar: str, ts: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return the (break_start, break_end) of ``ts``'s own session.

        ``None`` when that session has no official intraday break. Only
        meaningful for a calendar :func:`~quantlab.data.calendar.
        has_session_break` already confirmed has one.
        """
        day = ts.normalize()
        schedule = _sessions(calendar, day, day)
        if schedule.empty:
            return None
        row = schedule.iloc[0]
        break_start = row.get("break_start")
        if break_start is None or pd.isna(break_start):
            return None
        return pd.Timestamp(break_start), pd.Timestamp(row["break_end"])

    def _check_symbol_coverage(
        self,
        symbol: str,
        group: pd.DataFrame,
        report: DataQualityReport,
        start: date | None = None,
        end: date | None = None,
    ) -> None:
        calendar = self.symbol_calendars.get(symbol)
        if calendar is None:
            raise DataValidationError(f"No calendar configured for symbol {symbol!r}.")
        is_247_market = _is_247(calendar)
        ts = pd.to_datetime(group[TIMESTAMP]).sort_values()
        deltas = ts.diff().dropna()
        observed_step = deltas.median() if not deltas.empty else pd.Timedelta(0)
        expected_step = (
            _FREQUENCY_TIMEDELTA.get(self.expected_frequency)
            if self.expected_frequency is not None
            else None
        )
        step = expected_step or observed_step
        # Two rows are enough to detect a declared-frequency mismatch.
        if observed_step > pd.Timedelta(0):
            self._check_declared_frequency(symbol, calendar, ts, deltas, report)

        # Continuous markets have no legitimate exchange closures.
        gap_periods = 1 if is_247_market else self.max_gap_periods
        base_tolerance = (
            step * gap_periods if step > pd.Timedelta(0) else pd.Timedelta(0)
        )
        if is_247_market:
            # Make an edge lag of exactly one expected period fail.
            tolerance = max(
                base_tolerance - pd.Timedelta(microseconds=1), pd.Timedelta(0)
            )
        else:
            tolerance = max(base_tolerance, pd.Timedelta(days=1))

        range_start = pd.Timestamp(start) if start is not None else None
        range_end = pd.Timestamp(end) if end is not None else None
        if not is_247_market:
            if range_start is not None:
                range_start = _first_trading_day_on_or_after(
                    range_start, calendar=calendar
                )
            if range_end is not None:
                range_end = _last_trading_day_on_or_before(range_end, calendar=calendar)
        if start is not None and len(ts):
            assert range_start is not None
            lag = ts.iloc[0] - range_start
            if lag > tolerance:
                report.warnings.append(
                    f"{symbol}: data starts {lag} after the requested start "
                    f"{start} (first observation {ts.iloc[0].date()})."
                )
        if end is not None and len(ts):
            assert range_end is not None
            expected_last = range_end
            if pd.Timedelta(0) < step < pd.Timedelta(days=1):
                expected_last += pd.Timedelta(days=1) - step
            lag = expected_last - ts.iloc[-1]
            if lag > tolerance:
                report.warnings.append(
                    f"{symbol}: data ends {lag} before the requested end "
                    f"{end} (last observation {ts.iloc[-1].date()})."
                )

        if len(ts) < self.min_coverage_rows:
            report.warnings.append(
                f"Short coverage for {symbol}: only {len(ts)} rows "
                f"(< {self.min_coverage_rows})."
            )
            # Short coverage and internal gaps are independent findings.
        if step <= pd.Timedelta(0):
            return
        intraday_threshold = step * gap_periods
        if not is_247_market and self.expected_frequency in _DAILY_FREQUENCIES:
            start_days = (
                ts.shift(1).loc[deltas.index].to_numpy().astype("datetime64[D]")
            )
            end_days = ts.loc[deltas.index].to_numpy().astype("datetime64[D]")
            holidays = (
                _holidays_between(calendar, ts.min(), ts.max())
                .to_numpy()
                .astype("datetime64[D]")
            )
            session_steps = pd.Series(
                np.busday_count(
                    start_days,
                    end_days,
                    weekmask=_session_weekmask(calendar),
                    holidays=holidays,
                ),
                index=deltas.index,
            )
            gap_candidates = session_steps > gap_periods
        else:
            gap_candidates = deltas > intraday_threshold
        if not is_247_market and step < pd.Timedelta(days=1) and gap_candidates.any():
            # Evaluate cross-day gaps against this symbol's own calendar
            # sessions, not wall-clock time.
            starts = ts.shift(1).loc[deltas.index]
            ends = ts.loc[deltas.index]

            # Infer typical edge times to detect truncated bordering
            # sessions. Group by each timestamp's real trading session (see
            # session_labels), not a naive UTC calendar-day boundary.
            by_day = ts.groupby(_session_labels(calendar, ts))
            last_by_day = by_day.max().dt.time
            first_by_day = by_day.min().dt.time
            # Ties prefer the widest observed session.
            last_modes = last_by_day.mode()
            first_modes = first_by_day.mode()
            typical_last_time = last_modes.max() if not last_modes.empty else None
            typical_first_time = first_modes.min() if not first_modes.empty else None

            for pos in deltas.index[gap_candidates]:
                start_ts, end_ts = starts.loc[pos], ends.loc[pos]
                start_date, end_date = start_ts.date(), end_ts.date()
                if start_date == end_date:
                    # Same session: genuinely abnormal, UNLESS it's fully
                    # explained by the calendar's own official intraday
                    # break (e.g. XHKG's lunch recess) -- a real provider
                    # legitimately has no bars during it.
                    if _has_session_break(calendar):
                        session_break = self._session_break(calendar, start_ts)
                        if session_break is not None:
                            break_start, break_end = session_break
                            residual = (end_ts - start_ts) - (break_end - break_start)
                            if (
                                start_ts <= break_start
                                and end_ts >= break_end
                                and residual <= intraday_threshold
                            ):
                                gap_candidates.loc[pos] = False
                    continue
                if typical_last_time is not None and typical_first_time is not None:
                    expected_tail_end = pd.Timestamp.combine(
                        start_date, typical_last_time
                    )
                    expected_head_start = pd.Timestamp.combine(
                        end_date, typical_first_time
                    )
                    if (
                        expected_tail_end - start_ts > intraday_threshold
                        or end_ts - expected_head_start > intraday_threshold
                    ):
                        continue  # one side of the boundary is truncated
                skipped = pd.bdate_range(
                    start=start_date + pd.Timedelta(days=1),
                    end=end_date - pd.Timedelta(days=1),
                    freq=_business_day_offset(calendar),
                )
                if len(skipped) == 0:
                    gap_candidates.loc[pos] = False
        gap_positions = deltas[gap_candidates]
        for pos in gap_positions.index:
            end_ts = ts.loc[pos]
            start_ts = ts.shift(1).loc[pos]
            report.missing_periods.append(
                MissingPeriod(
                    symbol=symbol,
                    start=start_ts.to_pydatetime(),
                    end=end_ts.to_pydatetime(),
                )
            )
        if len(gap_positions):
            report.warnings.append(
                f"{len(gap_positions)} abnormal gap(s) for {symbol} "
                f"(at least {gap_periods} missing expected bar(s))."
            )

    def _check_declared_frequency(
        self,
        symbol: str,
        calendar: str,
        ts: pd.Series,
        deltas: pd.Series,
        report: DataQualityReport,
    ) -> None:
        """Compare median, matching fraction and 24/7 mean spacing.

        Daily equity deltas are measured in the symbol's own calendar
        sessions. Intraday equity matching excludes cross-session deltas,
        which are legitimate closures.
        """
        if self.expected_frequency is None or deltas.empty:
            return
        expected = _FREQUENCY_TIMEDELTA.get(self.expected_frequency)
        if expected is None or expected <= pd.Timedelta(0):
            return
        is_247_market = _is_247(calendar)
        daily_equity = not is_247_market and expected == pd.Timedelta(days=1)
        if daily_equity:
            # Count sessions so weekends and holidays have zero duration.
            starts = ts.shift(1).loc[deltas.index].to_numpy().astype("datetime64[D]")
            ends = ts.loc[deltas.index].to_numpy().astype("datetime64[D]")
            holidays = (
                _holidays_between(calendar, ts.min(), ts.max())
                .to_numpy()
                .astype("datetime64[D]")
            )
            ratios = pd.Series(
                np.busday_count(
                    starts,
                    ends,
                    weekmask=_session_weekmask(calendar),
                    holidays=holidays,
                ).astype(float),
                index=deltas.index,
            )
        else:
            ratios = deltas.dt.total_seconds() / expected.total_seconds()
        ratio = ratios.median()
        observed_step = expected * ratio
        tolerance = _FREQUENCY_MISMATCH_TOLERANCE
        # Boundary values are mismatches, not part of the tolerated interior.
        if ratio >= tolerance or ratio <= (1.0 / tolerance):
            report.warnings.append(
                f"{symbol}: observed bar spacing (~{observed_step}) does not "
                f"match the declared frequency '{self.expected_frequency}' "
                f"(~{expected}); annualisation (periods_per_year) and any "
                "frequency-dependent logic are likely wrong for this run."
            )
            return

        is_equity_subdaily = not is_247_market and expected < pd.Timedelta(days=1)
        if is_equity_subdaily:
            # Real trading-session labels, not naive UTC calendar dates (see
            # session_labels): otherwise a session that straddles UTC
            # midnight in local terms (e.g. Sydney, +10/+11) would have its
            # own intraday deltas wrongly excluded as "cross-session".
            labels = _session_labels(calendar, ts)
            same_day = (labels == labels.shift(1)).reindex(
                deltas.index, fill_value=False
            )
            intraday_deltas = deltas[same_day]
            if intraday_deltas.empty:
                return
            if _has_session_break(calendar):
                # A delta that fully spans the calendar's own official
                # intraday break (e.g. XHKG's lunch recess) is not "missing
                # a bar" -- no real provider has one during the break -- so
                # it must not count against the declared-frequency matching
                # fraction either, the same reasoning already applied to
                # abnormal-gap detection above.
                schedule = _sessions(calendar, ts.min(), ts.max())
                break_start_map = schedule.get("break_start", pd.Series(dtype=object))
                break_end_map = schedule.get("break_end", pd.Series(dtype=object))
                row_break_start = labels.map(break_start_map)
                row_break_end = labels.map(break_end_map)
                previous_ts = ts.shift(1)
                explained = (
                    row_break_start.notna()
                    & (previous_ts <= row_break_start)
                    & (ts >= row_break_end)
                )
                intraday_deltas = intraday_deltas[
                    ~explained.reindex(intraday_deltas.index, fill_value=False)
                ]
                if intraday_deltas.empty:
                    return
            intraday_ratios = (
                intraday_deltas.dt.total_seconds() / expected.total_seconds()
            )
            intraday_matching_fraction = (
                (intraday_ratios >= 1.0 / tolerance) & (intraday_ratios <= tolerance)
            ).mean()
            if intraday_matching_fraction <= _FREQUENCY_MATCH_MINIMUM_FRACTION_247:
                report.warnings.append(
                    f"{symbol}: only {intraday_matching_fraction:.0%} of "
                    "*same-day* observed bar spacings actually match the "
                    f"declared frequency '{self.expected_frequency}' "
                    f"(~{expected}) — a bar may be systematically missing "
                    "from partway through each session; annualisation and "
                    "any frequency-dependent logic are likely wrong for a "
                    "meaningful fraction of this run."
                )
            return
        matching_fraction = ((ratios >= 1.0 / tolerance) & (ratios <= tolerance)).mean()
        minimum_fraction = (
            _FREQUENCY_MATCH_MINIMUM_FRACTION_247
            if is_247_market
            else _FREQUENCY_MATCH_MINIMUM_FRACTION
        )
        if is_247_market:
            # The mean catches sparse large gaps that a fraction floor can miss.
            mean_step = cast("pd.Timedelta", deltas.mean())
            mean_ratio = mean_step / expected
            if mean_ratio >= tolerance or mean_ratio <= (1.0 / tolerance):
                report.warnings.append(
                    f"{symbol}: mean observed bar spacing (~{mean_step}) "
                    f"does not match the declared frequency "
                    f"'{self.expected_frequency}' (~{expected}) — the "
                    "median and matching-fraction checks alone did not "
                    "catch this; annualisation (periods_per_year) and any "
                    "frequency-dependent logic are likely wrong for this "
                    "run."
                )
                return
        if matching_fraction <= minimum_fraction:
            report.warnings.append(
                f"{symbol}: only {matching_fraction:.0%} of observed bar "
                f"spacings actually match the declared frequency "
                f"'{self.expected_frequency}' (~{expected}) — the median "
                "spacing looks correct, but a large minority of bars are a "
                "different spacing entirely (e.g. a mix of the declared "
                "step and exactly double it); annualisation and any "
                "frequency-dependent logic are likely wrong for a "
                "meaningful fraction of this run."
            )
