"""Versioned, Parquet-backed market-data storage and download cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from pandas.tseries.offsets import CustomBusinessDay

from quantlab.constants import CACHE_DIR, METADATA_DIR, SYMBOL, TIMESTAMP
from quantlab.data.base import ensure_canonical_schema
from quantlab.data.calendar import (
    DAILY_FREQUENCIES,
    FREQUENCY_TIMEDELTA,
    MONTHLY_FREQUENCIES,
    PERIODIC_FREQUENCIES,
    bar_bucket_end,
    business_day_offset,
    first_trading_day_on_or_after,
    is_247,
    last_trading_day_on_or_before,
    monthly_bucket_settlement,
    session_labels,
    sessions,
    weekly_bucket_settlement,
    weekly_bucket_start,
)
from quantlab.exceptions import DataValidationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

# Increment when cached data or its normalization contract changes. A new
# namespace prevents older files from silently surviving a code-level fix.
# v3: adds the per-row `_fetched_at` provenance column (see
# `_FETCHED_AT_COLUMN`) -- a v2 file has no such column and no way to
# retrofit one honestly, so it must never be silently reused as if its
# bars' fetch-vs-settlement timing were known.
_CACHE_FORMAT_VERSION = "v3"
_DAILY_POSTING_LAG = pd.Timedelta(hours=12)
_INTRADAY_POSTING_LAG = pd.Timedelta(minutes=30)
_PERIODIC_POSTING_LAG = pd.Timedelta(hours=12)
_LOCK_TIMEOUT_SECONDS = 30.0

# Internal-only column (never part of the canonical OHLCV schema, always
# stripped before data leaves this module): the wall-clock instant each row
# was last written by `write_symbol`. Tracked per row, not per file --
# `write_symbol` merges incoming rows into a file that may already hold
# older rows fetched at a different time, and a later write touching only
# an unrelated date range must not be mistaken for having refreshed every
# row in the file (see `_frame_covers`).
_FETCHED_AT_COLUMN = "_fetched_at"


def _utc_now() -> pd.Timestamp:
    """Return the current instant as timezone-naive UTC."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _has_internal_gap(
    timestamps: pd.Series,
    expected_step: pd.Timedelta | CustomBusinessDay,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> bool:
    """Return whether an expected timestamp is absent inside the range."""
    if range_start > range_end:
        return False
    expected = pd.date_range(range_start, range_end, freq=expected_step)
    present = pd.DatetimeIndex(pd.to_datetime(timestamps))
    return not expected.isin(present).all()


def _has_internal_month_gap(
    timestamps: pd.Series, range_start: pd.Timestamp, range_end: pd.Timestamp
) -> bool:
    """Return whether a touched calendar month contains no bar."""
    if range_start > range_end:
        return False
    expected = pd.period_range(
        range_start.to_period("M"), range_end.to_period("M"), freq="M"
    )
    present = set(pd.to_datetime(timestamps).dt.to_period("M"))
    return any(period not in present for period in expected)


def _has_internal_week_gap(
    timestamps: pd.Series,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    *,
    calendar: str,
) -> bool:
    """Return whether a touched trading week (``calendar``'s own week) has no bar.

    Buckets by :func:`~quantlab.data.calendar.weekly_bucket_start`, not a
    fixed Monday-Sunday ``Period("W")`` -- a calendar whose trading week
    isn't Western (e.g. XSAU, Sunday-Thursday) has its Sunday session
    grouped with the *previous* ISO week by ``.to_period("W")``, splitting
    one real trading week's coverage across two periods and reporting a
    perfectly complete cache as having an internal gap.
    """
    if range_start > range_end:
        return False
    expected_starts: set[pd.Timestamp] = set()
    cursor = weekly_bucket_start(range_start, calendar=calendar)
    last_bucket = weekly_bucket_start(range_end, calendar=calendar)
    while cursor <= last_bucket:
        expected_starts.add(cursor)
        cursor = cursor + pd.Timedelta(days=7)
    present = {
        weekly_bucket_start(pd.Timestamp(value), calendar=calendar)
        for value in pd.to_datetime(timestamps)
    }
    return any(bucket not in present for bucket in expected_starts)


_HASH_SUFFIX_SHAPE = re.compile(r"-[0-9a-f]{10}$")


def _safe(token: str) -> str:
    """Encode an arbitrary token as a deterministic filename component.

    Safe tokens remain unchanged. Tokens needing replacement
    receive a hash suffix, and hash-shaped raw tokens are kept disjoint from
    those generated names.
    """
    sanitized = "".join(c if c.isalnum() or c in "-._" else "_" for c in token)
    if sanitized == token and not _HASH_SUFFIX_SHAPE.search(token):
        return sanitized
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]
    return f"{sanitized}-{digest}"


def _metadata_component(name: str) -> str:
    """Return a bounded, case-safe component for free-form metadata names."""
    base = _safe(name).rstrip(". ")[:96] or "metadata"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{base}-{digest}"


def _sanitize_non_finite_floats(value: object) -> object:
    """Recursively replace non-finite Python and NumPy floats with ``None``."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _sanitize_non_finite_floats(value.item())
    if isinstance(value, dict):
        return {k: _sanitize_non_finite_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_non_finite_floats(v) for v in value]
    return value


def _drop_still_open_bars(
    data: pd.DataFrame, frequency: str, *, calendar: str
) -> pd.DataFrame:
    """Exclude bars not yet safely final.

    A bar isn't just excluded while its own trading bucket is still open
    (``bucket_end > now``) -- it's excluded until ``_posting_lag_for``'s own
    tolerance has *also* elapsed (``bucket_end + posting_lag <= now``), the
    same threshold every coverage check elsewhere in this module already
    uses to decide whether a bar is safe to trust. A provider can still
    revise a bar shortly after its bucket closes (this is exactly what the
    posting-lag tolerance exists to accommodate for cache *coverage*
    checks); serving it to a caller the moment the bucket closes, before
    that tolerance has passed, would let a not-yet-finalised value straight
    into a backtest even though the rest of the system doesn't yet
    consider it settled.
    """
    if data.empty:
        return data
    bucket_end = bar_bucket_end(
        pd.to_datetime(data[TIMESTAMP]),
        frequency,
        calendar=calendar,
    )
    safe_at = bucket_end + _posting_lag_for(frequency)
    return data.loc[safe_at <= _utc_now()].reset_index(drop=True)


def _latest_safe_daily_bar_date(now: pd.Timestamp, *, calendar: str) -> pd.Timestamp:
    """Return the date of the latest daily bar expected to be final."""
    safe_cutoff = now - _DAILY_POSTING_LAG
    if is_247(calendar):
        return safe_cutoff.normalize() - pd.Timedelta(days=1)

    schedule = sessions(
        calendar,
        safe_cutoff.normalize() - pd.Timedelta(days=60),
        safe_cutoff.normalize(),
    )
    eligible = schedule[schedule["market_close"] <= safe_cutoff]
    if eligible.empty:
        return safe_cutoff.normalize() - pd.Timedelta(days=60)
    return pd.Timestamp(eligible.index[-1]).normalize()


def _daily_cache_covers(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    calendar: str,
    now: pd.Timestamp,
) -> bool:
    effective_end = min(end, _latest_safe_daily_bar_date(now, calendar=calendar))
    if effective_end < start:
        return True

    expected_first = first_trading_day_on_or_after(start, calendar=calendar)
    expected_last = last_trading_day_on_or_before(effective_end, calendar=calendar)
    if timestamps.min() > expected_first + _DAILY_POSTING_LAG:
        return False
    if timestamps.max() < expected_last - _DAILY_POSTING_LAG:
        return False
    step = pd.Timedelta(days=1) if is_247(calendar) else business_day_offset(calendar)
    return not _has_internal_gap(timestamps, step, expected_first, expected_last)


def _hourly_247_cache_covers(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    now: pd.Timestamp,
) -> bool:
    step = pd.Timedelta(hours=1)
    request_boundary = end + pd.Timedelta(days=1)
    safe_boundary = (now - _INTRADAY_POSTING_LAG).floor("h")
    effective_boundary = min(request_boundary, safe_boundary)
    if effective_boundary <= start:
        return True

    expected_last = effective_boundary - step
    if timestamps.min() > start or timestamps.max() < expected_last:
        return False
    return not _has_internal_gap(timestamps, step, start, expected_last)


def _equity_intraday_cache_covers(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    step: pd.Timedelta,
    calendar: str,
    now: pd.Timestamp,
) -> bool:
    """Check every requested session on ``calendar``, including both range edges."""
    safe_cutoff = min(end + pd.Timedelta(days=1), now - _INTRADAY_POSTING_LAG)
    schedule_end = min(end, safe_cutoff.normalize())
    if schedule_end < start:
        return True
    schedule = sessions(calendar, start, schedule_end)
    if schedule.empty:
        return True

    requested = pd.to_datetime(timestamps)
    # Widen by a day on each side before filtering: a calendar whose local
    # session crosses UTC midnight (e.g. XASX, UTC+10/+11) opens its
    # session labeled `start` on the UTC calendar day *before* `start`
    # itself -- a raw `requested >= start` bound would strip that session's
    # genuine early bars before they're ever counted, permanently
    # undercounting every session on such a calendar.
    window = requested[
        (requested >= start - pd.Timedelta(days=1))
        & (requested < end + pd.Timedelta(days=2))
    ]
    # Group by each bar's real trading-session date (see
    # quantlab.data.calendar.session_labels), not a naive UTC calendar-day
    # boundary: a calendar whose local session crosses UTC midnight would
    # otherwise have one real session's bars split across two different
    # "days", undercounting that session and triggering a spurious
    # re-download -- or, the other direction, letting a late bar bleed into
    # the following session's count and mask a real gap there. Extra
    # entries this window pulls in for days outside `[start, schedule_end]`
    # are harmless: the loop below only ever looks up a `session_day`
    # already constrained to that range.
    if window.empty:
        by_day: dict[pd.Timestamp, pd.DatetimeIndex] = {}
    else:
        labels = session_labels(calendar, window)
        by_day = {
            pd.Timestamp(day): pd.DatetimeIndex(values).sort_values()
            for day, values in window.groupby(labels)
        }

    for session_day_raw, row in schedule.iterrows():
        session_day = pd.Timestamp(str(session_day_raw))
        market_open = pd.Timestamp(row["market_open"])
        market_close = pd.Timestamp(row["market_close"])
        starts = pd.date_range(market_open, market_close, freq=step, inclusive="left")
        # A calendar with an official intraday break (e.g. XHKG's lunch
        # recess) trades in two disjoint intervals, not one continuous
        # [market_open, market_close) block -- a real provider has no bars
        # during the break, so counting it as "expected" would make a
        # genuinely complete cache look incomplete and trigger a spurious
        # re-download every time.
        break_start = row.get("break_start")
        break_start_ts = (
            pd.Timestamp(break_start)
            if break_start is not None and pd.notna(break_start)
            else None
        )
        break_end_ts = (
            pd.Timestamp(row["break_end"]) if break_start_ts is not None else None
        )
        if break_start_ts is not None and break_end_ts is not None:
            starts = starts[(starts < break_start_ts) | (starts >= break_end_ts)]
        settlements = pd.DatetimeIndex(
            [min(bar_start + step, market_close) for bar_start in starts]
        )
        expected_count = int((settlements <= safe_cutoff).sum())
        if expected_count == 0:
            continue

        day_values = by_day.get(pd.Timestamp(session_day).normalize())
        if day_values is None or len(day_values) < expected_count:
            return False
        if len(day_values) > 1:
            series = pd.Series(day_values)
            deltas = series.diff()
            excessive = deltas > step * 1.5
            if (
                excessive.any()
                and break_start_ts is not None
                and break_end_ts is not None
            ):
                # A gap that straddles the official break isn't a real
                # internal gap once the break's own duration is accounted
                # for -- same reasoning as excluding it from expected_count
                # above.
                break_duration = break_end_ts - break_start_ts
                previous = series.shift(1)
                explained = (
                    excessive
                    & (previous <= break_start_ts)
                    & (series >= break_end_ts)
                    & (deltas - break_duration <= step * 1.5)
                )
                excessive = excessive & ~explained
            if excessive.fillna(False).any():
                return False
    return True


def _period_settlement(
    timestamp: pd.Timestamp, frequency: str, *, calendar: str
) -> pd.Timestamp:
    if frequency in MONTHLY_FREQUENCIES:
        return monthly_bucket_settlement(timestamp, calendar=calendar)
    return weekly_bucket_settlement(timestamp, calendar=calendar)


def _previous_period_end(timestamp: pd.Timestamp, frequency: str) -> pd.Timestamp:
    """Return any representative date within the period before ``timestamp``'s.

    Which exact date doesn't matter -- :func:`_period_settlement` resolves
    every date within a period to the same settlement -- only that it lands
    unambiguously one period back. A full 7-day step always does that for a
    weekly cadence regardless of which weekday a calendar's own week starts
    on (e.g. XSAU trades Sunday-Thursday); the previous ``dayofweek``-based
    computation assumed a fixed Monday-Sunday ISO week, which is wrong for
    such a calendar.
    """
    if frequency in MONTHLY_FREQUENCIES:
        month_start = pd.Timestamp(timestamp.year, timestamp.month, 1)
        return month_start - pd.Timedelta(days=1)
    return timestamp.normalize() - pd.Timedelta(days=7)


def _latest_safe_period_date(
    end: pd.Timestamp,
    frequency: str,
    *,
    calendar: str,
    now: pd.Timestamp,
) -> pd.Timestamp:
    candidate = min(end, now.normalize())
    safe_cutoff = now - _PERIODIC_POSTING_LAG
    if _period_settlement(candidate, frequency, calendar=calendar) > safe_cutoff:
        candidate = _previous_period_end(candidate, frequency)
    return candidate


def _periodic_cache_covers(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    *,
    calendar: str,
    now: pd.Timestamp,
) -> bool:
    effective_end = _latest_safe_period_date(
        end,
        frequency,
        calendar=calendar,
        now=now,
    )
    if effective_end < start:
        return True

    effective_start = first_trading_day_on_or_after(start, calendar=calendar)
    if timestamps.min() > effective_start + _PERIODIC_POSTING_LAG:
        return False

    required_settlement = _period_settlement(
        effective_end, frequency, calendar=calendar
    )
    latest_bucket_end = _period_settlement(
        pd.Timestamp(timestamps.max()),
        frequency,
        calendar=calendar,
    )
    if latest_bucket_end < required_settlement:
        return False

    relevant = timestamps[(timestamps >= start) & (timestamps <= effective_end)]
    if relevant.empty:
        return False
    relevant_index = pd.DatetimeIndex(relevant)
    if relevant_index.hasnans:
        raise DataValidationError("Cached bar timestamps must not be missing.")
    relevant_settlements = pd.DatetimeIndex(
        [
            _period_settlement(timestamp, frequency, calendar=calendar)
            for timestamp in relevant_index
        ]
    )
    service_boundary = effective_end + pd.Timedelta(days=1)
    if not (relevant_settlements <= service_boundary).any():
        return False

    if frequency in MONTHLY_FREQUENCIES:
        return not _has_internal_month_gap(timestamps, start, effective_end)
    return not _has_internal_week_gap(
        timestamps, start, effective_end, calendar=calendar
    )


def _posting_lag_for(frequency: str) -> pd.Timedelta:
    """Return the posting-delay tolerance a frequency's coverage check uses.

    The same three lag constants ``_daily_cache_covers``/
    ``_periodic_cache_covers``/``_hourly_247_cache_covers``/
    ``_equity_intraday_cache_covers`` already apply for "how long after
    settlement a provider might still be finalising a bar" -- the per-row
    staleness check in ``_frame_covers`` uses the same tolerance, so a bar
    is force-refreshed only once it has had a fair chance to actually be
    posted, not the instant its bucket closes.
    """
    if frequency in DAILY_FREQUENCIES:
        return _DAILY_POSTING_LAG
    if frequency in PERIODIC_FREQUENCIES:
        return _PERIODIC_POSTING_LAG
    return _INTRADAY_POSTING_LAG


class ParquetStorage:
    """Read and atomically update canonical Parquet market-data files."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        metadata_dir: Path = METADATA_DIR,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = Path(metadata_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _lock(path: Path) -> FileLock:
        return FileLock(f"{path}.lock", timeout=_LOCK_TIMEOUT_SECONDS)

    @staticmethod
    def _atomic_save_unlocked(data: pd.DataFrame, path: Path) -> None:
        """Write beside ``path`` and atomically replace the final file."""
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            data.to_parquet(temporary, index=False, engine="pyarrow")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, data: pd.DataFrame, path: Path) -> Path:
        """Atomically write a frame to Parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(path):
            self._atomic_save_unlocked(data, path)
        logger.info("Wrote %d rows to %s", len(data), path)
        return path

    def load(self, path: Path) -> pd.DataFrame:
        """Read a Parquet frame from ``path``."""
        # PyArrow is a core dependency; pandas selects it automatically.
        return pd.read_parquet(Path(path))

    def _cache_path(self, source: str, symbol: str, frequency: str) -> Path:
        source_key = source.strip().lower()
        symbol_key = symbol.strip().upper()
        frequency_key = frequency.strip().lower()
        return (
            self.cache_dir
            / _CACHE_FORMAT_VERSION
            / _safe(source_key)
            / f"{_safe(symbol_key)}_{_safe(frequency_key)}.parquet"
        )

    def _read_cache_file(self, path: Path, symbol: str) -> pd.DataFrame | None:
        """Read the raw cache file, including the internal `_fetched_at` column.

        A file with no `_fetched_at` column (written directly via
        :meth:`save`, bypassing :meth:`write_symbol` -- production never
        does this) gets it filled with ``NaT``: unknown provenance, treated
        by :meth:`_frame_covers` as stale rather than trusted once a row's
        bucket has safely closed (it offers no proof of a fresh fetch).
        """
        if not path.is_file():
            return None
        try:
            raw = self.load(path)
            canonical = ensure_canonical_schema(raw)
            canonical[_FETCHED_AT_COLUMN] = (
                pd.to_datetime(raw[_FETCHED_AT_COLUMN].to_numpy())
                if _FETCHED_AT_COLUMN in raw.columns
                else pd.NaT
            )
        except Exception as exc:  # pragma: no cover - corrupt cache is rare
            logger.warning("Failed to read cache %s: %s", path, exc)
            return None

        present_symbols = set(canonical[SYMBOL].unique())
        if present_symbols and present_symbols != {symbol}:
            logger.warning(
                "Ignoring cache %s: expected symbol %s, found %s.",
                path,
                symbol,
                sorted(present_symbols),
            )
            return None
        return (
            canonical.drop_duplicates(subset=[TIMESTAMP], keep="last")
            .sort_values(TIMESTAMP)
            .reset_index(drop=True)
        )

    def _read_filtered_with_provenance(
        self, source: str, symbol: str, frequency: str, *, calendar: str
    ) -> pd.DataFrame | None:
        """Settlement-filtered view, still carrying the internal `_fetched_at`.

        Shared by :meth:`read_symbol` (which drops the column before
        returning) and :meth:`read_covered_symbol` (which additionally needs
        it for :meth:`_frame_covers`'s per-row staleness check).
        """
        normalized_symbol = symbol.strip().upper()
        path = self._cache_path(source, normalized_symbol, frequency)
        data = self._read_cache_file(path, normalized_symbol)
        if data is None:
            return None
        return _drop_still_open_bars(data, frequency, calendar=calendar)

    def read_symbol(
        self,
        source: str,
        symbol: str,
        frequency: str,
        *,
        calendar: str,
    ) -> pd.DataFrame | None:
        """Return one cached symbol with provisional bars filtered out.

        Filters the returned view only -- never rewrites the file. Two
        experiments can legitimately share one cache key (same source,
        symbol, frequency) while requesting different calendars (e.g. XNYS
        vs 24/7), which settle a bar's bucket at different instants; if a
        read purged the file using whichever caller happened to ask first,
        one calendar's "still open" opinion could permanently delete a bar
        another calendar had already correctly settled and stored. Leaving
        the file untouched keeps the cache genuinely calendar-independent:
        :meth:`write_symbol` never drops still-open bars either (see its own
        docstring), so the persisted file's content never depends on which
        calendar happened to write last -- only each caller's own read
        applies its own settlement opinion.
        """
        filtered = self._read_filtered_with_provenance(
            source, symbol, frequency, calendar=calendar
        )
        if filtered is None:
            return None
        return filtered.drop(columns=[_FETCHED_AT_COLUMN])

    def write_symbol(
        self,
        data: pd.DataFrame,
        source: str,
        symbol: str,
        frequency: str,
        *,
        calendar: str,
        replace_start: date | None = None,
        replace_end: date | None = None,
    ) -> Path:
        """Atomically merge one symbol into its cache file.

        ``replace_start`` and ``replace_end`` may delimit a freshly downloaded
        interval whose previous cached rows must be removed before merging.

        Deliberately does *not* drop still-open bars before persisting --
        the cache key is source/symbol/frequency only, with no calendar
        component, so filtering the stored file by whichever calendar
        happened to call last would make the same file's on-disk content
        depend on write order between experiments using different
        calendars, silently reintroducing the same calendar dependence
        :meth:`read_symbol` is designed to avoid. A still-open bar is
        instead naturally superseded once its real, closed value is next
        downloaded -- and this is now actually enforced, not merely
        aspirational: every incoming row is stamped with the current instant
        under the internal ``_fetched_at`` column, preserved per row across
        merges (a row untouched by this write keeps its *previous*
        ``_fetched_at``, never inherits this write's), which
        :meth:`_frame_covers` uses to force a redownload of any row whose
        bucket was still open when it was actually fetched and has since
        settled. ``calendar`` is still used, but only to make the
        ``replace_start``/``replace_end`` purge itself calendar-aware (see
        below) -- never to decide which rows are settled/still-open, which
        stays exclusively a read-time decision.
        """
        normalized_symbol = symbol.strip().upper()
        incoming = ensure_canonical_schema(data)
        incoming[_FETCHED_AT_COLUMN] = _utc_now()
        present_symbols = set(incoming[SYMBOL].unique())
        if present_symbols and present_symbols != {normalized_symbol}:
            raise DataValidationError(
                f"Cannot write cache key {normalized_symbol!r} with rows for "
                f"{sorted(present_symbols)}."
            )
        if (replace_start is None) != (replace_end is None):
            raise ValueError("replace_start and replace_end must be provided together.")
        if (
            replace_start is not None
            and replace_end is not None
            and replace_end < replace_start
        ):
            raise ValueError("replace_end must be on or after replace_start.")

        path = self._cache_path(source, normalized_symbol, frequency)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(path):
            existing = self._read_cache_file(path, normalized_symbol)
            if (
                existing is not None
                and replace_start is not None
                and replace_end is not None
            ):
                timestamps = pd.to_datetime(existing[TIMESTAMP])
                lower = pd.Timestamp(replace_start)
                # A naive UTC-midnight `lower` would let a genuine
                # `replace_start` session bar survive a forced replacement
                # for a calendar whose local session opens before UTC
                # midnight of its own label date (e.g. XASX under daylight
                # saving, UTC+11 -- its session dated `replace_start` can
                # open on the previous UTC calendar day), the same crossing
                # `DataLoader._slice_range` already accounts for on read.
                if not is_247(calendar):
                    start_schedule = sessions(calendar, lower, lower)
                    if not start_schedule.empty:
                        lower = min(
                            lower, pd.Timestamp(start_schedule.iloc[0]["market_open"])
                        )
                upper = pd.Timestamp(replace_end) + pd.Timedelta(days=1)
                existing = existing.loc[(timestamps < lower) | (timestamps >= upper)]

            frames = [frame for frame in (existing, incoming) if frame is not None]
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=[TIMESTAMP], keep="last")
                .sort_values(TIMESTAMP)
                .reset_index(drop=True)
            )
            self._atomic_save_unlocked(merged, path)
        logger.info("Wrote %d cached rows to %s", len(merged), path)
        return path

    @staticmethod
    def _frame_covers(
        cached: pd.DataFrame,
        frequency: str,
        start: date,
        end: date,
        *,
        calendar: str,
    ) -> bool:
        """Apply frequency-specific coverage rules to a cached frame.

        Also guards against a stale row anywhere within the requested
        ``[start, end]`` range, not only the newest one in the file: a
        symbol whose frontier keeps advancing (new bars appended over time)
        can leave an older, still-provisional-when-fetched bar buried
        mid-file, never revisited again -- checking only the newest row
        would stop catching it the moment a newer bar arrives. A row counts
        as covering its bucket only when it carries proof (the per-row
        ``_fetched_at`` column ``write_symbol`` always sets, see its own
        docstring) of having been fetched at or after that bucket's safe
        instant (``bucket_end`` plus this frequency's posting-lag tolerance,
        see ``_posting_lag_for``) -- once that instant has passed, a row
        without such proof (a too-early fetch, or unknown provenance from a
        file written directly via :meth:`save`, bypassing
        :meth:`write_symbol`; production never does this) is treated as
        stale rather than trusted. Rows whose own bucket doesn't overlap the
        requested range are ignored: a narrow re-download can't refresh
        them, so counting them here would make the cache look permanently
        incomplete for requests that never touch them.
        """
        if cached.empty:
            return False
        if frequency not in FREQUENCY_TIMEDELTA:
            raise DataValidationError(
                f"Unsupported cache frequency {frequency!r}; expected one of "
                f"{sorted(FREQUENCY_TIMEDELTA)}."
            )
        if end < start:
            raise ValueError("end must be on or after start.")

        timestamps = pd.to_datetime(cached[TIMESTAMP]).sort_values()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        now = _utc_now()

        if _FETCHED_AT_COLUMN in cached.columns:
            fetched_at = pd.to_datetime(
                cached.loc[timestamps.index, _FETCHED_AT_COLUMN]
            )
            bucket_end = bar_bucket_end(timestamps, frequency, calendar=calendar)
            safe_at = bucket_end + _posting_lag_for(frequency)
            # A row's own bucket may sit entirely outside [start, end] (an
            # older bar the frontier has since moved past) -- such a row can
            # never be refreshed by a write that only replaces the requested
            # range, so counting it here would make the cache permanently
            # "not covering" every future request that doesn't happen to
            # touch it too. Scope staleness to buckets overlapping the
            # requested window, using the same day-inclusive widening as the
            # frequency-specific coverage checks below (``end`` is a
            # calendar day, so its bucket can extend past midnight).
            in_requested_range = (timestamps < end_ts + pd.Timedelta(days=1)) & (
                bucket_end > start_ts
            )
            # Once a row's own safe-at instant has passed, only a *confirmed*
            # fetch at or after that instant proves the bar is the settled
            # value, not a provisional one -- a row whose provenance is
            # unknown (``_fetched_at`` missing/``NaT``, e.g. a file written
            # directly via :meth:`save`, bypassing :meth:`write_symbol`;
            # production never does this) offers no such proof and must be
            # treated the same as a row confirmed too-early, not silently
            # trusted.
            verified_by_now = safe_at <= now
            confirmed_fresh = fetched_at.notna() & (fetched_at >= safe_at)
            stale = in_requested_range & verified_by_now & ~confirmed_fresh
            if stale.any():
                return False

        if frequency in DAILY_FREQUENCIES:
            return _daily_cache_covers(
                timestamps,
                start_ts,
                end_ts,
                calendar=calendar,
                now=now,
            )
        if is_247(calendar) and frequency in {"1h", "1H"}:
            return _hourly_247_cache_covers(timestamps, start_ts, end_ts, now=now)
        if frequency in PERIODIC_FREQUENCIES:
            return _periodic_cache_covers(
                timestamps,
                start_ts,
                end_ts,
                frequency,
                calendar=calendar,
                now=now,
            )
        return _equity_intraday_cache_covers(
            timestamps,
            start_ts,
            end_ts,
            step=FREQUENCY_TIMEDELTA[frequency],
            calendar=calendar,
            now=now,
        )

    def read_covered_symbol(
        self,
        source: str,
        symbol: str,
        frequency: str,
        start: date,
        end: date,
        *,
        calendar: str,
    ) -> pd.DataFrame | None:
        """Read a symbol once and return it only when it covers the request."""
        filtered = self._read_filtered_with_provenance(
            source, symbol, frequency, calendar=calendar
        )
        if filtered is None:
            return None
        if not self._frame_covers(filtered, frequency, start, end, calendar=calendar):
            return None
        return filtered.drop(columns=[_FETCHED_AT_COLUMN])

    def cache_covers(
        self,
        source: str,
        symbol: str,
        frequency: str,
        start: date,
        end: date,
        *,
        calendar: str,
    ) -> bool:
        """Return whether the cache contains every safely final requested bar."""
        return (
            self.read_covered_symbol(
                source,
                symbol,
                frequency,
                start,
                end,
                calendar=calendar,
            )
            is not None
        )

    @staticmethod
    def hash_frame(data: pd.DataFrame) -> str:
        """Return a deterministic hash of values, dtypes, order, and index."""
        payload = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
        schema = "|".join(f"{column}:{dtype}" for column, dtype in data.dtypes.items())
        return hashlib.sha256(schema.encode("utf-8") + payload).hexdigest()

    def write_metadata(self, name: str, metadata: dict[str, object]) -> Path:
        """Atomically write a strict-JSON metadata sidecar."""
        path = self.metadata_dir / f"{_metadata_component(name)}.json"
        payload = _sanitize_non_finite_floats(metadata)
        encoded = json.dumps(payload, indent=2, default=str, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with self._lock(path):
            try:
                temporary.write_text(encoded, encoding="utf-8")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return path
