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
    XNYS_BUSINESS_DAY,
    bar_bucket_end,
    first_trading_day_on_or_after,
    last_trading_day_on_or_before,
    monthly_bucket_settlement,
    weekly_bucket_settlement,
    xnys_sessions,
)
from quantlab.exceptions import DataValidationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

# Increment when cached data or its normalization contract changes. A new
# namespace prevents older files from silently surviving a code-level fix.
_CACHE_FORMAT_VERSION = "v2"
_DAILY_POSTING_LAG = pd.Timedelta(hours=12)
_INTRADAY_POSTING_LAG = pd.Timedelta(minutes=30)
_PERIODIC_POSTING_LAG = pd.Timedelta(hours=12)
_LOCK_TIMEOUT_SECONDS = 30.0


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
    timestamps: pd.Series, range_start: pd.Timestamp, range_end: pd.Timestamp
) -> bool:
    """Return whether a touched calendar week contains no bar."""
    if range_start > range_end:
        return False
    expected = pd.period_range(
        range_start.to_period("W"), range_end.to_period("W"), freq="W"
    )
    present = set(pd.to_datetime(timestamps).dt.to_period("W"))
    return any(period not in present for period in expected)


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
    data: pd.DataFrame, frequency: str, *, is_247_market: bool = False
) -> pd.DataFrame:
    """Exclude bars whose settlement instant is still in the future."""
    if data.empty:
        return data
    bucket_end = bar_bucket_end(
        pd.to_datetime(data[TIMESTAMP]),
        frequency,
        is_247_market=is_247_market,
    )
    return data.loc[bucket_end <= _utc_now()].reset_index(drop=True)


def _latest_safe_daily_bar_date(
    now: pd.Timestamp, *, is_247_market: bool
) -> pd.Timestamp:
    """Return the date of the latest daily bar expected to be final."""
    safe_cutoff = now - _DAILY_POSTING_LAG
    if is_247_market:
        return safe_cutoff.normalize() - pd.Timedelta(days=1)

    schedule = xnys_sessions(
        safe_cutoff.normalize() - pd.Timedelta(days=60), safe_cutoff.normalize()
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
    is_247_market: bool,
    now: pd.Timestamp,
) -> bool:
    effective_end = min(
        end, _latest_safe_daily_bar_date(now, is_247_market=is_247_market)
    )
    if effective_end < start:
        return True

    expected_first = first_trading_day_on_or_after(start, is_247_market=is_247_market)
    expected_last = last_trading_day_on_or_before(
        effective_end, is_247_market=is_247_market
    )
    if timestamps.min() > expected_first + _DAILY_POSTING_LAG:
        return False
    if timestamps.max() < expected_last - _DAILY_POSTING_LAG:
        return False
    step = pd.Timedelta(days=1) if is_247_market else XNYS_BUSINESS_DAY
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
    now: pd.Timestamp,
) -> bool:
    """Check every requested XNYS session, including both range edges."""
    safe_cutoff = min(end + pd.Timedelta(days=1), now - _INTRADAY_POSTING_LAG)
    schedule_end = min(end, safe_cutoff.normalize())
    if schedule_end < start:
        return True
    schedule = xnys_sessions(start, schedule_end)
    if schedule.empty:
        return True

    requested = pd.to_datetime(timestamps)
    requested = requested[
        (requested >= start) & (requested < end + pd.Timedelta(days=1))
    ]
    by_day = {
        pd.Timestamp(day): pd.DatetimeIndex(values).sort_values()
        for day, values in requested.groupby(requested.dt.normalize())
    }

    for session_day_raw, row in schedule.iterrows():
        session_day = pd.Timestamp(str(session_day_raw))
        market_open = pd.Timestamp(row["market_open"])
        market_close = pd.Timestamp(row["market_close"])
        starts = pd.date_range(market_open, market_close, freq=step, inclusive="left")
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
            deltas = pd.Series(day_values).diff().dropna()
            if (deltas > step * 1.5).any():
                return False
    return True


def _period_settlement(
    timestamp: pd.Timestamp, frequency: str, *, is_247_market: bool
) -> pd.Timestamp:
    if frequency in MONTHLY_FREQUENCIES:
        return monthly_bucket_settlement(timestamp, is_247_market=is_247_market)
    return weekly_bucket_settlement(timestamp, is_247_market=is_247_market)


def _previous_period_end(timestamp: pd.Timestamp, frequency: str) -> pd.Timestamp:
    if frequency in MONTHLY_FREQUENCIES:
        month_start = pd.Timestamp(timestamp.year, timestamp.month, 1)
        return month_start - pd.Timedelta(days=1)
    week_start = timestamp.normalize() - pd.Timedelta(days=timestamp.dayofweek)
    return week_start - pd.Timedelta(days=1)


def _latest_safe_period_date(
    end: pd.Timestamp,
    frequency: str,
    *,
    is_247_market: bool,
    now: pd.Timestamp,
) -> pd.Timestamp:
    candidate = min(end, now.normalize())
    safe_cutoff = now - _PERIODIC_POSTING_LAG
    if (
        _period_settlement(candidate, frequency, is_247_market=is_247_market)
        > safe_cutoff
    ):
        candidate = _previous_period_end(candidate, frequency)
    return candidate


def _periodic_cache_covers(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    *,
    is_247_market: bool,
    now: pd.Timestamp,
) -> bool:
    effective_end = _latest_safe_period_date(
        end,
        frequency,
        is_247_market=is_247_market,
        now=now,
    )
    if effective_end < start:
        return True

    effective_start = first_trading_day_on_or_after(start, is_247_market=is_247_market)
    if timestamps.min() > effective_start + _PERIODIC_POSTING_LAG:
        return False

    required_settlement = _period_settlement(
        effective_end, frequency, is_247_market=is_247_market
    )
    latest_bucket_end = _period_settlement(
        pd.Timestamp(timestamps.max()),
        frequency,
        is_247_market=is_247_market,
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
            _period_settlement(timestamp, frequency, is_247_market=is_247_market)
            for timestamp in relevant_index
        ]
    )
    service_boundary = effective_end + pd.Timedelta(days=1)
    if not (relevant_settlements <= service_boundary).any():
        return False

    if frequency in MONTHLY_FREQUENCIES:
        return not _has_internal_month_gap(timestamps, start, effective_end)
    return not _has_internal_week_gap(timestamps, start, effective_end)


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
        if not path.is_file():
            return None
        try:
            data = ensure_canonical_schema(self.load(path))
        except Exception as exc:  # pragma: no cover - corrupt cache is rare
            logger.warning("Failed to read cache %s: %s", path, exc)
            return None

        present_symbols = set(data[SYMBOL].unique())
        if present_symbols and present_symbols != {symbol}:
            logger.warning(
                "Ignoring cache %s: expected symbol %s, found %s.",
                path,
                symbol,
                sorted(present_symbols),
            )
            return None
        return (
            data.drop_duplicates(subset=[TIMESTAMP], keep="last")
            .sort_values(TIMESTAMP)
            .reset_index(drop=True)
        )

    def read_symbol(
        self,
        source: str,
        symbol: str,
        frequency: str,
        *,
        is_247_market: bool = False,
    ) -> pd.DataFrame | None:
        """Return one cached symbol after purging provisional bars."""
        normalized_symbol = symbol.strip().upper()
        path = self._cache_path(source, normalized_symbol, frequency)
        data = self._read_cache_file(path, normalized_symbol)
        if data is None:
            return None
        filtered = _drop_still_open_bars(data, frequency, is_247_market=is_247_market)
        if filtered.equals(data):
            return filtered

        # Re-read under the lock so a concurrent writer cannot be overwritten
        # by a purge based on an older snapshot.
        with self._lock(path):
            current = self._read_cache_file(path, normalized_symbol)
            if current is None:
                return None
            filtered = _drop_still_open_bars(
                current, frequency, is_247_market=is_247_market
            )
            self._atomic_save_unlocked(filtered, path)
        return filtered

    def write_symbol(
        self,
        data: pd.DataFrame,
        source: str,
        symbol: str,
        frequency: str,
        *,
        is_247_market: bool = False,
        replace_start: date | None = None,
        replace_end: date | None = None,
    ) -> Path:
        """Atomically merge one symbol into its cache file.

        ``replace_start`` and ``replace_end`` may delimit a freshly downloaded
        interval whose previous cached rows must be removed before merging.
        """
        normalized_symbol = symbol.strip().upper()
        incoming = ensure_canonical_schema(data)
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
                upper = pd.Timestamp(replace_end) + pd.Timedelta(days=1)
                existing = existing.loc[(timestamps < lower) | (timestamps >= upper)]

            frames = [frame for frame in (existing, incoming) if frame is not None]
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=[TIMESTAMP], keep="last")
                .sort_values(TIMESTAMP)
                .reset_index(drop=True)
            )
            merged = _drop_still_open_bars(
                merged, frequency, is_247_market=is_247_market
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
        is_247_market: bool,
    ) -> bool:
        """Apply frequency-specific coverage rules to a cached frame."""
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

        if frequency in DAILY_FREQUENCIES:
            return _daily_cache_covers(
                timestamps,
                start_ts,
                end_ts,
                is_247_market=is_247_market,
                now=now,
            )
        if is_247_market and frequency in {"1h", "1H"}:
            return _hourly_247_cache_covers(timestamps, start_ts, end_ts, now=now)
        if frequency in PERIODIC_FREQUENCIES:
            return _periodic_cache_covers(
                timestamps,
                start_ts,
                end_ts,
                frequency,
                is_247_market=is_247_market,
                now=now,
            )
        return _equity_intraday_cache_covers(
            timestamps,
            start_ts,
            end_ts,
            step=FREQUENCY_TIMEDELTA[frequency],
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
        is_247_market: bool = False,
    ) -> pd.DataFrame | None:
        """Read a symbol once and return it only when it covers the request."""
        cached = self.read_symbol(
            source, symbol, frequency, is_247_market=is_247_market
        )
        if cached is None:
            return None
        if not self._frame_covers(
            cached,
            frequency,
            start,
            end,
            is_247_market=is_247_market,
        ):
            return None
        return cached

    def cache_covers(
        self,
        source: str,
        symbol: str,
        frequency: str,
        start: date,
        end: date,
        *,
        is_247_market: bool = False,
    ) -> bool:
        """Return whether the cache contains every safely final requested bar."""
        return (
            self.read_covered_symbol(
                source,
                symbol,
                frequency,
                start,
                end,
                is_247_market=is_247_market,
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
