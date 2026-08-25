"""Binance OHLCV data source.

Uses the public Binance REST ``klines`` endpoint. Handles API pagination
(1000-candle limit), converts millisecond timestamps, avoids duplicates, and
backs off on rate-limit responses. No API key is required or logged.
Crypto has no corporate actions, so ``adjusted_close == close``.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import requests

from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.data.base import (
    MarketDataSource,
    SymbolSuggestion,
    ensure_canonical_schema,
)
from quantlab.exceptions import DataDownloadError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.binance.com/api/v3/klines"
_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
_INTERVAL = {"1d": "1d", "1h": "1h", "1w": "1w"}
_MAX_LIMIT = 1000  # Binance hard limit per request
_MAX_RETRY_DELAY_SECONDS = 60.0


class BinanceDataSource(MarketDataSource):
    """Download OHLCV candles from Binance.

    Args:
        max_retries: Attempts per request before failing.
        session: Optional pre-built ``requests.Session`` (injected in tests).
    """

    name = "binance"

    def __init__(
        self,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries <= 0
        ):
            raise ValueError("max_retries must be a positive integer.")
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()

    def download(
        self,
        symbols: list[str],
        start: date,
        end: date,
        frequency: str = "1d",
        *,
        calendar: str = "24/7",
    ) -> pd.DataFrame:
        """Download candles for ``symbols`` and normalise to canonical schema.

        ``calendar`` is unused because Binance provides each kline's explicit
        ``close_time`` and every Binance instrument's calendar is always
        ``"24/7"`` (enforced by :class:`~quantlab.config.InstrumentConfig`).
        """
        if not isinstance(symbols, list) or not symbols:
            raise DataDownloadError("Binance download requires at least one symbol.")
        normalised_symbols: list[str] = []
        for position, symbol in enumerate(symbols):
            if not isinstance(symbol, str) or not symbol.strip():
                raise DataDownloadError(
                    f"Binance symbol at position {position} must be a non-empty string."
                )
            normalised_symbols.append(symbol.strip().upper())
        if not isinstance(start, date) or not isinstance(end, date):
            raise DataDownloadError("Binance start and end must be date values.")
        if start > end:
            raise DataDownloadError("Binance start must be on or before end.")
        if not isinstance(frequency, str):
            raise DataDownloadError("Binance frequency must be a string.")

        interval = _INTERVAL.get(frequency)
        if interval is None:
            raise DataDownloadError(
                f"Unsupported frequency '{frequency}' for Binance. "
                f"Supported: {sorted(_INTERVAL)}."
            )
        # Use one cutoff instant for every symbol in this download.
        now_ms = int(time.time() * 1000)
        frames = [
            self._download_one(s, start, end, interval, now_ms)
            for s in normalised_symbols
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise DataDownloadError(
                f"Binance returned no data for {symbols} in [{start}, {end}]."
            )
        return ensure_canonical_schema(pd.concat(frames, ignore_index=True))

    def list_trading_symbols(self) -> list[SymbolSuggestion]:
        """Fetch every currently active Binance spot trading pair.

        For dashboard autocomplete only: returns an empty list (rather than
        raising) on any network or parsing failure. The full list is meant to
        be fetched once and filtered locally per keystroke, not re-fetched on
        every query.
        """
        try:
            response = self.session.get(_EXCHANGE_INFO_URL, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Binance symbol list fetch failed: %s", exc)
            return []
        entries = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []
        suggestions: list[SymbolSuggestion] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("status") != "TRADING":
                continue
            symbol = entry.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            base_asset = entry.get("baseAsset") or ""
            quote_asset = entry.get("quoteAsset") or ""
            description = (
                f"{base_asset}/{quote_asset}" if base_asset and quote_asset else ""
            )
            suggestions.append(
                SymbolSuggestion(symbol=symbol.strip().upper(), description=description)
            )
        return suggestions

    # ------------------------------------------------------------------ #
    def _download_one(
        self, symbol: str, start: date, end: date, interval: str, now_ms: int
    ) -> pd.DataFrame:
        start_ms = _to_millis(start)
        # Binance's endTime is inclusive (open_time <= endTime), so the request
        # bound must stop one millisecond before the day *after* `end` — using
        # the start of that next day would let its own candle slip in as an
        # extra, unrequested day.
        end_ms = _to_millis(end) + 86_400_000 - 1
        rows: list[list] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self._request(symbol, interval, cursor, end_ms)
            if not batch:
                break
            rows.extend(batch)
            last_open = max(int(row[0]) for row in batch)
            next_cursor = last_open + 1
            if next_cursor <= cursor:  # safety against non-advancing cursor
                break
            cursor = next_cursor
            if len(batch) < _MAX_LIMIT:
                break  # reached the end of available data
        if not rows:
            return pd.DataFrame()
        return self._normalise(rows, symbol, now_ms=now_ms, end_ms=end_ms)

    def _request(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> list[list]:
        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": _MAX_LIMIT,
        }
        last_error: Exception = DataDownloadError("No request was attempted.")
        for attempt in range(1, self.max_retries + 1):
            wait = float(attempt)
            try:
                resp = self.session.get(_BASE_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                last_error = exc
            else:
                status = resp.status_code
                if status == 429:
                    last_error = DataDownloadError("Binance rate limit (HTTP 429).")
                    wait = _retry_delay(resp.headers.get("Retry-After"), 2 * attempt)
                elif 500 <= status < 600:
                    last_error = DataDownloadError(
                        f"Binance server error (HTTP {status})."
                    )
                else:
                    try:
                        resp.raise_for_status()
                    except requests.HTTPError as exc:
                        raise DataDownloadError(
                            f"Binance rejected the request for {symbol} "
                            f"with HTTP {status}."
                        ) from exc
                    try:
                        return _validate_klines_payload(resp.json())
                    except (ValueError, DataDownloadError) as exc:
                        last_error = exc

            if attempt < self.max_retries:
                logger.warning(
                    "Binance request for %s failed (%s); retrying in %.1fs.",
                    symbol,
                    last_error,
                    wait,
                )
                time.sleep(wait)
        raise DataDownloadError(
            f"Binance request for {symbol} failed after {self.max_retries} "
            f"attempts: {last_error}"
        )

    @staticmethod
    def _normalise(
        rows: list[list], symbol: str, *, now_ms: int, end_ms: int
    ) -> pd.DataFrame:
        """Convert klines to canonical OHLCV and retain only closed bars.

        A bar must be closed both as of ``now_ms`` and within the request's
        inclusive end-day boundary.
        """
        frame = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "trades",
                "tbav",
                "tqav",
                "ignore",
            ],
        )
        frame = frame.drop_duplicates(subset=["open_time"], keep="last")
        close_time = frame["close_time"].astype("int64")
        frame = frame[(close_time < now_ms) & (close_time <= end_ms)]
        ts = pd.to_datetime(frame["open_time"].astype("int64"), unit="ms")
        close = pd.to_numeric(frame["close"], errors="coerce")
        return pd.DataFrame(
            {
                TIMESTAMP: ts,
                SYMBOL: symbol.upper(),
                OPEN: pd.to_numeric(frame["open"], errors="coerce"),
                HIGH: pd.to_numeric(frame["high"], errors="coerce"),
                LOW: pd.to_numeric(frame["low"], errors="coerce"),
                CLOSE: close,
                ADJUSTED_CLOSE: close,  # no corporate actions in crypto
                VOLUME: pd.to_numeric(frame["volume"], errors="coerce"),
            }
        )


def _retry_delay(value: str | None, fallback: int) -> float:
    """Return a finite, bounded retry delay in seconds."""
    try:
        delay = float(value) if value is not None else float(fallback)
    except (TypeError, ValueError):
        delay = float(fallback)
    if not math.isfinite(delay) or delay < 0:
        delay = float(fallback)
    return min(delay, _MAX_RETRY_DELAY_SECONDS)


def _validate_klines_payload(payload: Any) -> list[list[Any]]:
    """Validate the structural contract of a Binance klines response."""
    if not isinstance(payload, list):
        raise DataDownloadError("Binance returned a non-list klines payload.")

    rows: list[list[Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, (list, tuple)) or len(row) != 12:
            raise DataDownloadError(
                f"Binance kline {index} must contain exactly 12 fields."
            )
        if any(isinstance(row[position], bool) for position in (0, 6)):
            raise DataDownloadError(
                f"Binance kline {index} has an invalid open/close timestamp."
            )
        try:
            int(row[0])
            int(row[6])
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataDownloadError(
                f"Binance kline {index} has an invalid open/close timestamp."
            ) from exc
        rows.append(list(row))
    return rows


def _to_millis(d: date) -> int:
    """Convert a date to a UTC millisecond epoch."""
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
