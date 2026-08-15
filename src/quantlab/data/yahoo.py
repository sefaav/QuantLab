"""Yahoo Finance data source.

Downloads Yahoo instruments via :mod:`yfinance`, retries provider failures and
normalises them to canonical OHLCV. Adjusted close is preserved when supplied;
otherwise a logged raw-close fallback is used. ``yfinance`` is optional.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from math import isfinite
from numbers import Real

import pandas as pd

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
from quantlab.data.base import MarketDataSource, ensure_canonical_schema
from quantlab.data.calendar import (
    daily_equity_bucket_settlement,
    monthly_bucket_settlement,
    weekly_bucket_settlement,
)
from quantlab.exceptions import DataDownloadError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

#: Map QuantLab frequency strings to yfinance ``interval`` values.
_INTERVAL = {"1d": "1d", "1h": "1h", "1w": "1wk", "1mo": "1mo"}

#: Fixed bucket lengths; monthly settlement uses calendar arithmetic.
_INTERVAL_TIMEDELTA: dict[str, pd.Timedelta] = {
    "1d": pd.Timedelta(days=1),
    "1h": pd.Timedelta(hours=1),
    "1wk": pd.Timedelta(weeks=1),
}


class YahooFinanceDataSource(MarketDataSource):
    """Download market data from Yahoo Finance.

    Args:
        max_retries: Number of attempts per symbol before giving up.
        retry_backoff_seconds: Base sleep between retries (linear backoff).
    """

    name = "yahoo"

    def __init__(
        self, max_retries: int = 3, retry_backoff_seconds: float = 1.0
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries <= 0
        ):
            raise ValueError("max_retries must be a positive integer.")
        if (
            isinstance(retry_backoff_seconds, bool)
            or not isinstance(retry_backoff_seconds, Real)
            or not isfinite(float(retry_backoff_seconds))
            or retry_backoff_seconds < 0
        ):
            raise ValueError(
                "retry_backoff_seconds must be a finite non-negative number."
            )
        self.max_retries: int = max_retries
        self.retry_backoff_seconds: float = float(retry_backoff_seconds)

    def download(
        self,
        symbols: list[str],
        start: date,
        end: date,
        frequency: str = "1d",
        *,
        is_247_market: bool = False,
    ) -> pd.DataFrame:
        """Download and normalise data for ``symbols``.

        Args:
            symbols: Yahoo tickers.
            start: Inclusive start date.
            end: Inclusive end date.
            frequency: One of ``1d``, ``1h``, ``1w``, ``1mo``.
            is_247_market: Use continuous UTC settlement instead of XNYS.

        Returns:
            Canonical long OHLCV frame for all successfully downloaded symbols.

        Raises:
            DataDownloadError: If *no* symbol could be downloaded.
        """
        if not isinstance(symbols, list) or not symbols:
            raise DataDownloadError("Yahoo download requires at least one symbol.")
        normalised_symbols: list[str] = []
        for position, symbol in enumerate(symbols):
            if not isinstance(symbol, str) or not symbol.strip():
                raise DataDownloadError(
                    f"Yahoo symbol at position {position} must be a non-empty string."
                )
            normalised_symbols.append(symbol.strip().upper())
        if not isinstance(start, date) or not isinstance(end, date):
            raise DataDownloadError("Yahoo start and end must be date values.")
        if start > end:
            raise DataDownloadError("Yahoo start must be on or before end.")
        if not isinstance(is_247_market, bool):
            raise DataDownloadError("is_247_market must be a boolean.")

        if not isinstance(frequency, str):
            raise DataDownloadError("Yahoo frequency must be a string.")
        interval = _INTERVAL.get(frequency)
        if interval is None:
            raise DataDownloadError(
                f"Unsupported frequency '{frequency}' for Yahoo. "
                f"Supported: {sorted(_INTERVAL)}."
            )

        # Use one cutoff instant for every symbol in this request.
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        frames: list[pd.DataFrame] = []
        failures: dict[str, str] = {}
        for symbol in normalised_symbols:
            try:
                frames.append(
                    self._download_one(symbol, start, end, interval, now, is_247_market)
                )
            except DataDownloadError as exc:
                logger.error("Giving up on %s: %s", symbol, exc)
                failures[symbol] = str(exc)

        if not frames:
            details = "; ".join(
                f"{symbol}: {message}" for symbol, message in failures.items()
            )
            raise DataDownloadError(
                f"Failed to download any symbol from Yahoo: {normalised_symbols}. "
                f"Details: {details}."
            )
        if failures:
            logger.warning("Some symbols failed to download: %s", failures)
        return ensure_canonical_schema(pd.concat(frames, ignore_index=True))

    def _download_one(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str,
        now: pd.Timestamp,
        is_247_market: bool = False,
    ) -> pd.DataFrame:
        """Download a single symbol with retries and normalise it."""
        try:
            import yfinance as yf  # lazy import (optional dependency)
        except ImportError as exc:
            raise DataDownloadError(
                "Yahoo support requires the optional 'yfinance' dependency. "
                'Install it with `python -m pip install -e ".[yahoo]"`.'
            ) from exc

        # yfinance treats ``end`` as exclusive; add a day to include it.
        end_exclusive = end + timedelta(days=1)
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Downloading %s from Yahoo (attempt %d/%d)",
                    symbol,
                    attempt,
                    self.max_retries,
                )
                raw = yf.download(
                    symbol,
                    start=start,
                    end=end_exclusive,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:  # provider/network failures may be transient
                last_error = exc
            else:
                if raw is None or raw.empty:
                    last_error = DataDownloadError(f"Empty response for {symbol}.")
                else:
                    try:
                        normalised = self._normalise(
                            raw, symbol, interval, now, end, is_247_market
                        )
                    except Exception as exc:
                        raise DataDownloadError(
                            f"Yahoo returned an invalid schema for {symbol}: {exc}"
                        ) from exc
                    if normalised.empty:
                        raise DataDownloadError(
                            f"Yahoo returned data for {symbol}, but every bar was "
                            "still forming or settled after the requested end."
                        )
                    return normalised
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_seconds * attempt)
        raise DataDownloadError(
            f"Could not download {symbol} after {self.max_retries} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _normalise(
        raw: pd.DataFrame,
        symbol: str,
        interval: str,
        now: pd.Timestamp,
        end: date,
        is_247_market: bool = False,
    ) -> pd.DataFrame:
        """Convert one Yahoo frame to canonical OHLCV.

        Bars are retained only after their XNYS or 24/7 settlement boundary
        and only when that boundary is inside the requested inclusive range.
        """
        df = raw.copy()
        # Flatten a possible MultiIndex column (field, ticker) → field.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        # Locate the datetime column (named "Date" or "Datetime").
        ts_col = next(
            (c for c in df.columns if str(c).lower() in {"date", "datetime"}),
            df.columns[0],
        )
        if "Adj Close" in df.columns:
            adjusted_column = "Adj Close"
        else:
            adjusted_column = "Close"
            logger.warning(
                "Yahoo response for %s has no adjusted close; using raw close. "
                "Returns may omit distributions or split adjustments.",
                symbol,
            )
        out = pd.DataFrame(
            {
                # Convert aware timestamps to UTC before removing the timezone.
                TIMESTAMP: pd.to_datetime(df[ts_col], utc=True).dt.tz_localize(None),
                SYMBOL: symbol.upper(),
                OPEN: pd.to_numeric(df["Open"], errors="coerce"),
                HIGH: pd.to_numeric(df["High"], errors="coerce"),
                LOW: pd.to_numeric(df["Low"], errors="coerce"),
                CLOSE: pd.to_numeric(df["Close"], errors="coerce"),
                ADJUSTED_CLOSE: pd.to_numeric(df[adjusted_column], errors="coerce"),
                # Preserve missing values for the configured cleaning policy.
                VOLUME: pd.to_numeric(df["Volume"], errors="coerce"),
            }
        )
        if not out.empty:
            timestamps = pd.DatetimeIndex(out[TIMESTAMP])
            if timestamps.hasnans:
                raise ValueError("Yahoo returned a missing or invalid timestamp.")
            if interval == "1mo":
                bar_end = pd.DatetimeIndex(
                    [
                        monthly_bucket_settlement(
                            timestamp, is_247_market=is_247_market
                        )
                        for timestamp in timestamps
                    ]
                )
            elif interval == "1wk":
                bar_end = pd.DatetimeIndex(
                    [
                        weekly_bucket_settlement(timestamp, is_247_market=is_247_market)
                        for timestamp in timestamps
                    ]
                )
            elif interval == "1d" and not is_247_market:
                bar_end = pd.DatetimeIndex(
                    [
                        daily_equity_bucket_settlement(timestamp)
                        for timestamp in timestamps
                    ]
                )
            else:
                bar_end = timestamps + _INTERVAL_TIMEDELTA.get(
                    interval, pd.Timedelta(0)
                )
            end_boundary = pd.Timestamp(end) + pd.Timedelta(days=1)
            out = out[(bar_end <= now) & (bar_end <= end_boundary)].reset_index(
                drop=True
            )
        return out
