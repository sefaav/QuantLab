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

#: Map QuantLab frequency strings to yfinance ``interval`` values.
_INTERVAL = {"1d": "1d", "1h": "1h", "1w": "1wk", "1mo": "1mo"}

#: Yahoo's unofficial, public, unauthenticated symbol-search endpoint. Used
#: only for dashboard autocomplete, not for downloading price data.
_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
_SEARCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantLab/1.0)"}


class YahooFinanceDataSource(MarketDataSource):
    """Download market data from Yahoo Finance.

    Args:
        max_retries: Number of attempts per symbol before giving up.
        retry_backoff_seconds: Base sleep between retries (linear backoff).
        session: Optional pre-built ``requests.Session``, used only by
            :meth:`search_symbols` (injected in tests). Downloading price data
            goes through ``yfinance`` instead.
    """

    name = "yahoo"

    def __init__(
        self,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        session: requests.Session | None = None,
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
        self.session: requests.Session = (
            session if session is not None else requests.Session()
        )

    def search_symbols(
        self, query: str, max_results: int = 8
    ) -> list[SymbolSuggestion]:
        """Search Yahoo Finance's symbol directory for ``query``.

        For dashboard autocomplete only: returns an empty list (rather than
        raising) on any network or parsing failure.

        Args:
            query: Free-text ticker or company-name fragment.
            max_results: Maximum number of matches to return.

        Returns:
            Matching symbols, most relevant first per Yahoo's own ranking.
        """
        query = query.strip()
        if not query:
            return []
        params: dict[str, str | int] = {
            "q": query,
            "quotesCount": max_results,
            "newsCount": 0,
            "listsCount": 0,
        }
        try:
            response = self.session.get(
                _SEARCH_URL,
                params=params,
                headers=_SEARCH_HEADERS,
                # (connect, read): a single float applies to *each* phase, so
                # a plain `timeout=10` can take ~20s worst case before giving
                # up — too long for an interactive dashboard search box.
                timeout=(3, 5),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Yahoo symbol search failed for %r: %s", query, exc)
            return []
        quotes = payload.get("quotes") if isinstance(payload, dict) else None
        if not isinstance(quotes, list):
            return []
        suggestions: list[SymbolSuggestion] = []
        for quote in quotes:
            if not isinstance(quote, dict):
                continue
            symbol = quote.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            name = quote.get("shortname") or quote.get("longname") or ""
            exchange = quote.get("exchDisp") or quote.get("exchange") or ""
            description = " · ".join(part for part in (name, exchange) if part)
            suggestions.append(
                SymbolSuggestion(symbol=symbol.strip().upper(), description=description)
            )
        return suggestions[:max_results]

    def download(
        self,
        symbols: list[str],
        start: date,
        end: date,
        frequency: str = "1d",
        *,
        calendar: str = "XNYS",
    ) -> pd.DataFrame:
        """Download and normalise data for ``symbols``.

        Args:
            symbols: Yahoo tickers.
            start: Inclusive start date.
            end: Inclusive end date.
            frequency: One of ``1d``, ``1h``, ``1w``, ``1mo``.
            calendar: Accepted for interface parity with other sources but
                unused: the returned frame is never filtered by settlement
                here. Two experiments can request the same Yahoo symbol under
                different calendars, and the download cache (keyed only by
                source/symbol/frequency, with no calendar component) must
                stay identical either way — settlement-dependent filtering
                (which bars are still forming, and which fall within the
                requested end date) is applied only after any cache read/
                write, by :class:`~quantlab.data.loader.DataLoader` (see
                :meth:`~quantlab.data.storage.ParquetStorage.write_symbol`'s
                own docstring). Compare
                :class:`~quantlab.data.binance.BinanceDataSource`, which
                likewise accepts but ignores ``calendar``.

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
        if not isinstance(calendar, str) or not calendar.strip():
            raise DataDownloadError("calendar must be a non-empty string.")

        if not isinstance(frequency, str):
            raise DataDownloadError("Yahoo frequency must be a string.")
        interval = _INTERVAL.get(frequency)
        if interval is None:
            raise DataDownloadError(
                f"Unsupported frequency '{frequency}' for Yahoo. "
                f"Supported: {sorted(_INTERVAL)}."
            )

        frames: list[pd.DataFrame] = []
        failures: dict[str, str] = {}
        for symbol in normalised_symbols:
            try:
                frames.append(self._download_one(symbol, start, end, interval))
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
                        return self._normalise(raw, symbol, interval)
                    except Exception as exc:
                        raise DataDownloadError(
                            f"Yahoo returned an invalid schema for {symbol}: {exc}"
                        ) from exc
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
    ) -> pd.DataFrame:
        """Convert one Yahoo frame to canonical OHLCV.

        Returns every row Yahoo provided, including a bar for a still-forming
        session -- settlement-dependent filtering (which bars are still
        forming, and which fall within the requested end date) is applied
        only after any cache read/write, by
        :class:`~quantlab.data.loader.DataLoader`. Baking a settlement
        opinion in here, before the frame is ever cached, would make the
        cache's on-disk content depend on which calendar the first caller to
        fill it happened to request, even though the cache key (source/
        symbol/frequency) carries no calendar component -- see
        :meth:`~quantlab.data.storage.ParquetStorage.write_symbol`'s own
        docstring for the invariant this preserves.
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
        if interval in {"1d", "1wk", "1mo"}:
            # A calendar-date granularity (daily/weekly/monthly) has no
            # meaningful time-of-day -- the timestamp represents a *local*
            # trading date, not a specific UTC instant. Converting through
            # UTC first (as the intraday branch below correctly does) would
            # shift that date for any exchange ahead of UTC (e.g. XASX
            # +10/+11, XHKG +8): a tz-aware Yahoo response of, say, Monday
            # 00:00 AEDT becomes Sunday 13:00 UTC, silently turning a
            # Monday session into "Sunday". Stripping the timezone directly
            # keeps the local wall-clock date unchanged; a no-op when Yahoo
            # already returns naive daily timestamps.
            bar_timestamps = pd.to_datetime(df[ts_col]).dt.tz_localize(None)
        else:
            # Intraday: the timestamp is a genuine instant, so the UTC
            # conversion below is correct and required (see
            # test_yahoo_intraday_timezone_converted_to_utc_not_stripped_naively).
            bar_timestamps = pd.to_datetime(df[ts_col], utc=True).dt.tz_localize(None)
        out = pd.DataFrame(
            {
                TIMESTAMP: bar_timestamps,
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
        return out
