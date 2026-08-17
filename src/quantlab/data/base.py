"""Market-data source interface and canonical-schema helpers.

Every data source normalises its output to the canonical *long* OHLCV format:
one row per ``(timestamp, symbol)`` with the columns in
:data:`quantlab.constants.OHLCV_COLUMNS`. Downstream stages (features,
strategies, backtest) operate on *wide* matrices (``index=timestamp``,
``columns=symbol``) built with :func:`pivot_field`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import NamedTuple

import pandas as pd

from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    OHLCV_COLUMNS,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.exceptions import DataValidationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)


class SymbolSuggestion(NamedTuple):
    """A single symbol-search match, for dashboard autocomplete display.

    ``description`` is a short human-readable label (e.g. a company name and
    exchange, or a base/quote asset pair) and may be empty when the source
    provides none.
    """

    symbol: str
    description: str


class MarketDataSource(ABC):
    """Abstract base class for market-data providers.

    Concrete sources implement :meth:`download`; the returned frame must already
    be in the canonical long OHLCV schema.
    """

    #: Human-readable source identifier, matched against ``config.data.source``.
    name: str = "base"

    @abstractmethod
    def download(
        self,
        symbols: list[str],
        start: date,
        end: date,
        frequency: str,
        *,
        is_247_market: bool = False,
    ) -> pd.DataFrame:
        """Download raw data and return it in canonical long OHLCV format.

        Args:
            symbols: Tickers to download.
            start: Inclusive start date.
            end: Inclusive end date.
            frequency: Bar frequency, e.g. ``"1d"``.
            is_247_market: Whether bar settlement follows a continuous
                calendar instead of exchange sessions.

        Returns:
            A DataFrame with the columns of
            :data:`quantlab.constants.OHLCV_COLUMNS`.
        """
        raise NotImplementedError


def ensure_canonical_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder a frame to the canonical long OHLCV schema.

    Args:
        data: Candidate long-format market data.

    Returns:
        A copy with exactly the canonical columns in canonical order, with
        ``timestamp`` coerced to timezone-naive UTC and ``symbol`` to string.
        Timezone-aware values are converted to UTC; naive values are assumed
        to already represent UTC and are not shifted.

    Raises:
        DataValidationError: If required columns are missing.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in data.columns]
    if missing:
        raise DataValidationError(
            f"Market data is missing required columns: {missing}. "
            f"Expected canonical schema {OHLCV_COLUMNS}."
        )
    out = data.loc[:, list(OHLCV_COLUMNS)].copy()
    # Convert aware inputs to UTC before removing the timezone. Naive inputs
    # follow QuantLab's canonical convention and are interpreted as UTC.
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True).dt.tz_localize(None)
    out[SYMBOL] = out[SYMBOL].astype("string").astype(str)
    return out


def pivot_field(data: pd.DataFrame, field: str = ADJUSTED_CLOSE) -> pd.DataFrame:
    """Pivot a canonical long frame into a wide ``dates × symbols`` matrix.

    Args:
        data: Canonical long OHLCV frame.
        field: Column to extract (e.g. ``adjusted_close``, ``close``, ``volume``).

    Returns:
        A DataFrame indexed by sorted timestamps with one column per symbol.

    Raises:
        DataValidationError: If ``field`` is absent or timestamp/symbol pairs
            are duplicated.
    """
    if field not in data.columns:
        raise DataValidationError(
            f"Cannot pivot on unknown field '{field}'. Available: {list(data.columns)}."
        )
    duplicate_mask = data.duplicated(subset=[TIMESTAMP, SYMBOL], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise DataValidationError(
            f"Cannot pivot data with {duplicate_count} duplicate "
            "(timestamp, symbol) rows. Clean or validate the source first."
        )
    wide = data.pivot(index=TIMESTAMP, columns=SYMBOL, values=field)
    wide = wide.sort_index()
    wide.columns.name = None
    return wide


def price_matrix(data: pd.DataFrame, adjusted: bool = True) -> pd.DataFrame:
    """Return the price matrix used for return computation.

    Adjusted close is the default because it can represent distributions and
    splits when the data provider supplies it correctly. Custom data sources
    remain responsible for the meaning and quality of that field.
    """
    return pivot_field(data, ADJUSTED_CLOSE if adjusted else CLOSE)


def volume_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return the ``dates × symbols`` volume matrix (for slippage models)."""
    return pivot_field(data, VOLUME)
