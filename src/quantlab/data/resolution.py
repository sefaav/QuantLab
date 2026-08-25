"""Symbol -> source/calendar suggestions, for the dashboard only.

Pure, deterministic, offline heuristics used to pre-fill a form — never
consulted by config validation, the data loader, or the validator. Once a
config is built (YAML or dashboard), it is explicit and fully resolved;
:class:`~quantlab.config.InstrumentConfig` always carries a concrete
source/calendar, never a value inferred here at run time.
"""

from __future__ import annotations

import re

from quantlab.config import DataSourceName

#: Quote assets covering the overwhelming majority of active Binance pairs.
_BINANCE_QUOTE_ASSETS = frozenset(
    {
        "USDT",
        "BUSD",
        "USDC",
        "FDUSD",
        "TUSD",
        "DAI",
        "BTC",
        "ETH",
        "BNB",
        "EUR",
        "GBP",
        "TRY",
        "BRL",
    }
)
_BINANCE_SHAPE = re.compile(r"^[A-Z0-9]{2,20}$")

#: A bare US ticker, or one with a short exchange suffix (e.g. "1211.HK").
_YAHOO_SHAPE = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z]{1,3})?$")

#: Hand-maintained, conservative: an unmapped suffix returns None rather than
#: a wrong guess (e.g. Yahoo suffixes not listed here).
_YAHOO_SUFFIX_CALENDAR: dict[str | None, str] = {
    None: "XNYS",
    "L": "LSE",
    "HK": "XHKG",
    "T": "XTKS",
    "PA": "XPAR",
    "DE": "XFRA",
    "MI": "XMIL",
    "AS": "XAMS",
    "SW": "XSWX",
    "TO": "XTSE",
    "SI": "XSES",
    "AX": "XASX",
    "KS": "XKRX",
}


def detect_source(symbol: str) -> DataSourceName | None:
    """Best-effort guess at a symbol's data source, or ``None`` if unsure.

    Never returns ``csv``: nothing about a bare ticker string implies a
    local file, so a csv-sourced instrument always needs an explicit choice.
    """
    candidate = symbol.strip().upper()
    if not candidate:
        return None
    if _BINANCE_SHAPE.match(candidate) and any(
        candidate.endswith(quote) and len(candidate) > len(quote) + 1
        for quote in _BINANCE_QUOTE_ASSETS
    ):
        return DataSourceName.BINANCE
    if _YAHOO_SHAPE.match(candidate):
        return DataSourceName.YAHOO
    return None


def detect_calendar(symbol: str, source: DataSourceName) -> str | None:
    """Best-effort guess at a symbol's calendar given its (resolved) source."""
    if source is DataSourceName.BINANCE:
        return "24/7"
    if source is DataSourceName.YAHOO:
        candidate = symbol.strip().upper()
        suffix = candidate.split(".", 1)[1] if "." in candidate else None
        return _YAHOO_SUFFIX_CALENDAR.get(suffix)
    return None
