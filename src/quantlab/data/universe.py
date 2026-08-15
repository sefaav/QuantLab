"""Ordered and immutable investable-universe definitions.

The built-in lists are static examples, not point-in-time constituent histories.
Reports surface the resulting survivorship-bias limitation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True, init=False)
class Universe:
    """An ordered, immutable and de-duplicated collection of symbols."""

    name: str
    symbols: tuple[str, ...]

    def __init__(self, symbols: Iterable[str], name: str = "custom") -> None:
        normalised = self._normalise(symbols)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "symbols", normalised)

    @staticmethod
    def _normalise(symbols: Iterable[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for position, raw in enumerate(symbols):
            if not isinstance(raw, str):
                raise TypeError(
                    f"Symbol at position {position} must be a string, "
                    f"not {type(raw).__name__}."
                )
            sym = raw.strip().upper()
            if not sym:
                raise ValueError(f"Symbol at position {position} must not be empty.")
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
        if not out:
            raise ValueError("A universe must contain at least one symbol.")
        return tuple(out)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the universe's symbols in order."""
        return iter(self.symbols)

    def __len__(self) -> int:
        """Return the number of symbols in the universe."""
        return len(self.symbols)

    def __repr__(self) -> str:
        """Return an unambiguous, debuggable representation."""
        return f"Universe(name={self.name!r}, n={len(self)}, symbols={self.symbols})"

    @classmethod
    def from_symbols(cls, symbols: Iterable[str], name: str = "custom") -> Universe:
        """Build a universe from an explicit list of symbols."""
        return cls(symbols, name=name)

    @classmethod
    def from_csv(
        cls, path: str | Path, column: str = "symbol", name: str | None = None
    ) -> Universe:
        """Build a universe from a CSV file containing a symbol column."""
        path = Path(path)
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Universe CSV {path} is empty.") from exc
        if column not in frame.columns:
            raise ValueError(
                f"Universe CSV {path} has no {column!r} column. "
                f"Available columns: {list(frame.columns)}."
            )
        return cls(frame[column].tolist(), name=name or path.stem)

    @classmethod
    def crypto_major(cls) -> Universe:
        """Major liquid crypto pairs (Binance naming)."""
        return cls(
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
            name="crypto_major",
        )

    @classmethod
    def us_sector_etfs(cls) -> Universe:
        """A static nine-fund SPDR US sector ETF example."""
        return cls(
            ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
            name="us_sector_etfs",
        )

    @classmethod
    def liquid_multi_asset_etfs(cls) -> Universe:
        """A liquid, multi-asset-class ETF universe."""
        return cls(
            ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"],
            name="liquid_multi_asset_etfs",
        )

    def to_list(self) -> list[str]:
        """Return the symbols as a plain list."""
        return list(self.symbols)
