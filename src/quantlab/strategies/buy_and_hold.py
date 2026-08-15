"""Constant long-signal reference strategy."""

from __future__ import annotations

import pandas as pd

from quantlab.strategies.base import BaseStrategy, register_strategy


@register_strategy("buy_and_hold")
class BuyAndHoldStrategy(BaseStrategy):
    """Emit a long signal wherever an asset has a valid price.

    The allocator and rebalance schedule determine the resulting portfolio;
    for multiple assets this is not necessarily a literal buy-once portfolio.
    """

    def __init__(self) -> None:
        self._freeze_parameters()

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return an all-long signal matrix aligned to prices."""
        prices = self._prices(data)
        signals = prices.notna().astype(float)
        return self._validate_signals(signals, prices)
