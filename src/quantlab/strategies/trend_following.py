"""Moving-average trend direction strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantlab.features._validation import boolean, positive_int
from quantlab.features.momentum import ma_crossover_signal
from quantlab.strategies.base import BaseStrategy, register_strategy


@register_strategy("trend_following")
class TrendFollowingStrategy(BaseStrategy):
    """Emit the direction of a trailing fast/slow moving-average crossover.

    Volatility sizing belongs to the portfolio allocator so it is applied once
    and remains consistent with every other strategy.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 100,
        long_only: bool = True,
    ) -> None:
        values = self.validate_parameters(
            {
                "fast_window": fast_window,
                "slow_window": slow_window,
                "long_only": long_only,
            }
        )
        self.fast_window = values["fast_window"]
        self.slow_window = values["slow_window"]
        self.long_only = values["long_only"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate moving-average windows and direction mode."""
        values = dict(parameters)
        values["fast_window"] = positive_int(values["fast_window"], name="fast_window")
        values["slow_window"] = positive_int(values["slow_window"], name="slow_window")
        values["long_only"] = boolean(values["long_only"], name="long_only")
        if values["fast_window"] >= values["slow_window"]:
            raise ValueError("fast_window must be smaller than slow_window.")
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return trailing moving-average crossover directions."""
        prices = self._prices(data)
        signal = ma_crossover_signal(
            prices, fast_window=self.fast_window, slow_window=self.slow_window
        )
        if self.long_only:
            signal = signal.clip(lower=0.0)
        return self._validate_signals(signal, prices)
