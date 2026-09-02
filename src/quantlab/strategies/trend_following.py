"""Moving-average trend direction strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features._validation import boolean, choice, positive_int
from quantlab.features.momentum import ma_crossover_signal, moving_average
from quantlab.strategies.base import (
    PRICE_TYPES,
    BaseStrategy,
    SignalReasons,
    register_strategy,
    validate_risk_control_parameters,
)


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
        price_type: str = "adjusted_close",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        values = self.validate_parameters(
            {
                "fast_window": fast_window,
                "slow_window": slow_window,
                "long_only": long_only,
                "price_type": price_type,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        )
        self.fast_window = values["fast_window"]
        self.slow_window = values["slow_window"]
        self.long_only = values["long_only"]
        self.price_type = values["price_type"]
        self.stop_loss_pct = values["stop_loss_pct"]
        self.take_profit_pct = values["take_profit_pct"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate moving-average windows and direction mode."""
        values = dict(parameters)
        values["fast_window"] = positive_int(values["fast_window"], name="fast_window")
        values["slow_window"] = positive_int(values["slow_window"], name="slow_window")
        values["long_only"] = boolean(values["long_only"], name="long_only")
        values["price_type"] = choice(
            values["price_type"], name="price_type", options=PRICE_TYPES
        )
        if values["fast_window"] >= values["slow_window"]:
            raise ValueError("fast_window must be smaller than slow_window.")
        values["stop_loss_pct"], values["take_profit_pct"] = (
            validate_risk_control_parameters(
                values["stop_loss_pct"], values["take_profit_pct"]
            )
        )
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return trailing moving-average crossover directions."""
        prices = self._prices(data)
        signal = self._native_feature(
            prices,
            lambda p: ma_crossover_signal(
                p, fast_window=self.fast_window, slow_window=self.slow_window
            ),
        )
        if self.long_only:
            signal = signal.clip(lower=0.0)
        return self._validate_signals(signal, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons:
        """Explain each transition as a bullish/bearish MA crossover.

        The signal is memoryless -- recomputed fresh every row from the
        CURRENT fast/slow moving averages (see ``generate_signals``), no
        persistent state. A transition is "today's final (post-
        ``long_only``) signal differs from yesterday's", computed on the
        exact same clipped signal ``generate_signals()`` returns, so the
        two can never disagree. An increase in the final signal always
        means the fast MA crossed above the slow one (bullish); a
        decrease always means it crossed below (bearish) -- true whether
        the move is a fresh entry, an exit forced by ``long_only``
        clipping, or a direct long<->short reversal, since the final
        signal is a monotonic function of ``sign(fast_ma - slow_ma)``.
        """
        prices = self._prices(data)
        fast_ma = self._native_feature(
            prices, lambda p: moving_average(p, self.fast_window)
        ).to_numpy()
        slow_ma = self._native_feature(
            prices, lambda p: moving_average(p, self.slow_window)
        ).to_numpy()
        signal = self._native_feature(
            prices,
            lambda p: ma_crossover_signal(
                p, fast_window=self.fast_window, slow_window=self.slow_window
            ),
        )
        if self.long_only:
            signal = signal.clip(lower=0.0)
        final = signal.fillna(0.0).to_numpy()
        previous = np.vstack([np.zeros((1, final.shape[1])), final[:-1]])

        detail_code = np.empty(final.shape, dtype=object)
        details = np.empty(final.shape, dtype=object)
        for row in range(final.shape[0]):
            for col in range(final.shape[1]):
                if final[row, col] - previous[row, col] > EPSILON:
                    detail_code[row, col] = "bullish_crossover"
                    details[row, col] = (
                        f"fast MA {fast_ma[row, col]:.4f} crossed above "
                        f"slow MA {slow_ma[row, col]:.4f}"
                    )
                elif previous[row, col] - final[row, col] > EPSILON:
                    detail_code[row, col] = "bearish_crossover"
                    details[row, col] = (
                        f"fast MA {fast_ma[row, col]:.4f} crossed below "
                        f"slow MA {slow_ma[row, col]:.4f}"
                    )

        return self._validate_signal_reasons(
            pd.DataFrame(
                detail_code, index=prices.index, columns=prices.columns, dtype=object
            ),
            pd.DataFrame(
                details, index=prices.index, columns=prices.columns, dtype=object
            ),
            prices,
        )
