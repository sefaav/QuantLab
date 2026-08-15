"""Stateful mean reversion around a trailing price z-score."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.features._validation import boolean, finite_real, positive_int
from quantlab.features.mean_reversion import rolling_zscore
from quantlab.strategies.base import BaseStrategy, register_strategy


def _walk_positions(
    z: np.ndarray,
    entry: float,
    exit_: float,
    stop: float | None,
    long_only: bool,
) -> np.ndarray:
    """Convert z-scores into persistent positions in ``{-1, 0, 1}``."""
    positions = np.zeros_like(z, dtype=float)
    state = 0.0
    for index, value in enumerate(z):
        if np.isnan(value) or (stop is not None and abs(value) > stop):
            state = 0.0
        elif state == 0.0:
            if value < -entry:
                state = 1.0
            elif value > entry and not long_only:
                state = -1.0
        elif (state == 1.0 and value > -exit_) or (state == -1.0 and value < exit_):
            state = 0.0
        positions[index] = state
    return positions


@register_strategy("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    """Trade deviations from a trailing mean until exit or stop thresholds."""

    def __init__(
        self,
        lookback_period: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float | None = 4.0,
        long_only: bool = True,
    ) -> None:
        values = self.validate_parameters(
            {
                "lookback_period": lookback_period,
                "entry_zscore": entry_zscore,
                "exit_zscore": exit_zscore,
                "stop_zscore": stop_zscore,
                "long_only": long_only,
            }
        )
        self.lookback_period = values["lookback_period"]
        self.entry_zscore = values["entry_zscore"]
        self.exit_zscore = values["exit_zscore"]
        self.stop_zscore = values["stop_zscore"]
        self.long_only = values["long_only"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate z-score windows, thresholds and direction mode."""
        values = dict(parameters)
        values["lookback_period"] = positive_int(
            values["lookback_period"], name="lookback_period", minimum=2
        )
        values["entry_zscore"] = finite_real(
            values["entry_zscore"], name="entry_zscore", minimum=0.0, strict=True
        )
        values["exit_zscore"] = finite_real(
            values["exit_zscore"], name="exit_zscore", minimum=0.0
        )
        if values["stop_zscore"] is not None:
            values["stop_zscore"] = finite_real(
                values["stop_zscore"], name="stop_zscore", minimum=0.0
            )
        values["long_only"] = boolean(values["long_only"], name="long_only")
        if values["entry_zscore"] <= values["exit_zscore"]:
            raise ValueError("entry_zscore must exceed exit_zscore.")
        if (
            values["stop_zscore"] is not None
            and values["stop_zscore"] <= values["entry_zscore"]
        ):
            raise ValueError("stop_zscore must exceed entry_zscore.")
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return the trailing z-score state for every asset."""
        prices = self._prices(data)
        zscore = rolling_zscore(prices, self.lookback_period)
        signals = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns, dtype=float
        )
        for symbol in prices.columns:
            signals[symbol] = _walk_positions(
                zscore[symbol].to_numpy(dtype=float),
                entry=self.entry_zscore,
                exit_=self.exit_zscore,
                stop=self.stop_zscore,
                long_only=self.long_only,
            )
        return self._validate_signals(signals, prices)
