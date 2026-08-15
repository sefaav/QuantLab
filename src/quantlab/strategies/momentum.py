"""Trailing time-series and cross-sectional momentum strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.features._validation import (
    boolean,
    finite_real,
    non_negative_int,
    positive_int,
)
from quantlab.features.cross_sectional import select_top_bottom
from quantlab.features.momentum import momentum
from quantlab.features.returns import simple_returns
from quantlab.features.volatility import realized_volatility
from quantlab.strategies.base import BaseStrategy, register_strategy

_TIME_SERIES_SCALINGS = frozenset({"binary", "continuous", "volatility_adjusted"})


def _scaling(value: object, *, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"signal_scaling must be one of {sorted(allowed)}, got {value!r}."
        )
    return value


@register_strategy("time_series_momentum")
class TimeSeriesMomentumStrategy(BaseStrategy):
    """Map each asset's trailing momentum to a directional signal.

    ``binary`` uses only direction, ``continuous`` standardises momentum by
    its trailing dispersion, and ``volatility_adjusted`` divides momentum by
    trailing annualised volatility.
    """

    def __init__(
        self,
        lookback_period: int = 252,
        skip_period: int = 21,
        long_only: bool = True,
        signal_scaling: str = "binary",
        volatility_window: int = 63,
        periods_per_year: int = 252,
    ) -> None:
        values = self.validate_parameters(
            {
                "lookback_period": lookback_period,
                "skip_period": skip_period,
                "long_only": long_only,
                "signal_scaling": signal_scaling,
                "volatility_window": volatility_window,
                "periods_per_year": periods_per_year,
            }
        )
        self.lookback_period = values["lookback_period"]
        self.skip_period = values["skip_period"]
        self.long_only = values["long_only"]
        self.signal_scaling = values["signal_scaling"]
        self.volatility_window = values["volatility_window"]
        self.periods_per_year = values["periods_per_year"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate momentum windows, scaling and annualisation."""
        values = dict(parameters)
        values["lookback_period"] = positive_int(
            values["lookback_period"], name="lookback_period"
        )
        values["skip_period"] = non_negative_int(
            values["skip_period"], name="skip_period"
        )
        if values["skip_period"] >= values["lookback_period"]:
            raise ValueError("skip_period must be smaller than lookback_period.")
        values["long_only"] = boolean(values["long_only"], name="long_only")
        values["signal_scaling"] = _scaling(
            values["signal_scaling"], allowed=_TIME_SERIES_SCALINGS
        )
        if values["signal_scaling"] == "continuous" and values["lookback_period"] < 2:
            raise ValueError(
                "lookback_period must be at least 2 for continuous scaling."
            )
        values["volatility_window"] = positive_int(
            values["volatility_window"], name="volatility_window", minimum=2
        )
        values["periods_per_year"] = positive_int(
            values["periods_per_year"], name="periods_per_year"
        )
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return per-asset trailing-momentum signals."""
        prices = self._prices(data)
        score = momentum(prices, self.lookback_period, self.skip_period)

        if self.signal_scaling == "binary":
            signal = pd.DataFrame(
                np.sign(score), index=score.index, columns=score.columns
            )
        elif self.signal_scaling == "continuous":
            dispersion = score.rolling(
                self.lookback_period, min_periods=min(20, self.lookback_period)
            ).std(ddof=1)
            signal = (score / dispersion).clip(-1.0, 1.0)
        elif self.signal_scaling == "volatility_adjusted":
            volatility = realized_volatility(
                simple_returns(prices),
                window=self.volatility_window,
                periods_per_year=self.periods_per_year,
            )
            signal = (score / volatility).clip(-1.0, 1.0)
        else:  # pragma: no cover - constructor invariant
            raise RuntimeError(f"Unsupported signal scaling: {self.signal_scaling!r}.")

        if self.long_only:
            signal = signal.clip(lower=0.0)
        return self._validate_signals(signal, prices)


@register_strategy("cross_sectional_momentum")
class CrossSectionalMomentumStrategy(BaseStrategy):
    """Select the strongest assets and optionally short the weakest."""

    def __init__(
        self,
        lookback_period: int = 252,
        skip_period: int = 21,
        top_fraction: float = 0.25,
        bottom_fraction: float = 0.25,
        long_short: bool = False,
        signal_scaling: str = "binary",
    ) -> None:
        values = self.validate_parameters(
            {
                "lookback_period": lookback_period,
                "skip_period": skip_period,
                "top_fraction": top_fraction,
                "bottom_fraction": bottom_fraction,
                "long_short": long_short,
                "signal_scaling": signal_scaling,
            }
        )
        self.lookback_period = values["lookback_period"]
        self.skip_period = values["skip_period"]
        self.top_fraction = values["top_fraction"]
        self.bottom_fraction = values["bottom_fraction"]
        self.long_short = values["long_short"]
        self.signal_scaling = values["signal_scaling"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate momentum windows and disjoint selection fractions."""
        values = dict(parameters)
        values["lookback_period"] = positive_int(
            values["lookback_period"], name="lookback_period"
        )
        values["skip_period"] = non_negative_int(
            values["skip_period"], name="skip_period"
        )
        if values["skip_period"] >= values["lookback_period"]:
            raise ValueError("skip_period must be smaller than lookback_period.")
        values["top_fraction"] = finite_real(
            values["top_fraction"], name="top_fraction", minimum=0.0
        )
        values["bottom_fraction"] = finite_real(
            values["bottom_fraction"], name="bottom_fraction", minimum=0.0
        )
        if values["top_fraction"] > 1.0 or values["bottom_fraction"] > 1.0:
            raise ValueError("top_fraction and bottom_fraction must not exceed 1.")
        values["long_short"] = boolean(values["long_short"], name="long_short")
        if (
            values["long_short"]
            and values["top_fraction"] + values["bottom_fraction"] > 1.0
        ):
            raise ValueError(
                "top_fraction + bottom_fraction must not exceed 1 when "
                "long_short is enabled."
            )
        values["signal_scaling"] = _scaling(
            values["signal_scaling"], allowed=frozenset({"binary"})
        )
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return disjoint cross-sectional selections."""
        prices = self._prices(data)
        score = momentum(prices, self.lookback_period, self.skip_period)
        bottom = self.bottom_fraction if self.long_short else 0.0
        selection = select_top_bottom(
            score, top_fraction=self.top_fraction, bottom_fraction=bottom
        )
        return self._validate_signals(selection, prices)
