"""ADF-gated pairs trading on a trailing price-level relationship."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.exceptions import StrategyError
from quantlab.features._validation import (
    boolean,
    finite_real,
    numeric_pandas,
    positive_int,
    same_axes,
)
from quantlab.features.mean_reversion import rolling_zscore
from quantlab.logging_config import get_logger
from quantlab.strategies.base import BaseStrategy, register_strategy

logger = get_logger(__name__)


def adf_pvalue(series: pd.Series) -> float | None:
    """Return an ADF p-value, or ``None`` when the test is inconclusive."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    validated = numeric_pandas(series, name="series")
    values = validated.dropna().to_numpy(dtype=float)
    if len(values) < 20 or np.allclose(values, values[0]):
        return None
    try:
        from statsmodels.tsa.stattools import adfuller

        pvalue = float(adfuller(values, autolag="AIC")[1])
    except Exception as exc:  # pragma: no cover - third-party numerical failures
        logger.warning("ADF test failed: %s", exc)
        return None
    return pvalue if np.isfinite(pvalue) else None


def _ols_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return intercept and slope for ``y = intercept + slope * x``."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have identical shapes.")
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2:
        return np.nan, np.nan
    clean_x = x_values[mask]
    clean_y = y_values[mask]
    if float(np.ptp(clean_x)) <= EPSILON:
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(clean_x)), clean_x])
    coefficients, *_ = np.linalg.lstsq(design, clean_y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the OLS slope of y on x with an intercept."""
    return _ols_coefficients(x, y)[1]


def rolling_hedge_parameters(
    a: pd.Series, b: pd.Series, window: int, dynamic: bool
) -> tuple[pd.Series, pd.Series]:
    """Return trailing OLS intercept and slope without pre-formation values."""
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise TypeError("a and b must be pandas Series.")
    validated_a = numeric_pandas(a, name="a", strictly_positive=True)
    validated_b = numeric_pandas(b, name="b", strictly_positive=True)
    same_axes(validated_a, validated_b, names=("b",))
    length = positive_int(window, name="window", minimum=2)
    use_dynamic = boolean(dynamic, name="dynamic")

    intercept = pd.Series(np.nan, index=validated_a.index, dtype=float)
    beta = pd.Series(np.nan, index=validated_a.index, dtype=float)
    if len(validated_a) <= length:
        return intercept, beta

    if not use_dynamic:
        alpha_value, beta_value = _ols_coefficients(
            validated_b.iloc[:length].to_numpy(dtype=float),
            validated_a.iloc[:length].to_numpy(dtype=float),
        )
        intercept.iloc[length:] = alpha_value
        beta.iloc[length:] = beta_value
        return intercept, beta

    a_values = validated_a.to_numpy(dtype=float)
    b_values = validated_b.to_numpy(dtype=float)
    for position in range(length, len(validated_a)):
        start = position - length
        alpha_value, beta_value = _ols_coefficients(
            b_values[start:position], a_values[start:position]
        )
        intercept.iloc[position] = alpha_value
        beta.iloc[position] = beta_value
    return intercept, beta


def _rolling_hedge_ratio(
    a: pd.Series, b: pd.Series, window: int, dynamic: bool
) -> pd.Series:
    """Return the trailing OLS hedge slope."""
    return rolling_hedge_parameters(a, b, window, dynamic)[1]


def _walk_pairs_positions(
    zscore: np.ndarray,
    tradable: np.ndarray,
    entry: float,
    exit_: float,
    stop: float | None,
) -> np.ndarray:
    """Convert spread z-scores into persistent positions in ``{-1, 0, 1}``."""
    if len(zscore) != len(tradable):
        raise ValueError("zscore and tradable must have the same length.")
    positions = np.zeros_like(zscore, dtype=float)
    state = 0.0
    for index, value in enumerate(zscore):
        if not np.isfinite(value) or (stop is not None and abs(value) > stop):
            state = 0.0
        elif state == 0.0 and bool(tradable[index]):
            if value < -entry:
                state = 1.0
            elif value > entry:
                state = -1.0
        elif (state == 1.0 and value > -exit_) or (state == -1.0 and value < exit_):
            state = 0.0
        positions[index] = state
    return positions


@register_strategy("pairs_trading")
class PairsTradingStrategy(BaseStrategy):
    """Trade the residual of a trailing price-level regression between two assets.

    The ADF test gates new entries. Open positions still follow their z-score
    exit and stop rules, and any undefined z-score forces the pair flat.
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        formation_window: int = 252,
        zscore_window: int = 63,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float | None = 4.0,
        dynamic_hedge_ratio: bool = True,
        adf_pvalue_threshold: float = 0.10,
    ) -> None:
        values = self.validate_parameters(
            {
                "symbol_a": symbol_a,
                "symbol_b": symbol_b,
                "formation_window": formation_window,
                "zscore_window": zscore_window,
                "entry_zscore": entry_zscore,
                "exit_zscore": exit_zscore,
                "stop_zscore": stop_zscore,
                "dynamic_hedge_ratio": dynamic_hedge_ratio,
                "adf_pvalue_threshold": adf_pvalue_threshold,
            }
        )
        self.symbol_a = values["symbol_a"]
        self.symbol_b = values["symbol_b"]
        self.formation_window = values["formation_window"]
        self.zscore_window = values["zscore_window"]
        self.entry_zscore = values["entry_zscore"]
        self.exit_zscore = values["exit_zscore"]
        self.stop_zscore = values["stop_zscore"]
        self.dynamic_hedge_ratio = values["dynamic_hedge_ratio"]
        self.adf_pvalue_threshold = values["adf_pvalue_threshold"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate symbols, windows, thresholds and ADF confidence."""
        values = dict(parameters)
        for key in ("symbol_a", "symbol_b"):
            value = values[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string.")
            values[key] = value.strip().upper()
        if values["symbol_a"] == values["symbol_b"]:
            raise ValueError("symbol_a and symbol_b must differ.")
        values["formation_window"] = positive_int(
            values["formation_window"], name="formation_window", minimum=20
        )
        values["zscore_window"] = positive_int(
            values["zscore_window"], name="zscore_window", minimum=2
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
        if values["entry_zscore"] <= values["exit_zscore"]:
            raise ValueError("entry_zscore must exceed exit_zscore.")
        if (
            values["stop_zscore"] is not None
            and values["stop_zscore"] <= values["entry_zscore"]
        ):
            raise ValueError("stop_zscore must exceed entry_zscore.")
        values["dynamic_hedge_ratio"] = boolean(
            values["dynamic_hedge_ratio"], name="dynamic_hedge_ratio"
        )
        values["adf_pvalue_threshold"] = finite_real(
            values["adf_pvalue_threshold"],
            name="adf_pvalue_threshold",
            minimum=0.0,
            strict=True,
        )
        if values["adf_pvalue_threshold"] >= 1.0:
            raise ValueError("adf_pvalue_threshold must be strictly below 1.")
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return dollar-hedged signals for the two configured legs."""
        prices = self._prices(data)
        for symbol in (self.symbol_a, self.symbol_b):
            if symbol not in prices.columns:
                raise StrategyError(
                    f"Pairs trading needs symbol '{symbol}' in the data; "
                    f"available: {list(prices.columns)}."
                )
        a = prices[self.symbol_a]
        b = prices[self.symbol_b]
        intercept, beta = rolling_hedge_parameters(
            a, b, self.formation_window, self.dynamic_hedge_ratio
        )
        spread = a - intercept - beta * b
        zscore = rolling_zscore(spread, self.zscore_window)
        state = pd.Series(
            _walk_pairs_positions(
                zscore.to_numpy(dtype=float),
                self._stationarity_gate(a, b),
                entry=self.entry_zscore,
                exit_=self.exit_zscore,
                stop=self.stop_zscore,
            ),
            index=prices.index,
            dtype=float,
        )

        raw_legs = pd.DataFrame(
            {
                self.symbol_a: state * a,
                self.symbol_b: -state * beta * b,
            },
            index=prices.index,
        )
        largest_leg = raw_legs.abs().max(axis=1).where(lambda value: value > EPSILON)
        pair_signals = raw_legs.div(largest_leg, axis=0)
        signals = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns, dtype=float
        )
        signals.loc[:, [self.symbol_a, self.symbol_b]] = pair_signals
        return self._validate_signals(signals, prices)

    def _stationarity_gate(self, a: pd.Series, b: pd.Series) -> np.ndarray:
        """Test full trailing formation residuals at bounded intervals."""
        gate = np.zeros(len(a), dtype=bool)
        last_pvalue: float | None = None
        static_coefficients: tuple[float, float] | None = None
        if not self.dynamic_hedge_ratio and len(a) >= self.formation_window:
            static_coefficients = _ols_coefficients(
                b.iloc[: self.formation_window].to_numpy(dtype=float),
                a.iloc[: self.formation_window].to_numpy(dtype=float),
            )
        for position in range(self.formation_window, len(a)):
            if (position - self.formation_window) % self.zscore_window == 0:
                start = position - self.formation_window
                window = pd.concat(
                    {"a": a.iloc[start:position], "b": b.iloc[start:position]},
                    axis=1,
                ).dropna()
                if len(window) != self.formation_window:
                    last_pvalue = None
                else:
                    intercept, beta = (
                        _ols_coefficients(
                            window["b"].to_numpy(dtype=float),
                            window["a"].to_numpy(dtype=float),
                        )
                        if self.dynamic_hedge_ratio
                        else static_coefficients or (np.nan, np.nan)
                    )
                    if not np.isfinite(intercept) or not np.isfinite(beta):
                        last_pvalue = None
                    else:
                        residual = window["a"] - intercept - beta * window["b"]
                        last_pvalue = adf_pvalue(residual)
            gate[position] = (
                last_pvalue is not None and last_pvalue <= self.adf_pvalue_threshold
            )
        return gate
