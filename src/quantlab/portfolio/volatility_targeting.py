"""Trailing ex-ante portfolio-volatility targeting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON, TRADING_DAYS_PER_YEAR
from quantlab.portfolio._validation import (
    finite_real,
    positive_int,
    require_same_axes,
    validate_frame,
)


def estimated_portfolio_volatility(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    window: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Estimate annualised volatility for each row's current portfolio.

    Current weights are applied to complete observations inside the trailing
    return window. A missing return matters only when that asset has a non-zero
    weight; incomplete observations are excluded rather than treated as zero.
    """
    validated_weights = validate_frame(weights, name="weights")
    validated_returns = validate_frame(returns, name="returns", allow_missing=True)
    require_same_axes(
        validated_weights,
        validated_returns,
        reference_name="weights",
        other_name="returns",
    )
    length = positive_int(window, name="window", minimum=2)
    annual_periods = positive_int(periods_per_year, name="periods_per_year")
    minimum_observations = _minimum_observations(length)

    weight_values = validated_weights.to_numpy(dtype=float)
    return_values = validated_returns.to_numpy(dtype=float, na_value=np.nan)
    estimates = np.full(len(validated_weights), np.nan)
    for row_number in range(len(validated_weights)):
        start = max(0, row_number - length + 1)
        segment = return_values[start : row_number + 1]
        active = np.abs(weight_values[row_number]) > EPSILON
        if active.any():
            complete = np.isfinite(segment[:, active]).all(axis=1)
            segment = segment[complete]
            if len(segment) < minimum_observations:
                continue
            pseudo_returns = segment[:, active] @ weight_values[row_number, active]
        else:
            if len(segment) < minimum_observations:
                continue
            pseudo_returns = np.zeros(len(segment), dtype=float)
        estimates[row_number] = float(pseudo_returns.std(ddof=1))
    return pd.Series(estimates, index=validated_weights.index) * math.sqrt(
        annual_periods
    )


def volatility_target_leverage(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_volatility: float,
    *,
    window: int = 63,
    maximum_leverage: float = 1.5,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Return the bounded multiplier used to target annualised volatility.

    Before the estimator's minimum warm-up, exposure remains at the smaller
    of one and ``maximum_leverage``. A missing estimate after warm-up receives
    zero leverage because risk could not be measured reliably.
    """
    target = finite_real(
        target_volatility,
        name="target_volatility",
        minimum=0.0,
        strict=True,
    )
    length = positive_int(window, name="window", minimum=2)
    leverage_cap = finite_real(
        maximum_leverage,
        name="maximum_leverage",
        minimum=0.0,
        strict=True,
    )
    annual_periods = positive_int(periods_per_year, name="periods_per_year")
    estimated = estimated_portfolio_volatility(weights, returns, length, annual_periods)
    leverage = (target / (estimated + EPSILON)).clip(lower=0.0, upper=leverage_cap)
    warmup_count = _minimum_observations(length) - 1
    if warmup_count > 0:
        leverage.iloc[:warmup_count] = leverage.iloc[:warmup_count].fillna(
            min(1.0, leverage_cap)
        )
    return leverage.fillna(0.0)


def apply_volatility_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_volatility: float,
    *,
    window: int = 63,
    maximum_leverage: float = 1.5,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Scale weights by the validated volatility-target multiplier."""
    validated_weights = validate_frame(weights, name="weights")
    leverage = volatility_target_leverage(
        validated_weights,
        returns,
        target_volatility,
        window=window,
        maximum_leverage=maximum_leverage,
        periods_per_year=periods_per_year,
    )
    return validated_weights.mul(leverage, axis=0)


def _minimum_observations(window: int) -> int:
    return max(2, window // 2)
