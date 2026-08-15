"""Small transformations used by robustness scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import ExperimentConfig
from quantlab.risk._validation import finite_real, nonnegative_int, numeric_series


def scale_costs(
    config: ExperimentConfig,
    *,
    commission_mult: float = 1.0,
    spread_mult: float = 1.0,
    slippage_mult: float = 1.0,
) -> ExperimentConfig:
    """Return a revalidated config with each cost component scaled."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    commission = finite_real(commission_mult, name="commission_mult")
    spread = finite_real(spread_mult, name="spread_mult")
    slippage = finite_real(slippage_mult, name="slippage_mult")
    if min(commission, spread, slippage) < 0.0:
        raise ValueError("cost multipliers must be non-negative.")
    new_execution = config.execution.revalidated_copy(
        update={
            "commission_bps": config.execution.commission_bps * commission,
            "spread_bps": config.execution.spread_bps * spread,
            "slippage_bps": config.execution.slippage_bps * slippage,
            "impact_coefficient": config.execution.impact_coefficient * slippage,
        }
    )
    return config.revalidated_copy(update={"execution": new_execution})


def remove_best_days(returns: pd.Series, n: int = 10) -> pd.Series:
    """Set exactly the ``n`` largest finite observations to zero."""
    clean = numeric_series(returns, name="returns", allow_nan=True)
    count = nonnegative_int(n, name="n")
    if count == 0 or clean.empty:
        return clean
    finite_positions = np.flatnonzero(clean.notna().to_numpy())
    if len(finite_positions) == 0:
        return clean
    values = clean.iloc[finite_positions].to_numpy(dtype=float)
    order = np.argsort(-values, kind="stable")[:count]
    output = clean.copy()
    output.iloc[finite_positions[order]] = 0.0
    return output


def delay_execution(returns: pd.Series, periods: int = 1) -> pd.Series:
    """Shift realised returns as a post-hoc, cost-free delay approximation.

    Use a full backtest re-simulation when delayed weights, turnover and costs
    must also be modelled.
    """
    clean = numeric_series(returns, name="returns", allow_nan=True)
    delay = nonnegative_int(periods, name="periods")
    if delay == 0 or clean.empty:
        return clean
    result = clean.shift(delay)
    result.iloc[: min(delay, len(result))] = 0.0
    return result
