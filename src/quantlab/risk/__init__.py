"""Risk and performance metrics."""

from __future__ import annotations

from quantlab.risk.drawdown import (
    average_drawdown,
    drawdown_durations,
    drawdown_series,
    longest_drawdown,
    max_drawdown,
    max_drawdown_details,
)
from quantlab.risk.exposure import (
    average_gross_exposure,
    average_net_exposure,
    gross_exposure_series,
    net_exposure_series,
)
from quantlab.risk.metrics import (
    annualized_alpha,
    annualized_volatility,
    beta,
    cagr,
    calmar_ratio,
    compute_metrics,
    equity_from_returns,
    hit_rate,
    information_ratio,
    kurtosis,
    profit_factor,
    rolling_sharpe_ratio,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    total_return,
    tracking_error,
)
from quantlab.risk.stress import delay_execution, remove_best_days, scale_costs
from quantlab.risk.var import historical_cvar, historical_var

__all__ = [
    "annualized_alpha",
    "annualized_volatility",
    "average_drawdown",
    "average_gross_exposure",
    "average_net_exposure",
    "beta",
    "cagr",
    "calmar_ratio",
    "compute_metrics",
    "delay_execution",
    "drawdown_durations",
    "drawdown_series",
    "equity_from_returns",
    "gross_exposure_series",
    "historical_cvar",
    "historical_var",
    "hit_rate",
    "information_ratio",
    "kurtosis",
    "longest_drawdown",
    "max_drawdown",
    "max_drawdown_details",
    "net_exposure_series",
    "profit_factor",
    "remove_best_days",
    "rolling_sharpe_ratio",
    "scale_costs",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "total_return",
    "tracking_error",
]
