"""Stress scenarios and a random-sign test for realised returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.constants import SYMBOL
from quantlab.exceptions import QuantLabError
from quantlab.logging_config import get_logger
from quantlab.risk import metrics as M
from quantlab.risk._validation import (
    finite_real,
    nonnegative_int,
    numeric_series,
    positive_int,
)
from quantlab.risk.drawdown import max_drawdown
from quantlab.risk.stress import remove_best_days, scale_costs

logger = get_logger(__name__)

_STRESS_COLUMNS = [
    "scenario",
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "status",
    "error",
]


def _metrics_row(
    name: str, returns: pd.Series, ppy: int, risk_free_rate: float = 0.0
) -> dict[str, object]:
    """Build one successful stress-scenario row."""
    equity = M.equity_from_returns(returns)
    return {
        "scenario": name,
        "total_return": M.total_return(equity),
        "cagr": M.cagr(equity, ppy),
        "sharpe": M.sharpe_ratio(returns, risk_free_rate, ppy),
        "max_drawdown": max_drawdown(equity),
        "status": "ok",
        "error": None,
    }


def _failed_row(name: str, error: QuantLabError) -> dict[str, object]:
    """Keep an expected scenario failure visible in the result table."""
    return {
        "scenario": name,
        "total_return": float("nan"),
        "cagr": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "status": "failed",
        "error": str(error),
    }


def run_stress_tests(data: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Re-run the experiment under cost, delay and universe perturbations."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    if SYMBOL not in data.columns:
        raise ValueError(f"data must contain a {SYMBOL!r} column.")
    periods_per_year = positive_int(config.periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(config.risk_free_rate, name="risk_free_rate")

    baseline = run_backtest_from_config(data, config)
    rows = [
        _metrics_row("baseline", baseline.returns, periods_per_year, risk_free_rate)
    ]

    scenarios = {
        "commission x2": scale_costs(config, commission_mult=2.0),
        "commission x5": scale_costs(config, commission_mult=5.0),
        "slippage x2": scale_costs(config, slippage_mult=2.0),
    }
    for name, scenario_config in scenarios.items():
        result = run_backtest_from_config(data, scenario_config)
        rows.append(
            _metrics_row(name, result.returns, periods_per_year, risk_free_rate)
        )

    delayed = run_backtest_from_config(data, config, execution_delay=1)
    rows.append(
        _metrics_row(
            "execution delay +1",
            delayed.returns,
            periods_per_year,
            risk_free_rate,
        )
    )
    rows.append(
        _metrics_row(
            "best 10 days removed",
            remove_best_days(baseline.returns, 10),
            periods_per_year,
            risk_free_rate,
        )
    )

    if len(config.symbols) > 2:
        reduced_symbols = config.symbols[:-1]
        data_config = config.data.revalidated_copy(update={"symbols": reduced_symbols})
        reduced_config = config.revalidated_copy(update={"data": data_config})
        required_symbols = set(reduced_symbols)
        if config.benchmark_symbol is not None:
            required_symbols.add(config.benchmark_symbol)
        subset = data[data[SYMBOL].isin(required_symbols)].reset_index(drop=True)
        try:
            result = run_backtest_from_config(subset, reduced_config)
        except QuantLabError as exc:
            logger.warning("Reduced-universe scenario failed: %s", exc)
            rows.append(_failed_row("reduced universe", exc))
        else:
            rows.append(
                _metrics_row(
                    "reduced universe",
                    result.returns,
                    periods_per_year,
                    risk_free_rate,
                )
            )

    return pd.DataFrame(rows, columns=_STRESS_COLUMNS)


def monte_carlo_permutation(
    returns: pd.Series,
    *,
    n_iterations: int = 1000,
    seed: int = 42,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compare realised Sharpe with random sign flips of excess returns.

    The test preserves return magnitudes and randomises direction around the
    per-period risk-free return. Its empirical p-value is evidence against
    this specific random-sign null, not a probability of future profitability.
    """
    n_iterations = positive_int(n_iterations, name="n_iterations")
    seed = nonnegative_int(seed, name="seed")
    periods_per_year = positive_int(periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(risk_free_rate, name="risk_free_rate")
    validated = numeric_series(
        returns,
        name="returns",
        allow_nan=True,
        require_unique_index=True,
        require_sorted_index=True,
    )
    clean = validated.dropna().to_numpy(dtype=float)
    if (clean < -1.0).any():
        raise ValueError("returns must not contain values below -1.0.")
    if len(clean) < 2:
        return {
            "real_sharpe": 0.0,
            "p_value": 1.0,
            "n_iterations": float(n_iterations),
        }

    rng = np.random.default_rng(seed)
    risk_free_per_period = risk_free_rate / periods_per_year
    excess = clean - risk_free_per_period
    real = M.sharpe_ratio(pd.Series(clean), risk_free_rate, periods_per_year)
    count = 0
    for _ in range(n_iterations):
        signs = rng.choice([-1.0, 1.0], size=len(clean))
        random_returns = risk_free_per_period + excess * signs
        random_sharpe = M.sharpe_ratio(
            pd.Series(random_returns), risk_free_rate, periods_per_year
        )
        if random_sharpe >= real:
            count += 1
    return {
        "real_sharpe": float(real),
        "p_value": float((count + 1) / (n_iterations + 1)),
        "n_iterations": float(n_iterations),
    }
