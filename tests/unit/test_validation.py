"""Tests for walk-forward, sensitivity, bootstrap and stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tests.conftest import geometric_series, make_ohlcv

from quantlab.config import ExperimentConfig
from quantlab.validation.bootstrap import bootstrap_returns
from quantlab.validation.parameter_sensitivity import (
    run_parameter_sensitivity,
    sensitivity_heatmap_data,
)
from quantlab.validation.robustness import (
    monte_carlo_permutation,
    run_stress_tests,
)
from quantlab.validation.walk_forward import WalkForwardValidator


def _panel(n: int = 900) -> pd.DataFrame:
    frames = []
    for sym, seed, mu in [("AAA", 1, 0.0008), ("BBB", 2, 0.0002), ("CCC", 3, 0.0005)]:
        prices = geometric_series(n, mu=mu, sigma=0.012, s0=100.0, seed=seed)
        frames.append(make_ohlcv(sym, prices, start="2018-01-01"))
    return pd.concat(frames, ignore_index=True)


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "val",
            "data": {
                "source": "csv",
                "symbols": ["AAA", "BBB", "CCC"],
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
                "market_calendar": "XNYS",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 120,
                    "skip_period": 5,
                    "top_fraction": 0.4,
                },
            },
            "portfolio": {"allocator": "inverse_volatility", "maximum_weight": 0.6},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {"initial_capital": 100_000, "benchmark_symbol": "AAA"},
            "validation": {"optimization_metric": "sharpe"},
        }
    )


def test_walk_forward_produces_oos_curve() -> None:
    data = _panel()
    validator = WalkForwardValidator(_config())
    result = validator.run(
        data,
        parameter_grid={"lookback_period": [60, 120]},
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
    )
    assert len(result.folds) >= 1
    # OOS curve exists and every chosen param is from the grid.
    assert len(result.oos_returns) > 0
    for fold in result.folds:
        assert fold.best_params["lookback_period"] in (60, 120)
    table = result.summary_table()
    assert "test_sharpe" in table.columns


def test_parameter_sensitivity_grid() -> None:
    data = _panel()
    sens = run_parameter_sensitivity(
        data,
        _config(),
        parameter_x="lookback_period",
        values_x=[60, 120],
        parameter_y="top_fraction",
        values_y=[0.3, 0.5],
    )
    assert len(sens) == 4
    assert {"sharpe", "cagr", "max_drawdown"} <= set(sens.columns)
    heat = sensitivity_heatmap_data(sens, "lookback_period", "top_fraction", "sharpe")
    assert heat.shape == (2, 2)


def test_bootstrap_summary_percentiles() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    boot = bootstrap_returns(returns, n_iterations=200, block_size=5, seed=42)
    summary = boot.summary()
    assert {"cagr", "sharpe", "max_drawdown", "final_value"} == set(
        summary["statistic"]
    )
    # Percentile ordering holds.
    for _, row in summary.iterrows():
        assert row["p05"] <= row["median"] <= row["p95"]


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300))
    a = bootstrap_returns(returns, n_iterations=100, seed=7).samples
    b = bootstrap_returns(returns, n_iterations=100, seed=7).samples
    pd.testing.assert_frame_equal(a, b)


def test_stress_tests_include_expected_scenarios() -> None:
    data = _panel()
    table = run_stress_tests(data, _config())
    scenarios = set(table["scenario"])
    assert "baseline" in scenarios
    assert "commission x5" in scenarios
    assert "best 10 days removed" in scenarios
    # Higher commissions cannot improve the net total return vs baseline.
    base = table.loc[table["scenario"] == "baseline", "total_return"].iloc[0]
    c5 = table.loc[table["scenario"] == "commission x5", "total_return"].iloc[0]
    assert c5 <= base + 1e-9


def test_monte_carlo_permutation_reports_pvalue() -> None:
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    out = monte_carlo_permutation(returns, n_iterations=200, seed=42)
    assert 0.0 <= out["p_value"] <= 1.0
    assert "real_sharpe" in out
