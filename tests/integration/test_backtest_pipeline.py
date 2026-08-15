"""End-to-end backtest pipeline test.

Load data → features → strategy → backtest → metrics, exercising the full stack
through the config-driven runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv

from quantlab.backtesting.engine import BacktestEngine
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.constants import OHLCV_COLUMNS
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionModel
from quantlab.portfolio.allocator import EqualWeightAllocator, PortfolioAllocator
from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy


def _panel() -> pd.DataFrame:
    frames = []
    for sym, seed, mu in [("AAA", 1, 0.0007), ("BBB", 2, 0.0003), ("CCC", 3, 0.0005)]:
        prices = geometric_series(600, mu=mu, sigma=0.012, s0=100.0, seed=seed)
        frames.append(make_ohlcv(sym, prices, start="2019-01-01"))
    return pd.concat(frames, ignore_index=True)


def _config(strategy: dict, allocator: str = "equal_weight") -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "integration",
            "data": {
                "source": "csv",
                "symbols": ["AAA", "BBB", "CCC"],
                "start_date": "2019-01-01",
                "end_date": "2021-06-01",
                "market_calendar": "XNYS",
            },
            "strategy": strategy,
            "portfolio": {
                "allocator": allocator,
                "maximum_weight": 0.6,
                "volatility_window": 40,
                "rebalance_frequency": "monthly",
            },
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {
                "initial_capital": 100_000,
                "benchmark_symbol": "AAA",
                "risk_free_rate": 0.02,
                "periods_per_year": 252,
            },
        }
    )


@pytest.mark.parametrize(
    "strategy",
    [
        {"name": "buy_and_hold"},
        {
            "name": "cross_sectional_momentum",
            "parameters": {
                "lookback_period": 120,
                "skip_period": 10,
                "top_fraction": 0.4,
            },
        },
        {
            "name": "time_series_momentum",
            "parameters": {"lookback_period": 100, "skip_period": 5, "long_only": True},
        },
    ],
)
def test_full_pipeline_runs(strategy: dict) -> None:
    data = _panel()
    result = run_backtest_from_config(data, _config(strategy))

    # Structural checks on the result object.
    assert len(result.equity_curve) > 100
    assert result.equity_curve.index.is_monotonic_increasing
    assert result.equity_curve.iloc[0] == pytest.approx(100_000, rel=0.05)
    assert np.isfinite(result.equity_curve.to_numpy()).all()
    # Metrics present and finite.
    for key in ["total_return", "cagr", "sharpe_ratio", "max_drawdown"]:
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])
    assert result.metrics["total_cost_fraction"] == pytest.approx(
        result.total_cost_fraction()
    )
    assert "total_cost" not in result.metrics
    # Weights never breach the per-asset cap after constraints.
    assert result.weights.abs().to_numpy().max() <= 0.6 + 1e-6
    # No NaNs in the returns series.
    assert not result.returns.isna().any()


def test_costs_reduce_performance() -> None:
    data = _panel()
    strat = {
        "name": "cross_sectional_momentum",
        "parameters": {"lookback_period": 120, "skip_period": 10, "top_fraction": 0.4},
    }
    result = run_backtest_from_config(data, _config(strat))
    comp = result.gross_net_comparison()
    # Net return cannot exceed gross return once costs are charged.
    assert comp["net_total_return"] <= comp["gross_total_return"] + 1e-9
    assert "gross_sharpe" in comp
    assert np.isfinite(comp["gross_sharpe"])
    assert "Total costs (currency units)" in result.summary()


def test_reproducible_same_inputs() -> None:
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    r1 = run_backtest_from_config(data, cfg)
    r2 = run_backtest_from_config(data, cfg)
    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
    assert r1.metrics["sharpe_ratio"] == r2.metrics["sharpe_ratio"]


@pytest.mark.parametrize(
    "benchmark_kind", ["symbol", "equal_weight", "first_asset", "cash"]
)
def test_all_configured_benchmark_kinds_run_end_to_end(benchmark_kind: str) -> None:
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    benchmark_update = {
        "benchmark_kind": benchmark_kind,
        "benchmark_symbol": "AAA" if benchmark_kind == "symbol" else None,
    }
    cfg = cfg.revalidated_copy(
        update={"backtest": cfg.backtest.revalidated_copy(update=benchmark_update)}
    )

    result = run_backtest_from_config(data, cfg)

    assert result.benchmark_returns is not None
    assert len(result.benchmark_returns) == len(result.returns)
    assert not result.benchmark_returns.isna().any()


def test_equal_weight_benchmark_ignores_data_outside_configured_universe() -> None:
    data = _panel()
    extra = make_ohlcv(
        "ZZZ",
        geometric_series(600, mu=0.01, sigma=0.001, s0=100.0, seed=99),
        start="2019-01-01",
    )
    cfg = _config({"name": "buy_and_hold"})
    cfg = cfg.revalidated_copy(
        update={
            "backtest": cfg.backtest.revalidated_copy(
                update={"benchmark_kind": "equal_weight", "benchmark_symbol": None}
            )
        }
    )

    baseline = run_backtest_from_config(data, cfg)
    with_extra = run_backtest_from_config(pd.concat([data, extra]), cfg)

    assert baseline.benchmark_returns is not None
    assert with_extra.benchmark_returns is not None
    pd.testing.assert_series_equal(
        baseline.benchmark_returns, with_extra.benchmark_returns
    )


@pytest.mark.parametrize("execution_delay", [-1, 1.5, True])
def test_invalid_execution_delay_raises(execution_delay: object) -> None:
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})

    with pytest.raises(BacktestError, match="execution_delay"):
        run_backtest_from_config(
            data,
            cfg,
            execution_delay=execution_delay,  # type: ignore[arg-type]
        )


def test_backtest_refuses_a_silently_reduced_configured_universe() -> None:
    data = _panel()
    data = data[data["symbol"] != "CCC"].reset_index(drop=True)
    cfg = _config({"name": "buy_and_hold"})

    with pytest.raises(BacktestError, match=r"missing configured tradable.*CCC"):
        run_backtest_from_config(data, cfg)


def _run_engine_with_invalid_data(data: object) -> None:
    cfg = _config({"name": "buy_and_hold"})
    BacktestEngine().run(
        data,  # type: ignore[arg-type]
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )


def test_backtest_public_api_rejects_non_dataframe_data() -> None:
    with pytest.raises(BacktestError, match="pandas DataFrame"):
        _run_engine_with_invalid_data([{"symbol": "AAA"}])


@pytest.mark.parametrize("present_column", ["timestamp", "symbol"])
def test_backtest_public_api_rejects_missing_data_axes(
    present_column: str,
) -> None:
    with pytest.raises(BacktestError, match="missing required column"):
        _run_engine_with_invalid_data(pd.DataFrame({present_column: ["value"]}))


def test_backtest_public_api_reports_empty_canonical_frame() -> None:
    with pytest.raises(BacktestError, match="empty data"):
        _run_engine_with_invalid_data(pd.DataFrame(columns=OHLCV_COLUMNS))


class _InvalidAllocator(PortfolioAllocator):
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "missing_symbol":
            return signals.drop(columns=signals.columns[-1])
        output = signals.copy()
        output.iloc[-1, -1] = np.nan
        return output


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_symbol", "exactly the price-matrix"),
        ("missing_value", "allocator output must contain only finite values"),
    ],
)
def test_backtest_rejects_invalid_allocator_output(mode: str, message: str) -> None:
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    execution = ExecutionModel.from_config(cfg.execution)

    with pytest.raises(BacktestError, match=message):
        BacktestEngine().run(
            data,
            BuyAndHoldStrategy(),
            _InvalidAllocator(mode),
            execution,
            cfg,
        )
