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
                "instruments": [
                    {"symbol": s, "source": "csv", "calendar": "XNYS"}
                    for s in ["AAA", "BBB", "CCC"]
                ],
                "start_date": "2019-01-01",
                "end_date": "2021-06-01",
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
                "benchmark": {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
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


def test_direct_api_custom_execution_model_is_the_single_source_of_truth() -> None:
    """docs/api.md recommends using BacktestEngine directly to supply a
    custom execution-model instance. A config declaring 2 bps commission,
    run instead with a custom ExecutionModel at 100 bps, must have its
    equity curve, trade log and metadata/report all describe the *actual*
    100 bps model, never a mix where accounting charges 100 bps but the
    trade log or the report still claim the YAML's 2 bps."""
    from quantlab.execution.costs import CommissionModel, SpreadModel
    from quantlab.execution.slippage import ConstantSlippageModel
    from quantlab.reporting.research_summary import methodology

    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    assert cfg.commission_bps == 2.0  # confirms the YAML/actual values differ

    custom_model = ExecutionModel(
        commission=CommissionModel(100.0),
        spread=SpreadModel(50.0),
        slippage=ConstantSlippageModel(5.0),
    )
    result = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        custom_model,
        cfg,
    )

    # Metadata must record the actual model, not the YAML's.
    assert result.metadata["commission_bps"] == 100.0
    assert result.metadata["spread_bps"] == 50.0
    assert result.metadata["slippage_bps"] == 5.0

    # The trade log's own per-fill commission must match: on any date with
    # a real fill, commission = traded_notional * 100bps/10_000, not
    # 2bps/10_000.
    filled = result.trades.loc[result.trades["traded_notional"] > 0]
    assert len(filled) > 0
    implied_bps = (filled["commission"] / filled["traded_notional"]) * 10_000
    assert implied_bps.round(6).unique().tolist() == [100.0]

    # The report's methodology text must describe the actual model too.
    text = methodology(result)
    assert "commission 100.0 bps" in text
    assert "50.0 bps full quoted spread" in text
    assert "constant slippage 5.0 bps" in text
    assert "2.0 bps" not in text


def test_direct_api_records_the_actual_strategy_parameters_used() -> None:
    """config.yaml in a saved bundle is still whatever config the caller
    happened to pass alongside a custom strategy object -- it can declare
    lookback_period=252 while the strategy actually used was built with
    lookback_period=10. metadata must record the real, effective value and
    flag that config.yaml doesn't reflect it, rather than leaving the only
    persisted record of "what ran" silently wrong."""
    from quantlab.strategies.mean_reversion import MeanReversionStrategy

    data = _panel()
    cfg = _config({"name": "mean_reversion", "parameters": {"lookback_period": 252}})
    assert cfg.strategy.parameters["lookback_period"] == 252

    mismatched = BacktestEngine().run(
        data,
        MeanReversionStrategy(lookback_period=10),
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert mismatched.metadata["strategy_parameters"]["lookback_period"] == 10
    assert mismatched.metadata["config_yaml_reflects_strategy"] is False

    matching = BacktestEngine().run(
        data,
        MeanReversionStrategy(lookback_period=252),
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert matching.metadata["strategy_parameters"]["lookback_period"] == 252
    assert matching.metadata["config_yaml_reflects_strategy"] is True


def test_direct_api_catches_a_mismatch_on_an_undeclared_default_parameter() -> None:
    """config.yaml can omit a parameter entirely, leaving it at the
    strategy's own constructor default (20 for mean_reversion's
    lookback_period) -- comparing only *declared* config keys would never
    even examine lookback_period here, silently missing a mismatch on the
    exact parameter the config never mentions."""
    from quantlab.strategies.mean_reversion import MeanReversionStrategy

    data = _panel()
    cfg = _config({"name": "mean_reversion", "parameters": {}})
    assert "lookback_period" not in cfg.strategy.parameters

    mismatched = BacktestEngine().run(
        data,
        MeanReversionStrategy(lookback_period=10),
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert mismatched.metadata["config_yaml_reflects_strategy"] is False

    matching = BacktestEngine().run(
        data,
        MeanReversionStrategy(lookback_period=20),  # the constructor's own default
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert matching.metadata["config_yaml_reflects_strategy"] is True


def test_direct_api_records_whether_config_yaml_reflects_the_allocator() -> None:
    """Same guarantee as config_yaml_reflects_strategy, for the allocator: a
    custom allocator instance can silently diverge from config.yaml's own
    portfolio.allocator settings."""
    from quantlab.portfolio.allocator import InverseVolatilityAllocator
    from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy

    data = _panel()
    # _config's own defaults: maximum_weight=0.6, volatility_window=40,
    # backtest.periods_per_year=252.
    cfg = _config({"name": "buy_and_hold"}, allocator="inverse_volatility")

    mismatched = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        InverseVolatilityAllocator(
            volatility_window=10, maximum_weight=0.6, periods_per_year=252
        ),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert mismatched.metadata["config_yaml_reflects_allocator"] is False

    matching = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        InverseVolatilityAllocator(
            volatility_window=40, maximum_weight=0.6, periods_per_year=252
        ),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert matching.metadata["config_yaml_reflects_allocator"] is True


def test_direct_api_records_whether_config_yaml_reflects_execution() -> None:
    """Same guarantee as config_yaml_reflects_strategy/_allocator, for the
    execution model: a config declaring 2 bps commission, run instead with a
    custom ExecutionModel at 100 bps, must have config_yaml_reflects_execution
    report False -- and the HTML report's footer (see
    test_reporting_hardening.py) must not claim reproducibility from
    config.yaml when only this one of the three flags is false."""
    from quantlab.execution.costs import CommissionModel, SpreadModel
    from quantlab.execution.slippage import ConstantSlippageModel
    from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy

    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    assert cfg.commission_bps == 2.0

    mismatched = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel(
            commission=CommissionModel(100.0),
            spread=SpreadModel(50.0),
            slippage=ConstantSlippageModel(5.0),
        ),
        cfg,
    )
    assert mismatched.metadata["commission_bps"] == 100.0
    assert mismatched.metadata["config_yaml_reflects_execution"] is False
    # The other two components are untouched and still built correctly.
    assert mismatched.metadata["config_yaml_reflects_strategy"] is True
    assert mismatched.metadata["config_yaml_reflects_allocator"] is True

    matching = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert matching.metadata["config_yaml_reflects_execution"] is True


def test_config_yaml_reflects_strategy_catches_a_behavior_overriding_subclass() -> None:
    """A strategy subclass that overrides behaviour (here: always stays in
    cash) without changing any constructor parameter is indistinguishable
    from its base class by parameter comparison alone, or by an
    `isinstance` check, or by `.name` (a class attribute the subclass
    inherits unchanged) -- only exact class identity catches it."""
    from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy

    class NeverBuys(BuyAndHoldStrategy):
        def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame | None = None
        ) -> pd.DataFrame:
            return super().generate_signals(data, features) * 0.0

    data = _panel()
    cfg = _config({"name": "buy_and_hold"})

    result = BacktestEngine().run(
        data,
        NeverBuys(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert result.metadata["config_yaml_reflects_strategy"] is False


def test_config_yaml_reflects_allocator_catches_a_behavior_overriding_subclass() -> (
    None
):
    """Same guarantee as the strategy case, for the allocator."""
    from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy

    class AlwaysZeroAllocator(EqualWeightAllocator):
        def allocate(self, signals, data):  # type: ignore[no-untyped-def]
            return super().allocate(signals, data) * 0.0

    data = _panel()
    cfg = _config({"name": "buy_and_hold"})

    result = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        AlwaysZeroAllocator(),
        ExecutionModel.from_config(cfg.execution),
        cfg,
    )
    assert result.metadata["config_yaml_reflects_allocator"] is False


def test_config_yaml_reflects_execution_catches_a_commission_subclass() -> None:
    """A commission subclass that overrides the actual cost calculation
    while still reporting the same `commission_bps` value is
    indistinguishable by the numeric comparison alone -- only exact class
    identity (`commission_class`) catches it."""
    from quantlab.execution.costs import CommissionModel, SpreadModel
    from quantlab.execution.slippage import ConstantSlippageModel
    from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy

    class FreeCommission(CommissionModel):
        def calculate(self, traded_notional: pd.DataFrame) -> pd.Series:
            return super().calculate(traded_notional) * 0.0

    data = _panel()
    cfg = _config({"name": "buy_and_hold"})

    result = BacktestEngine().run(
        data,
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel(
            commission=FreeCommission(cfg.commission_bps),
            spread=SpreadModel(cfg.execution.spread_bps),
            slippage=ConstantSlippageModel(cfg.execution.slippage_bps),
        ),
        cfg,
    )
    assert result.metadata["config_yaml_reflects_execution"] is False


def test_config_yaml_reflects_execution_catches_a_manipulated_volume_adv() -> None:
    """Two volume-based slippage models with the same scalar parameters but
    a different average_daily_volume must not compare as a match -- the
    ADV itself is part of what actually drives the cost, via a deterministic
    hash rather than embedding the whole matrix into metadata.json."""
    from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy

    data = _panel()
    base = _config({"name": "buy_and_hold"})
    cfg = base.revalidated_copy(
        update={
            "execution": base.execution.revalidated_copy(
                update={"slippage_model": "volume", "impact_coefficient": 0.1}
            )
        }
    )

    manipulated = ExecutionModel.from_config(
        cfg.execution, average_daily_volume=999_999.0
    )
    result = BacktestEngine().run(
        data, BuyAndHoldStrategy(), EqualWeightAllocator(), manipulated, cfg
    )
    assert result.metadata["config_yaml_reflects_execution"] is False


@pytest.mark.parametrize(
    "benchmark_kind", ["symbol", "equal_weight", "first_asset", "cash"]
)
def test_all_configured_benchmark_kinds_run_end_to_end(benchmark_kind: str) -> None:
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    benchmark_update = {
        "benchmark_kind": benchmark_kind,
        "benchmark": (
            {"symbol": "AAA", "source": "csv", "calendar": "XNYS"}
            if benchmark_kind == "symbol"
            else None
        ),
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
                update={"benchmark_kind": "equal_weight", "benchmark": None}
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
