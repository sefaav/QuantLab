"""End-to-end backtest pipeline test.

Load data → features → strategy → backtest → metrics, exercising the full stack
through the config-driven runner.
"""

from __future__ import annotations

from typing import Any

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


def test_spy_portfolio_with_btcusdt_benchmark_runs_end_to_end() -> None:
    """Regression guard for the originally-reported SPY portfolio /
    BTCUSDT benchmark scenario (`MergeError: incompatible merge keys ...
    dtype('<M8[ms]') and dtype('<M8[us]')`). Did not reproduce with current
    code -- `session_labels`'s own merge_asof already normalizes both sides
    to `[ns]` first -- but had no permanent end-to-end guard against this
    class of cross-source datetime-precision issue reappearing silently
    (e.g. a pandas/pyarrow upgrade changing default parquet round-trip
    precision)."""
    spy_prices = geometric_series(60, mu=0.0005, sigma=0.008, s0=300.0, seed=11)
    spy = make_ohlcv("SPY", spy_prices, start="2019-01-02", freq="B")
    btc_prices = geometric_series(90, mu=0.001, sigma=0.02, s0=3000.0, seed=12)
    btc = make_ohlcv("BTCUSDT", btc_prices, start="2019-01-01", freq="D")
    data = pd.concat([spy, btc], ignore_index=True)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "spy_btc_benchmark",
            "data": {
                "instruments": [
                    {"symbol": "SPY", "source": "yahoo", "calendar": "XNYS"}
                ],
                "start_date": "2019-01-02",
                "end_date": "2019-03-15",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {
                "benchmark_kind": "symbol",
                "benchmark": {
                    "symbol": "BTCUSDT",
                    "source": "binance",
                    "calendar": "24/7",
                },
            },
        }
    )

    result = run_backtest_from_config(data, cfg)

    assert result.benchmark_returns is not None
    assert not result.benchmark_returns.isna().any()


def test_btcusdt_portfolio_with_spy_benchmark_runs_end_to_end() -> None:
    """A BTCUSDT portfolio with a SPY benchmark must run cleanly even
    though BTCUSDT trades on 2019-01-01, a date XNYS's own calendar marks
    as a holiday closure -- SPY has no observation for it at all."""
    btc_prices = geometric_series(45, mu=0.001, sigma=0.02, s0=3000.0, seed=21)
    btc = make_ohlcv("BTCUSDT", btc_prices, start="2019-01-01", freq="D")
    spy_prices = geometric_series(45, mu=0.0005, sigma=0.008, s0=300.0, seed=22)
    spy = make_ohlcv("SPY", spy_prices, start="2019-01-02", freq="B")
    data = pd.concat([btc, spy], ignore_index=True)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "btc_spy_benchmark",
            "data": {
                "instruments": [
                    {"symbol": "BTCUSDT", "source": "binance", "calendar": "24/7"}
                ],
                "start_date": "2019-01-01",
                "end_date": "2019-02-10",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {
                "benchmark_kind": "symbol",
                "benchmark": {"symbol": "SPY", "source": "yahoo", "calendar": "XNYS"},
            },
        }
    )

    result = run_backtest_from_config(data, cfg)

    assert result.benchmark_returns is not None
    assert not result.benchmark_returns.isna().any()
    assert result.benchmark_returns.iloc[0] == 0.0


def test_model_weight_drift_never_trades_a_closed_instrument() -> None:
    """A mixed-calendar portfolio (SPY on XNYS, BTCUSDT on 24/7) with
    `model_weight_drift=True` must never record a trade for SPY on a date
    XNYS's own calendar closes it -- even though BTCUSDT keeps trading,
    and even on a monthly rebalance date that happens to land on a day
    SPY itself is shut (e.g. 2019-01-01, New Year's Day). Regression
    guard for the anchor-collapse bug: a scalar, whole-row anchor flag
    used to force EVERY column to re-trade whenever ANY column anchored,
    including a currently-closed one."""
    btc_prices = geometric_series(150, mu=0.001, sigma=0.02, s0=3000.0, seed=31)
    btc = make_ohlcv("BTCUSDT", btc_prices, start="2019-01-01", freq="D")
    spy_prices = geometric_series(150, mu=0.0005, sigma=0.008, s0=300.0, seed=32)
    spy = make_ohlcv("SPY", spy_prices, start="2019-01-01", freq="D")
    data = pd.concat([btc, spy], ignore_index=True)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "mixed_calendar_drift",
            "data": {
                "instruments": [
                    {"symbol": "SPY", "source": "yahoo", "calendar": "XNYS"},
                    {"symbol": "BTCUSDT", "source": "binance", "calendar": "24/7"},
                ],
                "start_date": "2019-01-01",
                "end_date": "2019-05-30",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {
                "allocator": "equal_weight",
                "rebalance_frequency": "monthly",
                "model_weight_drift": True,
            },
            "execution": {"commission_bps": 5.0},
            "backtest": {"initial_capital": 100_000, "periods_per_year": 252},
        }
    )

    result = run_backtest_from_config(data, cfg)

    assert np.isfinite(result.equity_curve.to_numpy()).all()
    spy_trades = result.trades[result.trades["symbol"] == "SPY"]
    assert len(spy_trades) > 0  # sanity: SPY does trade on its own open days

    from quantlab.data.calendar import is_session_day

    spy_trade_dates = pd.DatetimeIndex(spy_trades["timestamp"])
    open_mask = is_session_day("XNYS", spy_trade_dates)
    assert open_mask.all(), (
        f"SPY has trade log rows on closed dates: "
        f"{spy_trade_dates[~open_mask].tolist()}"
    )


def test_rebalance_date_flag_is_never_true_on_a_closed_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `engine.py`'s `_align_bool` reused `executed_weights`
    (built for *weights*, where a closed row correctly repeats the last
    tradable row's frozen value) to align the `rebalance_date` boolean flag
    onto the executed timeline. Applied to a flag instead of a weight, that
    same repetition kept the flag True for every row a column stayed closed
    right after a landing -- `apply_weight_drift`'s own documented
    precondition explicitly forbids this, since it re-anchors ordinary
    debt to a stale target and executes an unscheduled trade the moment the
    column reopens. `rebalance_frequency=daily` deterministically triggers
    the closure-adjacent pattern via SPY's own weekend closures (no
    reliance on a specific holiday landing)."""
    btc_prices = geometric_series(150, mu=0.001, sigma=0.02, s0=3000.0, seed=31)
    btc = make_ohlcv("BTCUSDT", btc_prices, start="2019-01-01", freq="D")
    spy_prices = geometric_series(150, mu=0.0005, sigma=0.008, s0=300.0, seed=32)
    spy = make_ohlcv("SPY", spy_prices, start="2019-01-01", freq="D")
    data = pd.concat([btc, spy], ignore_index=True)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "mixed_calendar_daily_rebalance",
            "data": {
                "instruments": [
                    {"symbol": "SPY", "source": "yahoo", "calendar": "XNYS"},
                    {"symbol": "BTCUSDT", "source": "binance", "calendar": "24/7"},
                ],
                "start_date": "2019-01-01",
                "end_date": "2019-05-30",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {
                "allocator": "equal_weight",
                "rebalance_frequency": "daily",
                "model_weight_drift": True,
            },
            "execution": {"commission_bps": 5.0},
            "backtest": {"initial_capital": 100_000, "periods_per_year": 252},
        }
    )

    import quantlab.backtesting.engine as engine_mod

    captured: dict[str, pd.DataFrame] = {}
    orig_run_accounting = engine_mod.run_accounting

    def spy_run_accounting(*args: Any, **kwargs: Any) -> Any:
        captured["tradable"] = kwargs["tradable"].copy()
        captured["rebalance_date"] = kwargs["rebalance_date"].copy()
        return orig_run_accounting(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "run_accounting", spy_run_accounting)

    run_backtest_from_config(data, cfg)

    violation = captured["rebalance_date"] & ~captured["tradable"]
    assert not violation.to_numpy().any(), (
        f"rebalance_date is True on a closed row: {violation[violation.any(axis=1)]}"
    )


def test_model_weight_drift_changes_returns_and_rebalances_on_schedule() -> None:
    """`PortfolioConfig.model_weight_drift=True` (the default) must run
    cleanly end-to-end through the full config-driven pipeline and produce
    genuinely different (finite, sensible) equity from the legacy
    constant-weight step function (`model_weight_drift=False`) -- the whole
    point of the weight-drift feedback loop.

    Turnover must be exactly zero between scheduled rebalance dates --
    drift itself is never read back as a series of phantom trades, since
    `apply_weight_drift`'s own `trade_changes` output separates real
    trades from organic drift -- but NONZERO roughly once per scheduled
    rebalance date, even for a buy-and-hold/equal_weight strategy whose
    freshly-decided target NEVER numerically changes: real drift-driven
    turnover snaps the portfolio back to target on schedule via
    `rebalance_date`, rather than being silently absorbed into ongoing
    drift just because the target number happens to match the previous
    one. The legacy constant-weight baseline never rebalances at all past
    its first anchor (a genuinely constant target has nothing to correct
    back to), so drifted turnover is strictly higher in total.
    """
    data = _panel()
    base_cfg = _config({"name": "buy_and_hold"})
    legacy_cfg = base_cfg.revalidated_copy(
        update={
            "portfolio": base_cfg.portfolio.revalidated_copy(
                update={"model_weight_drift": False}
            )
        }
    )
    drift_cfg = base_cfg.revalidated_copy(
        update={
            "portfolio": base_cfg.portfolio.revalidated_copy(
                update={"model_weight_drift": True}
            )
        }
    )

    baseline = run_backtest_from_config(data, legacy_cfg)
    drifted = run_backtest_from_config(data, drift_cfg)

    assert np.isfinite(drifted.equity_curve.to_numpy()).all()
    assert not drifted.equity_curve.equals(baseline.equity_curve)
    assert drifted.metrics["annual_turnover"] > baseline.metrics["annual_turnover"]

    from quantlab.portfolio.rebalancing import rebalance_dates

    # One nonzero-turnover date per scheduled rebalance (the executed
    # timeline is the decision timeline shifted forward by exactly one
    # look-ahead-barrier row -- see quantlab.execution.orders.
    # executed_weights) -- never one per drift row (that would be the
    # phantom-turnover bug the drift/turnover separation in
    # apply_weight_drift's own trade_changes output exists to prevent).
    assert drifted.turnover is not None
    schedule = rebalance_dates(
        pd.DatetimeIndex(drifted.turnover.index),
        base_cfg.portfolio.rebalance_frequency,
    )
    nonzero_turnover_dates = drifted.turnover[drifted.turnover.abs() > 1e-9].index
    assert len(nonzero_turnover_dates) == len(schedule)


@pytest.mark.parametrize(
    ("frequency", "execution_delay"),
    [
        ("daily", 0),
        ("weekly", 0),
        ("monthly", 1),
        ("quarterly", 0),
    ],
)
def test_model_weight_drift_never_exceeds_maximum_turnover(
    frequency: str, execution_delay: int
) -> None:
    """A drift-caused anchor catch-up (including a constant-target
    schedule that never numerically changes -- see
    `apply_weight_drift`'s own `rebalance_date`-forced-rebalance
    mechanism) must never push realized turnover past
    `PortfolioConfig.maximum_turnover`, across rebalance frequencies and
    with a nonzero execution delay."""
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    cfg = cfg.revalidated_copy(
        update={
            "portfolio": cfg.portfolio.revalidated_copy(
                update={
                    "model_weight_drift": True,
                    "rebalance_frequency": frequency,
                    "maximum_turnover": 0.05,
                }
            )
        }
    )
    result = run_backtest_from_config(data, cfg, execution_delay=execution_delay)
    assert result.turnover is not None
    assert result.turnover.max() <= 0.05 + 1e-6


def test_model_weight_drift_progresses_toward_the_true_target_never_backward() -> None:
    """The executed position must climb monotonically toward the true
    target at exactly the configured `maximum_turnover` cap rate, never
    reversing, even once organic appreciation under
    `model_weight_drift=True` has carried the real position past an
    intermediate decision-level value. `maximum_turnover` must be applied
    ONLY ONCE -- at the decision level (`rebalance_and_cap_turnover`) OR
    in `apply_weight_drift`'s own ordinary-debt mechanism (the decision
    layer's own cap is disabled whenever the drift layer owns it
    instead), never both: double-applying it would hand the drift layer
    an already-capped INTERMEDIATE target instead of the constant TRUE
    target (1.0, for a single-instrument buy_and_hold), which the drift
    layer would then treat as "the" target -- capable of trading the
    portfolio BACKWARD, opposite the strategy's own direction, once
    organic appreciation had already carried the real position past that
    stale intermediate value."""
    prices = geometric_series(150, mu=0.01, sigma=0.0, s0=100.0, seed=1)
    data = make_ohlcv("AAA", prices, start="2019-01-01")
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "drift_no_reversal",
            "data": {
                "instruments": [{"symbol": "AAA", "source": "csv", "calendar": "XNYS"}],
                "start_date": "2019-01-01",
                "end_date": "2019-12-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {
                "allocator": "equal_weight",
                "maximum_turnover": 0.1,
                "model_weight_drift": True,
                "rebalance_frequency": "monthly",
            },
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000, "benchmark_kind": "cash"},
        }
    )

    result = run_backtest_from_config(data, cfg)

    # The decision-level target is the constant true target throughout --
    # never a stale, already-capped intermediate value.
    assert np.allclose(result.weights["AAA"].to_numpy(), 1.0)
    positions = result.positions["AAA"]
    assert positions.iloc[0] == pytest.approx(0.0)
    # Every executed transaction progresses toward the true target: the
    # position never decreases, and reaches it exactly once fully caught up.
    assert (positions.diff().dropna() >= -1e-9).all()
    assert positions.iloc[-1] == pytest.approx(1.0)
    assert result.turnover is not None
    assert result.turnover.max() <= 0.1 + 1e-6


@pytest.mark.parametrize(
    ("maximum_gross_exposure", "maximum_leverage", "expected_gross_cap"),
    [
        (None, 1.5, 1.5),  # unset -> maximum_leverage alone
        (0.8, 1.5, 0.8),  # maximum_gross_exposure is the tighter cap
        (2.0, 1.5, 1.5),  # maximum_leverage is the tighter cap
    ],
)
def test_model_weight_drift_forwards_the_combined_gross_exposure_cap(
    maximum_gross_exposure: float | None,
    maximum_leverage: float,
    expected_gross_cap: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """engine.py combines `maximum_gross_exposure`/`maximum_leverage` via
    `min(...)` (or falls back to `maximum_leverage` alone when `maximum_
    gross_exposure` is unset) before forwarding into `run_accounting`'s own
    `maximum_gross_exposure` -- the sole place the drift-compliance LP
    reads it once `model_weight_drift` is active. Before this test, zero
    coverage exercised this specific forwarding: a reversed ternary or a
    wrong field would have passed every other test. Captures the actual
    kwargs `run_accounting` is called with (mirroring
    `test_rebalance_date_flag_is_never_true_on_a_closed_row`'s own
    technique) rather than inferring the cap indirectly from whether a
    breach happens to occur."""
    data = _panel()
    cfg = _config({"name": "buy_and_hold"})
    portfolio_update: dict[str, Any] = {
        "model_weight_drift": True,
        "maximum_leverage": maximum_leverage,
    }
    if maximum_gross_exposure is not None:
        portfolio_update["maximum_gross_exposure"] = maximum_gross_exposure
    cfg = cfg.revalidated_copy(
        update={"portfolio": cfg.portfolio.revalidated_copy(update=portfolio_update)}
    )

    import quantlab.backtesting.engine as engine_mod

    captured: dict[str, Any] = {}
    orig_run_accounting = engine_mod.run_accounting

    def spy_run_accounting(*args: Any, **kwargs: Any) -> Any:
        captured["maximum_gross_exposure"] = kwargs["maximum_gross_exposure"]
        captured["maximum_weight"] = kwargs["maximum_weight"]
        captured["maximum_net_exposure"] = kwargs["maximum_net_exposure"]
        captured["long_only"] = kwargs["long_only"]
        return orig_run_accounting(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "run_accounting", spy_run_accounting)

    run_backtest_from_config(data, cfg)

    assert captured["maximum_gross_exposure"] == pytest.approx(expected_gross_cap)
    assert captured["maximum_weight"] == cfg.portfolio.maximum_weight
    assert captured["maximum_net_exposure"] == cfg.portfolio.maximum_net_exposure
    assert captured["long_only"] == cfg.portfolio.long_only


def test_model_weight_drift_restores_a_breach_of_combined_caps_end_to_end() -> None:
    """End-to-end (not a unit test of the LP directly): a two-asset
    portfolio with `model_weight_drift=True`, a tight `maximum_weight`, a
    long rebalance interval (quarterly, so drift has room to build up) and
    one asset engineered to strongly outgrow the other must genuinely
    breach `maximum_weight` from pure organic drift, and that breach must
    never persist for more than the one documented row of look-ahead-free
    lag (`apply_weight_drift`'s own temporal convention: a breach detected
    at `t` queues a correction that lands at `t+1`, never retroactively
    touching `t` itself) -- i.e. two CONSECUTIVE breaching rows for the
    same symbol would mean a queued correction failed to land on schedule.
    The trade log's `drift_compliance` adjustment code must also actually
    appear, proving the LP genuinely fired rather than the test vacuously
    passing because no breach ever occurred. This is exactly the
    combination (`model_weight_drift` + these caps, through the full
    `engine.py` pipeline) the prior integration tests never exercised
    together."""
    winner = geometric_series(260, mu=0.02, sigma=0.01, s0=100.0, seed=11)
    loser = geometric_series(260, mu=-0.005, sigma=0.01, s0=100.0, seed=12)
    data = pd.concat(
        [
            make_ohlcv("WIN", winner, start="2019-01-01"),
            make_ohlcv("LOSE", loser, start="2019-01-01"),
        ],
        ignore_index=True,
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "drift_compliance_end_to_end",
            "data": {
                "instruments": [
                    {"symbol": "WIN", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "LOSE", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2019-01-01",
                "end_date": "2019-12-31",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {
                "allocator": "equal_weight",
                "rebalance_frequency": "quarterly",
                "model_weight_drift": True,
                "maximum_weight": 0.55,
                "maximum_gross_exposure": 1.0,
                "maximum_leverage": 1.0,
                "long_only": True,
            },
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {"initial_capital": 100_000, "periods_per_year": 252},
        }
    )

    result = run_backtest_from_config(data, cfg)

    # `positions` is `accounting.executed_weights` -- the REAL, post-drift,
    # post-compliance-correction book (`weights` is the pre-drift decision
    # timeline, which never breaches anything by construction and so would
    # not exercise this at all).
    executed = result.positions
    assert executed is not None
    breach = executed.abs() > 0.55 + 1e-6
    assert breach.to_numpy().any(), (
        "expected WIN's organic drift to actually breach maximum_weight at "
        "least once -- otherwise this test does not engineer the scenario "
        "it claims to"
    )
    consecutive_breach = breach & breach.shift(1, fill_value=False)
    assert not consecutive_breach.to_numpy().any(), (
        f"a breach persisted for 2+ consecutive rows (never restored on "
        f"schedule): {consecutive_breach[consecutive_breach.any(axis=1)]}"
    )
    assert (executed.sum(axis=1).abs().to_numpy() <= 1.0 + 1e-6).all()
    assert (executed.to_numpy() >= -1e-9).all()  # long_only
    codes = result.trades["adjustment_reason_codes"].fillna("")
    assert codes.str.contains("drift_compliance").any(), (
        "expected at least one drift_compliance-attributed correction -- "
        "otherwise this test does not actually exercise the LP path"
    )


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
