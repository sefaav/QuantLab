"""Regression tests for execution behavior."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv
from tests.regression_helpers import (
    _flat_execution_model,
    _holdout_config,
    _rf_test_setup,
)

from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError


def test_target_volatility_applies_with_any_allocator() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv(sym, geometric_series(500, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [
            ("AAA", 1, 0.0007),
            ("BBB", 2, 0.0003),
            ("CCC", 3, 0.0005),
        ]
    ]
    data = pd.concat(frames, ignore_index=True)
    base: dict[str, Any] = {
        "experiment_name": "x",
        "data": {
            "source": "csv",
            "market_calendar": "XNYS",
            "symbols": ["AAA", "BBB", "CCC"],
            "start_date": "2020-01-01",
            "end_date": "2021-06-01",
        },
        "strategy": {
            "name": "cross_sectional_momentum",
            "parameters": {
                "lookback_period": 60,
                "skip_period": 5,
                "top_fraction": 0.5,
            },
        },
        "portfolio": {"allocator": "inverse_volatility", "volatility_window": 30},
        "execution": {"commission_bps": 2.0, "spread_bps": 3.0, "slippage_bps": 2.0},
        "backtest": {"initial_capital": 100_000},
    }
    cfg_no_target = ExperimentConfig.from_dict(base)
    cfg_with_target = ExperimentConfig.from_dict(
        {
            **base,
            "portfolio": {
                **base["portfolio"],
                "target_volatility": 0.05,
                "maximum_leverage": 1.5,
            },
        }
    )
    r1 = run_backtest_from_config(data, cfg_no_target)
    r2 = run_backtest_from_config(data, cfg_with_target)
    assert (
        abs(r2.metrics["annualized_volatility"] - r1.metrics["annualized_volatility"])
        > 0.005
    )


def test_inverse_volatility_allocator_honors_maximum_weight(
    synthetic_panel: pd.DataFrame,
) -> None:
    from quantlab.portfolio.allocator import InverseVolatilityAllocator

    idx = synthetic_panel["timestamp"].drop_duplicates().sort_values()
    signals = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB", "CCC"])
    alloc = InverseVolatilityAllocator(volatility_window=30, maximum_weight=0.30)
    weights = alloc.allocate(signals, synthetic_panel).dropna()
    assert weights.abs().max().max() <= 0.30 + 1e-6


def test_custom_rebalance_raises() -> None:
    from quantlab.portfolio.rebalancing import rebalance_dates

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    with pytest.raises(InvalidConfigurationError):
        rebalance_dates(idx, "custom")


def test_trade_log_uses_prior_period_price() -> None:
    from quantlab.backtesting.trade_log import build_trade_log
    from quantlab.execution.slippage import ConstantSlippageModel

    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    executed = pd.DataFrame({"AAA": [0.0, 1.0, 1.0, 1.0]}, index=idx)
    changes = pd.DataFrame({"AAA": [0.0, 1.0, 0.0, 0.0]}, index=idx)
    equity = pd.Series([100.0, 100.0, 110.0, 120.0], index=idx)
    prices = pd.DataFrame({"AAA": [10.0, 20.0, 30.0, 40.0]}, index=idx)

    trades = build_trade_log(
        executed,
        changes,
        equity,
        prices,
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_model=ConstantSlippageModel(0.0),
    )
    assert len(trades) == 1
    # The fill recorded at day index 1 (price 20) must use day 0's price (10),
    # not its own day's close.
    assert trades.iloc[0]["reference_price"] == pytest.approx(10.0)


def test_volume_slippage_is_not_degenerate() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv(sym, geometric_series(300, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [
            ("AAA", 1, 0.0007),
            ("BBB", 2, 0.0003),
            ("CCC", 3, 0.0005),
        ]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "vol_slip",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA", "BBB", "CCC"],
                "start_date": "2020-01-01",
                "end_date": "2020-10-01",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 60,
                    "skip_period": 5,
                    "top_fraction": 0.5,
                },
            },
            "portfolio": {"allocator": "equal_weight"},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 5.0,
                "slippage_model": "volume",
            },
            "backtest": {"initial_capital": 100_000},
        }
    )
    result = run_backtest_from_config(data, cfg)
    # The participation ratio (and therefore the slippage cost) must not
    # collapse to numerically zero just because a weight-change fraction is
    # compared against raw share volume with mismatched units.
    assert result.costs["slippage"].sum() > 1e-6


def test_negative_maximum_net_exposure_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
                "portfolio": {"maximum_net_exposure": -0.2},
            }
        )


def test_minimum_weight_above_maximum_weight_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
                "portfolio": {"minimum_weight": 0.5, "maximum_weight": 0.3},
            }
        )


def test_unknown_allocator_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
                "portfolio": {"allocator": "not_a_real_allocator"},
            }
        )


def test_unknown_slippage_model_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
                "execution": {"slippage_model": "typo"},
            }
        )


def test_bankruptcy_stops_trading_and_zeros_post_ruin_returns() -> None:
    from quantlab.backtesting.accounting import run_accounting

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.DataFrame({"A": [1.0] * 6}, index=idx)
    # A total wipeout on day 3, then ordinary-looking price action after.
    asset_returns = pd.DataFrame(
        {"A": [0.01, 0.01, -1.0, 0.02, -0.03, 0.05]}, index=idx
    )

    result = run_accounting(held, asset_returns, _flat_execution_model(), 100.0)
    assert result.equity.tolist() == [100.0, 101.0, 0.0, 0.0, 0.0, 0.0]
    # The fake post-ruin returns must be zeroed, not left at the raw
    # 0.02 / -0.03 / 0.05 the price series would otherwise produce.
    assert result.net_returns.iloc[3:].tolist() == [0.0, 0.0, 0.0]
    # No lingering "position" after ruin, and no phantom closing trade.
    assert (result.executed_weights.iloc[3:] == 0.0).all().all()
    assert (result.turnover.iloc[3:] == 0.0).all()


def test_volume_slippage_converges_to_self_consistent_net_equity() -> None:
    from quantlab.backtesting.accounting import run_accounting
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    held = pd.DataFrame({"A": np.where(np.arange(n) % 2 == 0, 1.0, -1.0)}, index=idx)
    asset_returns = pd.DataFrame({"A": rng.normal(0.0, 0.01, n)}, index=idx)
    adv = pd.DataFrame({"A": [50_000.0] * n}, index=idx)
    exec_cfg = ExecutionConfig(
        commission_bps=15.0,
        spread_bps=10.0,
        slippage_bps=0.0,
        slippage_model=cast(Any, "volume"),
        impact_coefficient=3.0,
    )
    model = ExecutionModel.from_config(exec_cfg, average_daily_volume=adv)
    result = run_accounting(held, asset_returns, model, initial_capital=100_000.0)

    # Gross and net equity must have meaningfully diverged in this
    # cost-heavy scenario — otherwise the test can't distinguish converged
    # net-equity sizing from a gross-equity shortcut.
    assert result.gross_equity.iloc[-1] / result.equity.iloc[-1] > 1.05

    # Self-consistency: recomputing slippage from `equity_for_costs` (what
    # this result says it used to size orders) must reproduce
    # `costs.slippage` exactly.
    executed = held.shift(1).fillna(0.0)
    weight_changes = executed - executed.shift(1).fillna(0.0)
    prior_equity = result.equity_for_costs.shift(1).fillna(100_000.0)
    recomputed_slippage = model.slippage.calculate(weight_changes, equity=prior_equity)
    pd.testing.assert_series_equal(
        recomputed_slippage, result.costs.slippage, check_names=False
    )

    # `equity_for_costs` must actually be the converged *net* equity, not
    # a gross-equity shortcut.
    assert result.equity_for_costs.iloc[-1] == pytest.approx(result.equity.iloc[-1])
    assert result.equity_for_costs.iloc[-1] != pytest.approx(
        result.gross_equity.iloc[-1]
    )


def test_solve_accounting_raises_when_fixed_point_does_not_converge() -> None:
    """An unconverged cost/equity solution must not be reported as final."""
    from quantlab.backtesting import accounting
    from quantlab.config import ExecutionConfig
    from quantlab.exceptions import BacktestError
    from quantlab.execution.execution_model import ExecutionModel

    original_cap = accounting._MAX_COST_EQUITY_ITERATIONS
    accounting._MAX_COST_EQUITY_ITERATIONS = 3
    try:
        n = 60
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.default_rng(3)
        held = pd.DataFrame(
            {"A": np.where(np.arange(n) % 2 == 0, 1.0, -1.0)}, index=idx
        )
        asset_returns = pd.DataFrame({"A": rng.normal(0.0, 0.01, n)}, index=idx)
        adv = pd.DataFrame({"A": [50_000.0] * n}, index=idx)
        exec_cfg = ExecutionConfig(
            commission_bps=15.0,
            spread_bps=10.0,
            slippage_bps=0.0,
            slippage_model=cast(Any, "volume"),
            impact_coefficient=3.0,
        )
        model = ExecutionModel.from_config(exec_cfg, average_daily_volume=adv)
        with pytest.raises(BacktestError, match="did not converge"):
            accounting.run_accounting(
                held, asset_returns, model, initial_capital=100_000.0
            )
    finally:
        accounting._MAX_COST_EQUITY_ITERATIONS = original_cap


def test_extreme_commission_does_not_produce_negative_notional() -> None:
    from quantlab.backtesting.accounting import run_accounting
    from quantlab.backtesting.trade_log import build_trade_log
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    held = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.0, -0.1, 0.0, 0.05]}, index=idx)
    exec_cfg = ExecutionConfig(
        commission_bps=20_000.0, spread_bps=0.0, slippage_bps=0.0
    )
    model = ExecutionModel.from_config(exec_cfg, average_daily_volume=1.0)

    result = run_accounting(held, asset_returns, model, initial_capital=100.0)
    prices = pd.DataFrame({"A": [100.0, 90.0, 90.0, 94.5]}, index=idx)
    trades = build_trade_log(
        result.executed_weights,
        result.weight_changes,
        result.equity,
        prices,
        commission_bps=20_000.0,
        spread_bps=0.0,
        slippage_model=model.slippage,
    )
    assert (trades["traded_notional"] >= 0.0).all()
    assert (trades["commission"] >= 0.0).all()


def test_volume_slippage_rejects_negative_params() -> None:
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    with pytest.raises(ValueError, match="base_slippage_bps"):
        VolumeBasedSlippageModel(base_slippage_bps=-1.0)
    with pytest.raises(ValueError, match="impact_coefficient"):
        VolumeBasedSlippageModel(impact_coefficient=-0.5)


def test_cost_and_slippage_models_reject_nan_and_infinity() -> None:
    from quantlab.execution.costs import CommissionModel, SpreadModel
    from quantlab.execution.slippage import (
        ConstantSlippageModel,
        VolumeBasedSlippageModel,
    )

    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            CommissionModel(bad)
        with pytest.raises(ValueError, match="finite"):
            SpreadModel(bad)
        with pytest.raises(ValueError, match="finite"):
            ConstantSlippageModel(bad)
        with pytest.raises(ValueError, match="finite"):
            VolumeBasedSlippageModel(base_slippage_bps=bad)
        with pytest.raises(ValueError, match="finite"):
            VolumeBasedSlippageModel(impact_coefficient=bad)
    # Sanity: ordinary, finite values remain accepted.
    assert CommissionModel(5.0).commission_bps == 5.0


def test_volume_slippage_rejects_invalid_average_daily_volume() -> None:
    import numpy as np
    import pandas as pd

    from quantlab.execution.slippage import VolumeBasedSlippageModel

    for bad_scalar in (-100.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="average_daily_volume"):
            VolumeBasedSlippageModel(average_daily_volume=bad_scalar)

    negative_cell = pd.DataFrame({"A": [100.0, -50.0], "B": [200.0, 300.0]})
    with pytest.raises(ValueError, match="average_daily_volume"):
        VolumeBasedSlippageModel(average_daily_volume=negative_cell)

    infinite_cell = pd.DataFrame({"A": [100.0, float("inf")], "B": [200.0, 300.0]})
    with pytest.raises(ValueError, match="average_daily_volume"):
        VolumeBasedSlippageModel(average_daily_volume=infinite_cell)

    # Sanity: a missing (NaN) cell -- the documented "no volume data yet"
    # case -- remains accepted, and ordinary positive values are unaffected.
    nan_cell = pd.DataFrame({"A": [100.0, np.nan], "B": [200.0, 300.0]})
    VolumeBasedSlippageModel(average_daily_volume=nan_cell)
    VolumeBasedSlippageModel(average_daily_volume=100.0)


def test_run_accounting_raises_on_a_non_finite_cost() -> None:
    from quantlab.backtesting.accounting import run_accounting
    from quantlab.exceptions import BacktestError
    from quantlab.execution.costs import CommissionModel, SpreadModel
    from quantlab.execution.execution_model import ExecutionModel
    from quantlab.execution.slippage import ConstantSlippageModel

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    weights = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01, 0.01]}, index=idx)

    commission = CommissionModel(1.0)
    commission.commission_bps = float("nan")  # bypass the constructor guard
    exec_model = ExecutionModel(
        commission, SpreadModel(0.0), ConstantSlippageModel(0.0)
    )

    with pytest.raises(BacktestError, match="commission cost is negative"):
        run_accounting(weights, asset_returns, exec_model, 100.0)


def test_cap_turnover_rejects_invalid_public_api_parameters() -> None:
    from quantlab.portfolio.rebalancing import cap_turnover

    targets = pd.DataFrame({"A": [0.5, 0.5]})
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(InvalidConfigurationError):
            cap_turnover(targets, maximum_turnover=bad)
    for name in ("maximum_weight", "maximum_gross_exposure", "maximum_net_exposure"):
        for bad in (-0.1, float("nan")):
            with pytest.raises(InvalidConfigurationError):
                cap_turnover(targets, maximum_turnover=0.5, **{name: bad})  # type: ignore[arg-type]
    # Sanity: ordinary usage remains unaffected.
    out = cap_turnover(targets, maximum_turnover=0.3)
    assert out.to_numpy().tolist() == [[0.3], [0.5]]


def test_maximum_positions_redistributes_freed_capital() -> None:
    """Dropping positions beyond `maximum_positions` must redistribute the
    freed capital onto the survivors, the same as `maximum_weight`
    already does — not leave the book silently under-invested."""
    from quantlab.portfolio.constraints import ConstraintSet

    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame(
        [[0.3, 0.3, 0.2, 0.1, 0.1]], index=idx, columns=["A", "B", "C", "D", "E"]
    )
    out = ConstraintSet(maximum_positions=3).apply(weights)
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    # Relative proportions among the survivors are preserved.
    assert out["A"].iloc[0] == pytest.approx(out["B"].iloc[0])
    assert out["A"].iloc[0] > out["C"].iloc[0]
    assert out["D"].iloc[0] == 0.0
    assert out["E"].iloc[0] == 0.0


def test_minimum_weight_redistributes_freed_capital() -> None:
    """Zeroing out dust positions below `minimum_weight` must redistribute
    the freed capital onto the survivors, not leave it
    un-redistributed."""
    from quantlab.portfolio.constraints import ConstraintSet

    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame(
        [[0.5, 0.45, 0.03, 0.02]], index=idx, columns=["A", "B", "C", "D"]
    )
    out = ConstraintSet(minimum_weight=0.05).apply(weights)
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert out["C"].iloc[0] == 0.0
    assert out["D"].iloc[0] == 0.0


def test_redistribution_does_not_manufacture_exposure_from_nothing() -> None:
    """If every position in a row is dropped, there is nothing left to
    redistribute onto — the row must stay at zero, not blow up."""
    from quantlab.portfolio.constraints import ConstraintSet

    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame([[0.01, 0.01, 0.01]], index=idx, columns=["A", "B", "C"])
    out = ConstraintSet(minimum_weight=0.5).apply(weights)
    assert (out.iloc[0] == 0.0).all()


def test_minimum_weight_survives_exposure_caps_applied_after_it() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame([[0.5, 0.5]], index=idx, columns=["A", "B"])
    out = ConstraintSet(minimum_weight=0.30, maximum_gross_exposure=0.50).apply(weights)
    # Every surviving weight must be either exactly 0 or >= minimum_weight —
    # 0.25 (what a naive single-pass gross cap produces) is neither.
    values = out.iloc[0]
    assert ((values == 0.0) | (values.abs() >= 0.30 - 1e-9)).all(), values.tolist()


def test_constraint_set_does_not_needlessly_liquidate_the_whole_book() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame([[0.5, 0.5]], index=idx, columns=["A", "B"])
    out = ConstraintSet(minimum_weight=0.30, maximum_gross_exposure=0.50).apply(weights)
    values = out.iloc[0]
    assert values.abs().sum() == pytest.approx(0.5)
    assert not (values == 0.0).all()
    assert ((values == 0.0) | (values.abs() >= 0.30 - 1e-9)).all(), values.tolist()


def test_constraint_set_liquidation_rescue_never_breaches_exposure() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    violations: list[str] = []
    for trial in range(2000):
        rng = np.random.default_rng(trial)
        n_symbols = int(rng.integers(1, 6))
        min_weight = float(rng.uniform(0.05, 0.4))
        max_gross = float(rng.uniform(0.2, 1.5)) if rng.random() < 0.7 else None
        max_net = float(rng.uniform(0.1, 1.0)) if rng.random() < 0.7 else None
        max_leverage = float(rng.uniform(0.2, 1.5)) if rng.random() < 0.3 else None
        max_positions = (
            int(rng.integers(1, n_symbols + 1)) if rng.random() < 0.3 else None
        )
        cs = ConstraintSet(
            minimum_weight=min_weight,
            maximum_gross_exposure=max_gross,
            maximum_net_exposure=max_net,
            maximum_leverage=max_leverage,
            maximum_positions=max_positions,
        )
        raw = pd.DataFrame(
            [rng.uniform(-1, 1, n_symbols)],
            columns=[f"S{i}" for i in range(n_symbols)],
        )
        row = cs.apply(raw).iloc[0].to_numpy()
        tol = 1e-6
        if max_gross is not None and np.abs(row).sum() > max_gross + tol:
            violations.append(f"trial {trial}: gross exposure breached")
        if max_leverage is not None and np.abs(row).sum() > max_leverage + tol:
            violations.append(f"trial {trial}: leverage breached")
        if max_net is not None and abs(row.sum()) > max_net + tol:
            violations.append(f"trial {trial}: net exposure breached")
        bad = [v for v in row if 1e-9 < abs(v) < min_weight - 1e-9]
        if bad:
            violations.append(f"trial {trial}: minimum_weight breached ({bad})")
    assert not violations, "\n".join(violations)


def test_constraint_set_rescue_respects_maximum_weight() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    cs = ConstraintSet(
        maximum_weight=0.3,
        minimum_weight=0.25,
        maximum_gross_exposure=1.0,
        maximum_leverage=10.0,
    )
    raw = pd.DataFrame({"A": [0.6], "B": [0.4]})
    out = cs.apply(raw)
    assert out.abs().max(axis=1).iloc[0] <= 0.3 + 1e-9


def test_constraint_set_rescue_preserves_numeric_column_labels() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    cs = ConstraintSet(
        minimum_weight=0.3,
        maximum_net_exposure=0.1,
        maximum_leverage=1.0,
    )
    raw = pd.DataFrame({0: [0.5], 1: [0.5], 2: [-0.5]})
    out = cs.apply(raw)
    assert out.columns.tolist() == [0, 1, 2]
    assert all(isinstance(c, int) for c in out.columns)
    assert out.loc[0, 0] == pytest.approx(0.5)
    assert out.loc[0, 2] == pytest.approx(-0.5)


def test_constraint_set_maximum_weight_fuzz_no_violations() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    violations: list[str] = []
    for trial in range(3000):
        rng = np.random.default_rng(trial)
        n_symbols = int(rng.integers(1, 6))
        max_weight = float(rng.uniform(0.2, 0.9))
        min_weight = float(rng.uniform(0.01, max_weight))
        max_gross = float(rng.uniform(0.3, 1.5)) if rng.random() < 0.7 else None
        max_leverage = float(rng.uniform(0.3, 1.5)) if rng.random() < 0.7 else None
        max_net = float(rng.uniform(0.1, 1.0)) if rng.random() < 0.5 else None
        max_positions = (
            int(rng.integers(1, n_symbols + 1)) if rng.random() < 0.5 else None
        )
        cs = ConstraintSet(
            maximum_weight=max_weight,
            minimum_weight=min_weight,
            maximum_gross_exposure=max_gross,
            maximum_leverage=max_leverage,
            maximum_net_exposure=max_net,
            maximum_positions=max_positions,
            long_only=bool(rng.random() < 0.3),
        )
        raw = pd.DataFrame(
            [rng.uniform(-1, 1, n_symbols)],
            columns=[f"S{i}" for i in range(n_symbols)],
        )
        worst = cs.apply(raw).abs().max(axis=1).iloc[0]
        if worst > max_weight + 1e-6:
            violations.append(
                f"trial {trial}: weight {worst} exceeds maximum_weight {max_weight}"
            )
    assert not violations, "\n".join(violations)


def test_minimum_weight_zeroing_does_not_worsen_net_exposure() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    max_gross, max_net, max_positions, min_weight = 1.0, 0.5, 3, 0.05
    cs = ConstraintSet(
        maximum_gross_exposure=max_gross,
        maximum_net_exposure=max_net,
        maximum_positions=max_positions,
        minimum_weight=min_weight,
    )
    violations = []
    for trial in range(500):
        rng = np.random.default_rng(trial)
        n_symbols = 5
        raw = pd.DataFrame(
            [rng.uniform(-1, 1, n_symbols), rng.uniform(-1, 1, n_symbols)],
            columns=[f"S{i}" for i in range(n_symbols)],
        )
        out = cs.apply(raw)
        net = out.sum(axis=1)
        gross = out.abs().sum(axis=1)
        active = (out.abs() > 1e-9).sum(axis=1)
        dust = ((out.abs() > 1e-9) & (out.abs() < min_weight)).sum(axis=1)
        tol = 1e-6
        if (net.abs() > max_net + tol).any():
            violations.append(f"trial {trial}: net exposure breached")
        if (gross > max_gross + tol).any():
            violations.append(f"trial {trial}: gross exposure breached")
        if (active > max_positions).any():
            violations.append(f"trial {trial}: position count breached")
        if (dust > 0).any():
            violations.append(f"trial {trial}: dust below minimum_weight found")
    assert not violations, "\n".join(violations)


def test_maximum_turnover_field_exists_and_is_enforced() -> None:
    """`PortfolioConfig.maximum_turnover` is a real field wired into
    `BacktestEngine.run` via `cap_turnover`, not just accepted and
    ignored."""
    from quantlab.config import PortfolioConfig

    cfg = PortfolioConfig(allocator="equal_weight", maximum_turnover=0.1)
    assert cfg.maximum_turnover == pytest.approx(0.1)

    from quantlab.portfolio.rebalancing import cap_turnover

    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, -1.0]}, index=idx)
    capped = cap_turnover(held, maximum_turnover=0.1)
    # Turnover from period 0 -> 1 is capped at 0.1 (L1), not the full 2.0 move.
    turnover = (capped.iloc[1] - capped.iloc[0]).abs().sum()
    assert turnover == pytest.approx(0.1)


def test_execution_model_rejects_a_negative_cost_component_even_if_it_cancels_out() -> (
    None
):
    from quantlab.exceptions import BacktestError
    from quantlab.execution.costs import CommissionModel, SpreadModel
    from quantlab.execution.execution_model import ExecutionModel
    from quantlab.execution.slippage import ConstantSlippageModel

    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    weight_changes = pd.DataFrame({"A": [0.1, 0.1]}, index=idx)

    commission = CommissionModel(1.0)
    spread = SpreadModel(1.0)
    # Directly mutate a constructed model's rate to a negative value,
    # bypassing its own constructor guard -- exactly the scenario a custom
    # `SlippageModel` subclass with a looser contract could also produce.
    spread.spread_bps = -2.0 * commission.commission_bps
    exec_model = ExecutionModel(commission, spread, ConstantSlippageModel(0.0))

    with pytest.raises(BacktestError, match="spread cost is negative"):
        exec_model.compute(weight_changes)


def test_btc_trend_sizing_and_annualisation_live_in_portfolio() -> None:
    """BTC trend direction stays separate from portfolio risk sizing."""
    import pathlib

    from quantlab.backtesting.runner import build_strategy_from_config

    cfg = ExperimentConfig.from_yaml(
        pathlib.Path(__file__).resolve().parents[2] / "configs" / "btc_trend.yaml"
    )
    assert "periods_per_year" not in cfg.strategy.parameters
    strategy = build_strategy_from_config(cfg)
    assert not hasattr(strategy, "periods_per_year")
    assert cfg.periods_per_year == 365
    assert cfg.portfolio.target_volatility == pytest.approx(0.40)


def test_execution_delay_is_a_true_resimulation_not_a_returns_shift() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv(sym, geometric_series(300, mu, 0.015, 100.0, seed=seed))
        for sym, seed, mu in [("AAA", 1, 0.0006), ("BBB", 2, 0.0003)]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA", "BBB"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "frequency": "1d",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 30,
                    "skip_period": 5,
                    "top_fraction": 0.5,
                },
            },
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "weekly"},
            "execution": {
                "commission_bps": 5.0,
                "spread_bps": 5.0,
                "slippage_bps": 5.0,
            },
            "backtest": {},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    base = run_backtest_from_config(data, cfg)
    delayed = run_backtest_from_config(data, cfg, execution_delay=1)

    # A genuine delay must change the trading schedule, hence cost/turnover —
    # the fake (returns-shift-only) version left these bit-for-bit identical.
    assert delayed.metrics["total_cost_fraction"] != pytest.approx(
        base.metrics["total_cost_fraction"]
    )
    assert delayed.metrics["annual_turnover"] != pytest.approx(
        base.metrics["annual_turnover"]
    )
    # And the returns curve itself must differ from the naive shift too.
    from quantlab.risk.stress import delay_execution

    naive_shift = delay_execution(base.returns, 1)
    assert not delayed.returns.equals(naive_shift)


def test_execution_delay_zero_is_a_no_op() -> None:
    """`execution_delay=0` (the default) must reproduce the exact baseline —
    the delay machinery must not perturb the normal path."""
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    base = run_backtest_from_config(data, cfg)
    explicit_zero = run_backtest_from_config(data, cfg, execution_delay=0)
    pd.testing.assert_series_equal(base.returns, explicit_zero.returns)


def test_scale_costs_slippage_mult_also_scales_impact_coefficient() -> None:
    from quantlab.risk.stress import scale_costs

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {
                "slippage_bps": 0.0,
                "slippage_model": "volume",
                "impact_coefficient": 0.2,
            },
            "backtest": {},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    scaled = scale_costs(cfg, slippage_mult=2.0)
    assert scaled.execution.impact_coefficient == pytest.approx(0.4)
    assert scaled.execution.slippage_bps == pytest.approx(0.0)


def test_trade_log_volume_slippage_matches_accounting() -> None:
    from quantlab.backtesting.accounting import run_accounting
    from quantlab.backtesting.trade_log import build_trade_log
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    held = pd.DataFrame({"A": [1.0, 1.0, 0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.0, 0.2, 0.1, 0.0]}, index=idx)
    # Deliberately high cost relative to a small ADV: at low cost levels
    # gross and net equity are nearly identical, so a test that (wrongly)
    # compares against gross equity would still pass by accident — this
    # needs enough cost drag for gross/net to visibly diverge, or the test
    # can't actually catch the equity-basis mistake it's guarding against.
    adv = pd.DataFrame({"A": [1_000.0] * 4}, index=idx)
    exec_cfg = ExecutionConfig(
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=50.0,
        slippage_model=cast(Any, "volume"),
        impact_coefficient=5.0,
    )
    model = ExecutionModel.from_config(exec_cfg, average_daily_volume=adv)
    accounting = run_accounting(held, asset_returns, model, initial_capital=1000.0)

    prices = pd.DataFrame({"A": [100.0] * 4}, index=idx)
    trades = build_trade_log(
        accounting.executed_weights,
        accounting.weight_changes,
        accounting.equity,
        prices,
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_model=model.slippage,
        slippage_equity=accounting.equity_for_costs,
    )
    # Reconstruct the aggregate accounting-level slippage cost in dollars and
    # compare per-date totals against the trade log's per-fill sum. Every
    # fraction-of-equity cost in this codebase (commission, spread, slippage,
    # net_returns itself) is consistently a fraction of the prior period's
    # *net* equity, not gross — `trade_log.py`'s own final dollar conversion
    # uses `prev_equity` (net) for exactly this reason, and the internal
    # volume-based *ratio* converges to net equity via `_solve_accounting`'s
    # fixed-point loop rather than using `gross_equity` (gross equity
    # diverges from net by exactly the cumulative cost drag — see the module
    # docstring). Comparing against gross equity here instead would only
    # pass by accident at low cost levels, where gross and net are nearly
    # identical; the elevated slippage_bps=50/impact_coefficient=5 above
    # makes them diverge enough for this comparison to actually be
    # meaningful.
    prior_net_equity = accounting.equity.shift(1).fillna(1000.0)
    aggregate_dollar_cost = accounting.costs.slippage * prior_net_equity
    trade_log_totals = trades.groupby("timestamp")["slippage_cost"].sum()
    for ts, expected_cost in aggregate_dollar_cost.items():
        if abs(expected_cost) < 1e-9:
            continue
        assert trade_log_totals.get(ts, 0.0) == pytest.approx(expected_cost, rel=1e-9)


def test_trade_log_defaults_to_equity_without_slippage_equity() -> None:
    from quantlab.backtesting.trade_log import build_trade_log
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)
    changes = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, index=idx)
    equity = pd.Series([1000.0, 1000.0, 1100.0], index=idx)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    model = VolumeBasedSlippageModel(
        base_slippage_bps=5.0,
        impact_coefficient=1.0,
        average_daily_volume=pd.DataFrame({"A": [1000.0] * 3}, index=idx),
    )
    kwargs: dict[str, Any] = {
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_model": model,
    }

    omitted = build_trade_log(executed, changes, equity, prices, **kwargs)
    explicit = build_trade_log(
        executed, changes, equity, prices, slippage_equity=equity, **kwargs
    )
    pd.testing.assert_frame_equal(omitted, explicit)

    # And it must be sensitive to slippage_equity, so the equality above
    # isn't vacuously true regardless of what's passed in.
    different = build_trade_log(
        executed, changes, equity, prices, slippage_equity=equity * 2.0, **kwargs
    )
    assert not different["slippage_cost"].equals(omitted["slippage_cost"])


def test_turnover_cap_completes_a_full_rotation_between_disjoint_sets() -> None:
    from quantlab.portfolio.constraints import ConstraintSet
    from quantlab.portfolio.rebalancing import cap_turnover

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    raw = pd.DataFrame(
        [[0.5, 0.5, 0.0, 0.0]] + [[0.0, 0.0, 0.5, 0.5]] * 5,
        index=idx,
        columns=["A", "B", "C", "D"],
    )
    constrained = ConstraintSet(maximum_positions=2).apply(raw)
    capped = cap_turnover(constrained, maximum_turnover=0.5)

    previous = pd.Series(0.0, index=raw.columns)
    for ts, row in capped.iterrows():
        realised_turnover = (row - previous).abs().sum()
        assert realised_turnover <= 0.5 + 1e-9, (
            f"turnover {realised_turnover} exceeds the 0.5 cap at {ts}"
        )
        previous = row

    # The rotation must still complete eventually (not get stuck at zero
    # forever) — by the last period the target's assets are fully held.
    assert capped.iloc[-1][["C", "D"]].sum() == pytest.approx(1.0)
    assert capped.iloc[-1][["A", "B"]].sum() == pytest.approx(0.0)


def test_engine_turnover_cap_never_exceeds_the_configured_budget() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv(sym, geometric_series(120, mu, 0.02, 100.0, seed=seed))
        for sym, seed, mu in [
            ("AAA", 1, 0.002),
            ("BBB", 2, -0.002),
            ("CCC", 3, 0.001),
            ("DDD", 4, -0.001),
        ]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA", "BBB", "CCC", "DDD"],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
                "frequency": "1d",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 10,
                    "skip_period": 0,
                    "top_fraction": 0.5,
                },
            },
            "portfolio": {
                "allocator": "equal_weight",
                "target_maximum_positions": 2,
                "maximum_turnover": 0.1,
                "rebalance_frequency": "daily",
            },
            "execution": {},
            "backtest": {},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    result = run_backtest_from_config(data, cfg)
    assert result.target_weights is not None
    target_nonzero = (result.target_weights.abs() > 1e-9).sum(axis=1)
    assert target_nonzero.max() <= 2
    realised_turnover = (
        (result.positions - result.positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    )
    assert realised_turnover.max() <= 0.1 + 1e-9


def test_infeasible_position_weight_combo_warns(caplog: Any) -> None:
    import logging

    from quantlab.config import PortfolioConfig

    with caplog.at_level(logging.WARNING):
        PortfolioConfig(
            allocator="equal_weight", target_maximum_positions=2, maximum_weight=0.30
        )
    assert any(
        "under-invest" in rec.message or "under-invest" in rec.getMessage()
        for rec in caplog.records
    )


def test_feasible_position_weight_combo_does_not_warn(caplog: Any) -> None:
    import logging

    from quantlab.config import PortfolioConfig

    with caplog.at_level(logging.WARNING):
        PortfolioConfig(
            allocator="equal_weight", target_maximum_positions=5, maximum_weight=0.30
        )
    assert not any("under-invest" in rec.getMessage() for rec in caplog.records)


def test_cap_turnover_net_exposure_is_preserved_by_construction() -> None:
    from quantlab.portfolio.rebalancing import cap_turnover

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        [[1.0, -0.6, 0.0], [1.0, -0.6, 0.0], [0.0, -0.6, 0.1]],
        index=idx,
        columns=["A", "B", "C"],
    )
    capped = cap_turnover(raw, maximum_turnover=1.0)
    net = capped.sum(axis=1)
    assert (net.abs() <= 0.5 + 1e-9).all(), f"net exposure breached: {net.tolist()}"


def test_cap_turnover_only_trades_on_rebalance_dates() -> None:
    from quantlab.config import RebalanceFrequency
    from quantlab.portfolio.rebalancing import (
        apply_rebalancing,
        cap_turnover,
        rebalance_dates,
    )

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    target = pd.DataFrame({"A": [1.0] * 6}, index=idx)
    rebalanced = apply_rebalancing(target, RebalanceFrequency.MONTHLY)
    dates = rebalance_dates(idx, RebalanceFrequency.MONTHLY)
    assert len(dates) == 1  # a single-month window has exactly one rebalance

    capped = cap_turnover(rebalanced, maximum_turnover=0.2, rebalance_index=dates)
    assert capped["A"].tolist() == [0.2] * 6

    # Without a schedule, a direct caller gets the "catch-up" behaviour
    # instead: turnover applies every day, not just at rebalances. This
    # confirms the difference above comes from `rebalance_index`, not
    # something else about the inputs.
    capped_no_schedule = cap_turnover(rebalanced, maximum_turnover=0.2)
    assert capped_no_schedule["A"].tolist() == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.0]
    )


def test_cap_turnover_dust_snap_never_exceeds_the_turnover_budget() -> None:
    from quantlab.portfolio.rebalancing import cap_turnover

    idx = pd.date_range("2020-01-01", periods=14, freq="D")
    targets = [0.9] * 6 + [0.0] * 8
    raw = pd.DataFrame({"A": targets}, index=idx)
    capped = cap_turnover(raw, maximum_turnover=0.3)
    values = capped["A"].tolist()

    # Core invariant: no single rebalance ever moves the position by more
    # than the configured turnover budget.
    previous = 0.0
    for v in values:
        assert abs(v - previous) <= 0.3 + 1e-9, (
            f"turnover budget exceeded: {previous} -> {v}"
        )
        previous = v

    # The exit leg (from period 6, where the target is 0) must be
    # monotonically non-increasing in magnitude — no bounce-back — and must
    # reach exactly zero.
    exit_leg = values[6:]
    assert all(exit_leg[i] >= exit_leg[i + 1] - 1e-9 for i in range(len(exit_leg) - 1))
    assert exit_leg[-1] == 0.0


def test_cap_turnover_convex_constraints_never_breached_property() -> None:
    from quantlab.portfolio.constraints import ConstraintSet
    from quantlab.portfolio.rebalancing import cap_turnover

    max_gross, max_net, max_weight, max_positions, min_weight = 1.0, 0.5, 0.4, 3, 0.05
    cs = ConstraintSet(
        maximum_gross_exposure=max_gross,
        maximum_net_exposure=max_net,
        maximum_weight=max_weight,
        maximum_positions=max_positions,
        minimum_weight=min_weight,
    )
    violations: list[str] = []
    for trial in range(800):
        rng = np.random.default_rng(trial)
        n_symbols = int(rng.integers(2, 8))
        n_periods = int(rng.integers(2, 5))
        raw = pd.DataFrame(
            [rng.uniform(-1, 1, n_symbols) for _ in range(n_periods)],
            columns=[f"S{i}" for i in range(n_symbols)],
        )
        constrained = cs.apply(raw)
        constrained.index = pd.date_range("2020-01-01", periods=n_periods, freq="D")
        maximum_turnover = float(rng.uniform(0.05, 0.6))
        out = cap_turnover(
            constrained,
            maximum_turnover=maximum_turnover,
            maximum_weight=max_weight,
            maximum_gross_exposure=max_gross,
            maximum_net_exposure=max_net,
        )
        net = out.sum(axis=1)
        gross = out.abs().sum(axis=1)
        turnover = out.diff().abs().sum(axis=1)
        turnover.iloc[0] = out.iloc[0].abs().sum()
        tol = 1e-6
        if (net.abs() > max_net + tol).any():
            violations.append(f"trial {trial}: net exposure breached")
        if (gross > max_gross + tol).any():
            violations.append(f"trial {trial}: gross exposure breached")
        if (out.abs() > max_weight + tol).to_numpy().any():
            violations.append(f"trial {trial}: per-asset weight cap breached")
        if (turnover > maximum_turnover + tol).any():
            violations.append(f"trial {trial}: turnover budget breached")
    assert not violations, "\n".join(violations)


def test_cap_turnover_converges_exactly_to_a_fixed_target() -> None:
    from quantlab.portfolio.constraints import ConstraintSet
    from quantlab.portfolio.rebalancing import cap_turnover

    max_gross, max_net, max_positions = 1.0, 0.5, 2
    cs = ConstraintSet(
        maximum_gross_exposure=max_gross,
        maximum_net_exposure=max_net,
        maximum_positions=max_positions,
    )
    failures: list[str] = []
    n_periods = 60
    for trial in range(300):
        rng = np.random.default_rng(trial)
        n_symbols = 4
        # Same target held for every rebalance after the first (a strategy
        # whose signal is unchanging) -- the scenario most likely to expose
        # a permanent deadlock, unlike a freshly-random target every period.
        first = rng.uniform(-1, 1, n_symbols)
        rest = rng.uniform(-1, 1, n_symbols)
        raw = pd.DataFrame(
            [first] + [rest] * (n_periods - 1),
            columns=[f"S{i}" for i in range(n_symbols)],
        )
        constrained = cs.apply(raw)
        constrained.index = pd.date_range("2020-01-01", periods=n_periods, freq="D")
        maximum_turnover = float(rng.uniform(0.05, 0.3))
        out = cap_turnover(
            constrained,
            maximum_turnover=maximum_turnover,
            maximum_gross_exposure=max_gross,
            maximum_net_exposure=max_net,
        )
        target_row = constrained.iloc[-1]
        final_row = out.iloc[-1]
        if not np.allclose(final_row.to_numpy(), target_row.to_numpy(), atol=1e-6):
            failures.append(
                f"trial {trial}: did not converge - target={target_row.tolist()} "
                f"final={final_row.tolist()}"
            )
    assert not failures, "\n".join(failures)


def test_cap_turnover_fuzz_settles_then_converges_exactly_with_exposure_caps() -> None:
    from quantlab.portfolio.rebalancing import cap_turnover

    def make_compliant_state(
        rng: np.random.Generator,
        n_assets: int,
        max_weight: float | None,
        max_gross: float | None,
        max_net: float | None,
    ) -> np.ndarray:
        state = rng.uniform(-1, 1, n_assets)
        if max_weight is not None:
            state = np.clip(state, -max_weight, max_weight)
        for _ in range(50):
            gross = np.abs(state).sum()
            net = state.sum()
            scale = 1.0
            if max_gross is not None and gross > max_gross:
                scale = min(scale, max_gross / gross)
            if max_net is not None and abs(net) > max_net:
                scale = min(scale, max_net / abs(net))
            if scale >= 1.0:
                break
            state = state * scale
        return state

    rng = np.random.default_rng(0)
    tested = 0
    non_convergent = 0
    for _ in range(600):
        n_assets = int(rng.integers(2, 6))
        max_turn = float(rng.uniform(0.05, 0.8))
        max_weight = float(rng.uniform(0.3, 1.0)) if rng.random() < 0.5 else None
        max_gross = float(rng.uniform(0.3, 1.5)) if rng.random() < 0.6 else None
        max_net = float(rng.uniform(0.1, 1.0)) if rng.random() < 0.6 else None
        cols = [f"A{i}" for i in range(n_assets)]

        previous = make_compliant_state(rng, n_assets, max_weight, max_gross, max_net)
        target = make_compliant_state(rng, n_assets, max_weight, max_gross, max_net)

        tested += 1
        settle_periods = max(20, int(np.ceil(np.abs(previous).sum() / max_turn)) + 5)
        main_periods = max(
            20, int(np.ceil(np.abs(target - previous).sum() / max_turn)) + 5
        )
        idx = pd.date_range(
            "2020-01-01", periods=settle_periods + main_periods, freq="D"
        )
        rows = [previous] * settle_periods + [target] * main_periods
        df = pd.DataFrame(rows, columns=cols, index=idx)
        out = cap_turnover(
            df,
            maximum_turnover=max_turn,
            maximum_weight=max_weight,
            maximum_gross_exposure=max_gross,
            maximum_net_exposure=max_net,
        )
        assert np.allclose(
            out.iloc[settle_periods - 1].to_numpy(), previous, atol=1e-6
        ), "settling phase itself did not reach the intended previous state"
        if not np.allclose(out.iloc[-1].to_numpy(), target, atol=1e-6):
            non_convergent += 1

    assert tested > 400
    assert non_convergent == 0, f"{non_convergent}/{tested} failed to converge exactly"


def test_cap_turnover_rejects_a_non_compliant_target_row() -> None:
    from quantlab.portfolio.rebalancing import cap_turnover

    idx = pd.date_range("2020-01-01", periods=1, freq="D")

    bad_weight = pd.DataFrame({"A": [0.9]}, index=idx)
    with pytest.raises(InvalidConfigurationError, match="maximum_weight"):
        cap_turnover(bad_weight, maximum_turnover=0.5, maximum_weight=0.5)

    bad_gross = pd.DataFrame({"A": [0.9], "B": [0.9]}, index=idx)
    with pytest.raises(InvalidConfigurationError, match="maximum_gross_exposure"):
        cap_turnover(bad_gross, maximum_turnover=0.5, maximum_gross_exposure=1.0)

    bad_net = pd.DataFrame({"A": [0.9], "B": [0.9]}, index=idx)
    with pytest.raises(InvalidConfigurationError, match="maximum_net_exposure"):
        cap_turnover(bad_net, maximum_turnover=0.5, maximum_net_exposure=1.0)

    bad_long_only = pd.DataFrame({"A": [-0.1]}, index=idx)
    with pytest.raises(InvalidConfigurationError, match="long_only"):
        cap_turnover(bad_long_only, maximum_turnover=0.5, long_only=True)


def test_cap_turnover_preserves_float_precision_for_integer_input() -> None:
    from quantlab.portfolio.rebalancing import cap_turnover

    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    targets = pd.DataFrame({"A": [1, 1]}, index=idx)
    assert targets["A"].dtype.kind == "i"

    out = cap_turnover(targets, maximum_turnover=0.5)
    assert out["A"].dtype == float
    assert out["A"].tolist() == pytest.approx([0.5, 1.0])
    turnover = out.diff().abs().sum(axis=1)
    turnover.iloc[0] = out.iloc[0].abs().sum()
    assert (turnover <= 0.5 + 1e-9).all()


def test_engine_only_trades_cap_turnover_on_rebalance_dates() -> None:
    """End-to-end: a monthly-rebalanced, turnover-capped backtest must not
    trade on non-rebalance dates."""
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _rf_test_setup()
    cfg = cfg.revalidated_copy(
        update={
            "portfolio": cfg.portfolio.revalidated_copy(
                update={
                    "maximum_turnover": 0.1,
                    "rebalance_frequency": "monthly",
                }
            )
        }
    )
    result = run_backtest_from_config(data, cfg)
    from quantlab.portfolio.rebalancing import rebalance_dates

    dates = rebalance_dates(pd.DatetimeIndex(result.positions.index), "monthly")
    # `result.positions` is `executed_weights = held.shift(1)` (the
    # look-ahead barrier), so a rebalance decided on date `d` only shows up
    # as a change one period *later* in `result.positions` — shift the
    # rebalance-date set to match before checking where trades landed.
    positions_index = result.positions.index
    rebalance_locs = positions_index.get_indexer(dates)
    shifted_locs = np.array(
        [loc + 1 for loc in rebalance_locs if 0 <= loc + 1 < len(positions_index)]
    )
    shifted_dates = positions_index[shifted_locs]
    non_rebalance_mask = ~positions_index.isin(shifted_dates)
    turnover_on_non_rebalance_days = (
        (result.positions - result.positions.shift(1).fillna(0.0))
        .abs()
        .sum(axis=1)[non_rebalance_mask]
    )
    assert (turnover_on_non_rebalance_days == 0.0).all()
