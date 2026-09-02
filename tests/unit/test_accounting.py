"""Accounting and anti-look-ahead tests.

These are the most important correctness tests in the project: they prove the
engine cannot earn a return on a position before it was actually put on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.accounting import (
    _detect_stop_loss_take_profit,
    compute_asset_returns,
    run_accounting,
)
from quantlab.config import ExecutionConfig
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionModel


def _zero_cost_model() -> ExecutionModel:
    return ExecutionModel.from_config(
        ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    )


def test_no_lookahead_gain_starts_one_period_late() -> None:
    """Signal goes long at date 2 → gain must start at date 3.

    The position decided at *t* only earns from *t+1* because executed weights
    are held-weights shifted by one.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    # Asset jumps +10% on date index 3 (the 4th day).
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.0, 0.10, 0.0]}, index=idx)
    # Held weights: flat, then long from date index 2 onward.
    held = pd.DataFrame({"AAA": [0.0, 0.0, 1.0, 1.0, 1.0]}, index=idx)

    acc = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)

    # executed = held.shift(1) → long only from date index 3.
    assert acc.executed_weights["AAA"].iloc[2] == 0.0
    assert acc.executed_weights["AAA"].iloc[3] == 1.0
    # The +10% on date 3 IS captured (position was on from date 3).
    assert acc.gross_returns.iloc[3] == pytest.approx(0.10)
    # Critically, date 2 earns nothing despite the signal turning long at date 2.
    assert acc.gross_returns.iloc[2] == pytest.approx(0.0)


def test_short_position_profits_on_decline() -> None:
    """A negative weight on a negative return yields a gain."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, -0.05]}, index=idx)
    held = pd.DataFrame({"AAA": [-1.0, -1.0, -1.0]}, index=idx)
    acc = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    # executed[-1] = -1, return[-1] = -5% → contribution +5%.
    assert acc.gross_returns.iloc[2] == pytest.approx(0.05)


def test_equity_compounding_formula() -> None:
    """equity_t = equity_{t-1} × (1 + net_return_t), equity_0 = capital."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.1, -0.1, 0.2]}, index=idx)
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    acc = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    manual = 100_000.0
    for r in acc.net_returns:
        manual *= 1.0 + r
    assert acc.equity.iloc[-1] == pytest.approx(manual)


def test_costs_reduce_net_below_gross() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.01, 0.01, 0.01]}, index=idx)
    # Alternate weights so there is turnover (and therefore cost).
    held = pd.DataFrame({"AAA": [1.0, 0.0, 1.0, 0.0]}, index=idx)
    costly = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=50.0, spread_bps=50.0, slippage_bps=50.0)
    )
    acc = run_accounting(held, asset_returns, costly, 100_000.0)
    assert acc.equity.iloc[-1] < acc.gross_equity.iloc[-1]
    assert (acc.costs.total >= 0).all()


def test_compute_asset_returns_matches_pct_change() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    rets = compute_asset_returns(prices)
    assert np.isnan(rets["AAA"].iloc[0])
    assert rets["AAA"].iloc[1] == pytest.approx(0.10)
    assert rets["AAA"].iloc[2] == pytest.approx(-0.10)


def test_missing_return_for_held_position_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="Asset return is missing"):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


@pytest.mark.parametrize("bad_capital", [0.0, -1.0, np.nan, np.inf, True])
def test_invalid_initial_capital_raises(bad_capital: object) -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="initial_capital"):
        run_accounting(held, asset_returns, _zero_cost_model(), bad_capital)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model_weight_drift": "yes"}, "model_weight_drift"),
        ({"long_only": 1}, "long_only"),
        ({"stop_loss_pct": -1.0}, "stop_loss_pct"),
        ({"stop_loss_pct": 0.0}, "stop_loss_pct"),
        ({"take_profit_pct": -0.5}, "take_profit_pct"),
        ({"model_weight_drift": True, "maximum_weight": -0.1}, "maximum_weight"),
        ({"model_weight_drift": True, "maximum_weight": 1.5}, "maximum_weight"),
        (
            {"model_weight_drift": True, "maximum_gross_exposure": -0.1},
            "maximum_gross_exposure",
        ),
        (
            {"model_weight_drift": True, "maximum_net_exposure": -0.1},
            "maximum_net_exposure",
        ),
        (
            {"model_weight_drift": True, "maximum_turnover": -0.1},
            "maximum_turnover",
        ),
        (
            {"model_weight_drift": True, "maximum_turnover": 0.0},
            "maximum_turnover",
        ),
    ],
)
def test_run_accounting_rejects_invalid_direct_api_arguments(
    kwargs: dict[str, object], match: str
) -> None:
    """A direct caller of `run_accounting` (bypassing PortfolioConfig's own
    field validation and strategies.base.validate_risk_control_parameters
    entirely) must not be able to silently pass a truthy non-bool flag, a
    non-positive stop_loss_pct/take_profit_pct, or an out-of-range
    exposure cap that would otherwise reach the drift-compliance LP as a
    confusing "bug in the algorithm" error instead of a clear,
    immediate one."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match=match):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            **kwargs,  # type: ignore[arg-type]
        )


def test_run_accounting_rejects_a_non_execution_model_instance() -> None:
    """A wrong-type `execution_model` must raise `BacktestError` (per this
    function's own documented contract), never let a missing `.compute`
    attribute surface as a confusing `AttributeError` from deep inside
    `_solve_accounting`."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="execution_model"):
        run_accounting(held, asset_returns, object(), 100_000.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rebalance_date", "match"),
    [
        (
            pd.DataFrame({"AAA": ["False", "True", "False"]}),
            "boolean",
        ),
        (
            pd.DataFrame({"AAA": [1, 0, 1]}),
            "boolean",
        ),
        (
            pd.DataFrame({"AAA": [True, None, False]}),
            "missing values",
        ),
    ],
)
def test_run_accounting_rejects_a_non_boolean_rebalance_date(
    rebalance_date: pd.DataFrame, match: str
) -> None:
    """A non-boolean `rebalance_date` column (e.g. the string `'False'`,
    which Python/pandas would otherwise silently coerce to a truthy
    non-empty string, or a `0`/`1` integer column) must raise, never
    silently be treated as `True` and force a phantom rebalance."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rebalance_date = rebalance_date.set_index(idx)
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match=match):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            model_weight_drift=True,
            rebalance_date=rebalance_date,
        )


@pytest.mark.parametrize("invalid_input", ["weights", "returns"])
def test_non_finite_accounting_input_raises(invalid_input: str) -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01]}, index=idx)
    if invalid_input == "weights":
        held.iloc[1, 0] = np.inf
        message = "held_weights"
    else:
        asset_returns.iloc[1, 0] = np.inf
        message = "asset_returns"

    with pytest.raises(BacktestError, match=message):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


def test_asset_simple_return_below_total_loss_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, -1.01]}, index=idx)

    with pytest.raises(BacktestError, match=r"below -1\.0"):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


def test_missing_return_remains_valid_when_asset_is_not_held() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, np.nan]}, index=idx)

    result = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)

    assert result.net_returns.eq(0.0).all()


def test_asset_returns_must_cover_held_weight_axes() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"BBB": [np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="must cover every"):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


def test_tradable_tolerates_a_differently_ordered_frame() -> None:
    """`tradable`'s column order need not match `held_weights`' own order --
    only the *set* of dates and symbols must agree."""
    idx = pd.date_range("2024-01-05", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5], "BBB": [0.5, 0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.01, -0.01], "BBB": [0.0, 0.02, 0.01]}, index=idx
    )
    # Same labels as `held`, deliberately reversed column order.
    tradable = pd.DataFrame(
        {"BBB": [True, True, True], "AAA": [True, True, True]}, index=idx
    )

    result = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    result_with_mask = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, tradable=tradable
    )
    pd.testing.assert_series_equal(result.net_returns, result_with_mask.net_returns)


def test_tradable_must_cover_the_same_set_of_symbols_as_held_weights() -> None:
    """A genuine set mismatch (not just reordering) must still raise --
    silently defaulting an unrecognized symbol to "tradable" could let it
    trade on a date it should have stayed closed."""
    idx = pd.date_range("2024-01-05", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.01], "BBB": [0.0, 0.02]}, index=idx)
    tradable = pd.DataFrame({"AAA": [True, True]}, index=idx)  # missing BBB

    with pytest.raises(BacktestError, match="dates and symbols"):
        run_accounting(
            held, asset_returns, _zero_cost_model(), 100_000.0, tradable=tradable
        )


# --------------------------------------------------------------------------- #
# Stop-loss / take-profit -- operates on the REAL executed position, never a
# raw strategy signal (see `_detect_stop_loss_take_profit`'s docstring).
# --------------------------------------------------------------------------- #
def test_stop_loss_take_profit_disabled_by_default_changes_nothing() -> None:
    """The single most important non-regression guarantee: leaving both
    thresholds at their `None` default must produce byte-identical
    accounting to today's behavior, for any weights/returns."""
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, -0.30, -0.30, 0.05, 0.05]}, index=idx
    )
    baseline = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    with_none = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        stop_loss_pct=None,
        take_profit_pct=None,
        position_groups=None,
    )
    pd.testing.assert_frame_equal(baseline.executed_weights, with_none.executed_weights)
    pd.testing.assert_series_equal(baseline.equity, with_none.equity)
    assert not with_none.stop_loss_triggered.to_numpy().any()
    assert not with_none.take_profit_triggered.to_numpy().any()


def test_stop_loss_forces_flat_the_bar_after_the_cumulative_breach() -> None:
    """Hand-computed: long AAA throughout, -6% then another -6% (cumulative
    0.94*0.94-1 = -11.64%, past a 10% stop) -- the LOSS-REALIZING bar itself
    keeps its return (no look-ahead: that loss already happened), and the
    position is force-flattened starting the NEXT bar."""
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.DataFrame({"AAA": [1.0] * 6}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, -0.06, -0.06, 0.0, 0.0]}, index=idx)

    acc = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, stop_loss_pct=0.10
    )

    assert acc.executed_weights["AAA"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    # The trigger array marks only the FIRST force-flattened date (the
    # "exit" event itself), not every date the position stays flat
    # thereafter -- that ongoing status already lives in executed_weights
    # (0.0) and would double as a "trigger" on every subsequent bar if not
    # deliberately restricted to the transition.
    assert acc.stop_loss_triggered["AAA"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert not acc.take_profit_triggered.to_numpy().any()
    # The bar that realized the breaching loss (index 3) is NOT itself
    # force-flattened -- that loss had already happened.
    assert acc.gross_returns.iloc[3] == pytest.approx(-0.06)


def test_stop_loss_gates_the_real_drifted_position_not_a_step_function() -> None:
    """Regression test: `model_weight_drift=True` is now the DEFAULT, so
    stop-loss/take-profit combined with organic weight drift is close to
    the ordinary path, not an exotic combination -- yet `_detect_stop_
    loss_take_profit`'s `before_state = executed - weight_changes` gated-
    turnover patch was previously never exercised with drift active.

    A and B start 50/50; A loses value every day (cumulative loss exceeds
    the 10% stop by the bar entering 2024-01-05); with weight drift on,
    A's weight organically SHRINKS below its own anchor value each day
    (never a step function) while B's grows to compensate (E shrinks from
    A's losses alone). The stop-loss must gate the REAL drifted weight A
    was actually sitting at when it flattens -- not the stale 0.5 anchor
    -- and the resulting turnover must reflect exactly that real value."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    held = pd.DataFrame({"A": [0.5] * 6, "B": [0.5] * 6}, index=dates)
    asset_returns = pd.DataFrame(
        {
            "A": [np.nan, -0.05, -0.05, -0.03, 0.0, 0.0],
            "B": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )

    result = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        model_weight_drift=True,
        stop_loss_pct=0.10,
    )

    # Organic drift entering rows 2-3 (never the flat 0.5 anchor) --
    # confirms this scenario genuinely exercises drift, not a step
    # function: dollar_A = 0.5*(1-0.05) = 0.475, E = 1+(0.5*-0.05) = 0.975,
    # weight_A = 0.475/0.975.
    assert result.executed_weights.loc[dates[2], "A"] == pytest.approx(
        0.475 / 0.975, rel=1e-6
    )
    assert result.executed_weights.loc[dates[2], "A"] != pytest.approx(0.5)

    # Cumulative A return since anchor: 0.95*0.95*0.97 - 1 ≈ -12.46%, past
    # the 10% stop -- triggers starting the bar after the breaching return
    # is realized (index 4), B is entirely unaffected (independent group).
    assert bool(result.stop_loss_triggered.loc[dates[4], "A"])
    assert not bool(result.stop_loss_triggered.loc[dates[4], "B"])
    assert not result.take_profit_triggered.to_numpy().any()
    assert result.executed_weights.loc[dates[4], "A"] == pytest.approx(0.0)
    assert result.executed_weights.loc[dates[4], "B"] == pytest.approx(
        0.533212, abs=1e-6
    )

    # The gated turnover must equal the REAL drifted weight A was sitting
    # at just before this row's own forced flatten (0.466788 -- verified
    # independently against the undisturbed drift path with no stop-loss
    # configured at all), never the stale 0.5 anchor or a value from an
    # earlier row.
    assert result.turnover.loc[dates[4]] == pytest.approx(0.466788, abs=1e-6)
    assert result.turnover.loc[dates[4]] != pytest.approx(0.5)

    # No trade at all for B on the stop-loss row -- gating A must never
    # spill into an unrelated, independent position group.
    assert result.weight_changes.loc[dates[4], "B"] == pytest.approx(0.0)

    assert np.isfinite(result.equity.to_numpy()).all()
    assert np.isfinite(result.net_returns.to_numpy()).all()
    assert np.isfinite(result.costs.total.to_numpy()).all()


def test_take_profit_forces_flat_the_bar_after_the_cumulative_gain() -> None:
    """Mirror of the stop-loss test, on the favorable side."""
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.DataFrame({"AAA": [1.0] * 6}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.06, 0.06, 0.0, 0.0]}, index=idx)

    acc = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, take_profit_pct=0.10
    )

    assert acc.executed_weights["AAA"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert acc.take_profit_triggered["AAA"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert not acc.stop_loss_triggered.to_numpy().any()


def test_stop_loss_never_triggers_when_no_position_is_actually_executed() -> None:
    """The test that would have failed the FIRST (rejected) design: a
    strategy can emit a signal, or the allocator can decide a target, but
    if it never becomes an actually-held (executed) position -- here
    ``held`` is always 0 -- a stop-loss must never fire, no matter how
    extreme the asset's own return is."""
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.DataFrame({"AAA": [0.0] * 6}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, -0.5, -0.5, 0.0, 0.0]}, index=idx)

    acc = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, stop_loss_pct=0.10
    )

    assert not acc.stop_loss_triggered.to_numpy().any()
    assert (acc.executed_weights["AAA"] == 0.0).all()


def test_stop_loss_does_not_immediately_reenter_at_a_rebased_price() -> None:
    """Once stopped, the position stays flat until the NEXT flat-to-non-
    flat transition of the (still-nonzero) held weight -- not an
    immediate re-entry the following bar, matching mean_reversion's own
    stop_threshold re-entry convention."""
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    # Held stays long throughout (the raw held target never goes back to
    # flat) -- a naive design might re-enter as soon as the position
    # "recovers"; the real one must not, since the group never actually
    # returned to flat.
    held = pd.DataFrame({"AAA": [1.0] * 8}, index=idx)
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, -0.20, 0.0, 0.20, 0.20, 0.20, 0.20]}, index=idx
    )

    acc = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, stop_loss_pct=0.10
    )

    # Stopped once at index 2 (-20% > 10% stop), forced flat from index 3
    # onward for the REST of the series -- even though later returns are
    # strongly positive, there is no re-entry since held never returns to 0.
    assert acc.executed_weights["AAA"].tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_position_groups_use_combined_pair_pnl_not_each_leg_separately() -> None:
    """Hand-computed pair scenario with a rebalance that HALVES both legs'
    magnitude mid-hold (simulating a hedge-ratio/position-size change):
    executed A=[1.0, 0.5, 0.25], B=[-1.0, -0.5, -0.25] (fixed 1:1 ratio,
    scaled down), returns A=-20%/-20%/-20%, B=0%/0%/0%. The GROUP return
    per unit of gross exposure is a CONSTANT -10% every period regardless
    of the scaling (gross_exposure exactly cancels the position-size
    change) -- this is the whole point of normalizing by realized
    exposure rather than dollar contribution. Cumulative: 0.90 -> 0.81
    (-19%, past a 15% stop, set after that bar) -> forces BOTH legs flat
    starting the next bar."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    executed = pd.DataFrame(
        {"A": [0.0, 1.0, 0.5, 0.25], "B": [0.0, -1.0, -0.5, -0.25]}, index=idx
    )
    asset_returns = pd.DataFrame(
        {"A": [0.0, -0.20, -0.20, -0.20], "B": [0.0, 0.0, 0.0, 0.0]}, index=idx
    )

    gated, stop_loss, take_profit, _ = _detect_stop_loss_take_profit(
        executed, asset_returns, [("A", "B")], 0.15, None
    )

    assert gated["A"].tolist() == [0.0, 1.0, 0.5, 0.0]
    assert gated["B"].tolist() == [0.0, -1.0, -0.5, 0.0]
    assert stop_loss["A"].tolist() == [False, False, False, True]
    assert stop_loss["B"].tolist() == [False, False, False, True]
    assert not take_profit.to_numpy().any()


def test_position_groups_default_to_one_independent_group_per_symbol() -> None:
    """`position_groups=None` (the default) must behave identically to
    declaring every symbol its own singleton group -- two unrelated
    symbols' stop-losses must never interact."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    executed = pd.DataFrame(
        {"A": [0.0, 1.0, 1.0, 1.0], "B": [0.0, 1.0, 1.0, 1.0]}, index=idx
    )
    # A breaches a 10% stop; B never does.
    asset_returns = pd.DataFrame(
        {"A": [0.0, -0.06, -0.06, 0.0], "B": [0.0, 0.01, 0.01, 0.01]}, index=idx
    )

    gated, stop_loss, _, _ = _detect_stop_loss_take_profit(
        executed, asset_returns, None, 0.10, None
    )

    assert gated["A"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert gated["B"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert stop_loss["A"].tolist() == [False, False, False, True]
    assert not stop_loss["B"].any()


def test_direct_long_to_short_reversal_starts_a_fresh_episode() -> None:
    """A same-bar sign flip (long directly to short, no intermediate flat
    row) must start a brand-new stop-loss episode for the new direction --
    it must NOT inherit the opposite-direction position's already-
    triggered stop and stay force-flattened forever, since the reversed
    position was never actually the one that breached.

    Long AAA, -20% breaches a 10% stop (forced flat from index 2). At
    index 3 the raw signal reverses directly to short (skipping flat) --
    the new short position must be allowed to hold, since -5%/-5% moves
    against it never breach 10% on their OWN fresh cumulative return."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    executed = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, -1.0, -1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, -0.20, 0.0, 0.05, 0.05]}, index=idx)

    gated, stop_loss, _, _ = _detect_stop_loss_take_profit(
        executed, asset_returns, None, 0.10, None
    )

    assert gated["AAA"].tolist() == [1.0, 1.0, 0.0, -1.0, -1.0]
    assert stop_loss["AAA"].tolist() == [False, False, True, False, False]


def test_position_groups_reject_malformed_declarations() -> None:
    """An empty group, a self-duplicate, an unknown symbol, or two groups
    claiming the same symbol must raise loudly -- silently accepting any
    of these would double-process (or mis-key) a symbol across two
    different entry-timing episodes instead of failing fast."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    executed = pd.DataFrame(
        {"A": [0.0, 1.0], "B": [0.0, 1.0], "C": [0.0, 1.0]}, index=idx
    )
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.01], "B": [0.0, 0.01], "C": [0.0, 0.01]}, index=idx
    )

    with pytest.raises(BacktestError, match="empty group"):
        _detect_stop_loss_take_profit(executed, asset_returns, [()], 0.10, None)
    with pytest.raises(BacktestError, match="repeats a symbol"):
        _detect_stop_loss_take_profit(executed, asset_returns, [("A", "A")], 0.10, None)
    with pytest.raises(BacktestError, match="not present"):
        _detect_stop_loss_take_profit(
            executed, asset_returns, [("A", "ZZZ")], 0.10, None
        )
    with pytest.raises(BacktestError, match="overlaps symbol"):
        _detect_stop_loss_take_profit(
            executed, asset_returns, [("A", "B"), ("B", "C")], 0.10, None
        )
