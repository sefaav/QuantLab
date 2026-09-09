"""Tests for the ``next_bar_open`` overnight/intraday return decomposition.

These are the load-bearing correctness tests for the execution-timing fix
that makes a decision's return attribution causally realizable (see
``ExecutionConfig.execution_timing`` and
``accounting._gross_returns_and_cost_basis``). Every scenario is
hand-computed, not just asserted against the implementation's own output.

The two legs are COMPOUNDED, never simply added:
``(1+overnight_leg)*(1+intraday_leg) - 1`` -- adding them instead would
silently drop the cross-compounding term whenever a real trade happens
between them, see ``test_a_full_rotation_compounds_the_two_legs_not_
adds_them`` below (a 100% A -> 100% B rotation with +10% overnight then
+10% intraday must earn 21%, not 20%).

Convention used throughout: ``held`` is the *decision* frame passed to
``run_accounting`` (before its internal one-period shift);
``executed[T] = held[T-1]`` is the frame these tests reason about directly
(row 0 is always flat, per ``executed_weights``' own convention).
``overnight_returns[T]`` is ``close[T-1] -> open[T]``; the corresponding
``asset_returns[T]`` (plain close-to-close) is always constructed from the
identity ``1 + asset_return = (1 + overnight) * (1 + intraday)`` so a test
can pick overnight/intraday directly and derive the close-to-close input
``run_accounting`` also requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.accounting import (
    _close_of_previous_row,
    _gross_returns_and_cost_basis,
    _group_return_with_execution_timing,
    _zero_safe_multiply,
    run_accounting,
)
from quantlab.backtesting.trade_log import build_trade_log
from quantlab.config import ExecutionConfig
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionModel


def _zero_cost_model() -> ExecutionModel:
    return ExecutionModel.from_config(
        ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    )


def _combine(overnight: float, intraday: float) -> float:
    """The close-to-close simple return implied by an overnight/intraday split."""
    return (1.0 + overnight) * (1.0 + intraday) - 1.0


def test_a_full_rotation_compounds_the_two_legs_not_adds_them() -> None:
    """100% in A, A gains +10% overnight, the book then rotates entirely
    into B at the open, B then gains +10% intraday. The overnight gain
    funds a LARGER dollar position in B, which itself then earns +10% --
    the economically correct answer is ``1.10*1.10-1 = 21%``, not the
    naively additive ``10%+10% = 20%``. Verified independently by
    dollar-tracing: start with $100 all in A; overnight A -> $110; rotate
    fully into B at $110; intraday B -> $121; return = $121/$100-1 = 21%.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    overnight_a, intraday_b = 0.10, 0.10
    held = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    # executed[T] = held[T-1]: executed_A = [0, 1, 0], executed_B = [0, 0, 1]
    # -> rotation at index 2.
    asset_returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, overnight_a],  # A's own close-to-close: pure overnight move
            "B": [0.0, 0.0, intraday_b],  # B's own close-to-close: pure intraday move
        },
        index=idx,
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, overnight_a], "B": [0.0, 0.0, 0.0]}, index=idx
    )
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    assert acc.gross_returns.iloc[2] == pytest.approx(0.21)
    assert acc.gross_returns.iloc[2] != pytest.approx(0.20)


def test_zero_safe_multiply_tolerates_a_tiny_floating_point_residual() -> None:
    """Regression test: a semantically-flat weight derived arithmetically
    (``executed - weight_changes``) can carry a tiny nonzero float
    residual (e.g. ~1e-17) instead of an exact ``0.0``. An exact ``!=
    0.0`` comparison would not recognize that cell as flat, letting the
    residual multiply a legitimately-missing ``returns`` value into a
    real ``NaN`` (``1e-17 * NaN`` is still ``NaN``) that then poisons a
    ``min_count=1`` sum downstream -- the whole point of this helper is
    to force exactly ``0.0`` there instead."""
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    tiny_residual = pd.DataFrame({"AAA": [1e-17]}, index=idx)
    missing_return = pd.DataFrame({"AAA": [float("nan")]}, index=idx)
    result = _zero_safe_multiply(tiny_residual, missing_return)
    assert result["AAA"].iloc[0] == 0.0
    assert not np.isnan(result["AAA"].iloc[0])

    # A genuinely material weight must still multiply through normally,
    # NaN included -- this helper only special-cases near-zero weights.
    real_weight = pd.DataFrame({"AAA": [0.5]}, index=idx)
    still_nan = _zero_safe_multiply(real_weight, missing_return)
    assert np.isnan(still_nan["AAA"].iloc[0])


def test_no_position_change_uses_plain_close_to_close_unaffected_by_overnight() -> None:
    """A steady-state holding day must be byte-identical to today's formula.

    Deliberately picks an ``overnight_returns`` value that would give a
    materially different (wrong) answer if it were blended in -- proving
    the decomposition is correctly gated on "did a trade happen today",
    not applied unconditionally.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    # executed = [0, 1, 1, 1, 1] -> entry at index 1, steady state at 2/3/4.
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, 0.05, 0.0, 0.0]}, index=idx
    )  # plain close-to-close on the steady-state date (index 2) is 5%.
    # A deliberately "wrong" overnight value on the steady-state date: if it
    # were used, the (unconditional, rejected) decomposed formula would give
    # w*overnight + w*intraday != w*asset_return, which is exactly the bug
    # this gating avoids.
    overnight_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.5, 0.0, 0.0]}, index=idx)
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    assert acc.gross_returns.iloc[2] == pytest.approx(0.05)


def test_full_entry_from_cash_earns_only_the_intraday_leg() -> None:
    """Entering fresh at the open must not retroactively earn the overnight gap."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    overnight, intraday = 0.02, 0.03
    held = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=idx)
    # executed = [0, 0, 1] -> entry happens at index 2.
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, _combine(overnight, intraday)]}, index=idx
    )
    overnight_returns = pd.DataFrame({"AAA": [0.0, 0.0, overnight]}, index=idx)
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    assert acc.gross_returns.iloc[2] == pytest.approx(intraday)
    # The plain close-to-close figure would have been strictly larger here
    # (overnight and intraday are both positive) -- confirming the fix
    # actually changes the number, not just that it runs.
    assert acc.gross_returns.iloc[2] < _combine(overnight, intraday)


def test_full_exit_to_cash_earns_only_the_overnight_leg() -> None:
    """Exiting at the open must still capture the overnight gap held into it."""
    overnight, intraday = 0.02, 0.03
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 0.0, 0.0]}, index=idx)
    # executed = [0, 1, 1, 1, 0] -> exit happens at index 4.
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, 0.0, 0.0, _combine(overnight, intraday)]}, index=idx
    )
    overnight_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, 0.0, 0.0, overnight]}, index=idx
    )
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    assert acc.gross_returns.iloc[4] == pytest.approx(overnight)


def test_partial_rebalance_splits_pre_and_post_holdings_correctly() -> None:
    """100% A -> 50% A / 50% B: overnight on the old book, intraday on the new one.

    A was still 100% exposed to the overnight gap (nothing traded yet),
    then the book became 50/50 for the intraday leg once the rebalance
    actually executed -- a per-day open-to-close treatment of the whole
    changed day would get this wrong.
    """
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    overnight_a, intraday_a = 0.02, -0.01
    overnight_b, intraday_b = -0.015, 0.04
    held_a = pd.DataFrame({"A": [1.0, 1.0, 0.5, 0.5]}, index=idx)["A"]
    held_b = pd.DataFrame({"B": [0.0, 0.0, 0.5, 0.5]}, index=idx)["B"]
    held = pd.concat([held_a, held_b], axis=1)
    # executed_A = [0, 1, 1, 0.5], executed_B = [0, 0, 0, 0.5]
    # -> the 100%A -> 50/50 rebalance is visible at index 3.
    asset_returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0, _combine(overnight_a, intraday_a)],
            "B": [0.0, 0.0, 0.0, _combine(overnight_b, intraday_b)],
        },
        index=idx,
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, overnight_a], "B": [0.0, 0.0, 0.0, overnight_b]},
        index=idx,
    )
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    # pre_A=1.0 (full overnight exposure to A, nothing traded yet),
    # post_A=0.5 (half the intraday move), pre_B=0.0, post_B=0.5. The two
    # legs COMPOUND, not add: the overnight leg is the portfolio-wide
    # pre-trade exposure's own return; the intraday leg is the portfolio-
    # wide post-trade exposure's own return.
    overnight_leg = 1.0 * overnight_a + 0.0 * overnight_b
    intraday_leg = 0.5 * intraday_a + 0.5 * intraday_b
    expected = (1.0 + overnight_leg) * (1.0 + intraday_leg) - 1.0
    assert acc.gross_returns.iloc[3] == pytest.approx(expected)


def test_rotation_a_to_b_earns_overnight_on_a_and_intraday_on_b() -> None:
    """100% A -> 100% B: still holding A overnight, holding B intraday."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    overnight_a, intraday_a = 0.01, 0.02
    overnight_b, intraday_b = -0.02, 0.03
    held = pd.DataFrame(
        {"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=idx
    )
    # executed_A = [0, 1, 1, 0], executed_B = [0, 0, 0, 1] -> rotation at index 3.
    asset_returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0, _combine(overnight_a, intraday_a)],
            "B": [0.0, 0.0, 0.0, _combine(overnight_b, intraday_b)],
        },
        index=idx,
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, overnight_a], "B": [0.0, 0.0, 0.0, overnight_b]},
        index=idx,
    )
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    # pre_A=1 post_A=0; pre_B=0 post_B=1 -- the overnight leg (earned on A,
    # the pre-trade holding) and the intraday leg (earned on B, the
    # post-trade holding) COMPOUND, not add (see
    # test_a_full_rotation_compounds_the_two_legs_not_adds_them for the
    # exact bug this guards against).
    expected = (1.0 + overnight_a) * (1.0 + intraday_b) - 1.0
    assert acc.gross_returns.iloc[3] == pytest.approx(expected)


def test_several_assets_changed_simultaneously() -> None:
    """Three assets each move to a different, independently-set target at once."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    legs = {
        # symbol -> (pre, overnight, post, intraday) -- every leg's pre != post,
        # so every leg genuinely trades this day (a "changed" cell).
        "A": (0.5, -0.02, 0.6, 0.01),
        "B": (0.0, 0.03, 0.4, -0.02),
        "C": (-0.5, 0.0, 0.3, 0.02),
    }
    # Build so executed[2] = pre, executed[3] = post for every leg.
    held = pd.DataFrame(index=idx)
    asset_returns = pd.DataFrame(index=idx)
    overnight_returns = pd.DataFrame(index=idx)
    overnight_leg = 0.0
    intraday_leg = 0.0
    for symbol, (pre, overnight, post, intraday) in legs.items():
        held[symbol] = [pre, pre, post, post]  # held[1]=pre -> executed[2]=pre
        asset_returns[symbol] = [0.0, 0.0, 0.0, _combine(overnight, intraday)]
        overnight_returns[symbol] = [0.0, 0.0, 0.0, overnight]
        overnight_leg += pre * overnight
        intraday_leg += post * intraday
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    # The overnight leg (every symbol's own PRE-trade exposure) and the
    # intraday leg (every symbol's own POST-trade exposure) each sum
    # across symbols, then the two portfolio-wide legs COMPOUND.
    expected = (1.0 + overnight_leg) * (1.0 + intraday_leg) - 1.0
    assert acc.gross_returns.iloc[3] == pytest.approx(expected)


def test_overnight_returns_none_is_a_strict_no_op() -> None:
    """`overnight_returns=None` (same_bar_close) matches today's formula exactly."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 0.5, 1.0, 0.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.02, -0.01, 0.03]}, index=idx)
    with_none = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    with_explicit_none = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, overnight_returns=None
    )
    pd.testing.assert_series_equal(
        with_none.gross_returns, with_explicit_none.gross_returns
    )


def test_changed_cell_never_uses_the_same_close_for_both_decision_and_fill() -> None:
    """Causality: a traded cell's contribution must differ from the naive
    "decide and fill at the same close" formula whenever overnight and
    intraday returns diverge -- proving the fix actually changes behavior
    for a real trade, not just for entries/exits already covered above."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    overnight, intraday = 0.10, -0.10
    held = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, _combine(overnight, intraday)]}, index=idx
    )
    overnight_returns = pd.DataFrame({"AAA": [0.0, 0.0, overnight]}, index=idx)
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    naive_same_close_fill = asset_returns["AAA"].iloc[
        2
    ]  # what same_bar_close would give
    assert acc.gross_returns.iloc[2] == pytest.approx(intraday)
    assert acc.gross_returns.iloc[2] != pytest.approx(naive_same_close_fill)


def test_turnover_and_costs_are_sized_on_equity_at_the_open_not_prior_close() -> None:
    """A real, partial rebalance's turnover/cost basis must reflect the
    equity that exists AT the open (after the overnight leg), not stale
    close(T-1) equity -- a real order at the open moves against what the
    portfolio is worth then, not what it was worth the day before.

    Setup: 50% held in AAA, a genuine +20% overnight gap, then a
    scheduled rebalance UP to 100%. Naively re-diffing weights
    (``executed - pre = 1.0 - 0.5 = 0.5``) ignores that the pre-trade
    weight actually DRIFTED (via the overnight leg) to
    ``weight_at_open = 0.5*1.20 / (1+0.5*0.20) = 0.5455``, so the real
    trade at the open is only ``1.0 - 0.5455 = 0.4545`` of the (larger)
    open-time equity -- independently hand-derived below, not just
    compared against the implementation's own intermediate values.
    """
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    overnight_gap = 0.20
    held = pd.DataFrame({"AAA": [0.5, 1.0, 1.0]}, index=idx)
    # executed = [0, 0.5, 1.0] -> the rebalance to 100% lands at index 2.
    # asset_returns.iloc[2] == overnight_gap exactly -> zero intraday leg,
    # isolating the cost-basis question from the (separately tested)
    # overnight/intraday compounding mechanics.
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, overnight_gap]}, index=idx)
    overnight_returns = pd.DataFrame({"AAA": [0.0, 0.0, overnight_gap]}, index=idx)
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=100.0, spread_bps=0.0, slippage_bps=0.0)
    )
    acc = run_accounting(
        held, asset_returns, model, 100_000.0, overnight_returns=overnight_returns
    )

    pre = 0.5
    overnight_leg = pre * overnight_gap
    weight_at_open = pre * (1.0 + overnight_gap) / (1.0 + overnight_leg)
    expected_turnover = 1.0 - weight_at_open
    assert acc.turnover.iloc[2] == pytest.approx(expected_turnover)
    # Materially different from the naive (buggy) prior-close-basis
    # turnover a plain re-diff would have given.
    assert acc.turnover.iloc[2] != pytest.approx(1.0 - pre)

    equity_at_open = acc.equity.iloc[1] * (1.0 + overnight_leg)
    # commission_bps=100 -> 1% -- this is the REAL dollar commission, sized
    # (correctly) against equity at the open, independent of how it is
    # later expressed as a return fraction.
    expected_commission_dollars = 0.01 * expected_turnover * equity_at_open
    assert expected_commission_dollars == pytest.approx(497.5, rel=1e-3)
    # `acc.costs.commission` (and every other cost component) is a
    # fraction of the SAME close(T-1)-equity basis `gross_returns` uses,
    # not of the open-equity basis the dollar trade was actually sized
    # against -- dividing the real dollar cost by the PRIOR close equity
    # (not the open equity) is what proves the two bases were never
    # silently mixed.
    expected_commission_close_basis = expected_commission_dollars / acc.equity.iloc[1]
    assert acc.costs.commission.iloc[2] == pytest.approx(
        expected_commission_close_basis
    )
    # The real proof this module cares about: the dollar cost must be
    # exactly what disappears between open equity and the period's final
    # net equity -- not silently over- or under-charged by mixing bases.
    # Valid because the intraday leg is exactly zero here (asset_returns
    # == overnight_gap): open equity IS close equity before the cost.
    expected_net_equity = equity_at_open - expected_commission_dollars
    assert acc.equity.iloc[2] == pytest.approx(expected_net_equity)


def test_costs_are_correctly_deducted_after_a_negative_overnight_gap() -> None:
    """Same cost-basis proof as the test above, but with a NEGATIVE gap and
    a FULL rotation (not a partial one) -- the bug this guards against
    (subtracting an open-basis cost fraction straight from a close-basis
    return) would OVERSTATE costs after a negative gap, since ``(1 +
    overnight_leg)`` is below 1 there, the opposite direction from the
    positive-gap case above."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    overnight_gap = -0.20
    held = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    # executed: [0,0] -> [1,0] -> [0,1]; full A->B rotation lands at index 2.
    # asset_returns.iloc[2][A] == overnight_gap exactly -> zero intraday
    # leg on A, isolating the cost-basis question.
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.0, overnight_gap], "B": [0.0, 0.0, 0.0]}, index=idx
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, overnight_gap], "B": [0.0, 0.0, 0.0]}, index=idx
    )
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=100.0, spread_bps=0.0, slippage_bps=0.0)
    )
    acc = run_accounting(
        held, asset_returns, model, 100_000.0, overnight_returns=overnight_returns
    )

    pre = 1.0  # 100% A held going into the gap
    overnight_leg = pre * overnight_gap
    turnover = 2.0  # full rotation: sell 100% A, buy 100% B
    equity_at_open = acc.equity.iloc[1] * (1.0 + overnight_leg)
    commission_rate = 0.01  # commission_bps=100 -> 1%
    expected_cost_dollars = commission_rate * turnover * equity_at_open
    expected_net_equity = equity_at_open - expected_cost_dollars
    assert acc.turnover.iloc[2] == pytest.approx(turnover)
    assert acc.equity.iloc[2] == pytest.approx(expected_net_equity)
    # Sanity: proves the fix actually matters here -- the (buggy) naive
    # subtraction of an unconverted open-basis cost fraction would have
    # OVERSTATED the cost after this negative gap (multiplying by the
    # smaller close-basis equity via a larger raw fraction instead of the
    # correct (1+overnight_leg) conversion factor, which is below 1
    # here), giving a materially different equity.
    naive_cost_fraction = commission_rate * turnover  # open-basis, unconverted
    naive_net_equity = acc.equity.iloc[1] * (
        1.0 + acc.gross_returns.iloc[2] - naive_cost_fraction
    )
    assert acc.equity.iloc[2] != pytest.approx(naive_net_equity)


def test_costs_are_correctly_deducted_for_a_full_rotation_with_volume_slippage() -> (
    None
):
    """The same open-basis -> close-basis cost conversion must apply to
    volume-based slippage exactly like commission/spread -- it is computed
    from the same ``weight_changes_for_costs`` input and returned in the
    same units (a fraction of whatever equity that input is itself
    relative to), regardless of the dollar-ADV lookup happening inside it."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    overnight_gap = 0.15
    held = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.0, overnight_gap], "B": [0.0, 0.0, 0.0]}, index=idx
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, overnight_gap], "B": [0.0, 0.0, 0.0]}, index=idx
    )
    model = ExecutionModel.from_config(
        ExecutionConfig(
            commission_bps=0.0,
            spread_bps=0.0,
            slippage_bps=10.0,
            slippage_model="volume",
            impact_coefficient=0.0,
        ),
        average_daily_volume=1_000_000.0,
    )
    acc = run_accounting(
        held, asset_returns, model, 100_000.0, overnight_returns=overnight_returns
    )

    pre = 1.0
    overnight_leg = pre * overnight_gap
    turnover = 2.0
    equity_at_open = acc.equity.iloc[1] * (1.0 + overnight_leg)
    # impact_coefficient=0.0 collapses VolumeBasedSlippageModel to a flat
    # base_slippage_bps rate, same formula as the constant model.
    expected_slippage_dollars = 0.0010 * turnover * equity_at_open
    expected_net_equity = equity_at_open - expected_slippage_dollars
    assert acc.equity.iloc[2] == pytest.approx(expected_net_equity)


def test_overnight_returns_validated_like_asset_returns() -> None:
    """overnight_returns must go through the same finiteness/floor checks."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    bad_overnight = pd.DataFrame({"AAA": [0.0, -2.0]}, index=idx)
    with pytest.raises(BacktestError, match=r"below -1\.0"):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=bad_overnight,
        )


def test_overnight_returns_exactly_minus_one_is_rejected() -> None:
    """A -100% overnight leg (e.g. a zero adjusted open) must not reach the
    intraday-return division -- (1+overnight) in the denominator would be
    exactly zero, poisoning gross_returns with inf/NaN. Rejected at or
    below -1.0, not just strictly below it."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    exact_wipeout_overnight = pd.DataFrame({"AAA": [0.0, -1.0]}, index=idx)
    with pytest.raises(BacktestError, match=r"at or below -1\.0"):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=exact_wipeout_overnight,
        )


def test_gross_returns_derives_pre_fresh_never_from_a_cached_snapshot() -> None:
    """With ``drift_active=False`` (the default), ``pre`` is
    ``executed - weight_changes`` from whichever pair a given call
    receives, never a snapshot carried over from an earlier pass -- two
    calls for the SAME date, one before and one after a stop-loss-style
    gating that changes ``executed``/``weight_changes``, must derive two
    DIFFERENT values for ``pre``, matching each call's own (post-gating)
    state. A cached snapshot would instead reuse the pre-gating value
    even after a forced flatten, silently mispricing the overnight leg on
    the row right after the flatten.
    """
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    overnight, intraday = 0.02, 0.03
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, _combine(overnight, intraday)]}, index=idx
    )
    overnight_returns = pd.DataFrame({"AAA": [0.0, overnight]}, index=idx)

    # "Before gating": organic drift already carried the position to 0.6
    # by the close of the previous row; nothing trades today (weight_
    # changes == 0), so this row falls to the plain undecomposed formula
    # -- this call only establishes the pre-gating baseline.
    executed_ungated = pd.DataFrame({"AAA": [0.6, 0.6]}, index=idx)
    weight_changes_ungated = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    ungated_returns, _costs, _leg = _gross_returns_and_cost_basis(
        executed_ungated,
        asset_returns,
        weight_changes_ungated,
        overnight_returns=overnight_returns,
    )
    assert ungated_returns.iloc[1] == pytest.approx(asset_returns["AAA"].iloc[1] * 0.6)

    # "After gating": a stop-loss instead forced this exact row fully
    # flat -- executed AND weight_changes both change to reflect the real
    # forced-exit trade (0.6 -> 0.0), exactly as `run_accounting`'s own
    # stop-loss re-solve does.
    executed_gated = pd.DataFrame({"AAA": [0.6, 0.0]}, index=idx)
    weight_changes_gated = pd.DataFrame({"AAA": [0.0, -0.6]}, index=idx)
    gated_returns, _costs, _leg = _gross_returns_and_cost_basis(
        executed_gated,
        asset_returns,
        weight_changes_gated,
        overnight_returns=overnight_returns,
    )
    # pre = executed - weight_changes = 0.0 - (-0.6) = 0.6 (correctly
    # picked up FRESH from this call's own pair): the overnight leg is
    # still earned on the 0.6 that was genuinely held overnight, and the
    # intraday leg correctly uses the POST-gating executed (0.0, so its
    # own contribution is exactly zero) -- never the stale 0.6 a cached
    # snapshot would have supplied. With the intraday leg at exactly
    # zero, compounding and adding coincide here, so this also equals
    # ``0.6 * overnight``.
    expected = (1.0 + 0.6 * overnight) * (1.0 + 0.0) - 1.0
    assert gated_returns.iloc[1] == pytest.approx(expected)
    # Sanity: must differ from what the stale cached snapshot (0.6 for
    # BOTH legs, ignoring the forced exit) would have given -- proving
    # this is not just coincidentally the same answer either way.
    stale_snapshot_answer = (1.0 + 0.6 * overnight) * (1.0 + 0.6 * intraday) - 1.0
    assert gated_returns.iloc[1] != pytest.approx(stale_snapshot_answer)


def test_run_accounting_accepts_weight_drift_with_overnight_returns() -> None:
    """Public-API smoke test: ``model_weight_drift=True`` together with
    ``overnight_returns`` (i.e. ``next_bar_open``) through the public
    ``run_accounting`` entry point. A steady, unchanging target under
    drift should still reproduce the plain close-to-close return every
    gated test above already establishes for a no-trade day.
    """
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.01, -0.02, 0.03]}, index=idx)
    overnight_returns = pd.DataFrame({"AAA": [0.0, 0.005, -0.01, 0.015]}, index=idx)
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        model_weight_drift=True,
        overnight_returns=overnight_returns,
    )
    assert np.isfinite(acc.equity.to_numpy()).all()
    assert np.isfinite(acc.gross_returns.to_numpy()).all()


def test_run_accounting_combines_drift_stop_loss_and_overnight_returns() -> None:
    """End-to-end, hand-computed test: weight drift + a stop-loss-triggered
    forced flatten immediately followed by a same-bar reversal, all under
    ``next_bar_open``, through the full public ``run_accounting`` pipeline.

    Single-asset, fully invested (+-1.0, no cash buffer), so organic drift
    is a no-op -- isolating the stop-loss/reversal/overnight interaction.
    ``executed = [0, 1, 1, 1, -1, -1]``. Row 2's -20% return breaches the
    10% stop, force-flattening row 3. Row 4's same-bar reversal (long
    directly to short) must measure its own trade against ``pre[4] = 0.0``
    (genuinely flat after row 3's forced exit), so its contribution is
    purely the intraday leg on the fresh short.
    """
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, -0.20, 0.0, 0.05, 0.05]}, index=idx)
    overnight_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, -0.01, 0.0, 0.02, 0.02]}, index=idx
    )
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        model_weight_drift=True,
        stop_loss_pct=0.10,
        overnight_returns=overnight_returns,
    )
    assert np.isfinite(acc.equity.to_numpy()).all()
    assert np.isfinite(acc.gross_returns.to_numpy()).all()
    assert acc.stop_loss_triggered["AAA"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    # Row 2: no trade (pure drift continuation) -- plain close-to-close.
    assert acc.gross_returns.iloc[2] == pytest.approx(-0.20)
    # Row 3: the forced exit itself -- both legs are zero-return bars, so
    # the contribution is 0 regardless of `pre`.
    assert acc.gross_returns.iloc[3] == pytest.approx(0.0)
    # Row 4: pre=0 (genuinely flat after row 3) -> purely the intraday leg.
    correct_intraday = (1.0 + 0.05) / (1.0 + 0.02) - 1.0
    correct_row4 = -1.0 * correct_intraday
    # A stale, ungated pre=1.0 would instead add its own overnight leg on
    # top of the fresh short's intraday leg -- a materially different,
    # still-finite number, so an isfinite-only check would miss this.
    buggy_row4 = 1.0 * 0.02 + (-1.0) * correct_intraday
    assert acc.gross_returns.iloc[4] == pytest.approx(correct_row4)
    assert acc.gross_returns.iloc[4] != pytest.approx(buggy_row4)


def test_exit_day_with_missing_asset_return_keeps_the_overnight_leg() -> None:
    """An exit cell (executed == 0 after a real trade) can legitimately have
    a missing asset_return (e.g. no more data once fully closed out) -- the
    real, nonzero pre * overnight leg for that day must survive, not be
    silently wiped out by a NaN executed * intraday term (0 * NaN = NaN in
    IEEE arithmetic, which a downstream skipna sum would then drop
    entirely)."""
    overnight = 0.04
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 0.0, 0.0]}, index=idx)
    # executed = [0, 1, 1, 1, 0] -> exit happens at index 4 (executed[4]=0).
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.0, 0.0, float("nan")]}, index=idx)
    overnight_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, 0.0, 0.0, overnight]}, index=idx
    )
    acc = run_accounting(
        held,
        asset_returns,
        _zero_cost_model(),
        100_000.0,
        overnight_returns=overnight_returns,
    )
    # pre[4]=executed[3]=1.0 (still held through the overnight gap),
    # post[4]=executed[4]=0.0 (exited at the open) -- contribution must be
    # exactly the overnight leg, not NaN.
    assert acc.gross_returns.iloc[4] == pytest.approx(overnight)


def test_overnight_returns_missing_dates_or_symbols_is_rejected() -> None:
    """overnight_returns must cover every held_weights date and symbol --
    a silently-reindexed gap would turn into an all-NaN column/row that the
    missing_overnight guard below only catches when a real trade happened
    on the affected date, not unconditionally."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0, 1.0], "BBB": [0.0, 0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0]}, index=idx
    )
    missing_symbol_overnight = pd.DataFrame({"AAA": [0.0, 0.0, 0.0]}, index=idx)
    with pytest.raises(
        BacktestError, match=r"must cover every held_weights date and symbol"
    ):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=missing_symbol_overnight,
        )
    missing_date_overnight = pd.DataFrame(
        {"AAA": [0.0, 0.0], "BBB": [0.0, 0.0]}, index=idx[:2]
    )
    with pytest.raises(
        BacktestError, match=r"must cover every held_weights date and symbol"
    ):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=missing_date_overnight,
        )


def test_overnight_returns_duplicate_axis_labels_are_rejected() -> None:
    """Regression test: ``held_weights``/``asset_returns`` both already
    reject a duplicate-labeled index/columns up front (see the loop just
    above their own coverage check) -- a directly-supplied
    ``overnight_returns`` needs the exact same guard, or a duplicate
    label surfaces only much later as pandas' own raw, uninformative
    ``ValueError: cannot reindex on an axis with duplicate labels``
    instead of a clear ``BacktestError``."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.0]}, index=idx)

    duplicate_date_idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]
    )
    duplicate_date_overnight = pd.DataFrame(
        {"AAA": [0.0, 0.0, 0.0, 0.0]}, index=duplicate_date_idx
    )
    with pytest.raises(
        BacktestError, match=r"overnight_returns index must not contain duplicate"
    ):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=duplicate_date_overnight,
        )

    duplicate_column_overnight = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], columns=["AAA", "AAA"], index=idx
    )
    with pytest.raises(
        BacktestError, match=r"overnight_returns columns must not contain duplicate"
    ):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=duplicate_column_overnight,
        )


def test_overnight_returns_wrong_type_is_rejected() -> None:
    """A non-DataFrame overnight_returns (e.g. a bare dict or Series) must
    be rejected up front rather than failing later with a confusing
    attribute error."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    with pytest.raises(BacktestError, match=r"must be a pandas DataFrame"):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns={"AAA": [0.0, 0.0]},  # type: ignore[arg-type]
        )


def test_overnight_returns_non_numeric_is_rejected() -> None:
    """Non-numeric overnight_returns (e.g. accidental strings) must be
    rejected with a clear message rather than propagating a raw numpy
    casting error."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    non_numeric_overnight = pd.DataFrame({"AAA": [0.0, "oops"]}, index=idx)
    with pytest.raises(BacktestError, match=r"must contain only numeric values"):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=non_numeric_overnight,
        )


def test_overnight_returns_infinity_is_rejected() -> None:
    """An infinite overnight_returns value must be rejected explicitly,
    not left to silently poison every downstream decomposed cell."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    infinite_overnight = pd.DataFrame({"AAA": [0.0, float("inf")]}, index=idx)
    with pytest.raises(BacktestError, match=r"must not contain Infinity"):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=infinite_overnight,
        )


def test_missing_overnight_return_on_a_trade_date_is_rejected() -> None:
    """A NaN overnight_returns cell on a date/symbol where a real trade
    happened (weight actually changed) must be rejected -- the
    decomposition needs that leg's value, and it cannot legitimately be
    unknown on the very day it is used."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=idx)
    # executed = [0, 0, 1] -> the trade (entry) lands at index 2, not index 1.
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.0]}, index=idx)
    overnight_with_gap = pd.DataFrame({"AAA": [0.0, 0.0, float("nan")]}, index=idx)
    with pytest.raises(BacktestError, match=r"Overnight return is missing"):
        run_accounting(
            held,
            asset_returns,
            _zero_cost_model(),
            100_000.0,
            overnight_returns=overnight_with_gap,
        )


def _overnight_wipeout_scenario() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """2x leveraged A, a -50% overnight gap wipes it out exactly (-100%),
    with a scheduled rotation into B landing at the open of that same row
    -- there is no equity left for that rotation to actually execute
    against. Returns ``(held, asset_returns, overnight_returns)``."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    held = pd.DataFrame(
        {"A": [2.0, 2.0, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0, 1.0]}, index=idx
    )
    # executed: [0,0] -> [2,0] -> [2,0] -> [0,1] -> [0,1]; the A->B
    # rotation lands at index 3, the same row the gap hits.
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, -0.50, 0.0], "B": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, -0.50, 0.0], "B": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    return held, asset_returns, overnight_returns


def test_overnight_wipeout_forbids_the_same_row_rotation_it_cannot_afford() -> None:
    """Regression test: a 2x position wiped out EXACTLY by its own overnight
    leg (-100%) must not still be allowed a same-row rotation into a new
    asset priced as if real capital existed to fund it: turnover and
    costs on that row must be exactly zero, not the huge-but-finite
    ``weight_at_open`` that dividing by the necessarily EPSILON-floored
    ``(1+overnight_leg)`` would otherwise produce."""
    held, asset_returns, overnight_returns = _overnight_wipeout_scenario()
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=100.0, spread_bps=0.0, slippage_bps=0.0)
    )
    acc = run_accounting(
        held, asset_returns, model, 100_000.0, overnight_returns=overnight_returns
    )
    assert acc.gross_returns.iloc[3] == pytest.approx(-1.0)
    assert acc.equity.iloc[3] == pytest.approx(0.0)
    assert acc.turnover.iloc[3] == pytest.approx(0.0)
    assert acc.costs.total.iloc[3] == pytest.approx(0.0)
    # The wipeout row itself -- not just row 4 onward -- must report flat,
    # never the scheduled A->B rotation no real capital could have funded
    # (mirrors the weight-drift version of this same scenario below).
    assert (acc.executed_weights.iloc[3] == 0.0).all()
    assert (acc.weight_changes.iloc[3] == 0.0).all()
    # Ruin is recognized starting the row AFTER the wipeout -- exactly the
    # same convention an equivalent same_bar_close crash already uses
    # (the crash row itself still reports its own real loss; only later
    # rows, which would otherwise try to trade against capital that no
    # longer exists, are force-flattened).
    assert not acc.ruined.iloc[3]
    assert acc.ruined.iloc[4]
    assert (acc.executed_weights.iloc[4] == 0.0).all()
    assert acc.turnover.iloc[4] == pytest.approx(0.0)


def test_overnight_wipeout_forbids_the_same_row_rotation_under_weight_drift() -> None:
    """Same wipeout, with ``model_weight_drift=True`` -- must produce
    identical numbers to the non-drift case above, since a total wipeout
    leaves no drifted position for the drift mechanism to distinguish
    from a plain step function.

    Unlike the plain (non-drift) case above -- where ``executed_weights``
    is simply the caller's own input step function, disconnected from
    real capital until ``ruined``'s later flattening catches up starting
    the row AFTER the wipeout -- ``apply_weight_drift`` computes
    ``executed_weights`` ITSELF, and must not report the scheduled A->B
    rotation on the wipeout row as though real capital existed at the
    open to fund it: regression test for exactly this (a rotation into a
    brand-new symbol, on the SAME row an overnight leg wipes the
    portfolio out, silently landing anyway against the flattening
    mechanism's own fictitious reset baseline)."""
    held, asset_returns, overnight_returns = _overnight_wipeout_scenario()
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=100.0, spread_bps=0.0, slippage_bps=0.0)
    )
    acc = run_accounting(
        held,
        asset_returns,
        model,
        100_000.0,
        overnight_returns=overnight_returns,
        model_weight_drift=True,
    )
    assert acc.gross_returns.iloc[3] == pytest.approx(-1.0)
    assert acc.equity.iloc[3] == pytest.approx(0.0)
    assert acc.turnover.iloc[3] == pytest.approx(0.0)
    assert acc.costs.total.iloc[3] == pytest.approx(0.0)
    assert not acc.ruined.iloc[3]
    assert acc.ruined.iloc[4]
    # The wipeout row itself -- not just row 4 onward -- must report flat,
    # never the scheduled B rotation no real capital could have funded.
    assert (acc.executed_weights.iloc[3] == 0.0).all()
    assert (acc.weight_changes.iloc[3] == 0.0).all()


def test_overnight_wipeout_stays_exactly_minus_one_under_extreme_leverage() -> None:
    """Regression test: a wipeout row's ``gross_returns`` must collapse to
    exactly -1 even when a freshly-decided (changed) cell's raw
    ``executed`` weight is extreme and combines with a large intraday
    return -- relying solely on ``(1+overnight_leg)`` being near-zero to
    swamp that product would silently fail once the row lands just below
    the EPSILON wipeout threshold (nonzero, not exactly 0) and the
    changed cell's own contribution is large enough to matter numerically
    at that scale."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    # 2x leverage in A, overnight gap tuned so (1+overnight_leg) lands
    # just BELOW EPSILON (1e-12) but is not exactly zero.
    overnight_leg_residual = 5e-13
    overnight_a = (overnight_leg_residual - 1.0) / 2.0
    held = pd.DataFrame(
        {"A": [2.0, 2.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0e8, 1.0e8]}, index=idx
    )
    # executed: [0,0] -> [2,0] -> [2,0] -> [0,1e8]; the extreme rotation
    # into B lands on the same row the gap hits (index 3).
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, overnight_a], "B": [0.0, 0.0, 0.0, 1.0]}, index=idx
    )
    overnight_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, overnight_a], "B": [0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    )
    acc = run_accounting(
        held, asset_returns, model, 100_000.0, overnight_returns=overnight_returns
    )
    assert acc.gross_returns.iloc[3] == pytest.approx(-1.0, abs=1e-9)
    assert acc.equity.iloc[3] == pytest.approx(0.0, abs=1e-3)
    assert acc.turnover.iloc[3] == pytest.approx(0.0)


def test_group_return_stays_exactly_minus_one_under_extreme_leverage() -> None:
    """Regression test: ``_group_return_with_execution_timing`` (the
    stop-loss/take-profit group-level twin of ``_gross_returns_and_cost_
    basis``) needs the exact same wipeout guard, for the exact same
    reason -- unmasked, it can produce a materially wrong finite value
    instead of collapsing to -1 on a genuine group wipeout.

    A pairs-style group (long A, short B, both 2x the group's own gross
    exposure in magnitude) whose short leg (B) suffers a ~200% overnight
    move wipes out the GROUP's aggregate overnight leg to just below the
    EPSILON threshold, even though A's own pre-trade dollar amount (100)
    stays large. Unmasked, that large numerator divided by the near-zero
    EPSILON-floored denominator makes ``weight_at_open`` blow up (~1e12),
    and combining it with A's 100% intraday return produces
    ``group_return == -0.5`` -- not -1 -- because the huge intraday
    contribution (rather than the near-zero ``(1+overnight_leg)`` factor)
    numerically dominates the compounding. This exact reduced scenario
    was confirmed to reproduce -0.5 against a copy of this function with
    the ``wiped_out`` masking removed.
    """
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    pre = pd.DataFrame({"A": [100.0], "B": [-100.0]}, index=idx)
    executed = pd.DataFrame({"A": [100.0], "B": [-100.0]}, index=idx)
    weight_changes = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=idx)
    # Tuned so the group's aggregate (1+overnight_leg) lands just BELOW
    # EPSILON (1e-12) but is not exactly zero -- pre_exposure=200,
    # overnight_dollar = 100*0 + (-100)*overnight_b = 200 - 1e-10, so
    # (1+overnight_leg) = (pre_exposure+overnight_dollar)/pre_exposure = 5e-13.
    overnight_b = 2.0 - 1e-12
    overnight_returns = pd.DataFrame({"A": [0.0], "B": [overnight_b]}, index=idx)
    # A's own asset_return is a full +100% intraday move (overnight leg is
    # 0 for A); B's asset_return equals its own overnight return (no
    # intraday move for B).
    asset_returns = pd.DataFrame({"A": [1.0], "B": [overnight_b]}, index=idx)
    columns = ["A", "B"]
    gross_exposure = executed.abs().sum(axis=1).to_numpy(dtype=float)

    group_return = _group_return_with_execution_timing(
        pre,
        executed,
        asset_returns,
        weight_changes,
        columns,
        gross_exposure,
        overnight_returns=overnight_returns,
    )
    assert group_return[0] == pytest.approx(-1.0, abs=1e-9)


def test_group_return_uses_one_shared_basis_for_traded_and_untraded_cells() -> None:
    """Regression test: a multi-symbol group where ONE symbol trades this
    row and another does not must combine their contributions on the SAME
    basis, not silently mix the untraded cell's drift-corrected
    ``weight_at_open`` (normalized by the group's OWN open-time exposure)
    with the traded cell's ``executed`` normalized by a DIFFERENT
    quantity (the caller's raw, post-trade ``gross_exposure``) -- that
    mismatch under- or over-weights whichever cell happens to use the
    smaller denominator, corrupting the group's compounded return and
    with it any stop-loss/take-profit threshold comparison built on it.

    A pre-existing group holds A and B at 0.1 each. Overnight, A gains
    10% and B is flat (pre_exposure=0.2, overnight_dollar=0.01,
    overnight_leg=0.05). At the open, A is traded down to executed=0.05
    (a fresh decision, already expressed relative to the grown open-time
    equity, per this module's own convention); B is untraded, so its
    contribution must be its organically-drifted weight_at_open, not its
    stale pre-trade value. A also earns a +20% intraday move; B earns 0%.

    Hand-derived (dollar P&L, divided by the group's PRE-trade exposure --
    the correct "beginning of period" basis, see this function's own
    docstring): total dollar P&L = overnight_dollar (0.01) + A's intraday
    contribution (executed_A * intraday_A = 0.05 * 0.20 = 0.01) + B's
    intraday contribution (0, flat) = 0.02. group_return = 0.02 / 0.2 =
    0.10 -- not 0.12, the value this exact scenario produces if the
    traded cell is instead (wrongly) normalized by gross_exposure=0.15.
    """
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    pre = pd.DataFrame({"A": [0.1], "B": [0.1]}, index=idx)
    executed = pd.DataFrame({"A": [0.05], "B": [0.1]}, index=idx)
    weight_changes = pd.DataFrame({"A": [-0.06], "B": [0.0]}, index=idx)
    overnight_returns = pd.DataFrame({"A": [0.10], "B": [0.0]}, index=idx)
    intraday_a = 0.20
    asset_returns = pd.DataFrame(
        {"A": [(1.0 + 0.10) * (1.0 + intraday_a) - 1.0], "B": [0.0]}, index=idx
    )
    columns = ["A", "B"]
    gross_exposure = executed.abs().sum(axis=1).to_numpy(dtype=float)

    group_return = _group_return_with_execution_timing(
        pre,
        executed,
        asset_returns,
        weight_changes,
        columns,
        gross_exposure,
        overnight_returns=overnight_returns,
    )
    assert group_return[0] == pytest.approx(0.10, abs=1e-9)


def test_group_return_matches_the_plain_formula_when_nothing_trades() -> None:
    """The basis fix above must not disturb the (already-guaranteed) case
    where nothing in the group trades this row -- the decomposed
    two-leg formula must still collapse EXACTLY to the plain,
    undecomposed ``(executed*asset_returns).sum()/gross_exposure``
    formula, matching ``overnight_returns=None``'s own behavior."""
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    pre = pd.DataFrame({"A": [0.1], "B": [0.1]}, index=idx)
    executed = pre.copy()
    weight_changes = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=idx)
    overnight_returns = pd.DataFrame({"A": [0.10], "B": [0.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.10], "B": [0.0]}, index=idx)
    columns = ["A", "B"]
    gross_exposure = executed.abs().sum(axis=1).to_numpy(dtype=float)

    group_return = _group_return_with_execution_timing(
        pre,
        executed,
        asset_returns,
        weight_changes,
        columns,
        gross_exposure,
        overnight_returns=overnight_returns,
    )
    plain = (executed[columns] * asset_returns[columns]).sum(axis=1).to_numpy() / (
        gross_exposure
    )
    assert group_return[0] == pytest.approx(plain[0])


def test_group_return_stays_sane_on_a_fresh_entry_from_flat() -> None:
    """Regression test: a group with NO pre-trade exposure at all (a
    fresh entry from flat -- the ordinary, common case for any brand-new
    position, not an edge case) must not explode. Basis fix: this
    function's own `pre_exposure * (1 + overnight_leg)` normalization
    for changed cells collapses to ~0 when nothing was held before today
    (`pre_exposure == 0`), so dividing by it (even EPSILON-floored)
    inflates a fresh position's own `executed` weight by a factor of
    roughly `1/EPSILON` -- confirmed to reproduce `group_return ~= 2e9`
    for this exact scenario before falling back to `gross_exposure`
    (today's actual, as-executed exposure) whenever `pre_exposure` is
    ~0. With nothing held overnight, `overnight_leg` is correctly 0
    already, so the day's return is purely the intraday leg on the
    freshly-entered position -- exactly the plain, undecomposed formula
    result."""
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    pre = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=idx)
    executed = pd.DataFrame({"A": [0.10], "B": [0.0]}, index=idx)
    weight_changes = pd.DataFrame({"A": [0.10], "B": [0.0]}, index=idx)
    overnight_returns = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.02], "B": [0.0]}, index=idx)
    columns = ["A", "B"]
    gross_exposure = executed.abs().sum(axis=1).to_numpy(dtype=float)

    group_return = _group_return_with_execution_timing(
        pre,
        executed,
        asset_returns,
        weight_changes,
        columns,
        gross_exposure,
        overnight_returns=overnight_returns,
    )
    assert group_return[0] == pytest.approx(0.02)


def test_close_of_previous_row_stays_zero_after_a_leveraged_wipeout() -> None:
    """Regression test: ``_close_of_previous_row`` (recovers ``pre`` for
    weight-drift-active next_bar_open accounting) needs the exact same
    wipeout guard as its sibling ``_gross_returns_and_cost_basis`` --
    unmasked, a row whose portfolio-wide intraday leg is a total loss
    makes ``(1+intraday_leg)`` collapse to (near-)zero, and dividing an
    unbounded (leveraged, offsetting long/short) ``executed`` value by
    that EPSILON-floored denominator produces an astronomical, not zero,
    ``close_weight`` -- which then silently corrupts the NEXT row's own
    ``pre``/overnight leg.

    Long A (+100%) and short B (-100%), intraday returns -60%/+40%: the
    portfolio-wide intraday leg is exactly -100% (1.0*-0.6 + (-1.0)*0.4
    = -1.0), a genuine total wipeout despite each individual leg's
    return being modest. Confirmed to reproduce close_weight ==
    [4e11, -1.4e12] instead of [0, 0] before this guard was added.
    """
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [-1.0, -1.0, 0.0]}, index=idx)
    intraday_returns = pd.DataFrame(
        {"A": [0.0, -0.6, 0.0], "B": [0.0, 0.4, 0.0]}, index=idx
    )

    result = _close_of_previous_row(executed, intraday_returns)
    # Row 3's `pre` is row 2's close_weight (shifted forward by one row) --
    # the wiped-out row's own contribution.
    assert result.iloc[2].tolist() == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    "with_gap", [False, True], ids=["same_bar_close", "next_bar_open"]
)
def test_trade_log_costs_reconcile_exactly_with_accounting_costs(
    with_gap: bool,
) -> None:
    """The trade log's per-fill commission/spread/slippage must sum, per
    date, to EXACTLY the same dollar figures ``AccountingResult.costs``
    itself reports -- for both a plain run (``overnight_returns=None``,
    also covers ``immediate_after_close``, identical at this level) and a
    next_bar_open run with a genuine overnight gap on the very day of a
    rotation. Requires sourcing the trade log's notional from
    ``weight_changes_for_costs``/``execution_equity``, not the plain
    ``weight_changes``/``equity`` pair, which under a next_bar_open gap
    are on a different basis than what ``costs`` was actually sized from.
    """
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    held = pd.DataFrame(
        {"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 1.0]}, index=idx
    )
    # executed: [0,0] -> [1,0] -> [1,0] -> [0,1]; the A->B rotation lands
    # at index 3, the same row the (optional) gap hits.
    gap = 0.20 if with_gap else 0.0
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0, gap], "B": [0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    overnight_returns = (
        pd.DataFrame({"A": [0.0, 0.0, 0.0, gap], "B": [0.0, 0.0, 0.0, 0.0]}, index=idx)
        if with_gap
        else None
    )
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=50.0, spread_bps=20.0, slippage_bps=10.0)
    )
    acc = run_accounting(
        held, asset_returns, model, 100_000.0, overnight_returns=overnight_returns
    )
    prices = pd.DataFrame({"A": [100.0] * 4, "B": [50.0] * 4}, index=idx)
    trades = build_trade_log(
        acc.executed_weights,
        acc.weight_changes_for_costs,
        acc.equity,
        prices,
        commission_bps=50.0,
        spread_bps=20.0,
        slippage_model=model.slippage,
        slippage_equity=acc.equity_for_costs,
        execution_equity=acc.execution_equity,
    )
    trade_log_total_by_date = (
        trades.groupby("timestamp")["total_cost"].sum().reindex(idx, fill_value=0.0)
    )
    # `costs.total` is a fraction of close(T-1) equity; the real dollar
    # figure it corresponds to is that fraction times the equity it was
    # subtracted from, i.e. equity.shift(1) (same basis for every
    # execution_timing mode -- see accounting.py's own cost-basis fix).
    accounting_dollar_cost = acc.costs.total * acc.equity.shift(1).fillna(100_000.0)
    pd.testing.assert_series_equal(
        trade_log_total_by_date,
        accounting_dollar_cost,
        check_names=False,
        check_freq=False,
    )
    assert trade_log_total_by_date.sum() > 0.0  # not vacuously all-zero
