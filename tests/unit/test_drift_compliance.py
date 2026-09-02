"""Direct tests for the drift-compliance LP (`restore_drift_compliance`).

Each test asserts against a hand-derived closed-form minimal-L1-distance
solution, not just "no crash" -- this is the riskiest piece of math behind
weight drift, so its correctness must be nailed down before it is wired
into `accounting.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quantlab.exceptions import BacktestError
from quantlab.portfolio.drift_compliance import restore_drift_compliance


def test_lone_maximum_weight_breach_clips_exactly_to_the_cap() -> None:
    result = restore_drift_compliance(
        np.array([0.5]),
        ["A"],
        np.array([True]),
        [("A",)],
        maximum_weight=0.3,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert not result.pending
    assert result.corrected[0] == pytest.approx(0.3)


def test_lone_gross_exposure_breach_clips_the_only_column() -> None:
    result = restore_drift_compliance(
        np.array([0.5]),
        ["A"],
        np.array([True]),
        [("A",)],
        maximum_weight=None,
        maximum_gross_exposure=0.3,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert not result.pending
    assert result.corrected[0] == pytest.approx(0.3)


def test_lone_long_only_breach_snaps_to_zero() -> None:
    """Nearest point on `w >= 0` to a negative drifted value is exactly 0."""
    result = restore_drift_compliance(
        np.array([-0.2]),
        ["A"],
        np.array([True]),
        [("A",)],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=True,
    )
    assert not result.pending
    assert result.corrected[0] == pytest.approx(0.0)


def test_closed_long_tradable_short_counterexample_pushes_short_more_negative() -> None:
    """The scenario that disproves "clip then scale tradable columns toward
    0" as a general solution: a large UNTRADABLE long position plus a
    TRADABLE short breaching maximum_net_exposure needs the short pushed
    MORE negative (away from 0), not scaled toward 0."""
    drifted = np.array([0.9, -0.1])  # A untradable, B tradable
    result = restore_drift_compliance(
        drifted,
        ["A", "B"],
        np.array([False, True]),
        [("A",), ("B",)],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=0.5,
        long_only=False,
    )
    assert not result.pending
    # A is untradable -- unchanged.
    assert result.corrected[0] == pytest.approx(0.9)
    # Feasible region for B: 0.9 + w_B in [-0.5, 0.5] => w_B in [-1.4, -0.4].
    # Nearest point to drifted B=-0.1 is -0.4 -- MORE negative, not toward 0.
    assert result.corrected[1] == pytest.approx(-0.4)
    assert result.corrected[1] < drifted[1]


def test_uninvolved_compliant_column_is_never_liquidated_by_an_unrelated_breach() -> (
    None
):
    """An uninvolved, already-compliant column must never be moved just
    because an UNRELATED constraint violation is being fixed elsewhere in
    the row: a single-stage slack-only relaxation that minimizes ONLY the
    constraint-violation slacks, with no term penalizing movement of such
    a column, would let the solver pick any optimal vertex, including one
    that arbitrarily liquidates B even though `maximum_weight` (a purely
    per-column cap) doesn't even reference B. A closed at 0.6 alone
    already breaches `maximum_weight=0.5` (tradability-caused, pending);
    B, open and compliant at 0.4, must stay exactly where it is -- the
    two-stage lexicographic fix (fix the minimal violation, then minimize
    L1 deviation among solutions achieving it) has no reason to move a
    column the violation doesn't involve."""
    result = restore_drift_compliance(
        np.array([0.6, 0.4]),
        ["A", "B"],
        np.array([False, True]),
        [("A",), ("B",)],
        maximum_weight=0.5,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert result.pending
    assert result.corrected[0] == pytest.approx(0.6)
    assert result.corrected[1] == pytest.approx(0.4)


def test_never_opens_a_new_position_on_a_column_drifted_to_exactly_zero() -> None:
    """Regression test: the LP must never invent a brand-new position
    (long OR short) on a column the drifted book does not already hold,
    even when doing so would have been the cheapest (or only) way to
    restore compliance. Before sign/support-preservation, this exact
    scenario would have opened a short on B (the strict LP was feasible
    by setting w_B=-0.1) -- a hedge the strategy never asked for and
    portfolio.long_only=False alone would have silently allowed. B must
    now stay fixed at exactly 0, and since the untradable A alone (0.6)
    already exceeds the 0.5 cap, this becomes a tradability-caused
    pending correction instead -- an honest "still breaching, waiting
    for A to reopen" rather than a silently manufactured hedge."""
    drifted = np.array([0.6, 0.0])  # A untradable long 0.6, B tradable AT ZERO
    result = restore_drift_compliance(
        drifted,
        ["A", "B"],
        np.array([False, True]),
        [("A",), ("B",)],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=0.5,
        long_only=False,
    )
    assert result.pending
    assert result.corrected[0] == pytest.approx(0.6)
    assert result.corrected[1] == pytest.approx(0.0)


def test_existing_long_may_not_flip_to_a_new_short_or_vice_versa() -> None:
    """A currently-LONG column may shrink toward 0 (or grow further long)
    but must never cross into short territory, and vice versa -- crossing
    zero is just as much "inventing a position the drifted book didn't
    hold" as starting from exactly zero would be."""
    # A long-only column (0.4) would need to go negative to satisfy this
    # net cap alongside an untradable 0.3 -- but it may only shrink to 0.
    drifted = np.array([0.3, 0.4])  # A untradable, B tradable LONG
    result = restore_drift_compliance(
        drifted,
        ["A", "B"],
        np.array([False, True]),
        [("A",), ("B",)],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=0.2,
        long_only=False,
    )
    # 0.3 (fixed) alone already exceeds the 0.2 cap -- tradability-caused,
    # B shrinks to its floor of 0 (not below), still pending.
    assert result.pending
    assert result.corrected[0] == pytest.approx(0.3)
    assert result.corrected[1] == pytest.approx(0.0)


def test_position_group_moves_both_legs_via_one_shared_scalar() -> None:
    """A declared group's legs must move together (`k_g`), never as
    independent free variables -- a per-column LP could satisfy the
    objective by moving only one leg, breaking the pair's hedge ratio."""
    drifted = np.array([0.6, -0.3])  # X, Y -- one group, both tradable
    result = restore_drift_compliance(
        drifted,
        ["X", "Y"],
        np.array([True, True]),
        [("X", "Y")],
        maximum_weight=0.5,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert not result.pending
    # k_g <= 0.5/0.6 = 0.8333... (from X's own maximum_weight bound);
    # minimal-L1 picks k_g as close to 1 as feasible => k_g = 0.8333...
    expected_k = 0.5 / 0.6
    assert result.corrected[0] == pytest.approx(expected_k * 0.6)
    assert result.corrected[1] == pytest.approx(expected_k * -0.3)
    # The ratio between the two legs is exactly preserved (coherent move).
    assert result.corrected[0] / result.corrected[1] == pytest.approx(
        drifted[0] / drifted[1]
    )


def test_position_group_with_one_untradable_leg_is_fixed_entirely() -> None:
    """A group is only eligible to move when EVERY leg is tradable -- one
    untradable leg must fix `k_g` at exactly 1 (the whole group frozen at
    its drifted proportions), never let the tradable leg move alone. Only
    the trivial both-tradable/both-untradable cases were tested before
    this: X alone (untradable, 0.6) already exceeds `maximum_weight=0.5`,
    so this is tradability-caused and pending -- the group stays exactly
    at its drifted values rather than Y moving independently."""
    drifted = np.array([0.6, -0.3])  # X untradable, Y tradable, one group
    result = restore_drift_compliance(
        drifted,
        ["X", "Y"],
        np.array([False, True]),
        [("X", "Y")],
        maximum_weight=0.5,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert result.pending
    assert result.corrected[0] == pytest.approx(0.6)
    assert result.corrected[1] == pytest.approx(-0.3)


def test_position_group_gross_exposure_cap_scales_k_g() -> None:
    """Group + `maximum_gross_exposure` interaction -- only per-leg
    `maximum_weight`/net-exposure combos with a group were tested before
    this. Both legs tradable, gross = |0.6| + |-0.3| = 0.9 at k_g=1;
    capped to 0.6 forces k_g <= 0.6/0.9 = 0.6667, and minimal-L1 picks the
    largest feasible k_g (closest to 1)."""
    drifted = np.array([0.6, -0.3])
    result = restore_drift_compliance(
        drifted,
        ["X", "Y"],
        np.array([True, True]),
        [("X", "Y")],
        maximum_weight=None,
        maximum_gross_exposure=0.6,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert not result.pending
    expected_k = 0.6 / 0.9
    assert result.corrected[0] == pytest.approx(expected_k * 0.6)
    assert result.corrected[1] == pytest.approx(expected_k * -0.3)
    assert result.corrected[0] / result.corrected[1] == pytest.approx(
        drifted[0] / drifted[1]
    )


def test_untradable_independent_column_negative_under_long_only_is_pending() -> None:
    """Regression test: `long_only` was only ever baked into a TRADABLE
    independent column's own lower bound -- silently a no-op for an
    UNTRADABLE (fixed) column, whose bound is pinned at its drifted value
    regardless of sign. Before the fix, the strict LP had no constraint
    that could ever reject a fixed negative value under long_only, so it
    trivially "succeeded" (`pending=False`) over a row that still
    genuinely violated long_only -- exactly the silent formulation bug
    this module exists to avoid elsewhere. A fixed column alone violating
    long_only must be diagnosed as tradability-caused (mirroring
    `_fixed_positions_alone_violate`'s own long_only branch, previously
    unreachable) and returned as a `pending`, best-effort (here: fully
    unchanged, since nothing else is free) correction."""
    from quantlab.portfolio.rebalancing import _compliance_violations

    result = restore_drift_compliance(
        np.array([-0.2]),
        ["A"],
        np.array([False]),
        [("A",)],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=True,
    )
    assert result.pending
    assert result.corrected[0] == pytest.approx(-0.2)
    assert _compliance_violations(
        result.corrected,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=True,
    ) == ["long_only"]


def test_untradable_group_with_a_negative_leg_under_long_only_is_pending() -> None:
    """Same bug as the independent-column case above, for a group: an
    untradable group fixes `k_g=1`, and a negative leg at k_g=1 was
    silently accepted as "compliant" before this constraint existed."""
    result = restore_drift_compliance(
        np.array([-0.2, 0.1]),
        ["X", "Y"],
        np.array([False, False]),
        [("X", "Y")],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=True,
    )
    assert result.pending
    assert result.corrected[0] == pytest.approx(-0.2)
    assert result.corrected[1] == pytest.approx(0.1)


def test_position_group_can_grow_past_one_when_genuinely_minimal() -> None:
    """`k_g` is bounded only by `k_g >= 0`, never capped at 1 -- capping it
    would incorrectly exclude a real minimal-L1 solution that requires
    growing a group. Here an untradable, fixed column `C` alone already
    pushes net exposure to 0.8; the only way to bring it back within
    `maximum_net_exposure=0.5` is to grow the (net-negative) hedge group
    past its own drifted proportions, offsetting C -- shrinking or leaving
    it at k_g=1 cannot satisfy the constraint at all, since C cannot move.
    """
    drifted = np.array([0.8, 0.1, -0.3])  # group net (at k_g=1) = -0.2
    result = restore_drift_compliance(
        drifted,
        ["C", "X", "Y"],
        np.array([False, True, True]),
        [("C",), ("X", "Y")],
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=0.5,
        long_only=False,
    )
    assert not result.pending
    # C is untradable and stays fixed; solving 0.8 - 0.2*k_g == 0.5 (the
    # nearest feasible net exposure to k_g=1) gives k_g == 1.5.
    expected_k = 1.5
    assert result.corrected[0] == pytest.approx(0.8)
    assert result.corrected[1] == pytest.approx(expected_k * 0.1)
    assert result.corrected[2] == pytest.approx(expected_k * -0.3)
    assert result.corrected[1] / result.corrected[2] == pytest.approx(
        drifted[1] / drifted[2]
    )


def test_tradability_caused_infeasibility_uses_slack_relaxation_and_flags_pending() -> (
    None
):
    """An untradable column's own drifted value already violates
    maximum_weight -- no amount of free-column movement can fix it. The
    slack-relaxation fallback must fire (not raise) and flag `pending`."""
    result = restore_drift_compliance(
        np.array([0.9]),
        ["A"],
        np.array([False]),
        [("A",)],
        maximum_weight=0.5,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert result.pending
    # Nothing free exists to help -- the untradable column is unchanged.
    assert result.corrected[0] == pytest.approx(0.9)


def test_tradability_caused_infeasibility_uses_free_columns_to_help() -> None:
    """When some OTHER column is free, the slack-relaxation solve should
    still let it move to reduce the aggregate breach as much as
    achievable, even though the untradable column itself can't be fixed."""
    drifted = np.array([0.9, 0.3])  # A untradable & breaches alone; B tradable
    result = restore_drift_compliance(
        drifted,
        ["A", "B"],
        np.array([False, True]),
        [("A",), ("B",)],
        maximum_weight=None,
        maximum_gross_exposure=0.5,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert result.pending
    assert result.corrected[0] == pytest.approx(0.9)
    # B is free and should be pulled toward 0 to minimize the slack needed.
    assert result.corrected[1] < drifted[1]


def test_always_feasible_and_never_pending_when_everything_is_tradable() -> None:
    """With every column tradable, 0 is always a feasible point for every
    free variable, so the strict LP can never be genuinely infeasible for
    a reason other than tradability -- this is what makes the "loud raise"
    branch an unreachable defensive invariant (mirroring `_assert_
    holdings_compliant`'s identical philosophy) under any valid,
    non-negative constraint configuration."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        drifted = rng.normal(scale=1.5, size=4)
        result = restore_drift_compliance(
            drifted,
            ["A", "B", "C", "D"],
            np.array([True, True, True, True]),
            [("A",), ("B",), ("C",), ("D",)],
            maximum_weight=0.4,
            maximum_gross_exposure=1.0,
            maximum_net_exposure=0.6,
            long_only=False,
        )
        assert not result.pending
        assert np.all(np.abs(result.corrected) <= 0.4 + 1e-6)
        assert np.sum(np.abs(result.corrected)) <= 1.0 + 1e-6
        assert abs(np.sum(result.corrected)) <= 0.6 + 1e-6


def test_untradable_columns_are_bit_for_bit_unchanged_by_the_lp() -> None:
    drifted = np.array([0.2, 0.9, -0.3])
    result = restore_drift_compliance(
        drifted,
        ["A", "B", "C"],
        np.array([True, False, True]),
        [("A",), ("B",), ("C",)],
        maximum_weight=0.5,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert result.corrected[1] == drifted[1]


def test_shape_mismatch_raises() -> None:
    with pytest.raises(BacktestError):
        restore_drift_compliance(
            np.array([0.1, 0.2]),
            ["A"],
            np.array([True]),
            [("A",)],
            maximum_weight=None,
            maximum_gross_exposure=None,
            maximum_net_exposure=None,
            long_only=False,
        )


def _fake_linprog_result(status: int) -> Any:
    from scipy.optimize import OptimizeResult

    return OptimizeResult(x=np.zeros(1), status=status, message="synthetic status")


def test_strict_lp_non_infeasible_solver_failure_raises_not_silently_diagnosed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HiGHS status that is neither optimal (0) nor infeasible (2) -- an
    iteration limit, an unbounded report, or numerical difficulties -- is a
    genuine solver failure, never the ordinary "tradability caused this"
    infeasibility the caller's diagnosis branch expects. Folding it into
    the same `None` return the real infeasible case uses would let a
    solver hiccup silently masquerade as an expected, best-effort
    correction (or an unrelated 'bug in the algorithm' report) instead of
    surfacing loudly with the solver's own status."""
    import quantlab.portfolio.drift_compliance as drift_compliance_mod

    monkeypatch.setattr(
        drift_compliance_mod, "linprog", lambda *a, **k: _fake_linprog_result(4)
    )

    with pytest.raises(BacktestError, match="non-infeasible, non-optimal"):
        restore_drift_compliance(
            np.array([0.5]),
            ["A"],
            np.array([True]),
            [("A",)],
            maximum_weight=0.3,
            maximum_gross_exposure=None,
            maximum_net_exposure=None,
            long_only=False,
        )


def test_slack_stage1_non_infeasible_solver_failure_raises() -> None:
    """Same principle as the strict-LP case, for stage 1 of the slack-mode
    solve: only genuine infeasibility (status 2) is treated as the
    expected 'nothing to relax' case the always-feasible slack LP should
    never actually hit; every other non-optimal status is a real failure
    and must raise, not vanish into the caller's generic 'unexpectedly
    infeasible' message."""
    from scipy.optimize import linprog as real_linprog

    import quantlab.portfolio.drift_compliance as drift_compliance_mod

    calls = {"n": 0}

    def fake_linprog(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            # The strict LP: force genuine infeasibility so the caller
            # proceeds to the tradability diagnosis and slack mode.
            return _fake_linprog_result(2)
        # Stage 1 of the slack-mode solve.
        return _fake_linprog_result(1)

    drift_compliance_mod.linprog = fake_linprog
    try:
        with pytest.raises(BacktestError, match="stage 1 failed"):
            # A alone (untradable) already violates maximum_weight -- a
            # genuine tradability-caused breach, so the strict LP's forced
            # infeasibility above is consistent with the real diagnosis.
            restore_drift_compliance(
                np.array([0.6]),
                ["A"],
                np.array([False]),
                [("A",)],
                maximum_weight=0.5,
                maximum_gross_exposure=None,
                maximum_net_exposure=None,
                long_only=False,
            )
    finally:
        drift_compliance_mod.linprog = real_linprog


def test_slack_stage2_failure_raises_not_stage1s_unpenalized_solution() -> None:
    """Stage 1 only minimizes total slack, with zero cost on every other
    free variable -- if stage 2 (which adds the real L1-deviation
    objective) unexpectedly fails to solve, silently substituting stage
    1's own solution could return an arbitrary, needlessly destructive
    correction (e.g. liquidating an uninvolved column) with no indication
    the fallback path was taken. This must raise instead."""
    from scipy.optimize import linprog as real_linprog

    import quantlab.portfolio.drift_compliance as drift_compliance_mod

    calls = {"n": 0}

    def fake_linprog(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            # Strict LP (infeasible) then stage 1 (slack) -- both real, so
            # stage 1's own solution is genuinely feasible for stage 2.
            return real_linprog(*args, **kwargs)
        # Stage 2.
        return _fake_linprog_result(4)

    drift_compliance_mod.linprog = fake_linprog
    try:
        with pytest.raises(BacktestError, match="stage 2 unexpectedly failed"):
            restore_drift_compliance(
                np.array([0.6]),
                ["A"],
                np.array([False]),
                [("A",)],
                maximum_weight=0.5,
                maximum_gross_exposure=None,
                maximum_net_exposure=None,
                long_only=False,
            )
    finally:
        drift_compliance_mod.linprog = real_linprog
