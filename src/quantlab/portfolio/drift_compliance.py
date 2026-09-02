"""Minimal-L1-turnover restoration of a drift-breached row's compliance.

Between rebalances, organic price drift (see
:func:`quantlab.backtesting.accounting.apply_weight_drift`) can push a row
past a hard portfolio-level risk limit (``maximum_weight``,
``maximum_gross_exposure``/``maximum_leverage``, ``maximum_net_exposure``,
``long_only``) even though the last REAL decision was itself fully
compliant. Restoring compliance off-schedule is a genuine constrained
optimization, not a heuristic: a naive "clip then scale toward 0" fix can
move exposure in the WRONG direction whenever some of the breaching
exposure sits in a currently-untradable column (see
:func:`restore_drift_compliance`'s own docstring for the exact
counterexample this module exists to avoid).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from quantlab.constants import EPSILON
from quantlab.exceptions import BacktestError

#: scipy.optimize.linprog's HiGHS status code for "solved to optimality".
_LINPROG_OPTIMAL = 0
#: HiGHS status code for "provably infeasible" -- the ONLY non-optimal
#: status treated as an ordinary, expected outcome (tradability-caused).
#: Every other non-optimal status (1: iteration limit, 3: unbounded, 4:
#: numerical difficulties) is a genuine solver failure, never folded into
#: the same "infeasible" bucket -- see `_solve`'s own docstring.
_LINPROG_INFEASIBLE = 2


@dataclass(frozen=True)
class DriftComplianceResult:
    """One row's outcome from :func:`restore_drift_compliance`.

    ``corrected`` is the row's new weights. ``pending`` is ``True`` only
    when full compliance was genuinely unachievable given which columns
    are currently tradable (the slack-relaxation fallback fired) --
    ``corrected`` is then the best achievable correction, not a fully
    compliant row, and the caller is expected to retry this row's
    successor once the responsible column(s) reopen.
    """

    corrected: np.ndarray
    pending: bool


def restore_drift_compliance(
    drifted: np.ndarray,
    columns: Sequence[str],
    tradable_row: np.ndarray,
    groups: Sequence[tuple[str, ...]],
    *,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
) -> DriftComplianceResult:
    """Return the minimal-L1-turnover row that restores compliance.

    Internal, low-level primitive: its sole caller,
    :func:`quantlab.backtesting.accounting.apply_weight_drift`, is the
    validated public entry point (frame shape/dtype/finiteness, tradable
    mask, etc.). This function only asserts array shapes match ``columns``
    below -- it trusts ``drifted``/``tradable_row`` are otherwise already
    clean numeric/boolean data, and is not meant to be called directly on
    unvalidated input. Full LP formulation (free variables, objective,
    constraints, the infeasibility-diagnosis/slack-relaxation fallback,
    and the two disclosed scope limits): see
    docs/drift_compliance.md#compliance-restoration-lp.

    Key invariants a caller relies on: a declared position group (e.g.
    pairs_trading's two legs, via ``groups``) always moves as one
    coherent unit via a single shared scaling factor, never one leg
    alone; an untradable column/group is always returned bit-for-bit
    unchanged from ``drifted``; the result never invents a brand-new
    position (long or short) on a column/group the drifted book did not
    already hold, even when that would be the cheapest fix. ``pending``
    is ``True`` only when full compliance was genuinely unachievable
    given current tradability (``corrected`` is then the best achievable
    partial fix, not a fully compliant row) -- a constraint configuration
    that is infeasible for any OTHER reason raises ``BacktestError``
    instead of returning a result, since that indicates a bug (a
    contradictory configuration ``_validate_target_row_compliant`` should
    already have rejected upstream), never a legitimate runtime outcome.
    """
    n = len(columns)
    if drifted.shape != (n,) or tradable_row.shape != (n,):
        raise BacktestError(
            "drifted and tradable_row must be 1-D arrays matching columns."
        )
    column_index = {name: i for i, name in enumerate(columns)}

    indep_columns: list[str] = []
    group_legs: list[list[int]] = []  # column indices per multi-column group
    for group in groups:
        if len(group) == 1:
            indep_columns.append(group[0])
        else:
            group_legs.append([column_index[symbol] for symbol in group])

    indep_idx = [column_index[symbol] for symbol in indep_columns]
    indep_tradable = [bool(tradable_row[i]) for i in indep_idx]
    group_tradable = [all(bool(tradable_row[i]) for i in legs) for legs in group_legs]
    group_l1_norm = [float(np.sum(np.abs(drifted[legs]))) for legs in group_legs]
    group_net = [float(np.sum(drifted[legs])) for legs in group_legs]

    gross_cap = maximum_gross_exposure

    solved = _solve(
        drifted,
        indep_idx,
        indep_tradable,
        group_legs,
        group_tradable,
        group_l1_norm,
        group_net,
        maximum_weight=maximum_weight,
        gross_cap=gross_cap,
        maximum_net_exposure=maximum_net_exposure,
        long_only=long_only,
        allow_slack=False,
    )
    if solved is not None:
        return DriftComplianceResult(corrected=solved, pending=False)

    # Diagnose: is infeasibility explained by the fixed (untradable)
    # positions alone? Build the row that WOULD result from every fixed
    # position at its drifted value and every free position at exactly
    # its own drifted value too (i.e. "no correction at all") and check
    # whether the fixed subset's own contribution already breaches a cap
    # that no amount of free-column movement could ever repair (a
    # portfolio-level cap breached by the fixed positions alone, or a
    # per-asset cap breached by a fixed position's own value).
    fixed_only_violation = _fixed_positions_alone_violate(
        drifted,
        indep_idx,
        indep_tradable,
        group_legs,
        group_tradable,
        maximum_weight=maximum_weight,
        gross_cap=gross_cap,
        maximum_net_exposure=maximum_net_exposure,
        long_only=long_only,
    )
    if not fixed_only_violation:  # pragma: no cover - defensive, should be unreachable
        # 0 is always a feasible point for every free (tradable) variable
        # under any valid, non-negative constraint configuration (see
        # test_always_feasible_and_never_pending_when_everything_is_
        # tradable's own reasoning) -- so a strict-LP infeasibility not
        # explained by the fixed/untradable subset alone should never
        # actually happen for input `_validate_drift_and_risk_options`/
        # `_validate_target_row_compliant` have already validated. Kept as
        # a loud, explicit guard rather than silently reaching the slack-
        # relaxation path for a reason tradability doesn't actually explain.
        raise BacktestError(
            "Drift-compliance restoration is infeasible for a reason other "
            "than tradability -- this indicates a bug in the algorithm "
            "(e.g. a contradictory constraint configuration that "
            "_validate_target_row_compliant should already have rejected "
            "at the target), not a legitimate runtime condition."
        )

    relaxed = _solve(
        drifted,
        indep_idx,
        indep_tradable,
        group_legs,
        group_tradable,
        group_l1_norm,
        group_net,
        maximum_weight=maximum_weight,
        gross_cap=gross_cap,
        maximum_net_exposure=maximum_net_exposure,
        long_only=long_only,
        allow_slack=True,
    )
    if relaxed is None:  # pragma: no cover - defensive, should be unreachable
        raise BacktestError(
            "Drift-compliance slack-relaxation LP unexpectedly infeasible "
            "despite a tradability-caused diagnosis -- this indicates a "
            "bug in the algorithm."
        )
    return DriftComplianceResult(corrected=relaxed, pending=True)


def _fixed_positions_alone_violate(
    drifted: np.ndarray,
    indep_idx: list[int],
    indep_tradable: list[bool],
    group_legs: list[list[int]],
    group_tradable: list[bool],
    *,
    maximum_weight: float | None,
    gross_cap: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
) -> bool:
    """Whether fixed positions alone already violate a constraint.

    True when the untradable columns'/groups' own drifted values alone
    already violate a constraint no amount of free-column movement could
    fix -- the signature of a tradability-caused (expected) infeasibility
    rather than a genuine bug.
    """
    if maximum_weight is not None:
        for i, tradable in zip(indep_idx, indep_tradable, strict=True):
            if not tradable and abs(drifted[i]) > maximum_weight + EPSILON:
                return True
        for legs, tradable in zip(group_legs, group_tradable, strict=True):
            if not tradable and any(
                abs(drifted[i]) > maximum_weight + EPSILON for i in legs
            ):
                return True
    if long_only:
        for i, tradable in zip(indep_idx, indep_tradable, strict=True):
            if not tradable and drifted[i] < -EPSILON:
                return True
        for legs, tradable in zip(group_legs, group_tradable, strict=True):
            if not tradable and any(drifted[i] < -EPSILON for i in legs):
                return True
    fixed_gross = sum(
        abs(drifted[i])
        for i, tradable in zip(indep_idx, indep_tradable, strict=True)
        if not tradable
    ) + sum(
        sum(abs(drifted[i]) for i in legs)
        for legs, tradable in zip(group_legs, group_tradable, strict=True)
        if not tradable
    )
    if gross_cap is not None and fixed_gross > gross_cap + EPSILON:
        return True
    fixed_net = sum(
        drifted[i]
        for i, tradable in zip(indep_idx, indep_tradable, strict=True)
        if not tradable
    ) + sum(
        sum(drifted[i] for i in legs)
        for legs, tradable in zip(group_legs, group_tradable, strict=True)
        if not tradable
    )
    return (
        maximum_net_exposure is not None
        and abs(fixed_net) > maximum_net_exposure + EPSILON
    )


def _solve(
    drifted: np.ndarray,
    indep_idx: list[int],
    indep_tradable: list[bool],
    group_legs: list[list[int]],
    group_tradable: list[bool],
    group_l1_norm: list[float],
    group_net: list[float],
    *,
    maximum_weight: float | None,
    gross_cap: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
    allow_slack: bool,
) -> np.ndarray | None:
    """Build and solve one LP (strict or slack-relaxed); ``None`` if infeasible.

    Variable layout: ``w_i`` (independent columns), ``k_g`` (groups),
    ``u_i``/``v_g`` (``|w_i - drifted_i|``/``|k_g - 1|`` L1-deviation
    auxiliaries, ALWAYS present -- see below), ``p_i``/``q_g``
    (``|w_i|``/``|k_g|`` auxiliaries, only wired into a constraint row
    when a gross-exposure cap is configured), and, in slack mode only,
    four shared slack variables (``s_mw``, ``s_gross``, ``s_net``,
    ``s_long``) added to each cap's own right-hand side -- ``s_long``
    relaxes the ``long_only`` inequality (below), the same way the other
    three relax their own numeric cap.

    Strict mode (``allow_slack=False``) minimizes the L1 deviation
    directly in one solve -- every constraint is satisfied exactly, no
    slack variables exist at all.

    Slack mode (``allow_slack=True``) is a genuine two-stage LEXICOGRAPHIC
    solve, not a single relaxed objective: stage 1 minimizes ONLY
    ``s_mw + s_gross + s_net + s_long`` (the unavoidable violation, exactly
    what the tradability-caused-infeasibility diagnosis already proved is
    nonzero); stage 2 then FIXES those slacks at their stage-1-optimal
    values (via tight bounds) and re-solves for the MINIMAL L1 deviation
    among every point achieving that same minimal violation. A single-
    stage "minimize slack only" solve would leave every OTHER free
    column's own objective coefficient at zero, so the solver is free to
    move an already-compliant, uninvolved column to an arbitrary value
    (e.g. liquidating it to 0) with no penalty for doing so, since nothing
    in that objective discourages it -- stage 2 is what rules that out.

    Returns ``None`` ONLY when the solve is provably infeasible (HiGHS
    status ``_LINPROG_INFEASIBLE``) -- the caller's own tradability
    diagnosis treats this as the ordinary, expected outcome. Any OTHER
    non-optimal status (an iteration limit, an unbounded problem, or a
    numerical-difficulties report) is a genuine solver failure, never
    folded into the same "infeasible, diagnose via tradability" bucket --
    it raises `BacktestError` immediately with the solver's own status and
    message, since silently treating it as an ordinary infeasibility could
    misreport a real solver hiccup as a tradability-caused breach (or vice
    versa), and a caller has no way to tell the two apart from `None` alone.
    """
    n_i = len(indep_idx)
    n_g = len(group_legs)
    w_at = list(range(n_i))
    k_at = list(range(n_i, n_i + n_g))
    u_at = list(range(n_i + n_g, 2 * n_i + n_g))
    v_at = list(range(2 * n_i + n_g, 2 * n_i + 2 * n_g))
    p_at = list(range(2 * n_i + 2 * n_g, 3 * n_i + 2 * n_g))
    q_at = list(range(3 * n_i + 2 * n_g, 3 * n_i + 3 * n_g))
    n_vars = 3 * (n_i + n_g) + (4 if allow_slack else 0)
    # Only ever read/written inside an `if allow_slack:` guard below, so
    # these remain valid indices even though `n_vars` excludes them when
    # `allow_slack` is False (kept as plain ints, not `int | None`, so
    # every use site stays a simple, unconditional index expression).
    s_mw = 3 * (n_i + n_g)
    s_gross = s_mw + 1
    s_net = s_mw + 2
    s_long = s_mw + 3

    c_deviation = np.zeros(n_vars)
    for pos in u_at:
        c_deviation[pos] = 1.0
    for pos, norm in zip(v_at, group_l1_norm, strict=True):
        c_deviation[pos] = norm
    # Allocated unconditionally (only ever READ inside `if allow_slack:`
    # below, mirroring s_mw/s_gross/s_net's own "harmless when unused"
    # convention above) so every use site stays a simple, unconditional
    # expression rather than requiring a definite-assignment analysis that
    # spans two separate `if allow_slack`/`if not allow_slack` statements.
    c_slack = np.zeros(n_vars)
    if allow_slack:
        c_slack[s_mw] = 1.0
        c_slack[s_gross] = 1.0
        c_slack[s_net] = 1.0
        c_slack[s_long] = 1.0

    bounds: list[tuple[float, float]] = [(0.0, 0.0)] * n_vars
    for k, (i, tradable) in enumerate(zip(indep_idx, indep_tradable, strict=True)):
        if not tradable:
            bounds[w_at[k]] = (float(drifted[i]), float(drifted[i]))
        else:
            # Sign/support-preserving: a currently-LONG column may shrink
            # toward 0 or grow further long, a currently-SHORT column may
            # shrink toward 0 or grow further short, but neither may CROSS
            # zero, and a column already AT zero stays fixed there -- the
            # LP is never allowed to invent a brand-new position (long or
            # short) on an asset the drifted book doesn't already hold.
            # See restore_drift_compliance's own docstring: the closed-
            # long/tradable-short counterexample this module exists to
            # solve correctly only ever needs an ALREADY-nonzero column
            # free to move FURTHER in its own direction, never a zero
            # column becoming nonzero, so this loses no real solution.
            drifted_i = float(drifted[i])
            if drifted_i > EPSILON:
                lo, hi = 0.0, np.inf
            elif drifted_i < -EPSILON:
                lo, hi = -np.inf, 0.0
            else:
                lo = hi = 0.0
            if long_only:
                lo = max(lo, 0.0)
            bounds[w_at[k]] = (lo, hi)
    for k, tradable in enumerate(group_tradable):
        bounds[k_at[k]] = (1.0, 1.0) if not tradable else (0.0, np.inf)
    for pos in u_at:
        bounds[pos] = (0.0, np.inf)
    for pos in v_at:
        bounds[pos] = (0.0, np.inf)
    if gross_cap is not None:
        for pos in p_at:
            bounds[pos] = (0.0, np.inf)
        for pos in q_at:
            bounds[pos] = (0.0, np.inf)
    if allow_slack:
        bounds[s_mw] = (0.0, np.inf)
        bounds[s_gross] = (0.0, np.inf)
        bounds[s_net] = (0.0, np.inf)
        bounds[s_long] = (0.0, np.inf)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    def _row() -> np.ndarray:
        return np.zeros(n_vars)

    # Deviation: u_i >= |w_i - drifted_i|; v_g >= |k_g - 1|. Built
    # UNCONDITIONALLY (not just in strict mode) so slack mode's stage 2
    # (below) can minimize this same L1 deviation once the unavoidable
    # violation has been pinned at its stage-1-optimal value -- otherwise
    # a free, already-compliant column has no cost keeping it near its own
    # drifted value and the solver may move it anywhere.
    for k, i in enumerate(indep_idx):
        row = _row()
        row[w_at[k]] = 1.0
        row[u_at[k]] = -1.0
        a_ub.append(row)
        b_ub.append(float(drifted[i]))
        row = _row()
        row[w_at[k]] = -1.0
        row[u_at[k]] = -1.0
        a_ub.append(row)
        b_ub.append(-float(drifted[i]))
    for k in range(n_g):
        row = _row()
        row[k_at[k]] = 1.0
        row[v_at[k]] = -1.0
        a_ub.append(row)
        b_ub.append(1.0)
        row = _row()
        row[k_at[k]] = -1.0
        row[v_at[k]] = -1.0
        a_ub.append(row)
        b_ub.append(-1.0)

    if long_only:
        # `long_only` is otherwise only baked into a TRADABLE independent
        # column's own lower bound (`lo = max(lo, 0.0)` above) -- silently
        # a no-op for an UNTRADABLE (fixed) column or group, whose bound
        # is pinned at its drifted value regardless of sign. Without an
        # explicit inequality here, a fixed column/group already negative
        # under long_only (which "shouldn't happen" upstream, per this
        # function's own sign-preservation docstring, but is not actually
        # enforced anywhere for a FIXED value) would let the strict LP
        # trivially "succeed" over a value that still violates long_only --
        # exactly the kind of formulation bug this module must never
        # produce. Redundant (never binding) for every TRADABLE column,
        # whose own bounds already enforce this; only a FIXED column/group
        # can make it bind, correctly turning that into LP infeasibility
        # so the normal tradability diagnosis + slack-relaxation path
        # handles it like any other fixed-value violation (see
        # `_fixed_positions_alone_violate`'s own long_only check).
        for k in range(n_i):
            row = _row()
            row[w_at[k]] = -1.0
            if allow_slack:
                row[s_long] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
        for k, legs in enumerate(group_legs):
            for i in legs:
                d = float(drifted[i])
                row = _row()
                row[k_at[k]] = -d
                if allow_slack:
                    row[s_long] = -1.0
                a_ub.append(row)
                b_ub.append(0.0)

    if maximum_weight is not None:
        for k in range(n_i):
            row = _row()
            row[w_at[k]] = 1.0
            if allow_slack:
                row[s_mw] = -1.0
            a_ub.append(row)
            b_ub.append(maximum_weight)
            row = _row()
            row[w_at[k]] = -1.0
            if allow_slack:
                row[s_mw] = -1.0
            a_ub.append(row)
            b_ub.append(maximum_weight)
        for k, legs in enumerate(group_legs):
            for i in legs:
                d = float(drifted[i])
                row = _row()
                row[k_at[k]] = d
                if allow_slack:
                    row[s_mw] = -1.0
                a_ub.append(row)
                b_ub.append(maximum_weight)
                row = _row()
                row[k_at[k]] = -d
                if allow_slack:
                    row[s_mw] = -1.0
                a_ub.append(row)
                b_ub.append(maximum_weight)

    if gross_cap is not None:
        # p_i >= |w_i|; q_g >= |k_g|.
        for k in range(n_i):
            row = _row()
            row[w_at[k]] = 1.0
            row[p_at[k]] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
            row = _row()
            row[w_at[k]] = -1.0
            row[p_at[k]] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
        for k in range(n_g):
            row = _row()
            row[k_at[k]] = 1.0
            row[q_at[k]] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
            row = _row()
            row[k_at[k]] = -1.0
            row[q_at[k]] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
        row = _row()
        for pos in p_at:
            row[pos] = 1.0
        for pos, norm in zip(q_at, group_l1_norm, strict=True):
            row[pos] = norm
        if allow_slack:
            row[s_gross] = -1.0
        a_ub.append(row)
        b_ub.append(gross_cap)

    if maximum_net_exposure is not None:
        row = _row()
        for pos in w_at:
            row[pos] = 1.0
        for pos, net in zip(k_at, group_net, strict=True):
            row[pos] = net
        row_pos = row.copy()
        row_neg = -row.copy()
        if allow_slack:
            row_pos[s_net] = -1.0
            row_neg[s_net] = -1.0
        a_ub.append(row_pos)
        b_ub.append(maximum_net_exposure)
        a_ub.append(row_neg)
        b_ub.append(maximum_net_exposure)

    a_ub_arr = np.array(a_ub) if a_ub else None
    b_ub_arr = np.array(b_ub) if b_ub else None

    if not allow_slack:
        result = linprog(
            c_deviation, A_ub=a_ub_arr, b_ub=b_ub_arr, bounds=bounds, method="highs"
        )
        if result.status == _LINPROG_INFEASIBLE:
            return None
        if result.status != _LINPROG_OPTIMAL:
            raise BacktestError(
                "Drift-compliance strict LP failed with a non-infeasible, "
                f"non-optimal HiGHS status ({result.status}: "
                f"{result.message}) -- this is a genuine solver failure, "
                "not a tradability-caused infeasibility, and must be "
                "investigated directly."
            )
        x = result.x
    else:
        # Stage 1: the unavoidable violation alone.
        stage1 = linprog(
            c_slack, A_ub=a_ub_arr, b_ub=b_ub_arr, bounds=bounds, method="highs"
        )
        if stage1.status == _LINPROG_INFEASIBLE:
            return None
        if stage1.status != _LINPROG_OPTIMAL:
            raise BacktestError(
                "Drift-compliance slack-relaxation stage 1 failed with a "
                f"non-infeasible, non-optimal HiGHS status ({stage1.status}: "
                f"{stage1.message}) -- this is a genuine solver failure, "
                "not a tradability-caused infeasibility, and must be "
                "investigated directly."
            )
        # Stage 2: pin that violation at its stage-1-optimal value, then
        # minimize the L1 deviation among every point achieving it -- this
        # is what keeps an already-compliant, uninvolved free column at
        # (or near) its own drifted value instead of moving it arbitrarily.
        stage2_bounds = list(bounds)
        for pos in (s_mw, s_gross, s_net, s_long):
            pinned = float(stage1.x[pos])
            stage2_bounds[pos] = (pinned, pinned)
        stage2 = linprog(
            c_deviation,
            A_ub=a_ub_arr,
            b_ub=b_ub_arr,
            bounds=stage2_bounds,
            method="highs",
        )
        # Stage 1's own solution is always feasible for stage 2 (same
        # constraints, slacks pinned at the value it itself produced), so
        # a non-optimal status here is always a genuine bug -- never
        # silently substituted with stage 1's own solution, which has no
        # penalty on any OTHER free column and so could be an arbitrary,
        # needlessly destructive correction (see this function's own
        # docstring on why stage 2 exists at all).
        if stage2.status != _LINPROG_OPTIMAL:
            raise BacktestError(
                "Drift-compliance slack-relaxation stage 2 unexpectedly "
                f"failed (HiGHS status {stage2.status}: {stage2.message}) "
                "despite stage 1's own solution being feasible for stage 2 "
                "by construction -- this indicates a bug in the algorithm."
            )
        x = stage2.x

    out = drifted.copy()
    for k, i in enumerate(indep_idx):
        out[i] = x[w_at[k]]
    for k, legs in enumerate(group_legs):
        kg = x[k_at[k]]
        for i in legs:
            out[i] = kg * drifted[i]
    return out
