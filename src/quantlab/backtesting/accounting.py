"""Vectorised portfolio accounting.

Weights are shifted by one period before returns are applied, so a decision
made at t can only earn from t+1. Turnover is the L1 change in executed
weights. Costs are fractions of equity; volume-based slippage is solved
against prior net equity by :func:`_solve_accounting`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from quantlab.constants import EPSILON
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionCosts, ExecutionModel
from quantlab.execution.orders import executed_weights as compute_executed_weights
from quantlab.execution.orders import validate_execution_frame
from quantlab.execution.orders import weight_changes as compute_weight_changes
from quantlab.logging_config import get_logger
from quantlab.portfolio.drift_compliance import restore_drift_compliance
from quantlab.portfolio.rebalancing import _compliance_violations
from quantlab.risk.exposure import average_gross_exposure, average_net_exposure

logger = get_logger(__name__)

#: Maximum iterations for equity-dependent slippage convergence.
_MAX_COST_EQUITY_ITERATIONS = 20

#: Convergence tolerance as a fraction of initial capital.
_COST_EQUITY_CONVERGENCE_TOLERANCE = 1e-9


@dataclass
class AccountingResult:
    """All series produced by the accounting step."""

    executed_weights: pd.DataFrame  # weights actually in force each period
    weight_changes: pd.DataFrame  # per-symbol Δ of the executed book
    asset_returns: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: ExecutionCosts
    turnover: pd.Series
    equity: pd.Series  # net-of-cost equity curve
    gross_equity: pd.Series  # gross (cost-free) equity curve
    # Net-equity estimate used to size volume-dependent slippage. Reuse it in
    # the trade log to keep per-fill and aggregate costs consistent.
    equity_for_costs: pd.Series
    # True at every date from which prior-period equity was <= 0 -- trading
    # stops there (see run_accounting's docstring/logging). Real provenance
    # for the trade log's forced_liquidation adjustment, not a reconstruction:
    # this is the exact same boolean condition run_accounting already uses to
    # decide when to force positions flat.
    ruined: pd.Series
    # True on every (date, symbol) cell whose position was force-flattened
    # by a stop-loss/take-profit breach on the REAL executed position (see
    # `_detect_stop_loss_take_profit`) -- real provenance for the trade
    # log's stop_loss/take_profit adjustments, mirroring how `ruined`
    # already documents `forced_liquidation`. All-``False`` (never ``None``)
    # when neither `stop_loss_pct` nor `take_profit_pct` was configured.
    stop_loss_triggered: pd.DataFrame
    take_profit_triggered: pd.DataFrame
    # Real provenance from `apply_weight_drift` -- see `DriftProvenance`'s
    # own field docs. All-``False`` (never ``None``) when
    # `model_weight_drift` was not enabled for this run.
    drift_compliance_forced: pd.DataFrame
    drift_compliance_pending: pd.DataFrame
    drift_turnover_actively_limited: pd.DataFrame
    drift_turnover_touched: pd.DataFrame


def portfolio_metrics_from_accounting(
    accounting: AccountingResult, periods_per_year: int
) -> dict[str, float]:
    """Exposure and turnover metrics shared by every accounting consumer.

    Shared by :class:`~quantlab.backtesting.engine.BacktestEngine` and
    :class:`~quantlab.validation.walk_forward.WalkForwardValidator` so a
    single-backtest ``BacktestResult`` and a stitched walk-forward
    out-of-sample ``BacktestResult`` report these the same way.
    """
    turnover = accounting.turnover
    return {
        "annual_turnover": float(turnover.mean() * periods_per_year)
        if len(turnover)
        else 0.0,
        "average_gross_exposure": average_gross_exposure(accounting.executed_weights),
        "average_net_exposure": average_net_exposure(accounting.executed_weights),
    }


def compute_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns of the adjusted-close matrix."""
    return prices.pct_change(fill_method=None)


def _floor_at_total_loss(returns: pd.Series, label: str) -> pd.Series:
    """Clip per-period losses at -100% so compounded equity stays non-negative.

    This models liquidation at zero equity: without fresh capital, subsequent
    portfolio equity remains zero.
    """
    if (returns < -1.0).any():
        logger.warning(
            "%s return below -100%% in at least one period (min %.2f) — "
            "flooring at -100%% (total loss). Check leverage and cost "
            "configuration.",
            label,
            float(returns.min()),
        )
    return returns.clip(lower=-1.0)


def _run_accounting_steps(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    execution_model: ExecutionModel,
    initial_capital: float,
    *,
    force_flat: pd.Series | None = None,
    cost_equity: pd.Series | None = None,
    weight_changes_override: pd.DataFrame | None = None,
) -> AccountingResult:
    """Compute turnover, costs, returns and equity for one executed book.

    Args:
        executed: Weights actually in force each period (post look-ahead
            shift), ``dates × symbols``.
        asset_returns: Per-asset simple returns aligned to ``executed``.
        execution_model: Cost model.
        initial_capital: Starting equity.
        force_flat: Dates on which turnover is forced to zero after
            bankruptcy, preventing a closing trade with no remaining capital.
        cost_equity: Previous net-equity estimate used to size
            volume-dependent slippage. The first pass uses gross equity.
        weight_changes_override: When given, used as turnover/cost input
            INSTEAD OF ``executed``'s own row-to-row diff -- required when
            ``executed`` is not a plain step function (weight drift is
            active), since consecutive rows then genuinely differ from
            organic price movement alone, never a real trade; a naive diff
            would charge phantom turnover/costs for every drifting row.
            ``None`` (the default) computes turnover from ``executed``'s
            own plain row-to-row diff.
    """
    # Turnover is the L1 change in the executed book -- from the real-
    # trade-only override when given (see the docstring above), else the
    # plain diff (correct on its own for a step-function `executed`).
    weight_changes = (
        compute_weight_changes(executed)
        if weight_changes_override is None
        else weight_changes_override
    )
    if force_flat is not None:
        weight_changes = weight_changes.copy()
        weight_changes.loc[force_flat, :] = 0.0
    turnover = weight_changes.abs().sum(axis=1)

    # A missing return is harmless only when no material position is held.
    missing_held_returns = asset_returns.isna() & (executed.abs() > EPSILON)
    if missing_held_returns.to_numpy().any():
        first_row, first_column = np.argwhere(missing_held_returns.to_numpy())[0]
        first_date = missing_held_returns.index[int(first_row)]
        first_symbol = missing_held_returns.columns[int(first_column)]
        raise BacktestError(
            "Asset return is missing while the portfolio holds a non-zero "
            f"position: {first_symbol!r} on {first_date!r}."
        )

    # Missing returns on unheld assets do not
    # contribute; a row with no valid contribution is treated as zero.
    contributions = executed * asset_returns
    gross_returns = contributions.sum(axis=1, min_count=1).fillna(0.0)
    gross_returns = _floor_at_total_loss(gross_returns, "gross")
    gross_equity = initial_capital * (1.0 + gross_returns).cumprod()

    # Size costs from prior-period equity. The first fixed-point
    # pass uses gross equity; later passes feed back the latest net estimate.
    equity_for_costs = gross_equity if cost_equity is None else cost_equity
    prior_equity = equity_for_costs.shift(1).fillna(initial_capital)
    costs = execution_model.compute(weight_changes, equity=prior_equity)

    # Align costs to returns. An absent date means no recorded cost;
    # a non-finite value on an existing date is invalid and must not become zero.
    reindexed_costs = costs.total.reindex(gross_returns.index)
    already_present = reindexed_costs.index.isin(costs.total.index)
    if not np.isfinite(reindexed_costs[already_present]).all():
        bad_dates = reindexed_costs.index[already_present][
            ~np.isfinite(reindexed_costs[already_present])
        ]
        raise BacktestError(
            f"Execution costs are not finite on {list(bad_dates)[:5]}"
            f"{'…' if len(bad_dates) > 5 else ''} — check for a "
            "misconfigured commission/spread/slippage rate."
        )
    net_returns = gross_returns - reindexed_costs.fillna(0.0)
    net_returns = _floor_at_total_loss(net_returns, "net")

    # Build the net equity curve from the configured initial capital.
    equity = initial_capital * (1.0 + net_returns).cumprod()

    return AccountingResult(
        executed_weights=executed,
        weight_changes=weight_changes,
        asset_returns=asset_returns,
        gross_returns=gross_returns,
        net_returns=net_returns,
        costs=costs,
        turnover=turnover,
        equity=equity,
        gross_equity=gross_equity,
        equity_for_costs=equity_for_costs,
        # Overwritten by run_accounting (the only real caller of this
        # internal helper) with the actual ruin/stop-loss/take-profit
        # provenance -- these placeholders are never observed externally.
        ruined=pd.Series(False, index=gross_returns.index),
        stop_loss_triggered=pd.DataFrame(
            False, index=executed.index, columns=executed.columns
        ),
        take_profit_triggered=pd.DataFrame(
            False, index=executed.index, columns=executed.columns
        ),
        drift_compliance_forced=pd.DataFrame(
            False, index=executed.index, columns=executed.columns
        ),
        drift_compliance_pending=pd.DataFrame(
            False, index=executed.index, columns=executed.columns
        ),
        drift_turnover_actively_limited=pd.DataFrame(
            False, index=executed.index, columns=executed.columns
        ),
        drift_turnover_touched=pd.DataFrame(
            False, index=executed.index, columns=executed.columns
        ),
    )


def _solve_accounting(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    execution_model: ExecutionModel,
    initial_capital: float,
    *,
    force_flat: pd.Series | None = None,
    weight_changes_override: pd.DataFrame | None = None,
) -> AccountingResult:
    """Solve equity-dependent costs to a self-consistent equity curve.

    Volume impact at t is sized from equity at t-1, which depends on earlier
    costs. Repeated vectorised passes feed the latest net-equity curve back
    into the cost model until the maximum difference reaches the tolerance.
    """
    result = _run_accounting_steps(
        executed,
        asset_returns,
        execution_model,
        initial_capital,
        force_flat=force_flat,
        weight_changes_override=weight_changes_override,
    )
    if not len(result.equity):
        return result
    tolerance = _COST_EQUITY_CONVERGENCE_TOLERANCE * initial_capital
    residual = float("inf")
    for _ in range(_MAX_COST_EQUITY_ITERATIONS - 1):
        next_result = _run_accounting_steps(
            executed,
            asset_returns,
            execution_model,
            initial_capital,
            force_flat=force_flat,
            cost_equity=result.equity,
            weight_changes_override=weight_changes_override,
        )
        residual = float((next_result.equity - result.equity).abs().max())
        result = next_result
        if residual <= tolerance:
            break
    else:
        raise BacktestError(
            "Execution-cost/equity fixed point did not converge within "
            f"{_MAX_COST_EQUITY_ITERATIONS} iterations (residual "
            f"{residual:.6f}, tolerance {tolerance:.6f}). Check the "
            "slippage, leverage and liquidity assumptions."
        )
    return result


def _resolve_position_groups(
    columns: pd.Index, position_groups: Sequence[tuple[str, ...]] | None
) -> list[tuple[str, ...]]:
    """Expand ``position_groups`` into a complete partition of ``columns``.

    A symbol not mentioned in any declared group (or ``position_groups``
    being ``None`` entirely) means "its own independent group" -- a
    caller only needs to declare GENUINE multi-symbol groups (e.g.
    pairs_trading's two legs via ``BaseStrategy.position_groups()``),
    never every symbol individually.

    Raises:
        BacktestError: If any declared group is empty, repeats a symbol
            within itself, references a symbol absent from ``columns``, or
            overlaps a symbol already claimed by another declared group --
            each would otherwise silently corrupt the stop-loss/take-profit
            and weight-drift-compliance walks (double-processing a symbol
            under two different entry timings, or applying the group-return
            formula to a nonexistent column).
    """
    grouped: set[str] = set()
    groups: list[tuple[str, ...]] = []
    if position_groups is not None:
        available = set(columns)
        for group in position_groups:
            members = tuple(group)
            if not members:
                raise BacktestError(
                    "position_groups entries must be non-empty; got an empty group."
                )
            if len(set(members)) != len(members):
                raise BacktestError(
                    f"position_groups entry {members!r} repeats a symbol within itself."
                )
            unknown = [symbol for symbol in members if symbol not in available]
            if unknown:
                raise BacktestError(
                    f"position_groups entry {members!r} references symbol(s) "
                    f"{unknown} not present among the executed weights columns."
                )
            overlap = grouped & set(members)
            if overlap:
                raise BacktestError(
                    f"position_groups entry {members!r} overlaps symbol(s) "
                    f"{sorted(overlap)} already claimed by another declared "
                    "group -- every symbol may belong to at most one group."
                )
            groups.append(members)
            grouped.update(members)
    for column in columns:
        if column not in grouped:
            groups.append((column,))
    return groups


def _walk_group_stop_loss_take_profit(
    gross_exposure: np.ndarray,
    group_return: np.ndarray,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    reversed_without_flat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk one position-group's own return path since its last entry.

    ``group_return[t]`` is this group's realized return for period t,
    per unit of gross exposure it represented that period (see
    :func:`_detect_stop_loss_take_profit`'s docstring for the exact
    formula and why it is correct under a static or dynamic hedge ratio,
    rebalancing, long/short and partial entries/exits). "Entry" is the
    first date after the group was fully flat (``gross_exposure <=
    EPSILON``) that it becomes non-flat again, OR a date any leg's sign
    flips directly (long to short or vice versa) without an intermediate
    flat row (``reversed_without_flat[t]``) -- a same-bar reversal is
    economically a close-then-reopen, so the new direction must start its
    own fresh cumulative-return episode rather than silently inheriting
    the old (opposite-direction) position's running total, which would
    misattribute a break/breach to a position that was never actually
    held.

    A breach detected using periods THROUGH t (inclusive) force-flattens
    period t+1 onward -- never period t itself, since t's own return has
    already been realized by the time this decision could be made (no
    look-ahead). Once force-flattened, the group stays flat until its
    next flat-to-non-flat transition or same-bar reversal (no immediate
    re-entry at a rebased price -- the same convention `mean_reversion`'s
    own indicator-based stop uses, for consistency).

    Returns ``(force_flat, stop_loss_triggered, take_profit_triggered)``,
    each a boolean array aligned to ``gross_exposure``. The trigger
    arrays mark the FIRST force-flattened date for their respective
    cause (an "exit" event), not the date the breach was internally
    detected.
    """
    n = len(gross_exposure)
    force_flat = np.zeros(n, dtype=bool)
    stop_loss_triggered = np.zeros(n, dtype=bool)
    take_profit_triggered = np.zeros(n, dtype=bool)
    reversed_flags = (
        np.zeros(n, dtype=bool)
        if reversed_without_flat is None
        else reversed_without_flat
    )
    was_flat = True
    cumulative = 1.0
    stopped = False
    stopped_reason: str | None = None
    trigger_marked = False
    for t in range(n):
        if gross_exposure[t] <= EPSILON:
            was_flat = True
            stopped = False
            stopped_reason = None
            trigger_marked = False
            cumulative = 1.0
            continue
        if was_flat or reversed_flags[t]:
            cumulative = 1.0
            stopped = False
            stopped_reason = None
            trigger_marked = False
            was_flat = False
        if stopped:
            force_flat[t] = True
            if not trigger_marked:
                if stopped_reason == "stop_loss":
                    stop_loss_triggered[t] = True
                else:
                    take_profit_triggered[t] = True
                trigger_marked = True
            continue
        r = group_return[t]
        if np.isfinite(r):
            cumulative *= 1.0 + r
        total_return = cumulative - 1.0
        if stop_loss_pct is not None and total_return <= -stop_loss_pct:
            stopped = True
            stopped_reason = "stop_loss"
        elif take_profit_pct is not None and total_return >= take_profit_pct:
            stopped = True
            stopped_reason = "take_profit"
    return force_flat, stop_loss_triggered, take_profit_triggered


def _detect_stop_loss_take_profit(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    position_groups: Sequence[tuple[str, ...]] | None,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    weight_changes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Detect and gate stop-loss/take-profit breaches on the REAL executed position.

    Operates on ``executed`` (the actual post-shift, post-constraint,
    post-rebalance/turnover-cap position a real portfolio would hold),
    never on a strategy's raw signal -- a signal is not necessarily a
    realized position (the allocator, portfolio constraints, rebalancing
    schedule and turnover cap all sit between them), so gating on the
    signal directly could force-flatten a position that was never
    actually opened, or miss one that was.

    For a position group ``G`` (one symbol, or e.g. pairs_trading's two
    legs via ``position_groups``), at each date::

        gross_exposure[t] = sum(|executed[s][t]| for s in G)
        group_return[t]   = sum(executed[s][t] * asset_returns[s][t] for s in G)
                             / gross_exposure[t]

    ``group_return`` is the group's return per unit of ITS OWN gross
    exposure at that date -- not a dollar contribution to total portfolio
    equity (which would depend on how much capital was allocated to it,
    irrelevant to "has this position itself moved against me by X%").
    This normalization by the ACTUAL exposure held each period (not the
    exposure at entry) is what makes the formula correct regardless of a
    static or dynamic hedge ratio, weight changes, rebalancing, long/
    short direction, or partial entries/exits: every period contributes
    its realized return weighted by whatever was really held that
    period, using EXACTLY the same ``executed``/``asset_returns`` this
    module already computes internally (never a second, potentially
    diverging calculation). For a single-symbol group this reduces
    exactly to ``sign(executed[t]) * asset_returns[t]`` -- the standard
    definition of a price-based stop-loss/take-profit.

    Thresholds are evaluated on GROSS (pre-cost) return: QuantLab's
    execution cost model is portfolio-level only (no per-symbol/per-group
    cost decomposition exists), so an exact net-of-cost trigger is not
    presently computable. This is a deliberate, disclosed design
    convention -- not "the" universal definition of a stop-loss/take-
    profit -- documented on ``stop_loss_pct``/``take_profit_pct``
    themselves; a net-of-cost variant could be added separately if
    per-position cost attribution is ever built.

    Returns ``(gated_executed, stop_loss_triggered, take_profit_triggered,
    gated_weight_changes)`` -- the trigger frames are booleans broadcast
    across every column of the breaching group (matching the trade log's
    row-per-symbol grain), all ``False`` when neither threshold is
    configured. ``gated_weight_changes`` mirrors ``weight_changes`` (the
    caller's own real-trade-only turnover series, e.g. from
    :func:`apply_weight_drift`) with the forced flatten's own turnover
    patched in correctly -- ``None`` in, ``None`` out (the caller then
    falls back to plain re-diffing ``gated_executed``, which is already
    exactly correct when every row-to-row change genuinely is a trade,
    i.e. weight drift is not active).
    """
    if stop_loss_pct is None and take_profit_pct is None:
        empty = pd.DataFrame(False, index=executed.index, columns=executed.columns)
        return executed, empty, empty.copy(), weight_changes

    groups = _resolve_position_groups(executed.columns, position_groups)
    gated = executed.copy()
    gated_weight_changes = None if weight_changes is None else weight_changes.copy()
    # What was genuinely HELD immediately before whatever (if anything) this
    # row itself already traded -- derived generically from the pre-gating
    # (executed, weight_changes) pair, correct whether that row was a pure
    # drift row (weight_changes == 0, so this is just `executed` itself) or
    # a real trade/anchor row (subtracting that row's own delta recovers
    # the pre-trade state) -- never a second, potentially diverging
    # recomputation of the drift trajectory.
    before_state = None if weight_changes is None else executed - weight_changes
    stop_loss_triggered = pd.DataFrame(
        False, index=executed.index, columns=executed.columns
    )
    take_profit_triggered = stop_loss_triggered.copy()
    for group in groups:
        columns = list(group)
        group_executed = executed[columns]
        gross_exposure = group_executed.abs().sum(axis=1).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            group_return = (
                (group_executed * asset_returns[columns]).sum(axis=1) / gross_exposure
            ).to_numpy(dtype=float)
        # A leg that flips sign directly (long to short or back) without an
        # intermediate flat row is economically a close-then-reopen -- the
        # new direction must start a fresh episode, never inherit the old
        # (opposite) position's running cumulative return.
        signs = np.sign(group_executed.to_numpy(dtype=float))
        previous_signs = np.vstack([np.zeros((1, signs.shape[1])), signs[:-1]])
        reversed_without_flat = np.any(
            (previous_signs != 0.0) & (signs != 0.0) & (previous_signs != signs),
            axis=1,
        )
        force_flat, sl, tp = _walk_group_stop_loss_take_profit(
            gross_exposure,
            group_return,
            stop_loss_pct,
            take_profit_pct,
            reversed_without_flat,
        )
        if force_flat.any():
            gated.loc[force_flat, columns] = 0.0
            stop_loss_triggered.loc[sl, columns] = True
            take_profit_triggered.loc[tp, columns] = True
            if gated_weight_changes is not None and before_state is not None:
                force_flat_series = pd.Series(force_flat, index=executed.index)
                transition = force_flat_series & ~force_flat_series.shift(
                    1, fill_value=False
                )
                # No organic drift-turnover is credited while flat, only the
                # real closing trade on the first forced-flat row.
                gated_weight_changes.loc[force_flat_series, columns] = 0.0
                gated_weight_changes.loc[transition, columns] = (
                    0.0 - before_state.loc[transition, columns]
                )
    return gated, stop_loss_triggered, take_profit_triggered, gated_weight_changes


@dataclass(frozen=True)
class DriftProvenance:
    """Real, cell-level provenance from :func:`apply_weight_drift`.

    ``drift_compliance_forced`` is True on the row a queued compliance
    correction actually LANDED -- a fresh anchor, exactly like any other
    real trade (turnover/costs/the trade log already pick this up
    generically, with no special-casing, since it is just another row-to-
    row change in the frame ``run_accounting`` is given).
    ``drift_compliance_pending`` is True on every row a breach is known
    and not yet (fully) resolved -- including the very row it was first
    detected on, and every later row it is retried while blocked by
    tradability. Mutually exclusive with ``drift_compliance_forced`` for
    every (date, symbol) cell -- enforced in ``__post_init__`` below, since
    the two masks are written by two independent, same-row `_try_restore`
    calls in :func:`apply_weight_drift` with no shared memory of each
    other's own verdict.
    ``drift_turnover_actively_limited``/``drift_turnover_touched`` are the
    exact analogue of :class:`~quantlab.portfolio.rebalancing.
    TurnoverProvenance`'s own identically-named fields, but for ordinary
    rebalance debt capped HERE (the only place ``maximum_turnover`` is
    enforced once weight drift is active -- see ``engine.py``'s own
    decision-level call).
    """

    drift_compliance_forced: pd.DataFrame
    drift_compliance_pending: pd.DataFrame
    drift_turnover_actively_limited: pd.DataFrame
    drift_turnover_touched: pd.DataFrame

    def __post_init__(self) -> None:
        """Enforce that forced/pending are never both True for the same cell."""
        overlap = self.drift_compliance_forced & self.drift_compliance_pending
        if overlap.to_numpy().any():
            raise BacktestError(
                "drift_compliance_forced and drift_compliance_pending must "
                "be mutually exclusive per cell -- this indicates a bug in "
                "apply_weight_drift's same-row compliance handling."
            )


def _validate_drift_and_risk_options(
    *,
    long_only: bool,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    model_weight_drift: bool | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    maximum_turnover: float | None = None,
) -> None:
    """Validate the same invariants ``PortfolioConfig`` already enforces.

    A strategy-driven caller (``engine.py``) only ever gets here with
    already-validated values (``PortfolioConfig``'s own field constraints,
    ``quantlab.strategies.base.validate_risk_control_parameters``) -- but
    ``run_accounting``/``apply_weight_drift`` are BOTH public, directly
    callable functions (tests, scripts, a future programmatic caller)
    that bypass both, so this module must not silently accept a truthy
    non-bool ``model_weight_drift``, a negative ``stop_loss_pct``, or a
    negative ``maximum_weight`` reaching the drift-compliance LP as a
    genuinely infeasible constraint and raising a confusing "bug in the
    algorithm" error instead of a clear, immediate input-validation one.
    """
    flags: list[tuple[str, object]] = [("long_only", long_only)]
    if model_weight_drift is not None:
        flags.append(("model_weight_drift", model_weight_drift))
    for flag_name, flag_value in flags:
        if not isinstance(flag_value, (bool, np.bool_)):
            raise BacktestError(f"{flag_name} must be a boolean, got {flag_value!r}.")
    for pct_name, pct_value in (
        ("stop_loss_pct", stop_loss_pct),
        ("take_profit_pct", take_profit_pct),
    ):
        if pct_value is None:
            continue
        if isinstance(pct_value, (bool, np.bool_)) or not isinstance(pct_value, Real):
            raise BacktestError(
                f"{pct_name} must be a finite number, got {pct_value!r}."
            )
        if not np.isfinite(float(pct_value)) or float(pct_value) <= 0.0:
            raise BacktestError(
                f"{pct_name} must be strictly positive, got {pct_value!r}."
            )
    for cap_name, cap_value, strict, upper in (
        ("maximum_weight", maximum_weight, True, 1.0),
        ("maximum_gross_exposure", maximum_gross_exposure, True, None),
        ("maximum_net_exposure", maximum_net_exposure, False, None),
        ("maximum_turnover", maximum_turnover, True, None),
    ):
        if cap_value is None:
            continue
        if isinstance(cap_value, (bool, np.bool_)) or not isinstance(cap_value, Real):
            raise BacktestError(
                f"{cap_name} must be a finite number, got {cap_value!r}."
            )
        value = float(cap_value)
        out_of_range = not np.isfinite(value) or (
            value <= 0.0 if strict else value < 0.0
        )
        if out_of_range:
            bound = "> 0" if strict else ">= 0"
            raise BacktestError(
                f"{cap_name} must be a finite number {bound}, got {cap_value!r}."
            )
        if upper is not None and value > upper:
            raise BacktestError(f"{cap_name} must not exceed {upper}.")


def _validate_tradable_mask(
    tradable: pd.DataFrame, reference: pd.DataFrame, *, reference_name: str
) -> pd.DataFrame:
    """Validate and axis-align a strictly boolean ``tradable`` mask.

    Shared by :func:`run_accounting` and :func:`apply_weight_drift` -- both
    are directly callable public functions, so neither may silently accept
    a mask on different axes than the frame it is meant to gate, or a
    non-boolean column (e.g. the string ``"False"``, which would otherwise
    coerce to truthy on the plain ``.to_numpy(dtype=bool)`` cast every
    caller of this mask ultimately performs).
    """
    if not isinstance(tradable, pd.DataFrame):
        raise BacktestError("tradable must be a pandas DataFrame.")
    if not tradable.index.is_unique:
        raise BacktestError("tradable index must not contain duplicate labels.")
    if set(tradable.index) != set(reference.index) or set(tradable.columns) != set(
        reference.columns
    ):
        raise BacktestError(
            f"tradable must have the same dates and symbols as {reference_name}."
        )
    if tradable.isna().to_numpy().any():
        raise BacktestError("tradable must not contain missing values.")
    non_bool_columns = [
        column for column, dtype in tradable.dtypes.items() if not is_bool_dtype(dtype)
    ]
    if non_bool_columns:
        raise BacktestError(
            f"tradable must contain only boolean values; column(s) "
            f"{non_bool_columns} are not boolean dtype (e.g. a string "
            "'False' would otherwise silently coerce to True)."
        )
    return tradable.reindex(index=reference.index, columns=reference.columns)


def apply_weight_drift(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    tradable: pd.DataFrame | None,
    position_groups: Sequence[tuple[str, ...]] | None,
    *,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
    rebalance_date: pd.DataFrame | None = None,
    maximum_turnover: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DriftProvenance]:
    """Evolve ``executed`` forward by organic price drift between real trades.

    Walks ``executed`` -- the ALREADY shift-respecting-tradability, look-
    ahead-barrier-applied real executed book (see :func:`quantlab.
    execution.orders.executed_weights`), never the pre-shift decision
    timeline :mod:`quantlab.portfolio.rebalancing` produces -- forward
    between genuine trades via a per-column ``dollar[i]`` exposure and a
    single shared relative equity ``E`` (``weight[i] = dollar[i] / E``).
    Full mechanism -- the two kinds of per-column debt (hard-risk-limit
    compliance debt via :func:`~quantlab.portfolio.drift_compliance.
    restore_drift_compliance`'s LP, and turnover-capped ordinary rebalance
    debt), their priority order, anchor detection, same-row combination
    checks, and the bankruptcy guard -- is documented in
    docs/backtesting.md#weight-drift, not repeated here.

    Returns ``(pre_period_weights, trade_changes, provenance)``.
    ``pre_period_weights`` is the weight HELD GOING INTO each row, BEFORE
    that row's own return is applied (consistent with ``executed_weights
    = held.shift(1)`` elsewhere in this module -- returning the
    post-return value would double-count it). ``trade_changes`` is the
    real per-row trade delta -- zero on a pure-drift row, the actual size
    on a row that lands a rebalance or a compliance correction -- required
    so a naive diff of ``pre_period_weights`` never sees organic drift
    itself as a "trade" (see ``_run_accounting_steps``'s own
    ``weight_changes_override``). Never raises or produces ``inf``/
    ``NaN`` from this recursion itself -- a bankrupt anchor-episode
    (relative ``E <= EPSILON``) is force-flattened and logged instead,
    mirroring ``ruined``'s own handling. (A believed-fully-restored row
    that still violates a constraint DOES raise -- see the compliance
    re-check right before each row is finalized -- since that specific
    case is a genuine bug in the LP, not a legitimate runtime outcome.)

    ``tradable``/``position_groups`` are the same frames ``run_accounting``
    already threads through elsewhere (tradability-aware shifting, stop-
    loss/take-profit position groups) -- ``tradable is None`` treats every
    column as always tradable (single-calendar short-circuit).
    ``rebalance_date`` is a boolean, ``dates x symbols`` :class:`pandas.
    DataFrame` matching ``executed``'s own index and columns exactly,
    already shifted onto the executed timeline exactly like ``executed``
    itself (see ``run_accounting``'s own docstring and ``engine.py``'s
    construction of it) -- a column must never be marked ``True`` on a
    date it is not itself genuinely tradable. ``None`` falls back to
    value-diff-only anchor detection.
    """
    _validate_drift_and_risk_options(
        long_only=long_only,
        maximum_weight=maximum_weight,
        maximum_gross_exposure=maximum_gross_exposure,
        maximum_net_exposure=maximum_net_exposure,
        maximum_turnover=maximum_turnover,
    )
    # A directly-callable public function (see the module docstring's own
    # "run_accounting/apply_weight_drift are BOTH public" note) must not
    # silently accept a malformed `executed`/`asset_returns`/`tradable` --
    # unlike `run_accounting`, which is reached only through its own
    # up-front validation, a caller can invoke this function directly with
    # entirely unvalidated data.
    executed = validate_execution_frame(executed, name="executed")
    if not isinstance(asset_returns, pd.DataFrame):
        raise BacktestError("asset_returns must be a pandas DataFrame.")
    if not asset_returns.index.is_unique:
        raise BacktestError("asset_returns index must not contain duplicate labels.")
    if not asset_returns.columns.is_unique:
        raise BacktestError("asset_returns columns must not contain duplicate labels.")
    missing_dates = executed.index.difference(asset_returns.index)
    missing_symbols = executed.columns.difference(asset_returns.columns)
    if len(missing_dates) or len(missing_symbols):
        raise BacktestError(
            "asset_returns must cover every executed date and symbol "
            f"(missing dates: {list(missing_dates)[:5]}, missing symbols: "
            f"{list(missing_symbols)[:5]})."
        )
    asset_returns = asset_returns.reindex_like(executed)
    try:
        returns_values = asset_returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("asset_returns must contain only numeric values.") from exc
    if np.isinf(returns_values).any():
        raise BacktestError("asset_returns must not contain Infinity.")
    finite_returns = returns_values[np.isfinite(returns_values)]
    if (finite_returns < -1.0).any():
        raise BacktestError(
            "asset_returns must not contain simple returns below -1.0 (-100%)."
        )
    # Row 0 is exempt: it is always this series' own anchor (`previous_row
    # is None`), with no prior row to have earned a return during -- a
    # caller may legitimately start `executed` already non-zero (assuming
    # a pre-existing position) with no return recorded for how it got
    # there, mirroring `run_accounting`'s own `executed = held.shift(1)`
    # convention, which always makes the production pipeline's row 0
    # exactly 0 regardless of `held`. A genuinely MISSING return on an
    # ALREADY-established held position from row 1 onward is still real,
    # unambiguous missing data and must still raise.
    missing_held_returns = np.isnan(returns_values) & (
        np.abs(executed.to_numpy(dtype=float)) > EPSILON
    )
    if missing_held_returns.shape[0] > 0:
        missing_held_returns[0, :] = False
    if missing_held_returns.any():
        bad = np.argwhere(missing_held_returns)[0]
        raise BacktestError(
            "asset_returns is missing a return for a held position: "
            f"{executed.index[bad[0]]!r}/{executed.columns[bad[1]]!r}."
        )
    if tradable is not None:
        tradable = _validate_tradable_mask(
            tradable, executed, reference_name="executed"
        )
    columns = list(executed.columns)
    n_rows, n_cols = executed.shape
    groups = _resolve_position_groups(executed.columns, position_groups)
    if rebalance_date is not None:
        if not isinstance(rebalance_date, pd.DataFrame):
            raise BacktestError("rebalance_date must be a pandas DataFrame.")
        if not rebalance_date.index.is_unique:
            raise BacktestError(
                "rebalance_date index must not contain duplicate labels."
            )
        if set(rebalance_date.index) != set(executed.index) or set(
            rebalance_date.columns
        ) != set(executed.columns):
            raise BacktestError(
                "rebalance_date must have the same dates and symbols as executed."
            )
        if rebalance_date.isna().to_numpy().any():
            raise BacktestError("rebalance_date must not contain missing values.")
        non_bool_columns = [
            column
            for column, dtype in rebalance_date.dtypes.items()
            if not is_bool_dtype(dtype)
        ]
        if non_bool_columns:
            raise BacktestError(
                f"rebalance_date must contain only boolean values; column(s) "
                f"{non_bool_columns} are not boolean dtype (e.g. a string "
                "'False' would otherwise silently coerce to True)."
            )
        rebalance_date_np = rebalance_date.reindex(
            index=executed.index, columns=executed.columns
        ).to_numpy(dtype=bool)
    else:
        rebalance_date_np = None

    executed_np = executed.to_numpy(dtype=float)
    returns_np = returns_values
    if tradable is not None:
        tradable_np = tradable.reindex(
            index=executed.index, columns=executed.columns
        ).to_numpy(dtype=bool)
    else:
        tradable_np = np.ones((n_rows, n_cols), dtype=bool)

    out = np.zeros((n_rows, n_cols))
    trade_changes = np.zeros((n_rows, n_cols))
    drift_compliance_forced = np.zeros((n_rows, n_cols), dtype=bool)
    drift_compliance_pending = np.zeros((n_rows, n_cols), dtype=bool)
    drift_turnover_actively_limited = np.zeros((n_rows, n_cols), dtype=bool)
    drift_turnover_touched = np.zeros((n_rows, n_cols), dtype=bool)

    def _violations(row: np.ndarray) -> list[str]:
        return _compliance_violations(
            row,
            maximum_weight=maximum_weight,
            maximum_gross_exposure=maximum_gross_exposure,
            maximum_net_exposure=maximum_net_exposure,
            long_only=long_only,
        )

    def _try_restore(
        row: np.ndarray, row_tradable: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool] | None:
        """``None`` if already compliant; else (corrected, relevant_mask, pending).

        ``relevant_mask`` is the set of columns the caller must track for
        this correction: every column the LP actually moved, plus --
        ONLY when full compliance was not achievable (``pending=True``)
        -- every currently-untradable column still holding a non-zero
        position. The latter matters because a single untradable column
        that IS the entire breach (nothing free exists to move at all)
        would otherwise report an EMPTY moved-set despite a real,
        unresolved violation -- silently losing both the provenance
        marking and the "wait for this column to reopen" eligibility
        check the pending state depends on.
        """
        if not _violations(row):
            return None
        result = restore_drift_compliance(
            row,
            columns,
            row_tradable,
            groups,
            maximum_weight=maximum_weight,
            maximum_gross_exposure=maximum_gross_exposure,
            maximum_net_exposure=maximum_net_exposure,
            long_only=long_only,
        )
        moved = np.abs(result.corrected - row) > EPSILON
        if result.pending:
            still_held_and_closed = (~row_tradable) & (np.abs(row) > EPSILON)
            moved = moved | still_held_and_closed
        return result.corrected, moved, result.pending

    dollar = np.zeros(n_cols)
    equity = 1.0
    previous_row: np.ndarray | None = None
    # Per-column ordinary rebalance debt: `ordinary_target[i]` is the
    # decided value column `i` is being walked toward, meaningful only
    # where `ordinary_mask[i]` is True. A fresh per-column decision (a
    # value-diff or scheduled-rebalance anchor) overwrites BOTH for that
    # column alone -- it never touches any OTHER column's own outstanding
    # debt.
    ordinary_mask = np.zeros(n_cols, dtype=bool)
    ordinary_target = np.zeros(n_cols)
    # Whether a hard-risk-limit breach is currently outstanding anywhere
    # in the portfolio -- see the docstring's "compliance debt" section.
    compliance_pending = False

    for t in range(n_rows):
        row = executed_np[t]
        row_tradable = tradable_np[t]
        if previous_row is None:
            fresh = np.ones(n_cols, dtype=bool)
        else:
            fresh = np.abs(row - previous_row) > EPSILON
        if rebalance_date_np is not None:
            fresh = fresh | rebalance_date_np[t]
        previous_row = row

        # `dollar/equity` currently hold the state finalized at the END of
        # row (t-1)'s own advance -- the weight HELD GOING INTO row t,
        # before row t's own return. Every subsequent step below measures
        # a real trade against this PRE-transaction value.
        weight_prev = dollar / equity if equity > EPSILON else np.zeros(n_cols)

        # Debt already outstanding BEFORE this row's own fresh decisions are
        # folded in -- used below only to distinguish "this row's move is
        # continuing a catch-up that was already running" from "this is a
        # brand-new decision" for turnover-touched provenance; it plays no
        # role in what actually executes.
        carried_in_mask = ordinary_mask.copy()
        ordinary_target = np.where(fresh, row, ordinary_target)
        ordinary_mask = ordinary_mask | fresh

        landed = weight_prev
        landed_compliance_mask = np.zeros(n_cols, dtype=bool)

        # --- Compliance debt: highest priority, exempt from
        # `maximum_turnover`, re-solved fresh from THIS row's own weights
        # (never a stale stored target -- see the docstring). A hard-limit
        # correction is ORTHOGONAL to ordinary rebalance debt -- it never
        # clears it, even for a column it moves: `ordinary_target` is
        # always itself a validated-compliant value (upstream constraint
        # enforcement, or an earlier compliance-restored point), so once
        # this correction lands, resuming the walk toward that target is
        # always safe, and a still-more-conservative target must not be
        # silently abandoned just because compliance intervened first.
        if compliance_pending:
            restored = _try_restore(landed, row_tradable)
            if restored is None:
                compliance_pending = False
            else:
                corrected, moved, still_pending = restored
                landed = np.where(moved, corrected, landed)
                landed_compliance_mask = moved
                if still_pending:
                    drift_compliance_pending[t, moved] = True
                else:
                    drift_compliance_forced[t, moved] = True
                compliance_pending = still_pending

        # --- Ordinary rebalance debt: turnover-capped, for whatever
        # columns are both owed a decision AND actually tradable this
        # row -- a closed column's own debt simply waits, untouched, for
        # a later row it reopens on, whether it is a fresh anchor or a
        # multi-row catch-up.
        eligible = ordinary_mask & row_tradable & ~landed_compliance_mask
        desired = np.where(ordinary_mask, ordinary_target, landed)
        change = np.where(eligible, desired - landed, 0.0)
        requested = float(np.abs(change).sum())
        if maximum_turnover is None or requested <= maximum_turnover + EPSILON:
            fraction = 1.0
        else:
            fraction = maximum_turnover / requested
        landed = landed + fraction * change
        # Real, cell-level turnover-limiting provenance -- the exact
        # analogue of `quantlab.portfolio.rebalancing.cap_turnover`'s own
        # `turnover_actively_limited`/`turnover_touched` frames, since
        # this is now the ONLY place `maximum_turnover` is actually
        # enforced when weight drift is active (see `engine.py`'s own
        # decision-level call, which passes `maximum_turnover=None` in
        # that case for exactly this reason).
        actively_limited = eligible & (fraction < 1.0 - EPSILON)
        drift_turnover_actively_limited[t] = actively_limited
        drift_turnover_touched[t] = eligible & (actively_limited | carried_in_mask)
        if fraction >= 1.0 - EPSILON:
            ordinary_mask = ordinary_mask & ~eligible

        # --- A genuinely NEW violation can emerge purely from COMBINING
        # this row's just-decided/corrected columns with another column's
        # frozen or still-drifting value -- unlike organic drift (an
        # exogenous price move needing a one-row lag to react to), every
        # input to this combination is already known before finalizing
        # this row's own output, so there is no look-ahead concern in
        # resolving it immediately. Only runs when something was actually
        # decided this row; a pure, undisturbed drift row instead keeps
        # the ordinary one-row-lag detect-then-queue behavior below.
        decided_this_row = bool(landed_compliance_mask.any() or eligible.any())
        if decided_this_row and not compliance_pending:
            restored = _try_restore(landed, row_tradable)
            if restored is not None:
                corrected, moved, still_pending = restored
                landed = np.where(moved, corrected, landed)
                # This is a SECOND, independent `_try_restore` call (Step 1
                # above may have already run its own on this same row) --
                # its own write to `landed` for `moved` cells supersedes
                # whatever Step 1 already recorded for the SAME cell
                # earlier this row, since Step 1's `landed` value for
                # those cells has just been overwritten above. Clear
                # first so `forced`/`pending` can never both be True for
                # the same cell in the same row -- this call's own
                # verdict is authoritative for any cell it touches.
                drift_compliance_forced[t, moved] = False
                drift_compliance_pending[t, moved] = False
                if still_pending:
                    drift_compliance_pending[t, moved] = True
                    compliance_pending = True
                else:
                    drift_compliance_forced[t, moved] = True
        elif not decided_this_row and not compliance_pending:
            # Pure organic drift newly breaching a limit -- queued, never
            # applied to this row's own output (the portfolio genuinely
            # held the breaching value for one row; correcting it
            # retroactively would be look-ahead). Lands starting the next
            # row via the compliance-debt branch above.
            restored = _try_restore(landed, row_tradable)
            if restored is not None:
                _corrected, moved, _still_pending = restored
                drift_compliance_pending[t, moved] = True
                compliance_pending = True

        if not compliance_pending:
            # Defensive check, mirroring `rebalancing._assert_holdings_
            # compliant`'s own "never trust the invariant blindly"
            # philosophy: whenever the row-walk believes no compliance
            # debt remains outstanding for THIS row, `landed` must
            # actually be compliant -- checked on the row's final,
            # fully-assembled value (not inside `_try_restore` itself,
            # which can legitimately return `pending=False` for an
            # INTERMEDIATE state that a later, independent `_try_restore`
            # call in this same row -- see the "combining this row's
            # just-decided columns" branch above -- is specifically
            # responsible for re-checking against a value it hadn't seen
            # yet). A violation here would mean a bug in the LP
            # formulation (or in how this loop combines its calls), not a
            # bad input -- but a silent violation would be an expensive,
            # hard-to-diagnose out-of-mandate position, so this fails
            # loudly rather than reporting a clean "compliance restored"
            # trade-log event over a row that still breaches a limit.
            remaining = _violations(landed)
            if remaining:
                raise BacktestError(
                    "apply_weight_drift produced a row believed fully "
                    f"compliant (no pending debt) that still violates: "
                    f"{', '.join(remaining)} -- this indicates a bug in "
                    "the drift-compliance restoration, not a legitimate "
                    "runtime condition."
                )
        out[t] = landed
        trade_changes[t] = landed - weight_prev
        touched = np.abs(landed - weight_prev) > EPSILON
        dollar = np.where(touched, landed * equity, dollar)
        if not ordinary_mask.any() and not compliance_pending:
            # Pure numerical hygiene, never a behavior change: `weight =
            # dollar / equity` is invariant under uniformly rescaling
            # both, so renormalizing to `E = 1.0` is always safe whenever
            # no debt of any kind remains outstanding.
            dollar = landed.copy()
            equity = 1.0

        # === Advance state using row t's OWN return, with out[t] (this
        # row's just-finalized output) as the base weight. `dollar/equity`
        # already equal `out[t]` here. The result becomes `weight_prev` --
        # and so, by default, `out[t+1]` -- for the next row. ===
        r = returns_np[t]
        r = np.where(np.isfinite(r), r, 0.0)
        gross_return_t = float(np.sum(out[t] * r))
        dollar = dollar * (1.0 + r)
        equity = equity * (1.0 + gross_return_t)

        if equity <= EPSILON:
            dollar = np.zeros(n_cols)
            equity = 1.0
            ordinary_mask = np.zeros(n_cols, dtype=bool)
            compliance_pending = False
            logger.warning(
                "Weight-drift equity (relative to its own last anchor, "
                "gross/pre-cost) reached zero or below at %s -- "
                "flattening this episode's drifted positions. Check "
                "leverage and cost configuration.",
                executed.index[t],
            )

    pre_period_weights = pd.DataFrame(
        out, index=executed.index, columns=executed.columns
    )
    trade_changes_frame = pd.DataFrame(
        trade_changes, index=executed.index, columns=executed.columns
    )
    provenance = DriftProvenance(
        drift_compliance_forced=pd.DataFrame(
            drift_compliance_forced, index=executed.index, columns=executed.columns
        ),
        drift_compliance_pending=pd.DataFrame(
            drift_compliance_pending, index=executed.index, columns=executed.columns
        ),
        drift_turnover_actively_limited=pd.DataFrame(
            drift_turnover_actively_limited,
            index=executed.index,
            columns=executed.columns,
        ),
        drift_turnover_touched=pd.DataFrame(
            drift_turnover_touched, index=executed.index, columns=executed.columns
        ),
    )
    return pre_period_weights, trade_changes_frame, provenance


def run_accounting(
    held_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    execution_model: ExecutionModel,
    initial_capital: float,
    *,
    tradable: pd.DataFrame | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    position_groups: Sequence[tuple[str, ...]] | None = None,
    model_weight_drift: bool = False,
    maximum_weight: float | None = None,
    maximum_gross_exposure: float | None = None,
    maximum_net_exposure: float | None = None,
    long_only: bool = False,
    rebalance_date: pd.DataFrame | None = None,
    maximum_turnover: float | None = None,
) -> AccountingResult:
    """Run the vectorised accounting loop.

    Args:
        held_weights: Target weights actually held (step function after
            rebalancing), ``dates × symbols``.
        asset_returns: Per-asset simple returns aligned to ``held_weights``.
        execution_model: Cost model.
        initial_capital: Starting equity.
        tradable: When given, the mandatory look-ahead-barrier shift becomes
            per-symbol tradability-aware (see
            :func:`quantlab.execution.orders.executed_weights`): a decision
            made right before a closure executes on that symbol's next real
            tradable row, not the raw next row, so it is never misattributed
            as trading during the closure itself (e.g. a weekend row that
            only exists because another, always-open instrument shares the
            same combined index). Also the tradability `apply_weight_drift`
            (when enabled) uses for its own compliance-restoration LP.
        stop_loss_pct: Fractional (e.g. 0.10 = 10%) gross-return threshold
            that force-flattens a position/group -- see
            :func:`_detect_stop_loss_take_profit` for the exact formula
            and why it operates on the real executed position rather than
            a raw strategy signal. ``None`` (default) disables it, with
            strictly no change to accounting's own numbers.
        take_profit_pct: Same, on the favorable side.
        position_groups: Column groups (e.g. pairs_trading's two legs)
            whose combined P&L, not each column's own, drives the stop-
            loss/take-profit check -- see :func:`_resolve_position_groups`.
            A column absent from every group is its own independent group.
            Also the groups `apply_weight_drift`'s compliance-restoration
            LP moves coherently via one shared scalar.
        model_weight_drift: When ``True``, evolve ``executed`` forward by
            organic price drift between real trades (see
            :func:`apply_weight_drift`) instead of holding it constant
            until the next scheduled rebalance -- the constant-weight
            step function is a documented approximation this flag
            replaces with a materially more accurate one. ``False``
            (default here) is strictly a no-op: ``executed`` passes
            through unchanged.
        maximum_weight: Per-asset hard cap, re-enforced on every drift row
            when `model_weight_drift` is enabled (see
            :func:`quantlab.portfolio.drift_compliance.
            restore_drift_compliance`). Ignored when `model_weight_drift`
            is ``False``.
        maximum_gross_exposure: Portfolio-level gross cap, same treatment.
        maximum_net_exposure: Portfolio-level net cap, same treatment.
        long_only: Same treatment.
        rebalance_date: Ignored unless `model_weight_drift` is `True`.
            Boolean, ``dates x symbols`` DataFrame matching `held_weights`'
            own dates and symbols, aligned to the EXECUTED timeline
            exactly like `tradable` -- `True` on a (row, column) whose
            underlying decision for THAT symbol was made on a genuine
            scheduled rebalance date; a column must never be marked
            `True` on a date it is not itself tradable. `None` falls back
            to anchor detection from `executed`'s own row-to-row diff
            alone. See :func:`apply_weight_drift`'s own docstring and
            docs/backtesting.md#weight-drift for why this matters (a
            constant-target schedule is otherwise invisible to value-
            diffing) and the full anchor-detection mechanics.
        maximum_turnover: Forwarded to :func:`apply_weight_drift` when
            `model_weight_drift` is enabled, bounding the L1 size of an
            ordinary anchor's catch-up trade (exempting a hard-risk-limit
            drift-compliance correction -- see docs/backtesting.md#weight-
            drift for the exact mechanics). Ignored when `model_weight_
            drift` is `False` (the decision-level `rebalance_and_cap_
            turnover` cap already applies there, unaffected by this
            parameter either way). `None` (default) leaves anchor catch-
            ups uncapped.

    Returns:
        A populated :class:`AccountingResult`.

    Raises:
        BacktestError: If capital, weights, returns or labels are invalid, a
            held position has no return, or the cost/equity solve fails.
    """
    if isinstance(initial_capital, (bool, np.bool_)) or not isinstance(
        initial_capital, Real
    ):
        raise BacktestError("initial_capital must be a finite number greater than 0.")
    try:
        capital = float(initial_capital)
    except (TypeError, ValueError) as exc:
        raise BacktestError(
            "initial_capital must be a finite number greater than 0."
        ) from exc
    if not np.isfinite(capital) or capital <= 0.0:
        raise BacktestError("initial_capital must be a finite number greater than 0.")
    if not isinstance(execution_model, ExecutionModel):
        raise BacktestError(
            f"execution_model must be an ExecutionModel instance, got "
            f"{execution_model!r}."
        )
    _validate_drift_and_risk_options(
        long_only=long_only,
        maximum_weight=maximum_weight,
        maximum_gross_exposure=maximum_gross_exposure,
        maximum_net_exposure=maximum_net_exposure,
        model_weight_drift=model_weight_drift,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        maximum_turnover=maximum_turnover,
    )

    for name, frame in (
        ("held_weights", held_weights),
        ("asset_returns", asset_returns),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise BacktestError(f"{name} must be a pandas DataFrame.")
        if not frame.index.is_unique:
            raise BacktestError(f"{name} index must not contain duplicate labels.")
        if not frame.columns.is_unique:
            raise BacktestError(f"{name} columns must not contain duplicate labels.")

    if held_weights.empty:
        raise BacktestError("held_weights must contain at least one date and symbol.")

    missing_dates = held_weights.index.difference(asset_returns.index)
    missing_symbols = held_weights.columns.difference(asset_returns.columns)
    if len(missing_dates) or len(missing_symbols):
        raise BacktestError(
            "asset_returns must cover every held_weights date and symbol "
            f"(missing dates: {list(missing_dates)[:5]}, missing symbols: "
            f"{list(missing_symbols)[:5]})."
        )

    if tradable is not None:
        # tradable is only ever built internally from held_weights' own
        # (date, symbol) grid, never user input, so a mismatched axis set
        # always means an upstream wiring bug -- raise loudly rather than
        # silently default an unrecognized cell to "tradable" and risk
        # trading a symbol that should have stayed closed. Axis *order*
        # alone is not a mismatch: a caller may build tradable from a
        # declared symbol list while held_weights comes from an
        # alphabetically-pivoted price matrix. Strictly boolean dtype
        # (never e.g. the string "False", which would otherwise coerce to
        # truthy) is enforced by the shared helper.
        tradable = _validate_tradable_mask(
            tradable, held_weights, reference_name="held_weights"
        )

    held = held_weights.sort_index()
    if tradable is not None:
        tradable = tradable.reindex(index=held.index, columns=held.columns)
    asset_returns = asset_returns.reindex_like(held)
    try:
        return_values = asset_returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("asset_returns must contain only numeric values.") from exc
    if np.isinf(return_values).any():
        raise BacktestError("asset_returns must not contain Infinity.")
    finite_returns = return_values[np.isfinite(return_values)]
    if (finite_returns < -1.0).any():
        raise BacktestError(
            "asset_returns must not contain simple returns below -1.0 (-100%)."
        )

    # Shift weights so period-t return uses weights chosen at t-1. `tradable`
    # was already validated above to share held_weights' exact axes and
    # sorted alongside it, so it needs no further alignment here.
    executed = compute_executed_weights(held, tradable=tradable)

    # Weight drift operates on this ALREADY shift-respecting-tradability,
    # look-ahead-barrier-applied real executed book -- see
    # apply_weight_drift's own docstring for why this must be the post-
    # shift executed timeline, never the pre-shift decision timeline
    # rebalancing.py produces. `False` (default) is a provable no-op.
    drift_provenance: DriftProvenance | None = None
    # Real-trade-only turnover/cost input (see apply_weight_drift's own
    # docstring): `None` outside drift, where `executed` is already a
    # plain step function and every row-to-row diff genuinely IS a trade.
    weight_changes_override: pd.DataFrame | None = None
    if model_weight_drift:
        executed, weight_changes_override, drift_provenance = apply_weight_drift(
            executed,
            asset_returns,
            tradable,
            position_groups,
            maximum_weight=maximum_weight,
            maximum_gross_exposure=maximum_gross_exposure,
            maximum_net_exposure=maximum_net_exposure,
            long_only=long_only,
            rebalance_date=rebalance_date,
            maximum_turnover=maximum_turnover,
        )

    result = _solve_accounting(
        executed,
        asset_returns,
        execution_model,
        capital,
        weight_changes_override=weight_changes_override,
    )

    # Detect stop-loss/take-profit breaches on the REAL executed position
    # from this initial pass, then -- exactly like the ruin handling below
    # -- gate the affected cells and re-solve so turnover, costs, returns
    # and equity are all self-consistent with the forced flatten. Checked
    # BEFORE ruin: a stop-loss is a strategy-level risk control, not the
    # portfolio-wide catastrophe ruin represents (which still overrides
    # it below if both occur).
    gated_executed, stop_loss_triggered, take_profit_triggered, gated_weight_changes = (
        _detect_stop_loss_take_profit(
            result.executed_weights,
            asset_returns,
            position_groups,
            stop_loss_pct,
            take_profit_pct,
            weight_changes=weight_changes_override,
        )
    )
    if stop_loss_triggered.to_numpy().any() or take_profit_triggered.to_numpy().any():
        executed = gated_executed
        weight_changes_override = gated_weight_changes
        result = _solve_accounting(
            executed,
            asset_returns,
            execution_model,
            capital,
            weight_changes_override=weight_changes_override,
        )

    # After equity reaches zero, flatten later positions and recompute so
    # returns, turnover, costs and the trade log contain no post-ruin trades.
    ruined = result.equity.shift(1).fillna(capital) <= 0.0
    if ruined.any():
        logger.warning(
            "Portfolio equity reached zero at %s — no margin calls are "
            "modeled, so trading stops there: positions, turnover, costs "
            "and returns are held flat for every subsequent period instead "
            "of continuing to simulate trades against capital that no "
            "longer exists.",
            result.equity.index[result.equity <= 0.0][0],
        )
        executed = executed.copy()
        executed.loc[ruined, :] = 0.0
        result = _solve_accounting(
            executed,
            asset_returns,
            execution_model,
            capital,
            force_flat=ruined,
            weight_changes_override=weight_changes_override,
        )
    result.ruined = ruined
    # `_solve_accounting` above (both the stop-loss/take-profit re-run and
    # this ruin re-run) returns a fresh AccountingResult whose stop_loss_
    # triggered/take_profit_triggered are all-False placeholders -- restore
    # the real provenance detected earlier. `forced_liquidation` still wins
    # over these in the trade log's own reason-priority ordering when both
    # coincide on the same cell, so no need to clear them on ruined dates.
    result.stop_loss_triggered = stop_loss_triggered
    result.take_profit_triggered = take_profit_triggered
    if drift_provenance is not None:
        result.drift_compliance_forced = drift_provenance.drift_compliance_forced
        result.drift_compliance_pending = drift_provenance.drift_compliance_pending
        result.drift_turnover_actively_limited = (
            drift_provenance.drift_turnover_actively_limited
        )
        result.drift_turnover_touched = drift_provenance.drift_turnover_touched
        # `DriftProvenance.__post_init__` already enforced forced/pending
        # mutual exclusion at construction time, but `AccountingResult` is
        # a plain mutable dataclass with no invariant of its own -- that
        # guarantee does not survive being flattened onto its separately-
        # mutable fields above. Cheap re-check here too, so a future
        # change to this flattening step (the only place that does it)
        # can't silently reintroduce an overlap nothing downstream would
        # otherwise catch.
        overlap = result.drift_compliance_forced & result.drift_compliance_pending
        if overlap.to_numpy().any():
            raise BacktestError(
                "AccountingResult.drift_compliance_forced and "
                "drift_compliance_pending are not mutually exclusive after "
                "being attached in run_accounting -- this indicates a bug "
                "in that flattening step, not in DriftProvenance's own "
                "construction."
            )

    logger.info(
        "Accounting: %d periods, final equity %.2f (gross %.2f), avg turnover %.4f",
        len(result.equity),
        float(result.equity.iloc[-1]) if len(result.equity) else capital,
        float(result.gross_equity.iloc[-1]) if len(result.gross_equity) else capital,
        float(result.turnover.mean()) if len(result.turnover) else 0.0,
    )
    return result
