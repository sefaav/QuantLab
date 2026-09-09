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
    # Per-symbol Δ of the executed book vs. the pre-overnight holding --
    # "which cell just traded", not a dollar-notional/turnover basis.
    # Identical to ``weight_changes_for_costs`` below outside
    # ``execution_timing="next_bar_open"``.
    weight_changes: pd.DataFrame
    asset_returns: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: ExecutionCosts
    # L1 weight change actually traded, sized against equity AT execution
    # time (open(T), after the overnight leg, under next_bar_open; else
    # close(T-1)) -- see ``weight_changes_for_costs`` below.
    turnover: pd.Series
    # The per-cell delta ``turnover``/``costs`` above are sized from.
    # A trade log (or any other per-fill dollar breakdown) must use THIS
    # field for notional, never ``weight_changes`` -- its per-fill total
    # would otherwise silently stop matching ``costs.total`` under
    # next_bar_open (see ``_gross_returns_and_cost_basis``'s docstring).
    weight_changes_for_costs: pd.DataFrame
    equity: pd.Series  # net-of-cost equity curve
    gross_equity: pd.Series  # gross (cost-free) equity curve
    # Net-equity estimate used to size volume-dependent slippage. Reuse it in
    # the trade log to keep per-fill and aggregate costs consistent.
    equity_for_costs: pd.Series
    # The REAL equity a trade executes against (see ``turnover``'s own
    # doc above). Pairs with ``weight_changes_for_costs`` to size a
    # per-fill dollar notional that reconciles exactly with
    # ``costs``/``turnover``, for every execution_timing mode uniformly.
    execution_equity: pd.Series
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


def _zero_safe_multiply(weight: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """``weight * returns``, forced exactly ``0.0`` wherever ``weight`` is ~zero.

    True ``0 * x = 0``, not IEEE's ``0 * NaN = NaN`` -- a zero-weight leg
    contributes zero regardless of whether its own return is defined that
    day (e.g. no data once a fully-exited symbol stops being quoted),
    rather than poisoning a downstream ``skipna`` sum with NaN for a leg
    that shouldn't count at all.

    Compares against ``EPSILON``, not exact ``== 0.0``: ``weight`` is
    often itself an arithmetic result (e.g. a subtraction), so a
    semantically-flat cell can carry a tiny floating-point residual an
    exact-equality check would miss.
    """
    return (weight * returns).where(weight.abs() > EPSILON, 0.0)


def _close_of_previous_row(
    executed: pd.DataFrame, intraday_returns: pd.DataFrame
) -> pd.DataFrame:
    """The weight actually held at the close of the PREVIOUS row.

    Needed whenever weight drift is active: ``executed - weight_changes``
    (the identity that recovers it for a plain step-function book) no
    longer works once ``weight_changes`` is measured against the
    open-time baseline instead of the close(T-1) one (see
    ``apply_weight_drift``'s own ``overnight_returns`` handling). This
    grows each row's own ``executed`` forward by that SAME row's own
    intraday leg, then shifts by one row -- purely a function of
    ``executed``/``intraday_returns``, so it is recomputed fresh on every
    call and stays correct across a stop-loss- or ruin-gated re-solve,
    never a value cached from an earlier pass.

    A row whose intraday leg is a total (or worse) loss makes
    ``(1+intraday_leg)`` collapse to (near-)zero -- relying on the
    EPSILON-floored denominator alone would turn that into a huge-but-
    finite ``close_weight`` for any row with an unbounded (e.g.
    leveraged) ``executed`` value, instead of the "nothing survived"
    zero this row's own zero equity implies. Masked to exactly 0.0 on
    such a row, mirroring ``_gross_returns_and_cost_basis``'s own
    ``wiped_out`` guard.
    """
    dollar_at_close = _zero_safe_multiply(executed, 1.0 + intraday_returns)
    intraday_leg = (
        _zero_safe_multiply(executed, intraday_returns)
        .sum(axis=1, min_count=1)
        .fillna(0.0)
    )
    intraday_leg = _floor_at_total_loss(intraday_leg, "intraday leg")
    wiped_out = (1.0 + intraday_leg) <= EPSILON
    close_weight = dollar_at_close.div((1.0 + intraday_leg).clip(lower=EPSILON), axis=0)
    close_weight = close_weight.where(~wiped_out, 0.0)
    return close_weight.shift(1, fill_value=0.0)


def _gross_returns_and_cost_basis(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    weight_changes: pd.DataFrame,
    *,
    overnight_returns: pd.DataFrame | None,
    drift_active: bool = False,
) -> tuple[pd.Series, pd.DataFrame, pd.Series | None]:
    """Return ``(gross_returns, weight_changes_for_costs, overnight_leg)``.

    ``overnight_returns=None`` (``execution_timing="same_bar_close"``):
    ``gross_returns`` is the plain ``(executed * asset_returns).sum(axis=1)``,
    and ``weight_changes_for_costs`` is ``weight_changes`` itself, unaltered.

    Otherwise (``execution_timing="next_bar_open"``), every date's return
    is decomposed into a ``close[T-1] -> open[T]`` (overnight) leg earned
    on the PRE-trade holding and an ``open[T] -> close[T]`` (intraday) leg
    earned on the POST-trade holding, and the two legs are COMPOUNDED --
    ``(1+overnight_leg)*(1+intraday_leg) - 1``, never added. Addition would
    drop the cross term: a portfolio 100% in A that earns +10% overnight
    then rotates entirely into B, which earns +10% intraday, has genuinely
    earned ``1.10*1.10-1 = 21%`` (the overnight gain funds a LARGER
    position in B), not ``10%+10% = 20%``.

    Both legs are computed PORTFOLIO-WIDE -- every column, not only the
    ones that traded that day -- because the whole book (including
    untraded positions and any uninvested cash) earns the overnight leg
    together before any trade executes, and that resulting open-time
    holding is what the intraday leg keys off. An untraded column's own
    row-to-row-constant ``executed`` value is therefore NOT its true
    weight at the open: its dollar amount drifted overnight too.
    ``weight_at_open`` recovers that drifted value and stands in for
    ``executed`` on every untraded column when assembling the intraday
    leg; a column that genuinely traded today (``weight_changes``
    nonzero) uses its literal, freshly-decided ``executed`` value there
    instead, since a fresh decision is meaningful only relative to the
    equity that exists when it executes -- the open.

    This is exactly (not approximately) the plain undecomposed formula
    whenever nothing traded anywhere in the book that day: every column's
    ``intraday_basis`` collapses to its own ``weight_at_open``, and
    ``overnight + (1+overnight)*intraday`` telescopes to
    ``(1+overnight)*(1+intraday) - 1 = asset_return``. This holds even
    when different columns carry different overnight/intraday splits,
    which a per-cell ADDITIVE combination would not: expanding
    ``(1+w1*o1+w2*o2)*(1+w1*i1+w2*i2)-1`` for two untraded columns
    carries a spurious cross term (``w1*w2*o1*i2``) that routing each
    column's own contribution through its own ``weight_at_open`` avoids.

    ``weight_changes_for_costs`` mirrors ``weight_changes`` except on
    traded cells, where it reports the trade size measured against the
    open-time holding (``executed - weight_at_open``) rather than the
    stale pre-overnight holding (``executed - pre``) -- turnover and
    every cost derived from it (commission, spread, volume-dependent
    slippage) should be sized on what a real order at the open actually
    has to move. Untraded cells keep their original (near-zero)
    ``weight_changes`` value unconditionally -- organic overnight drift
    is never itself a trade, the same principle ``weight_changes_override``
    protects elsewhere in this module.

    ``pre`` (what was held going into today, before today's own trade) is
    ``executed - weight_changes`` for a plain step-function book
    (``drift_active=False``), never a separately cached snapshot -- an
    exact identity there. Weight drift changes what ``weight_changes``
    measures (see :func:`apply_weight_drift`'s own ``overnight_returns``
    handling): once it is the REAL trade size against the open-time
    baseline, that subtraction recovers open(T), not close(T-1). With
    ``drift_active=True``, ``pre`` is instead
    :func:`_close_of_previous_row` -- purely a function of ``executed``/
    ``intraday_returns``, so a stop-loss- or ruin-gated re-solve always
    gets the correctly-gated close(T-1), never a stale value.

    ``overnight_leg`` is the portfolio-wide overnight-leg return per date
    (``None`` when ``overnight_returns`` is ``None``), returned so the
    caller can size costs on the equity that actually exists at the open
    (``equity[T-1] * (1+overnight_leg[T])``), not stale ``equity[T-1]``.
    """
    if overnight_returns is None:
        gross_returns = (executed * asset_returns).sum(axis=1, min_count=1).fillna(0.0)
        return gross_returns, weight_changes, None

    # (1+overnight)*(1+intraday) = 1+asset_return, by construction of the
    # two legs -- derive intraday from the two already-validated return
    # series instead of re-pivoting a third price matrix. Computed before
    # `pre` since the weight-drift branch below also needs it.
    intraday_returns = (1.0 + asset_returns) / (1.0 + overnight_returns) - 1.0
    pre = (
        _close_of_previous_row(executed, intraday_returns)
        if drift_active
        else executed - weight_changes
    )
    changed = weight_changes.abs() > EPSILON

    # Overnight leg: every column's PRE-trade holding earns its own
    # overnight return -- portfolio-wide, not gated on `changed`, since
    # the whole book (traded columns included) was still holding `pre`
    # through the overnight gap.
    overnight_leg = (
        _zero_safe_multiply(pre, overnight_returns).sum(axis=1, min_count=1).fillna(0.0)
    )
    # A total-wipeout (or worse) overnight leg would make `1+overnight_leg`
    # zero or negative, which the division below cannot tolerate -- floor
    # it exactly like `_floor_at_total_loss` floors every other return
    # series in this module, so a pathological input degrades to "flat at
    # zero" instead of raising or dividing by (near-)zero.
    overnight_leg = _floor_at_total_loss(overnight_leg, "overnight leg")
    # A total (or worse) overnight wipeout leaves no equity for any order
    # this row to actually execute against -- `weight_at_open` and every
    # trade size derived from it are economically meaningless once
    # `(1+overnight_leg)` collapses to zero. Detected BEFORE the division
    # below, not from its output: the (necessarily EPSILON-floored)
    # denominator would otherwise turn a wipeout into a huge-but-finite
    # `weight_at_open` instead of the "no orders, no turnover, no cost"
    # this row's own real (zero) equity implies.
    wiped_out = (1.0 + overnight_leg) <= EPSILON

    # `weight_at_open`: each column's OWN pre-trade holding, organically
    # drifted by its OWN overnight return, then renormalized by the
    # portfolio's overall overnight growth -- the weight-drift mechanism
    # (`apply_weight_drift`'s own dollar/equity recursion) applied as a
    # single, portfolio-wide step for the overnight leg alone.
    dollar_at_open = _zero_safe_multiply(pre, 1.0 + overnight_returns)
    weight_at_open = dollar_at_open.div(
        (1.0 + overnight_leg).clip(lower=EPSILON), axis=0
    )
    weight_at_open = weight_at_open.where(~wiped_out, 0.0)

    # Every column's intraday-leg basis: its freshly-decided `executed`
    # value where it genuinely traded today, else its own drifted
    # `weight_at_open` -- see this function's own docstring for why an
    # untraded column's stale `executed` value would otherwise silently
    # misrepresent its true open-time weight. Forced to 0 on a wiped-out
    # row for the same reason `weight_at_open` is above: relying on
    # `(1+overnight_leg)` alone to collapse `gross_returns` to exactly -1
    # would silently fail if a changed cell's raw `executed` (unbounded
    # here, e.g. under high leverage) combined with an extreme intraday
    # return ever overcame that near-zero factor numerically.
    intraday_basis = executed.where(changed, weight_at_open)
    intraday_basis = intraday_basis.where(~wiped_out, 0.0)
    intraday_leg = (
        _zero_safe_multiply(intraday_basis, intraday_returns)
        .sum(axis=1, min_count=1)
        .fillna(0.0)
    )

    # (1+overnight_leg) is already exactly 0 (or clipped from something
    # even more negative) on a wiped-out row, so this compounds to
    # exactly -1 regardless of `intraday_leg` -- correctly preserves the
    # real, already-realized overnight loss on the row it happens on.
    gross_returns = (1.0 + overnight_leg) * (1.0 + intraday_leg) - 1.0

    weight_changes_for_costs = weight_changes.where(~changed, executed - weight_at_open)
    # No real trade is possible with no equity left -- forcing every
    # column's contribution to exactly 0.0 on a wiped-out row (not just
    # `weight_at_open`) is what keeps `turnover`/costs sane instead of
    # reporting a phantom rotation into whatever was merely DECIDED that
    # row, priced as if real capital still existed to move.
    weight_changes_for_costs = weight_changes_for_costs.where(~wiped_out, 0.0)
    return gross_returns, weight_changes_for_costs, overnight_leg


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


@dataclass
class _AccountingBasis:
    """Cost-equity-INDEPENDENT quantities.

    Computed once by :func:`_prepare_accounting_basis` and reused unchanged
    across every iteration of :func:`_solve_accounting`'s fixed-point loop
    -- see that function's own docstring for why none of this depends on
    ``cost_equity``.
    """

    executed: pd.DataFrame
    weight_changes: pd.DataFrame
    gross_returns: pd.Series
    gross_equity: pd.Series
    turnover: pd.Series
    weight_changes_for_costs: pd.DataFrame
    overnight_leg: pd.Series | None


def _prepare_accounting_basis(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    initial_capital: float,
    *,
    force_flat: pd.Series | None = None,
    weight_changes_override: pd.DataFrame | None = None,
    overnight_returns: pd.DataFrame | None = None,
) -> _AccountingBasis:
    """Validate inputs and derive every cost-equity-INDEPENDENT quantity.

    ``turnover``/``gross_returns``/``weight_changes_for_costs``/
    ``overnight_leg`` are a pure function of ``executed``/``asset_returns``/
    ``overnight_returns`` alone -- none of them depend on ``cost_equity``.
    Computing them once here (rather than inside the per-iteration step,
    :func:`_accounting_result_from_basis`) means :func:`_solve_accounting`'s
    fixed-point loop -- up to ``_MAX_COST_EQUITY_ITERATIONS`` passes -- pays
    for the full-panel overnight/intraday decomposition (and its own
    up-front validation) exactly once per :func:`run_accounting`/
    :func:`apply_weight_drift` call, not once per iteration of a loop whose
    only actual unknown is the equity curve costs are sized against.

    Args:
        executed: Weights actually in force each period (post look-ahead
            shift), ``dates × symbols``.
        asset_returns: Per-asset simple returns aligned to ``executed``.
        initial_capital: Starting equity.
        force_flat: Dates on which turnover is forced to zero after
            bankruptcy, preventing a closing trade with no remaining capital.
        weight_changes_override: When given, used as turnover/cost input
            INSTEAD OF ``executed``'s own row-to-row diff -- required when
            ``executed`` is not a plain step function (weight drift is
            active), since consecutive rows then genuinely differ from
            organic price movement alone, never a real trade; a naive diff
            would charge phantom turnover/costs for every drifting row.
            ``None`` (the default) computes turnover from ``executed``'s
            own plain row-to-row diff.
        overnight_returns: ``close[T-1] -> open[T]`` per-asset returns.
            ``None`` (default, ``execution_timing="same_bar_close"``) is a
            strict no-op on gross-return computation. Otherwise, see
            :func:`_gross_returns_and_cost_basis` for the exact
            decomposition this enables.
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
    if overnight_returns is not None:
        # Required wherever the decomposition actually consults
        # overnight_returns for a nonzero-weight contribution: every cell
        # held going INTO today (`pre`, the overnight leg's own basis) or
        # held coming OUT of today's trade (`executed`, needed even on an
        # untraded cell to derive `intraday_returns`, and on a freshly-
        # entered cell as `intraday_basis` directly) -- the overnight leg
        # is computed portfolio-wide, not gated on `weight_changes` alone.
        drift_active = weight_changes_override is not None
        if drift_active:
            intraday_returns_for_check = (1.0 + asset_returns) / (
                1.0 + overnight_returns
            ) - 1.0
            pre_for_check = _close_of_previous_row(executed, intraday_returns_for_check)
        else:
            pre_for_check = executed - weight_changes
        overnight_needed = (pre_for_check.abs() > EPSILON) | (executed.abs() > EPSILON)
        missing_overnight = overnight_returns.isna() & overnight_needed
        if missing_overnight.to_numpy().any():
            first_row, first_column = np.argwhere(missing_overnight.to_numpy())[0]
            first_date = missing_overnight.index[int(first_row)]
            first_symbol = missing_overnight.columns[int(first_column)]
            raise BacktestError(
                "Overnight return is missing on a cell held before or "
                f"after today's trade: {first_symbol!r} on {first_date!r}."
            )

    gross_returns, weight_changes_for_costs, overnight_leg = (
        _gross_returns_and_cost_basis(
            executed,
            asset_returns,
            weight_changes,
            overnight_returns=overnight_returns,
            drift_active=weight_changes_override is not None,
        )
    )
    gross_returns = _floor_at_total_loss(gross_returns, "gross")
    gross_equity = initial_capital * (1.0 + gross_returns).cumprod()
    turnover = weight_changes_for_costs.abs().sum(axis=1)

    if overnight_leg is not None:
        # A row whose OVERNIGHT leg alone wipes out the portfolio
        # (`1+overnight_leg` collapsing to ~0) means the scheduled trade
        # this row's `executed`/`weight_changes` describe never actually
        # happens: under `next_bar_open`, that trade executes AT the open,
        # by which point there is no capital left to move -- exactly the
        # `wiped_out` condition `_gross_returns_and_cost_basis` already
        # uses to zero `weight_changes_for_costs`/`turnover`/`costs`
        # above. Left unzeroed here, `executed_weights`/`weight_changes`
        # (this function's own OUTPUT fields) would still show a
        # transaction into a position that was never really taken --
        # `AccountingResult.executed_weights` is documented as "weights
        # actually in force", and a bankrupt row has none. Mirrors
        # `apply_weight_drift`'s own `_reset_if_bankrupt` handling of the
        # identical scenario on its own (per-row) code path -- idempotent
        # when `executed`/`weight_changes` are already zeroed there, since
        # both derive `wiped_out` from the same `(1+overnight_leg)<=
        # EPSILON` condition.
        wiped_out = (1.0 + overnight_leg) <= EPSILON
        if wiped_out.any():
            executed = executed.copy()
            executed.loc[wiped_out, :] = 0.0
            weight_changes = weight_changes.copy()
            weight_changes.loc[wiped_out, :] = 0.0

    return _AccountingBasis(
        executed=executed,
        weight_changes=weight_changes,
        gross_returns=gross_returns,
        gross_equity=gross_equity,
        turnover=turnover,
        weight_changes_for_costs=weight_changes_for_costs,
        overnight_leg=overnight_leg,
    )


def _accounting_result_from_basis(
    basis: _AccountingBasis,
    asset_returns: pd.DataFrame,
    execution_model: ExecutionModel,
    initial_capital: float,
    *,
    cost_equity: pd.Series | None = None,
) -> AccountingResult:
    """Apply the one COST-EQUITY-DEPENDENT step on top of a shared ``basis``.

    Args:
        basis: Precomputed by :func:`_prepare_accounting_basis`, once per
            :func:`_solve_accounting` call -- reused unchanged across every
            iteration of its fixed-point loop.
        asset_returns: Per-asset simple returns aligned to ``basis.executed``.
        execution_model: Cost model.
        initial_capital: Starting equity.
        cost_equity: Previous net-equity estimate used to size
            volume-dependent slippage. The first pass uses gross equity.
    """
    executed = basis.executed
    weight_changes = basis.weight_changes
    gross_returns = basis.gross_returns
    gross_equity = basis.gross_equity
    turnover = basis.turnover
    weight_changes_for_costs = basis.weight_changes_for_costs
    overnight_leg = basis.overnight_leg

    # Size costs from prior-period equity. The first fixed-point
    # pass uses gross equity; later passes feed back the latest net estimate.
    equity_for_costs = gross_equity if cost_equity is None else cost_equity
    prior_equity = equity_for_costs.shift(1).fillna(initial_capital)
    if overnight_leg is not None:
        # A next_bar_open trade executes at the open -- cost/slippage
        # sizing must use that same equity-at-open basis.
        prior_equity = prior_equity * (1.0 + overnight_leg)
    costs = execution_model.compute(weight_changes_for_costs, equity=prior_equity)
    if overnight_leg is not None:
        # Every cost component is a fraction of open(T) equity (the same
        # basis `weight_changes_for_costs` itself is measured against),
        # but `gross_returns` below is a fraction of close(T-1) equity --
        # rescale by the same `(1+overnight_leg)` factor used for
        # `prior_equity` above so the subtraction below (and every other
        # consumer of `costs`, e.g. the trade log) isn't mixing two
        # different euros.
        growth = 1.0 + overnight_leg
        costs = ExecutionCosts(
            commission=costs.commission * growth,
            spread=costs.spread * growth,
            slippage=costs.slippage * growth,
        )

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
        weight_changes_for_costs=weight_changes_for_costs,
        equity=equity,
        gross_equity=gross_equity,
        equity_for_costs=equity_for_costs,
        execution_equity=prior_equity,
        # Overwritten by run_accounting (the only real caller of
        # _solve_accounting) with the actual ruin/stop-loss/take-profit
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
    overnight_returns: pd.DataFrame | None = None,
) -> AccountingResult:
    """Solve equity-dependent costs to a self-consistent equity curve.

    Volume impact at t is sized from equity at t-1, which depends on earlier
    costs. Repeated vectorised passes feed the latest net-equity curve back
    into the cost model until the maximum difference reaches the tolerance
    -- see :func:`_prepare_accounting_basis` for why everything BUT that
    equity-dependent cost step is computed only once, up front, rather than
    on every pass.
    """
    basis = _prepare_accounting_basis(
        executed,
        asset_returns,
        initial_capital,
        force_flat=force_flat,
        weight_changes_override=weight_changes_override,
        overnight_returns=overnight_returns,
    )
    result = _accounting_result_from_basis(
        basis, asset_returns, execution_model, initial_capital
    )
    if not len(result.equity):
        return result
    tolerance = _COST_EQUITY_CONVERGENCE_TOLERANCE * initial_capital
    residual = float("inf")
    for _ in range(_MAX_COST_EQUITY_ITERATIONS - 1):
        next_result = _accounting_result_from_basis(
            basis,
            asset_returns,
            execution_model,
            initial_capital,
            cost_equity=result.equity,
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


def _group_return_with_execution_timing(
    pre: pd.DataFrame,
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    weight_changes: pd.DataFrame,
    columns: list[str],
    gross_exposure: np.ndarray,
    *,
    overnight_returns: pd.DataFrame | None,
) -> np.ndarray:
    """Group-level analogue of ``_gross_returns_and_cost_basis``'s two-leg return.

    Same compounding fix, same reason: a group that entered AT the open
    (``execution_timing="next_bar_open"``) must not have the overnight gap
    that happened BEFORE its own entry credited to (or blamed on) its
    stop-loss/take-profit cumulative return -- it was not yet held then.
    Rescales ``pre``/``executed`` to the group's OWN exposure (not
    portfolio equity, which this stop-loss/take-profit module never
    references at all) before compounding, exactly mirroring
    ``_gross_returns_and_cost_basis``'s per-portfolio derivation at the
    group scale instead.

    The returned value is the group's dollar P&L for the day (overnight
    leg plus intraday leg, correctly compounded so the overnight gain/loss
    also scales the capital the intraday leg trades with -- see
    ``_gross_returns_and_cost_basis``'s own cross-term docstring) divided
    by ``pre_exposure`` -- the group's exposure going INTO the day, before
    today's own trade -- not the caller-supplied ``gross_exposure`` (the
    group's raw, as-executed exposure AFTER today's trade). Both a
    changed cell's ``executed`` (already, by this module's own
    convention, a target relative to the CURRENT, already-grown-by-the-
    overnight-leg equity -- see ``_gross_returns_and_cost_basis``'s own
    docstring on why a fresh decision is never rescaled) and an unchanged
    cell's drifted ``weight_at_open`` are therefore expressed relative to
    the SAME basis (``pre_exposure * (1 + overnight_leg)``) before being
    combined -- using the caller's ``gross_exposure`` for one and this
    basis for the other would silently mix two different scales into one
    sum. ``pre_exposure`` coincides exactly with ``gross_exposure``
    whenever nothing in the group traded that day (``executed`` is then
    identically ``pre``), which is also why the ``overnight_returns=None``
    branch below can keep dividing by ``gross_exposure`` unchanged --
    there is no open/close split to reconcile a second basis against in
    that case. When ``pre_exposure`` is itself ~0 (the group entered
    fresh from flat, with nothing held overnight at all), that basis is
    undefined and the division falls back to ``gross_exposure`` instead
    -- correct here too, since ``overnight_leg`` is then also exactly 0
    (nothing to earn an overnight return on), so the return for the day
    is purely the intraday leg on the freshly-entered position, exactly
    matching what dividing by ``gross_exposure`` gives.

    ``overnight_returns=None`` (``same_bar_close``/``immediate_after_close``):
    unchanged existing behaviour, ``(executed*asset_returns).sum(axis=1) /
    gross_exposure``.
    """
    group_asset_returns = asset_returns[columns]
    group_executed = executed[columns]
    if overnight_returns is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = (
                (group_executed * group_asset_returns).sum(axis=1) / gross_exposure
            ).to_numpy(dtype=float)
        return np.asarray(result, dtype=float)

    group_pre = pre[columns]
    group_overnight = overnight_returns[columns]
    group_changed = weight_changes[columns].abs() > EPSILON
    pre_exposure = group_pre.abs().sum(axis=1).to_numpy(dtype=float)

    overnight_dollar = (
        _zero_safe_multiply(group_pre, group_overnight)
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        overnight_leg = np.where(
            pre_exposure > EPSILON, overnight_dollar / pre_exposure, 0.0
        )
    overnight_leg = np.clip(overnight_leg, -1.0 + EPSILON, None)
    # A wiped-out group's `weight_at_open` denominator is necessarily
    # EPSILON-floored below, which would otherwise turn a total overnight
    # loss into a huge-but-finite weight instead of the "no real exposure
    # left" this row's own zero equity implies -- and, unmasked, a changed
    # cell's unbounded `normalized_executed` combined with an extreme
    # intraday return could numerically overcome the near-zero
    # `(1+overnight_leg)` factor and stop `group_return` from collapsing
    # to exactly -1. Mirrors `_gross_returns_and_cost_basis`'s own
    # `wiped_out` guard at the group scale.
    wiped_out = pd.Series((1.0 + overnight_leg) <= EPSILON, index=group_pre.index)

    intraday_returns = (1.0 + group_asset_returns) / (1.0 + group_overnight) - 1.0
    # Shared by both `weight_at_open` (untraded cells) and
    # `normalized_executed` (changed cells) below -- see this function's
    # own docstring on why using two DIFFERENT denominators here (the
    # caller's `gross_exposure` for one, this basis for the other) would
    # silently mix two different scales into `intraday_basis`. A group
    # with NO pre-trade exposure at all (`pre_exposure` ~ 0, e.g. a fresh
    # entry from flat -- the ordinary, common case for any new position)
    # has no real "exposure at the open" to normalize against: falls back
    # to `gross_exposure` (today's actual, as-executed exposure) instead
    # of the EPSILON-floored `pre_exposure*(1+overnight_leg)`, which would
    # otherwise make `normalized_executed` explode by a factor of roughly
    # `1/EPSILON` for every ordinary fresh entry.
    open_time_group_exposure = pd.Series(
        np.where(
            pre_exposure > EPSILON,
            pre_exposure * (1.0 + overnight_leg),
            gross_exposure,
        ),
        index=group_pre.index,
    ).clip(lower=EPSILON)
    with np.errstate(invalid="ignore", divide="ignore"):
        weight_at_open = _zero_safe_multiply(group_pre, 1.0 + group_overnight).div(
            open_time_group_exposure, axis=0
        )
    weight_at_open = weight_at_open.where(~wiped_out, 0.0)
    normalized_executed = group_executed.div(open_time_group_exposure, axis=0)
    intraday_basis = normalized_executed.where(group_changed, weight_at_open)
    intraday_basis = intraday_basis.where(~wiped_out, 0.0)
    intraday_leg = (
        _zero_safe_multiply(intraday_basis, intraday_returns)
        .sum(axis=1)
        .to_numpy(dtype=float)
    )

    group_return = (1.0 + overnight_leg) * (1.0 + intraday_leg) - 1.0
    # A group with no exposure at all this period (gross_exposure <=
    # EPSILON) is handled by the caller's own flat-detection, but avoid
    # ever returning a spurious finite value derived from a clipped
    # near-zero denominator for such a row.
    return np.where(gross_exposure > EPSILON, group_return, np.nan)


def _detect_stop_loss_take_profit(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    position_groups: Sequence[tuple[str, ...]] | None,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    weight_changes: pd.DataFrame | None = None,
    *,
    overnight_returns: pd.DataFrame | None = None,
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

    ``overnight_returns`` (``execution_timing="next_bar_open"`` only):
    ``group_return`` is instead the group's own compounded overnight/
    intraday two-leg return (see
    :func:`_group_return_with_execution_timing`) -- a group that enters
    AT the open must not have the overnight gap that happened BEFORE its
    own entry credited toward (or against) its cumulative barrier return.
    ``None`` (default) uses the plain formula above unchanged.

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
    # recomputation of the drift trajectory. Available unconditionally
    # (falling back to `executed`'s own plain shift when the caller has no
    # weight-drift-provenance series to hand) since the two-leg
    # `overnight_returns` path below needs `pre` regardless of drift mode,
    # not only for the phantom-turnover patching further down.
    if weight_changes is None:
        before_state = executed.shift(1, fill_value=0.0)
    elif overnight_returns is not None:
        # Same swap as _gross_returns_and_cost_basis's own `pre`: once
        # weight drift is active, `executed - weight_changes` no longer
        # recovers close(T-1) (see that function's docstring), so this
        # must be derived the same formula-based way instead of reusing
        # the plain-step-function identity.
        intraday_returns_for_groups = (1.0 + asset_returns) / (
            1.0 + overnight_returns
        ) - 1.0
        before_state = _close_of_previous_row(executed, intraday_returns_for_groups)
    else:
        before_state = executed - weight_changes
    effective_weight_changes = (
        weight_changes if weight_changes is not None else executed - before_state
    )
    stop_loss_triggered = pd.DataFrame(
        False, index=executed.index, columns=executed.columns
    )
    take_profit_triggered = stop_loss_triggered.copy()
    for group in groups:
        columns = list(group)
        group_executed = executed[columns]
        gross_exposure = group_executed.abs().sum(axis=1).to_numpy(dtype=float)
        group_return = _group_return_with_execution_timing(
            before_state,
            executed,
            asset_returns,
            effective_weight_changes,
            columns,
            gross_exposure,
            overnight_returns=overnight_returns,
        )
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
                # The row the forced flatten ENDS on -- a fresh re-entry,
                # whether the group's own (pre-gating) exposure genuinely
                # passed through flat first, or `reversed_without_flat`
                # ended it directly on a same-bar sign flip. Never covered
                # by `force_flat_series`/`transition` above (both are
                # False here by construction -- this row is exactly where
                # `_walk_group_stop_loss_take_profit` resets `stopped` to
                # False again). Left unpatched, this row's trade would
                # still be measured against the UNGATED, pre-liquidation
                # `before_state` (the organically-drifted weight the
                # position would have had if it had never actually been
                # forced flat) instead of the REAL pre-trade state, which
                # this gating already forced to exactly zero -- a phantom
                # overnight/turnover/cost basis with no warning. The real
                # trade is always "from zero" here: whatever this row's
                # `gated` value is, minus the zero the previous row was
                # actually held at.
                exit_transition = ~force_flat_series & force_flat_series.shift(
                    1, fill_value=False
                )
                gated_weight_changes.loc[exit_transition, columns] = gated.loc[
                    exit_transition, columns
                ]
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


def _validate_overnight_returns(
    overnight_returns: object, reference: pd.DataFrame, *, reference_name: str
) -> pd.DataFrame:
    """Validate and axis-align an ``overnight_returns`` matrix against ``reference``.

    Shared by :func:`run_accounting` and :func:`apply_weight_drift` -- both
    are directly callable public functions, so neither may silently accept
    a malformed or misaligned overnight-return matrix. Index/columns
    uniqueness is checked before ``reindex_like`` below: reindexing FROM a
    duplicate-labeled axis is itself ambiguous and raises pandas' own raw,
    uninformative ``ValueError: cannot reindex on an axis with duplicate
    labels`` rather than this module's usual clear ``BacktestError``.
    Strictly greater than -1.0 (unlike ``asset_returns``' own ``< -1.0``
    check elsewhere in both callers): the overnight/intraday decomposition
    derives ``intraday_returns`` as ``(1+asset_returns)/(1+overnight_returns)
    - 1``, so an exact -1.0 here (a total-wipeout overnight leg, e.g. a
    zero adjusted open) divides by zero and would poison ``gross_returns``
    with inf/NaN on every decomposed cell for that (date, symbol) -- caught
    here, loudly, rather than silently downstream.
    """
    if not isinstance(overnight_returns, pd.DataFrame):
        raise BacktestError("overnight_returns must be a pandas DataFrame.")
    if not overnight_returns.index.is_unique:
        raise BacktestError(
            "overnight_returns index must not contain duplicate labels."
        )
    if not overnight_returns.columns.is_unique:
        raise BacktestError(
            "overnight_returns columns must not contain duplicate labels."
        )
    missing_dates = reference.index.difference(overnight_returns.index)
    missing_symbols = reference.columns.difference(overnight_returns.columns)
    if len(missing_dates) or len(missing_symbols):
        raise BacktestError(
            f"overnight_returns must cover every {reference_name} date and "
            f"symbol (missing dates: {list(missing_dates)[:5]}, missing "
            f"symbols: {list(missing_symbols)[:5]})."
        )
    overnight_returns = overnight_returns.reindex_like(reference)
    try:
        overnight_values = overnight_returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError(
            "overnight_returns must contain only numeric values."
        ) from exc
    if np.isinf(overnight_values).any():
        raise BacktestError("overnight_returns must not contain Infinity.")
    finite_overnight = overnight_values[np.isfinite(overnight_values)]
    if (finite_overnight <= -1.0).any():
        raise BacktestError(
            "overnight_returns must not contain simple returns at or "
            "below -1.0 (-100%): a -100% overnight leg would divide by "
            "zero when deriving the intraday leg."
        )
    return overnight_returns


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
    overnight_returns: pd.DataFrame | None = None,
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

    ``overnight_returns`` (``execution_timing="next_bar_open"`` only):
    every row's ``close[T-1] -> open[T]`` leg is applied to ``dollar``/
    ``equity`` FIRST, before any decision this row is made, so ``weight_
    prev`` (the reference every anchor/turnover-cap/compliance check
    below uses) reflects the REAL open(T) weight -- not the stale
    close(T-1) weight a rebalance, a turnover cap or a hard risk limit
    would otherwise be (mis)evaluated against for a full row. Only the
    REMAINING ``open[T] -> close[T]`` leg is then applied at the row's
    end to advance toward the next row's overnight leg. ``None``
    (default) applies the whole row's own return as one leg at the end,
    exactly as before -- a strict no-op.
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
    if overnight_returns is not None:
        # Same defensive rigor as asset_returns above -- a directly-
        # callable public function, never reached only through
        # run_accounting's own up-front validation.
        overnight_returns = _validate_overnight_returns(
            overnight_returns, executed, reference_name="executed"
        )
        overnight_values = overnight_returns.to_numpy(dtype=float)
        # A position held into this row's overnight leg -- either already
        # held coming out of the PREVIOUS row, or freshly targeted THIS
        # row -- needs a real overnight return; row 0 is exempt (no prior
        # close to have drifted from, mirroring asset_returns' own row-0
        # exemption above).
        prev_executed_np = np.vstack(
            [np.zeros((1, executed.shape[1])), executed.to_numpy(dtype=float)[:-1]]
        )
        overnight_needed = (np.abs(prev_executed_np) > EPSILON) | (
            np.abs(executed.to_numpy(dtype=float)) > EPSILON
        )
        missing_overnight = np.isnan(overnight_values) & overnight_needed
        if missing_overnight.shape[0] > 0:
            missing_overnight[0, :] = False
        if missing_overnight.any():
            bad = np.argwhere(missing_overnight)[0]
            raise BacktestError(
                "overnight_returns is missing a return for a position held "
                f"before or after a trade: {executed.index[bad[0]]!r}/"
                f"{executed.columns[bad[1]]!r}."
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
    if overnight_returns is not None:
        overnight_arr = overnight_returns.to_numpy(dtype=float)
        # (1+overnight)*(1+intraday) = 1+asset_return -- same identity
        # _gross_returns_and_cost_basis derives intraday from, so the two
        # legs recompose exactly to the validated asset_returns.
        intraday_arr = (1.0 + returns_np) / (1.0 + overnight_arr) - 1.0
        overnight_returns_np: np.ndarray | None = overnight_arr
        intraday_returns_np: np.ndarray | None = intraday_arr
    else:
        overnight_returns_np = None
        intraday_returns_np = None
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

    def _advance_return_leg(weight_basis: np.ndarray, leg_returns: np.ndarray) -> None:
        """Advance ``dollar``/``equity`` by one return leg, in place."""
        nonlocal dollar, equity
        leg_returns = np.where(np.isfinite(leg_returns), leg_returns, 0.0)
        leg_gross = float(np.sum(weight_basis * leg_returns))
        dollar = dollar * (1.0 + leg_returns)
        equity = equity * (1.0 + leg_gross)

    def _reset_if_bankrupt(t: int, when: str) -> bool:
        """Flatten a bankrupt anchor-episode, mirroring ``ruined``'s own handling.

        Returns whether a reset happened -- the "overnight leg" call site
        needs this to also force THIS row's own output flat (see the main
        loop), since resetting the internal ``dollar``/``equity`` state
        alone does not, by itself, stop the row's own freshly-decided
        target from being landed against that just-reset (fictitious)
        baseline as if real capital existed to fund it.
        """
        nonlocal dollar, equity, ordinary_mask, compliance_pending
        if equity > EPSILON:
            return False
        dollar = np.zeros(n_cols)
        equity = 1.0
        ordinary_mask = np.zeros(n_cols, dtype=bool)
        compliance_pending = False
        logger.warning(
            "Weight-drift equity (relative to its own last anchor, "
            "gross/pre-cost) reached zero or below at %s (%s) -- "
            "flattening this episode's drifted positions. Check leverage.",
            executed.index[t],
            when,
        )
        return True

    previous_row: np.ndarray | None = None

    for t in range(n_rows):
        row = executed_np[t]
        row_tradable = tradable_np[t]

        overnight_bankrupt = False
        if overnight_returns_np is not None:
            # `dollar/equity` still hold the state finalized at the END of
            # row (t-1)'s own advance -- CLOSE(t-1). Apply the overnight
            # leg FIRST, before any decision below, so `weight_prev`
            # reflects the REAL open(t) weight rather than the stale
            # close(t-1) one every anchor/turnover-cap/compliance check
            # would otherwise be evaluated against for a full row (see
            # this function's own docstring).
            weight_close_prev = (
                dollar / equity if equity > EPSILON else np.zeros(n_cols)
            )
            _advance_return_leg(weight_close_prev, overnight_returns_np[t])
            overnight_bankrupt = _reset_if_bankrupt(t, "overnight leg")

        if overnight_bankrupt:
            # No real capital exists at the open to fund ANY position
            # this row, including the row's own freshly-decided target --
            # `_reset_if_bankrupt` only resets the internal `dollar`/
            # `equity` STATE (so a later row starts from a clean, fresh
            # baseline); it does not, by itself, stop THIS row's own
            # decision from being landed against that just-reset
            # fictitious equity=1.0 baseline as if real capital existed
            # to fund it. Forcing `out[t]` flat here too (not just for
            # every row STRICTLY AFTER this one, which `run_accounting`'s
            # own `ruined` post-processing already handles) keeps
            # `executed_weights`/`weight_changes` from ever reporting a
            # position or a trade that no real capital could have funded.
            # `previous_row` is reset to all-zero (not `row`) so the NEXT
            # row's own fresh-decision detection correctly treats
            # whatever is decided then as a genuinely new entry against
            # the just-reset baseline, rather than silently dropping it
            # for happening to equal this row's own never-funded target.
            landed = np.zeros(n_cols)
            out[t] = landed
            trade_changes[t] = 0.0
            previous_row = np.zeros(n_cols)
            if intraday_returns_np is not None:
                _advance_return_leg(out[t], intraday_returns_np[t])
                _reset_if_bankrupt(t, "intraday leg")
            else:
                _advance_return_leg(out[t], returns_np[t])
                _reset_if_bankrupt(t, "return")
            continue

        if previous_row is None:
            fresh = np.ones(n_cols, dtype=bool)
        else:
            fresh = np.abs(row - previous_row) > EPSILON
        if rebalance_date_np is not None:
            fresh = fresh | rebalance_date_np[t]
        previous_row = row

        # `dollar/equity` now hold the weight HELD GOING INTO row t's own
        # trade decision -- open(t) when overnight_returns is active,
        # else still close(t-1) (unchanged). Every subsequent step below
        # measures a real trade against this PRE-transaction value.
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

        # === Advance state using row t's REMAINING return, with out[t]
        # (this row's just-finalized output) as the base weight.
        # `dollar/equity` already equal `out[t]` here. When overnight_
        # returns is active, the overnight leg was already consumed at
        # the top of THIS iteration, so only the intraday leg remains --
        # otherwise (unchanged) the whole row's own return is one leg.
        # The result becomes `weight_close_prev`/`weight_prev` -- and so,
        # by default, `out[t+1]` -- for the next row. ===
        if intraday_returns_np is not None:
            _advance_return_leg(out[t], intraday_returns_np[t])
            _reset_if_bankrupt(t, "intraday leg")
        else:
            _advance_return_leg(out[t], returns_np[t])
            _reset_if_bankrupt(t, "return")

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
    overnight_returns: pd.DataFrame | None = None,
) -> AccountingResult:
    """Run the vectorised accounting loop.

    Args:
        held_weights: Target weights actually held (step function after
            rebalancing), ``dates × symbols``.
        asset_returns: Per-asset simple returns aligned to ``held_weights``.
        execution_model: Cost model.
        initial_capital: Starting equity.
        overnight_returns: ``close[T-1] -> open[T]`` per-asset simple
            returns, required only when ``config.execution.execution_timing
            == "next_bar_open"`` -- *not* simply "!= same_bar_close":
            ``"immediate_after_close"`` also leaves this ``None``, since a
            continuous 24/7 market has no session gap to decompose (see
            :func:`quantlab.backtesting.engine.BacktestEngine.run`, which
            derives it via :mod:`quantlab.backtesting.execution_prices`
            only for ``"next_bar_open"``).
            ``None`` (default) is a strict no-op: every gross-return cell
            uses today's plain ``executed * asset_returns`` formula,
            unchanged. When given, every date's return is decomposed
            portfolio-wide into an overnight leg (pre-trade holdings) and
            an intraday leg (post-trade holdings), compounded -- see
            :func:`_gross_returns_and_cost_basis` for the exact formula
            and why it is portfolio-wide, not limited to cells that traded
            that day. Must COVER every ``held_weights`` date and symbol --
            NOT required to already be exactly aligned like
            ``asset_returns`` is: a caller may pass a wider, unsliced
            frame (e.g. the full available history) and this reindexes
            it down internally, which simplifies callers that would
            otherwise need to slice per fold/window (see
            :mod:`quantlab.validation.walk_forward`'s own call sites).
            ``Infinity`` and any FINITE value at or below -1.0 (an exact
            -100% overnight leg is rejected too, not just values below
            it: it would divide by zero deriving the intraday leg) are
            rejected globally, over every reindexed cell, regardless of
            whether it is ever held or traded -- weight drift has not
            run yet at this point, so there is no held/traded mask
            available yet to scope either check down to. ``NaN`` is
            different: it is tolerated HERE unconditionally (never
            itself finite, so neither check above ever applies to it),
            but only genuinely permitted where no pre- or post-trade
            position actually needs that cell's return -- a separate,
            held/traded-scoped check further down (see
            ``missing_overnight`` in this function's own body) raises an
            explicit error the moment a real position depends on a
            missing value, rather than at this upfront, unscoped point.
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

    if overnight_returns is not None:
        overnight_returns = _validate_overnight_returns(
            overnight_returns, held, reference_name="held_weights"
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
            overnight_returns=overnight_returns,
        )

    result = _solve_accounting(
        executed,
        asset_returns,
        execution_model,
        capital,
        weight_changes_override=weight_changes_override,
        overnight_returns=overnight_returns,
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
            overnight_returns=overnight_returns,
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
            overnight_returns=overnight_returns,
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
            overnight_returns=overnight_returns,
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
