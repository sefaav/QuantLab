"""Vectorised portfolio accounting.

Weights are shifted by one period before returns are applied, so a decision
made at t can only earn from t+1. Turnover is the L1 change in executed
weights. Costs are fractions of equity; volume-based slippage is solved
against prior net equity by :func:`_solve_accounting`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionCosts, ExecutionModel
from quantlab.execution.orders import executed_weights as compute_executed_weights
from quantlab.execution.orders import weight_changes as compute_weight_changes
from quantlab.logging_config import get_logger
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
    """
    # Turnover is the L1 change in the executed book.
    weight_changes = compute_weight_changes(executed)
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
    )


def _solve_accounting(
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    execution_model: ExecutionModel,
    initial_capital: float,
    *,
    force_flat: pd.Series | None = None,
) -> AccountingResult:
    """Solve equity-dependent costs to a self-consistent equity curve.

    Volume impact at t is sized from equity at t-1, which depends on earlier
    costs. Repeated vectorised passes feed the latest net-equity curve back
    into the cost model until the maximum difference reaches the tolerance.
    """
    result = _run_accounting_steps(
        executed, asset_returns, execution_model, initial_capital, force_flat=force_flat
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


def run_accounting(
    held_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    execution_model: ExecutionModel,
    initial_capital: float,
    *,
    tradable: pd.DataFrame | None = None,
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
            same combined index).

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
        # Exact same *set* of dates and symbols, no missing values -- unlike
        # asset_returns (which may legitimately come from a wider price
        # matrix), tradable is only ever built internally from held_weights'
        # own (date, symbol) grid, never user input. A mismatched set always
        # means an upstream wiring bug, so it must raise loudly rather than
        # silently default an unrecognized cell to "tradable" and risk
        # trading a symbol that should have stayed closed. Axis *order*
        # alone is not a mismatch: a caller may build tradable from a
        # declared symbol list while held_weights comes from an
        # alphabetically-pivoted price matrix.
        if set(tradable.index) != set(held_weights.index) or set(
            tradable.columns
        ) != set(held_weights.columns):
            raise BacktestError(
                "tradable must have the same dates and symbols as held_weights."
            )
        if tradable.isna().to_numpy().any():
            raise BacktestError("tradable must not contain missing values.")

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

    result = _solve_accounting(executed, asset_returns, execution_model, capital)

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
        )

    logger.info(
        "Accounting: %d periods, final equity %.2f (gross %.2f), avg turnover %.4f",
        len(result.equity),
        float(result.equity.iloc[-1]) if len(result.equity) else capital,
        float(result.gross_equity.iloc[-1]) if len(result.gross_equity) else capital,
        float(result.turnover.mean()) if len(result.turnover) else 0.0,
    )
    return result
