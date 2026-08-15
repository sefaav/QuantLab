"""Build synthetic fills from executed portfolio-weight changes.

Each row records a non-zero change, its estimated notional and modelled costs.
It does not simulate share quantities or partial order execution.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from quantlab.constants import BPS_TO_FRACTION, EPSILON
from quantlab.exceptions import BacktestError
from quantlab.execution.orders import (
    equity_before_period,
    traded_notional,
    validate_execution_frame,
)
from quantlab.execution.slippage import (
    SlippageModel,
    validate_slippage_cost_frame,
)

#: Column order of the trade log.
TRADE_LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "previous_weight",
    "new_weight",
    "weight_change",
    "side",
    "reference_price",
    "traded_notional",
    "commission",
    "spread_cost",
    "slippage_cost",
    "total_cost",
]


def _non_negative_rate(value: object, name: str) -> float:
    """Return a finite, non-negative numeric rate or raise ``BacktestError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise BacktestError(f"{name} must be a finite non-negative number.")
    rate = float(value)
    if not np.isfinite(rate) or rate < 0.0:
        raise BacktestError(f"{name} must be a finite non-negative number.")
    return rate


def _validate_unique_axes(frame: pd.DataFrame, name: str) -> None:
    """Reject duplicate labels that would make scalar fills ambiguous."""
    if not frame.index.is_unique:
        raise BacktestError(f"{name} index must not contain duplicate labels.")
    if not frame.columns.is_unique:
        raise BacktestError(f"{name} columns must not contain duplicate labels.")


def build_trade_log(
    executed_weights: pd.DataFrame,
    weight_changes: pd.DataFrame,
    equity: pd.Series,
    reference_prices: pd.DataFrame,
    *,
    commission_bps: float,
    spread_bps: float,
    slippage_model: SlippageModel,
    slippage_equity: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the trade log from executed weight changes.

    Args:
        executed_weights: Weights actually in force each period.
        weight_changes: Per-symbol change in the executed book.
        equity: Net equity curve (used to size notional off prior equity).
        reference_prices: Price matrix for the reference price of each fill.
        commission_bps: Commission in bps.
        spread_bps: Full spread in bps (half applied).
        slippage_model: The same per-symbol model used by accounting.
        slippage_equity: Per-date equity passed to the slippage model's
            ``equity`` argument. Pass ``AccountingResult.equity_for_costs``
            to reproduce volume-based accounting costs.

    Returns:
        A DataFrame with :data:`TRADE_LOG_COLUMNS`, one row per non-zero fill.
    """
    if not isinstance(executed_weights, pd.DataFrame):
        raise BacktestError("executed_weights must be a pandas DataFrame.")
    if not isinstance(weight_changes, pd.DataFrame):
        raise BacktestError("weight_changes must be a pandas DataFrame.")
    if not isinstance(reference_prices, pd.DataFrame):
        raise BacktestError("reference_prices must be a pandas DataFrame.")
    if not isinstance(equity, pd.Series):
        raise BacktestError("equity must be a pandas Series.")
    if slippage_equity is not None and not isinstance(slippage_equity, pd.Series):
        raise BacktestError("slippage_equity must be a pandas Series.")
    if not isinstance(slippage_model, SlippageModel):
        raise BacktestError("slippage_model must implement SlippageModel.")

    commission_rate = _non_negative_rate(commission_bps, "commission_bps")
    spread_rate = _non_negative_rate(spread_bps, "spread_bps")

    for name, frame in (
        ("executed_weights", executed_weights),
        ("weight_changes", weight_changes),
        ("reference_prices", reference_prices),
    ):
        _validate_unique_axes(frame, name)
    if not equity.index.is_unique:
        raise BacktestError("equity index must not contain duplicate labels.")
    if slippage_equity is not None and not slippage_equity.index.is_unique:
        raise BacktestError("slippage_equity index must not contain duplicate labels.")
    if not weight_changes.index.is_monotonic_increasing:
        raise BacktestError("weight_changes index must be sorted in increasing order.")
    if not executed_weights.index.equals(weight_changes.index) or not (
        executed_weights.columns.equals(weight_changes.columns)
    ):
        raise BacktestError(
            "executed_weights and weight_changes must have identical axes."
        )

    if weight_changes.empty:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    executed = validate_execution_frame(executed_weights, name="executed_weights")
    changes = validate_execution_frame(weight_changes, name="weight_changes")
    executed_values = executed.to_numpy()
    change_values = changes.to_numpy()
    previous_values = np.vstack(
        [np.zeros((1, executed_values.shape[1])), executed_values[:-1]]
    )

    previous_equity_series = equity_before_period(equity, changes.index)
    previous_equity = previous_equity_series.to_numpy()
    notional_values = traded_notional(changes, equity).to_numpy()

    slippage_ref = slippage_equity if slippage_equity is not None else equity
    previous_slippage_series = equity_before_period(
        slippage_ref,
        changes.index,
        name="slippage_equity",
    )

    slippage_fraction = slippage_model.per_symbol_cost(
        changes, previous_slippage_series
    )
    slippage_values = validate_slippage_cost_frame(
        slippage_fraction, changes
    ).to_numpy()
    slippage_currency = slippage_values * previous_equity[:, None]

    reference_aligned = reference_prices.reindex(
        index=weight_changes.index, columns=weight_changes.columns
    )
    try:
        reference_values = reference_aligned.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("reference_prices must contain numeric values.") from exc
    lagged_reference_values = np.vstack(
        [np.full((1, reference_values.shape[1]), np.nan), reference_values[:-1]]
    )

    records: list[dict[str, object]] = []
    # Iterate only over non-zero changes to keep the log compact.
    nonzero = np.abs(change_values) > EPSILON
    for row_number, timestamp in enumerate(weight_changes.index):
        changed_columns = np.flatnonzero(nonzero[row_number])
        if not len(changed_columns):
            continue
        for column_number in changed_columns:
            column_index = int(column_number)
            symbol = weight_changes.columns[column_index]
            delta = float(change_values[row_number, column_index])
            notional = float(notional_values[row_number, column_index])
            commission = notional * commission_rate * BPS_TO_FRACTION
            spread = notional * spread_rate * BPS_TO_FRACTION / 2.0
            slippage = float(slippage_currency[row_number, column_index])
            price = float(lagged_reference_values[row_number, column_index])
            if not np.isfinite(price) or price <= 0.0:
                raise BacktestError(
                    "A positive finite prior-period reference price is required "
                    f"for {symbol!r} on {timestamp!r}."
                )
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "previous_weight": float(previous_values[row_number, column_index]),
                    "new_weight": float(executed_values[row_number, column_index]),
                    "weight_change": delta,
                    "side": "buy" if delta > 0 else "sell",
                    "reference_price": price,
                    "traded_notional": notional,
                    "commission": commission,
                    "spread_cost": spread,
                    "slippage_cost": slippage,
                    "total_cost": commission + spread + slippage,
                }
            )
    if not records:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    return pd.DataFrame.from_records(records)[TRADE_LOG_COLUMNS]
