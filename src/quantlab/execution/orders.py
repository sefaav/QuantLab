"""Portfolio-weight and traded-notional helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.exceptions import BacktestError


def validate_execution_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return a finite float execution matrix with unambiguous ordered axes."""
    if not isinstance(frame, pd.DataFrame):
        raise BacktestError(f"{name} must be a pandas DataFrame.")
    if not frame.index.is_unique:
        raise BacktestError(f"{name} index must not contain duplicate labels.")
    if not frame.columns.is_unique:
        raise BacktestError(f"{name} columns must not contain duplicate labels.")
    if not frame.index.is_monotonic_increasing:
        raise BacktestError(f"{name} index must be sorted in increasing order.")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError(f"{name} must contain only numeric values.") from exc
    if not np.isfinite(values).all():
        raise BacktestError(f"{name} must contain only finite values.")
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def validate_equity_series(
    equity: pd.Series,
    index: pd.Index,
    *,
    name: str = "equity",
) -> pd.Series:
    """Return finite non-negative equity aligned exactly to ``index``."""
    if not isinstance(equity, pd.Series):
        raise BacktestError(f"{name} must be a pandas Series.")
    if not equity.index.is_unique:
        raise BacktestError(f"{name} index must not contain duplicate labels.")
    if not equity.index.equals(index):
        raise BacktestError(f"{name} must have exactly the execution-date index.")
    try:
        values = equity.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError(f"{name} must contain only numeric values.") from exc
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise BacktestError(f"{name} must contain only finite non-negative values.")
    return pd.Series(values, index=index, name=equity.name)


def equity_before_period(
    equity: pd.Series,
    index: pd.Index,
    *,
    name: str = "equity",
) -> pd.Series:
    """Equity available before each period, using the first value initially."""
    validated = validate_equity_series(equity, index, name=name)
    previous = validated.shift(1)
    if len(previous):
        previous.iloc[0] = validated.iloc[0]
    return previous


def executed_weights(held_weights: pd.DataFrame) -> pd.DataFrame:
    """Shift decisions by one period so period-t returns use t-1 weights."""
    held = validate_execution_frame(held_weights, name="held_weights")
    executed = held.shift(1)
    if len(executed):
        executed.iloc[0, :] = 0.0
    return executed


def weight_changes(executed: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol executed-book changes, with a zero starting position."""
    validated = validate_execution_frame(executed, name="executed_weights")
    previous = validated.shift(1)
    if len(previous):
        previous.iloc[0, :] = 0.0
    return validated - previous


def traded_notional(
    weight_changes_frame: pd.DataFrame, equity: pd.Series
) -> pd.DataFrame:
    """Absolute traded notional per date and symbol."""
    changes = validate_execution_frame(weight_changes_frame, name="weight_changes")
    previous_equity = equity_before_period(equity, changes.index)
    return changes.abs().mul(previous_equity, axis=0)
