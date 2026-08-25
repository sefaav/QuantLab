"""Portfolio-weight and traded-notional helpers."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

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


def shift_respecting_tradability(
    frame: pd.DataFrame, periods: int, tradable: pd.DataFrame
) -> pd.DataFrame:
    """Shift each column by ``periods`` steps among its own tradable rows.

    A raw ``frame.shift(periods)`` treats every row as a valid execution
    opportunity for every symbol -- wrong whenever a symbol is untradable on
    some rows (a closed market on a mixed-calendar timeline): a decision
    made on the last date a symbol was tradable must "execute" on that
    symbol's *next real tradable opportunity*, not on the next raw row,
    which may be a date the symbol can't actually trade on at all (e.g. a
    weekend row that only exists because another, always-open instrument
    shares the same combined index). For each column, a closed row repeats
    the last value already produced for a tradable row (frozen, no
    reallocation happens while closed); a tradable row takes the value from
    ``periods`` tradable rows back for that column, never a raw row-count
    lookback.
    """
    if isinstance(periods, (bool, np.bool_)) or not isinstance(periods, Integral):
        raise BacktestError("periods must be a non-negative integer.")
    if int(periods) < 0:
        raise BacktestError("periods must be a non-negative integer.")
    if not isinstance(tradable, pd.DataFrame):
        raise BacktestError("tradable must be a pandas DataFrame.")
    if not frame.index.equals(tradable.index) or not frame.columns.equals(
        tradable.columns
    ):
        raise BacktestError(
            "tradable must have exactly the same index and columns as frame, "
            "in the same order."
        )
    non_bool_columns = [
        column for column, dtype in tradable.dtypes.items() if not is_bool_dtype(dtype)
    ]
    if non_bool_columns:
        raise BacktestError(
            f"tradable must contain only boolean values; column(s) "
            f"{non_bool_columns} are not boolean dtype (e.g. a string "
            "'False' would otherwise silently coerce to True)."
        )
    result = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        mask = tradable[column].to_numpy(dtype=bool)
        tradable_positions = np.flatnonzero(mask)
        # A column's first `periods` tradable positions have no tradable
        # row "behind" them to reference -- there is no prior decision,
        # so (matching the same "start flat" convention as
        # executed_weights' own row-0 override) this is 0.0, not NaN. Left
        # as NaN, the ffill below would propagate NaN through every row
        # before the column's first real shifted value -- e.g. a symbol
        # closed on the very first rows of a mixed-calendar history that
        # opens on, say, a Friday -- and executed_weights' row-0-only
        # override can't reach those later rows to fix them.
        shifted_at_tradable = np.zeros(len(tradable_positions))
        if len(tradable_positions) > periods:
            source = tradable_positions[: len(tradable_positions) - periods]
            shifted_at_tradable[periods:] = values[source]
        full = np.full(len(frame.index), np.nan)
        full[tradable_positions] = shifted_at_tradable
        column_result = pd.Series(full, index=frame.index)
        # Same "no prior decision -> flat" convention for any row before a
        # column's first tradable one (a symbol closed on the very first
        # row(s) of the whole history): ffill alone cannot reach *leading*
        # NaN, since there is nothing earlier to carry forward.
        result[column] = column_result.ffill().fillna(0.0)
    return result


def executed_weights(
    held_weights: pd.DataFrame, *, tradable: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Shift decisions by one period so period-t returns use t-1 weights.

    When ``tradable`` is given, the shift is per-symbol tradability-aware
    (see :func:`shift_respecting_tradability`): a symbol's decision only
    appears in the returned frame on its own next tradable row, not the raw
    next row -- a decision made right before a closure (e.g. Friday, before
    a weekend) is never misattributed as trading during the closure itself.
    """
    held = validate_execution_frame(held_weights, name="held_weights")
    if tradable is not None:
        executed = shift_respecting_tradability(held, 1, tradable)
    else:
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
