"""Compute a rolling-window feature on each symbol's own native calendar.

A closure-bar-padded multi-calendar universe (see
``quantlab.data.closures``) shares one combined timeline across every
symbol, including days a given symbol's own calendar has no session for
(e.g. a 24/7 crypto instrument's weekend rows appearing alongside a
session-bound equity's own calendar). A rolling window computed directly
on that padded timeline therefore spans more real calendar days than
periods for any session-bound symbol sharing it with an always-open one,
diluting the estimate. :func:`compute_native_then_align` removes this:
each symbol is sliced to its own verified native session rows before the
feature is computed, and the result is reindexed/forward-filled back onto
the combined timeline afterward -- the same "nothing changed while closed"
convention the raw OHLCV closure padding itself already uses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from quantlab.data.calendar import is_session_day, uniform_calendar
from quantlab.data.closures import verified_closure_mask


def compute_native_then_align(
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    symbol_calendars: Mapping[str, str] | None,
    combined_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Run ``compute_fn`` per symbol on its own native calendar, then align.

    ``compute_fn`` must be a plain, calendar-agnostic rolling-window
    feature (e.g. ``momentum``, ``rolling_zscore``) that treats each of
    ``prices``'s columns independently -- it is called once per symbol on
    a single-column frame, never on the full multi-symbol matrix, so any
    function with genuine cross-column interaction is out of scope here.

    Dilution only exists when the universe genuinely spans more than one
    calendar (a symbol's own window is only stretched by a closure that
    exists BECAUSE a differently-scheduled symbol shares its timeline) --
    so ``symbol_calendars=None``, or every ``prices`` column sharing the
    exact same one calendar (even one with its own ordinary closures,
    e.g. a single-calendar equity universe's weekends/holidays -- nothing
    dilutes there, since every column is equally subject to the same
    closures and `prices.index` already reflects exactly that calendar's
    own sessions), short-circuits straight to a single vectorized
    ``compute_fn(prices)`` call. A column with no
    verified closure at all in ``prices.index`` for its own calendar
    short-circuits individually to ``compute_fn`` on its own untouched
    column.
    """
    if not symbol_calendars:
        return compute_fn(prices)
    calendars = {
        symbol: symbol_calendars[symbol]
        for symbol in prices.columns
        if symbol in symbol_calendars
    }
    if not calendars:
        return compute_fn(prices)
    if len(calendars) == len(prices.columns) and (
        uniform_calendar(calendars.values()) is not None
    ):
        return compute_fn(prices)
    closure = verified_closure_mask(
        pd.DatetimeIndex(prices.index), list(calendars), calendars
    )
    if not bool(closure.to_numpy().any()):
        return compute_fn(prices)

    columns: dict[str, pd.Series] = {}
    for symbol in prices.columns:
        calendar = calendars.get(symbol)
        if calendar is None or not bool(closure[symbol].any()):
            columns[symbol] = compute_fn(prices[[symbol]]).iloc[:, 0]
            continue
        native_index = prices.index[~closure[symbol].to_numpy()]
        native_result = compute_fn(prices.loc[native_index, [symbol]]).iloc[:, 0]
        aligned = native_result.reindex(combined_index)
        combined_closure = ~is_session_day(calendar, pd.DatetimeIndex(combined_index))
        fillable = aligned.isna() & combined_closure
        aligned = aligned.mask(fillable, aligned.ffill())
        columns[symbol] = aligned
    return pd.DataFrame(columns, index=combined_index)[list(prices.columns)]
