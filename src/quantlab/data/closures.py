"""Verified-closure detection for a multi-instrument tradable universe.

Distinguishes a *verified closure* (a date that is not a trading session on a
symbol's own calendar) from a *genuinely missing* bar (a date the calendar
says should be a session, but no provider row exists for it) — only the
former is handled here. A genuinely missing bar remains a data-quality
problem governed by ``missing_value_policy``, unchanged.

Callers must restrict ``data``/``symbols`` to the *tradable* universe only
(never an external benchmark) — see :mod:`quantlab.data.loader` — so that a
benchmark on a different calendar can never inflate the portfolio's own
timeline with synthetic closure bars that no tradable instrument needs.

Closure semantics are only well-defined at daily granularity: sub-daily
session boundaries need open/close *times*, not just date membership, which
the validator's existing intraday gap tolerance already absorbs separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.data.base import pivot_field
from quantlab.data.calendar import is_session_day
from quantlab.exceptions import DataValidationError

#: insert_verified_closure_bars/tradable_mask_for are no-ops outside this.
DAILY_FREQUENCY = "1d"


def verified_closure_mask(
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    calendar_for_symbol: Mapping[str, str],
) -> pd.DataFrame:
    """Return a ``dates x symbols`` bool frame: True where verifiably closed.

    A cell is True when ``date`` is not a trading session on that symbol's
    own calendar (weekend/holiday), independent of whether a row happens to
    exist there.
    """
    columns = {
        symbol: ~is_session_day(calendar_for_symbol[symbol], dates)
        for symbol in symbols
    }
    return pd.DataFrame(columns, index=dates, columns=list(symbols))


def tradable_mask_for(
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    calendar_for_symbol: Mapping[str, str],
) -> pd.DataFrame:
    """Return a ``dates x symbols`` bool frame: True where tradable (open)."""
    return ~verified_closure_mask(dates, symbols, calendar_for_symbol)


def _drop_real_bars_on_closures(
    data: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    closure: pd.DataFrame,
    strict: bool,
    warnings: list[str] | None,
) -> pd.DataFrame:
    """Discard a real row whose own symbol's calendar was closed that day.

    A verified closure means the market did not open, so no legitimate trade
    could have produced this row -- keeping it would let a data anomaly (bad
    provider row, timezone slip, a stray weekend print) inject a fictitious
    price move into the return series, silently breaking the "a verified
    closure is never traded" guarantee. Dropping it here, before the fill
    logic below runs, makes the date fall back to an ordinary verified
    closure (flat, last known price) -- exactly as if no row had ever been
    supplied for it.
    """
    date_positions = dates.get_indexer(pd.Index(data[TIMESTAMP]))
    symbol_positions = pd.Index(symbols).get_indexer(pd.Index(data[SYMBOL]))
    covered = symbol_positions >= 0
    is_closure_row = np.zeros(len(data), dtype=bool)
    is_closure_row[covered] = closure.to_numpy()[
        date_positions[covered], symbol_positions[covered]
    ]
    if not is_closure_row.any():
        return data

    anomalous = data.loc[is_closure_row]
    examples = ", ".join(
        f"{row[SYMBOL]}@{row[TIMESTAMP].date()}"
        for _, row in anomalous.head(5).iterrows()
    )
    message = (
        f"{int(is_closure_row.sum())} row(s) fall on a verified market "
        f"closure for their own symbol's calendar (e.g. {examples}) and "
        "were discarded -- a closed market cannot produce a real trade."
    )
    if strict:
        raise DataValidationError(message)
    if warnings is not None:
        warnings.append(message)
    return data.loc[~is_closure_row].reset_index(drop=True)


def insert_verified_closure_bars(
    data: pd.DataFrame,
    *,
    symbol_calendars: Mapping[str, str],
    frequency: str,
    strict: bool = False,
    warnings: list[str] | None = None,
    counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Insert a synthetic bar for each verified closure that needs one.

    A synthetic bar is inserted for ``(date, symbol)`` only when: (a) some
    *other* symbol in ``data`` already has a real row on ``date`` (dates are
    drawn from the union of what's actually present, so this holds by
    construction — never invents a date nothing in ``data`` trades on), (b)
    ``date`` is a verified closure for ``symbol``'s own calendar, and (c)
    ``symbol`` already has at least one earlier real bar to carry forward
    (never extrapolates before a symbol's first observed bar).

    ``open``/``high``/``low``/``close`` all repeat the last known ``close``
    (a flat, zero-range bar); ``adjusted_close`` separately repeats the last
    known ``adjusted_close`` — never derived from ``close``, since the two
    can differ (splits/dividends) and each must stay flat independently so
    ``pct_change`` is exactly 0 on both series. ``volume`` is 0 explicitly
    (never forward-filled), so no strategy or volume-based slippage model can
    mistake a verified closure for real trading activity.

    A *real* row that already sits on a verified closure (a data anomaly, not
    a gap) is discarded rather than trusted -- see
    :func:`_drop_real_bars_on_closures`. In ``strict`` mode this raises
    :class:`~quantlab.exceptions.DataValidationError`; otherwise a
    description is appended to ``warnings`` (when given) and the row is
    dropped before the fill logic below runs.

    A no-op when ``frequency`` isn't daily, or when nothing needs filling.

    ``counts``, when given, is updated in place with ``"discarded"`` (real
    rows removed by :func:`_drop_real_bars_on_closures`) and ``"inserted"``
    (synthetic rows added below) as two separate non-negative numbers --
    never netted against each other, unlike a single delta-of-lengths count,
    which can go negative and misrepresent what actually happened to the
    data (see :class:`~quantlab.data.validator.DataQualityReport`).
    """
    if frequency != DAILY_FREQUENCY or data.empty:
        return data

    symbols = sorted(symbol_calendars)
    dates = pd.DatetimeIndex(sorted(data[TIMESTAMP].unique()))
    closure = verified_closure_mask(dates, symbols, symbol_calendars)

    before_drop = len(data)
    data = _drop_real_bars_on_closures(
        data,
        dates=dates,
        symbols=symbols,
        closure=closure,
        strict=strict,
        warnings=warnings,
    )
    if counts is not None:
        counts["discarded"] = before_drop - len(data)
    if data.empty:
        return data
    dates = pd.DatetimeIndex(sorted(data[TIMESTAMP].unique()))
    closure = verified_closure_mask(dates, symbols, symbol_calendars)

    close_wide = pivot_field(data, CLOSE).reindex(index=dates, columns=symbols)
    adjusted_wide = pivot_field(data, ADJUSTED_CLOSE).reindex(
        index=dates, columns=symbols
    )
    filled_close = close_wide.ffill()
    filled_adjusted = adjusted_wide.ffill()

    # Verified closure, no real row that day, and a prior real bar to carry
    # forward (ffill leaves leading NaN before a symbol's first observation).
    needs_fill = close_wide.isna() & closure & filled_close.notna()
    if not needs_fill.to_numpy().any():
        return data

    rows, columns = np.where(needs_fill.to_numpy())
    fill_dates = dates[rows]
    fill_symbols = [symbols[column] for column in columns]
    close_values = filled_close.to_numpy()[rows, columns]
    adjusted_values = filled_adjusted.to_numpy()[rows, columns]
    synthetic = pd.DataFrame(
        {
            TIMESTAMP: fill_dates,
            SYMBOL: fill_symbols,
            OPEN: close_values,
            HIGH: close_values,
            LOW: close_values,
            CLOSE: close_values,
            ADJUSTED_CLOSE: adjusted_values,
            VOLUME: 0.0,
        }
    )
    if counts is not None:
        counts["inserted"] = len(synthetic)
    combined = pd.concat([data, synthetic], ignore_index=True)
    return combined.sort_values([TIMESTAMP, SYMBOL]).reset_index(drop=True)
