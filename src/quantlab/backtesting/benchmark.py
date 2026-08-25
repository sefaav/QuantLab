"""Construct benchmark returns aligned to a portfolio timeline."""

from __future__ import annotations

import math
from numbers import Integral, Real

import pandas as pd

from quantlab.constants import SYMBOL, TRADING_DAYS_PER_YEAR
from quantlab.data.base import price_matrix
from quantlab.data.calendar import is_session_day
from quantlab.exceptions import BacktestError
from quantlab.features.returns import simple_returns

_BENCHMARK_KINDS = frozenset({"symbol", "equal_weight", "cash", "first_asset"})


def buy_and_hold_returns(data: pd.DataFrame, symbol: str) -> pd.Series:
    """Return series for buying and holding a single symbol.

    Pivots only ``symbol``'s own rows, never the whole (possibly
    multi-symbol) ``data`` frame: pivoting together with other symbols on a
    wider or differently-shaped calendar (e.g. an external benchmark sharing
    a frame with an already closure-filled, 24/7-inclusive tradable
    universe) would reindex gaps into ``symbol``'s own dense series that it
    never actually had, which then cascade into ``simple_returns`` (NaN
    divided by NaN) on real trading days adjacent to those gaps.
    """
    symbol_rows = data.loc[data[SYMBOL] == symbol]
    if symbol_rows.empty:
        raise KeyError(f"Benchmark symbol '{symbol}' not in data.")
    prices = price_matrix(symbol_rows, adjusted=True)
    return simple_returns(prices[symbol])


def equal_weight_returns(data: pd.DataFrame) -> pd.Series:
    """Return series of an equal-weight portfolio rebalanced every period."""
    prices = price_matrix(data, adjusted=True)
    asset_returns = simple_returns(prices)
    # Do not silently redistribute a missing asset's weight across the assets
    # that happen to have data on that date.
    return asset_returns.mean(axis=1, skipna=False)


def cash_returns(
    index: pd.DatetimeIndex,
    risk_free_rate: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Cash returns using ``annual_rate / periods_per_year`` per period."""
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral):
        raise ValueError("periods_per_year must be a positive integer.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if isinstance(risk_free_rate, bool) or not isinstance(risk_free_rate, Real):
        raise ValueError("risk_free_rate must be a finite number.")
    if not math.isfinite(float(risk_free_rate)):
        raise ValueError("risk_free_rate must be a finite number.")
    per_period = risk_free_rate / periods_per_year
    return pd.Series(per_period, index=index)


def _align_returns(
    series: pd.Series,
    portfolio_index: pd.DatetimeIndex,
    *,
    calendar: str | None = None,
) -> pd.Series:
    """Align benchmark returns onto ``portfolio_index``.

    A point-sample ``reindex`` would be wrong whenever the benchmark's own
    calendar differs from the portfolio's: it silently drops any benchmark
    session that falls *between* two portfolio dates (e.g. a 24/7 benchmark's
    weekend moves get discarded entirely rather than compounded into the
    next portfolio date, materially understating or overstating its real
    return). Compounding into a cumulative equity curve first and reindexing
    that instead fixes this unconditionally: for a portfolio date that is
    also one of the benchmark's own dates, this reduces to exactly the
    benchmark's own return that period (no approximation, no calendar
    needed); for a benchmark session that falls between two portfolio dates,
    it is correctly compounded into the following portfolio date rather than
    dropped.

    Separately, any date whose price is missing (whether ``series`` never
    had a row for it at all, or a wider shared price matrix reindexed it in
    as NaN) is a *verified closure*, not missing data, when the benchmark's
    own calendar has no session there (e.g. an equity benchmark reindexed
    onto a mixed-calendar portfolio's weekend rows) -- exactly like on the
    tradable side, it should read as flat (zero return), never raise. When
    ``calendar`` is given, such dates are forward-filled from the
    benchmark's last known level; a date calendar says *should* be a
    session but still has no data is a genuine gap and still raises, same
    as before (this is exactly why ``equal_weight``/``first_asset``, whose
    series are derived from already closure-filled tradable data, never
    pass a calendar here -- a missing value for them is never a calendar
    closure, always a real defect, and must always raise). Without a
    ``calendar``, any missing date raises unconditionally.
    """
    if series.empty or len(portfolio_index) == 0:
        return series.reindex(portfolio_index)
    original_index = series.index
    equity = (1.0 + series).cumprod()
    # A returns series' own first element is *always* NaN by construction
    # (pct_change has no prior value to diff against) -- a universal,
    # expected artifact, never a data defect. Left as NaN, it would divide
    # the equity curve's second pct_change by NaN and falsely flag that
    # period as missing too. Treat it as the compounding origin (1.0):
    # cumprod already computed every later value as if this were the case
    # (a leading NaN is skipped, not zeroed, by cumprod's own semantics), so
    # this only fixes position zero and changes nothing downstream.
    if pd.isna(equity.iloc[0]):
        equity.iloc[0] = 1.0
    combined_index = original_index.union(portfolio_index)
    equity_on_combined = equity.reindex(combined_index)
    if calendar is not None:
        closure = pd.Series(
            ~is_session_day(calendar, pd.DatetimeIndex(combined_index)),
            index=combined_index,
        )
        fillable = equity_on_combined.isna() & closure
        equity_on_combined = equity_on_combined.mask(
            fillable, equity_on_combined.ffill()
        )
    aligned_equity = equity_on_combined.reindex(portfolio_index)
    # `fill_method` must be pinned explicitly: older pandas releases allowed
    # by `pandas>=2.1` default `pct_change` to forward-fill a gap before
    # diffing (silently turning a genuine missing value into a 0.0 return
    # instead of leaving it NaN for the check below to catch), while pandas 3
    # never fills. Pinning `None` makes this call's behaviour identical on
    # every supported pandas version.
    aligned = aligned_equity.pct_change(fill_method=None)
    aligned.iloc[0] = 0.0
    missing = aligned.isna()
    if missing.any():
        missing_dates = list(aligned.index[missing][:5])
        raise BacktestError(
            "Benchmark returns are missing on portfolio dates "
            f"{missing_dates}. Check benchmark coverage and market calendar."
        )
    return aligned


def build_benchmark(
    data: pd.DataFrame,
    portfolio_index: pd.DatetimeIndex,
    *,
    benchmark_symbol: str | None = None,
    benchmark_calendar: str | None = None,
    first_asset_symbol: str | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    kind: str = "symbol",
) -> pd.Series | None:
    """Build benchmark returns aligned to the portfolio dates.

    Args:
        data: Canonical long OHLCV frame.
        portfolio_index: Dates to align the benchmark to.
        benchmark_symbol: External symbol to track (if ``kind='symbol'``).
        benchmark_calendar: The ``symbol`` benchmark's own calendar, used
            only to distinguish a verified closure (flat, never an error)
            from a genuine data gap (still an error) when aligning it onto
            ``portfolio_index``. Ignored for every other ``kind``, whose
            series are derived from already closure-filled tradable data.
        first_asset_symbol: Explicit universe symbol used by
            ``kind='first_asset'``.
        risk_free_rate: Annualised rate for the cash benchmark.
        periods_per_year: Annualisation factor for the cash benchmark.
        kind: ``symbol`` / ``equal_weight`` / ``cash`` / ``first_asset``.

    Returns:
        Benchmark returns reindexed to ``portfolio_index``, or ``None`` if no
        benchmark applies (e.g. ``symbol`` requested but none configured).
    """
    benchmark_kind = kind.strip().lower()
    if benchmark_kind not in _BENCHMARK_KINDS:
        raise ValueError(
            f"Unknown benchmark kind {kind!r}; expected one of "
            f"{sorted(_BENCHMARK_KINDS)}."
        )

    calendar_for_alignment: str | None = None
    if benchmark_kind == "cash":
        series = cash_returns(portfolio_index, risk_free_rate, periods_per_year)
    elif benchmark_kind == "equal_weight":
        series = equal_weight_returns(data)
    elif benchmark_kind == "first_asset":
        if first_asset_symbol is None:
            raise ValueError(
                "first_asset_symbol is required when benchmark kind is 'first_asset'."
            )
        try:
            series = buy_and_hold_returns(data, first_asset_symbol)
        except KeyError as exc:
            raise BacktestError(
                f"First-asset benchmark symbol {first_asset_symbol!r} is absent "
                "from the loaded data."
            ) from exc
    else:
        if benchmark_symbol is None:
            return None
        try:
            series = buy_and_hold_returns(data, benchmark_symbol)
        except KeyError as exc:
            raise BacktestError(
                f"Configured benchmark symbol {benchmark_symbol!r} is absent "
                "from the loaded data."
            ) from exc
        calendar_for_alignment = benchmark_calendar
    return _align_returns(series, portfolio_index, calendar=calendar_for_alignment)
