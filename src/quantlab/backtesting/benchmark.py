"""Construct benchmark returns aligned to a portfolio timeline."""

from __future__ import annotations

import math
from numbers import Integral, Real

import pandas as pd

from quantlab.constants import TRADING_DAYS_PER_YEAR
from quantlab.data.base import price_matrix
from quantlab.exceptions import BacktestError
from quantlab.features.returns import simple_returns

_BENCHMARK_KINDS = frozenset({"symbol", "equal_weight", "cash", "first_asset"})


def buy_and_hold_returns(data: pd.DataFrame, symbol: str) -> pd.Series:
    """Return series for buying and holding a single symbol."""
    prices = price_matrix(data, adjusted=True)
    if symbol not in prices.columns:
        raise KeyError(f"Benchmark symbol '{symbol}' not in data.")
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


def _align_returns(series: pd.Series, portfolio_index: pd.DatetimeIndex) -> pd.Series:
    """Align returns, allowing only the expected first-period missing value."""
    aligned = series.reindex(portfolio_index)
    if aligned.empty:
        return aligned
    aligned = aligned.copy()
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
    return _align_returns(series, portfolio_index)
