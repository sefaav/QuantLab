"""Trailing volatility and dependence estimators."""

from __future__ import annotations

import math
from typing import TypeVar, cast

import pandas as pd

from quantlab.constants import TRADING_DAYS_PER_YEAR
from quantlab.features._validation import (
    boolean,
    finite_real,
    numeric_pandas,
    positive_int,
    same_axes,
)

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def historical_volatility(
    returns: PandasT,
    window: int,
    *,
    annualize: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PandasT:
    """Return the trailing sample standard deviation of period returns."""
    validated = numeric_pandas(returns, name="returns")
    length = positive_int(window, name="window", minimum=2)
    use_annualization = boolean(annualize, name="annualize")
    ppy = positive_int(periods_per_year, name="periods_per_year")
    vol = validated.rolling(length, min_periods=length).std(ddof=1)
    if use_annualization:
        vol = vol * math.sqrt(ppy)
    return vol


def realized_volatility(
    returns: PandasT,
    window: int = 63,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PandasT:
    """Return annualised trailing historical volatility."""
    return historical_volatility(
        returns, window, annualize=True, periods_per_year=periods_per_year
    )


def ewma_volatility(
    returns: PandasT,
    *,
    halflife: float = 21.0,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PandasT:
    """Return exponentially weighted volatility with a finite warm-up."""
    validated = numeric_pandas(returns, name="returns")
    decay = finite_real(halflife, name="halflife", minimum=0.0, strict=True)
    use_annualization = boolean(annualize, name="annualize")
    ppy = positive_int(periods_per_year, name="periods_per_year")
    min_periods = max(2, math.ceil(decay))
    vol = validated.ewm(halflife=decay, min_periods=min_periods, adjust=False).std(
        bias=False
    )
    if use_annualization:
        vol = vol * math.sqrt(ppy)
    return vol


def downside_volatility(
    returns: PandasT,
    window: int,
    *,
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PandasT:
    """Return the rolling root-mean-square shortfall below ``threshold``.

    Each downside observation is measured as ``return - threshold``. Returns
    at or above the threshold contribute zero; missing returns remain missing.
    """
    validated = numeric_pandas(returns, name="returns")
    length = positive_int(window, name="window")
    target = finite_real(threshold, name="threshold")
    use_annualization = boolean(annualize, name="annualize")
    ppy = positive_int(periods_per_year, name="periods_per_year")
    shortfall = (validated - target).clip(upper=0.0)
    vol = (shortfall**2).rolling(length, min_periods=length).mean().pow(0.5)
    if use_annualization:
        vol = vol * math.sqrt(ppy)
    return vol


def average_true_range(
    high: PandasT, low: PandasT, close: PandasT, window: int = 14
) -> PandasT:
    """Return Wilder-style Average True Range.

    True range is ``max(high-low, |high-prev_close|, |low-prev_close|)``. Its
    first observation uses ``high-low`` because no previous close exists. ATR
    then applies Wilder's exponential smoothing after a full-window warm-up.
    """
    validated_high = numeric_pandas(high, name="high", strictly_positive=True)
    validated_low = numeric_pandas(low, name="low", strictly_positive=True)
    validated_close = numeric_pandas(close, name="close", strictly_positive=True)
    same_axes(
        validated_high,
        validated_low,
        validated_close,
        names=("low", "close"),
    )
    length = positive_int(window, name="window")
    if (
        (
            (validated_high < validated_low)
            & validated_high.notna()
            & validated_low.notna()
        )
        .to_numpy()
        .any()
    ):
        raise ValueError("high must be greater than or equal to low where defined.")
    complete = validated_high.notna() & validated_low.notna() & validated_close.notna()
    if (
        (
            ((validated_close > validated_high) | (validated_close < validated_low))
            & complete
        )
        .to_numpy()
        .any()
    ):
        raise ValueError("close must lie between low and high where all are defined.")

    if isinstance(validated_high, pd.Series):
        if not isinstance(validated_low, pd.Series) or not isinstance(
            validated_close, pd.Series
        ):
            raise TypeError("ATR inputs must use the same pandas container type.")
        result = _average_true_range_series(
            validated_high, validated_low, validated_close, length
        )
    else:
        if not isinstance(validated_low, pd.DataFrame) or not isinstance(
            validated_close, pd.DataFrame
        ):
            raise TypeError("ATR inputs must use the same pandas container type.")
        result = _average_true_range_frame(
            validated_high, validated_low, validated_close, length
        )
    return cast(PandasT, result)  # type: ignore[redundant-cast]


def _average_true_range_series(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = (high - low).abs()
    high_gap = (high - prev_close).abs()
    low_gap = (low - prev_close).abs()
    true_range = true_range.where(high_gap.isna() | (true_range >= high_gap), high_gap)
    true_range = true_range.where(low_gap.isna() | (true_range >= low_gap), low_gap)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def _average_true_range_frame(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int
) -> pd.DataFrame:
    prev_close = close.shift(1)
    true_range = (high - low).abs()
    high_gap = (high - prev_close).abs()
    low_gap = (low - prev_close).abs()
    true_range = true_range.where(high_gap.isna() | (true_range >= high_gap), high_gap)
    true_range = true_range.where(low_gap.isna() | (true_range >= low_gap), low_gap)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def rolling_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Return trailing beta without compressing gaps in either time series."""
    if not isinstance(returns, pd.Series) or not isinstance(
        benchmark_returns, pd.Series
    ):
        raise TypeError("returns and benchmark_returns must be pandas Series.")
    asset = numeric_pandas(returns, name="returns")
    benchmark = numeric_pandas(benchmark_returns, name="benchmark_returns")
    length = positive_int(window, name="window", minimum=2)
    aligned_benchmark = benchmark.reindex(asset.index)
    cov = asset.rolling(length, min_periods=length).cov(aligned_benchmark)
    variance = aligned_benchmark.rolling(length, min_periods=length).var(ddof=1)
    return cov / variance.where(variance > 0.0)


def rolling_correlation(
    returns: pd.Series, benchmark_returns: pd.Series, window: int = 63
) -> pd.Series:
    """Return trailing Pearson correlation without compressing temporal gaps."""
    if not isinstance(returns, pd.Series) or not isinstance(
        benchmark_returns, pd.Series
    ):
        raise TypeError("returns and benchmark_returns must be pandas Series.")
    asset = numeric_pandas(returns, name="returns")
    benchmark = numeric_pandas(benchmark_returns, name="benchmark_returns")
    length = positive_int(window, name="window", minimum=2)
    aligned_benchmark = benchmark.reindex(asset.index)
    return asset.rolling(length, min_periods=length).corr(aligned_benchmark)


def annualize_volatility(
    periodic_vol: float, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Scale a non-negative single-period volatility to annual terms."""
    value = finite_real(periodic_vol, name="periodic_vol", minimum=0.0)
    ppy = positive_int(periods_per_year, name="periods_per_year")
    return value * math.sqrt(ppy)
