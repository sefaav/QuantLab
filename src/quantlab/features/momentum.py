"""Momentum features.

Every function is trailing: the value at date *t* uses only prices at or before
*t*. The optional ``skip`` excludes the most recent periods:

    momentum_t = P_{t-skip} / P_{t-lookback} - 1
"""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import pandas as pd

from quantlab.features._validation import (
    non_negative_int,
    numeric_pandas,
    positive_int,
)
from quantlab.features.returns import simple_returns
from quantlab.features.volatility import realized_volatility

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def momentum(prices: PandasT, lookback_period: int, skip_period: int = 0) -> PandasT:
    """Momentum score with an optional recent-period exclusion.

    Args:
        prices: Price level(s).
        lookback_period: Total look-back window in periods.
        skip_period: Most recent periods to skip (e.g. 21 to drop the last
            month). Must satisfy ``0 <= skip_period < lookback_period``.

    Returns:
        ``P_{t-skip} / P_{t-lookback} - 1``.
    """
    lookback = positive_int(lookback_period, name="lookback_period")
    skip = non_negative_int(skip_period, name="skip_period")
    if skip >= lookback:
        raise ValueError(
            f"Require skip_period ({skip}) < lookback_period ({lookback})."
        )
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    return validated.shift(skip) / validated.shift(lookback) - 1.0


def rate_of_change(prices: PandasT, periods: int) -> PandasT:
    """Rate of change over ``periods`` — momentum with no skip."""
    return simple_returns(prices, positive_int(periods, name="periods"))


def moving_average(prices: PandasT, window: int) -> PandasT:
    """Simple moving average over a trailing ``window``."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    length = positive_int(window, name="window")
    return validated.rolling(length, min_periods=length).mean()


def ma_crossover_signal(prices: PandasT, fast_window: int, slow_window: int) -> PandasT:
    """Fast/slow moving-average crossover indicator.

    Returns ``+1`` above, ``-1`` below, ``0`` when the MAs are equal and ``NaN``
    during warm-up. The MAs are trailing.
    """
    fast_length = positive_int(fast_window, name="fast_window")
    slow_length = positive_int(slow_window, name="slow_window")
    if fast_length >= slow_length:
        raise ValueError(
            f"fast_window ({fast_length}) must be < slow_window ({slow_length})."
        )
    fast = moving_average(prices, fast_length)
    slow = moving_average(prices, slow_length)
    diff = fast - slow
    # NumPy preserves the pandas container through ``__array_ufunc__``.
    return cast(PandasT, np.sign(diff))


def price_above_ma(prices: PandasT, window: int) -> PandasT:
    """Trend sign: ``+1`` above, ``-1`` below, ``0`` equal, ``NaN`` warm-up."""
    ma = moving_average(prices, window)
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    return cast(PandasT, np.sign(validated - ma))


def volatility_adjusted_momentum(
    prices: PandasT,
    lookback_period: int,
    skip_period: int = 0,
    volatility_window: int = 63,
    periods_per_year: int = 252,
) -> PandasT:
    """Momentum scaled by trailing annualised volatility.

    Dividing the raw momentum by realised volatility puts assets of different
    riskiness on a comparable scale (a crude risk-adjusted signal).
    """
    volatility_length = positive_int(
        volatility_window, name="volatility_window", minimum=2
    )
    annual_periods = positive_int(periods_per_year, name="periods_per_year")
    raw = momentum(prices, lookback_period, skip_period)
    rets = simple_returns(prices)
    vol = realized_volatility(
        rets, window=volatility_length, periods_per_year=annual_periods
    )
    if isinstance(raw, pd.Series):
        if not isinstance(vol, pd.Series):
            raise TypeError("Momentum and volatility have incompatible pandas types.")
        result = raw / vol.where(vol > 0.0, np.nan)
    else:
        if not isinstance(vol, pd.DataFrame):
            raise TypeError("Momentum and volatility have incompatible pandas types.")
        result = raw / vol.where(vol > 0.0, np.nan)
    return cast(PandasT, result)  # type: ignore[redundant-cast]
