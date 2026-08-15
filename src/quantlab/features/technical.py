"""Trailing technical indicators."""

from __future__ import annotations

from typing import TypeVar, cast

import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features._validation import numeric_pandas, positive_int

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def exponential_moving_average(prices: PandasT, span: int) -> PandasT:
    """Return an exponential moving average after a full-span warm-up."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    valid_span = positive_int(span, name="span")
    return validated.ewm(span=valid_span, min_periods=valid_span, adjust=False).mean()


def macd(
    prices: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Return MACD, its signal line and their difference.

    The signal line starts only after ``signal_span`` defined MACD values.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    fast_n = positive_int(fast_span, name="fast_span")
    slow_n = positive_int(slow_span, name="slow_span")
    signal_n = positive_int(signal_span, name="signal_span")
    if fast_n >= slow_n:
        raise ValueError("fast_span must be smaller than slow_span.")

    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    fast = exponential_moving_average(validated, fast_n)
    slow = exponential_moving_average(validated, slow_n)
    macd_line = fast - slow
    signal = macd_line.ewm(span=signal_n, min_periods=signal_n, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal, "histogram": macd_line - signal}
    )


def rolling_max(prices: PandasT, window: int) -> PandasT:
    """Return the trailing price maximum after a full-window warm-up."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    valid_window = positive_int(window, name="window")
    return validated.rolling(valid_window, min_periods=valid_window).max()


def rolling_min(prices: PandasT, window: int) -> PandasT:
    """Return the trailing price minimum after a full-window warm-up."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    valid_window = positive_int(window, name="window")
    return validated.rolling(valid_window, min_periods=valid_window).min()


def donchian_position(prices: PandasT, window: int) -> PandasT:
    """Return the price position in its trailing channel.

    Defined values lie in ``[0, 1]``. A flat channel has neutral position
    ``0.5`` rather than an undefined division by zero.
    """
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    valid_window = positive_int(window, name="window")
    hi = validated.rolling(valid_window, min_periods=valid_window).max()
    lo = validated.rolling(valid_window, min_periods=valid_window).min()
    width = hi - lo
    if isinstance(validated, pd.Series):
        if not isinstance(lo, pd.Series) or not isinstance(width, pd.Series):
            raise TypeError("Donchian inputs produced incompatible pandas objects.")
        position = (validated - lo) / width.where(width.abs() > EPSILON)
        result = position.mask(width.abs() <= EPSILON, 0.5)
    else:
        if not isinstance(lo, pd.DataFrame) or not isinstance(width, pd.DataFrame):
            raise TypeError("Donchian inputs produced incompatible pandas objects.")
        position = (validated - lo) / width.where(width.abs() > EPSILON)
        result = position.mask(width.abs() <= EPSILON, 0.5)
    return cast(PandasT, result)  # type: ignore[redundant-cast]
