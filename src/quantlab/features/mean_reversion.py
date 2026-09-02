"""Trailing mean-reversion indicators and a full-sample half-life estimator."""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features._validation import finite_real, numeric_pandas, positive_int

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def rolling_zscore(series: PandasT, window: int) -> PandasT:
    """Rolling z-score ``(x - mean) / std`` over a trailing window."""
    validated = numeric_pandas(series, name="series")
    length = positive_int(window, name="window", minimum=2)
    mean = validated.rolling(length, min_periods=length).mean()
    std = validated.rolling(length, min_periods=length).std(ddof=1)
    return (validated - mean) / (std + EPSILON)


def distance_to_moving_average(prices: PandasT, window: int) -> PandasT:
    """Signed distance between price and its trailing moving average."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    length = positive_int(window, name="window")
    ma = validated.rolling(length, min_periods=length).mean()
    return validated - ma


def normalized_distance_to_mean(prices: PandasT, window: int) -> PandasT:
    """Distance to the moving average expressed as a fraction of the MA."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    length = positive_int(window, name="window")
    ma = validated.rolling(length, min_periods=length).mean()
    return (validated - ma) / ma


def rolling_percentile_rank(
    prices: PandasT, window: int, *, strictly_positive: bool = True
) -> PandasT:
    """Trailing percentile rank of the current price within its own window.

    ``[0, 1]``: ``0`` when the current price is the lowest in the trailing
    window, ``1`` when it is the highest, ``0.5`` in the middle. Ties are
    averaged (pandas' default rank behavior). ``strictly_positive`` defaults
    to ``True`` (the usual price-series case); pass ``False`` for a series
    that can legitimately be zero or negative (e.g. a pairs-trading spread
    residual) -- the rank computation itself is sign-agnostic, only the
    input validation differs.
    """
    validated = numeric_pandas(
        prices, name="prices", strictly_positive=strictly_positive
    )
    length = positive_int(window, name="window", minimum=2)

    def _percentile_of_last(window_values: np.ndarray) -> float:
        return float(pd.Series(window_values).rank(pct=True).iloc[-1])

    return validated.rolling(length, min_periods=length).apply(
        _percentile_of_last, raw=True
    )


def rsi(
    prices: PandasT, window: int = 14, *, strictly_positive: bool = True
) -> PandasT:
    """Relative Strength Index using Wilder-style exponential smoothing.

    Returns values in ``[0, 100]``; a flat window is neutral at ``50``.
    ``strictly_positive`` defaults to ``True`` (the usual price-series
    case); pass ``False`` for a series that can legitimately be zero or
    negative (e.g. a pairs-trading spread residual) -- RSI is computed
    from period-over-period changes, which are sign-agnostic; only the
    input validation differs.
    """
    validated = numeric_pandas(
        prices, name="prices", strictly_positive=strictly_positive
    )
    length = positive_int(window, name="window", minimum=2)
    delta = validated.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    if isinstance(avg_gain, pd.Series):
        if not isinstance(avg_loss, pd.Series):
            raise TypeError("RSI smoothing produced incompatible pandas objects.")
        result = _finish_rsi_series(avg_gain, avg_loss)
    else:
        if not isinstance(avg_loss, pd.DataFrame):
            raise TypeError("RSI smoothing produced incompatible pandas objects.")
        result = _finish_rsi_frame(avg_gain, avg_loss)
    return cast(PandasT, result)  # type: ignore[redundant-cast]


def _finish_rsi_series(avg_gain: pd.Series, avg_loss: pd.Series) -> pd.Series:
    both_flat = avg_gain.eq(0.0) & avg_loss.eq(0.0)
    only_gains = avg_gain.gt(0.0) & avg_loss.eq(0.0)
    rs = avg_gain / avg_loss.where(avg_loss > 0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    return result.mask(both_flat, 50.0).mask(only_gains, 100.0)


def _finish_rsi_frame(avg_gain: pd.DataFrame, avg_loss: pd.DataFrame) -> pd.DataFrame:
    both_flat = avg_gain.eq(0.0) & avg_loss.eq(0.0)
    only_gains = avg_gain.gt(0.0) & avg_loss.eq(0.0)
    rs = avg_gain / avg_loss.where(avg_loss > 0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    return result.mask(both_flat, 50.0).mask(only_gains, 100.0)


def bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns a DataFrame with ``lower``, ``middle``, ``upper`` and ``pct_b``
    (position within the band, 0 at the lower band, 1 at the upper).
    """
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    length = positive_int(window, name="window", minimum=2)
    width_multiplier = finite_real(num_std, name="num_std", minimum=0.0, strict=True)
    middle = validated.rolling(length, min_periods=length).mean()
    std = validated.rolling(length, min_periods=length).std(ddof=1)
    upper = middle + width_multiplier * std
    lower = middle - width_multiplier * std
    band_width = upper - lower
    pct_b = (validated - lower) / band_width.where(band_width.abs() > EPSILON)
    pct_b = pct_b.mask(band_width.abs() <= EPSILON, 0.5)
    return pd.DataFrame(
        {"lower": lower, "middle": middle, "upper": upper, "pct_b": pct_b}
    )


def half_life(series: pd.Series) -> float:
    """Approximate mean-reversion half-life of a series.

    Fits the discrete Ornstein-Uhlenbeck regression
    ``Δy_t = λ · y_{t-1} + c + ε`` by OLS and returns ``-ln(2) / λ``. A positive,
    finite result indicates mean reversion; ``inf`` means no reversion detected.
    The result is measured in observations and assumes each adjacent row is one
    period. Missing rows are not stitched across gaps.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    y = numeric_pandas(series, name="series")
    lagged = y.shift(1)
    pairs = pd.concat({"lagged": lagged, "delta": y - lagged}, axis=1).dropna()
    if len(pairs) < 2:
        return float("inf")
    x = np.column_stack([np.ones(len(pairs)), pairs["lagged"].to_numpy()])
    try:
        coef, *_ = np.linalg.lstsq(x, pairs["delta"].to_numpy(), rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    lam = float(coef[1])
    if not np.isfinite(lam) or lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)
