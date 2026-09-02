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
from quantlab.features.cross_sectional import select_top_bottom
from quantlab.features.returns import forward_returns, simple_returns
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


def momentum_persistence(
    prices: pd.Series,
    lookback_period: int,
    skip_period: int,
    holding_period: int,
) -> pd.DataFrame:
    """Pair each date's trailing momentum score with its subsequent return.

    Returns a two-column ``(past_momentum, future_return)`` DataFrame, one
    row per date where both are defined -- the basic building block for
    checking whether momentum actually persists on a given series (does a
    high past score tend to be followed by a high subsequent return, on
    this data): a positive relationship is descriptive-sample evidence FOR
    the strategy's premise, a flat or negative one against it -- not a
    hypothesis test (rows from overlapping holding periods are not
    independent observations, so this is not a significance claim).
    ``future_return`` looks strictly ahead of each row's own date -- these
    pairs describe the data, they are never a tradable signal themselves
    (see ``forward_returns``). This asks the TIME-SERIES question (does
    THIS asset's own past predict its own future); for the cross-sectional
    question (do higher-RANKED assets outperform lower-ranked ones), see
    :func:`cross_sectional_momentum_persistence`.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    past = momentum(prices, lookback_period, skip_period)
    horizon = positive_int(holding_period, name="holding_period")
    future = forward_returns(prices, horizon)
    return pd.concat({"past_momentum": past, "future_return": future}, axis=1).dropna()


def cross_sectional_momentum_persistence(
    prices: pd.DataFrame,
    lookback_period: int,
    skip_period: int,
    holding_period: int,
    *,
    top_fraction: float = 0.25,
    bottom_fraction: float | None = None,
) -> pd.DataFrame:
    """Date-by-date evidence for CROSS-SECTIONAL momentum persistence.

    Unlike :func:`momentum_persistence` (a single asset's own past-vs-
    future relationship -- the TIME-SERIES momentum question), this asks
    the question cross-sectional momentum actually trades: on each date,
    do assets ranked higher on trailing momentum go on to earn higher
    subsequent returns than assets ranked lower, RELATIVE TO EACH OTHER?
    An asset's own serial autocorrelation is neither necessary nor
    sufficient for that.

    Returns one row per date with at least 3 assets scored, containing the
    Spearman rank correlation between that date's momentum scores and
    subsequent ``holding_period``-period returns across the universe
    (``rank_correlation``), plus the realized ``top_return``/
    ``bottom_return``/``top_minus_bottom`` spread for the
    ``top_fraction``/``bottom_fraction`` selection (mirroring
    ``CrossSectionalMomentumStrategy``'s own selection via
    :func:`~quantlab.features.cross_sectional.select_top_bottom`) over the
    same horizon. Descriptive sample evidence, not a hypothesis test --
    overlapping holding periods across consecutive dates are not
    independent observations.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    bottom = bottom_fraction if bottom_fraction is not None else top_fraction
    scores = momentum(prices, lookback_period, skip_period)
    horizon = positive_int(holding_period, name="holding_period")
    future = forward_returns(prices, horizon)
    selection = select_top_bottom(scores, top_fraction, bottom)

    rows: list[dict[str, object]] = []
    for date in scores.index:
        score_row = scores.loc[date]
        future_row = future.loc[date]
        valid = score_row.notna() & future_row.notna()
        if int(valid.sum()) < 3:
            continue
        rank_correlation = score_row[valid].corr(future_row[valid], method="spearman")
        top_mask = valid & (selection.loc[date] == 1.0)
        bottom_mask = valid & (selection.loc[date] == -1.0)
        top_return = future_row[top_mask].mean() if top_mask.any() else np.nan
        bottom_return = future_row[bottom_mask].mean() if bottom_mask.any() else np.nan
        rows.append(
            {
                "date": date,
                "rank_correlation": rank_correlation,
                "top_return": top_return,
                "bottom_return": bottom_return,
                "top_minus_bottom": top_return - bottom_return,
            }
        )
    columns = ["rank_correlation", "top_return", "bottom_return", "top_minus_bottom"]
    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("date")
    return pd.DataFrame(rows).set_index("date")[columns]
