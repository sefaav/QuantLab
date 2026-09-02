"""Additional feature tests (technical indicators, returns, resampler)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv

from quantlab.data.resampler import resample_ohlcv
from quantlab.features import technical as T
from quantlab.features.returns import equity_curve, log_returns
from quantlab.features.volatility import (
    average_true_range,
    downside_volatility,
    ewma_volatility,
    rolling_beta,
    rolling_correlation,
)


def test_ema_and_macd() -> None:
    prices = pd.Series(np.linspace(100, 200, 80))
    ema = T.exponential_moving_average(prices, span=10)
    assert ema.dropna().iloc[-1] > ema.dropna().iloc[0]
    macd = T.macd(prices)
    assert {"macd", "signal", "histogram"} == set(macd.columns)


def test_rolling_channels_and_donchian() -> None:
    prices = pd.Series(
        np.concatenate([np.linspace(100, 150, 40), np.linspace(150, 120, 20)])
    )
    hi = T.rolling_max(prices, 10)
    lo = T.rolling_min(prices, 10)
    assert (hi.dropna() >= lo.dropna()).all()
    pos = T.donchian_position(prices, 10).dropna()
    assert pos.between(-0.01, 1.01).all()


def test_efficiency_ratio_is_high_for_a_clean_trend_and_low_for_noise() -> None:
    trending = pd.Series(np.linspace(100, 200, 60))
    rng = np.random.default_rng(0)
    choppy = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, 60)))
    trending_ratio = T.efficiency_ratio(trending, 20).dropna()
    choppy_ratio = T.efficiency_ratio(choppy, 20).dropna()
    assert trending_ratio.between(0.0, 1.0).all()
    assert choppy_ratio.between(0.0, 1.0).all()
    # A perfectly monotonic trend's net move equals its total path length.
    assert trending_ratio.iloc[-1] == pytest.approx(1.0)
    assert trending_ratio.mean() > choppy_ratio.mean()


def test_efficiency_ratio_is_neutral_for_a_flat_window() -> None:
    flat = pd.Series(np.full(30, 100.0))
    ratio = T.efficiency_ratio(flat, 10).dropna()
    assert (ratio == 0.5).all()


def test_log_returns_and_equity_curve() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    lr = log_returns(prices)
    assert lr.iloc[1] == pytest.approx(np.log(1.1))
    eq = equity_curve(pd.Series([np.nan, 0.1, -0.1]), initial=100.0)
    assert eq.iloc[-1] == pytest.approx(100 * 1.1 * 0.9)


def test_atr_and_downside_vol() -> None:
    data = make_ohlcv("AAA", geometric_series(200, 0.0005, 0.01, 100.0, 1))
    high = data.set_index("timestamp")["high"]
    low = data.set_index("timestamp")["low"]
    close = data.set_index("timestamp")["close"]
    atr = average_true_range(high, low, close, window=14)
    assert (atr.dropna() >= 0).all()
    rets = close.pct_change(fill_method=None)
    dvol = downside_volatility(rets, window=30)
    assert (dvol.dropna() >= 0).all()


def test_ewma_vol_and_rolling_beta() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 300))
    assert ewma_volatility(r, halflife=21).dropna().gt(0).all()
    b = pd.Series(rng.normal(0, 0.01, 300))
    strat = 1.2 * b
    beta = rolling_beta(strat, b, window=63).dropna()
    assert np.allclose(beta.to_numpy(), 1.2, atol=1e-6)
    corr = rolling_correlation(strat, b, window=63).dropna()
    assert np.allclose(corr.to_numpy(), 1.0, atol=1e-6)


def test_resample_ohlcv_weekly() -> None:
    data = make_ohlcv("AAA", geometric_series(60, 0.0005, 0.01, 100.0, 1), freq="B")
    weekly = resample_ohlcv(data, "1w")
    # Weekly bars are fewer than daily and preserve OHLC ordering.
    assert len(weekly) < len(data)
    assert (weekly["high"] >= weekly["low"]).all()
    assert set(weekly["symbol"]) == {"AAA"}
