"""Tests for trend-following strategy and volatility-targeting allocator/portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from quantlab.exceptions import InvalidConfigurationError
from quantlab.portfolio.allocator import VolatilityTargetingAllocator
from quantlab.portfolio.rebalancing import cap_turnover, rebalance_dates
from quantlab.portfolio.volatility_targeting import (
    apply_volatility_target,
    estimated_portfolio_volatility,
    volatility_target_leverage,
)
from quantlab.strategies.trend_following import TrendFollowingStrategy


def test_trend_following_long_only_uptrend() -> None:
    prices = np.linspace(100, 250, 200)
    data = make_ohlcv("AAA", prices)
    strat = TrendFollowingStrategy(
        fast_window=10,
        slow_window=40,
        long_only=True,
    )
    signals = strat.generate_signals(data)
    last = signals["AAA"].dropna().iloc[-1]
    assert last >= 0  # long-only never goes short
    assert last <= 1.0


def test_trend_following_short_allowed_on_downtrend() -> None:
    prices = np.linspace(250, 100, 200)
    data = make_ohlcv("AAA", prices)
    strat = TrendFollowingStrategy(
        fast_window=10,
        slow_window=40,
        long_only=False,
    )
    signals = strat.generate_signals(data)
    assert signals["AAA"].dropna().iloc[-1] < 0


def test_trend_following_rejects_bad_windows() -> None:
    with pytest.raises(ValueError, match="fast_window"):
        TrendFollowingStrategy(fast_window=50, slow_window=20)


def test_volatility_targeting_allocator_scales_exposure(
    synthetic_panel: pd.DataFrame,
) -> None:
    idx = synthetic_panel["timestamp"].drop_duplicates().sort_values()
    signals = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB", "CCC"])
    alloc = VolatilityTargetingAllocator(
        target_volatility=0.10, volatility_window=30, maximum_leverage=2.0
    )
    weights = alloc.allocate(signals, synthetic_panel)
    assert np.isfinite(weights.to_numpy()).all()
    # Leverage cap respected: gross exposure never exceeds max_leverage roughly.
    gross = weights.abs().sum(axis=1).dropna()
    assert (gross <= 2.0 + 1e-6).all()


def test_estimated_portfolio_volatility_and_leverage() -> None:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=300)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, 300)}, index=idx)
    weights = pd.DataFrame({"A": [1.0] * 300}, index=idx)
    vol = estimated_portfolio_volatility(weights, returns, window=30)
    assert (vol.dropna() >= 0).all()
    leverage = volatility_target_leverage(
        weights, returns, target_volatility=0.10, window=30, maximum_leverage=1.5
    )
    assert (leverage <= 1.5 + 1e-9).all()
    scaled = apply_volatility_target(
        weights, returns, target_volatility=0.10, window=30, maximum_leverage=1.5
    )
    # apply_volatility_target must scale weights by exactly the leverage
    # series computed above, row by row.
    assert scaled.shape == weights.shape
    expected = weights.mul(leverage, axis=0)
    assert np.allclose(scaled.to_numpy(), expected.to_numpy())


def test_rebalance_dates_daily() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    assert len(rebalance_dates(idx, "daily")) == 10


def test_rebalance_dates_custom_not_implemented() -> None:
    """'custom' must be refused explicitly, not silently behave as monthly."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    with pytest.raises(InvalidConfigurationError):
        rebalance_dates(idx, "custom")


def test_cap_turnover_limits_change() -> None:
    idx = pd.date_range("2020-01-01", periods=3)
    held = pd.DataFrame({"A": [1.0, 1.0, -1.0]}, index=idx)
    capped = cap_turnover(held, maximum_turnover=0.5)
    # Turnover from date1->date2 should be capped at 0.5 (not the full 2.0 jump).
    change = (capped.iloc[2] - capped.iloc[1]).abs().sum()
    assert change <= 0.5 + 1e-9
