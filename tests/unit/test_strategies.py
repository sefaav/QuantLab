"""Tests for trading strategies.

Each strategy is checked on a synthetic dataset with an obvious expected
behaviour, plus the universal contract: signals in ``[-1, 1]``, no NaNs, shape
matching the price panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from quantlab.strategies import (
    available_strategies,
    build_strategy,
)
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.momentum import (
    CrossSectionalMomentumStrategy,
    TimeSeriesMomentumStrategy,
)
from quantlab.strategies.pairs_trading import PairsTradingStrategy, adf_pvalue


def _assert_contract(signals: pd.DataFrame) -> None:
    arr = signals.to_numpy()
    assert np.isfinite(arr).all(), "signals must be finite"
    assert (arr >= -1.0 - 1e-9).all()
    assert (arr <= 1.0 + 1e-9).all()


# --------------------------------------------------------------------------- #
def test_registry_has_all_strategies() -> None:
    for name in [
        "buy_and_hold",
        "time_series_momentum",
        "cross_sectional_momentum",
        "mean_reversion",
        "trend_following",
        "pairs_trading",
    ]:
        assert name in available_strategies()


def test_buy_and_hold_all_long(synthetic_panel: pd.DataFrame) -> None:
    strat = build_strategy("buy_and_hold")
    signals = strat.generate_signals(synthetic_panel)
    _assert_contract(signals)
    # Everything with a price is fully long.
    assert (signals.to_numpy() == 1.0).all()


def test_time_series_momentum_long_on_uptrend() -> None:
    # Strictly rising prices → positive momentum → long signal.
    prices = np.linspace(100, 300, 320)
    data = make_ohlcv("AAA", prices)
    strat = TimeSeriesMomentumStrategy(
        lookback_period=100, skip_period=5, long_only=True, signal_scaling="binary"
    )
    signals = strat.generate_signals(data)
    _assert_contract(signals)
    assert signals["AAA"].iloc[-1] == 1.0


def test_time_series_momentum_flat_or_short_on_downtrend() -> None:
    prices = np.linspace(300, 100, 320)
    data = make_ohlcv("AAA", prices)
    # long_only=False so a downtrend can go short.
    strat = TimeSeriesMomentumStrategy(
        lookback_period=100, skip_period=5, long_only=False
    )
    signals = strat.generate_signals(data)
    _assert_contract(signals)
    assert signals["AAA"].iloc[-1] == -1.0


def test_cross_sectional_momentum_picks_winner(synthetic_panel: pd.DataFrame) -> None:
    # AAA trends up, BBB trends down (see conftest). Long-only top 1/3.
    strat = CrossSectionalMomentumStrategy(
        lookback_period=100, skip_period=5, top_fraction=0.34, long_short=False
    )
    signals = strat.generate_signals(synthetic_panel)
    _assert_contract(signals)
    last = signals.dropna().iloc[-1]
    assert last["AAA"] == 1.0  # strongest momentum → long
    assert last["BBB"] == 0.0  # weakest → not selected (long-only)


def test_mean_reversion_goes_long_after_crash() -> None:
    # Flat then a sharp drop → z-score deeply negative → long entry.
    prices = np.concatenate([np.full(40, 100.0), np.linspace(100, 70, 10)])
    data = make_ohlcv("AAA", prices)
    strat = MeanReversionStrategy(
        lookback_period=20, entry_zscore=1.5, exit_zscore=0.5, long_only=True
    )
    signals = strat.generate_signals(data)
    _assert_contract(signals)
    assert signals["AAA"].iloc[-1] == 1.0


def test_mean_reversion_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError, match="entry_zscore"):
        MeanReversionStrategy(entry_zscore=0.5, exit_zscore=2.0)


def test_pairs_trading_contract(two_symbol_panel: pd.DataFrame) -> None:
    strat = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=120,
        zscore_window=30,
        entry_zscore=1.5,
        exit_zscore=0.5,
    )
    signals = strat.generate_signals(two_symbol_panel)
    _assert_contract(signals)
    # The strategy must actually trade this panel, and legs move in opposite
    # directions whenever a position is on (long one leg, short the other).
    active = signals[(signals["EWA"] != 0) | (signals["EWB"] != 0)]
    assert len(active) > 0
    row = active.iloc[-1]
    assert np.sign(row["EWA"]) == -np.sign(row["EWB"])


def test_adf_pvalue_on_stationary_series() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)  # stationary AR(1)
    p = adf_pvalue(pd.Series(x))
    assert p is not None
    assert p < 0.1  # rejects unit root → stationary


def test_build_strategy_unknown_raises() -> None:
    from quantlab.exceptions import StrategyError

    with pytest.raises(StrategyError):
        build_strategy("does_not_exist")
