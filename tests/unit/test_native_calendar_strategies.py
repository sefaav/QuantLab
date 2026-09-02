"""Wiring tests: every rolling-window feature call site in a strategy (and
`runner.py`'s ADV) must route through `compute_native_then_align` via
`BaseStrategy.symbol_calendars` / `self._native_feature`, and must be a
provable no-op (byte-identical) when `symbol_calendars` is unset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)


def _long_from_wide(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol in prices.columns:
        series = prices[symbol]
        frames.append(
            pd.DataFrame(
                {
                    TIMESTAMP: series.index,
                    SYMBOL: symbol,
                    OPEN: series.to_numpy(),
                    HIGH: series.to_numpy(),
                    LOW: series.to_numpy(),
                    CLOSE: series.to_numpy(),
                    ADJUSTED_CLOSE: series.to_numpy(),
                    VOLUME: 1_000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _mixed_calendar_panel(periods: int = 42) -> pd.DataFrame:
    """AAA (session-bound, closed weekends) + BTC (24/7), 42 calendar days
    (30 native AAA trading days -- enough margin above pairs_trading's
    ``formation_window`` minimum of 20).

    AAA's weekend rows are flat-filled from the prior real close, exactly
    matching `insert_verified_closure_bars`'s own production convention.
    """
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")  # Jan 1 = Monday
    is_weekend = dates.weekday >= 5
    aaa = np.empty(len(dates))
    trading_values = np.linspace(100.0, 150.0, num=int((~is_weekend).sum()))
    aaa[~is_weekend] = trading_values
    last = np.nan
    for i in range(len(dates)):
        if is_weekend[i]:
            aaa[i] = last
        else:
            last = aaa[i]
    btc = np.linspace(40_000.0, 41_260.0, num=len(dates))
    return pd.DataFrame({"AAA": aaa, "BTC": btc}, index=dates)


_CALENDARS = {"AAA": "XNYS", "BTC": "24/7"}


def test_time_series_momentum_native_calendar_changes_diluted_output() -> None:
    from quantlab.strategies.momentum import TimeSeriesMomentumStrategy

    prices = _mixed_calendar_panel()
    data = _long_from_wide(prices)

    diluted = TimeSeriesMomentumStrategy(
        lookback_period=10, skip_period=0, signal_scaling="binary"
    )
    diluted_signals = diluted.generate_signals(data)

    native = TimeSeriesMomentumStrategy(
        lookback_period=10, skip_period=0, signal_scaling="binary"
    )
    native.symbol_calendars = _CALENDARS
    native_signals = native.generate_signals(data)

    # BTC has no closures at all -- untouched either way.
    pd.testing.assert_series_equal(
        diluted_signals["BTC"], native_signals["BTC"], check_names=False
    )
    # AAA's dilution genuinely changes at least one date's signal.
    assert not diluted_signals["AAA"].equals(native_signals["AAA"])


def test_cross_sectional_momentum_native_calendar_changes_diluted_output() -> None:
    from quantlab.strategies.momentum import CrossSectionalMomentumStrategy

    prices = _mixed_calendar_panel()
    data = _long_from_wide(prices)

    diluted = CrossSectionalMomentumStrategy(
        lookback_period=10, skip_period=0, top_fraction=0.5, signal_scaling="binary"
    )
    diluted_signals = diluted.generate_signals(data)

    native = CrossSectionalMomentumStrategy(
        lookback_period=10, skip_period=0, top_fraction=0.5, signal_scaling="binary"
    )
    native.symbol_calendars = _CALENDARS
    native_signals = native.generate_signals(data)

    assert not diluted_signals.equals(native_signals)


def test_trend_following_native_calendar_changes_diluted_output() -> None:
    from quantlab.strategies.trend_following import TrendFollowingStrategy

    prices = _mixed_calendar_panel()
    data = _long_from_wide(prices)

    diluted = TrendFollowingStrategy(fast_window=3, slow_window=10)
    diluted_signals = diluted.generate_signals(data)

    native = TrendFollowingStrategy(fast_window=3, slow_window=10)
    native.symbol_calendars = _CALENDARS
    native_signals = native.generate_signals(data)

    pd.testing.assert_series_equal(
        diluted_signals["BTC"], native_signals["BTC"], check_names=False
    )
    assert not diluted_signals["AAA"].equals(native_signals["AAA"])


def test_mean_reversion_native_calendar_changes_diluted_output() -> None:
    """Exercises `_centered_indicator` directly (rather than the full
    entry/exit state machine) -- this simple synthetic price panel never
    crosses the default z-score entry threshold, so the post-threshold
    SIGNAL would be an uninformative constant zero either way; the
    underlying INDICATOR is where the native-calendar wrapping actually
    shows up."""
    from quantlab.strategies.mean_reversion import _centered_indicator

    prices = _mixed_calendar_panel()

    diluted = _centered_indicator(prices, "zscore", 10, 2.0)
    native = _centered_indicator(prices, "zscore", 10, 2.0, _CALENDARS)

    pd.testing.assert_series_equal(diluted["BTC"], native["BTC"], check_names=False)
    assert not diluted["AAA"].equals(native["AAA"])


def test_engine_collapses_position_group_tradability_end_to_end() -> None:
    """`BacktestEngine.run()` must gate a declared position group's
    tradability as ONE unit (both legs eligible only on a date BOTH are
    open), not per-leg independently -- otherwise a rebalance could move
    one leg of a pair while the other stays frozen, introducing unmodeled
    legging risk. Checked at the `result.weights` level: across the whole
    backtest, AAA's executed weight must change on a date if and only if
    BTC's does too."""
    from tests.conftest import geometric_series

    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.config import ExperimentConfig

    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    is_weekend = dates.weekday >= 5
    aaa_native = geometric_series(
        int((~is_weekend).sum()), mu=0.0, sigma=0.02, s0=100.0, seed=11
    )
    aaa = np.empty(len(dates))
    aaa[~is_weekend] = aaa_native
    last = np.nan
    for i in range(len(dates)):
        if is_weekend[i]:
            aaa[i] = last
        else:
            last = aaa[i]
    btc = geometric_series(len(dates), mu=0.0, sigma=0.02, s0=100.0, seed=22)
    prices = pd.DataFrame({"AAA": aaa, "BTC": btc}, index=dates)
    data = _long_from_wide(prices)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "pairs_group_tradability",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BTC", "source": "csv", "calendar": "24/7"},
                ],
                "start_date": "2024-01-01",
                "end_date": "2024-03-30",
            },
            "strategy": {
                "name": "pairs_trading",
                "parameters": {
                    "symbol_a": "AAA",
                    "symbol_b": "BTC",
                    "formation_window": 20,
                    "indicator_window": 5,
                    "indicator": "percentile",
                    "entry_threshold": 0.30,
                    "exit_threshold": 0.05,
                    "adf_pvalue_threshold": None,
                    "dynamic_hedge_ratio": True,
                },
            },
            "portfolio": {
                "allocator": "signal_proportional",
                "rebalance_frequency": "daily",
            },
            "execution": {},
            "backtest": {"periods_per_year": 252},
        }
    )

    result = run_backtest_from_config(data, cfg)

    aaa_changed = result.weights["AAA"].diff().abs() > 1e-9
    btc_changed = result.weights["BTC"].diff().abs() > 1e-9
    pd.testing.assert_series_equal(aaa_changed, btc_changed, check_names=False)
    assert bool(aaa_changed.any())  # the invariant must be exercised, not vacuous


def test_pairs_trading_native_pair_context_uses_intersection_of_native_calendars() -> (
    None
):
    from quantlab.strategies.pairs_trading import PairsTradingStrategy

    prices = _mixed_calendar_panel()

    diluted = PairsTradingStrategy(
        symbol_a="AAA",
        symbol_b="BTC",
        formation_window=20,
        indicator_window=5,
        adf_pvalue_threshold=None,
    )
    _, _, diluted_indicator, _, diluted_tradable = diluted._native_pair_context(prices)

    native = PairsTradingStrategy(
        symbol_a="AAA",
        symbol_b="BTC",
        formation_window=20,
        indicator_window=5,
        adf_pvalue_threshold=None,
    )
    native.symbol_calendars = _CALENDARS
    _, _, native_indicator, _, native_tradable = native._native_pair_context(prices)

    assert not diluted_indicator.equals(native_indicator)
    # No injected calendar -> both legs always considered open.
    assert bool(diluted_tradable.all())
    # Injected mixed calendars -> weekends/holidays correctly block entry.
    assert not bool(native_tradable.all())


def test_pairs_trading_entry_gate_matches_symbol_a_native_calendar() -> None:
    """`symbol_b` (BTC) is 24/7, so the combined entry gate must reduce
    exactly to `symbol_a`'s (AAA/XNYS) own native session mask when the ADF
    stationarity gate is disabled."""
    from quantlab.data.calendar import is_session_day
    from quantlab.strategies.pairs_trading import PairsTradingStrategy

    prices = _mixed_calendar_panel()
    strategy = PairsTradingStrategy(
        symbol_a="AAA",
        symbol_b="BTC",
        formation_window=20,
        indicator_window=5,
        adf_pvalue_threshold=None,
    )
    strategy.symbol_calendars = _CALENDARS
    _, _, _, _, tradable = strategy._native_pair_context(prices)

    expected = is_session_day("XNYS", pd.DatetimeIndex(prices.index))
    np.testing.assert_array_equal(tradable, expected)


def test_build_execution_from_config_adv_uses_native_calendar() -> None:
    """A weekend closure's synthetic zero-volume bar must never drag down a
    session-bound symbol's own trailing dollar-ADV -- the window's content
    is computed on its own native calendar, not the closure-padded
    combined timeline."""
    from quantlab.backtesting.runner import build_execution_from_config
    from quantlab.config import ExperimentConfig
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    dates = pd.date_range("2024-01-01", periods=28, freq="D")
    is_weekend = dates.weekday >= 5
    aaa_close = np.full(len(dates), 100.0)
    aaa_volume = np.where(is_weekend, 0.0, 1_000_000.0)
    btc_close = np.full(len(dates), 40_000.0)
    btc_volume = np.full(len(dates), 500_000.0)

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    TIMESTAMP: dates,
                    SYMBOL: "AAA",
                    OPEN: aaa_close,
                    HIGH: aaa_close,
                    LOW: aaa_close,
                    CLOSE: aaa_close,
                    ADJUSTED_CLOSE: aaa_close,
                    VOLUME: aaa_volume,
                }
            ),
            pd.DataFrame(
                {
                    TIMESTAMP: dates,
                    SYMBOL: "BTC",
                    OPEN: btc_close,
                    HIGH: btc_close,
                    LOW: btc_close,
                    CLOSE: btc_close,
                    ADJUSTED_CLOSE: btc_close,
                    VOLUME: btc_volume,
                }
            ),
        ],
        ignore_index=True,
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_native",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BTC", "source": "csv", "calendar": "24/7"},
                ],
                "start_date": "2024-01-01",
                "end_date": "2024-01-28",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {"slippage_model": "volume", "slippage_bps": 5.0},
            "backtest": {"periods_per_year": 252},
        }
    )

    execution = build_execution_from_config(cfg, data)
    slippage = execution.slippage
    assert isinstance(slippage, VolumeBasedSlippageModel)
    adv = slippage.average_daily_volume
    assert isinstance(adv, pd.DataFrame)

    naive = (
        pd.DataFrame(
            {"AAA": aaa_volume * aaa_close, "BTC": btc_volume * btc_close}, index=dates
        )
        .rolling(21, min_periods=1)
        .mean()
        .shift(1)
    )
    assert not adv["AAA"].equals(naive["AAA"])
    # Every real AAA trading day has an identical dollar volume; its native
    # trailing ADV must therefore be exactly constant, never diluted below
    # this by a weekend's zero-volume synthetic bar.
    assert np.allclose(adv["AAA"].dropna().to_numpy(), 100_000_000.0)


def test_symbol_calendars_none_is_byte_identical_to_no_wrapper() -> None:
    """Regression safety gate: an unset `symbol_calendars` (the default,
    e.g. a strategy built directly in a unit test outside the engine) must
    reproduce today's plain vectorized computation exactly."""
    from quantlab.features.momentum import momentum
    from quantlab.strategies.momentum import TimeSeriesMomentumStrategy

    prices = _mixed_calendar_panel()
    data = _long_from_wide(prices)
    strategy = TimeSeriesMomentumStrategy(
        lookback_period=10, skip_period=0, signal_scaling="binary"
    )
    signals = strategy.generate_signals(data)

    expected_score = momentum(prices, 10, 0)
    expected = pd.DataFrame(
        np.sign(expected_score.to_numpy()),
        index=expected_score.index,
        columns=expected_score.columns,
    ).fillna(0.0)
    pd.testing.assert_frame_equal(
        signals, expected, check_dtype=False, check_names=False, check_freq=False
    )


def test_parameters_excludes_symbol_calendars() -> None:
    """`symbol_calendars` is engine-injected context, never a user-supplied
    hyperparameter -- `BaseStrategy.parameters()` must never surface it in
    a config-YAML round-trip, execution-model hash, or sweep-parameter
    enumeration (see `_NON_PARAMETER_ATTRIBUTES`). Relied on throughout
    this module (every other test here sets `.symbol_calendars` directly,
    bypassing the constructor-parameter freeze) but never directly
    asserted until now."""
    from quantlab.strategies.momentum import TimeSeriesMomentumStrategy

    strategy = TimeSeriesMomentumStrategy(
        lookback_period=10, skip_period=0, signal_scaling="binary"
    )
    strategy.symbol_calendars = _CALENDARS

    params = strategy.parameters()

    assert "symbol_calendars" not in params
    assert strategy.symbol_calendars == _CALENDARS
