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

from quantlab.exceptions import StrategyError
from quantlab.features.mean_reversion import rolling_zscore
from quantlab.strategies import (
    available_strategies,
    build_strategy,
)
from quantlab.strategies.base import BaseStrategy
from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy
from quantlab.strategies.mean_reversion import (
    INDICATORS,
    MeanReversionStrategy,
    _walk_positions_with_reasons,
)
from quantlab.strategies.momentum import (
    CrossSectionalMomentumStrategy,
    TimeSeriesMomentumStrategy,
)
from quantlab.strategies.pairs_trading import (
    PairsTradingStrategy,
    _walk_pairs_positions_with_reasons,
    adf_pvalue,
    rolling_hedge_parameters,
)
from quantlab.strategies.trend_following import TrendFollowingStrategy


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


def test_cross_sectional_magnitude_is_monotone_within_each_selected_leg() -> None:
    """Regression test: an earlier, cross-sectional-mean-centered version
    of this function was NOT monotone in score within a selected leg --
    scores [0, 1, 2] all selected as one long leg standardized to
    magnitudes [1, 0, 1] (mean 1, std 1), zeroing out the MIDDLE score
    while the best and worst tied at full weight. Rank-within-leg must
    fix this: strictly increasing with score in the long leg, strictly
    decreasing (more negative = higher magnitude) in the short leg, and
    never exactly zero for a selected asset."""
    from quantlab.strategies.momentum import _cross_sectional_magnitude

    idx = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [0.0], "B": [1.0], "C": [2.0]}, index=idx)
    selection = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]}, index=idx)
    magnitude = _cross_sectional_magnitude(score, selection)
    row = magnitude.iloc[0]
    assert row["A"] < row["B"] < row["C"]
    assert (row > 0.0).all()
    assert row["C"] == pytest.approx(1.0)

    # Mirrored on the short side: more negative score -> higher magnitude.
    short_score = pd.DataFrame({"A": [-2.0], "B": [-1.0], "C": [-0.5]}, index=idx)
    short_selection = pd.DataFrame({"A": [-1.0], "B": [-1.0], "C": [-1.0]}, index=idx)
    short_magnitude = _cross_sectional_magnitude(short_score, short_selection)
    short_row = short_magnitude.iloc[0]
    assert short_row["A"] > short_row["B"] > short_row["C"]
    assert (short_row > 0.0).all()
    assert short_row["A"] == pytest.approx(1.0)

    # Unselected assets stay at exactly zero regardless of their score.
    mixed_score = pd.DataFrame(
        {"A": [5.0], "B": [1.0], "C": [-1.0], "D": [-5.0]}, index=idx
    )
    mixed_selection = pd.DataFrame(
        {"A": [1.0], "B": [0.0], "C": [0.0], "D": [-1.0]}, index=idx
    )
    mixed_magnitude = _cross_sectional_magnitude(mixed_score, mixed_selection)
    mixed_row = mixed_magnitude.iloc[0]
    assert mixed_row["B"] == 0.0
    assert mixed_row["C"] == 0.0
    assert mixed_row["A"] == pytest.approx(1.0)
    assert mixed_row["D"] == pytest.approx(1.0)


def test_cross_sectional_magnitude_is_invariant_to_column_permutation() -> None:
    """Regression test: identical scores must get identical magnitudes
    regardless of which column order they happen to be pivoted into --
    ``rank(method="first")`` broke ties by column position, an arbitrary,
    non-economic artifact (e.g. two backtests over the same data loaded
    with a differently-ordered universe declaration would silently size
    tied positions differently). ``method="max"`` fixes this: every tied
    score shares the same rank."""
    from quantlab.strategies.momentum import _cross_sectional_magnitude

    idx = pd.date_range("2024-01-01", periods=1)
    score_abc = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]}, index=idx)
    selection_abc = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]}, index=idx)
    magnitude_abc = _cross_sectional_magnitude(score_abc, selection_abc)
    # A fully tied leg must resolve to magnitude 1.0 for EVERY member, not
    # a range spread across the tie group by column order.
    assert magnitude_abc.iloc[0].to_dict() == {"A": 1.0, "B": 1.0, "C": 1.0}

    score_cab = score_abc[["C", "A", "B"]]
    selection_cab = selection_abc[["C", "A", "B"]]
    magnitude_cab = _cross_sectional_magnitude(score_cab, selection_cab)
    assert magnitude_cab.iloc[0].to_dict() == {"C": 1.0, "A": 1.0, "B": 1.0}

    # A partial tie (two names share the best score) must also resolve
    # identically for both, regardless of order, while staying monotone
    # against the untied, lower-scored name.
    partial_score = pd.DataFrame({"A": [0.0], "B": [2.0], "C": [2.0]}, index=idx)
    partial_selection = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]}, index=idx)
    partial_magnitude = _cross_sectional_magnitude(partial_score, partial_selection)
    partial_row = partial_magnitude.iloc[0]
    assert partial_row["B"] == partial_row["C"] == pytest.approx(1.0)
    assert partial_row["A"] < partial_row["B"]

    reordered_score = partial_score[["C", "A", "B"]]
    reordered_selection = partial_selection[["C", "A", "B"]]
    reordered_magnitude = _cross_sectional_magnitude(
        reordered_score, reordered_selection
    )
    assert reordered_magnitude.iloc[0]["B"] == pytest.approx(partial_row["B"])
    assert reordered_magnitude.iloc[0]["C"] == pytest.approx(partial_row["C"])
    assert reordered_magnitude.iloc[0]["A"] == pytest.approx(partial_row["A"])


def test_mean_reversion_goes_long_after_crash() -> None:
    # Flat then a sharp drop → z-score deeply negative → long entry.
    prices = np.concatenate([np.full(40, 100.0), np.linspace(100, 70, 10)])
    data = make_ohlcv("AAA", prices)
    strat = MeanReversionStrategy(
        lookback_period=20, entry_threshold=1.5, exit_threshold=0.5, long_only=True
    )
    signals = strat.generate_signals(data)
    _assert_contract(signals)
    assert signals["AAA"].iloc[-1] == 1.0


def test_mean_reversion_rejects_bad_thresholds() -> None:
    """entry_threshold must exceed exit_threshold."""
    with pytest.raises(ValueError, match="entry_threshold"):
        MeanReversionStrategy(entry_threshold=0.5, exit_threshold=2.0)


def test_mean_reversion_explicit_none_disables_stop_threshold() -> None:
    """Passing stop_threshold=None explicitly must disable the stop
    entirely -- NOT silently resolve to the indicator's own default, the
    bug this sentinel-based design fixes."""
    strat = MeanReversionStrategy(lookback_period=20, stop_threshold=None)
    assert strat.stop_threshold is None


def test_mean_reversion_omitted_stop_threshold_uses_indicator_default() -> None:
    """Leaving stop_threshold out entirely must resolve to the chosen
    indicator's own default -- distinct from an explicit None (see the
    test above)."""
    strat = MeanReversionStrategy(lookback_period=20, indicator="zscore")
    assert strat.stop_threshold == 4.0


@pytest.mark.parametrize("indicator", sorted(INDICATORS))
def test_mean_reversion_every_indicator_produces_a_valid_signal(indicator: str) -> None:
    """Every one of the five indicators must drive the SAME state machine
    to a valid, actually-nonzero signal on a series constructed to deviate
    sharply from its own recent history -- not just avoid raising.

    A 30-period monotonic decline (longer than the 25-period lookback)
    ensures even `percentile`'s rank(pct=True) -- whose minimum is
    exactly 1/N, never 0 -- comfortably clears its own default entry
    threshold (percentile < 0.05, i.e. 1/25 = 0.04)."""
    rng = np.random.default_rng(7)
    prices = np.concatenate(
        [100.0 + np.cumsum(rng.normal(0.0, 0.2, 60)), np.linspace(100.0, 40.0, 30)]
    )
    data = make_ohlcv("AAA", prices)
    strat = MeanReversionStrategy(
        lookback_period=25, indicator=indicator, long_only=False
    )
    signals = strat.generate_signals(data)
    _assert_contract(signals)
    assert (signals["AAA"] != 0.0).any()


def test_walk_positions_with_reasons_covers_every_branch() -> None:
    """Direct test of the state machine's reason attribution -- one
    z-score path deliberately visits every branch: oversold entry,
    mean-reversion exit, overbought entry, stop-loss exit, a no-op NaN
    (already flat, no reason recorded) and a NaN-driven forced exit."""
    z = np.array(
        [
            0.0,  # flat, below threshold -> no transition
            -2.5,  # crosses -entry (-2.0) -> oversold_entry
            -0.3,  # crosses -exit_ (-0.5) -> mean_reversion_exit
            2.5,  # crosses entry (2.0) -> overbought_entry
            5.0,  # |z| > stop (4.0) -> stop_loss_exit
            np.nan,  # already flat -> no-op, no reason recorded
            -2.5,  # oversold_entry again
            np.nan,  # was long -> data_unavailable_exit
        ]
    )

    positions, detail_code, details = _walk_positions_with_reasons(
        z, entry=2.0, exit_=0.5, stop=4.0, long_only=False
    )

    assert positions.tolist() == [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]
    assert detail_code.tolist() == [
        None,
        "oversold_entry",
        "mean_reversion_exit",
        "overbought_entry",
        "stop_loss_exit",
        None,
        "oversold_entry",
        "data_unavailable_exit",
    ]
    assert details[1] is not None
    assert "entry threshold -2.0000" in details[1]
    assert details[2] is not None
    assert "exit threshold -0.5000" in details[2]
    assert details[3] is not None
    assert "entry threshold 2.0000" in details[3]
    assert details[4] is not None
    assert "stop threshold 4.0000" in details[4]
    assert details[7] is not None
    assert "unavailable" in details[7]


def test_walk_positions_with_reasons_long_only_suppresses_short_entry() -> None:
    """long_only=True must never record overbought_entry -- the branch is
    unreachable, matching generate_signals' own long_only gate."""
    z = np.array([0.0, 2.5])

    positions, detail_code, _ = _walk_positions_with_reasons(
        z, entry=2.0, exit_=0.5, stop=None, long_only=True
    )

    assert positions.tolist() == [0.0, 0.0]
    assert detail_code.tolist() == [None, None]


def test_mean_reversion_explain_signals_matches_generate_signals_transitions() -> None:
    """Every date generate_signals() actually changes AAA's position must
    have a non-None reason, and vice versa -- explain_signals() must
    never invent a reason for a date nothing happened, nor omit one where
    something did."""
    prices = np.concatenate([np.full(40, 100.0), np.linspace(100, 70, 10)])
    data = make_ohlcv("AAA", prices)
    strat = MeanReversionStrategy(
        lookback_period=20, entry_threshold=1.5, exit_threshold=0.5, long_only=True
    )

    signals = strat.generate_signals(data)
    reasons = strat.explain_signals(data)

    assert reasons.detail_code.index.equals(signals.index)
    assert reasons.detail_code.columns.equals(signals.columns)
    assert reasons.details.index.equals(signals.index)
    assert reasons.details.columns.equals(signals.columns)

    values = signals["AAA"].to_numpy()
    previous = np.concatenate([[0.0], values[:-1]])
    changed = np.abs(values - previous) > 1e-12
    has_reason = reasons.detail_code["AAA"].notna().to_numpy()
    assert (changed == has_reason).all()
    # Exactly one transition (flat -> long) drives this whole crash
    # scenario -- the FIRST row generate_signals() goes to 1.0 must read
    # as oversold_entry, not some other branch.
    entry_row = int(np.flatnonzero(changed)[0])
    assert reasons.detail_code["AAA"].iloc[entry_row] == "oversold_entry"


def test_mean_reversion_explain_signals_does_not_affect_generate_signals() -> None:
    """explain_signals() is a pure, independent recomputation -- calling
    it must not change what generate_signals() itself returns."""
    prices = np.concatenate([np.full(40, 100.0), np.linspace(100, 70, 10)])
    data = make_ohlcv("AAA", prices)
    strat = MeanReversionStrategy(
        lookback_period=20, entry_threshold=1.5, exit_threshold=0.5, long_only=True
    )

    before = strat.generate_signals(data)
    strat.explain_signals(data)
    after = strat.generate_signals(data)

    pd.testing.assert_frame_equal(before, after)


def test_buy_and_hold_explain_signals_matches_generate_signals_transitions() -> None:
    """A symbol whose price starts partway through the window (a
    staggered listing date) must read as price_became_available exactly
    on its first valid row -- the only thing this strategy's signal can
    ever depend on -- and every transition generate_signals() actually
    makes must have a matching non-None reason, and vice versa."""
    data_a = make_ohlcv("AAA", np.full(20, 100.0), start="2020-01-01")
    data_b = make_ohlcv("BBB", np.full(20, 50.0), start="2020-01-01").iloc[5:]
    data = pd.concat([data_a, data_b], ignore_index=True)

    strat = BuyAndHoldStrategy()
    signals = strat.generate_signals(data)
    reasons = strat.explain_signals(data)

    for symbol in ("AAA", "BBB"):
        values = signals[symbol].to_numpy()
        previous = np.concatenate([[0.0], values[:-1]])
        changed = np.abs(values - previous) > 1e-12
        has_reason = reasons.detail_code[symbol].notna().to_numpy()
        assert (changed == has_reason).all()

    first_valid_bbb_date = signals.index[5]
    assert (
        reasons.detail_code.at[first_valid_bbb_date, "BBB"] == "price_became_available"
    )
    assert reasons.detail_code.at[signals.index[0], "AAA"] == "price_became_available"


def test_trend_following_explain_signals_reports_crossover_codes() -> None:
    """A clean down-then-up price path forces exactly one bearish and one
    bullish crossover; codes and the fast/slow MA values in the details
    must match generate_signals()' own transitions."""
    prices = np.concatenate([np.linspace(100, 80, 60), np.linspace(80, 120, 60)])
    data = make_ohlcv("AAA", prices, start="2020-01-01")
    strat = TrendFollowingStrategy(fast_window=5, slow_window=20, long_only=False)

    signals = strat.generate_signals(data)
    reasons = strat.explain_signals(data)

    values = signals["AAA"].to_numpy()
    previous = np.concatenate([[0.0], values[:-1]])
    changed = np.abs(values - previous) > 1e-12
    has_reason = reasons.detail_code["AAA"].notna().to_numpy()
    assert (changed == has_reason).all()
    assert set(reasons.detail_code["AAA"].dropna().unique()) <= {
        "bullish_crossover",
        "bearish_crossover",
    }
    # The uptrend leg must eventually produce a bullish crossover, and
    # its details must cite real MA values.
    bullish = reasons.detail_code["AAA"] == "bullish_crossover"
    assert bullish.any()
    bullish_details = reasons.details["AAA"][bullish].iloc[0]
    assert "fast MA" in bullish_details
    assert "crossed above slow MA" in bullish_details


def test_time_series_momentum_binary_explain_signals_reports_entry_codes() -> None:
    prices = np.concatenate([np.full(30, 100.0), np.linspace(100, 160, 40)])
    data = make_ohlcv("AAA", prices, start="2020-01-01")
    strat = TimeSeriesMomentumStrategy(
        lookback_period=20, skip_period=1, signal_scaling="binary", long_only=True
    )

    signals = strat.generate_signals(data)
    reasons = strat.explain_signals(data)
    assert reasons is not None

    values = signals["AAA"].to_numpy()
    previous = np.concatenate([[0.0], values[:-1]])
    changed = np.abs(values - previous) > 1e-12
    has_reason = reasons.detail_code["AAA"].notna().to_numpy()
    assert (changed == has_reason).all()
    codes = set(reasons.detail_code["AAA"].dropna().unique())
    assert codes <= {
        "positive_momentum_entry",
        "negative_momentum_entry",
        "momentum_exit",
    }
    assert "positive_momentum_entry" in codes


@pytest.mark.parametrize("signal_scaling", ["continuous", "volatility_adjusted"])
def test_time_series_momentum_non_binary_explain_signals_returns_none(
    signal_scaling: str,
) -> None:
    """A continuously-scaled signal changes almost every rebalance date --
    the generic pipeline text already explains it fully, so this
    deliberately opts out of a strategy-specific attribution rather than
    inventing a label repeated on nearly every row."""
    prices = np.concatenate([np.full(30, 100.0), np.linspace(100, 160, 40)])
    data = make_ohlcv("AAA", prices, start="2020-01-01")
    strat = TimeSeriesMomentumStrategy(
        lookback_period=20, skip_period=1, signal_scaling=signal_scaling
    )
    assert strat.explain_signals(data) is None


def test_time_series_momentum_volatility_adjusted_masks_zero_volatility() -> None:
    """Regression test: a price jump followed by a long dead-flat stretch
    gives a zero trailing realized volatility while the (longer-lookback)
    momentum score is still positive. `generate_signals()` previously
    computed ``score / volatility`` inline and clipped the result, so
    ``positive / 0 == inf`` became a false full-conviction ``+1.0`` there --
    diverging from the public `volatility_adjusted_momentum()` helper (and
    the Strategy Explorer lab), which both mask a zero-volatility window to
    ``NaN`` (an inconclusive read, not a confident signal). The strategy
    must now agree with the helper: NaN there, filled to `0.0` by
    `_validate_signals()`, never `1.0`.
    """
    from quantlab.features.momentum import volatility_adjusted_momentum

    prices = np.concatenate([np.full(10, 100.0), np.full(90, 110.0)])
    data = make_ohlcv("AAA", prices, start="2020-01-01")
    lookback, skip, vol_window = 60, 0, 20
    strat = TimeSeriesMomentumStrategy(
        lookback_period=lookback,
        skip_period=skip,
        signal_scaling="volatility_adjusted",
        volatility_window=vol_window,
        long_only=False,
    )
    signals = strat.generate_signals(data)["AAA"]

    helper = volatility_adjusted_momentum(
        pd.Series(prices, index=data["timestamp"].unique()),
        lookback,
        skip,
        vol_window,
        252,
    ).clip(-1.0, 1.0)
    expected = helper.fillna(0.0)
    expected.index = signals.index
    pd.testing.assert_series_equal(signals, expected, check_names=False)
    # Not a vacuous comparison -- confirm the zero-volatility window this
    # test targets actually occurs and would previously have been 1.0.
    assert (expected == 0.0).any()


def test_cross_sectional_momentum_explain_signals_reports_selection_codes(
    synthetic_panel: pd.DataFrame,
) -> None:
    strat = CrossSectionalMomentumStrategy(
        lookback_period=60, skip_period=5, top_fraction=0.34, long_short=True
    )
    signals = strat.generate_signals(synthetic_panel)
    reasons = strat.explain_signals(synthetic_panel)
    assert reasons is not None

    for symbol in signals.columns:
        values = signals[symbol].to_numpy()
        previous = np.concatenate([[0.0], values[:-1]])
        changed = np.abs(values - previous) > 1e-12
        has_reason = reasons.detail_code[symbol].notna().to_numpy()
        assert (changed == has_reason).all()

    codes = set(np.unique(reasons.detail_code.to_numpy()[reasons.detail_code.notna()]))
    assert codes <= {
        "entered_top_selection",
        "left_top_selection",
        "entered_bottom_selection",
        "left_bottom_selection",
    }
    assert "entered_top_selection" in codes


def test_pairs_trading_contract(two_symbol_panel: pd.DataFrame) -> None:
    strat = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=120,
        indicator_window=30,
        entry_threshold=1.5,
        exit_threshold=0.5,
    )
    signals = strat.generate_signals(two_symbol_panel)
    _assert_contract(signals)
    # The strategy must actually trade this panel, and legs move in opposite
    # directions whenever a position is on (long one leg, short the other).
    active = signals[(signals["EWA"] != 0) | (signals["EWB"] != 0)]
    assert len(active) > 0
    row = active.iloc[-1]
    assert np.sign(row["EWA"]) == -np.sign(row["EWB"])


def test_pairs_trading_explain_signals_matches_generate_signals_transitions(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """Both legs must carry the SAME reason at the SAME date (one shared
    pair position), matching a direct re-walk of the state machine; every
    other symbol stays None."""
    strat = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=120,
        indicator_window=30,
        entry_threshold=1.5,
        exit_threshold=0.5,
    )
    prices = strat._prices(two_symbol_panel)
    a, b = prices["EWA"], prices["EWB"]
    intercept, beta = rolling_hedge_parameters(
        a, b, strat.formation_window, strat.dynamic_hedge_ratio
    )
    zscore = rolling_zscore(a - intercept - beta * b, strat.indicator_window)
    state, expected_detail_code, expected_details = _walk_pairs_positions_with_reasons(
        zscore.to_numpy(dtype=float),
        strat._stationarity_gate(a, b),
        entry=strat.entry_threshold,
        exit_=strat.exit_threshold,
        stop=strat.stop_threshold,
    )

    reasons = strat.explain_signals(two_symbol_panel)

    pd.testing.assert_series_equal(
        reasons.detail_code["EWA"], reasons.detail_code["EWB"], check_names=False
    )
    assert reasons.detail_code["EWA"].tolist() == list(expected_detail_code)
    assert reasons.details["EWA"].tolist() == list(expected_details)
    codes = set(reasons.detail_code["EWA"].dropna().unique())
    assert codes <= {
        "spread_oversold_entry",
        "spread_overbought_entry",
        "mean_reversion_exit",
        "stop_loss_exit",
        "data_unavailable_exit",
    }
    assert codes  # this panel is designed to actually trade
    assert (state != 0).any()


def test_pairs_trading_explain_signals_says_gate_disabled_when_adf_is_none(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """`adf_pvalue_threshold=None` disables the stationarity gate entirely
    -- an entry's reason text must say so, never claim "stationarity gate
    open" for a gate that was never even evaluated."""
    strat = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=120,
        indicator_window=30,
        entry_threshold=1.5,
        exit_threshold=0.5,
        adf_pvalue_threshold=None,
    )
    reasons = strat.explain_signals(two_symbol_panel)
    entry_details = reasons.details["EWA"][
        reasons.detail_code["EWA"].isin(
            ["spread_oversold_entry", "spread_overbought_entry"]
        )
    ]
    assert not entry_details.empty  # this panel is designed to actually trade
    assert entry_details.str.contains("gate disabled").all()
    assert not entry_details.str.contains("stationarity gate open").any()


def test_pairs_trading_decision_signal_matches_the_real_state_array(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """decision_signal() must return EXACTLY the same discrete `state`
    array (+-1/0) that generate_signals() computes internally -- a pure
    recalculation via the same shared helper, never a reconstruction that
    could diverge."""
    strat = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=120,
        indicator_window=30,
        entry_threshold=1.5,
        exit_threshold=0.5,
    )
    prices = strat._prices(two_symbol_panel)
    a, b = prices["EWA"], prices["EWB"]
    intercept, beta = rolling_hedge_parameters(
        a, b, strat.formation_window, strat.dynamic_hedge_ratio
    )
    zscore = rolling_zscore(a - intercept - beta * b, strat.indicator_window)
    expected_state, _, _ = _walk_pairs_positions_with_reasons(
        zscore.to_numpy(dtype=float),
        strat._stationarity_gate(a, b),
        entry=strat.entry_threshold,
        exit_=strat.exit_threshold,
        stop=strat.stop_threshold,
    )

    decision = strat.decision_signal(two_symbol_panel)

    assert decision is not None
    assert decision["EWA"].tolist() == list(expected_state)
    assert decision["EWB"].tolist() == list(expected_state)
    # Every other symbol in the universe stays 0 -- this strategy never
    # touches them.
    other_columns = [c for c in decision.columns if c not in ("EWA", "EWB")]
    for column in other_columns:
        assert (decision[column] == 0.0).all()


def test_pairs_trading_decision_signal_shares_index_and_columns_with_prices(
    two_symbol_panel: pd.DataFrame,
) -> None:
    strat = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=120,
        indicator_window=30,
        entry_threshold=1.5,
        exit_threshold=0.5,
    )
    prices = strat._prices(two_symbol_panel)

    decision = strat.decision_signal(two_symbol_panel)

    assert decision is not None
    assert decision.index.equals(prices.index)
    assert decision.columns.equals(prices.columns)
    assert np.isfinite(decision.to_numpy(dtype=float)).all()


def test_base_strategy_decision_signal_defaults_to_none() -> None:
    """Every built-in strategy except pairs_trading leaves decision_signal
    at its default -- generate_signals()'s own output is already a
    faithful decision proxy for them."""
    strat = MeanReversionStrategy(
        lookback_period=20, entry_threshold=1.5, exit_threshold=0.5
    )
    assert strat.decision_signal(pd.DataFrame()) is None


def test_validate_decision_signal_rejects_mismatched_shape() -> None:
    reference = pd.DataFrame(
        {"A": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2)
    )
    mismatched = pd.DataFrame(
        {"A": [1.0]}, index=pd.date_range("2020-01-01", periods=1)
    )
    with pytest.raises(StrategyError, match="index and columns"):
        BaseStrategy._validate_decision_signal(mismatched, reference)


def test_validate_decision_signal_rejects_mismatched_columns() -> None:
    idx = pd.date_range("2020-01-01", periods=2)
    reference = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    mismatched = pd.DataFrame({"B": [1.0, 2.0]}, index=idx)
    with pytest.raises(StrategyError, match="index and columns"):
        BaseStrategy._validate_decision_signal(mismatched, reference)


def test_validate_decision_signal_rejects_non_numeric_values() -> None:
    idx = pd.date_range("2020-01-01", periods=2)
    reference = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    non_numeric = pd.DataFrame({"A": ["x", "y"]}, index=idx)
    with pytest.raises(StrategyError, match="numeric"):
        BaseStrategy._validate_decision_signal(non_numeric, reference)


def test_validate_decision_signal_rejects_nan() -> None:
    idx = pd.date_range("2020-01-01", periods=2)
    reference = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    with_nan = pd.DataFrame({"A": [1.0, np.nan]}, index=idx)
    with pytest.raises(StrategyError, match="NaN or Infinity"):
        BaseStrategy._validate_decision_signal(with_nan, reference)


def test_validate_decision_signal_rejects_infinity() -> None:
    idx = pd.date_range("2020-01-01", periods=2)
    reference = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
    with_inf = pd.DataFrame({"A": [1.0, np.inf]}, index=idx)
    with pytest.raises(StrategyError, match="NaN or Infinity"):
        BaseStrategy._validate_decision_signal(with_inf, reference)


def test_decision_signal_never_affects_backtest_numerics(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """decision_signal() is a strictly diagnostic proxy (point 6): a
    monkeypatched version that always returns None (forcing the engine's
    fallback to the raw `signals`) must produce a BIT-IDENTICAL backtest
    -- weights, PnL, costs -- to the real override; only the trade log's
    reason columns may differ."""
    from quantlab.backtesting.engine import BacktestEngine
    from quantlab.config import ExperimentConfig
    from quantlab.execution.execution_model import ExecutionModel
    from quantlab.portfolio.allocator import build_allocator

    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "decision_signal_invariance",
            "data": {
                "instruments": [
                    {"symbol": "EWA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "EWB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
            "strategy": {
                "name": "pairs_trading",
                "parameters": {"symbol_a": "EWA", "symbol_b": "EWB"},
            },  # unused: an instance is passed directly to .run() below
            "portfolio": {
                "allocator": "signal_proportional",
                "rebalance_frequency": "daily",
            },
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000},
        }
    )
    strategy = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=60,
        indicator_window=15,
        entry_threshold=1.0,
        exit_threshold=0.3,
    )
    execution_model = ExecutionModel.from_config(config.execution)
    allocator = build_allocator("signal_proportional")
    with_decision = BacktestEngine().run(
        two_symbol_panel, strategy, allocator, execution_model, config
    )

    class _NoDecisionProxy(PairsTradingStrategy):
        def decision_signal(  # type: ignore[override]
            self, data: pd.DataFrame, features: pd.DataFrame | None = None
        ) -> None:
            return None

    fallback_strategy = _NoDecisionProxy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=60,
        indicator_window=15,
        entry_threshold=1.0,
        exit_threshold=0.3,
    )
    without_decision = BacktestEngine().run(
        two_symbol_panel, fallback_strategy, allocator, execution_model, config
    )

    pd.testing.assert_series_equal(
        with_decision.equity_curve, without_decision.equity_curve
    )
    pd.testing.assert_frame_equal(with_decision.weights, without_decision.weights)
    pd.testing.assert_frame_equal(with_decision.positions, without_decision.positions)
    pd.testing.assert_series_equal(with_decision.returns, without_decision.returns)
    for column in ("commission", "spread_cost", "slippage_cost", "total_cost"):
        pd.testing.assert_series_equal(
            with_decision.trades[column], without_decision.trades[column]
        )


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


def _split_like_data(symbol: str = "AAA") -> pd.DataFrame:
    """40 rows whose adjusted_close diverges from close via a simulated split.

    The first half of adjusted_close is halved relative to close, creating
    a real, structural divergence between the two price series -- not just
    numeric noise -- so ``_prices()`` must produce genuinely different
    matrices for ``price_type="close"`` vs ``"adjusted_close"``.
    """
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100, 140, n)
    adjusted_close = close.copy()
    adjusted_close[:20] /= 2.0
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adjusted_close": adjusted_close,
            "volume": 1_000_000.0,
        }
    )


@pytest.mark.parametrize(
    ("strategy_cls", "kwargs"),
    [
        (BuyAndHoldStrategy, {}),
        (TimeSeriesMomentumStrategy, {"lookback_period": 10, "skip_period": 0}),
        (CrossSectionalMomentumStrategy, {"lookback_period": 10, "skip_period": 0}),
        (MeanReversionStrategy, {"lookback_period": 10}),
        (TrendFollowingStrategy, {"fast_window": 3, "slow_window": 8}),
    ],
)
def test_prices_respects_signal_price_type_per_strategy(
    strategy_cls: type[BaseStrategy], kwargs: dict[str, object]
) -> None:
    """Each strategy's ``_prices()`` must read whichever ``price_type`` it
    was constructed with -- ``"close"`` and ``"adjusted_close"`` must
    produce genuinely different price matrices on data with a real
    divergence between the two fields."""
    data = _split_like_data()
    strategy_close = strategy_cls(price_type="close", **kwargs)  # type: ignore[call-arg]
    strategy_adjusted = strategy_cls(price_type="adjusted_close", **kwargs)  # type: ignore[call-arg]
    assert strategy_close.price_type == "close"
    assert strategy_adjusted.price_type == "adjusted_close"

    prices_close = strategy_close._prices(data)
    prices_adjusted = strategy_adjusted._prices(data)
    assert not prices_close.equals(prices_adjusted)
    pd.testing.assert_series_equal(
        prices_close["AAA"], data.set_index("timestamp")["close"], check_names=False
    )
    pd.testing.assert_series_equal(
        prices_adjusted["AAA"],
        data.set_index("timestamp")["adjusted_close"],
        check_names=False,
    )


def test_prices_respects_signal_price_type_for_pairs_trading() -> None:
    data = pd.concat(
        [_split_like_data("AAA"), _split_like_data("BBB")], ignore_index=True
    )
    strategy_close = PairsTradingStrategy(
        symbol_a="AAA", symbol_b="BBB", formation_window=20, price_type="close"
    )
    strategy_adjusted = PairsTradingStrategy(
        symbol_a="AAA", symbol_b="BBB", formation_window=20, price_type="adjusted_close"
    )
    prices_close = strategy_close._prices(data)
    prices_adjusted = strategy_adjusted._prices(data)
    assert not prices_close.equals(prices_adjusted)


def test_price_type_is_rejected_when_invalid() -> None:
    with pytest.raises(ValueError, match="price_type"):
        BuyAndHoldStrategy(price_type="vwap")
    with pytest.raises(StrategyError, match="price_type"):
        build_strategy("buy_and_hold", {"price_type": "vwap"})


def test_build_strategy_from_config_injects_signal_price_type() -> None:
    """build_strategy_from_config() must inject strategy.signal_price_type
    the same way it already injects periods_per_year -- only when the
    strategy accepts it and the YAML didn't already set it explicitly."""
    from quantlab.backtesting.runner import build_strategy_from_config
    from quantlab.config import ExperimentConfig

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "price_type_injection",
            "data": {
                "instruments": [{"symbol": "AAA", "source": "csv", "calendar": "XNYS"}],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
            },
            "strategy": {"name": "buy_and_hold", "signal_price_type": "close"},
        }
    )
    strategy = build_strategy_from_config(cfg)
    assert strategy.price_type == "close"

    default_cfg = cfg.revalidated_copy(
        update={
            "strategy": cfg.strategy.revalidated_copy(
                update={"signal_price_type": "adjusted_close"}
            )
        }
    )
    assert build_strategy_from_config(default_cfg).price_type == "adjusted_close"
