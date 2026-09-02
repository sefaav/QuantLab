"""Tests for the shared pairs-trading diagnostics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.base import price_matrix
from quantlab.features.pairs_diagnostics import compute_pair_diagnostics, spread


def test_spread_matches_manual_computation() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    a = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=index)
    b = pd.Series([5.0, 5.5, 6.0, 6.5, 7.0], index=index)
    intercept = pd.Series(1.0, index=index)
    beta = pd.Series(2.0, index=index)
    result = spread(a, b, intercept, beta)
    expected = a - 1.0 - 2.0 * b
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_spread_rejects_mismatched_axes() -> None:
    a = pd.Series([1.0, 2.0], index=[0, 1])
    b = pd.Series([1.0, 2.0], index=[0, 2])
    intercept = pd.Series([0.0, 0.0], index=[0, 1])
    beta = pd.Series([1.0, 1.0], index=[0, 1])
    with pytest.raises(ValueError, match="index"):
        spread(a, b, intercept, beta)


def test_compute_pair_diagnostics_on_a_cointegrated_pair(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """``two_symbol_panel`` builds EWB = 0.8 * EWA + 20 + small noise -- a
    strongly cointegrated, mean-reverting-spread pair by construction."""
    prices = price_matrix(two_symbol_panel)
    diagnostics = compute_pair_diagnostics(
        prices,
        "EWA",
        "EWB",
        formation_window=100,
        indicator_window=20,
        dynamic_hedge_ratio=True,
    )
    assert diagnostics.symbol_a == "EWA"
    assert diagnostics.symbol_b == "EWB"
    assert -1.0 <= diagnostics.correlation <= 1.0
    assert diagnostics.correlation > 0.5
    assert diagnostics.hedge_ratio.notna().sum() > 0
    assert diagnostics.spread.notna().sum() > 0
    assert diagnostics.indicator == "zscore"
    assert diagnostics.spread_indicator.notna().sum() > 0
    assert diagnostics.adf_result is not None
    assert diagnostics.adf_result.reject_null is True
    assert diagnostics.cointegration_result is not None
    assert diagnostics.cointegration_result.reject_null is True
    assert np.isfinite(diagnostics.half_life)
    assert diagnostics.half_life > 0
    assert np.isfinite(diagnostics.hedge_ratio_stability)
    assert diagnostics.rolling_adf_pvalue.notna().sum() > 0


def test_compute_pair_diagnostics_rejects_missing_symbol(
    two_symbol_panel: pd.DataFrame,
) -> None:
    prices = price_matrix(two_symbol_panel)
    with pytest.raises(ValueError, match="EWZ"):
        compute_pair_diagnostics(
            prices,
            "EWA",
            "EWZ",
            formation_window=100,
            indicator_window=20,
            dynamic_hedge_ratio=True,
        )


def test_compute_pair_diagnostics_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        compute_pair_diagnostics(
            pd.Series([1.0, 2.0]),  # type: ignore[arg-type]
            "EWA",
            "EWB",
            formation_window=100,
            indicator_window=20,
            dynamic_hedge_ratio=True,
        )


def test_compute_pair_diagnostics_static_hedge_ratio_is_stable_by_construction(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """A static hedge ratio never varies after formation, so its own
    stability diagnostic must read as exactly zero dispersion."""
    prices = price_matrix(two_symbol_panel)
    diagnostics = compute_pair_diagnostics(
        prices,
        "EWA",
        "EWB",
        formation_window=100,
        indicator_window=20,
        dynamic_hedge_ratio=False,
    )
    assert diagnostics.hedge_ratio_stability == pytest.approx(0.0)


@pytest.mark.parametrize("dynamic_hedge_ratio", [True, False])
def test_rolling_adf_pvalue_reproduces_the_live_strategys_own_gate(
    two_symbol_panel: pd.DataFrame, dynamic_hedge_ratio: bool
) -> None:
    """``PairDiagnostics.rolling_adf_pvalue``, thresholded, must equal
    ``PairsTradingStrategy._stationarity_gate`` bar for bar -- the whole
    point of both calling the same shared
    ``periodic_stationarity_pvalues`` function (see its docstring). A
    diagnostic that silently used a DIFFERENT computation (e.g. slicing an
    already-dynamically-refit spread series instead of fitting one fresh
    regression per checkpoint window) would show a pair as stationarity-
    gated on dates where a real backtest of it was not, or vice versa.

    ``two_symbol_panel`` is single-calendar, and ``_stationarity_gate`` is
    called directly here with the SAME ``prices["EWA"]``/``prices["EWB"]``
    series ``compute_pair_diagnostics`` itself uses -- this only proves
    "same function, same input -> same output". The live strategy's own
    entry gate instead reaches ``_stationarity_gate`` via
    ``_native_pair_context``, which feeds it each leg sliced to the
    intersection of both legs' native session dates: for a MIXED-calendar
    pair that input differs from what ``compute_pair_diagnostics`` uses,
    and the two genuinely diverge -- see
    ``test_rolling_adf_pvalue_diverges_from_the_live_gate_under_mixed_
    calendars`` below.
    """
    from quantlab.strategies.pairs_trading import PairsTradingStrategy

    prices = price_matrix(two_symbol_panel)
    formation_window = 100
    indicator_window = 20
    adf_pvalue_threshold = 0.10

    diagnostics = compute_pair_diagnostics(
        prices,
        "EWA",
        "EWB",
        formation_window=formation_window,
        indicator_window=indicator_window,
        dynamic_hedge_ratio=dynamic_hedge_ratio,
    )
    diagnostics_gate = (
        diagnostics.rolling_adf_pvalue.notna()
        & (diagnostics.rolling_adf_pvalue <= adf_pvalue_threshold)
    ).to_numpy()

    strategy = PairsTradingStrategy(
        symbol_a="EWA",
        symbol_b="EWB",
        formation_window=formation_window,
        indicator_window=indicator_window,
        dynamic_hedge_ratio=dynamic_hedge_ratio,
        adf_pvalue_threshold=adf_pvalue_threshold,
    )
    strategy_gate = strategy._stationarity_gate(prices["EWA"], prices["EWB"])

    np.testing.assert_array_equal(diagnostics_gate, strategy_gate)
    # Not a vacuous comparison of two all-False arrays.
    assert strategy_gate.any()


def test_rolling_adf_pvalue_diverges_from_the_live_gate_under_mixed_calendars() -> None:
    """Regression test for a documentation bug: earlier docstrings claimed
    ``rolling_adf_pvalue`` was "the one exception" among this module's
    diagnostics that always reproduces the live entry gate exactly,
    because it calls the exact same ``periodic_stationarity_pvalues``
    FUNCTION. Calling the same function is not the same as reproducing
    the same RESULT: under a mixed-calendar universe, the live gate (via
    ``PairsTradingStrategy._native_pair_context``) feeds that function
    each leg sliced to the intersection of both legs' own native session
    dates, while ``compute_pair_diagnostics`` feeds it the full combined,
    closure-padded timeline. Proven directly here by calling
    ``periodic_stationarity_pvalues`` both ways on the same mixed-calendar
    pair and showing the results genuinely differ on native session dates
    (not just on the padding itself, where they could trivially differ)."""
    from quantlab.data.calendar import is_session_day
    from quantlab.strategies.pairs_trading import periodic_stationarity_pvalues

    dates = pd.date_range("2019-01-01", periods=200, freq="D")  # includes weekends
    is_weekend = dates.weekday >= 5
    rng = np.random.default_rng(5)
    common = np.empty(len(dates))
    common[~is_weekend] = 100.0 + np.cumsum(
        rng.normal(0.05, 1.0, size=int((~is_weekend).sum()))
    )
    last = np.nan
    for i in range(len(dates)):
        if is_weekend[i]:
            common[i] = last
        else:
            last = common[i]
    noise = rng.normal(0.0, 0.5, size=len(dates))
    a = pd.Series(common, index=dates)  # AAA: XNYS, flat-filled on weekends
    b = pd.Series(0.8 * common + 20.0 + noise, index=dates)  # BTC: 24/7, tracks A

    both_open = is_session_day("XNYS", dates)
    native_index = dates[both_open]

    combined_pvalues = periodic_stationarity_pvalues(
        a, b, formation_window=60, stride=10, dynamic_hedge_ratio=True
    )
    native_pvalues = periodic_stationarity_pvalues(
        a.loc[native_index],
        b.loc[native_index],
        formation_window=60,
        stride=10,
        dynamic_hedge_ratio=True,
    )

    # Compare on native session dates only -- a mismatch there proves the
    # divergence isn't just an artefact of the padding/reindex itself.
    combined_on_native = combined_pvalues.loc[native_index]
    assert not combined_on_native.reset_index(drop=True).equals(
        native_pvalues.reset_index(drop=True)
    )


@pytest.mark.parametrize("indicator", ["zscore", "rsi", "percentile"])
def test_compute_pair_diagnostics_indicator_matches_the_strategys_own_series(
    two_symbol_panel: pd.DataFrame, indicator: str
) -> None:
    """``spread_indicator`` must match the SAME series
    ``PairsTradingStrategy._centered_spread_indicator`` computes for a
    given ``indicator`` -- not always the zscore, regardless of which
    indicator was requested."""
    from quantlab.strategies.pairs_trading import _centered_spread_indicator

    prices = price_matrix(two_symbol_panel)
    diagnostics = compute_pair_diagnostics(
        prices,
        "EWA",
        "EWB",
        formation_window=100,
        indicator_window=20,
        dynamic_hedge_ratio=True,
        indicator=indicator,
    )
    assert diagnostics.indicator == indicator
    expected = _centered_spread_indicator(diagnostics.spread, indicator, 20)
    pd.testing.assert_series_equal(
        diagnostics.spread_indicator, expected, check_names=False
    )
