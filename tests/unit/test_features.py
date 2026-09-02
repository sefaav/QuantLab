"""Tests for feature engineering.

The look-ahead-safety tests are the important ones: a feature at row *t* must
never depend on data strictly after *t*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.features.cross_sectional import (
    cross_sectional_zscore,
    select_top_bottom,
)
from quantlab.features.mean_reversion import (
    half_life,
    rolling_percentile_rank,
    rolling_zscore,
    rsi,
)
from quantlab.features.momentum import (
    cross_sectional_momentum_persistence,
    ma_crossover_signal,
    momentum,
    momentum_persistence,
)
from quantlab.features.pipeline import FeaturePipeline
from quantlab.features.returns import (
    cumulative_returns,
    forward_returns,
    simple_returns,
)
from quantlab.features.volatility import realized_volatility


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def test_simple_returns_manual_example() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    rets = simple_returns(prices)
    assert np.isnan(rets.iloc[0])
    assert rets.iloc[1] == pytest.approx(0.10)
    assert rets.iloc[2] == pytest.approx(-0.10)


def test_cumulative_returns() -> None:
    rets = pd.Series([np.nan, 0.10, -0.10])
    cum = cumulative_returns(rets)
    # (1)(1.1)(0.9) - 1 = -0.01
    assert cum.iloc[-1] == pytest.approx(0.99 - 1.0)


def test_forward_returns_is_future_shifted() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    fwd = forward_returns(prices, periods=1)
    # forward return at t=0 equals the *next* simple return.
    assert fwd.iloc[0] == pytest.approx(0.10)
    assert np.isnan(fwd.iloc[-1])  # nothing after the last bar


# --------------------------------------------------------------------------- #
# Momentum: strictly rising prices → positive momentum
# --------------------------------------------------------------------------- #
def test_momentum_positive_for_rising_prices(rising_prices: np.ndarray) -> None:
    prices = pd.Series(rising_prices)
    mom = momentum(prices, lookback_period=20, skip_period=0)
    assert mom.dropna().iloc[-1] > 0


def test_momentum_skip_uses_only_past() -> None:
    prices = pd.Series(np.linspace(100, 200, 100))
    mom = momentum(prices, lookback_period=30, skip_period=5)
    # Value at index t must not change if we truncate the series after t.
    t = 60
    truncated = momentum(prices.iloc[: t + 1], lookback_period=30, skip_period=5)
    assert truncated.iloc[t] == pytest.approx(mom.iloc[t])


def test_momentum_rejects_bad_skip() -> None:
    with pytest.raises(ValueError, match="skip_period"):
        momentum(pd.Series([1.0, 2.0, 3.0]), lookback_period=5, skip_period=5)


def test_momentum_persistence_pairs_past_score_with_future_return() -> None:
    prices = pd.Series(np.linspace(100, 200, 100))
    paired = momentum_persistence(
        prices, lookback_period=20, skip_period=0, holding_period=5
    )
    assert list(paired.columns) == ["past_momentum", "future_return"]
    assert not paired.empty
    # A strictly rising series must show positive past momentum for every
    # row that survives dropna() (the trailing/leading warm-up is excluded).
    assert (paired["past_momentum"] > 0).all()
    assert (paired["future_return"] > 0).all()


def test_cross_sectional_momentum_persistence_with_known_future_ranking() -> None:
    """5 assets with distinct constant growth rates: the momentum ranking
    (by construction) exactly matches the future-return ranking at every
    date, so the Spearman rank correlation must be (near) perfect and the
    top-minus-bottom spread must be strictly positive throughout -- the
    concrete counter-example to a single-asset past-vs-future-return
    scatter, which cannot even express a cross-sectional ranking claim."""
    n = 120
    growth_rates = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020]
    prices = pd.DataFrame(
        {
            f"A{i}": 100.0 * (1.0 + rate) ** np.arange(n)
            for i, rate in enumerate(growth_rates)
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    result = cross_sectional_momentum_persistence(
        prices,
        lookback_period=20,
        skip_period=0,
        holding_period=5,
        top_fraction=0.2,
        bottom_fraction=0.2,
    )
    assert list(result.columns) == [
        "rank_correlation",
        "top_return",
        "bottom_return",
        "top_minus_bottom",
    ]
    assert not result.empty
    assert (result["rank_correlation"] > 0.99).all()
    assert (result["top_minus_bottom"] > 0).all()


def test_cross_sectional_momentum_persistence_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        cross_sectional_momentum_persistence(
            pd.Series([1.0, 2.0]),  # type: ignore[arg-type]
            lookback_period=5,
            skip_period=0,
            holding_period=1,
        )


def test_cross_sectional_momentum_persistence_skips_dates_with_too_few_assets() -> None:
    """A date with fewer than 3 scored assets cannot support a meaningful
    rank correlation -- it must be excluded entirely, not produce a NaN row."""
    n = 60
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.01 ** np.arange(n)),
            "B": 100.0 * (1.02 ** np.arange(n)),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    result = cross_sectional_momentum_persistence(
        prices, lookback_period=10, skip_period=0, holding_period=5
    )
    assert result.empty


def test_momentum_persistence_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        momentum_persistence(
            pd.DataFrame({"a": [1.0, 2.0]}),  # type: ignore[arg-type]
            lookback_period=5,
            skip_period=0,
            holding_period=1,
        )


def test_ma_crossover_sign() -> None:
    prices = pd.Series(np.linspace(100, 200, 60))
    sig = ma_crossover_signal(prices, fast_window=5, slow_window=20)
    # In a persistent uptrend the fast MA sits above the slow MA → +1.
    assert sig.dropna().iloc[-1] == 1.0


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
def test_realized_volatility_scales_with_noise() -> None:
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 0.005, 300))
    wild = pd.Series(rng.normal(0, 0.02, 300))
    v_calm = realized_volatility(calm, window=63).dropna().mean()
    v_wild = realized_volatility(wild, window=63).dropna().mean()
    assert v_wild > v_calm


# --------------------------------------------------------------------------- #
# Mean reversion
# --------------------------------------------------------------------------- #
def test_rolling_zscore_bounds() -> None:
    prices = pd.Series(np.concatenate([np.full(30, 100.0), [130.0]]))
    z = rolling_zscore(prices, window=20)
    # A sharp jump above a flat window yields a large positive z-score.
    assert z.iloc[-1] > 3


def test_rsi_range() -> None:
    rng = np.random.default_rng(1)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    r = rsi(prices, window=14).dropna()
    assert r.between(0, 100).all()


def test_rolling_percentile_rank_is_one_for_a_new_high() -> None:
    prices = pd.Series(np.arange(1.0, 31.0))  # strictly increasing.
    rank = rolling_percentile_rank(prices, window=20)
    # The last observation of a strictly increasing window is its max.
    assert rank.iloc[-1] == pytest.approx(1.0)


def test_rolling_percentile_rank_is_lowest_for_a_new_low() -> None:
    prices = pd.Series(np.arange(30.0, 0.0, -1.0))  # strictly decreasing.
    rank = rolling_percentile_rank(prices, window=20)
    # pandas' rank(pct=True) is 1-indexed, so the minimum of a 20-window
    # scores 1/20, not exactly 0 -- still the lowest rank in that window.
    assert rank.iloc[-1] == pytest.approx(1.0 / 20.0)


def test_rolling_percentile_rank_is_bounded_and_nan_during_warmup() -> None:
    rng = np.random.default_rng(3)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)).clip(min=-90))
    rank = rolling_percentile_rank(prices, window=20)
    assert rank.iloc[:19].isna().all()
    assert rank.dropna().between(0.0, 1.0).all()


def test_half_life_detects_mean_reversion() -> None:
    # AR(1) with phi < 1 mean-reverts; half-life should be finite and positive.
    rng = np.random.default_rng(2)
    n = 500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(x))
    assert 0 < hl < np.inf


# --------------------------------------------------------------------------- #
# Cross-sectional: per-date normalisation
# --------------------------------------------------------------------------- #
def test_cross_sectional_zscore_row_mean_zero() -> None:
    df = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [2.0, 4.0], "C": [3.0, 6.0]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    z = cross_sectional_zscore(df)
    # Each row standardised independently → row mean ≈ 0.
    assert z.mean(axis=1).abs().max() < 1e-9


def test_select_top_bottom_counts() -> None:
    df = pd.DataFrame(
        {"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]},
        index=pd.date_range("2020-01-01", periods=1),
    )
    sel = select_top_bottom(df, top_fraction=0.25, bottom_fraction=0.25)
    assert sel.at[df.index[0], "A"] == 1.0  # best → long
    assert sel.at[df.index[0], "D"] == -1.0  # worst → short
    assert sel.at[df.index[0], "B"] == 0.0


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_feature_pipeline_records_metadata() -> None:
    prices = pd.DataFrame(
        np.linspace(100, 200, 80).reshape(-1, 1),
        columns=["AAA"],
        index=pd.date_range("2020-01-01", periods=80),
    )
    pipe = (
        FeaturePipeline()
        .add("mom20", lambda d: momentum(d, 20), params={"lookback": 20}, window=20)
        .add("z10", lambda d: rolling_zscore(d, 10), window=10)
    )
    out = pipe.fit_transform(prices)
    assert pipe.feature_names == ["mom20", "z10"]
    # The min usable date is driven by the longest warm-up (mom20 → 20 rows).
    assert pipe.min_usable_date is not None
    assert pipe.min_usable_date >= prices.index[20]
    assert isinstance(out.columns, pd.MultiIndex)
    meta = pipe.metadata()
    assert meta[0]["generated_nans"] >= 20


# --------------------------------------------------------------------------- #
# Public API (quantlab.features.__all__)
# --------------------------------------------------------------------------- #
def test_features_public_api_names_are_all_importable() -> None:
    """Every name in `quantlab.features.__all__` must actually resolve on
    the package -- in particular the stationarity/correlation/pairs-
    diagnostics/efficiency-ratio/momentum-persistence surface added
    alongside the Strategy Explorer feature, which `__init__.py` had
    stopped re-exporting even though its own docstring calls it the
    public API."""
    import quantlab.features as features

    assert features.__all__, "features.__all__ must not be empty"
    for name in features.__all__:
        assert hasattr(features, name), f"quantlab.features.{name} is missing"

    for name in (
        "efficiency_ratio",
        "momentum_persistence",
        "cross_sectional_momentum_persistence",
        "correlation_matrix",
        "ADFResult",
        "CointegrationResult",
        "adf_test",
        "cointegration_test",
        "hurst_exponent",
        "PairDiagnostics",
        "compute_pair_diagnostics",
    ):
        assert name in features.__all__, f"{name} missing from features.__all__"
