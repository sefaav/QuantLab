"""Regression tests for the hardened public feature API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import quantlab.features as features
from quantlab.features.cross_sectional import select_top_bottom
from quantlab.features.mean_reversion import bollinger_bands, half_life, rsi
from quantlab.features.momentum import (
    ma_crossover_signal,
    momentum,
    volatility_adjusted_momentum,
)
from quantlab.features.pipeline import FeaturePipeline
from quantlab.features.returns import cumulative_returns, equity_curve, log_returns
from quantlab.features.technical import donchian_position, macd
from quantlab.features.volatility import (
    annualize_volatility,
    average_true_range,
    downside_volatility,
    historical_volatility,
    rolling_beta,
    rolling_correlation,
)


def test_public_api_exports_all_public_feature_families() -> None:
    expected = {
        "equity_curve",
        "distance_to_moving_average",
        "annualize_volatility",
        "exponential_moving_average",
        "macd",
        "rolling_max",
        "rolling_min",
        "donchian_position",
    }
    assert expected <= set(features.__all__)
    assert all(hasattr(features, name) for name in expected)


@pytest.mark.parametrize("periods", [0, -1, True])
def test_return_periods_reject_non_positive_integers(periods: object) -> None:
    prices = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="periods"):
        features.simple_returns(prices, periods=periods)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="periods"):
        features.forward_returns(prices, periods=periods)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_log_returns_reject_non_positive_prices(bad_price: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        log_returns(pd.Series([1.0, bad_price, 2.0]))


def test_compounding_preserves_internal_missing_returns() -> None:
    returns = pd.Series([np.nan, 0.10, np.nan, 0.20])
    cumulative = cumulative_returns(returns)
    equity = equity_curve(returns, initial=100.0)
    assert cumulative.iloc[0] == 0.0
    assert cumulative.iloc[1] == pytest.approx(0.10)
    assert cumulative.iloc[2:].isna().all()
    assert equity.iloc[:2].tolist() == pytest.approx([100.0, 110.0])
    assert equity.iloc[2:].isna().all()


def test_momentum_and_ma_validate_direct_arguments() -> None:
    prices = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="lookback_period"):
        momentum(prices, lookback_period=-2)
    with pytest.raises(ValueError, match="fast_window"):
        ma_crossover_signal(prices, fast_window=3, slow_window=2)
    with pytest.raises(ValueError, match="window"):
        historical_volatility(prices, window=1)


def test_volatility_adjusted_momentum_preserves_numeric_dtype() -> None:
    prices = pd.Series(np.linspace(100.0, 140.0, 80))
    result = volatility_adjusted_momentum(
        prices, lookback_period=10, volatility_window=5
    )
    assert pd.api.types.is_float_dtype(result.dtype)


def test_flat_rsi_and_bollinger_band_position_are_neutral() -> None:
    prices = pd.Series([100.0] * 30)
    assert rsi(prices, window=5).dropna().eq(50.0).all()
    assert bollinger_bands(prices, window=5)["pct_b"].dropna().eq(0.5).all()


def test_half_life_does_not_stitch_across_missing_rows() -> None:
    sparse = pd.Series([1.0, np.nan, 0.8])
    assert half_life(sparse) == float("inf")


def test_top_bottom_resolves_ties_to_exact_disjoint_counts() -> None:
    scores = pd.DataFrame([[5.0, 5.0, 5.0, 5.0]], columns=list("ABCD"))
    selected = select_top_bottom(scores, top_fraction=0.25, bottom_fraction=0.25)
    assert (selected == 1.0).to_numpy().sum() == 1
    assert (selected == -1.0).to_numpy().sum() == 1
    assert selected.loc[0, "A"] == 1.0
    assert selected.loc[0, "B"] == -1.0


def test_top_bottom_rejects_unavoidably_overlapping_small_universe() -> None:
    scores = pd.DataFrame([[1.0]], columns=["A"])
    with pytest.raises(ValueError, match="disjoint"):
        select_top_bottom(scores, top_fraction=0.1, bottom_fraction=0.1)


def test_macd_has_explicit_signal_warmup() -> None:
    prices = pd.Series(np.arange(1.0, 20.0))
    result = macd(prices, fast_span=2, slow_span=4, signal_span=3)
    assert result["macd"].first_valid_index() == 3
    assert result["signal"].first_valid_index() == 5


def test_flat_donchian_channel_is_neutral() -> None:
    prices = pd.Series([10.0] * 10)
    assert donchian_position(prices, 3).dropna().eq(0.5).all()


def test_downside_volatility_uses_shortfall_and_preserves_missing() -> None:
    returns = pd.Series([-0.02, 0.005, np.nan, 0.02])
    result = downside_volatility(returns, window=1, threshold=0.01, annualize=False)
    expected = pd.Series([0.03, 0.005, np.nan, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_atr_keeps_first_true_range_and_validates_ohlc() -> None:
    high = pd.Series([12.0, 13.0])
    low = pd.Series([10.0, 11.0])
    close = pd.Series([11.0, 12.0])
    result = average_true_range(high, low, close, window=1)
    assert result.iloc[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="high"):
        average_true_range(pd.Series([9.0]), pd.Series([10.0]), pd.Series([9.5]))


@pytest.mark.parametrize(
    ("value", "periods"),
    [(-0.1, 252), (0.1, 0), (float("inf"), 252), (0.1, True)],
)
def test_annualize_volatility_validates_inputs(value: float, periods: object) -> None:
    with pytest.raises(ValueError, match=r"periodic_vol|periods_per_year"):
        annualize_volatility(value, periods_per_year=periods)  # type: ignore[arg-type]


@pytest.mark.parametrize("estimator", [rolling_beta, rolling_correlation])
def test_rolling_dependence_does_not_compress_temporal_gaps(
    estimator: Callable[[pd.Series, pd.Series, int], pd.Series],
) -> None:
    asset = pd.Series([1.0, np.nan, 3.0, 4.0])
    benchmark = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = estimator(asset, benchmark, 3)
    assert result.isna().all()


def test_pipeline_fit_transform_executes_each_transformer_once() -> None:
    calls = 0

    def transformer(data: pd.DataFrame) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return data * 2.0

    data = pd.DataFrame({"A": [1.0, 2.0]})
    result = FeaturePipeline().add("double", transformer).fit_transform(data)
    assert calls == 1
    assert result[("double", "A")].tolist() == [2.0, 4.0]


def test_pipeline_rejects_changed_output_axes() -> None:
    data = pd.DataFrame({"A": [1.0, 2.0]})
    pipeline = FeaturePipeline().add("drop", lambda frame: frame.iloc[1:])
    with pytest.raises(ValueError, match="index"):
        pipeline.fit(data)

    renamed = FeaturePipeline().add(
        "rename", lambda frame: frame.rename(columns={"A": "B"})
    )
    with pytest.raises(ValueError, match="columns"):
        renamed.fit(data)


def test_pipeline_fitted_state_is_enforced_and_invalidated_by_add() -> None:
    data = pd.DataFrame({"A": [1.0, 2.0]})
    pipeline = FeaturePipeline().add("identity", lambda frame: frame)
    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.transform(data)
    pipeline.fit(data)
    pipeline.add("double", lambda frame: frame * 2.0)
    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.transform(data)


def test_pipeline_failed_refit_invalidates_previous_fit() -> None:
    fail = False

    def transformer(data: pd.DataFrame) -> pd.DataFrame:
        if fail:
            raise ValueError("deliberate failure")
        return data

    data = pd.DataFrame({"A": [1.0, 2.0]})
    pipeline = FeaturePipeline().add("identity", transformer).fit(data)
    fail = True
    with pytest.raises(ValueError, match="deliberate"):
        pipeline.fit(data)
    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.transform(data)


@pytest.mark.parametrize(
    ("name", "transformer", "window"),
    [
        ("", lambda frame: frame, None),
        ("x", None, None),
        ("x", lambda frame: frame, 0),
    ],
)
def test_pipeline_add_validates_registration(
    name: str, transformer: object, window: int | None
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FeaturePipeline().add(name, cast(Any, transformer), window=window)


def test_pipeline_metadata_preserves_zero_index_label() -> None:
    data = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
    pipeline = FeaturePipeline().add("identity", lambda frame: frame).fit(data)
    assert pipeline.metadata()[0]["first_valid"] == "0"


def test_series_and_dataframe_typing_branches_are_numerically_equivalent() -> None:
    prices = pd.DataFrame(
        {
            "A": np.linspace(100.0, 140.0, 80),
            "B": np.linspace(80.0, 120.0, 80),
        }
    )
    returns = prices.pct_change(fill_method=None)

    for column in prices:
        pd.testing.assert_series_equal(
            rsi(prices, window=5)[column], rsi(prices[column], window=5)
        )
        pd.testing.assert_series_equal(
            volatility_adjusted_momentum(
                prices, lookback_period=10, volatility_window=5
            )[column],
            volatility_adjusted_momentum(
                prices[column], lookback_period=10, volatility_window=5
            ),
        )
        pd.testing.assert_series_equal(
            equity_curve(returns, initial=100.0)[column],
            equity_curve(returns[column], initial=100.0),
        )
        pd.testing.assert_series_equal(
            donchian_position(prices, window=5)[column],
            donchian_position(prices[column], window=5),
        )

    high = prices + 2.0
    low = prices - 2.0
    for column in prices:
        pd.testing.assert_series_equal(
            average_true_range(high, low, prices, window=5)[column],
            average_true_range(high[column], low[column], prices[column], window=5),
        )
