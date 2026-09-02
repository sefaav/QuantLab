"""Tests for stationarity/cointegration/persistence diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.features.stationarity import (
    adf_test,
    cointegration_test,
    hurst_exponent,
)


def _mean_reverting_series(
    n: int = 300, *, seed: int = 0, lam: float = 0.3
) -> pd.Series:
    """A strongly mean-reverting AR(1) series: x_t = (1 - lam) * x_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = (1.0 - lam) * values[t - 1] + rng.normal(0.0, 1.0)
    return pd.Series(values)


def _random_walk(n: int = 300, *, seed: int = 0, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 1.0, n)
    return pd.Series(np.cumsum(steps))


# --------------------------------------------------------------------------- #
# adf_test
# --------------------------------------------------------------------------- #
def test_adf_test_rejects_null_for_a_strongly_mean_reverting_series() -> None:
    result = adf_test(_mean_reverting_series())
    assert result is not None
    assert result.reject_null is True
    assert result.pvalue <= 0.05
    assert "stationarity" in result.interpretation


def test_adf_test_does_not_reject_null_for_a_random_walk() -> None:
    result = adf_test(_random_walk())
    assert result is not None
    assert result.reject_null is False
    assert result.pvalue > 0.05


def test_adf_test_returns_none_for_too_few_observations() -> None:
    assert adf_test(pd.Series(np.arange(10, dtype=float))) is None


def test_adf_test_returns_none_for_a_constant_series() -> None:
    assert adf_test(pd.Series(np.full(50, 3.0))) is None


def test_adf_test_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        adf_test(pd.DataFrame({"a": [1.0, 2.0]}))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_level", [0.0, 1.0, -0.1, 1.5])
def test_adf_test_rejects_bad_significance(bad_level: float) -> None:
    with pytest.raises(ValueError, match="significance"):
        adf_test(_random_walk(), significance=bad_level)


def test_adf_test_critical_values_and_metadata_are_populated() -> None:
    result = adf_test(_mean_reverting_series())
    assert result is not None
    assert set(result.critical_values) == {"1%", "5%", "10%"}
    assert result.n_obs > 0
    assert result.n_lags >= 0


def test_adf_test_returns_none_for_a_non_finite_statistic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite ADF statistic (even with a finite p-value) is a
    numerically degenerate result -- must be treated as inconclusive."""

    def _fake_adfuller(values: object, autolag: str) -> tuple[object, ...]:
        return (float("inf"), 0.01, 1, 100, {"1%": -3.5, "5%": -2.9, "10%": -2.6}, 0.0)

    monkeypatch.setattr(
        "statsmodels.tsa.stattools.adfuller",
        _fake_adfuller,
    )
    assert adf_test(_mean_reverting_series()) is None


def test_adf_test_returns_none_for_a_non_finite_critical_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_adfuller(values: object, autolag: str) -> tuple[object, ...]:
        return (
            -3.0,
            0.01,
            1,
            100,
            {"1%": float("nan"), "5%": -2.9, "10%": -2.6},
            0.0,
        )

    monkeypatch.setattr(
        "statsmodels.tsa.stattools.adfuller",
        _fake_adfuller,
    )
    assert adf_test(_mean_reverting_series()) is None


# --------------------------------------------------------------------------- #
# cointegration_test
# --------------------------------------------------------------------------- #
def test_cointegration_test_detects_a_cointegrated_pair() -> None:
    common_trend = _random_walk(seed=1)
    noise = pd.Series(np.random.default_rng(2).normal(0.0, 0.5, len(common_trend)))
    a = common_trend
    b = common_trend * 1.5 + noise
    result = cointegration_test(a, b)
    assert result is not None
    assert result.reject_null is True
    assert result.pvalue <= 0.05


def test_cointegration_test_does_not_reject_null_for_independent_walks() -> None:
    a = _random_walk(seed=10)
    b = _random_walk(seed=20)
    result = cointegration_test(a, b)
    assert result is not None
    assert result.reject_null is False


def test_cointegration_test_returns_none_for_near_perfect_collinearity() -> None:
    """``b = 2 * a`` is (near-)perfectly collinear -- statsmodels itself
    warns the test is numerically unreliable in this case (a spurious
    ``statistic=-inf``/``pvalue=0.0`` "confident" result otherwise). Must
    be treated as inconclusive (``None``), not returned as evidence of a
    stable long-run relationship."""
    a = _random_walk(seed=3)
    b = a * 2.0
    assert cointegration_test(a, b) is None


def test_cointegration_test_returns_none_for_a_non_finite_critical_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite critical value is as numerically degenerate as a
    non-finite statistic/p-value -- must be treated as inconclusive."""

    def _fake_coint(a: object, b: object) -> tuple[float, float, list[float]]:
        return (-3.0, 0.01, [float("nan"), -3.3, -3.0])

    monkeypatch.setattr("statsmodels.tsa.stattools.coint", _fake_coint)
    a = _random_walk(seed=1)
    b = _random_walk(seed=2)
    assert cointegration_test(a, b) is None


def test_cointegration_test_returns_none_for_too_few_observations() -> None:
    short_a = pd.Series(np.arange(10, dtype=float))
    short_b = pd.Series(np.arange(10, dtype=float) * 2)
    assert cointegration_test(short_a, short_b) is None


def test_cointegration_test_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        cointegration_test(pd.DataFrame({"a": [1.0]}), pd.Series([1.0]))  # type: ignore[arg-type]


def test_cointegration_test_rejects_mismatched_axes() -> None:
    a = pd.Series(np.arange(30, dtype=float))
    b = pd.Series(np.arange(30, dtype=float), index=np.arange(30, 60))
    with pytest.raises(ValueError, match="index"):
        cointegration_test(a, b)


# --------------------------------------------------------------------------- #
# hurst_exponent
# --------------------------------------------------------------------------- #
def test_hurst_exponent_is_low_for_a_mean_reverting_series() -> None:
    h = hurst_exponent(_mean_reverting_series(n=500, lam=0.5))
    assert h < 0.4


def test_hurst_exponent_is_near_half_for_a_random_walk() -> None:
    h = hurst_exponent(_random_walk(n=2000))
    assert 0.35 < h < 0.65


def test_hurst_exponent_is_high_for_a_trending_series() -> None:
    """A deterministic straight line isn't the right synthetic case here: its
    lag-k differences are a constant plus noise, so their *variance* doesn't
    grow with lag at all (the estimator reads that as H ~= 0, not high).
    Genuine persistence needs positively autocorrelated *increments* (each
    step likely continues the last one's direction) accumulated into a walk
    -- the standard way to simulate trending/persistent fBm-like data."""
    rng = np.random.default_rng(3)
    increments = np.zeros(500)
    for t in range(1, len(increments)):
        increments[t] = 0.6 * increments[t - 1] + rng.normal(0.0, 1.0)
    trend = pd.Series(np.cumsum(increments))
    h = hurst_exponent(trend)
    assert h > 0.6


def test_hurst_exponent_is_nan_for_too_short_a_series() -> None:
    assert np.isnan(hurst_exponent(pd.Series(np.arange(5, dtype=float)), max_lag=20))


def test_hurst_exponent_is_nan_for_a_constant_series() -> None:
    assert np.isnan(hurst_exponent(pd.Series(np.full(100, 5.0))))


def test_hurst_exponent_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        hurst_exponent(pd.DataFrame({"a": [1.0, 2.0]}))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_lag", [0, 1, -5])
def test_hurst_exponent_rejects_bad_max_lag(bad_lag: int) -> None:
    with pytest.raises(ValueError, match="max_lag"):
        hurst_exponent(_random_walk(), max_lag=bad_lag)
