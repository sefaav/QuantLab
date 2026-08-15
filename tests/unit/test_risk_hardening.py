"""Regression tests for the public risk-analysis contracts."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import quantlab.risk as risk
from quantlab.config import ExperimentConfig
from quantlab.risk import metrics as M
from quantlab.risk.drawdown import drawdown_series, max_drawdown_details
from quantlab.risk.exposure import gross_exposure_series
from quantlab.risk.stress import delay_execution, remove_best_days, scale_costs
from quantlab.risk.var import historical_cvar, historical_var


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_annualization_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        M.sharpe_ratio(pd.Series([0.01, -0.01]), periods_per_year=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_risk_free_rate_must_be_a_finite_real(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        M.sharpe_ratio(pd.Series([0.01, -0.01]), risk_free_rate=value)  # type: ignore[arg-type]


def test_equity_from_returns_rejects_invented_or_impossible_periods() -> None:
    with pytest.raises(ValueError, match="missing"):
        M.equity_from_returns(pd.Series([0.01, np.nan]))
    with pytest.raises(ValueError, match="infinite"):
        M.equity_from_returns(pd.Series([0.01, np.inf]))
    with pytest.raises(ValueError, match="-100%"):
        M.equity_from_returns(pd.Series([-1.01]))


def test_sortino_uses_full_sample_downside_deviation() -> None:
    returns = pd.Series([0.1, 0.1, 0.1, -0.1])
    assert M.sortino_ratio(returns, periods_per_year=1) == pytest.approx(1.0)


def test_constant_distribution_moments_are_nan_without_scipy_warnings() -> None:
    returns = pd.Series([0.01] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(M.skewness(returns))
        assert np.isnan(M.kurtosis(returns))


def test_compute_metrics_rejects_misaligned_or_inconsistent_equity() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    returns = pd.Series([0.0, 0.1, 0.0], index=index)
    with pytest.raises(ValueError, match="same index"):
        M.compute_metrics(returns, pd.Series([1.0, 1.1, 1.1]))
    inconsistent = pd.Series([1.0, 1.2, 1.2], index=index)
    with pytest.raises(ValueError, match="inconsistent"):
        M.compute_metrics(returns, inconsistent)


def test_drawdown_rejects_invalid_equity_and_empty_details_are_complete() -> None:
    with pytest.raises(ValueError, match="missing"):
        drawdown_series(pd.Series([1.0, np.nan]))
    with pytest.raises(ValueError, match="negative"):
        drawdown_series(pd.Series([1.0, -0.1]))
    with pytest.raises(ValueError, match="unique"):
        drawdown_series(pd.Series([1.0, 0.9], index=[0, 0]))
    with pytest.raises(ValueError, match="positive again"):
        drawdown_series(pd.Series([1.0, 0.0, 0.1]))
    assert max_drawdown_details(pd.Series(dtype=float))["depth"] == 0.0


def test_exposure_rejects_missing_weights_instead_of_treating_them_as_cash() -> None:
    weights = pd.DataFrame({"A": [np.nan], "B": [np.nan]})
    with pytest.raises(ValueError, match="missing"):
        gross_exposure_series(weights)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, True, np.inf])
def test_var_confidence_is_strictly_between_zero_and_one(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        historical_var(pd.Series([0.01, -0.02]), confidence=confidence)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        historical_cvar(pd.Series([0.01, -0.02]), confidence=confidence)  # type: ignore[arg-type]


def test_remove_best_days_is_positional_with_duplicate_labels() -> None:
    returns = pd.Series([0.10, 0.05, np.nan, -0.01], index=[0, 0, 1, 2])
    stressed = remove_best_days(returns, 1)
    assert stressed.iloc[0] == 0.0
    assert stressed.iloc[1] == 0.05
    assert np.isnan(stressed.iloc[2])


def test_delay_execution_only_fills_new_leading_periods() -> None:
    delayed = delay_execution(pd.Series([0.1, np.nan, 0.2]), 1)
    expected = pd.Series([0.0, 0.1, np.nan])
    pd.testing.assert_series_equal(delayed, expected)


@pytest.mark.parametrize("multiplier", [-1.0, np.inf, True])
def test_scale_costs_validates_direct_inputs(
    sample_config: ExperimentConfig, multiplier: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        scale_costs(sample_config, commission_mult=multiplier)  # type: ignore[arg-type]


def test_public_risk_api_exports_metric_and_stress_helpers() -> None:
    expected = {
        "drawdown_durations",
        "equity_from_returns",
        "kurtosis",
        "rolling_sharpe_ratio",
        "skewness",
        "delay_execution",
        "remove_best_days",
        "scale_costs",
    }
    assert expected <= set(risk.__all__)
    assert all(hasattr(risk, name) for name in expected)
