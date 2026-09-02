"""Tests for the multi-asset correlation matrix diagnostic."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantlab.features.correlation import correlation_matrix


def test_correlation_matrix_diagonal_is_one() -> None:
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, 100)),
            "B": 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, 100)),
        },
        index=index,
    )
    matrix = correlation_matrix(prices)
    assert cast(float, matrix.loc["A", "A"]) == pytest.approx(1.0)
    assert cast(float, matrix.loc["B", "B"]) == pytest.approx(1.0)
    assert cast(float, matrix.loc["A", "B"]) == pytest.approx(
        cast(float, matrix.loc["B", "A"])
    )


def test_correlation_matrix_detects_strongly_correlated_assets() -> None:
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    base_returns = rng.normal(0.0004, 0.01, 200)
    a = 100.0 * np.cumprod(1.0 + base_returns)
    b = 50.0 * np.cumprod(1.0 + base_returns + rng.normal(0.0, 0.0005, 200))
    c = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.015, 200))
    prices = pd.DataFrame({"A": a, "B": b, "C": c}, index=index)
    matrix = correlation_matrix(prices)
    ab = cast(float, matrix.loc["A", "B"])
    ac = cast(float, matrix.loc["A", "C"])
    assert ab > 0.9
    assert abs(ac) < ab


def test_correlation_matrix_rejects_unknown_method() -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="method"):
        correlation_matrix(prices, method="bogus")  # type: ignore[arg-type]


def test_correlation_matrix_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        correlation_matrix(pd.Series([1.0, 2.0]))  # type: ignore[arg-type]
