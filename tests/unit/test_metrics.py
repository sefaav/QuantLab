"""Tests for performance and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.risk import metrics as M
from quantlab.risk.drawdown import drawdown_series, longest_drawdown, max_drawdown
from quantlab.risk.var import historical_cvar, historical_var


def _equity(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def test_max_drawdown_manual_example() -> None:
    """Equity [100, 120, 90, 110] → 90/120 - 1 = -0.25."""
    eq = _equity([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown(eq) == pytest.approx(-0.25)


def test_drawdown_series_zero_at_new_highs() -> None:
    eq = _equity([100.0, 120.0, 90.0, 110.0])
    dd = drawdown_series(eq)
    assert dd.iloc[0] == 0.0  # first point is the peak
    assert dd.iloc[1] == 0.0  # new high
    assert dd.iloc[2] == pytest.approx(-0.25)


def test_longest_drawdown() -> None:
    eq = _equity([100.0, 90.0, 95.0, 105.0, 104.0])
    # Underwater for indices 1,2 (2 periods), then recovers, then 1 period.
    assert longest_drawdown(eq) == 2


# --------------------------------------------------------------------------- #
# Return metrics
# --------------------------------------------------------------------------- #
def test_total_return() -> None:
    eq = _equity([100.0, 150.0])
    assert M.total_return(eq) == pytest.approx(0.5)


def test_cagr_two_years() -> None:
    # 252 periods/year; 504 periods ≈ 2 years; double the money → ~41.4% CAGR.
    eq = pd.Series(
        np.linspace(100.0, 200.0, 505),
        index=pd.date_range("2020-01-01", periods=505),
    )
    got = M.cagr(eq, periods_per_year=252)
    assert got == pytest.approx(2 ** (1 / 2) - 1, rel=1e-2)


def test_annualized_volatility_scaling() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, 2520))
    vol = M.annualized_volatility(rets, periods_per_year=252)
    # ~0.01 * sqrt(252) ≈ 0.1587
    assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.1)


def test_sharpe_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.001, 0.01, 2520))
    assert M.sharpe_ratio(rets, risk_free_rate=0.0, periods_per_year=252) > 0


def test_sharpe_risk_free_reduces_ratio() -> None:
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.001, 0.01, 2520))
    s0 = M.sharpe_ratio(rets, 0.0, 252)
    s1 = M.sharpe_ratio(rets, 0.05, 252)
    assert s1 < s0


def test_sortino_only_penalises_downside() -> None:
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0.001, 0.01, 2520))
    # Sortino >= Sharpe when downside deviation < total std (typical).
    assert M.sortino_ratio(rets, 0.0, 252) >= M.sharpe_ratio(rets, 0.0, 252) - 1e-9


def test_calmar_ratio() -> None:
    eq = _equity([100.0, 120.0, 90.0, 130.0, 140.0])
    calmar = M.calmar_ratio(eq, periods_per_year=252)
    # Net gain over the window (100 -> 140) means CAGR is positive, so a
    # correctly signed Calmar ratio (CAGR / |max drawdown|) must be too.
    assert calmar > 0
    assert np.isfinite(calmar)


# --------------------------------------------------------------------------- #
# VaR / CVaR
# --------------------------------------------------------------------------- #
def test_var_cvar_ordering() -> None:
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0, 0.02, 5000))
    var95 = historical_var(rets, 0.95)
    var99 = historical_var(rets, 0.99)
    cvar95 = historical_cvar(rets, 0.95)
    assert var99 >= var95 > 0
    assert cvar95 >= var95  # expected shortfall is worse than VaR


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #
def test_beta_of_scaled_benchmark() -> None:
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.normal(0, 0.01, 2000))
    strat = 1.5 * bench  # perfectly correlated, 1.5x leverage
    assert M.beta(strat, bench) == pytest.approx(1.5, rel=1e-6)


def test_information_ratio_zero_when_identical() -> None:
    rng = np.random.default_rng(6)
    bench = pd.Series(rng.normal(0, 0.01, 2000))
    assert M.information_ratio(bench, bench) == 0.0


def test_compute_metrics_aggregate() -> None:
    rng = np.random.default_rng(7)
    rets = pd.Series(
        rng.normal(0.0004, 0.01, 1000),
        index=pd.date_range("2020-01-01", periods=1000),
    )
    equity = 100_000 * (1 + rets).cumprod()
    bench = pd.Series(rng.normal(0.0003, 0.01, 1000), index=rets.index)
    m = M.compute_metrics(
        rets, equity, benchmark_returns=bench, risk_free_rate=0.02, periods_per_year=252
    )
    for key in [
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "var_95",
        "cvar_99",
        "beta",
        "alpha",
        "tracking_error",
    ]:
        assert key in m
        assert np.isfinite(m[key])
