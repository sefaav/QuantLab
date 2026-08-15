"""Cross-check quantlab.risk metrics against empyrical and statsmodels.

Unit tests written by the same author who wrote the formula tend to share
its blind spots: a wrong ddof or a forgotten annualisation term looks
"correct" to both. These tests instead compare against an independent,
widely used implementation (empyrical) or a general-purpose regression
library (statsmodels) that was not written with quantlab's conventions in
mind, so a shared mistake is far less likely.

Matching requires lining up conventions first — annualisation factor,
whether the risk-free rate is annual or per-period, ddof — not assuming the
two libraries mean the same thing by the same name. Where a convention
genuinely differs (alpha's arithmetic vs geometric annualisation, see
below), the test reconciles it explicitly rather than papering over it with
a loose tolerance.
"""

from __future__ import annotations

from typing import Protocol, cast

import empyrical
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from quantlab.risk import metrics as qm
from quantlab.risk.drawdown import max_drawdown
from quantlab.risk.var import historical_cvar, historical_var
from quantlab.strategies.pairs_trading import _ols_coefficients

RISK_FREE_ANNUAL = 0.02
# 1001 observations makes (n - 1) * cutoff an exact integer at the 0.05 and
# 0.01 cutoffs used below, so the VaR/CVaR quantile lands on an actual order
# statistic instead of an interpolated point — the two implementations then
# select exactly the same tail observations instead of merely close ones.
N_OBSERVATIONS = 1001


class _SharpeReference(Protocol):
    def __call__(
        self,
        returns: pd.Series,
        *,
        risk_free: float,
        annualization: int,
    ) -> float: ...


class _SortinoReference(Protocol):
    def __call__(
        self,
        returns: pd.Series,
        *,
        required_return: float,
        annualization: int,
    ) -> float: ...


# Empyrical has no type information. These local protocols describe the
# supported call shapes used here, including their genuinely floating rates.
_empyrical_sharpe = cast(_SharpeReference, empyrical.sharpe_ratio)
_empyrical_sortino = cast(_SortinoReference, empyrical.sortino_ratio)

ANNUALIZATION_CASES = [(0, 12), (7, 252), (123, 365)]


def _synthetic_returns(seed: int) -> tuple[pd.Series, pd.Series]:
    """Return a strategy/benchmark return-series pair for a given seed."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=N_OBSERVATIONS, freq="D")
    benchmark = pd.Series(rng.normal(0.0004, 0.010, N_OBSERVATIONS), index=index)
    # A non-trivial market loading exercises beta/alpha more strongly than two
    # independent series whose sample beta happens to sit near zero.
    strategy = (
        0.0002
        + 0.8 * benchmark
        + pd.Series(rng.normal(0.0, 0.006, N_OBSERVATIONS), index=index)
    )
    return strategy, benchmark


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_sharpe_ratio_matches_empyrical(seed: int, periods_per_year: int) -> None:
    returns, _ = _synthetic_returns(seed)
    ours = qm.sharpe_ratio(returns, RISK_FREE_ANNUAL, periods_per_year)
    # empyrical's risk_free is a per-period rate; ours is annual.
    reference = _empyrical_sharpe(
        returns,
        risk_free=RISK_FREE_ANNUAL / periods_per_year,
        annualization=periods_per_year,
    )
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_sortino_ratio_matches_empyrical(seed: int, periods_per_year: int) -> None:
    returns, _ = _synthetic_returns(seed)
    ours = qm.sortino_ratio(returns, RISK_FREE_ANNUAL, periods_per_year)
    reference = _empyrical_sortino(
        returns,
        required_return=RISK_FREE_ANNUAL / periods_per_year,
        annualization=periods_per_year,
    )
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_cagr_matches_empyrical_annual_return(seed: int, periods_per_year: int) -> None:
    returns, _ = _synthetic_returns(seed)
    equity = qm.equity_from_returns(returns)
    ours = qm.cagr(equity, periods_per_year)
    reference = empyrical.annual_return(returns, annualization=periods_per_year)
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_total_return_matches_empyrical(seed: int) -> None:
    returns, _ = _synthetic_returns(seed)
    equity = qm.equity_from_returns(returns)
    ours = qm.total_return(equity)
    reference = empyrical.cum_returns_final(returns)
    assert ours == pytest.approx(reference, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_annualized_volatility_matches_empyrical(
    seed: int, periods_per_year: int
) -> None:
    returns, _ = _synthetic_returns(seed)
    ours = qm.annualized_volatility(returns, periods_per_year)
    reference = empyrical.annual_volatility(returns, annualization=periods_per_year)
    assert ours == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_max_drawdown_matches_empyrical(seed: int) -> None:
    returns, _ = _synthetic_returns(seed)
    equity = qm.equity_from_returns(returns)
    ours = max_drawdown(equity)
    reference = empyrical.max_drawdown(returns)
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_calmar_ratio_matches_empyrical(seed: int, periods_per_year: int) -> None:
    returns, _ = _synthetic_returns(seed)
    equity = qm.equity_from_returns(returns)
    ours = qm.calmar_ratio(equity, periods_per_year)
    reference = empyrical.calmar_ratio(returns, annualization=periods_per_year)
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize(("confidence", "cutoff"), [(0.95, 0.05), (0.99, 0.01)])
def test_historical_var_matches_empyrical(
    seed: int, confidence: float, cutoff: float
) -> None:
    returns, _ = _synthetic_returns(seed)
    ours = historical_var(returns, confidence)
    # empyrical reports the raw (typically negative) quantile return; ours
    # reports the loss as a non-negative magnitude, so the sign is flipped.
    reference = empyrical.value_at_risk(returns, cutoff=cutoff)
    assert ours == pytest.approx(max(0.0, -reference), rel=1e-9)


@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize(("confidence", "cutoff"), [(0.95, 0.05), (0.99, 0.01)])
def test_historical_cvar_matches_empyrical(
    seed: int, confidence: float, cutoff: float
) -> None:
    returns, _ = _synthetic_returns(seed)
    ours = historical_cvar(returns, confidence)
    reference = empyrical.conditional_value_at_risk(returns, cutoff=cutoff)
    assert ours == pytest.approx(max(0.0, -reference), rel=1e-9)


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_beta_matches_empyrical(seed: int) -> None:
    returns, benchmark = _synthetic_returns(seed)
    ours = qm.beta(returns, benchmark)
    reference = empyrical.beta(returns, benchmark)
    assert ours == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_annualized_alpha_matches_empyrical_once_reconciled(
    seed: int, periods_per_year: int
) -> None:
    """Compare alpha after reconciling the two annualisation conventions.

    quantlab annualises the mean per-period alpha arithmetically
    (``* periods_per_year``); empyrical compounds it geometrically
    (``(1 + alpha) ** periods_per_year - 1``). These genuinely differ (by a
    few percent at daily-equity alpha magnitudes) and neither is "the
    reference" — so instead of a loose tolerance, invert empyrical's
    compounding to recover its per-period alpha and re-annualise it
    arithmetically. If both implementations compute the same underlying
    per-period alpha (same beta, same risk-free adjustment, same mean),
    this matches to float precision; a real bug in either would not.
    """
    returns, benchmark = _synthetic_returns(seed)
    ours = qm.annualized_alpha(returns, benchmark, RISK_FREE_ANNUAL, periods_per_year)
    reference = empyrical.alpha(
        returns,
        benchmark,
        risk_free=RISK_FREE_ANNUAL / periods_per_year,
        annualization=periods_per_year,
    )
    implied_period_alpha = (1.0 + reference) ** (1.0 / periods_per_year) - 1.0
    reconciled = implied_period_alpha * periods_per_year
    assert ours == pytest.approx(reconciled, rel=1e-6)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_tracking_error_matches_empyrical(seed: int, periods_per_year: int) -> None:
    returns, benchmark = _synthetic_returns(seed)
    ours = qm.tracking_error(returns, benchmark, periods_per_year)
    reference = empyrical.annual_volatility(
        returns - benchmark, annualization=periods_per_year
    )
    assert ours == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_information_ratio_matches_empyrical(seed: int, periods_per_year: int) -> None:
    returns, benchmark = _synthetic_returns(seed)
    ours = qm.information_ratio(returns, benchmark, periods_per_year)
    # Empyrical's excess_sharpe is per-period; QuantLab annualises the ratio.
    reference = empyrical.excess_sharpe(returns, benchmark) * np.sqrt(periods_per_year)
    assert ours == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize(("seed", "periods_per_year"), ANNUALIZATION_CASES)
def test_rolling_sharpe_matches_empyrical(seed: int, periods_per_year: int) -> None:
    returns, _ = _synthetic_returns(seed)
    window = 63
    ours = qm.rolling_sharpe_ratio(
        returns, window, RISK_FREE_ANNUAL, periods_per_year
    ).dropna()
    reference = empyrical.roll_sharpe_ratio(
        returns,
        window,
        risk_free=RISK_FREE_ANNUAL / periods_per_year,
        annualization=periods_per_year,
    )
    assert isinstance(reference, pd.Series)
    pd.testing.assert_series_equal(
        ours, reference, check_names=False, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pairs_ols_hedge_ratio_matches_statsmodels(seed: int) -> None:
    """Cross-check the pairs-trading hedge-ratio regression against statsmodels.

    ``_ols_coefficients`` solves the same ``y = intercept + slope * x`` via
    ``np.linalg.lstsq`` (SVD-based) that statsmodels solves via QR/pinv —
    different numerical paths to the same least-squares problem, so any
    real disagreement points at a formula bug rather than solver noise.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(100.0, 5.0, 300).cumsum()
    y = 2.0 * x + rng.normal(0.0, 3.0, 300) + 10.0

    intercept, slope = _ols_coefficients(x, y)
    reference = sm.OLS(y, sm.add_constant(x)).fit().params

    assert intercept == pytest.approx(reference[0], rel=1e-6)
    assert slope == pytest.approx(reference[1], rel=1e-6)
