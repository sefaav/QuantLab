"""Stationarity, long-run-relationship and persistence diagnostics.

Structured results (never a bare float) so a caller always has the
statistic, the null/alternative hypotheses and a plain-language
interpretation available, not just a pass/fail number.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.features._validation import (
    finite_real,
    numeric_pandas,
    positive_int,
    same_axes,
)
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

#: Below this many non-missing observations, both ADF and Engle-Granger are
#: considered inconclusive rather than numerically unstable.
_MIN_TEST_OBSERVATIONS = 20


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey-Fuller stationarity test outcome for one series.

    H0: the series has a unit root (is non-stationary). H1: the series is
    stationary. A low ``pvalue`` is evidence against H0 for the sample
    tested -- it does not prove stationarity, and says nothing about
    whether that property will hold going forward.
    """

    statistic: float
    pvalue: float
    n_lags: int
    n_obs: int
    critical_values: dict[str, float]
    significance: float
    reject_null: bool
    interpretation: str


def adf_test(series: pd.Series, *, significance: float = 0.05) -> ADFResult | None:
    """Run an Augmented Dickey-Fuller test; ``None`` when inconclusive.

    Wraps ``statsmodels.tsa.stattools.adfuller`` with ``autolag="AIC"``.
    Returns ``None`` (never a raised error) for fewer than 20 observations,
    a constant series, or a numerical failure inside statsmodels -- these
    are "cannot conclude anything" cases, not test failures to propagate.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    validated = numeric_pandas(series, name="series")
    level = _significance(significance)
    values = validated.dropna().to_numpy(dtype=float)
    if len(values) < _MIN_TEST_OBSERVATIONS or np.allclose(values, values[0]):
        return None
    try:
        from statsmodels.tsa.stattools import adfuller

        statistic, pvalue, n_lags, n_obs, critical_values, _ = adfuller(
            values, autolag="AIC"
        )
    except Exception as exc:  # pragma: no cover - third-party numerical failures
        logger.warning("ADF test failed: %s", exc)
        return None
    if not np.isfinite(pvalue) or not np.isfinite(statistic):
        return None
    finite_critical_values = {
        key: float(value) for key, value in critical_values.items()
    }
    if not all(np.isfinite(value) for value in finite_critical_values.values()):
        return None
    reject_null = bool(pvalue <= level)
    return ADFResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        n_lags=int(n_lags),
        n_obs=int(n_obs),
        critical_values=finite_critical_values,
        significance=level,
        reject_null=reject_null,
        interpretation=_adf_interpretation(pvalue, level, reject_null),
    )


def _adf_interpretation(pvalue: float, level: float, reject_null: bool) -> str:
    verdict = (
        "reject the unit-root null -- evidence of stationarity"
        if reject_null
        else "cannot reject the unit-root null -- no evidence of stationarity"
    )
    return f"ADF p-value {pvalue:.4f} at the {level:g} level: {verdict}."


@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger cointegration test outcome for two price series.

    H0: the two series are not cointegrated (no stable long-run linear
    relationship). H1: they are cointegrated. Distinct from correlation
    (a short-run co-movement measure) and from running ADF on a spread
    built from an already-fitted hedge ratio -- this test fits and checks
    the relationship in one step.
    """

    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    significance: float
    reject_null: bool
    interpretation: str


def cointegration_test(
    a: pd.Series, b: pd.Series, *, significance: float = 0.05
) -> CointegrationResult | None:
    """Engle-Granger cointegration test between two price series.

    Wraps ``statsmodels.tsa.stattools.coint`` (Engle & Granger 1987's
    two-step method; regresses ``a`` on ``b`` and tests the residual for a
    unit root -- asymmetric in principle, though the two directions rarely
    disagree in practice). Assumes both series are individually I(1)
    (integrated of order one); the test is not meaningful otherwise.
    Returns ``None`` (never a raised error) for fewer than 20 paired
    observations, a numerical failure inside statsmodels, a non-finite
    statistic/p-value, or when the two series are (near-)perfectly
    collinear -- statsmodels' own ``CollinearityWarning`` flags this last
    case as numerically unreliable (e.g. a spurious ``statistic=-inf``,
    ``pvalue=0.0`` "confident" result for ``b = 2 * a``), so it is treated
    as inconclusive here rather than surfaced as a confident verdict.
    """
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise TypeError("a and b must be pandas Series.")
    validated_a = numeric_pandas(a, name="a")
    validated_b = numeric_pandas(b, name="b")
    same_axes(validated_a, validated_b, names=("b",))
    level = _significance(significance)
    paired = pd.concat({"a": validated_a, "b": validated_b}, axis=1).dropna()
    if len(paired) < _MIN_TEST_OBSERVATIONS:
        return None
    try:
        from statsmodels.tools.sm_exceptions import CollinearityWarning
        from statsmodels.tsa.stattools import coint

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", CollinearityWarning)
            statistic, pvalue, critical_values = coint(
                paired["a"].to_numpy(dtype=float), paired["b"].to_numpy(dtype=float)
            )
        if any(issubclass(w.category, CollinearityWarning) for w in caught):
            logger.warning(
                "Cointegration test: near-perfectly collinear series -- "
                "result is not numerically reliable, treating as inconclusive."
            )
            return None
    except Exception as exc:  # pragma: no cover - third-party numerical failures
        logger.warning("Cointegration test failed: %s", exc)
        return None
    if not np.isfinite(pvalue) or not np.isfinite(statistic):
        return None
    finite_critical_values = dict(
        zip(
            ("1%", "5%", "10%"),
            (float(value) for value in critical_values),
            strict=True,
        )
    )
    if not all(np.isfinite(value) for value in finite_critical_values.values()):
        return None
    reject_null = bool(pvalue <= level)
    verdict = (
        "reject the no-cointegration null -- evidence of a stable long-run relationship"
        if reject_null
        else "cannot reject the no-cointegration null -- no evidence of a "
        "stable long-run relationship"
    )
    return CointegrationResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        critical_values=finite_critical_values,
        significance=level,
        reject_null=reject_null,
        interpretation=f"Engle-Granger p-value {pvalue:.4f} at the {level:g} level: "
        f"{verdict}.",
    )


def hurst_exponent(series: pd.Series, *, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent via the variance-of-differences method.

    Regresses ``log(std(x[t+lag] - x[t]))`` on ``log(lag)`` for
    ``lag in [2, max_lag]``; the slope is the estimate. ``H < 0.5``
    suggests mean reversion, ``H ~= 0.5`` a random walk, ``H > 0.5`` a
    trending/persistent series -- a descriptive estimate on the sample
    given, not a hypothesis test with a p-value. Returns ``nan`` when
    there are too few observations (``< 2 * max_lag``) or the series is
    degenerate (e.g. constant) after dropping missing values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    validated = numeric_pandas(series, name="series")
    length = positive_int(max_lag, name="max_lag", minimum=2)
    values = validated.dropna().to_numpy(dtype=float)
    if len(values) < 2 * length:
        return float("nan")
    lags = np.arange(2, length + 1)
    spreads = np.array(
        [np.std(values[lag:] - values[:-lag]) for lag in lags], dtype=float
    )
    if not np.all(spreads > 0):
        return float("nan")
    slope, _ = np.polyfit(np.log(lags.astype(float)), np.log(spreads), 1)
    return float(slope)


def _significance(value: object) -> float:
    level = finite_real(value, name="significance", minimum=0.0, strict=True)
    if level >= 1.0:
        raise ValueError("significance must be strictly between 0 and 1.")
    return level
