"""Performance and risk metrics for regularly spaced observations.

Statistical metrics ignore missing returns but reject infinite values. Empty or
insufficient samples use the package's neutral ``0.0`` convention; undefined
distribution moments for constant samples remain ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.constants import EPSILON, TRADING_DAYS_PER_YEAR
from quantlab.risk._validation import (
    boolean,
    equity_series,
    finite_real,
    numeric_series,
    positive_int,
)
from quantlab.risk.drawdown import average_drawdown, longest_drawdown, max_drawdown
from quantlab.risk.var import historical_cvar, historical_var


def _returns(value: object, *, name: str = "returns") -> pd.Series:
    """Validate return observations and drop missing values."""
    return numeric_series(value, name=name, allow_nan=True).dropna()


def _annualization(value: object) -> int:
    return positive_int(value, name="periods_per_year")


def _risk_free_rate(value: object) -> float:
    return finite_real(value, name="risk_free_rate")


def equity_from_returns(
    returns: pd.Series, initial: float = 1.0, *, preserve_index: bool = False
) -> pd.Series:
    """Compound all period returns after an explicit baseline value.

    Missing and infinite returns are rejected because treating either as a
    zero-return period would invent performance. Returns below ``-100%`` are
    invalid. When ``preserve_index`` is true, a preceding timestamp is inferred
    for a chronological ``DatetimeIndex``.
    """
    clean = numeric_series(
        returns,
        name="returns",
        allow_nan=False,
        require_unique_index=preserve_index,
        require_sorted_index=preserve_index,
    )
    baseline = finite_real(initial, name="initial")
    keep_index = boolean(preserve_index, name="preserve_index")
    if baseline <= 0.0:
        raise ValueError("initial must be greater than zero.")
    if (clean < -1.0).any():
        raise ValueError("returns must not be below -100%.")

    growth = baseline * (1.0 + clean).cumprod()
    values = np.concatenate([[baseline], growth.to_numpy(dtype=float)])
    index: pd.Index
    if keep_index and isinstance(clean.index, pd.DatetimeIndex) and not clean.empty:
        step = (
            clean.index[1] - clean.index[0]
            if len(clean.index) > 1
            else pd.Timedelta(days=1)
        )
        if step <= pd.Timedelta(0):
            raise ValueError("returns index must have a positive inferred frequency.")
        index = pd.DatetimeIndex([clean.index[0] - step]).append(clean.index)
    else:
        index = pd.RangeIndex(len(values))
    return pd.Series(values, index=index, dtype=float)


def total_return(equity: pd.Series) -> float:
    """Return ``final / initial - 1`` for a finite equity curve."""
    clean = equity_series(equity, require_nonnegative=False, prevent_resurrection=False)
    if len(clean) < 2:
        return 0.0
    return float(clean.iloc[-1] / clean.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate under regular observation spacing.

    A non-positive terminal value is treated as a total loss and returns
    ``-1.0``.
    """
    ppy = _annualization(periods_per_year)
    clean = equity_series(equity, require_nonnegative=False, prevent_resurrection=False)
    if len(clean) < 2:
        return 0.0
    ratio = clean.iloc[-1] / clean.iloc[0]
    if ratio <= 0.0:
        return -1.0
    years = (len(clean) - 1) / ppy
    return float(ratio ** (1.0 / years) - 1.0)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised sample standard deviation of period returns."""
    ppy = _annualization(periods_per_year)
    clean = _returns(returns)
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(ppy))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio with an annual risk-free rate."""
    ppy = _annualization(periods_per_year)
    rf = _risk_free_rate(risk_free_rate)
    clean = _returns(returns)
    if len(clean) < 2:
        return 0.0
    excess = clean - rf / ppy
    std = clean.std(ddof=1)
    if std < EPSILON:
        return 0.0
    return float(excess.mean() / std * np.sqrt(ppy))


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Rolling annualised Sharpe ratio over a trailing window."""
    clean = numeric_series(returns, name="returns", allow_nan=True)
    size = positive_int(window, name="window")
    ppy = _annualization(periods_per_year)
    rf = _risk_free_rate(risk_free_rate)
    roll_mean = (clean - rf / ppy).rolling(size).mean()
    roll_std = clean.rolling(size).std(ddof=1)
    flat = roll_std < EPSILON
    result = roll_mean / roll_std.where(~flat, 1.0) * np.sqrt(ppy)
    output: pd.Series = result.where(~flat, 0.0)
    return output


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio using full-sample downside deviation."""
    ppy = _annualization(periods_per_year)
    rf = _risk_free_rate(risk_free_rate)
    clean = _returns(returns)
    if len(clean) < 2:
        return 0.0
    excess = clean - rf / ppy
    shortfall = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt((shortfall**2).mean()))
    if downside_deviation < EPSILON:
        return 0.0
    return float(excess.mean() / downside_deviation * np.sqrt(ppy))


def calmar_ratio(
    equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Return ``CAGR / abs(max_drawdown)``."""
    ppy = _annualization(periods_per_year)
    mdd = abs(max_drawdown(equity))
    if mdd < EPSILON:
        return 0.0
    return float(cagr(equity, ppy) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive observations among non-zero period returns."""
    clean = _returns(returns)
    nonzero = clean[clean != 0.0]
    if nonzero.empty:
        return 0.0
    return float((nonzero > 0.0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Positive period returns divided by absolute negative period returns."""
    clean = _returns(returns)
    gains = float(clean[clean > 0.0].sum())
    losses = float(-clean[clean < 0.0].sum())
    if losses < EPSILON:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def skewness(returns: pd.Series) -> float:
    """Bias-corrected skewness, or ``NaN`` for a constant sample."""
    clean = _returns(returns)
    if len(clean) < 3:
        return 0.0
    if clean.std(ddof=0) < EPSILON:
        return float("nan")
    return float(stats.skew(clean.to_numpy(dtype=float), bias=False))


def kurtosis(returns: pd.Series) -> float:
    """Bias-corrected excess kurtosis, or ``NaN`` for a constant sample."""
    clean = _returns(returns)
    if len(clean) < 4:
        return 0.0
    if clean.std(ddof=0) < EPSILON:
        return float("nan")
    return float(stats.kurtosis(clean.to_numpy(dtype=float), bias=False))


def _aligned_returns(returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    strategy = numeric_series(
        returns, name="returns", allow_nan=True, require_unique_index=True
    )
    benchmark = numeric_series(
        benchmark_returns,
        name="benchmark_returns",
        allow_nan=True,
        require_unique_index=True,
    )
    return pd.concat({"s": strategy, "b": benchmark}, axis=1).dropna()


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Return strategy beta relative to the benchmark."""
    aligned = _aligned_returns(returns, benchmark_returns)
    if len(aligned) < 2:
        return 0.0
    variance = aligned["b"].var(ddof=1)
    if variance < EPSILON:
        return 0.0
    return float(aligned["s"].cov(aligned["b"]) / variance)


def annualized_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Approximate annualised CAPM alpha from period means."""
    ppy = _annualization(periods_per_year)
    rf = _risk_free_rate(risk_free_rate)
    aligned = _aligned_returns(returns, benchmark_returns)
    if len(aligned) < 2:
        return 0.0
    coefficient = beta(aligned["s"], aligned["b"])
    rf_period = rf / ppy
    alpha_period = (aligned["s"].mean() - rf_period) - coefficient * (
        aligned["b"].mean() - rf_period
    )
    return float(alpha_period * ppy)


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised sample standard deviation of active returns."""
    ppy = _annualization(periods_per_year)
    aligned = _aligned_returns(returns, benchmark_returns)
    if len(aligned) < 2:
        return 0.0
    active = aligned["s"] - aligned["b"]
    return float(active.std(ddof=1) * np.sqrt(ppy))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised mean active return divided by tracking error."""
    ppy = _annualization(periods_per_year)
    aligned = _aligned_returns(returns, benchmark_returns)
    if len(aligned) < 2:
        return 0.0
    active = aligned["s"] - aligned["b"]
    std = active.std(ddof=1)
    if std < EPSILON:
        return 0.0
    return float(active.mean() / std * np.sqrt(ppy))


def _validate_metric_inputs(
    returns: pd.Series, equity: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Validate aggregate inputs and their chronology."""
    clean_returns = numeric_series(
        returns,
        name="returns",
        allow_nan=False,
        require_unique_index=True,
        require_sorted_index=True,
    )
    clean_equity = equity_series(equity)
    if len(clean_equity) == len(clean_returns):
        if not clean_equity.index.equals(clean_returns.index):
            raise ValueError("returns and equity must use the same index.")
        comparison_returns = clean_returns.iloc[1:].to_numpy(dtype=float)
        previous_equity = clean_equity.iloc[:-1].to_numpy(dtype=float)
        current_equity = clean_equity.iloc[1:].to_numpy(dtype=float)
    elif len(clean_equity) == len(clean_returns) + 1:
        if not clean_equity.index[1:].equals(clean_returns.index):
            raise ValueError("equity after its baseline must align with returns.")
        comparison_returns = clean_returns.to_numpy(dtype=float)
        previous_equity = clean_equity.iloc[:-1].to_numpy(dtype=float)
        current_equity = clean_equity.iloc[1:].to_numpy(dtype=float)
    else:
        raise ValueError(
            "equity must have the same length as returns or one baseline point more."
        )
    expected_equity = previous_equity * (1.0 + comparison_returns)
    if not np.allclose(current_equity, expected_equity, rtol=1e-9, atol=1e-10):
        raise ValueError("equity changes are inconsistent with returns.")
    after_ruin = (previous_equity == 0.0) & (comparison_returns != 0.0)
    if after_ruin.any():
        raise ValueError("returns must remain zero after equity reaches zero.")
    return clean_returns, clean_equity


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute the standard metric suite as a flat mapping."""
    ppy = _annualization(periods_per_year)
    rf = _risk_free_rate(risk_free_rate)
    clean_returns, clean_equity = _validate_metric_inputs(returns, equity)
    metrics: dict[str, float] = {
        "total_return": total_return(clean_equity),
        "cagr": cagr(clean_equity, ppy),
        "annualized_volatility": annualized_volatility(clean_returns, ppy),
        "sharpe_ratio": sharpe_ratio(clean_returns, rf, ppy),
        "sortino_ratio": sortino_ratio(clean_returns, rf, ppy),
        "calmar_ratio": calmar_ratio(clean_equity, ppy),
        "max_drawdown": max_drawdown(clean_equity),
        "average_drawdown": average_drawdown(clean_equity),
        "longest_drawdown": float(longest_drawdown(clean_equity)),
        "var_95": historical_var(clean_returns, 0.95),
        "var_99": historical_var(clean_returns, 0.99),
        "cvar_95": historical_cvar(clean_returns, 0.95),
        "cvar_99": historical_cvar(clean_returns, 0.99),
        "hit_rate": hit_rate(clean_returns),
        "best_period": float(clean_returns.max()) if len(clean_returns) else 0.0,
        "worst_period": float(clean_returns.min()) if len(clean_returns) else 0.0,
        "skewness": skewness(clean_returns),
        "kurtosis": kurtosis(clean_returns),
        "profit_factor": profit_factor(clean_returns),
    }
    if benchmark_returns is not None:
        metrics.update(
            {
                "beta": beta(clean_returns, benchmark_returns),
                "alpha": annualized_alpha(clean_returns, benchmark_returns, rf, ppy),
                "information_ratio": information_ratio(
                    clean_returns, benchmark_returns, ppy
                ),
                "tracking_error": tracking_error(clean_returns, benchmark_returns, ppy),
            }
        )
    return metrics
