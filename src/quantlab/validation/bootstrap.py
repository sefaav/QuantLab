"""Bootstrap estimates of historical return-sampling uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.constants import TRADING_DAYS_PER_YEAR
from quantlab.risk import metrics as M
from quantlab.risk._validation import (
    finite_real,
    nonnegative_int,
    numeric_series,
    positive_int,
)
from quantlab.risk.drawdown import max_drawdown

_SAMPLE_COLUMNS = ["cagr", "sharpe", "max_drawdown", "final_value"]


@dataclass(frozen=True)
class BootstrapResult:
    """Distribution of bootstrapped statistics with a percentile summary."""

    samples: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate the stable sample schema and detach caller-owned data."""
        if not isinstance(self.samples, pd.DataFrame):
            raise TypeError("samples must be a pandas DataFrame.")
        missing = set(_SAMPLE_COLUMNS).difference(self.samples.columns)
        if missing:
            raise ValueError(f"samples is missing columns: {sorted(missing)}.")
        object.__setattr__(self, "samples", self.samples.loc[:, _SAMPLE_COLUMNS].copy())

    def summary(self) -> pd.DataFrame:
        """Return the mean, dispersion and central percentile interval."""
        rows = []
        for column in self.samples.columns:
            series = self.samples[column].dropna()
            rows.append(
                {
                    "statistic": column,
                    "median": float(series.median()),
                    "p05": float(series.quantile(0.05)),
                    "p95": float(series.quantile(0.95)),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                }
            )
        return pd.DataFrame(rows)


def _resample_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` circular block-bootstrap indices."""
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    indices = np.empty(n, dtype=int)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        take = min(block_size, n - filled)
        indices[filled : filled + take] = (np.arange(start, start + take)) % n
        filled += take
    return indices


def bootstrap_returns(
    returns: pd.Series,
    *,
    n_iterations: int = 1000,
    block_size: int = 1,
    seed: int = 42,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.0,
) -> BootstrapResult:
    """Bootstrap CAGR, Sharpe ratio, drawdown and final portfolio value.

    Missing observations are dropped for the i.i.d. bootstrap. They are
    rejected for block bootstrapping because dropping them would join periods
    that were not adjacent in the original series. The output describes the
    sampled historical distribution, not future profitability.
    """
    n_iterations = positive_int(n_iterations, name="n_iterations")
    block_size = positive_int(block_size, name="block_size")
    seed = nonnegative_int(seed, name="seed")
    periods_per_year = positive_int(periods_per_year, name="periods_per_year")
    initial_capital = finite_real(initial_capital, name="initial_capital")
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be greater than zero.")
    risk_free_rate = finite_real(risk_free_rate, name="risk_free_rate")
    validated = numeric_series(
        returns,
        name="returns",
        allow_nan=True,
        require_unique_index=True,
        require_sorted_index=True,
    )
    if block_size > 1 and validated.isna().any():
        raise ValueError(
            "returns must not contain missing values for block bootstrapping."
        )
    clean = validated.dropna().to_numpy(dtype=float)
    if (clean < -1.0).any():
        raise ValueError("returns must not contain values below -1.0.")

    records: list[dict[str, float]] = []
    if len(clean) < 2:
        return BootstrapResult(pd.DataFrame(records, columns=_SAMPLE_COLUMNS))

    rng = np.random.default_rng(seed)
    for _ in range(n_iterations):
        indices = _resample_indices(len(clean), block_size, rng)
        sample = pd.Series(clean[indices])
        equity = M.equity_from_returns(sample, initial=initial_capital)
        records.append(
            {
                "cagr": M.cagr(equity, periods_per_year),
                "sharpe": M.sharpe_ratio(sample, risk_free_rate, periods_per_year),
                "max_drawdown": max_drawdown(equity),
                "final_value": float(equity.iloc[-1]),
            }
        )
    return BootstrapResult(pd.DataFrame(records, columns=_SAMPLE_COLUMNS))
