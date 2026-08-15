"""Historical Value-at-Risk and Conditional Value-at-Risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.risk._validation import finite_real, numeric_series


def _inputs(returns: pd.Series, confidence: float) -> tuple[pd.Series, float]:
    clean = numeric_series(returns, name="returns", allow_nan=True).dropna()
    level = finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    return clean, level


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Return historical VaR as a non-negative period-loss magnitude."""
    clean, level = _inputs(returns, confidence)
    if clean.empty:
        return 0.0
    quantile = float(np.quantile(clean.to_numpy(dtype=float), 1.0 - level))
    return max(0.0, -quantile)


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Return mean historical loss at or beyond the VaR threshold."""
    clean, level = _inputs(returns, confidence)
    if clean.empty:
        return 0.0
    threshold = float(np.quantile(clean.to_numpy(dtype=float), 1.0 - level))
    tail = clean[clean <= threshold]
    if tail.empty:
        return max(0.0, -threshold)
    return max(0.0, -float(tail.mean()))
