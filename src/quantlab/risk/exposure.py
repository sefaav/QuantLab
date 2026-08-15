"""Gross and net portfolio exposure analytics."""

from __future__ import annotations

import pandas as pd

from quantlab.risk._validation import numeric_frame


def gross_exposure_series(weights: pd.DataFrame) -> pd.Series:
    """Return per-observation gross exposure ``sum(abs(weight))``."""
    clean = numeric_frame(weights, name="weights")
    return clean.abs().sum(axis=1)


def net_exposure_series(weights: pd.DataFrame) -> pd.Series:
    """Return per-observation net exposure ``sum(weight)``."""
    clean = numeric_frame(weights, name="weights")
    return clean.sum(axis=1)


def average_gross_exposure(weights: pd.DataFrame) -> float:
    """Return time-average gross exposure."""
    exposure = gross_exposure_series(weights)
    return float(exposure.mean()) if len(exposure) else 0.0


def average_net_exposure(weights: pd.DataFrame) -> float:
    """Return time-average net exposure."""
    exposure = net_exposure_series(weights)
    return float(exposure.mean()) if len(exposure) else 0.0
