"""Drawdown depth and duration analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.risk._validation import equity_series


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return ``equity / running_peak - 1`` at each observation."""
    clean = equity_series(equity)
    return clean / clean.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Return the most negative drawdown, or zero for an empty curve."""
    drawdown = drawdown_series(equity)
    return float(drawdown.min()) if len(drawdown) else 0.0


def average_drawdown(equity: pd.Series) -> float:
    """Return the time average of the non-positive drawdown series."""
    drawdown = drawdown_series(equity)
    return float(drawdown.mean()) if len(drawdown) else 0.0


def drawdown_durations(equity: pd.Series) -> list[int]:
    """Return the observation count of each uninterrupted drawdown."""
    underwater = drawdown_series(equity) < 0.0
    durations: list[int] = []
    current = 0
    for flag in underwater.to_numpy(dtype=bool):
        if flag:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)
    return durations


def longest_drawdown(equity: pd.Series) -> int:
    """Return the longest drawdown in number of observations."""
    durations = drawdown_durations(equity)
    return max(durations) if durations else 0


def max_drawdown_details(equity: pd.Series) -> dict[str, object]:
    """Return the worst drawdown's depth and peak/trough labels."""
    clean = equity_series(equity)
    if clean.empty:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "depth": 0.0,
        }
    drawdown = clean / clean.cummax() - 1.0
    trough_position = int(np.argmin(drawdown.to_numpy(dtype=float)))
    peak_position = int(
        np.argmax(clean.iloc[: trough_position + 1].to_numpy(dtype=float))
    )
    depth = float(drawdown.iloc[trough_position])
    return {
        "max_drawdown": depth,
        "peak_date": clean.index[peak_position],
        "trough_date": clean.index[trough_position],
        "depth": depth,
    }
