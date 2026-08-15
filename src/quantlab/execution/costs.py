"""Commission and spread cost models.

Both costs are linear in traded notional. Because they are linear, feeding the
model *weight changes* (notional per unit of equity) yields the cost as a
fraction of equity — which is what the vectorised engine needs — while feeding
absolute traded notional yields absolute money. The formulas are identical
either way.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd

from quantlab.constants import BPS_TO_FRACTION
from quantlab.execution.orders import validate_execution_frame


def _validate_rate(value: object, *, name: str) -> float:
    """Return a finite non-negative rate, rejecting boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"{name} must be a non-negative, finite number, got {value!r}."
        )
    rate = float(value)
    if rate < 0 or not math.isfinite(rate):
        raise ValueError(
            f"{name} must be a non-negative, finite number, got {value!r}."
        )
    return rate


class CommissionModel:
    """Proportional commission.

    ``commission = traded_notional × commission_bps / 10_000``.

    Args:
        commission_bps: Commission in basis points of traded notional.
    """

    def __init__(self, commission_bps: float = 0.0) -> None:
        self.commission_bps = _validate_rate(commission_bps, name="commission_bps")

    def calculate(self, traded_notional: pd.DataFrame) -> pd.Series:
        """Per-date commission summed across symbols."""
        traded = validate_execution_frame(traded_notional, name="traded_notional")
        rate = (
            _validate_rate(self.commission_bps, name="commission_bps") * BPS_TO_FRACTION
        )
        return traded.abs().sum(axis=1) * rate


class SpreadModel:
    """Half-spread cost applied on position changes.

    ``spread_cost = traded_notional × spread_bps / 20_000``. The extra factor of
    2 (i.e. divide by 20 000, not 10 000) reflects paying the *half*-spread when
    crossing.

    Args:
        spread_bps: Full quoted spread in basis points.
    """

    def __init__(self, spread_bps: float = 0.0) -> None:
        self.spread_bps = _validate_rate(spread_bps, name="spread_bps")

    def calculate(self, traded_notional: pd.DataFrame) -> pd.Series:
        """Per-date spread cost summed across symbols."""
        traded = validate_execution_frame(traded_notional, name="traded_notional")
        rate = (
            _validate_rate(self.spread_bps, name="spread_bps") * BPS_TO_FRACTION / 2.0
        )
        return traded.abs().sum(axis=1) * rate
