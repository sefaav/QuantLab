"""Constant and volume-based slippage models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from quantlab.constants import BPS_TO_FRACTION, EPSILON
from quantlab.exceptions import BacktestError
from quantlab.execution.costs import _validate_rate
from quantlab.execution.orders import (
    validate_equity_series,
    validate_execution_frame,
)


def _validate_average_daily_volume(
    value: pd.DataFrame | float | None,
) -> pd.DataFrame | float:
    """Return a validated defensive copy of dollar ADV."""
    if value is None:
        raise ValueError("average_daily_volume is required for volume-based slippage.")
    if not isinstance(value, pd.DataFrame):
        return _validate_rate(value, name="average_daily_volume")
    if not value.index.is_unique:
        raise ValueError("average_daily_volume index must not contain duplicates.")
    if not value.columns.is_unique:
        raise ValueError("average_daily_volume columns must not contain duplicates.")
    try:
        values = value.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "average_daily_volume must contain numeric values or NaN."
        ) from exc
    finite_or_missing = np.isfinite(values) | np.isnan(values)
    if not finite_or_missing.all() or (values[np.isfinite(values)] < 0.0).any():
        raise ValueError(
            "average_daily_volume must be non-negative and finite wherever "
            "it is present."
        )
    return pd.DataFrame(values, index=value.index, columns=value.columns)


def validate_slippage_cost_frame(
    costs: pd.DataFrame,
    traded_notional: pd.DataFrame,
) -> pd.DataFrame:
    """Validate per-symbol slippage costs against the transaction matrix."""
    if not isinstance(costs, pd.DataFrame):
        raise BacktestError("slippage must return a pandas DataFrame.")
    if not costs.index.equals(traded_notional.index) or not costs.columns.equals(
        traded_notional.columns
    ):
        raise BacktestError(
            "slippage costs must have exactly the traded-notional axes."
        )
    try:
        values = costs.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("slippage costs must be numeric.") from exc
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise BacktestError(
            "slippage costs must contain only finite non-negative values."
        )
    return pd.DataFrame(values, index=costs.index, columns=costs.columns)


class SlippageModel(ABC):
    """Interface for per-symbol slippage estimates."""

    @abstractmethod
    def per_symbol_cost(
        self, traded_notional: pd.DataFrame, equity: pd.Series | None = None
    ) -> pd.DataFrame:
        """Return slippage cost per date and symbol, preserving input axes."""
        raise NotImplementedError

    def calculate(
        self, traded_notional: pd.DataFrame, equity: pd.Series | None = None
    ) -> pd.Series:
        """Return validated per-date slippage summed across symbols."""
        traded = validate_execution_frame(traded_notional, name="traded_notional")
        validated_equity = (
            validate_equity_series(equity, traded.index) if equity is not None else None
        )
        costs = self.per_symbol_cost(traded, validated_equity)
        return validate_slippage_cost_frame(costs, traded).sum(axis=1)


class ConstantSlippageModel(SlippageModel):
    """Flat slippage: ``traded_notional × slippage_bps / 10_000``."""

    def __init__(self, slippage_bps: float = 0.0) -> None:
        self.slippage_bps = _validate_rate(slippage_bps, name="slippage_bps")

    def per_symbol_cost(
        self, traded_notional: pd.DataFrame, equity: pd.Series | None = None
    ) -> pd.DataFrame:
        """Return flat per-symbol slippage; ``equity`` is not required."""
        traded = validate_execution_frame(traded_notional, name="traded_notional")
        rate = _validate_rate(self.slippage_bps, name="slippage_bps") * BPS_TO_FRACTION
        costs = traded.abs() * rate
        return validate_slippage_cost_frame(costs, traded)


class VolumeBasedSlippageModel(SlippageModel):
    """Square-root market impact using order size relative to dollar ADV.

    ``effective_bps = base_slippage_bps
        + impact_coefficient × sqrt(order_dollars / average_daily_volume)``

    Missing or zero ADV is allowed only for cells without a material trade.
    """

    def __init__(
        self,
        base_slippage_bps: float = 0.0,
        impact_coefficient: float = 0.1,
        average_daily_volume: pd.DataFrame | float | None = None,
    ) -> None:
        self.base_slippage_bps = _validate_rate(
            base_slippage_bps, name="base_slippage_bps"
        )
        self.impact_coefficient = _validate_rate(
            impact_coefficient, name="impact_coefficient"
        )
        self._average_daily_volume = _validate_average_daily_volume(
            average_daily_volume
        )

    @property
    def average_daily_volume(self) -> pd.DataFrame | float:
        """Dollar ADV, returned as a copy when it is a matrix."""
        if isinstance(self._average_daily_volume, pd.DataFrame):
            return self._average_daily_volume.copy(deep=True)
        return self._average_daily_volume

    def per_symbol_cost(
        self, traded_notional: pd.DataFrame, equity: pd.Series | None = None
    ) -> pd.DataFrame:
        """Return per-symbol square-root-impact slippage costs.

        ``traded_notional`` is interpreted as a fraction of equity when
        ``equity`` is supplied; otherwise it must use the same unit as ADV.
        """
        traded = validate_execution_frame(traded_notional, name="traded_notional")
        order = traded.abs()
        if equity is None:
            order_dollars = order
        else:
            validated_equity = validate_equity_series(equity, order.index)
            order_dollars = order.mul(validated_equity, axis=0)

        adv = _validate_average_daily_volume(self._average_daily_volume)
        materially_traded = order_dollars > EPSILON
        if isinstance(adv, pd.DataFrame):
            adv_aligned = adv.reindex_like(order)
            unavailable = adv_aligned.isna() | (adv_aligned <= 0.0)
            unavailable_trade = unavailable & materially_traded
            if unavailable_trade.to_numpy().any():
                row, column = np.argwhere(unavailable_trade.to_numpy())[0]
                date = unavailable_trade.index[int(row)]
                symbol = unavailable_trade.columns[int(column)]
                raise ValueError(
                    "average_daily_volume is missing or zero for "
                    f"{symbol!r} on {date!r}; positive ADV is required for "
                    "every material trade."
                )
            safe_adv = adv_aligned.mask(unavailable, 1.0)
            ratio = (order_dollars / safe_adv).mask(unavailable, 0.0)
        else:
            if adv <= 0.0:
                if materially_traded.to_numpy().any():
                    raise ValueError(
                        "Volume-based slippage requires positive ADV for every "
                        "material trade; average_daily_volume is zero."
                    )
                ratio = order_dollars * 0.0
            else:
                ratio = order_dollars / adv

        base_bps = _validate_rate(self.base_slippage_bps, name="base_slippage_bps")
        impact = _validate_rate(self.impact_coefficient, name="impact_coefficient")
        effective_bps = base_bps + impact * np.sqrt(ratio)
        costs = order * effective_bps * BPS_TO_FRACTION
        return validate_slippage_cost_frame(costs, traded)


def build_slippage_model(
    model: str,
    slippage_bps: float,
    *,
    impact_coefficient: float = 0.1,
    average_daily_volume: pd.DataFrame | float | None = None,
) -> SlippageModel:
    """Build a named slippage model."""
    if not isinstance(model, str):
        raise TypeError("model must be a string.")
    key = model.lower().strip()
    if key in {"constant", "fixed"}:
        return ConstantSlippageModel(slippage_bps)
    if key in {"volume", "volume_based", "sqrt"}:
        return VolumeBasedSlippageModel(
            base_slippage_bps=slippage_bps,
            impact_coefficient=impact_coefficient,
            average_daily_volume=average_daily_volume,
        )
    raise ValueError(f"Unknown slippage model '{model}'.")
