"""Allocators that convert strategy signals into target portfolio weights.

The built-in risk estimates are trailing. End-to-end causality still depends
on the supplied signals and on the execution timing enforced by the engine.
Portfolio constraints are applied separately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from quantlab.constants import TRADING_DAYS_PER_YEAR
from quantlab.data.base import price_matrix
from quantlab.exceptions import InvalidConfigurationError, QuantLabError
from quantlab.features.returns import simple_returns
from quantlab.features.volatility import realized_volatility
from quantlab.portfolio._validation import finite_real, positive_int, validate_frame
from quantlab.portfolio.position_sizing import (
    inverse_volatility_weights,
    normalize_gross,
    renormalize_within_cap,
)
from quantlab.portfolio.volatility_targeting import apply_volatility_target


class PortfolioAllocator(ABC):
    """Abstract signal-to-weight allocator."""

    name: str = "base"

    @abstractmethod
    def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Return a ``dates x symbols`` target-weight matrix."""
        raise NotImplementedError

    @staticmethod
    def _returns(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Return trailing asset returns aligned exactly to ``signals``."""
        validated_signals = _validate_allocator_inputs(signals, data)
        prices = price_matrix(data, adjusted=True)
        missing_symbols = validated_signals.columns.difference(prices.columns)
        missing_dates = validated_signals.index.difference(prices.index)
        if len(missing_symbols) or len(missing_dates):
            raise QuantLabError(
                "Signals contain symbols or dates that are absent from market data."
            )
        returns = simple_returns(prices)
        return returns.reindex(
            index=validated_signals.index, columns=validated_signals.columns
        )


AllocatorT = TypeVar("AllocatorT", bound=PortfolioAllocator)
_REGISTRY: dict[str, type[PortfolioAllocator]] = {}


def register_allocator(
    name: str,
) -> Callable[[type[AllocatorT]], type[AllocatorT]]:
    """Register an allocator class under a unique, non-empty name."""
    if not isinstance(name, str) or not name.strip():
        raise InvalidConfigurationError("Allocator name must be a non-empty string.")
    registry_name = name.strip()

    def _wrap(cls: type[AllocatorT]) -> type[AllocatorT]:
        if not issubclass(cls, PortfolioAllocator):
            raise InvalidConfigurationError(
                "Registered allocator must inherit PortfolioAllocator."
            )
        if registry_name in _REGISTRY:
            raise InvalidConfigurationError(
                f"Allocator '{registry_name}' is already registered."
            )
        _REGISTRY[registry_name] = cls
        cls.name = registry_name
        return cls

    return _wrap


def build_allocator(name: str, **kwargs: Any) -> PortfolioAllocator:
    """Instantiate a registered allocator by name."""
    if not isinstance(name, str) or not name.strip():
        raise InvalidConfigurationError("Allocator name must be a non-empty string.")
    registry_name = name.strip()
    if registry_name not in _REGISTRY:
        raise QuantLabError(
            f"Unknown allocator '{registry_name}'. Registered: {sorted(_REGISTRY)}."
        )
    try:
        return _REGISTRY[registry_name](**kwargs)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"Invalid parameters for allocator '{registry_name}': {exc}"
        ) from exc


def available_allocators() -> list[str]:
    """Return registered allocator names in sorted order."""
    return sorted(_REGISTRY)


@register_allocator("equal_weight")
class EqualWeightAllocator(PortfolioAllocator):
    """Give every active signal equal absolute weight and preserve its sign."""

    def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Return equal-magnitude weights with row-wise gross exposure one."""
        validated = _validate_allocator_inputs(signals, data)
        direction = pd.DataFrame(
            np.sign(validated), index=validated.index, columns=validated.columns
        )
        active_count = (validated.abs() > 0.0).sum(axis=1)
        return direction.div(active_count.where(active_count > 0), axis=0).fillna(0.0)


@register_allocator("signal_proportional")
class SignalProportionalAllocator(PortfolioAllocator):
    """Allocate in proportion to signed signal magnitude."""

    def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Return signal-proportional weights with gross exposure one."""
        validated = _validate_allocator_inputs(signals, data)
        return normalize_gross(validated, target_gross=1.0)


@register_allocator("inverse_volatility")
class InverseVolatilityAllocator(PortfolioAllocator):
    """Allocate inversely to trailing annualised volatility."""

    def __init__(
        self,
        volatility_window: int = 63,
        maximum_weight: float | None = None,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.volatility_window = positive_int(
            volatility_window, name="volatility_window", minimum=2
        )
        if maximum_weight is None:
            self.maximum_weight = None
        else:
            validated_cap = finite_real(
                maximum_weight, name="maximum_weight", minimum=0.0, strict=True
            )
            if validated_cap > 1.0:
                raise InvalidConfigurationError("maximum_weight must not exceed 1.")
            self.maximum_weight = validated_cap
        self.periods_per_year = positive_int(periods_per_year, name="periods_per_year")

    def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Return inverse-volatility weights with unavailable estimates flat."""
        validated = _validate_allocator_inputs(signals, data)
        returns = self._returns(data, validated)
        volatility = realized_volatility(
            returns,
            window=self.volatility_window,
            periods_per_year=self.periods_per_year,
        )
        raw = inverse_volatility_weights(validated, volatility)
        weights = normalize_gross(raw, target_gross=1.0)
        if self.maximum_weight is not None:
            weights = renormalize_within_cap(
                weights,
                target_gross=1.0,
                cap=self.maximum_weight,
            )
        return weights


@register_allocator("volatility_targeting")
class VolatilityTargetingAllocator(PortfolioAllocator):
    """Scale inverse-volatility weights toward an annual volatility target."""

    def __init__(
        self,
        target_volatility: float = 0.12,
        volatility_window: int = 63,
        maximum_leverage: float = 1.5,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.target_volatility = finite_real(
            target_volatility,
            name="target_volatility",
            minimum=0.0,
            strict=True,
        )
        self.volatility_window = positive_int(
            volatility_window, name="volatility_window", minimum=2
        )
        self.maximum_leverage = finite_real(
            maximum_leverage,
            name="maximum_leverage",
            minimum=0.0,
            strict=True,
        )
        self.periods_per_year = positive_int(periods_per_year, name="periods_per_year")
        self._base = InverseVolatilityAllocator(
            volatility_window=self.volatility_window,
            periods_per_year=self.periods_per_year,
        )

    def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Return inverse-volatility weights scaled by the risk target."""
        validated = _validate_allocator_inputs(signals, data)
        base = self._base.allocate(validated, data)
        returns = self._returns(data, validated)
        return apply_volatility_target(
            base,
            returns,
            self.target_volatility,
            window=self.volatility_window,
            maximum_leverage=self.maximum_leverage,
            periods_per_year=self.periods_per_year,
        )


def _validate_allocator_inputs(
    signals: pd.DataFrame, data: pd.DataFrame
) -> pd.DataFrame:
    validated = validate_frame(signals, name="signals")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    return validated
