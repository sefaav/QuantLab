"""Strategy interface, signal contract and registry."""

from __future__ import annotations

import copy
import inspect
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from quantlab.data.base import price_matrix
from quantlab.exceptions import StrategyError
from quantlab.features._validation import numeric_pandas

_REGISTRY: dict[str, type[BaseStrategy]] = {}
_StrategyT = TypeVar("_StrategyT", bound="BaseStrategy")


def _registry_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise StrategyError("Strategy name must be a non-empty string.")
    return name.strip()


def register_strategy(
    name: str, *, replace: bool = False
) -> Callable[[type[_StrategyT]], type[_StrategyT]]:
    """Register a :class:`BaseStrategy` subclass under a unique name."""
    registry_name = _registry_name(name)
    if not isinstance(replace, bool):
        raise StrategyError("replace must be a boolean.")

    def _wrap(cls: type[_StrategyT]) -> type[_StrategyT]:
        if not isinstance(cls, type) or not issubclass(cls, BaseStrategy):
            raise StrategyError("Registered strategies must inherit BaseStrategy.")
        if registry_name in _REGISTRY and not replace:
            raise StrategyError(
                f"Strategy '{registry_name}' is already registered to "
                f"{_REGISTRY[registry_name].__qualname__}."
            )
        _REGISTRY[registry_name] = cls
        cls.name = registry_name
        return cls

    return _wrap


def build_strategy(
    name: str, parameters: Mapping[str, Any] | None = None
) -> BaseStrategy:
    """Instantiate a registered strategy from keyword parameters."""
    registry_name = _registry_name(name)
    if registry_name not in _REGISTRY:
        raise StrategyError(
            f"Unknown strategy '{registry_name}'. Registered: {sorted(_REGISTRY)}."
        )
    if parameters is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(parameters, Mapping):
        if any(not isinstance(key, str) for key in parameters):
            raise StrategyError("Strategy parameter names must be strings.")
        kwargs = dict(parameters)
    else:
        raise StrategyError("parameters must be a mapping or None.")
    try:
        return _REGISTRY[registry_name](**kwargs)
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            f"Invalid parameters for strategy '{registry_name}': {exc}"
        ) from exc


def available_strategies() -> list[str]:
    """Return registered strategy names in sorted order."""
    return sorted(_REGISTRY)


def strategy_parameter_names(name: str) -> set[str]:
    """Return explicit constructor keywords accepted by a strategy."""
    registry_name = _registry_name(name)
    if registry_name not in _REGISTRY:
        raise StrategyError(
            f"Unknown strategy '{registry_name}'. Registered: {sorted(_REGISTRY)}."
        )
    signature = inspect.signature(_REGISTRY[registry_name].__init__)
    return {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def _unwrap_simple_type(annotation: Any) -> type | None:
    """Resolve plain and optional annotations used for early type checks."""
    if annotation is Any:
        return None
    if isinstance(annotation, type):
        return annotation
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [
            item for item in typing.get_args(annotation) if item is not type(None)
        ]
        if len(members) == 1 and isinstance(members[0], type):
            return members[0]
    return None


def _matches_annotation(value: object, expected: type) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return expected is bool
    if expected is int:
        return isinstance(value, Integral)
    if expected is float:
        return isinstance(value, Real)
    return isinstance(value, expected)


def _bound_parameters(
    cls: type[BaseStrategy], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    try:
        bound = signature.bind(None, **dict(parameters))
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameters for strategy '{cls.name}': {exc}"
        ) from exc
    bound.apply_defaults()
    return {
        name: value
        for name, value in bound.arguments.items()
        if name != "self"
        and signature.parameters[name].kind
        not in (
            signature.parameters[name].VAR_POSITIONAL,
            signature.parameters[name].VAR_KEYWORD,
        )
    }


def validate_strategy_parameters(name: str, parameters: Mapping[str, Any]) -> None:
    """Validate names, simple annotations and built-in parameter relations."""
    registry_name = _registry_name(name)
    if registry_name not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy '{registry_name}'. Registered: {sorted(_REGISTRY)}."
        )
    if not isinstance(parameters, Mapping):
        raise ValueError("Strategy parameters must be a mapping.")
    if any(not isinstance(key, str) for key in parameters):
        raise ValueError("Strategy parameter names must be strings.")

    accepted = strategy_parameter_names(registry_name)
    unknown = set(parameters) - accepted
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)} for strategy "
            f"'{registry_name}'. Accepted parameters: {sorted(accepted)}."
        )

    cls = _REGISTRY[registry_name]
    try:
        hints = typing.get_type_hints(cls.__init__)
    except (NameError, TypeError) as exc:
        raise ValueError(
            f"Cannot resolve constructor annotations for strategy "
            f"'{registry_name}': {exc}"
        ) from exc
    for key, value in parameters.items():
        expected = _unwrap_simple_type(hints.get(key))
        if (
            expected is not None
            and value is not None
            and not _matches_annotation(value, expected)
        ):
            raise ValueError(
                f"Parameter '{key}' for strategy '{registry_name}' must be "
                f"{expected.__name__}, got {type(value).__name__} ({value!r})."
            )

    complete = _bound_parameters(cls, parameters)
    cls.validate_parameters(complete)


class BaseStrategy(ABC):
    """Abstract base class for signal-generating strategies."""

    name: str = "base"

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a complete constructor mapping without creating an instance."""
        return dict(parameters)

    @abstractmethod
    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return a ``dates x symbols`` signal matrix in ``[-1, 1]``."""
        raise NotImplementedError

    @staticmethod
    def _prices(data: pd.DataFrame) -> pd.DataFrame:
        """Return a finite, positive adjusted-close matrix."""
        prices = price_matrix(data, adjusted=True)
        try:
            validated = numeric_pandas(
                prices, name="adjusted-close prices", strictly_positive=True
            )
        except (TypeError, ValueError) as exc:
            raise StrategyError(str(exc)) from exc
        if validated.empty or validated.shape[1] == 0:
            raise StrategyError(
                "Market data must contain at least one date and symbol."
            )
        return validated.astype(float)

    @staticmethod
    def _validate_signals(
        signals: pd.DataFrame, reference: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Validate signal values and optionally require exact reference axes."""
        if not isinstance(signals, pd.DataFrame):
            raise StrategyError("Signals must be a pandas DataFrame.")
        try:
            validated = numeric_pandas(signals, name="signals")
        except (TypeError, ValueError) as exc:
            raise StrategyError(str(exc)) from exc
        if reference is not None and (
            not validated.index.equals(reference.index)
            or not validated.columns.equals(reference.columns)
        ):
            raise StrategyError(
                "Signal index and columns must exactly match the price matrix."
            )
        values = validated.to_numpy(dtype=float, na_value=np.nan)
        present = ~np.isnan(values)
        if ((values[present] < -1.0) | (values[present] > 1.0)).any():
            raise StrategyError("Finite signals must remain within [-1, 1].")
        return validated.fillna(0.0).astype(float)

    def _freeze_parameters(self) -> None:
        object.__setattr__(self, "_strategy_parameters_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent public parameter mutation after construction."""
        if getattr(self, "_strategy_parameters_frozen", False) and not name.startswith(
            "_"
        ):
            raise AttributeError(
                "Strategy parameters are immutable after construction."
            )
        object.__setattr__(self, name, value)

    def parameters(self) -> dict[str, Any]:
        """Return a defensive copy of public strategy parameters."""
        return copy.deepcopy(
            {key: value for key, value in vars(self).items() if not key.startswith("_")}
        )

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        params = ", ".join(
            f"{key}={value!r}" for key, value in self.parameters().items()
        )
        return f"{type(self).__name__}({params})"
