"""Strategy interface, signal contract and registry."""

from __future__ import annotations

import copy
import inspect
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, TypeVar

import numpy as np
import pandas as pd

from quantlab.data.base import price_matrix
from quantlab.exceptions import StrategyError
from quantlab.features._validation import finite_real, numeric_pandas

#: Accepted values for every strategy's ``price_type`` constructor
#: parameter (see ``BaseStrategy._prices()``).
PRICE_TYPES = frozenset({"adjusted_close", "close"})

_REGISTRY: dict[str, type[BaseStrategy]] = {}
_StrategyT = TypeVar("_StrategyT", bound="BaseStrategy")


class UnsetType:
    """Sentinel distinguishing "not passed" from an explicit ``None``.

    Used for an optional, indicator/threshold-dependent constructor
    parameter (e.g. `MeanReversionStrategy`/`PairsTradingStrategy`'s
    ``stop_threshold``) whose sensible default depends on another
    parameter chosen in the SAME call (e.g. ``indicator``) and therefore
    cannot be a plain literal default. If ``None`` were the parameter's
    own default, "not passed" and "explicitly disabled" would collapse to
    the same value, making "disable this" inexpressible. `UNSET` is the
    constructor's actual default instead: `UNSET` resolves to whatever
    indicator-specific default applies, while an explicit ``None`` is
    respected as "disabled".
    """

    def __repr__(self) -> str:
        """Return a short, unambiguous debug representation."""
        return "<unset>"


UNSET = UnsetType()


def validate_risk_control_parameters(
    stop_loss_pct: object, take_profit_pct: object
) -> tuple[float | None, float | None]:
    """Validate a strategy's ``stop_loss_pct``/``take_profit_pct`` constructor pair.

    Shared by every strategy that accepts these two (rather than each
    duplicating the same two ``finite_real`` calls) -- both are optional,
    strictly positive fractions (e.g. ``0.10`` = 10%) with no relational
    constraint between them (unlike ``entry``/``exit``/``stop``, a
    stop-loss and a take-profit are independent conditions on opposite
    sides of zero return, not points on the same ordered scale).
    """
    validated_stop_loss = (
        None
        if stop_loss_pct is None
        else finite_real(stop_loss_pct, name="stop_loss_pct", minimum=0.0, strict=True)
    )
    validated_take_profit = (
        None
        if take_profit_pct is None
        else finite_real(
            take_profit_pct, name="take_profit_pct", minimum=0.0, strict=True
        )
    )
    return validated_stop_loss, validated_take_profit


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


def strategy_sweepable_parameter_names(name: str) -> set[str]:
    """Return constructor keywords suitable for a 2-parameter sensitivity sweep.

    Excludes boolean-defaulted parameters (structural switches such as
    long_only/long_short/dynamic_hedge_ratio): sweeping one changes which
    *other* parameters are even meaningful (e.g. bottom_fraction only
    matters when long_short=True), and candidate values typed as text
    (comma-separated) are easy to misparse as an int/str instead of a bool.
    ``default_parameter_grid`` already treats these as fixed, not swept, for
    walk-forward's own default grid — sensitivity applies the same rule.

    Also excludes ``cls.deprecated_parameter_names`` (a strategy's own
    deprecated backward-compatible aliases for a renamed parameter, if
    any are currently registered): offering both a deprecated alias and
    its canonical replacement as independent sweep axes would let a sweep
    set the alias to a value that conflicts with the canonical name
    already fixed elsewhere in the same config.
    """
    registry_name = _registry_name(name)
    if registry_name not in _REGISTRY:
        raise StrategyError(
            f"Unknown strategy '{registry_name}'. Registered: {sorted(_REGISTRY)}."
        )
    strategy_class = _REGISTRY[registry_name]
    signature = inspect.signature(strategy_class.__init__)
    return {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and not isinstance(parameter.default, bool)
        and parameter.name not in strategy_class.deprecated_parameter_names
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


@dataclass(frozen=True)
class SignalReasons:
    """Optional, strategy-specific explanation of ``generate_signals()``.

    Both frames must share ``generate_signals()``'s own ``dates x
    symbols`` shape and index/columns exactly. ``detail_code`` is a
    closed set of stable, machine-readable strings (or ``None`` where no
    transition happened that date); ``details`` is optional free text
    with the concrete values/thresholds involved, for human reading
    only. Mirrors ``trade_log.py``'s own ``trigger_reason_code``/
    ``trigger_reason_detail_code``/``trigger_reason_details`` split one
    level up: this is the strategy's own contribution to a row whose
    ``trigger_reason_code == "strategy_signal"``.
    """

    detail_code: pd.DataFrame
    details: pd.DataFrame


class BaseStrategy(ABC):
    """Abstract base class for signal-generating strategies."""

    name: str = "base"
    #: Price series ``_prices()`` reads for signal generation --
    #: "adjusted_close" (default) or "close". A class attribute fallback
    #: for any strategy that doesn't accept its own ``price_type``
    #: constructor parameter; every built-in strategy sets its own
    #: instance attribute of the same name, validated at construction.
    price_type: str = "adjusted_close"
    #: Fractional (e.g. 0.10 = 10%) gross-return thresholds that force-
    #: flatten this strategy's REAL executed position (see
    #: `quantlab.backtesting.accounting._detect_stop_loss_take_profit`) --
    #: class attribute fallbacks (mirroring `price_type` above) so calling
    #: code can read `strategy.stop_loss_pct`/`strategy.take_profit_pct`
    #: uniformly across every strategy, including ones that don't accept
    #: either as a constructor parameter. `None` (default) disables the
    #: check entirely, with strictly no change to accounting's numbers.
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    #: Per-symbol calendar names, engine-injected onto the strategy
    #: instance right before `generate_signals()`/`explain_signals()`/
    #: `decision_signal()` are called (see
    #: `quantlab.backtesting.engine.BacktestEngine.run`) so every rolling-
    #: window feature call site can compute on each symbol's own native
    #: calendar rather than a closure-padded combined timeline -- see
    #: `quantlab.features.native_calendar.compute_native_then_align`.
    #: Never a user-configured constructor hyperparameter: exempted from
    #: both the post-construction freeze (`__setattr__` below) and
    #: `parameters()` (must never appear in a config-YAML round-trip,
    #: execution-model hash, or sweep-parameter enumeration -- it is
    #: engine context, not a strategy parameter). `None` when the engine
    #: has not injected it (e.g. a strategy constructed directly in a
    #: unit test); every native-calendar call site must treat that the
    #: same as "no mixed calendars", falling back to a single vectorized
    #: computation.
    symbol_calendars: dict[str, str] | None = None
    #: Instance attributes exempted from both the freeze and `parameters()`
    #: -- engine-injected context, never a real strategy parameter.
    _NON_PARAMETER_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset(
        {"symbol_calendars"}
    )
    #: Constructor keyword(s) kept only as deprecated backward-compatible
    #: aliases for a renamed parameter. No built-in strategy currently
    #: registers any (defaults to empty) -- kept as generic infrastructure
    #: for the next time a strategy parameter is renamed. A registered
    #: alias would still be fully valid to pass, and still documented via
    #: `ParameterDoc` -- excluded only from
    #: `strategy_sweepable_parameter_names()`, so a sensitivity/robustness
    #: sweep never offers BOTH an alias and its canonical name as
    #: independent axes (which would let a sweep set the alias to a value
    #: that conflicts with the canonical name already fixed elsewhere in
    #: the same config).
    deprecated_parameter_names: ClassVar[frozenset[str]] = frozenset()

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

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons | None:
        """Optionally explain WHY ``generate_signals()`` changed its output.

        Must be a pure function of exactly the same ``data``/``features``
        given to ``generate_signals()`` -- no information unavailable at
        each row's own date (no look-ahead), and no dependency on state
        left over from a prior call. The default implementation returns
        ``None``, meaning no strategy-specific attribution is available;
        callers must treat that as "not analyzed", not as "unknown"/"no
        reason" -- the generic ``strategy_signal`` reason still applies.
        """
        return None

    def decision_signal(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Optional diagnostic representation of the strategy's decision state.

        Used ONLY for trade-reason attribution (trigger detection: "did
        the strategy make a new decision since the last rebalance") and
        strategic-origin tracking -- never a substitute for
        ``generate_signals()``'s own output in sizing, allocation,
        constraints, execution, PnL or cost calculations. Must be a pure
        function of exactly the same ``data``/``features`` given to
        ``generate_signals()`` (same no-look-ahead, no-leftover-state
        contract as :meth:`explain_signals`).

        The default (``None``) means ``generate_signals()``'s own output
        is already a faithful decision proxy -- true for every built-in
        strategy except :class:`~quantlab.strategies.pairs_trading.
        PairsTradingStrategy`, whose raw signal mixes a discrete decision
        state with purely mechanical rescaling (price/beta/normalization)
        that a plain "did the signal change" comparison cannot tell
        apart from a real new decision. Overriding this is an exception,
        not the norm: a strategy whose signal is itself the decision
        (the common case, including every strategy with a continuous or
        volatility-adjusted signal, where a magnitude change legitimately
        IS a new sizing decision) must leave this at its default.

        When overridden, the returned frame must share ``generate_
        signals()``'s own ``dates x symbols`` shape, index and columns
        exactly, contain only finite numeric values (no NaN/Infinity --
        raises :class:`~quantlab.exceptions.StrategyError` otherwise,
        never silently coerced or dropped), and is never reindexed by a
        caller to "align" a mismatched shape -- a caller that receives a
        mismatched frame must raise, not guess an alignment that could
        introduce a temporal offset.
        """
        return None

    def position_groups(self) -> tuple[tuple[str, ...], ...] | None:
        """Optional: symbol columns whose COMBINED P&L is one logical position.

        Used ONLY by a stop-loss/take-profit check (see
        :func:`quantlab.backtesting.accounting._detect_stop_loss_take_profit`)
        to decide whether a group of columns should be force-flattened
        together based on their combined realized return, rather than
        each column's own return independently. The default (``None``)
        means every symbol is its own independent group -- correct for
        every built-in strategy except :class:`~quantlab.strategies.
        pairs_trading.PairsTradingStrategy`, whose two legs (``symbol_a``/
        ``symbol_b``) form one economic position that a per-leg check
        would evaluate incorrectly (e.g. treating a leg that moves
        against the pair's own net P&L as a standalone loss). A symbol
        never mentioned in any returned group is still its own
        independent group -- this need not enumerate every column.
        """
        return None

    def _prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a finite, positive price matrix at ``self.price_type``."""
        if self.price_type not in PRICE_TYPES:
            raise StrategyError(
                f"Unknown price_type {self.price_type!r}; expected one of "
                f"{sorted(PRICE_TYPES)}."
            )
        adjusted = self.price_type == "adjusted_close"
        prices = price_matrix(data, adjusted=adjusted)
        try:
            validated = numeric_pandas(
                prices,
                name=f"{self.price_type.replace('_', '-')} prices",
                strictly_positive=True,
            )
        except (TypeError, ValueError) as exc:
            raise StrategyError(str(exc)) from exc
        if validated.empty or validated.shape[1] == 0:
            raise StrategyError(
                "Market data must contain at least one date and symbol."
            )
        return validated.astype(float)

    def _native_feature(
        self,
        prices: pd.DataFrame,
        compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute a rolling-window feature on each symbol's own calendar.

        Uses ``self.symbol_calendars``, aligned back onto
        ``prices.index`` -- see
        :func:`quantlab.features.native_calendar.compute_native_then_align`.
        ``self.symbol_calendars is None`` (not engine-injected, e.g. a
        strategy constructed directly in a unit test) short-circuits to
        calling ``compute_fn(prices)`` directly.
        """
        from quantlab.features.native_calendar import compute_native_then_align

        return compute_native_then_align(
            compute_fn, prices, self.symbol_calendars, pd.DatetimeIndex(prices.index)
        )

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

    @staticmethod
    def _validate_decision_signal(
        decision: pd.DataFrame, reference: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate a :meth:`decision_signal` result against its reference.

        Stricter than :meth:`_validate_signals`: NaN/Infinity are always
        an error (never silently filled), and axes must match ``reference``
        exactly -- no reindexing, which could otherwise mask a temporal
        misalignment (look-ahead) between the decision frame and the
        signal it is meant to diagnose.
        """
        if not isinstance(decision, pd.DataFrame):
            raise StrategyError("decision_signal() must return a pandas DataFrame.")
        if not decision.index.equals(reference.index) or not decision.columns.equals(
            reference.columns
        ):
            raise StrategyError(
                "decision_signal() index and columns must exactly match "
                "generate_signals()'s own output."
            )
        try:
            values = decision.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise StrategyError(
                "decision_signal() must contain only numeric values."
            ) from exc
        if not np.isfinite(values).all():
            raise StrategyError("decision_signal() must not contain NaN or Infinity.")
        return decision.astype(float)

    @staticmethod
    def _validate_signal_reasons(
        detail_code: pd.DataFrame, details: pd.DataFrame, reference: pd.DataFrame
    ) -> SignalReasons:
        """Validate an ``explain_signals()`` result against its reference.

        ``reference`` is whatever axes ``generate_signals()`` itself used
        (its own price/signal matrix) -- both frames must match exactly,
        the same requirement ``_validate_signals`` already enforces for
        the numeric signal matrix.
        """
        normalized: dict[str, pd.DataFrame] = {}
        for frame, name in ((detail_code, "detail_code"), (details, "details")):
            if not isinstance(frame, pd.DataFrame):
                raise StrategyError(f"SignalReasons.{name} must be a pandas DataFrame.")
            if not frame.index.equals(reference.index) or not frame.columns.equals(
                reference.columns
            ):
                raise StrategyError(
                    f"SignalReasons.{name} index and columns must exactly match "
                    "the price matrix."
                )
            # Normalize to plain object dtype with real None for missing
            # cells -- assigning a None/str column into a DataFrame can
            # get silently promoted to pandas' StringDtype, whose missing
            # marker is NaN rather than None (bites even careful callers,
            # not just naive ones), and downstream code (engine.py,
            # trade_log.py) relies on a strict `is None` check.
            frame = frame.astype(object).where(frame.notna(), None)
            bad = frame.map(
                lambda value: value is not None and not isinstance(value, str)
            )
            if bad.to_numpy().any():
                raise StrategyError(f"SignalReasons.{name} values must be str or None.")
            normalized[name] = frame
        return SignalReasons(
            detail_code=normalized["detail_code"], details=normalized["details"]
        )

    def _freeze_parameters(self) -> None:
        object.__setattr__(self, "_strategy_parameters_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent public parameter mutation after construction.

        ``_NON_PARAMETER_ATTRIBUTES`` (e.g. ``symbol_calendars``) is
        exempted: engine-injected context set on the instance after
        construction, not a user-supplied hyperparameter the freeze is
        meant to protect.
        """
        if (
            getattr(self, "_strategy_parameters_frozen", False)
            and not name.startswith("_")
            and name not in self._NON_PARAMETER_ATTRIBUTES
        ):
            raise AttributeError(
                "Strategy parameters are immutable after construction."
            )
        object.__setattr__(self, name, value)

    def parameters(self) -> dict[str, Any]:
        """Return a defensive copy of public strategy parameters."""
        return copy.deepcopy(
            {
                key: value
                for key, value in vars(self).items()
                if not key.startswith("_") and key not in self._NON_PARAMETER_ATTRIBUTES
            }
        )

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        params = ", ".join(
            f"{key}={value!r}" for key, value in self.parameters().items()
        )
        return f"{type(self).__name__}({params})"
