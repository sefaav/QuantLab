"""Strategy Explorer content registry.

Mirrors ``quantlab.strategies.base``'s registration pattern: each strategy
declares its own pedagogical profile in ``explorer/profiles/<name>.py``,
registered by calling :func:`register_profile` at import time. The Strategy
Explorer's own dispatch -- the gallery/detail pages and the optional
Results-tab/report diagnostics -- only ever asks "does the current
strategy's profile declare X", never special-cases a strategy by name. This
does not extend to the rest of the dashboard: the regular Backtest/
Walk-forward sidebar in ``app.py`` still branches on a strategy's name to
render its own config widgets, unrelated to this registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from quantlab.config import ExperimentConfig
    from quantlab.reporting.sections import DiagnosticsSection


@dataclass(frozen=True)
class ParameterDoc:
    """Full explanation of one strategy constructor parameter.

    ``what``/``where``/``why`` answer "what is it", "where in the signal
    pipeline does it act" and "why does it exist". ``effect_increase``/
    ``effect_decrease`` describe the observable consequence of moving it
    in each direction -- the strategy's interactive lab should let a user
    actually see this happen, not just read about it (see
    :attr:`StrategyProfile.lab`).
    """

    name: str
    what: str
    where: str
    why: str
    default: str
    typical_range: str
    effect_increase: str
    effect_decrease: str
    tradeoffs: str
    interactions: str = ""


@dataclass(frozen=True)
class ResultsDiagnostics:
    """A strategy's own extra Results-tab/report diagnostics, if any.

    ``compute`` takes the already-loaded canonical price/OHLCV frame and
    the experiment config and returns a strategy-specific structured
    result (e.g. ``PairDiagnostics``). ``render`` displays that result in
    the dashboard Results tab (``st`` injected as the first argument, the
    structured result as the second). ``report_section`` turns the same
    structured result into a generic :class:`~quantlab.reporting.sections.
    DiagnosticsSection` for the HTML report. ``key`` doubles as the
    ``robustness`` dict key the CLI attaches it under and the
    ``st.session_state`` key the dashboard stores it under -- must be
    unique across every registered profile.
    """

    key: str
    compute: Callable[[pd.DataFrame, ExperimentConfig], Any]
    render: Callable[[Any, Any], None]
    report_section: Callable[[Any], DiagnosticsSection]


@dataclass(frozen=True)
class StrategyProfile:
    """The Strategy Explorer's complete content for one registered strategy.

    ``strategy_name`` must match a name in ``quantlab.strategies.base.
    available_strategies()``. Every markdown field is plain text/Markdown
    rendered inside a collapsible ``st.expander`` section on the detail
    page; none are required to be exhaustive on their own -- the
    ``parameters`` list and the interactive ``lab`` carry the bulk of the
    "make every parameter genuinely understood" requirement.
    """

    strategy_name: str
    display_name: str
    category: str
    overview_md: str
    economic_intuition_md: str
    mathematical_definition_md: str
    assumptions_md: str
    diagnostics_md: str
    interpretation_md: str
    limitations_md: str
    parameters: list[ParameterDoc]
    lab: Callable[[Any], None]
    references_md: str | None = None
    results_diagnostics: ResultsDiagnostics | None = None


_REGISTRY: dict[str, StrategyProfile] = {}

#: Text fields every profile must actually fill in -- an empty one would
#: silently render as a blank expander section rather than fail loudly at
#: registration time.
_REQUIRED_MARKDOWN_FIELDS = (
    "overview_md",
    "economic_intuition_md",
    "mathematical_definition_md",
    "assumptions_md",
    "diagnostics_md",
    "interpretation_md",
    "limitations_md",
)


def register_profile(profile: StrategyProfile, *, replace: bool = False) -> None:
    """Register a strategy's Strategy Explorer content.

    Raises if a profile is already registered for ``profile.strategy_name``
    unless ``replace=True`` (mirrors ``register_strategy``'s own guard
    against accidental double-registration). Also enforces the contracts
    ``StrategyProfile``/``ResultsDiagnostics`` document but did not
    previously check: ``strategy_name`` names a real registered strategy,
    every markdown field actually has content, no ``ParameterDoc`` name is
    duplicated, ``compute``/``render``/``report_section`` are callable when
    ``results_diagnostics`` is set, and (when it is) its ``key`` does not
    collide with another already-registered profile's -- that key doubles
    as the ``robustness`` dict key and the dashboard ``st.session_state``
    key, so a collision would silently let two strategies clobber each
    other's diagnostics.
    """
    if not isinstance(profile, StrategyProfile):
        raise TypeError("profile must be a StrategyProfile.")
    if profile.strategy_name in _REGISTRY and not replace:
        raise ValueError(
            f"A profile is already registered for '{profile.strategy_name}'."
        )
    # Local import (not module-level): forces `quantlab.strategies` to
    # finish registering every built-in strategy right here if a caller
    # hasn't already imported it, so this check is correct regardless of
    # import order rather than only when this module happens to run after
    # `quantlab.strategies` elsewhere.
    from quantlab.strategies.base import available_strategies

    if profile.strategy_name not in available_strategies():
        raise ValueError(
            f"Profile strategy_name {profile.strategy_name!r} is not a "
            f"registered strategy. Registered: {available_strategies()}."
        )
    for field in _REQUIRED_MARKDOWN_FIELDS:
        if not getattr(profile, field).strip():
            raise ValueError(
                f"Profile '{profile.strategy_name}': {field} must not be empty."
            )
    if not profile.display_name.strip():
        raise ValueError(f"Profile '{profile.strategy_name}': display_name is empty.")
    if not profile.category.strip():
        raise ValueError(f"Profile '{profile.strategy_name}': category is empty.")
    parameter_names = [parameter.name for parameter in profile.parameters]
    duplicate_parameters = {
        name for name in parameter_names if parameter_names.count(name) > 1
    }
    if duplicate_parameters:
        raise ValueError(
            f"Profile '{profile.strategy_name}': duplicate ParameterDoc "
            f"name(s) {sorted(duplicate_parameters)}."
        )
    for parameter in profile.parameters:
        # `interactions` is deliberately excluded: "" is its documented
        # default for a parameter that genuinely has none to report.
        for parameter_field in (
            "name",
            "what",
            "where",
            "why",
            "default",
            "typical_range",
            "effect_increase",
            "effect_decrease",
            "tradeoffs",
        ):
            if not getattr(parameter, parameter_field).strip():
                raise ValueError(
                    f"Profile '{profile.strategy_name}': ParameterDoc "
                    f"{parameter.name!r}'s {parameter_field} must not be empty."
                )
    if profile.results_diagnostics is not None:
        for callback_name in ("compute", "render", "report_section"):
            if not callable(getattr(profile.results_diagnostics, callback_name)):
                raise ValueError(
                    f"Profile '{profile.strategy_name}': "
                    f"results_diagnostics.{callback_name} must be callable."
                )
        if not profile.results_diagnostics.key.strip():
            raise ValueError(
                f"Profile '{profile.strategy_name}': results_diagnostics.key "
                "must not be empty."
            )
        key = profile.results_diagnostics.key
        colliding = [
            name
            for name, existing in _REGISTRY.items()
            if name != profile.strategy_name
            and existing.results_diagnostics is not None
            and existing.results_diagnostics.key == key
        ]
        if colliding:
            raise ValueError(
                f"Profile '{profile.strategy_name}': results_diagnostics.key "
                f"{key!r} collides with already-registered profile(s) "
                f"{colliding} -- this key must be unique across every "
                "registered profile."
            )
    _REGISTRY[profile.strategy_name] = profile


def get_profile(strategy_name: str) -> StrategyProfile | None:
    """Return the registered profile for a strategy, or ``None``."""
    return _REGISTRY.get(strategy_name)


def available_profiles() -> list[str]:
    """Return the names of every strategy with a registered profile."""
    return sorted(_REGISTRY)
