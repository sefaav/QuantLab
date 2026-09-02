"""Strategy Explorer: the dashboard's per-strategy research/education mode.

The registry in :mod:`quantlab.dashboard.explorer.profile` is this
package's public extension point: a new strategy gains a gallery card and
detail page purely by registering a :class:`StrategyProfile` in its own
``explorer/profiles/<name>.py`` module (mirroring how
``quantlab.strategies.base.register_strategy`` works for trading logic
itself) -- nothing in ``app.py``/``cli.py``/``html_report.py`` needs to
change or name the new strategy.
"""

from __future__ import annotations

from quantlab.dashboard.explorer.profile import (
    ParameterDoc,
    ResultsDiagnostics,
    StrategyProfile,
    available_profiles,
    get_profile,
    register_profile,
)

__all__ = [
    "ParameterDoc",
    "ResultsDiagnostics",
    "StrategyProfile",
    "available_profiles",
    "get_profile",
    "register_profile",
]
