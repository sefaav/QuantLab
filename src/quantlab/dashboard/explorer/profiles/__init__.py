"""Importing this package registers every strategy's Strategy Explorer profile.

Mirrors how ``quantlab.strategies`` triggers each strategy's own
``@register_strategy`` decorator by importing every strategy module --
``get_profile()``/``available_profiles()`` (``quantlab.dashboard.explorer.
profile``) are the actual public entry points, so nothing here needs
re-exporting.
"""

from __future__ import annotations

# Imported for their registration side effects (each module calls
# register_profile() at import time).
from quantlab.dashboard.explorer.profiles import (
    buy_and_hold,  # noqa: F401
    cross_sectional_momentum,  # noqa: F401
    mean_reversion,  # noqa: F401
    pairs_trading,  # noqa: F401
    time_series_momentum,  # noqa: F401
    trend_following,  # noqa: F401
)
