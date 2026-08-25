"""Trading strategies.

Importing this package registers every built-in strategy so
:func:`build_strategy` can construct them by name from a config.
"""

from __future__ import annotations

from quantlab.strategies.base import (
    BaseStrategy,
    available_strategies,
    build_strategy,
    register_strategy,
    strategy_parameter_names,
    strategy_sweepable_parameter_names,
    validate_strategy_parameters,
)

# Import concrete strategies for their registration side effects.
from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.momentum import (
    CrossSectionalMomentumStrategy,
    TimeSeriesMomentumStrategy,
)
from quantlab.strategies.pairs_trading import (
    PairsTradingStrategy,
    adf_pvalue,
    rolling_hedge_parameters,
)
from quantlab.strategies.trend_following import TrendFollowingStrategy

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "CrossSectionalMomentumStrategy",
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "TimeSeriesMomentumStrategy",
    "TrendFollowingStrategy",
    "adf_pvalue",
    "available_strategies",
    "build_strategy",
    "register_strategy",
    "rolling_hedge_parameters",
    "strategy_parameter_names",
    "strategy_sweepable_parameter_names",
    "validate_strategy_parameters",
]
