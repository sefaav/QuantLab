# Strategies

## The contract

Every strategy implements `BaseStrategy.generate_signals(data, features) ->
pd.DataFrame`:

- **Index**: dates.
- **Columns**: symbols.
- **Values**: in `[-1, 1]` — `+1` max long, `0` flat, `-1` max short.

Strategies produce signals, not portfolio weights, costs, or returns. Their
output must have exactly the same dates and symbols as the adjusted-close
matrix. Missing warm-up signals become zero; infinite values, duplicate axes,
misaligned axes, and finite values outside `[-1, 1]` are rejected.

## Built-in strategies

| Name (config `strategy.name`) | Class | Idea |
| --- | --- | --- |
| `buy_and_hold` | `BuyAndHoldStrategy` | Constant long signal for every asset with a valid price; the allocator and rebalance cadence determine the actual portfolio. |
| `time_series_momentum` | `TimeSeriesMomentumStrategy` | Per-asset trailing momentum; binary, continuous, or volatility-adjusted scaling. |
| `cross_sectional_momentum` | `CrossSectionalMomentumStrategy` | Ranks assets by trailing momentum, longs the top fraction, and can short a disjoint bottom fraction. |
| `mean_reversion` | `MeanReversionStrategy` | Rolling z-score with a stateful entry, exit, and stop machine. |
| `trend_following` | `TrendFollowingStrategy` | Trailing MA-crossover direction; sizing belongs to the portfolio allocator. |
| `pairs_trading` | `PairsTradingStrategy` | Trailing regression residual, ADF-gated entries, and price-adjusted dollar hedge legs. |

## Signals and allocators

An allocator decides how signal values become portfolio weights:

- `equal_weight` preserves signs but discards relative signal magnitudes.
- `signal_proportional` preserves relative signed magnitudes.
- `inverse_volatility` and `volatility_targeting` add their own risk sizing.

Consequently, non-binary time-series momentum cannot use `equal_weight`, and
its `volatility_adjusted` mode cannot be combined with another inverse-
volatility allocator. Pairs trading requires `signal_proportional`; per-leg
weight/minimum-size constraints that would distort or remove one hedge leg are
rejected by configuration validation. A portfolio-level volatility target is
still valid because it scales both pair legs together.

The pairs regression estimates a share hedge ratio from the preceding
`formation_window`. QuantLab converts it to relative dollar notionals using
the current prices. The ADF test is run periodically on the complete trailing
formation-window residual and an inconclusive test prevents a new entry.

## Adding a new strategy

```python
from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantlab.strategies.base import BaseStrategy, register_strategy


@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def __init__(self, lookback_period: int = 20) -> None:
        values = self.validate_parameters({"lookback_period": lookback_period})
        self.lookback_period = values["lookback_period"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(parameters)
        if isinstance(values["lookback_period"], bool) or not isinstance(
            values["lookback_period"], int
        ):
            raise ValueError("lookback_period must be an integer")
        if values["lookback_period"] <= 0:
            raise ValueError("lookback_period must be positive")
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        prices = self._prices(data)
        signal = ...  # trailing-only strategy logic
        return self._validate_signals(signal, prices)
```

Register it by importing the module once, then reference its name in
`strategy.name`. `validate_parameters()` lets configuration loading validate
relations without constructing the strategy. Constructors should still
validate direct Python use, and `_freeze_parameters()` prevents later public
parameter mutation.

## Look-ahead safety

Every built-in strategy uses trailing windows only. `forward_returns` exists
for research labels and must never feed a trading signal. Accounting shifts
held weights by one bar before applying asset returns (see
[Backtesting](backtesting.md)); this prevents same-bar signal execution, but it
cannot make an incorrectly future-aware custom strategy causal. Custom
strategies remain responsible for never reading data after the signal date.
