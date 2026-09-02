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
| `mean_reversion` | `MeanReversionStrategy` | A stateful entry/exit/stop machine driven by a chosen `indicator` (rolling z-score, Bollinger %B, RSI, distance to moving average, or percentile rank). |
| `trend_following` | `TrendFollowingStrategy` | Trailing MA-crossover direction; sizing belongs to the portfolio allocator. |
| `pairs_trading` | `PairsTradingStrategy` | Trailing regression residual, ADF-gated entries, and legs sized by the relative dollar-notional hedge ratio implied by beta and current prices. |

## Signals and allocators

An allocator decides how signal values become portfolio weights:

- `equal_weight` preserves signs but discards relative signal magnitudes.
- `signal_proportional` preserves relative signed magnitudes.
- `inverse_volatility` and `volatility_targeting` add their own risk sizing.

Consequently, non-binary time-series or cross-sectional momentum signals
require an allocator that preserves signal magnitudes and cannot use
`equal_weight`; time-series momentum's `volatility_adjusted` mode
additionally cannot be combined with another inverse-volatility allocator.
Pairs trading requires `signal_proportional`; per-leg
weight/minimum-size constraints that would distort or remove one hedge leg are
rejected by configuration validation. A portfolio-level volatility target is
still valid because it scales both pair legs together.

The pairs regression estimates a share hedge ratio from the preceding
`formation_window`. QuantLab converts it to relative dollar notionals using
the current prices. The ADF test is run periodically on the complete trailing
formation-window residual and an inconclusive test prevents a new entry.

## Stop-loss / take-profit

Every strategy except `buy_and_hold` accepts optional `stop_loss_pct`/
`take_profit_pct` parameters (fractional, `None` by default). Unlike
`mean_reversion`'s `stop_threshold` (an indicator-level stop) or
`pairs_trading`'s own `stop_threshold` (a spread-indicator-level stop),
these operate on
the position actually executed by the allocator/constraints/rebalancing/
execution pipeline, not on the strategy's own signal or indicator — see
[Backtesting: Stop-loss / take-profit](backtesting.md#stop-loss-take-profit)
for the exact mechanism, including how `pairs_trading`'s two legs are
force-flattened together on their combined P&L via
`BaseStrategy.position_groups()`.

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

To also give it a dashboard [Strategy Explorer](strategy_explorer.md) page
(research content, an interactive lab, optionally its own Results-tab/report
diagnostics), see that document's "Adding Strategy Explorer support for a
new strategy" section — a separate, optional step from registering the
strategy itself.

## Look-ahead safety

Every built-in strategy uses trailing windows only. `forward_returns` exists
for research labels and must never feed a trading signal. Accounting shifts
held weights by one bar before applying asset returns (see
[Backtesting](backtesting.md)); this prevents same-bar signal execution, but it
cannot make an incorrectly future-aware custom strategy causal. Custom
strategies remain responsible for never reading data after the signal date.
