# Backtesting

## Pipeline order

The CLI and `BacktestEngine.run()` together execute this sequence:

1. Load, clean and validate data in `DataLoader`.
2. Isolate tradable symbols and compute adjusted-price returns.
3. Compute features inside the strategy and generate signals at `t`.
4. Transform signals into target weights with the allocator.
5. Apply optional portfolio-level volatility targeting.
6. Apply hard portfolio constraints.
7. Apply the configured rebalance schedule and turnover cap.
8. Apply any additional execution-delay stress assumption.
9. **Shift held weights one period before computing returns** — the
   look-ahead barrier.
10. Solve turnover, costs, gross/net returns and equity, including
    equity-dependent volume slippage.
11. Build the benchmark and trade log from the same dates and cost assumptions.
12. Compute metrics and assemble the `BacktestResult`.

## Preventing look-ahead bias

The return earned in period `t` must come from a position **decided before**
`t`. `quantlab.backtesting.accounting.run_accounting` enforces this with one
line:

```python
executed_weights = held_weights.shift(1).fillna(0.0)
gross_returns = (executed_weights * asset_returns).sum(axis=1)
```

This is directly unit-tested (`tests/unit/test_accounting.py`): a signal that
turns long on date *i* must show zero gain on date *i* and only starts
capturing returns from date *i + 1*.

## Equity accounting

```
equity_0 = initial_capital
equity_t = equity_{t-1} * (1 + net_return_t)
```

Gross and net equity curves are both kept so cost drag is always visible:
`result.gross_net_comparison()`.

## Missing data

If an asset return is missing while a non-zero position is held, accounting
raises `BacktestError`; it never invents a 0% move. A missing return on an
unheld asset contributes nothing. A row with no held exposure is treated as a
zero portfolio return.

## Short positions

`portfolio_return = weight * asset_return`. A negative weight combined with a
negative asset return correctly produces a gain — no special-casing needed;
this is a straight consequence of the formula and is unit-tested.

## Costs

Three components are charged from traded notional (expressed as a fraction of
equity when the model receives weight changes):

- **Commission**: `traded_notional * commission_bps / 10_000`
- **Spread**: `traded_notional * spread_bps / 20_000` (half-spread)
- **Slippage**: a constant rate (`bps / 10_000`) or a nonlinear volume-based
  impact rate (`base_bps + impact_coefficient * sqrt(order / ADV)`). The
  volume model requires finite positive ADV for every traded asset.

`ExecutionModel` aggregates all three; the trade log
(`quantlab.backtesting.trade_log`) itemises every fill with its own cost
breakdown.

## Rebalancing & turnover

Daily allocator output is a *target*; `apply_rebalancing` samples it on
rebalance dates (daily / weekly / monthly / quarterly) and holds it constant
between them — trades, and therefore costs, only occur at rebalances.
Turnover is `sum(|held_t - held_{t-1}|)` (`w_{-1} = 0`), directly matching the
manual example of capital 100k, turnover 0.5, 10 bps → cost 50.
