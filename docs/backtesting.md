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
10. When `portfolio.model_weight_drift` is enabled, evolve the shifted,
    executed weights forward by organic price drift between real trades
    (see [Weight drift](#weight-drift)) — otherwise unchanged, holding
    them constant until the next scheduled rebalance.
11. Solve turnover, costs, gross/net returns and equity, including
    equity-dependent volume slippage.
12. Build the benchmark and trade log from the same dates and cost assumptions.
13. Compute metrics and assemble the `BacktestResult`.

## Preventing look-ahead bias

The return earned in period `t` must come from a position **decided before**
`t`. `quantlab.backtesting.accounting.run_accounting` enforces this via
`quantlab.execution.orders.executed_weights`:

```python
executed_weights = compute_executed_weights(
    held_weights,
    tradable=tradable,
)
gross_returns = (executed_weights * asset_returns).sum(axis=1)
```

For a single-calendar universe (`tradable=None`) this reduces to the simple
mental model `held_weights.shift(1)` (first row zeroed); when a per-symbol
`tradable` mask is given, the shift is per-symbol tradability-aware instead
— a decision made right before a closure lands on that symbol's own next
tradable row, not the raw next row, so it is never misattributed as trading
during the closure itself.

This is directly unit-tested (`tests/unit/test_accounting.py`): a signal that
turns long on date *i* must show zero gain on date *i* and only starts
capturing returns from date *i + 1*.

## Equity accounting

```
equity_0 = initial_capital
equity_t = equity_{t-1} * (1 + net_return_t)
```

Gross and net equity curves are both retained, so cost drag can be
inspected via `result.gross_net_comparison()`.

## Stop-loss / take-profit

A strategy's `stop_loss_pct`/`take_profit_pct` (fractional, e.g. `0.10` =
10%, `None` by default on every built-in strategy -- disabled with
strictly no change to accounting's numbers) force-flatten a position when
its cumulative return since entry breaches the configured threshold. This
operates on the **real executed position** (`accounting.executed_weights`,
after the allocator, portfolio constraints, rebalancing schedule and
turnover cap), never on a strategy's raw signal -- a signal is not
necessarily a realized position. For a symbol/group `G`, at each date:

```
gross_exposure = sum(|executed_weight| for each symbol in G)
group_return   = sum(executed_weight * asset_return for each symbol in G) / gross_exposure
```

`group_return` is per unit of the group's *own* realized exposure that
date, not a dollar contribution to total portfolio equity — this makes it
correct regardless of a static or dynamic hedge ratio, rebalancing,
weight changes, long/short direction or partial fills. For a single
symbol it reduces to `sign(executed_weight) * asset_return`, the standard
price-based stop-loss/take-profit. A strategy declares a multi-symbol
group via `BaseStrategy.position_groups()` (e.g. `pairs_trading`'s two
legs, so a stop-loss triggers on the pair's *combined* P&L, not either
leg's own return in isolation) — every symbol not covered by a declared
group is its own independent group. Thresholds are evaluated on **gross
(pre-cost) return**: QuantLab's execution cost model is portfolio-level
only (no per-symbol/per-group cost decomposition), so an exact net-of-cost
trigger is not presently computable — a deliberate, disclosed design
convention. Once triggered, no immediate re-entry at a rebased price: the
position stays flat until its next real flat-to-non-flat transition. See
`quantlab.backtesting.accounting._detect_stop_loss_take_profit`.

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

This section describes `rebalancing.py`'s own DECISION-timeline output --
the *rebalance target* the strategy/allocator/constraints machinery
decides to chase. It is not automatically the *weight actually held*, and
its own turnover formula is not automatically the *real trade* executed:
when `portfolio.model_weight_drift` is enabled (the default -- see
[Weight drift](#weight-drift) below), organic price drift between real
trades changes the weight actually held continuously, and a genuine trade
is instead reported by `apply_weight_drift`'s own `trade_changes` output --
exactly zero on a pure-drift row (price moved; nothing was traded) and the
real size on an anchor or a landed compliance/turnover-cap correction.

Daily allocator output is a *target*; `apply_rebalancing` samples it on
rebalance dates (daily / weekly / monthly / quarterly) and holds it constant
between them — at THIS decision-timeline layer, trades, and therefore
costs, only occur at rebalances. Turnover is `sum(|held_t - held_{t-1}|)`
(`w_{-1} = 0`), directly matching the manual example of capital 100k,
turnover 0.5, 10 bps → cost 50.

For a mixed-calendar portfolio, `rebalance_and_cap_turnover` is
tradability-aware: a closed instrument (per its own calendar, see
[Data pipeline](data_pipeline.md#verified-closures-vs-missing-data)) never
trades on a closed date, and its rebalance target becomes a pending debt that
keeps retrying — at every following tradable session, not only the next
scheduled rebalance — until fully executed, even if `maximum_turnover` spreads
that execution across several sessions. Portfolio constraints (gross/net
exposure, max weight, long-only) are enforced on the actually-executed
holdings after accounting for frozen/closed instruments, not just on the
theoretical fully-open target, since freezing one instrument while others move
can push the real portfolio out of its mandate even when the target was
compliant. For a single-calendar experiment this machinery is a proven no-op:
behaviour is byte-identical to the plain rebalance/turnover-cap path above.

`rebalance_and_cap_turnover`'s output — including a pending target resolving
there on a reopening day — is still only a *decision*, dated that day. Every
decision, on a reopening day or an ordinary rebalance date alike, is subject
to the same one-period (tradability-respecting) look-ahead shift applied by
the accounting layer before it affects executed weights, turnover or costs —
a target that resolves in `held_weights` on a symbol's reopening day therefore
does not reach the accounting layer until that symbol's *next* tradable
session, not the reopening day itself.

## Weight drift

Everything above describes `rebalancing.py`'s own output: a decision-timeline
step function, constant between rebalance dates. A real portfolio does not
actually stay constant between trades — each asset's own price move drifts
its dollar exposure, and therefore its weight, continuously. When
`portfolio.model_weight_drift` is `True` (the default),
`quantlab.backtesting.accounting.apply_weight_drift` evolves the already
shifted, executed weights forward between genuine trades, via a per-column
dollar exposure and a single shared relative equity `E` (`weight[i] =
dollar[i] / E`).

Conceptually, two independent kinds of debt drive every row's output, in
priority order: a hard-risk-limit breach (`maximum_weight`/`maximum_gross_
exposure`/`maximum_net_exposure`/`long_only`) is corrected first, via a
genuine linear program — never a "clip and scale toward zero" heuristic,
which can move exposure in the wrong direction — that finds the minimal-
turnover point restoring compliance; an ordinary fresh rebalance decision
is applied second, turnover-capped like a decision-level rebalance. See
[Weight-drift mechanics and the compliance-restoration LP](drift_compliance.md)
for the full per-column debt priority order, anchor detection, the
bankruptcy guard, and the LP's exact formulation — this section only
states the invariants and limitations a caller needs to know:

- Output is always a *pre-period* value — the weight held going into a
  row, before that row's own return is applied — never the post-period
  value, which would double-count that row's own return.
- A declared position group (e.g. `pairs_trading`'s two legs, see
  `BaseStrategy.position_groups()`) is always corrected as one coherent
  unit via a single shared scaling factor, never one leg moving alone.
- The correction is sign/support-preserving: an existing long may shrink
  or grow further long, an existing short may shrink or grow further
  short, but neither crosses zero, and a column already at exactly zero
  is never opened into a brand-new position — it can never invent a hedge
  the strategy's own signal never asked for.
- A closed asset's dollar exposure does not move, but its weight still
  drifts purely through `E`'s own movement from every other tradable
  asset's real return.
- A hard risk-limit breach detected using row `t`'s own drift cannot
  execute until row `t+1` at the earliest — the same look-ahead barrier
  as every other decision in this module — and if the responsible
  exposure sits in a currently-closed column, full correction may be
  impossible until it reopens; the LP applies the best achievable fix
  meanwhile, carrying the residual as a pending breach rather than
  raising or silently dropping it.
- Never produces `inf`/`NaN`: a bankrupt anchor-episode (relative `E <=
  EPSILON`) is force-flattened and logged instead of dividing by
  (near-)zero.
- `model_weight_drift=False` remains available as an optional constant-
  weight compatibility mode (byte-identical to the step function described
  above), not the recommended path.
- The compliance-restoration LP's own basis is gross/pre-cost, the same
  disclosed convention already used by `stop_loss_pct`/`take_profit_pct`.

## Trade-log reason attribution

`quantlab.backtesting.trade_log._classify_reason` assigns every fill's
`trigger_reason_code`/`adjustment_reason_codes` from real, per-layer
provenance signals — never deduced after the fact from `new != desired`.
`execution_delay`/the rebalance-sampling frequency are not adjustments:
they are uniform timing conventions baked into every comparison below, so
they shift *when* a trigger is consumed, never *what* explains one trade's
execution differing from another's.

**Trigger** — the single most-upstream event currently consumed that
initiated the target change (not exhaustive: when `strategy_signal` is the
trigger, a downstream layer subsequently recomputing the target is a
mechanical consequence of that same event, not separately lost
information):

| Priority | Code | Fires when |
| --- | --- | --- |
| 1 | `strategy_signal` | The strategy's own decision changed since the last rebalance (from its diagnostic decision proxy — `decision_signal()` when provided, else the raw signal). |
| 2 | `portfolio_rebalance` | Only the allocator's output changed. |
| 3 | `volatility_target_adjustment` | Only vol-targeting changed the target. |
| — | *(none)* | Nothing above changed. |

**Adjustment(s)** — collected independently of trigger, from each layer's
own real provenance signal; a higher-priority cause fully explains the row
and suppresses lower-priority ones (the precise clip value of a lower
layer becomes moot once a higher one applies). Priorities 1–3 never touch
`trigger`, which keeps reflecting what the strategy actually wanted:

| Priority | Code(s) | Fires when |
| --- | --- | --- |
| 1 | `forced_liquidation` | Portfolio ruin (`AccountingResult.ruined`) — overrides everything else. |
| 2 | `stop_loss` / `take_profit` | A real force-flatten (`AccountingResult.stop_loss_triggered`/`take_profit_triggered`) — overrides ordinary constraints, overridden by `forced_liquidation`. |
| 3 | `drift_compliance` / `drift_compliance_pending` | The drift-compliance LP restored (or attempted to restore) a hard risk limit breached by organic drift — overrides ordinary constraint/tradability/turnover_cap adjustments, overridden by a stop-loss/take-profit breach on that same corrected weight. |
| 4 | Constraint name(s), `tradability`, `turnover_cap` | A contributing constraint (direct/redistribution), a closure catch-up/feasibility limit, or the turnover budget itself. |
| 5 | `position_rescaling` / `deferred_catchup` | Last-resort fallback: the target is still drifting with no known trigger (e.g. pairs_trading's price/beta rescaling), or the causal layer is genuinely unknown — reached only when nothing above explains the row.
