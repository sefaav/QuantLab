# Weight-drift mechanics and the compliance-restoration LP

Detailed internal mechanism behind `portfolio.model_weight_drift` — see
[Weight drift](backtesting.md#weight-drift) for the conceptual summary,
invariants and disclosed limitations. This page is for contributors
modifying `quantlab.backtesting.accounting.apply_weight_drift` or
`quantlab.portfolio.drift_compliance.restore_drift_compliance`, not
general usage.

## Per-column debt

`apply_weight_drift` evolves the already shifted, executed weights forward
between genuine trades via a per-column dollar exposure and a single
shared relative equity `E`:

```
weight[i] = dollar[i] / E
```

Output is always a *pre-period* value — the weight held going into a row,
before that row's own return is applied — consistent with `executed_weights
= held.shift(1)` elsewhere in the accounting module; returning the
post-period value instead would double-count that row's own return once
inside this recursion and again when it is multiplied by `asset_return` a
second time.

Two independent kinds of per-column debt drive every row's output, in this
priority order:

1. **Compliance debt** — a hard-risk-limit breach (`maximum_weight`/
   `maximum_gross_exposure`/`maximum_net_exposure`/`long_only`), corrected
   via the minimal-L1-turnover linear program described below — never a
   "clip and scale toward zero" heuristic, which can move exposure in the
   wrong direction whenever some of the breaching exposure sits in a
   currently-untradable column. A declared position group (e.g.
   `pairs_trading`'s two legs, see `BaseStrategy.position_groups()`) is
   corrected as one coherent unit via a single shared scaling factor,
   never one leg moving alone. Re-solved fresh every row it is
   outstanding, from that row's own current weights, never a stale stored
   target — a column the LP leaves untouched (fixed/untradable, or simply
   not needing to move) always reflects its own current, continued drift
   when the correction lands, not a snapshot from whenever the breach was
   first detected. Any improvement the LP finds is applied immediately,
   whether or not it fully resolves the breach — a tradability-blocked
   best-effort partial fix (recorded via `drift_compliance_pending`) still
   lands, rather than being recomputed and silently discarded every day
   while nothing improves. Never subject to `maximum_turnover` — a hard
   risk-limit override, not an ordinary rebalance — and always takes
   priority over ordinary debt below.
2. **Ordinary rebalance debt** — a fresh per-column decision (a column
   anchors when its own value in the executed book changed since the
   previous row — the same `EPSILON`-based "did a real trade happen"
   convention the trade log itself uses — or `rebalance_date.loc[t,
   column]` is `True`, a `dates x symbols` frame threaded down from
   `engine.py`, itself gated by that column's own tradability; the second
   condition catches a scheduled rebalance whose freshly-decided target
   happens to numerically equal the immediately preceding one, otherwise
   invisible to value-diffing alone), turnover-capped like a decision-level
   rebalance when `PortfolioConfig.maximum_turnover` is set. A column's own
   fresh decision replaces only that column's own debt, never any other
   column's — an unrelated column's own outstanding debt, including one
   belonging to a currently-closed instrument, survives untouched. All
   columns with outstanding debt that are also currently tradable share
   one combined per-row turnover budget; the unresolved remainder is
   retried against a fresh budget each subsequent row. A column with debt
   that is not currently tradable never trades — its debt simply waits,
   exactly like a decision-level pending-due-to-closure debt — whether
   that is its own fresh anchor or a multi-row catch-up already in
   progress. A column compliance debt (1) already moved this row is
   excluded from ordinary-debt eligibility this same row too — its landed
   value already reflects the higher-priority correction; ordinary debt
   resumes toward it again starting next row — which is what keeps the
   two kinds of debt from fighting over the same cell within a single row.

A closed asset's dollar exposure does not move (its return is `0` on a
synthetic closure bar), but its weight still drifts purely through `E`'s
own movement from every other tradable asset's real return — it is never
force-reset to a stale decision just because another column anchors or a
compliance correction lands the same row. Combining this row's own
just-decided/corrected columns with another column's frozen or
still-drifting value can itself create a new violation neither component
had alone (mixing is not a convex combination the way interpolating along
one line is) — unlike organic drift, which needs a one-row lag before
reacting to it (correcting it retroactively would be look-ahead), every
input to this combination is already known before the row's own output is
finalized, so it is checked and, where achievable, resolved in the same
row. A row where nothing was decided (pure drift) keeps the ordinary
one-row-lag detect-then-queue behavior: a breach found there is recorded
in `drift_compliance_pending` and only lands starting the next row, via
compliance debt (1) above.

If `E` (this anchor-episode's own relative equity — gross, pre-cost,
distinct from the portfolio's absolute equity curve) falls to zero or
below, that episode's positions are force-flattened and a warning is
logged, mirroring `ruined`'s own "flatten and continue" handling rather
than aborting.

Whenever the row-walk believes no compliance debt remains outstanding for
a row (no `drift_compliance_pending`), that row's landed weights are
re-verified against the same constraints one more time before being
finalized. A violation there is treated as an internal or numerical
failure and raises `BacktestError` — a numerical solver failure remains
possible in principle, so this is not asserted as strictly unreachable,
but it should never be a legitimate, expected runtime outcome (mirrors
`rebalancing._assert_holdings_compliant`'s identical "never trust the
invariant blindly" philosophy).

`model_weight_drift=False` remains available as an optional constant-
weight compatibility mode (byte-identical to the step function described
in [Rebalancing & turnover](backtesting.md#rebalancing-turnover)), not
the recommended path.

## Compliance-restoration LP

`quantlab.portfolio.drift_compliance.restore_drift_compliance` is the
minimal-L1-turnover linear program compliance debt (above) uses. Internal,
low-level primitive: its sole caller, `apply_weight_drift`, is the
validated public entry point (frame shape/dtype/finiteness, tradable mask,
etc.) — this function only asserts array shapes match `columns`, trusting
`drifted`/`tradable_row` are otherwise already clean numeric/boolean data.

**Free variables**: `w_i` for every independent (singleton-group) column,
and one shared scalar `k_g` per multi-column group with `w_i := k_g *
drifted_i` for every leg `i` of group `g` (a declared position group —
e.g. `pairs_trading`'s two legs — must move together, preserving its
current relative composition/hedge ratio exactly; `k_g` is bounded only by
`k_g >= 0`, never capped at `1`, since the L1 objective already penalizes
any movement away from `drifted` and capping would incorrectly exclude a
genuine minimal-L1 solution that requires growing a group). An untradable
independent column, or any group with at least one untradable leg, is
fixed at its drifted value (cannot move this row) — eligibility is the AND
of every member's own native tradability, mirroring
`quantlab.strategies.pairs_trading`'s own "both legs open" gate.

**Sign/support-preserving bounds**: an independent column's own
free-variable bound never invents a position the drifted book does not
already hold — a currently-long column (`drifted_i > 0`) may shrink toward
0 or grow further long, a currently-short column (`drifted_i < 0`) may
shrink toward 0 or grow further short, and a column already exactly at 0
is fixed there. `long_only` additionally clamps every tradable independent
column's lower bound to 0 (grouped legs are assumed already non-negative
under `long_only` by construction — every anchor this module drifts from
was itself validated compliant, and a positive quantity cannot cross zero
under any return greater than -100%, so `k_g >= 0` alone keeps a grouped
leg non-negative too). That assumption covers every ordinary case, but is
not itself enforced by anything for a fixed (untradable) column/group — an
explicit `w_i >= 0` (`k_g * drifted_i >= 0` for a group leg) inequality is
added for every independent column/group leg regardless of tradability, so
a fixed value that does turn out negative under `long_only` is correctly
reported as LP-infeasible (driving the tradability diagnosis below)
instead of the strict LP trivially "succeeding" over a value that still
violates `long_only`.

**Objective**: `minimize sum_i |w_i - drifted_i|` subject to `|w_i| <=
maximum_weight`, `sum_i |w_i| <= maximum_gross_exposure` (or
`maximum_leverage`, whichever is tighter — the caller passes the
already-combined cap), `|sum_i w_i| <= maximum_net_exposure`, and the
sign-preserving bounds above. This is **not** solved by a "clip each
weight, then scale the tradable columns toward 0" heuristic — that is
provably wrong in general: e.g. a large untradable long position plus an
already-open, tradable short position breaching `maximum_net_exposure` on
the long side needs that short pushed more negative, not scaled toward 0
(scaling toward 0 moves net exposure the wrong direction). The linear
program gets this right because `w_i` for that tradable short is free to
move further in its own direction (not constrained to move only toward
zero) — while still never opening a brand-new position on a column the
drifted book held at exactly zero.

**Infeasibility handling**: if that LP is infeasible, first diagnoses
whether this is explained entirely by the fixed (untradable) columns'/
groups' own values already violating a constraint on their own
(tradability-caused, expected) rather than a genuinely contradictory
constraint configuration (a bug — `_validate_target_row_compliant` should
already have rejected that at the target itself, before drift ever ran).
Tradability-caused infeasibility re-solves via an always-feasible,
two-stage lexicographic LP instead of requiring zero violation: stage 1
minimizes the sum of four non-negative slacks (one added to each cap's own
right-hand side — `maximum_weight`, gross, net, `long_only`), finding the
smallest unavoidable violation; stage 2 then fixes those slacks at their
stage-1-optimal values and re-solves for the minimal-L1-deviation point
among every solution achieving that same minimal violation — so an
already-compliant, uninvolved free column is never moved (e.g. liquidated)
just because nothing in a single-stage "minimize slack only" objective
would have penalized doing so. The result is the best achievable
correction using only what is currently tradable, returned with
`pending=True`. A genuine misconfiguration instead raises `BacktestError`,
mirroring `_assert_holdings_compliant`'s "a violation here means a bug in
the algorithm" philosophy.

**Two deliberate, disclosed scope limits**:

- Only the hard risk limits above are enforced here — `target_minimum_
  weight`/`target_maximum_positions` are not, consistent with
  `PortfolioConfig`'s own documented convention that these non-convex,
  target-portfolio-only constraints may be temporarily violated by any
  transitional weights (turnover-capped rebalancing already has this exact
  property; an off-schedule drift correction is the same kind of
  transitional state, not a new target).
- Sign/support-preservation means the strict LP can, in principle, be
  infeasible in a case an unrestricted LP (free to open a brand-new
  position) would still solve — in practice this never actually happens:
  shrinking every free tradable column toward 0 always drives
  `maximum_weight`/gross/net exposure back toward whatever the fixed
  (untradable) columns alone already produce, so whenever the fixed
  columns alone are compliant, a fully sign-preserving compliant point is
  always reachable by shrinking alone — the strict LP never spuriously
  fails for this reason. When the LP has multiple equally-minimal-L1-cost
  optimal solutions, HiGHS returns some optimal vertex with no secondary
  preference toward leaving more columns untouched — the returned row is
  always a genuinely minimal-L1, fully compliant, sign-preserving
  correction, but is not guaranteed to be the unique one that moves the
  fewest columns among several equally-cheap alternatives.
