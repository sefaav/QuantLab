# Validation

A full-sample backtest is exploratory evidence. QuantLab provides chronological
holdout, walk-forward evaluation and robustness tools to test narrower claims.

## Chronological splits and holdout

`quantlab.validation.chronological_split` creates contiguous train, validation
and test blocks in time order. Ratios must be finite, between zero and one, and
sum to exactly one within floating-point tolerance.

The holdout report slices one continuously simulated return series. Positions,
warm-up and costs therefore carry naturally across block boundaries; each block
is not restarted from cash. The test block is out of sample only when the
strategy and its parameters were fixed without consulting that block.

## Walk-forward validation

`WalkForwardValidator.run(data, parameter_grid, train_window, validation_window, test_window, expanding=True, step=None)`
performs these steps:

1. Evaluate candidate parameters on each validation block -- each candidate is
   its own fresh backtest, restarted from cash on that block alone, not
   chained to any other candidate or fold.
2. Select the best finite score using `validation.optimization_metric`.
3. Apply that choice to the untouched test block.
4. Stitch every fold's test-block returns into one continuous OOS curve,
   preserving portfolio, turnover and accounting state *across fold
   boundaries only* -- the OOS curve is one simulated run, but candidate
   selection within a fold never sees that chained state.
5. Report the stitched OOS curve as `WalkForwardResult.oos_result`.

`step` controls how far each fold's train window advances relative to the
previous one. It defaults to `test_window` (contiguous, non-overlapping test
blocks -- the still-recommended default). A smaller `step` overlaps test
blocks for denser evaluation (more folds, more compute); on an overlapping
date the stitched OOS curve keeps the most recent fold's decision, and two
folds whose test blocks collapse onto the same first execution date are
rejected outright rather than silently misattributing observations. `step`
must not exceed `test_window`: a larger step would leave gaps in the
stitched OOS curve that CAGR/annualisation (which assume regularly spaced
observations) cannot account for, so this is rejected at run time too.

The Python API accepts an explicit `parameter_grid`. A YAML experiment can set
the same candidates under `validation.parameter_grid`. The CLI and momentum
research notebook both call `parameter_grid_for_config(config)`, which returns
that YAML grid when present and otherwise uses the small built-in grid from
`default_parameter_grid(config)`. This keeps their candidate sets identical;
call the validator directly only when a grid should remain outside the saved
experiment definition.

```yaml
validation:
  method: walk_forward
  train_window: 1000
  validation_window: 252
  test_window: 126
  step: 126  # optional; defaults to test_window
  optimization_metric: sharpe
  parameter_grid:
    lookback_period: [126, 189, 252]
    skip_period: [0, 21]
```

Every candidate and every cross-parameter combination is validated when the
YAML is loaded. Structural choices such as long-only versus long/short should
normally remain fixed in `strategy.parameters`, so one experiment answers one
research question.

`WalkForwardResult.parameter_stability()` reports coefficients of variation for
numeric selected parameters. Low variation is descriptive evidence of
consistent selection, not proof that the parameter or strategy is robust.

```bash
quantlab walk-forward --config configs/momentum_sp500.yaml
```

This writes the walk-forward CSV artefacts and incorporates compatible evidence
into the generated HTML report. Progress (and an ETA) is shown live in the
terminal or dashboard while it runs. An interruption (Ctrl+C, a crash, closing
the terminal) is resumed automatically the next time the same command runs
against the same experiment, config, data and code — only *completed* folds
are skipped; whichever fold was still in progress at the moment of
interruption is discarded and recomputed from its start, not resumed
mid-fold. Pass `--fresh` to discard all saved progress and start over
instead.
The same applies to `stress-test`, `sensitivity` and `robustness` in
walk-forward mode.

## Parameter sensitivity

`run_parameter_sensitivity(data, config, parameter_x, values_x, parameter_y, values_y)`
records Sharpe, CAGR, drawdown, turnover and trade count for every
combination. Failed configurations remain visible with a status and error
message. The useful pattern is a region with comparable behaviour, not
merely one isolated optimum.

## Bootstrap

`bootstrap_returns(returns, n_iterations, block_size, seed)` resamples returns
i.i.d. when `block_size=1` or in circular blocks otherwise. It reports the
sampled distributions of CAGR, Sharpe, maximum drawdown and final value. Block
sampling retains dependence within each sampled block, but does not reproduce
the complete time-series process. These are historical sampling estimates, not
forecasts.

`BootstrapResult.summary(confidence_level=0.90)` reports each statistic's
median plus a `p_lower`/`p_upper` percentile band at the requested confidence
level (0.90 -> the 5th/95th percentiles, the default). Set
`robustness.bootstrap.confidence_level` in YAML to change it for a saved
experiment's own bootstrap run.

## Panel inference for walk-forward studies

`bootstrap_returns`/`monte_carlo_permutation` above make an inferential
statement about *one* stitched return series. A multi-asset walk-forward
study instead produces a *panel*: one row per entity (a pair, an asset)
per fold, where rows sharing a fold share a market environment
(cross-sectional dependence) and rows sharing an entity recur across folds
(the same entity is not an independent draw each fold). Treating every row
as independent -- a plain t-test, or OLS with i.i.d. standard errors --
understates uncertainty and can manufacture significance out of pure
cross-sectional co-movement within a fold.
`quantlab.validation.panel_inference` resamples, permutes, or clusters at
one caller-chosen dependence level instead of treating every row as an
independent draw:

- `cluster_bootstrap_difference(y, treated, cluster, statistic="mean"|"median", n_boot, seed)` --
  a CI for `stat(y | treated) - stat(y | control)`, resampling WHOLE
  clusters (e.g. folds) with replacement rather than individual rows.
- `within_cluster_permutation_test(y, treated, cluster, statistic, n_perm, seed)` --
  a p-value for the same quantity, keeping cluster membership fixed and
  instead shuffling the treated/control LABELS within each cluster (a
  different mechanism from the bootstrap above, not a second way to
  resample whole clusters). Both are single-dimensional: whichever
  `cluster` label is passed is the only dependence structure corrected
  for -- entity-level dependence (the same entity recurring across folds)
  is not separately modelled unless the caller clusters by entity instead
  of by fold.
- `fama_macbeth_difference(frame, outcome, treated, cluster)` /
  `per_cluster_spearman(frame, x, y, cluster)` -- one statistic per
  cluster, then a t-test across clusters (`n_clusters` observations, each
  treated as one independent unit for THIS test's own standard error, not
  `n_rows`). Also single-dimensional, with an additional assumption: the
  across-cluster t-test treats each cluster's own statistic as an
  independent draw, which is not automatically true for temporally
  adjacent or overlapping folds (e.g. a walk-forward run with `step <
  test_window`) -- serial correlation between neighbouring folds' own
  outcomes is not corrected for here.
- `clustered_regression(frame, outcome, regressor, fixed_effects=None, clusters=(...))` --
  OLS with optional fixed effects and one- **or two-way** cluster-robust
  standard errors. This is the one function in this module that can
  address fold and entity dependence *simultaneously* (e.g.
  `clusters=(fold_column, entity_column)`); every other function above
  addresses only whichever single `cluster` column it is given.
- `paired_cluster_bootstrap_sharpe_difference(returns_a, returns_b, cluster_of_date, n_boot, seed, periods_per_year, risk_free_rate)` --
  a paired cluster-block bootstrap CI for the Sharpe-ratio difference
  between two stitched return series sharing a calendar. `risk_free_rate`
  is an annual rate subtracted before computing each Sharpe ratio
  (`0.0` default, i.e. the inputs are assumed already excess returns
  unless set). `p_value_two_sided` is an approximate bootstrap TAIL
  PROBABILITY (the same CI-exclusion duality a 95% CI already implies),
  not the general equivalent of a dedicated null-recentered or
  permutation-based test -- see the function's own docstring.
  Single-dimensional (clusters by date-derived label only).
- `holm_adjust(p_values)` -- Holm step-down family-wise error control for
  a set of p-values computed together (e.g. one per policy under test).
- `cliffs_delta(x, y)` -- a scale-free effect size, not a test.

Every function above that clusters or resamples enforces at least 2
distinct clusters, but that is a bare computability floor, not a claim
that inference from that few is reliable -- cluster-robust standard
errors and bootstrap/permutation confidence intervals are well known to
be fragile with only a handful of clusters, regardless of how many rows
each one has. For `fama_macbeth_difference`/`per_cluster_spearman`
specifically, the floor counts *contributing* clusters (both a treated
and a control row, and a finite per-cluster statistic) rather than raw
distinct labels in the input `cluster` column, which can be higher.

Use these instead of `bootstrap_returns`/`monte_carlo_permutation`
whenever the object under study is a panel with a fold/entity structure,
not a single stitched series -- and use `clustered_regression` specifically
when both fold and entity dependence need correcting at once, since it is
the only function here that supports two-way clustering. Every function
documents its own exact null hypothesis; none of them establish causality
-- they quantify whether an observed association is larger than
dependence-respecting resampling variation would typically produce, under
the single dependence structure the caller supplied, not every dependence
structure the panel might actually have. Each result-producing function
returns its own frozen dataclass (`ClusterDifferenceResult`,
`PermutationTestResult`, `FamaMacBethResult`, `ClusterSpearmanResult`,
`PairedSharpeDifferenceResult`), not a bare dict.

## Formation-then-test folds

`walk_forward_windows` always produces a train/validation/test triple.
`formation_test_windows(index, formation_window, test_window, step=None)`
instead produces formation-then-test folds with no validation block, for
a Gatev, Goetzmann & Rouwenhorst (2006, "Pairs Trading: Performance of a
Relative-Value Arbitrage Rule")-style design where quantities are
estimated once on the formation window and a fixed rule is then applied
out of sample -- a middle validation block would only eat into the test
window for no purpose. Each `FormationTestWindow` carries positional
slices (`formation`, `test`) alongside start/end timestamps
(`formation_start`/`formation_end`/`test_start`/`test_end`); `step`
defaults to `test_window` like `walk_forward_windows`'s own, but may
exceed it here -- each fold is evaluated independently rather than
stitched into one continuous curve, so a gap between test windows is a
legitimate sparser design, not a CAGR/annualisation hazard.

## Random-sign test

`monte_carlo_permutation` randomly flips the sign of per-period excess returns
around the configured risk-free rate, while preserving their magnitudes. Its
empirical p-value is the share of randomised Sharpes at least as high as the
observed Sharpe, with a finite-sample correction. A low value is evidence
against this specific random-sign null; it is not the probability that the
strategy is genuine, profitable or likely to work in the future.

## Stress tests

`run_stress_tests(data, config)` evaluates elevated commissions and
slippage, an extra execution-delay period, removal of the best days, and a
reduced tradable universe. Every scenario -- including one whose universe is
too small to leave at least 2 tradable symbols, or that fails for any other
reason -- keeps its own row in the table with `status="failed"` and an error
message, rather than being silently omitted or aborting the whole run.

Every scenario's magnitude comes from `robustness.stress_test` in YAML, each
a list so more than one magnitude can be evaluated per scenario type (e.g.
`execution_delays: [1, 2, 5]` adds a scenario row per delay); an empty list
disables that scenario type entirely. `commission_multipliers`/
`slippage_multipliers` must be strictly greater than 1.0 -- these model
elevated, adverse costs, not a cheaper-than-baseline scenario. The default
configuration evaluates:

```yaml
robustness:
  stress_test:
    enabled: true
    commission_multipliers: [2.0, 5.0]
    slippage_multipliers: [2.0]
    execution_delays: [1]
    best_days_removed: [10]
    reduce_universe_by: [1]  # symbols dropped from the tail of the universe
```

In walk-forward mode, a commission/slippage scenario reuses the baseline's
own cached per-fold candidate *weights* -- a parameter combination's
signal/allocation output never depends on costs, only its score does, so
recomputing weights from scratch would be wasted work -- and re-scores
every one of them under the new costs, re-selecting each fold's winner
from those new scores. Which candidate wins **can and does change** under
stressed costs; this is not a cost-invariant shortcut. The
execution-delay and reduced-universe scenarios genuinely change the
weights themselves, so each of those still re-runs selection end to end
instead of reusing the cache.

## What none of this proves

Positive holdout, walk-forward, sensitivity, bootstrap and stress-test results
support a more careful research process. They do not guarantee future
profitability. See [Limitations](limitations.md).
