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

`WalkForwardValidator.run(data, parameter_grid, train_window,
validation_window, test_window, expanding=True, step=None)` performs these
steps:

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

`run_parameter_sensitivity(data, config, parameter_x, values_x, parameter_y,
values_y)` records Sharpe, CAGR, drawdown, turnover and trade count for every
combination. Failed configurations remain visible with a status and error
message. The useful pattern is a region with comparable behaviour, not merely
one isolated optimum.

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

## What none of this proves

Positive holdout, walk-forward, sensitivity, bootstrap and stress-test results
support a more careful research process. They do not guarantee future
profitability. See [Limitations](limitations.md).
