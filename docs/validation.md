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
validation_window, test_window, expanding=True)` performs these steps:

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

## Random-sign test

`monte_carlo_permutation` randomly flips the sign of per-period excess returns
around the configured risk-free rate, while preserving their magnitudes. Its
empirical p-value is the share of randomised Sharpes at least as high as the
observed Sharpe, with a finite-sample correction. A low value is evidence
against this specific random-sign null; it is not the probability that the
strategy is genuine, profitable or likely to work in the future.

## Stress tests

`run_stress_tests(data, config)` evaluates higher commissions and slippage, an
extra execution-delay period, removal of the ten best days, and—when the
universe is large enough—a reduced tradable universe. Expected scenario
failures are retained in the table instead of being silently omitted.

## What none of this proves

Positive holdout, walk-forward, sensitivity, bootstrap and stress-test results
support a more careful research process. They do not guarantee future
profitability. See [Limitations](limitations.md).
