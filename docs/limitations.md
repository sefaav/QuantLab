# Limitations

QuantLab is a research and education platform. It is designed to make its own
limitations visible rather than hide them — every generated report includes
this list automatically. Read this before drawing conclusions from any
result.

## Data

- **Adjusted prices** fold in dividends and splits retroactively; they are
  convenient for total-return backtests but were not exactly tradable as
  modelled at the time.
- **Survivorship bias**: universe helpers like `Universe.us_sector_etfs()` or
  `liquid_multi_asset_etfs()` reflect **current** composition. Backtests over
  them implicitly assume today's constituents existed throughout the period,
  which can overstate results.
- **Single venue** for crypto data (Binance): no consolidated tape.
- **One market calendar per experiment**: Yahoo experiments must choose either
  the XNYS approximation (for US-listed equities and ETFs) or the 24/7 calendar
  (for continuously traded instruments such as crypto). Binance uses 24/7.
  Futures, foreign exchanges and instruments with other trading schedules are
  not modelled accurately by either choice, and mixed-calendar universes are
  not supported.

## Execution

- **No real market impact** beyond a simplified slippage term (constant or
  square-root) — a large order does not actually move the book in the
  simulation the way it would in reality.
- **Simplified spread and slippage models**: commissions and spreads scale
  linearly with traded notional; optional volume impact uses a square-root
  participation-rate approximation rather than order-book dynamics.
- **No regulatory constraints**: no short-borrow limits, no margin calls, no
  position limits beyond what the config expresses.
- **No taxes**.
- **Liquidity is treated as effectively unlimited** relative to position size
  (aside from the optional volume-based slippage term).

## Methodology

- **Rebalancing is a step function**: weights are held constant between
  rebalance dates, ignoring intra-period drift from price moves — the
  standard simplification for a vectorised backtest.
- **Possible data-snooping**: trying many strategies, parameters or universes
  and reporting only the best one overstates expected performance. QuantLab's
  walk-forward and sensitivity tooling exists to mitigate this, but no
  tooling eliminates the risk entirely. The [validation guide](validation.md)
  explains what each robustness method can and cannot establish.
- **Historical results are not predictive**. Positive backtests, walk-forward,
  bootstrap and stress-test outcomes describe the past under stated
  assumptions; they do not guarantee future performance.
- **Long calculations are not checkpointed**: interrupting walk-forward,
  sensitivity, bootstrap, permutation or stress analysis requires restarting
  that calculation. Downloaded market data remain reusable through the cache.
- **Interface scope differs**: the dashboard currently exposes holdout and
  stress analysis, while walk-forward, sensitivity, bootstrap and permutation
  are available through the CLI, notebooks or Python API as documented in the
  validation guide.

## Reproducibility

Every saved run's `metadata.json` carries a content hash of the exact data
used, a best-effort Git commit hash (`null` if the working copy isn't a
detectable Git repository), and the installed versions of QuantLab and selected
numerical dependencies (pandas, numpy, pydantic, scipy and statsmodels). These
fields detect many relevant input, source and dependency differences, but they
do not describe the complete environment. `uv.lock` additionally pins every
transitive dependency. Running `uv sync --locked` restores that set and prevents
an unnoticed dependency upgrade from changing numerical results. This is still
not a full reproducibility manifest because the operating system, CPU and other
environment details are not captured.

Saved result bundles are replaced one file at a time using atomic file
replacement and carry an in-progress marker while being written. An interrupted
save is therefore detectable and individual files are not left half-written,
but the directory as a whole is not one cross-platform atomic transaction.
Concurrent saves to the same output directory are serialized by a persistent
sibling lock file; waiting for that lock fails with a clear error after 30
seconds. The in-progress marker still records interruption state and is not the
locking mechanism.

The Docker image uses the same `uv.lock` as CI, pins its base image by digest,
installs QuantLab non-editably in a multi-stage build and runs without root
privileges. Bind-mount permissions can still depend on the host UID/GID, and a
production deployment should continuously scan the final image.

## What would need to change for production use

This platform intentionally stops short of what a production trading system
requires: real-time data feeds, an event-driven (not vectorised) execution
model, broker integration, position reconciliation, regulatory compliance,
and operational monitoring. **QuantLab must not be used to manage real
capital as-is.**
