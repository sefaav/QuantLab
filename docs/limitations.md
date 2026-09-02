# Limitations

QuantLab is a research and education platform. It is designed to make its own
limitations visible rather than hide them — every generated report includes
an automatically generated limitations section (see `research_summary.
STANDARD_LIMITATIONS`); this page provides the more complete, project-level
discussion. Read this before drawing conclusions from any result.

## Data

- **Adjusted prices** fold in dividends and splits retroactively; they are
  convenient for total-return backtests but were not exactly tradable as
  modelled at the time.
- **Survivorship bias**: universe helpers like `Universe.us_sector_etfs()` or
  `liquid_multi_asset_etfs()` reflect **current** composition. Backtests over
  them implicitly assume today's constituents existed throughout the period,
  which can overstate results.
- **Single venue** for crypto data (Binance): no consolidated tape.
- **Multi-calendar support is an approximation, not a live feed.** Each
  instrument declares its own source and calendar (any name recognised by
  `pandas_market_calendars`, or `24/7`); a mixed-calendar portfolio (e.g. US
  equities alongside crypto) is supported at daily frequency — closed
  sessions are detected per symbol, valued at the last known price with zero
  return and zero volume, and never traded.

  *Handled on each instrument's own native calendar*: every built-in
  strategy's own signal generation (momentum lookbacks, technical
  indicators, every mean-reversion indicator), a pairs-trading spread's
  hedge fit and indicator (computed on the intersection of both legs' own
  native session dates), and `runner.py`'s ADV computation — each symbol is
  sliced to its own verified native session rows before computing, then
  reindexed/forward-filled back onto the combined timeline
  (`quantlab.features.native_calendar.compute_native_then_align`), so a
  session-bound instrument sharing a timeline with an always-open one is
  never diluted by the always-open instrument's own extra sessions in what
  actually gets traded. Weekly/monthly bucket settlement and Yahoo's daily
  timestamps also resolve against each instrument's own calendar rather
  than a UTC-midnight approximation.

  *Still computed on the combined, closure-padded timeline, not yet
  native-calendar-aware*: the `inverse_volatility`/`volatility_targeting`
  portfolio allocators' realized-volatility estimate, so a mixed-calendar
  universe's allocator weights can still be diluted by the always-open
  instrument's own extra sessions; weekly/monthly **rebalancing** buckets
  against the raw UTC boundary whenever the portfolio's instruments do not
  all share one calendar; and the Strategy Explorer's Results-tab
  diagnostics, which recompute their own illustrative indicators
  independently of the live strategy's signal path for `pairs_trading`
  (hedge fit, spread, indicator, and rolling ADF p-value — the last calls
  the same `periodic_stationarity_pvalues` function the live entry gate
  uses, but on the combined timeline rather than the native intersection
  the live gate itself feeds it, so calling the same function does not by
  itself make the result match), `time_series_momentum`,
  `cross_sectional_momentum` and `trend_following`. `mean_reversion`'s own
  Results-tab diagnostic is the one exception, wired onto the same
  native-calendar path the live strategy uses. The Strategy Explorer's
  interactive labs additionally assume the XNYS calendar for any Yahoo or
  CSV symbol (not detected/configurable there the way the main dashboard's
  per-instrument table is), so a non-US instrument may not be represented
  faithfully in a lab.

  *Not yet supported at all*: an official intraday session break (e.g.
  XHKG's lunch recess) is handled for hourly cache-coverage/gap-detection
  checks but not modelled anywhere else; coverage has only been exercised
  against the calendars this project's own test suite covers (XNYS, XHKG,
  XASX, XSAU, 24/7), not every calendar `pandas_market_calendars`
  recognises; the calendar itself is a static, best-effort schedule
  (holidays, weekends), not a real-time venue-status feed, so an
  unscheduled closure (an exchange halt, an outage) is not detected as a
  closure and instead falls under the ordinary `missing_value_policy`
  handling for gaps; and intraday (`1h`) frequency does not support mixed
  calendars at all and is rejected at config load.

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

- **Weight drift between rebalances is modeled by default**
  (`portfolio.model_weight_drift`, `True` by default): each asset's own
  price move drifts its executed weight between real trades, rather than
  holding it constant until the next scheduled rebalance, with hard
  portfolio-level risk limits re-checked every row and restored via a
  linear-programming projection (never a "clip and scale" heuristic) if
  breached off-schedule — see [Weight drift](backtesting.md#weight-drift)
  and [the compliance-restoration LP](drift_compliance.md) for the full
  mechanism. `model_weight_drift=False` remains available as an optional
  constant-weight compatibility mode, not the recommended path.
  The compliance-restoration LP's own basis is gross/pre-cost, the same
  disclosed convention already used by `stop_loss_pct`/`take_profit_pct`.
- **Possible data-snooping**: trying many strategies, parameters or universes
  and reporting only the best one overstates expected performance. QuantLab's
  walk-forward and sensitivity tooling exists to mitigate this, but no
  tooling eliminates the risk entirely. The [validation guide](validation.md)
  explains what each robustness method can and cannot establish.
- **Historical results are not predictive**. Positive backtests, walk-forward,
  bootstrap and stress-test outcomes describe the past under stated
  assumptions; they do not guarantee future performance.
- **Checkpointing covers fold-level progress, not every stage**: the CLI's
  `walk-forward`, `stress-test` and `sensitivity` commands persist progress
  after each completed fold, scenario block or grid cell (never a partially
  computed one) so an interrupted run resumes instead of restarting from
  scratch, gated on the run's config/data/code/dependency provenance still
  matching. Checkpoint files are pickle and trusted-local-file-only (see
  `quantlab.validation.checkpoint`) — never point one at a file from an
  untrusted source. Downloaded market data remain separately reusable
  through the cache regardless.
- **Interface scope**: the dashboard's Backtest mode covers a plain backtest,
  chronological holdout, bootstrap, permutation test and sensitivity;
  Walk-forward mode covers walk-forward validation itself plus its own
  walk-forward-OOS-aware stress tests and sensitivity, and also exposes
  bootstrap and permutation test — mode-agnostic techniques that resample
  already-realised returns rather than re-running selection, so the same
  functions apply directly to the walk-forward OOS series. The Python API and
  notebooks expose the same underlying functions directly for anything the
  dashboard doesn't surface.

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
