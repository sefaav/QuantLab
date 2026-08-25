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
- **Per-instrument calendars are an approximation, not a live feed**: each
  instrument declares its own source and calendar (any name recognised by
  `pandas_market_calendars`, or `24/7`), and a mixed-calendar portfolio (e.g.
  US equities alongside crypto) is supported at daily frequency — closed
  sessions are detected per symbol, valued at the last known price with zero
  return and zero volume, and never traded. Weekly/monthly bucket settlement
  and Yahoo's daily timestamps resolve against each instrument's own
  calendar rather than a UTC-midnight approximation — but weekly/monthly
  **rebalancing** only does so when every instrument in the portfolio shares
  one calendar; a genuinely mixed-calendar portfolio (e.g. equities
  alongside crypto) still buckets rebalance dates against the raw UTC
  boundary, the same documented approximation used elsewhere for a
  mixed-calendar universe. This coverage is deliberately narrower than "fully
  supported" for two reasons: it does not extend to every calendar detail
  (e.g. an official intraday session break, like XHKG's lunch recess, is
  handled for hourly cache-coverage and gap-detection checks but not modelled
  anywhere else), and it has not been exercised against every calendar
  `pandas_market_calendars` recognises — only the ones this project's own
  test suite covers (XNYS, XHKG, XASX, XSAU, 24/7). But the calendar is
  still a static,
  best-effort schedule (holidays, weekends), not a real-time venue-status
  feed, so an unscheduled closure (an exchange halt, an outage) is not
  detected as a closure and instead falls under the ordinary
  `missing_value_policy` handling for gaps. Intraday (`1h`) frequency does
  not support mixed calendars at all yet and is rejected at config load —
  verified-closure handling only operates at daily frequency.
- **Rolling-window features are diluted in a mixed-calendar universe**: a
  verified closure's synthetic bar is exactly flat (zero return, zero
  volume), but momentum lookbacks, volatility windows, ADV windows and
  technical indicators all still count it as one more *period* — for a
  session-bound instrument sharing a combined timeline with an always-open
  one (e.g. equities alongside crypto), a "252-period" window therefore spans
  *more* than 252 real trading sessions, and the flat bars pull volatility/ADV
  estimates down. QuantLab warns about this at config load
  (`DataConfig._warn_if_mixed_calendars_dilute_windowed_features`) but does
  not correct it: doing so properly would mean computing every instrument's
  features on its own native calendar before aligning signals, a
  substantially larger redesign than today's shared-timeline architecture.
  Prefer a single shared calendar per experiment when window-based estimates
  need to be precise; treat mixed-calendar results as directionally
  informative rather than exact until this is addressed.

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
