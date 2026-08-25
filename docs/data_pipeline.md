# Data pipeline

## Canonical schema

Every source is normalised to one long-format schema before entering the rest
of the pipeline:

| Column            | Type     | Constraint                          |
| ----------------- | -------- | ------------------------------------ |
| `timestamp`        | datetime | sorted per symbol                    |
| `symbol`          | string   | non-empty                            |
| `open, high, low, close, adjusted_close` | float | finite, strictly positive |
| `volume`          | float   | finite, non-negative                |

Timezone-aware timestamps are converted to UTC. Timezone-naive timestamps are
assumed to already represent UTC and are not shifted.

`quantlab.data.base.ensure_canonical_schema` enforces the column set;
`quantlab.data.validator.DataValidator` checks the constraints and produces a
`DataQualityReport` (duplicates, missing values, invalid prices, OHLC
consistency, coverage gaps).

## Instruments

`DataConfig.instruments` is a list of `InstrumentConfig` entries
(`symbol`/`source`/`calendar`), each fully explicit — no global source or
calendar for the whole experiment. A single portfolio can freely mix sources
and calendars, e.g. US equities from Yahoo (`XNYS`) alongside crypto from
Binance (`24/7`). `ExperimentConfig.benchmark` is itself an `InstrumentConfig`;
if its symbol duplicates a tradable instrument it must match that instrument's
source/calendar exactly and is never re-downloaded, and it never contaminates
the tradable universe's own timeline (an external 24/7 benchmark cannot inject
synthetic weekend bars into an all-equities portfolio).

## Sources

- **Yahoo Finance** (`quantlab.data.yahoo.YahooFinanceDataSource`) — Adjusted
  close is retained when Yahoo supplies it; a warning is logged when raw close
  must be used instead.
- **Binance** (`quantlab.data.binance.BinanceDataSource`) — crypto OHLCV,
  handles the 1000-candle pagination limit and 429 rate-limit backoff. No
  corporate actions, so `adjusted_close == close`. Always `calendar: "24/7"`.
- **CSV** (`source: csv` on an instrument) — reads `data/raw/<SYMBOL>.csv`,
  already in canonical schema. Used for fully offline experiments and tests;
  neutral with respect to frequency compatibility (its real frequency is only
  known after reading the file, checked by `DataValidator` post-load).

## Verified closures vs. missing data

A verified closure — a date that is a non-session day on an instrument's own
calendar (`quantlab.data.calendar.is_session_day`) — is distinguished from
genuinely missing data. `quantlab.data.closures.insert_verified_closure_bars`
fills a closed instrument with a flat synthetic bar (open/high/low/close and
adjusted_close each carried forward independently from the last known value,
volume forced to zero) whenever another instrument in the same tradable
universe has a real bar that day, so return is exactly zero and the instrument
is excluded from that day's rebalancing (see
[Backtesting](backtesting.md#rebalancing-turnover) for the tradability-aware execution
side). This is a no-op for a single-calendar experiment and for non-daily
frequencies. A gap that is *not* a calendar closure (e.g. a genuinely missing
trading day) is governed entirely by `missing_value_policy`, as below — but,
unlike a closure, it is never assumed to have zero return; the policy decides
whether it's rejected, dropped, or filled.

## Cleaning

The missing-value policy is always explicit, from the config, and applies at
two levels — a missing *value* inside an existing row, and a (date, symbol)
combination with *no row at all* for a real trading session are structurally
different problems, handled by different code with the same policy:

- `DataCleaner` (`quantlab.data.cleaner`) handles a missing value inside a row
  that exists.
- `DataLoader.load()` separately applies the identical policy to a genuine
  gap — a real session, on a symbol's own calendar, with no row whatsoever
  (`quantlab.data.loader._apply_missing_value_policy_to_genuine_gaps`),
  scoped to the tradable universe only (an external benchmark's own gaps are
  its own concern). Left ungoverned, such a gap would silently produce an
  incomplete panel — caught only later, confusingly, by the backtest engine's
  "asset return missing while held" error.

Both levels honour the same four policies:

- `drop` — remove rows with any missing price; for a genuine gap, remove the
  *entire date* from the tradable universe (every symbol, not just the one
  missing), so the resulting panel stays dense.
- `forward_fill` — carry prices forward within each symbol, never backward,
  for at most `data.forward_fill_limit` consecutive bars (default: one). For
  a genuine gap, a filled row is synthetic and flat (open/high/low/close and
  adjusted_close carried forward independently, volume forced to zero — the
  same shape as a verified-closure bar, but this is not one: it still counts
  toward the fill limit and is not exempt from rebalancing). Any unresolved
  or newly OHLC-inconsistent row, or a date where the fill limit is
  exceeded, is dropped the same way as under `drop`.
- `raise` — fail if any canonical field is missing, including volume,
  timestamp, or symbol; for a genuine gap, fail at load time naming the
  affected (date, symbol) pairs, rather than later inside the engine.
- `none` — leave gaps for the caller to handle, at both levels.

## Storage

`ParquetStorage` caches each `(source, symbol, frequency)` as one Parquet file
under `data/cache/` and reuses cache entries that pass coverage and internal-gap
checks. Remote gaps or forced refreshes are downloaded and merged.

`DataLoader.load()` is the entry point used by the CLI and dashboard:
download-or-cache → discard bars that have not settled → slice to the requested
range → inspect raw defects → clean → validate the final frame → insert
verified-closure bars → apply `missing_value_policy` to any remaining genuine
gap → trim to the tradable universe's common start/end coverage. Slicing
before forward filling prevents extra history in a wider cache from changing
the first requested observation.

## Universes

`quantlab.data.universe.Universe` provides convenience constructors
(`from_symbols`, `from_csv`, `crypto_major`, `us_sector_etfs`,
`liquid_multi_asset_etfs`). These reflect the **current** composition of an
index or sector set — they do not reconstruct historical membership, so
backtests over them carry a survivorship-bias caveat (see
[Limitations](limitations.md)).
