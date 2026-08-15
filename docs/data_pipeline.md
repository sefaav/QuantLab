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

## Sources

- **Yahoo Finance** (`quantlab.data.yahoo.YahooFinanceDataSource`) — Yahoo
  tickers on either the XNYS or 24/7 calendar. Adjusted close is retained when
  Yahoo supplies it; a warning is logged when raw close must be used instead.
- **Binance** (`quantlab.data.binance.BinanceDataSource`) — crypto OHLCV,
  handles the 1000-candle pagination limit and 429 rate-limit backoff. No
  corporate actions, so `adjusted_close == close`.
- **CSV** (`source: csv` in a config) — reads `data/raw/<SYMBOL>.csv`, already
  in canonical schema. Used for fully offline experiments and tests.

## Cleaning

`DataCleaner` never silently fills every gap. The missing-value policy is
always explicit, from the config:

- `drop` — remove rows with any missing price.
- `forward_fill` — carry prices forward within each symbol, never backward,
  for at most `data.forward_fill_limit` consecutive bars (default: one).
  Any unresolved or newly OHLC-inconsistent row is dropped.
- `raise` — fail if any canonical field is missing, including volume,
  timestamp, or symbol.
- `none` — leave gaps for the caller to handle.

## Storage

`ParquetStorage` caches each `(source, symbol, frequency)` as one Parquet file
under `data/cache/` and reuses cache entries that pass coverage and internal-gap
checks. Remote gaps or forced refreshes are downloaded and merged.

`DataLoader.load()` is the entry point used by the CLI and dashboard:
download-or-cache → discard bars that have not settled → slice to the requested
range → inspect raw defects → clean → validate the final frame. Slicing before
forward filling prevents extra history in a wider cache from changing the
first requested observation.

## Universes

`quantlab.data.universe.Universe` provides convenience constructors
(`from_symbols`, `from_csv`, `crypto_major`, `us_sector_etfs`,
`liquid_multi_asset_etfs`). These reflect the **current** composition of an
index or sector set — they do not reconstruct historical membership, so
backtests over them carry a survivorship-bias caveat (see
[Limitations](limitations.md)).
