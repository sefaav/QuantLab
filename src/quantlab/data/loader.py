"""Acquire, cache, restrict, clean, and validate experiment market data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.config import (
    BenchmarkKind,
    DataSourceName,
    ExperimentConfig,
    InstrumentConfig,
    MissingValuePolicy,
)
from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    DEMO_DATA_DIR,
    HIGH,
    LOW,
    OPEN,
    RAW_DATA_DIR,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.data.base import MarketDataSource, ensure_canonical_schema, pivot_field
from quantlab.data.calendar import FREQUENCY_TIMEDELTA, bar_bucket_end, is_247, sessions
from quantlab.data.cleaner import DataCleaner
from quantlab.data.closures import (
    DAILY_FREQUENCY,
    insert_verified_closure_bars,
    tradable_mask_for,
)
from quantlab.data.storage import ParquetStorage, _drop_still_open_bars
from quantlab.data.validator import DataQualityReport, DataValidator
from quantlab.exceptions import DataDownloadError, DataValidationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def build_source(name: str) -> MarketDataSource:
    """Instantiate a supported remote market-data source."""
    key = name.lower().strip()
    if key == "yahoo":
        from quantlab.data.yahoo import YahooFinanceDataSource

        return YahooFinanceDataSource()
    if key == "binance":
        from quantlab.data.binance import BinanceDataSource

        return BinanceDataSource()
    raise DataDownloadError(
        f"Unknown remote data source '{name}'. Known remote sources: yahoo, binance. "
        "CSV files are handled directly by DataLoader."
    )


def _apply_missing_value_policy_to_genuine_gaps(
    data: pd.DataFrame,
    *,
    symbol_calendars: Mapping[str, str],
    frequency: str,
    policy: MissingValuePolicy,
    forward_fill_limit: int,
    warnings: list[str],
) -> pd.DataFrame:
    """Govern a (date, symbol) combination with no row at all, by policy.

    ``missing_value_policy`` (applied by :class:`~quantlab.data.cleaner.
    DataCleaner`) only ever sees rows that already exist -- it can drop or
    fill a NaN *value* inside a row, but it has no way to notice that a row
    is entirely absent for a symbol on a date it should have traded. Left
    unhandled, that silently produces an incomplete panel (only caught much
    later, confusingly, by the engine's "asset return missing while held"
    guard). This runs after :func:`~quantlab.data.closures.
    insert_verified_closure_bars`, so a verified closure (a real non-session
    day) is never mistaken for a gap -- only a real trading session with no
    data counts here.

    A no-op when ``frequency`` isn't daily (closure/gap semantics are only
    well-defined at daily granularity, see ``quantlab.data.closures``), or
    under policy ``none`` (gaps stay exactly as they already are, governed
    by the existing "abnormal gap" warning).
    """
    if frequency != DAILY_FREQUENCY or data.empty or policy is MissingValuePolicy.NONE:
        return data
    symbols = sorted(symbol_calendars)
    dates = pd.DatetimeIndex(sorted(data[TIMESTAMP].unique()))
    # A date nothing in `data` trades on is never invented -- `dates` is
    # drawn only from what's actually present, same convention as
    # insert_verified_closure_bars.
    tradable = tradable_mask_for(dates, symbols, symbol_calendars)
    close_wide = pivot_field(data, CLOSE).reindex(index=dates, columns=symbols)
    # A symbol's coverage simply not having started yet, or having already
    # ended, is a *coverage-window* difference between symbols -- handled
    # separately by DataLoader.load()'s common-start/common-end trimming,
    # with its own clear warning. Only a hole strictly between a symbol's
    # own first and last observed dates is a genuine gap this function
    # should govern; leading/trailing absence relative to the symbol's own
    # history must never be double-handled here first.
    observed = close_wide.notna()
    within_own_range = observed.cummax() & observed[::-1].cummax()[::-1]
    missing = close_wide.isna() & tradable & within_own_range
    if not missing.to_numpy().any():
        return data

    if policy is MissingValuePolicy.RAISE:
        rows, cols = np.where(missing.to_numpy())
        examples = ", ".join(
            f"{symbols[int(c)]}@{dates[int(r)].date()}"
            for r, c in list(zip(rows, cols, strict=True))[:5]
        )
        raise DataValidationError(
            f"{len(rows)} (date, symbol) combination(s) are genuinely "
            f"missing under missing_value_policy 'raise' (e.g. {examples}) "
            "-- no row exists for a real trading session on that symbol's "
            "own calendar (not a verified closure)."
        )

    if policy is MissingValuePolicy.DROP:
        affected_dates = dates[missing.to_numpy().any(axis=1)]
        warnings.append(
            f"{len(affected_dates)} date(s) dropped from the tradable "
            "universe: at least one symbol was genuinely missing that day "
            "under missing_value_policy 'drop' (e.g. "
            f"{[d.date().isoformat() for d in affected_dates[:5]]})."
        )
        return data.loc[~data[TIMESTAMP].isin(affected_dates)].reset_index(drop=True)

    # forward_fill: fill from each symbol's own last known price, bounded by
    # forward_fill_limit; a date where even one symbol exceeds the limit is
    # dropped entirely (same as `drop` above) rather than left partially
    # filled, which would risk the same downstream "missing while held"
    # failure this function exists to prevent.
    filled_close = close_wide.ffill(limit=forward_fill_limit)
    adjusted_wide = pivot_field(data, ADJUSTED_CLOSE).reindex(
        index=dates, columns=symbols
    )
    filled_adjusted = adjusted_wide.ffill(limit=forward_fill_limit)
    unresolved = missing & filled_close.isna()
    if unresolved.to_numpy().any():
        affected_dates = dates[unresolved.to_numpy().any(axis=1)]
        warnings.append(
            f"{len(affected_dates)} date(s) dropped from the tradable "
            "universe: a genuinely missing (date, symbol) combination "
            f"exceeded the {forward_fill_limit}-bar forward-fill limit "
            f"(e.g. {[d.date().isoformat() for d in affected_dates[:5]]})."
        )
        data = data.loc[~data[TIMESTAMP].isin(affected_dates)].reset_index(drop=True)
        keep = ~dates.isin(affected_dates)
        dates = dates[keep]
        missing = missing.loc[dates]
        filled_close = filled_close.loc[dates]
        filled_adjusted = filled_adjusted.loc[dates]

    fillable = missing & filled_close.notna()
    if not fillable.to_numpy().any():
        return data
    rows, cols = np.where(fillable.to_numpy())
    fill_dates = dates[rows]
    fill_symbols = [symbols[c] for c in cols]
    close_values = filled_close.to_numpy()[rows, cols]
    adjusted_values = filled_adjusted.to_numpy()[rows, cols]
    synthetic = pd.DataFrame(
        {
            TIMESTAMP: fill_dates,
            SYMBOL: fill_symbols,
            OPEN: close_values,
            HIGH: close_values,
            LOW: close_values,
            CLOSE: close_values,
            ADJUSTED_CLOSE: adjusted_values,
            VOLUME: 0.0,
        }
    )
    warnings.append(
        f"{len(rows)} row(s) forward-filled for genuinely missing (date, "
        "symbol) combinations (not a verified closure)."
    )
    combined = pd.concat([data, synthetic], ignore_index=True)
    return combined.sort_values([TIMESTAMP, SYMBOL]).reset_index(drop=True)


class DataLoader:
    """Load clean, validated canonical market data for an experiment."""

    def __init__(
        self,
        storage: ParquetStorage | None = None,
        raw_dir: Path | None = None,
    ) -> None:
        self.storage = storage if storage is not None else ParquetStorage()
        # Resolve the default when constructing the loader so tests can patch it.
        self.raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
        self._bundled_demo_data_used = False

    def download(
        self, config: ExperimentConfig, *, force: bool = False
    ) -> pd.DataFrame:
        """Return raw canonical data for tradable and external benchmark instruments."""
        # Reset in case this instance is reused across multiple load() calls
        # -- a stale True from an earlier call must never leak into this one.
        self._bundled_demo_data_used = False
        instruments = list(config.data.instruments)
        external_benchmark = self._external_benchmark_instrument(config)
        if external_benchmark is not None:
            instruments.append(external_benchmark)
        return self._download_group(instruments, config, force=force)

    @staticmethod
    def _external_benchmark_instrument(
        config: ExperimentConfig,
    ) -> InstrumentConfig | None:
        """Return the benchmark instrument only when outside the tradable universe.

        A benchmark symbol already present in ``data.instruments`` reuses that
        instrument's data — a validator guarantees source/calendar agree in
        that case, so no separate fetch is needed.
        """
        benchmark = config.backtest.benchmark
        if benchmark is None or config.benchmark_kind is not BenchmarkKind.SYMBOL:
            return None
        if benchmark.symbol in config.symbols:
            return None
        return benchmark

    def _download_group(
        self,
        instruments: list[InstrumentConfig],
        config: ExperimentConfig,
        *,
        force: bool,
    ) -> pd.DataFrame:
        """Fetch every instrument, then apply calendar-dependent preparation.

        Calendar-dependent filtering (still-open bars, date-range slicing)
        runs only *after* every instrument's frame is assembled from cache or
        the provider — never baked into what gets cached — so the same
        provider history is never duplicated in the cache just because one
        experiment picked a different calendar than another for the same
        symbol/source/frequency.
        """
        by_source: dict[DataSourceName, list[InstrumentConfig]] = defaultdict(list)
        for instrument in instruments:
            by_source[instrument.source].append(instrument)

        raw_frames: list[pd.DataFrame] = []
        csv_instruments = by_source.pop(DataSourceName.CSV, [])
        if csv_instruments:
            raw_frames.append(
                self._load_csv(
                    [instrument.symbol for instrument in csv_instruments],
                    use_bundled_demo_data=config.data.use_bundled_demo_data,
                )
            )
        for source_name, group in by_source.items():
            source = build_source(source_name)
            raw_frames.extend(
                self._download_symbol(source, instrument, config, force=force)
                for instrument in group
            )
        raw = pd.concat(raw_frames, ignore_index=True)

        prepared_frames = [
            self._prepare_instrument_frame(
                raw.loc[raw[SYMBOL] == instrument.symbol], instrument, config
            )
            for instrument in instruments
        ]
        return ensure_canonical_schema(pd.concat(prepared_frames, ignore_index=True))

    @staticmethod
    def _prepare_instrument_frame(
        frame: pd.DataFrame, instrument: InstrumentConfig, config: ExperimentConfig
    ) -> pd.DataFrame:
        """Calendar-dependent filtering for one instrument.

        Applied after any cache read/write, never persisted — the cache stays
        calendar-agnostic (see :meth:`_download_group`).
        """
        frame = _drop_still_open_bars(
            frame, config.data.frequency, calendar=instrument.calendar
        )
        return DataLoader._slice_range(
            frame,
            config.start_date,
            config.end_date,
            config.frequency,
            calendar=instrument.calendar,
        )

    def load(
        self, config: ExperimentConfig, *, force: bool = False
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """Return clean, validated data restricted to the configured dates."""
        sliced_raw = self.download(config, force=force)

        symbol_calendars = {
            instrument.symbol: instrument.calendar
            for instrument in config.data.instruments
        }
        external_benchmark = self._external_benchmark_instrument(config)
        if external_benchmark is not None:
            symbol_calendars[external_benchmark.symbol] = external_benchmark.calendar

        validator = DataValidator(
            expected_frequency=config.data.frequency,
            symbol_calendars=symbol_calendars,
        )
        strict = config.data.missing_value_policy is MissingValuePolicy.RAISE
        # Record defects before deterministic cleaning removes them.
        pre_clean_report = validator.check_pre_clean_defects(sliced_raw, strict=strict)
        cleaner = DataCleaner(
            config.data.missing_value_policy,
            forward_fill_limit=config.data.forward_fill_limit,
        )
        sliced = cleaner.clean(sliced_raw)

        expected_symbols = self._symbols_to_fetch(config)
        report = validator.validate(
            sliced,
            start=config.start_date,
            end=config.end_date,
            strict=strict,
            expected_symbols=expected_symbols,
        )

        # Verified-closure bars are inserted only for the TRADABLE universe: an
        # external benchmark on a different calendar must never inflate the
        # portfolio's own timeline (see quantlab.data.closures).
        tradable_symbols = set(config.symbols)
        is_tradable = sliced[SYMBOL].isin(tradable_symbols)
        tradable_part = sliced.loc[is_tradable]
        benchmark_part = sliced.loc[~is_tradable]
        tradable_calendars = {
            instrument.symbol: instrument.calendar
            for instrument in config.data.instruments
        }
        closure_warnings: list[str] = []
        closure_counts = {"discarded": 0, "inserted": 0}
        filled_tradable = insert_verified_closure_bars(
            tradable_part,
            symbol_calendars=tradable_calendars,
            frequency=config.data.frequency,
            strict=strict,
            warnings=closure_warnings,
            counts=closure_counts,
        )
        report.warnings.extend(closure_warnings)
        report.closure_discarded_count = closure_counts["discarded"]
        report.closure_inserted_count = closure_counts["inserted"]

        # A real trading session with no row at all for a symbol -- a
        # genuine gap, never a verified closure (already handled above) --
        # must be explicitly governed by missing_value_policy, the same
        # guarantee documented for a missing *value* inside an existing
        # row. Scoped to the tradable universe only, same reasoning as
        # closure-fill: an external benchmark's own gaps are its own
        # concern, never forced onto the portfolio's tradable timeline.
        gap_warnings: list[str] = []
        filled_tradable = _apply_missing_value_policy_to_genuine_gaps(
            filled_tradable,
            symbol_calendars=tradable_calendars,
            frequency=config.data.frequency,
            policy=config.data.missing_value_policy,
            forward_fill_limit=config.data.forward_fill_limit,
            warnings=gap_warnings,
        )
        report.warnings.extend(gap_warnings)

        sliced = (
            pd.concat([filled_tradable, benchmark_part], ignore_index=True)
            .sort_values([TIMESTAMP, SYMBOL])
            .reset_index(drop=True)
        )

        # A mixed-calendar tradable universe's combined timeline (union of
        # every instrument's own sessions) can start earlier than a
        # session-bound instrument's actual first observation -- e.g. a 24/7
        # instrument already has a bar on a date a Yahoo/CSV equity simply
        # has no data for yet, which closure-fill correctly leaves alone
        # (never extrapolates before a symbol's first observed bar, see
        # quantlab.data.closures). Left unhandled, that produces an
        # unrecoverable NaN on the equity's own genuinely-first trading day
        # once price_matrix pivots to the union grid and pct_change cascades
        # -- caught only downstream, confusingly, by the benchmark-alignment
        # or missing-return-while-held guards. Trim the whole panel (both
        # tradable and any external-benchmark rows, which are unused before
        # this point anyway) to the date every tradable symbol has real
        # coverage from, so every downstream consumer starts from a panel
        # where each tradable symbol genuinely has a price on its first row.
        tradable_now = sliced.loc[sliced[SYMBOL].isin(tradable_symbols)]
        first_by_symbol = tradable_now.groupby(SYMBOL)[TIMESTAMP].min()
        if len(first_by_symbol) and first_by_symbol.max() > first_by_symbol.min():
            common_start = first_by_symbol.max()
            # The symbol(s) that push the common start this late (their own
            # first observation *is* common_start) -- not the ones losing
            # rows, which is every symbol whose coverage began earlier.
            limiting_symbols = sorted(
                first_by_symbol.index[first_by_symbol == common_start]
            )
            before = len(sliced)
            sliced = sliced.loc[sliced[TIMESTAMP] >= common_start].reset_index(
                drop=True
            )
            report.warnings.append(
                f"Tradable universe coverage effectively starts "
                f"{common_start.date()}, later than the requested start -- "
                f"{limiting_symbols} have no data before then (dropped "
                f"{before - len(sliced)} earlier row(s) from other symbols)."
            )

        # Symmetric case at the other end: one tradable symbol's data simply
        # stops earlier than another's (e.g. a stale feed), while the
        # combined union timeline (built from every instrument's own
        # sessions) keeps going. Those trailing dates aren't verified
        # closures for the stopped symbol -- they're real sessions on its own
        # calendar with no data at all -- so closure-fill correctly leaves
        # them alone, and an unheld/unrebalanced position would otherwise hit
        # an unrecoverable "asset return missing while held" failure deep in
        # the engine instead of a clear, load-time explanation. Trim the
        # whole panel the same way the start side already does.
        tradable_now = sliced.loc[sliced[SYMBOL].isin(tradable_symbols)]
        last_by_symbol = tradable_now.groupby(SYMBOL)[TIMESTAMP].max()
        if len(last_by_symbol) and last_by_symbol.min() < last_by_symbol.max():
            common_end = last_by_symbol.min()
            limiting_symbols = sorted(
                last_by_symbol.index[last_by_symbol == common_end]
            )
            before = len(sliced)
            sliced = sliced.loc[sliced[TIMESTAMP] <= common_end].reset_index(drop=True)
            report.warnings.append(
                f"Tradable universe coverage effectively ends "
                f"{common_end.date()}, earlier than the requested end -- "
                f"{limiting_symbols} have no data after then (dropped "
                f"{before - len(sliced)} later row(s) from other symbols)."
            )

        report.raw_row_count = len(sliced_raw)
        report.clean_row_count = len(sliced)
        report.bundled_demo_data_used = self._bundled_demo_data_used
        # row_count was set inside validate(), before closure-fill (which can
        # now both insert and discard rows, see quantlab.data.closures) and
        # the start/end coverage trims above ever ran -- refresh it so it
        # reflects the data this call actually returns, the same as
        # clean_row_count just above.
        report.row_count = len(sliced)
        present_symbols = set(sliced[SYMBOL].unique())
        missing_symbols = sorted(set(expected_symbols) - present_symbols)
        if missing_symbols:
            raise DataValidationError(
                f"Requested symbol(s) {missing_symbols} have zero usable rows after "
                "loading and cleaning; refusing to run on a reduced universe."
            )

        report.duplicate_count += pre_clean_report.duplicate_count
        report.invalid_price_count += pre_clean_report.invalid_price_count
        # ``none`` recounts the original gaps; max avoids doubling them.
        for column, count in pre_clean_report.missing_value_count.items():
            report.missing_value_count[column] = max(
                report.missing_value_count.get(column, 0), count
            )
        pre_clean_warnings = [
            warning
            for warning in pre_clean_report.warnings
            if warning not in report.warnings
        ]
        report.warnings = pre_clean_warnings + report.warnings
        logger.info(
            "Loaded %d rows for %d symbols (%s).",
            len(sliced),
            sliced[SYMBOL].nunique(),
            report.summary(),
        )
        return sliced, report

    @staticmethod
    def _symbols_to_fetch(config: ExperimentConfig) -> list[str]:
        """Return tradable symbols plus a separate symbol benchmark."""
        symbols = list(config.symbols)
        benchmark = (
            config.benchmark_symbol
            if config.benchmark_kind is BenchmarkKind.SYMBOL
            else None
        )
        if benchmark and benchmark not in symbols:
            symbols.append(benchmark)
        return symbols

    def _download_symbol(
        self,
        source: MarketDataSource,
        instrument: InstrumentConfig,
        config: ExperimentConfig,
        *,
        force: bool,
    ) -> pd.DataFrame:
        """Cache-aware fetch for one instrument.

        ``read_covered_symbol``/``read_symbol`` already mask any still-open
        bar from the returned view (calendar-dependent, but never rewrites
        the cache file -- see :meth:`~quantlab.data.storage.ParquetStorage.
        read_symbol`'s own docstring). What this method does *not* apply is
        the date-range restriction (``_slice_range``): that full
        "prepare" pass runs afterward, in :meth:`_prepare_instrument_frame`,
        which is what the "applied only after any cache read/write" claim
        there actually refers to.
        """
        symbol = instrument.symbol
        frequency = config.frequency
        start, end = config.start_date, config.end_date
        cached = None
        if not force:
            cached = self.storage.read_covered_symbol(
                source.name,
                symbol,
                frequency,
                start,
                end,
                calendar=instrument.calendar,
            )
        if cached is not None:
            logger.info("Cache hit for %s (%s).", symbol, source.name)
            return cached

        logger.info("Cache miss for %s; downloading.", symbol)
        downloaded = source.download(
            [symbol],
            start,
            end,
            frequency,
            calendar=instrument.calendar,
        )
        self.storage.write_symbol(
            downloaded,
            source.name,
            symbol,
            frequency,
            calendar=instrument.calendar,
            replace_start=start,
            replace_end=end,
        )
        persisted = self.storage.read_symbol(
            source.name,
            symbol,
            frequency,
            calendar=instrument.calendar,
        )
        if persisted is None:
            raise DataDownloadError(
                f"Downloaded {symbol}, but its merged cache could not be read back."
            )
        return persisted

    def _load_csv(
        self, symbols: list[str], *, use_bundled_demo_data: bool = False
    ) -> pd.DataFrame:
        """Load either a complete local CSV set or the complete demo set."""
        local_paths = [self.raw_dir / f"{symbol}.csv" for symbol in symbols]
        present_local = [path.is_file() for path in local_paths]
        if all(present_local):
            paths = local_paths
        elif any(present_local):
            missing = [
                str(path)
                for path, present in zip(local_paths, present_local, strict=True)
                if not present
            ]
            raise DataDownloadError(
                "CSV source is only partially available; refusing to mix local and "
                f"bundled synthetic data. Missing local files: {missing}."
            )
        elif use_bundled_demo_data:
            paths = [DEMO_DATA_DIR / f"{symbol}.csv" for symbol in symbols]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise DataDownloadError(
                    f"Bundled demo data is incomplete; missing files: {missing}."
                )
            self._bundled_demo_data_used = True
        else:
            raise DataDownloadError(
                "CSV source files were not found. Expected: "
                f"{[str(path) for path in local_paths]}."
            )

        frames: list[pd.DataFrame] = []
        for path in paths:
            try:
                frames.append(pd.read_csv(path, parse_dates=[TIMESTAMP]))
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise DataDownloadError(
                    f"Failed to read CSV file {path}: {exc}"
                ) from exc
        return ensure_canonical_schema(pd.concat(frames, ignore_index=True))

    @staticmethod
    def _slice_range(
        data: pd.DataFrame,
        start: date,
        end: date,
        frequency: str,
        *,
        calendar: str,
    ) -> pd.DataFrame:
        """Restrict raw timestamps and exclude bars settling after ``end``."""
        timestamps = pd.to_datetime(data[TIMESTAMP])
        end_boundary = pd.Timestamp(end) + pd.Timedelta(days=1)
        # A naive UTC midnight boundary would drop a session's genuine
        # early bars for a calendar whose local session opens before UTC
        # midnight of its own label date (e.g. XASX, UTC+10/+11 -- its
        # session dated `start` can open the previous UTC calendar day).
        # Only relevant when `start` is itself a real session for this
        # calendar; otherwise the naive boundary is already correct (the
        # next real session, whenever it falls, has no reason to start
        # before it).
        start_boundary = pd.Timestamp(start)
        if not is_247(calendar):
            start_schedule = sessions(calendar, start_boundary, start_boundary)
            if not start_schedule.empty:
                start_boundary = min(
                    start_boundary, pd.Timestamp(start_schedule.iloc[0]["market_open"])
                )
        mask = (timestamps >= start_boundary) & (timestamps < end_boundary)
        sliced = data.loc[mask]
        if frequency in FREQUENCY_TIMEDELTA and not sliced.empty:
            bucket_ends = bar_bucket_end(
                pd.to_datetime(sliced[TIMESTAMP]),
                frequency,
                calendar=calendar,
            )
            sliced = sliced.loc[bucket_ends <= end_boundary]
        return sliced.reset_index(drop=True)
