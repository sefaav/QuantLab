"""Acquire, cache, restrict, clean, and validate experiment market data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quantlab.config import BenchmarkKind, ExperimentConfig, MissingValuePolicy
from quantlab.constants import DEMO_DATA_DIR, RAW_DATA_DIR, SYMBOL, TIMESTAMP
from quantlab.data.base import MarketDataSource, ensure_canonical_schema
from quantlab.data.calendar import FREQUENCY_TIMEDELTA, bar_bucket_end
from quantlab.data.cleaner import DataCleaner
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

    def download(
        self, config: ExperimentConfig, *, force: bool = False
    ) -> pd.DataFrame:
        """Return raw canonical data for tradable and external benchmark symbols."""
        symbols = self._symbols_to_fetch(config)
        source_name = config.data.source
        if source_name == "csv":
            return self._load_csv(
                symbols, use_bundled_demo_data=config.data.use_bundled_demo_data
            )

        source = build_source(source_name)
        frames = [
            self._download_symbol(source, symbol, config, force=force)
            for symbol in symbols
        ]
        return ensure_canonical_schema(pd.concat(frames, ignore_index=True))

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

    def load(
        self, config: ExperimentConfig, *, force: bool = False
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """Return clean, validated data restricted to the configured dates."""
        raw = self.download(config, force=force)
        # Keep CSV and remote sources under the same settlement rule.
        raw = _drop_still_open_bars(
            raw, config.data.frequency, is_247_market=config.data.is_247_market
        )
        # Slice before forward-filling so a wider cache cannot affect a narrow run.
        sliced_raw = self._slice_range(
            raw,
            config.start_date,
            config.end_date,
            config.frequency,
            is_247_market=config.data.is_247_market,
        )

        validator = DataValidator(
            expected_frequency=config.data.frequency,
            is_247_market=config.data.is_247_market,
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
        report.raw_row_count = len(sliced_raw)
        report.clean_row_count = len(sliced)
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

    def _download_symbol(
        self,
        source: MarketDataSource,
        symbol: str,
        config: ExperimentConfig,
        *,
        force: bool,
    ) -> pd.DataFrame:
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
                is_247_market=config.data.is_247_market,
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
            is_247_market=config.data.is_247_market,
        )
        self.storage.write_symbol(
            downloaded,
            source.name,
            symbol,
            frequency,
            is_247_market=config.data.is_247_market,
            replace_start=start,
            replace_end=end,
        )
        persisted = self.storage.read_symbol(
            source.name,
            symbol,
            frequency,
            is_247_market=config.data.is_247_market,
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
        is_247_market: bool,
    ) -> pd.DataFrame:
        """Restrict raw timestamps and exclude bars settling after ``end``."""
        timestamps = pd.to_datetime(data[TIMESTAMP])
        end_boundary = pd.Timestamp(end) + pd.Timedelta(days=1)
        mask = (timestamps >= pd.Timestamp(start)) & (timestamps < end_boundary)
        sliced = data.loc[mask]
        if frequency in FREQUENCY_TIMEDELTA and not sliced.empty:
            bucket_ends = bar_bucket_end(
                pd.to_datetime(sliced[TIMESTAMP]),
                frequency,
                is_247_market=is_247_market,
            )
            sliced = sliced.loc[bucket_ends <= end_boundary]
        return sliced.reset_index(drop=True)
