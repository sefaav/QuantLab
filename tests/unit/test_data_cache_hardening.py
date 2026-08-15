"""Regression tests for loader, resampler, and cache hardening."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quantlab.data.storage as storage_module
from quantlab.config import ExperimentConfig
from quantlab.constants import OHLCV_COLUMNS
from quantlab.data.base import MarketDataSource
from quantlab.data.loader import DataLoader, build_source
from quantlab.data.resampler import resample_ohlcv
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import DataDownloadError, DataValidationError


def _ohlcv(
    timestamps: pd.DatetimeIndex | list[pd.Timestamp] | list[str],
    *,
    symbol: str = "AAA",
    volume: float = 100.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": volume,
        }
    )


def test_hourly_247_cache_requires_safely_closed_hours_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = pd.Timestamp("2026-08-08 15:40:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: now)
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")

    yesterday = pd.date_range("2026-08-07", periods=24, freq="h")
    storage.write_symbol(
        _ohlcv(yesterday, symbol="BTCUSDT"),
        "binance",
        "BTCUSDT",
        "1h",
        is_247_market=True,
    )
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2026, 8, 7),
        date(2026, 8, 8),
        is_247_market=True,
    )

    safely_closed_today = pd.date_range("2026-08-08", periods=15, freq="h")
    storage.write_symbol(
        _ohlcv(safely_closed_today, symbol="BTCUSDT"),
        "binance",
        "BTCUSDT",
        "1h",
        is_247_market=True,
    )
    assert storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2026, 8, 7),
        date(2026, 8, 8),
        is_247_market=True,
    )


def test_equity_hourly_cache_requires_the_final_requested_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2026-01-20 12:00:00")
    )
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    days = pd.date_range("2024-01-02", "2024-01-05", freq="B")
    hours = [day + pd.Timedelta(hours=hour) for day in days for hour in range(9, 16)]
    storage.write_symbol(_ohlcv(hours), "yahoo", "AAA", "1h")

    assert not storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 8)
    )

    monday = [pd.Timestamp("2024-01-08") + pd.Timedelta(hours=h) for h in range(9, 16)]
    storage.write_symbol(_ohlcv(monday), "yahoo", "AAA", "1h")
    assert storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 8)
    )


def test_resampler_rejects_a_finer_target() -> None:
    daily = _ohlcv(pd.date_range("2024-01-01", periods=3, freq="D"))
    with pytest.raises(DataValidationError, match="finer"):
        resample_ohlcv(daily, "1h")


def test_resampler_preserves_unknown_volume_and_canonical_schema() -> None:
    daily = _ohlcv(
        pd.date_range("2024-01-01", periods=5, freq="D"), volume=float("nan")
    )
    weekly = resample_ohlcv(daily, "1w", source_frequency="1d")
    assert list(weekly.columns) == list(OHLCV_COLUMNS)
    assert weekly["volume"].isna().all()
    assert weekly["close"].notna().all()


def test_resampler_returns_a_canonical_empty_frame() -> None:
    empty = pd.DataFrame(columns=OHLCV_COLUMNS)
    result = resample_ohlcv(empty, "1w", source_frequency="1d")
    assert result.empty
    assert list(result.columns) == list(OHLCV_COLUMNS)


def test_resampler_rejects_duplicates_and_unknown_frequencies() -> None:
    frame = _ohlcv(["2024-01-01", "2024-01-01"])
    with pytest.raises(DataValidationError, match="duplicate"):
        resample_ohlcv(frame, "1w", source_frequency="1d")
    with pytest.raises(DataValidationError, match="Unsupported resampling"):
        resample_ohlcv(frame.iloc[[0]], "2h", source_frequency="1h")


class _PartialSource(MarketDataSource):
    name = "partial"

    def download(
        self,
        symbols: list[str],
        start: date,
        end: date,
        frequency: str,
        *,
        is_247_market: bool = False,
    ) -> pd.DataFrame:
        del symbols, start, end, frequency, is_247_market
        return _ohlcv(["2024-01-03"])


def test_forced_download_returns_the_persisted_merged_frame(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    storage.write_symbol(_ohlcv(["2023-12-31", "2024-01-01"]), "partial", "AAA", "1d")
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "forced_consistency",
            "data": {
                "source": "csv",
                "symbols": ["AAA"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "market_calendar": "24/7",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(storage=storage)
    returned = loader._download_symbol(_PartialSource(), "AAA", config, force=True)
    persisted = storage.read_symbol("partial", "AAA", "1d", is_247_market=True)
    assert persisted is not None
    pd.testing.assert_frame_equal(returned, persisted)
    assert returned["timestamp"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-12-31",
        "2024-01-03",
    ]


def test_storage_deduplicates_the_first_write_and_validates_its_symbol(
    tmp_path: Path,
) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    duplicate = _ohlcv(["2024-01-01", "2024-01-01"])
    storage.write_symbol(duplicate, "yahoo", "AAA", "1d")
    cached = storage.read_symbol("yahoo", "AAA", "1d")
    assert cached is not None
    assert len(cached) == 1

    with pytest.raises(DataValidationError, match="rows for"):
        storage.write_symbol(_ohlcv(["2024-01-02"], symbol="BBB"), "yahoo", "AAA", "1d")


def test_storage_uses_a_versioned_cache_namespace(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    path = storage.write_symbol(_ohlcv(["2024-01-01"]), "yahoo", "AAA", "1d")
    assert path.relative_to(tmp_path / "cache").parts[0] == "v2"


def test_atomic_save_keeps_the_previous_file_when_serialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    path = storage.save(pd.DataFrame({"value": [1]}), tmp_path / "frame.parquet")

    def fail_to_parquet(self: pd.DataFrame, *_args: Any, **_kwargs: Any) -> None:
        del self
        raise OSError("simulated failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    with pytest.raises(OSError, match="simulated failure"):
        storage.save(pd.DataFrame({"value": [2]}), path)
    assert storage.load(path)["value"].tolist() == [1]
    assert not list(tmp_path.glob(".*.tmp"))


def test_loader_refuses_a_symbol_with_zero_usable_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _ohlcv(["2024-01-01", "2024-01-02"], symbol="BBB").to_csv(
        raw / "AAA.csv", index=False
    )
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "mislabeled_csv",
            "data": {
                "source": "csv",
                "symbols": ["AAA"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "market_calendar": "24/7",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    with pytest.raises(DataValidationError, match="reduced universe"):
        DataLoader(raw_dir=raw).load(config)


def test_build_source_distinguishes_csv_from_remote_sources() -> None:
    with pytest.raises(DataDownloadError, match="handled directly"):
        build_source("csv")


def test_loader_preserves_a_falsy_injected_storage(tmp_path: Path) -> None:
    class FalsyStorage(ParquetStorage):
        def __bool__(self) -> bool:
            return False

    storage = FalsyStorage(tmp_path / "cache", tmp_path / "metadata")
    assert DataLoader(storage=storage).storage is storage


def test_metadata_names_are_case_safe_and_windows_device_safe(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    upper = storage.write_metadata("NUL", {"value": np.float32(1.0)})
    lower = storage.write_metadata("nul", {"value": np.float32(2.0)})
    assert upper != lower
    assert upper.is_file()
    assert lower.is_file()
