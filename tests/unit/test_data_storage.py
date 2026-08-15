"""Tests for Parquet storage, universe and the CSV loader path."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tests.conftest import make_ohlcv

from quantlab.config import ExperimentConfig
from quantlab.constants import TIMESTAMP
from quantlab.data.loader import DataLoader
from quantlab.data.storage import ParquetStorage
from quantlab.data.universe import Universe


def test_parquet_roundtrip(tmp_path: Path) -> None:
    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    data = make_ohlcv("AAA", np.linspace(100, 110, 10))
    path = storage.save(data, tmp_path / "x.parquet")
    reloaded = storage.load(path)
    pd.testing.assert_frame_equal(
        data.reset_index(drop=True), reloaded.reset_index(drop=True)
    )


def test_symbol_cache_merge_and_cover(tmp_path: Path) -> None:
    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    early = make_ohlcv("AAA", np.linspace(100, 105, 6), start="2020-01-01")
    storage.write_symbol(early, "yahoo", "AAA", "1d")
    assert storage.cache_covers(
        "yahoo", "AAA", "1d", date(2020, 1, 1), date(2020, 1, 3)
    )
    # A later slice is merged in without duplicating timestamps.
    later = make_ohlcv("AAA", np.linspace(106, 112, 7), start="2020-01-09")
    storage.write_symbol(later, "yahoo", "AAA", "1d")
    merged = storage.read_symbol("yahoo", "AAA", "1d")
    assert merged is not None
    assert not merged.duplicated(subset=[TIMESTAMP]).any()


def test_hash_is_deterministic() -> None:
    data = make_ohlcv("AAA", np.linspace(100, 110, 10))
    assert ParquetStorage.hash_frame(data) == ParquetStorage.hash_frame(data.copy())


def test_universe_normalisation() -> None:
    u = Universe.from_symbols([" spy ", "spy", "QQQ"])
    assert u.to_list() == ["SPY", "QQQ"]
    assert len(Universe.liquid_multi_asset_etfs()) == 8
    assert "BTCUSDT" in Universe.crypto_major().to_list()


def test_universe_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "symbols.csv"
    pd.DataFrame({"symbol": ["SPY", "QQQ", "SPY"]}).to_csv(path, index=False)
    u = Universe.from_csv(path)
    assert u.to_list() == ["SPY", "QQQ"]


def test_csv_source_loader(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for sym, s0 in [("AAA", 100.0), ("BBB", 50.0)]:
        frame = make_ohlcv(sym, np.linspace(s0, s0 + 20, 60), start="2020-01-01")
        frame.to_csv(raw_dir / f"{sym}.csv", index=False)

    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "csv_test",
            "data": {
                "source": "csv",
                "symbols": ["AAA", "BBB"],
                "start_date": "2020-01-01",
                "end_date": "2020-02-15",
                "missing_value_policy": "drop",
                "market_calendar": "XNYS",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw_dir,
    )
    data, report = loader.load(config)
    assert set(data["symbol"].unique()) == {"AAA", "BBB"}
    assert report.row_count > 0
    # All rows within the requested window.
    ts = pd.to_datetime(data[TIMESTAMP])
    assert ts.min() >= pd.Timestamp("2020-01-01")
