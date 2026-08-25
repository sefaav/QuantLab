"""Integration test for the data pipeline step 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tests.conftest import geometric_series, make_ohlcv

from quantlab.config import ExperimentConfig
from quantlab.constants import OHLCV_COLUMNS, TIMESTAMP
from quantlab.data.loader import DataLoader
from quantlab.data.storage import ParquetStorage


def test_csv_to_clean_validated_panel(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for sym, seed in [("AAA", 1), ("BBB", 2)]:
        prices = geometric_series(200, mu=0.0005, sigma=0.01, s0=100.0, seed=seed)
        frame = make_ohlcv(sym, prices, start="2020-01-01")
        # Inject a duplicate + a NaN to exercise cleaning.
        frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
        frame.loc[10, "close"] = np.nan
        frame.to_csv(raw / f"{sym}.csv", index=False)

    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "data_pipe",
            "data": {
                "instruments": [
                    {"symbol": s, "source": "csv", "calendar": "XNYS"}
                    for s in ["AAA", "BBB"]
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-08-01",
                "missing_value_policy": "drop",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    data, report = loader.load(config)

    assert list(data.columns) == list(OHLCV_COLUMNS)
    # Duplicates removed, no NaN prices remaining under the drop policy.
    assert not data.duplicated(subset=[TIMESTAMP, "symbol"]).any()
    assert data["close"].notna().all()
    assert report.row_count == len(data)
    assert (data["close"] > 0).all()
