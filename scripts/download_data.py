#!/usr/bin/env python
"""Download the market data required by an experiment.

Usage from the project root:
    python scripts/download_data.py --config configs/momentum_sp500.yaml

Force a fresh download instead of reusing the existing cache:
    python scripts/download_data.py --config configs/momentum_sp500.yaml --force

This is a thin wrapper around the same ``DataLoader.download()`` path used by
``quantlab download``. Remote Yahoo Finance and Binance data are stored in the
local Parquet cache; CSV input is read directly and is not copied into it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Download experiment data.")
    parser.add_argument("--config", "-c", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Ignore the cache.")
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig.from_yaml(args.config)
    data = DataLoader().download(config, force=args.force)
    symbol_count = data["symbol"].nunique()
    if config.data_source == "csv":
        print(
            f"Loaded {len(data)} rows for {symbol_count} symbols from CSV files "
            "(CSV input is not written to the Parquet cache)."
        )
    else:
        print(
            f"Cached {len(data)} rows for {symbol_count} symbols from "
            f"{config.data_source}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
