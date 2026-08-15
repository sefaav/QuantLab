#!/usr/bin/env python
"""Run an experiment backtest and save its artefact bundle.

Usage from the project root:
    python scripts/run_backtest.py --config configs/momentum_sp500.yaml

Write the artefacts to a custom directory:
    python scripts/run_backtest.py -c configs/default.yaml -o reports/custom_run

This script uses the same data-loading, backtesting and saving pipeline as
``quantlab backtest``. It does not run walk-forward validation or stress tests.

Saving over an existing experiment directory removes existing walk-forward and
stress-test artefacts because this command does not recompute them. To
regenerate only the report while preserving compatible validation evidence,
use ``scripts/generate_report.py`` or ``quantlab report``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a backtest from a config.")
    parser.add_argument("--config", "-c", required=True, type=Path)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=("Output directory. Defaults to reports/generated/<experiment_name>/."),
    )
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig.from_yaml(args.config)
    data, report = DataLoader().load(config)
    if report.warnings:
        # Keep data-quality warnings visible for script users, matching the CLI.
        print(f"[data] {len(report.warnings)} warning(s):")
        for message in report.warnings:
            print(f"  - {message}")
    result = run_backtest_from_config(data, config, data_quality_report=report)
    out = result.save(args.output)
    print(result.summary())
    if result.save_warnings:
        # Numeric artefacts may still be saved when optional report rendering fails.
        # Keep the status marker ASCII-only for legacy Windows console encodings.
        print(
            f"\n[WARN] Saved to {out} — but with {len(result.save_warnings)} "
            "warning(s) (report/figures may be incomplete or missing; see log):"
        )
        for warning in result.save_warnings:
            print(f"  - {warning}")
    else:
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
