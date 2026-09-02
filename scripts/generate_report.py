#!/usr/bin/env python
"""(Re)generate the research report and backtest artefacts for an experiment.

Usage from the project root:
    python scripts/generate_report.py --config configs/momentum_sp500.yaml

The script reloads the experiment data, runs the full-sample backtest and
writes the complete artefact bundle under
``reports/generated/<experiment_name>/``, including ``report.html``, figures,
metrics, metadata and CSV result files.

This command does not rerun walk-forward validation. Compatible walk-forward
artefacts from an earlier run are reused only when their configuration, data,
code and saved-file provenance checks still pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.backtesting.result import (
    resolve_experiment_directory,
    save_with_walk_forward_reuse,
)
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.cli import _strategy_diagnostics_robustness
from quantlab.config import ExperimentConfig
from quantlab.constants import GENERATED_REPORTS_DIR
from quantlab.data.loader import DataLoader
from quantlab.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate a research report and backtest artefacts."
    )
    parser.add_argument("--config", "-c", required=True, type=Path)
    args = parser.parse_args()

    configure_logging()
    config = ExperimentConfig.from_yaml(args.config)
    # This script's entire purpose is producing an HTML report, so it always
    # renders one -- regardless of output.save_html_report/save_figures,
    # which only govern whether *other* runs render the presentation layer.
    config = config.revalidated_copy(
        update={
            "output": config.output.revalidated_copy(
                update={"save_html_report": True, "save_figures": True}
            )
        }
    )
    data, report = DataLoader().load(config)
    if report.warnings:
        # Keep data-quality warnings visible for script users, matching the CLI.
        print(f"[data] {len(report.warnings)} warning(s):")
        for message in report.warnings:
            print(f"  - {message}")
    result = run_backtest_from_config(data, config, data_quality_report=report)
    out_dir = resolve_experiment_directory(config, default_root=GENERATED_REPORTS_DIR)
    # Preserve compatible walk-forward evidence while regenerating the report.
    # Reuse them only when the same compatibility checks as `quantlab report` pass.
    robustness_extra = _strategy_diagnostics_robustness(data, config)
    out = save_with_walk_forward_reuse(
        result, out_dir, robustness_extra=robustness_extra
    )
    if result.save_warnings:
        # Numeric artefacts may still be saved when optional report rendering fails.
        # Keep the status marker ASCII-only for legacy Windows console encodings.
        print(
            f"[WARN] Report written to {out / 'report.html'} — but with "
            f"{len(result.save_warnings)} warning(s) (report/figures may be "
            "incomplete or missing; see log):"
        )
        for warning in result.save_warnings:
            print(f"  - {warning}")
    else:
        print(f"Report written to {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
