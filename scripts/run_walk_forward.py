#!/usr/bin/env python
"""Run walk-forward validation and robustness tests for an experiment.

Usage from the project root:
    python scripts/run_walk_forward.py --config configs/momentum_sp500.yaml

This compatibility wrapper delegates to ``quantlab.cli.walk_forward`` so its
parameter grids, robustness tests, artefact saving and provenance checks remain
aligned with the main ``quantlab walk-forward`` command.

For normal interactive use, the equivalent CLI command is:

    quantlab walk-forward --config configs/momentum_sp500.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import typer

from quantlab.cli import walk_forward


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation and robustness tests."
    )
    parser.add_argument("--config", "-c", required=True, type=Path)
    args = parser.parse_args()

    try:
        walk_forward(config=args.config, shipped_config=None)
    except typer.Exit as exc:
        # Convert Typer's direct-call exit exception into this wrapper's process status.
        return int(exc.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
