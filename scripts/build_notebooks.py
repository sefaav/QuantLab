#!/usr/bin/env python
"""Build and execute the research notebooks under ``notebooks/``.

The notebook cells are defined in ``scripts/notebook_cells.py``. This script
builds each notebook with ``nbformat``, executes every code cell with
``nbclient`` and stores the resulting outputs in the generated ``.ipynb``
files.

Usage:
    python scripts/build_notebooks.py
    python scripts/build_notebooks.py --notebook 03_mean_reversion_research.ipynb
    python scripts/build_notebooks.py --timeout 3600

The research configurations use Yahoo Finance data. ``DataLoader`` first reads
from ``data/cache/`` and automatically downloads any missing date ranges when
network access is available.

For an offline build, or to reuse the exact data already stored locally,
populate the cache beforehand:

    quantlab download --config configs/momentum_sp500.yaml
    quantlab download --config configs/mean_reversion_etfs.yaml
    quantlab download --config configs/pairs_trading.yaml

Install the required development, notebook and Yahoo dependencies with:

    python -m pip install -e ".[dev,notebooks,yahoo]"
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).parent))
from notebook_cells import (
    NB_01_DATA_QUALITY,
    NB_02_MOMENTUM_RESEARCH,
    NB_03_MEAN_REVERSION_RESEARCH,
    NB_04_PAIRS_TRADING_RESEARCH,
    NB_05_ROBUSTNESS_ANALYSIS,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOK_DEFINITIONS = {
    "01_data_quality.ipynb": NB_01_DATA_QUALITY,
    "02_momentum_research.ipynb": NB_02_MOMENTUM_RESEARCH,
    "03_mean_reversion_research.ipynb": NB_03_MEAN_REVERSION_RESEARCH,
    "04_pairs_trading_research.ipynb": NB_04_PAIRS_TRADING_RESEARCH,
    "05_robustness_analysis.ipynb": NB_05_ROBUSTNESS_ANALYSIS,
}
NOTEBOOK_CONFIGS = {
    "01_data_quality.ipynb": ("configs/momentum_sp500.yaml",),
    "02_momentum_research.ipynb": ("configs/momentum_sp500.yaml",),
    "03_mean_reversion_research.ipynb": ("configs/mean_reversion_etfs.yaml",),
    "04_pairs_trading_research.ipynb": ("configs/pairs_trading.yaml",),
    "05_robustness_analysis.ipynb": ("configs/momentum_sp500.yaml",),
}


def build(name: str, cells: list[tuple[str, str]], *, timeout: int = 3600) -> None:
    """Build, execute and save one notebook.

    Args:
        name: Output filename under notebooks/.
        cells: List of (kind, source) where kind is "md" or "code".
        timeout: Maximum execution time in seconds for each code cell.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    print(f"Executing {name} ...")
    client.execute()
    # Bind stored outputs to the exact QuantLab implementation that produced
    # them. The notebook consistency test detects stale outputs after a source
    # change, even when the cell text itself did not change.
    from quantlab.backtesting.engine import _source_hash

    nb["metadata"]["quantlab"] = {
        "code_hash": _source_hash(),
        "config_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in NOTEBOOK_CONFIGS[name]
        },
        "generator": "scripts/build_notebooks.py",
    }
    out_path = NOTEBOOKS_DIR / name
    nbf.write(nb, out_path)
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        choices=sorted(NOTEBOOK_DEFINITIONS),
        help="Build only this notebook. By default all notebooks are rebuilt.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=3600,
        metavar="SECONDS",
        help="Maximum execution time per code cell (default: 3600).",
    )
    args = parser.parse_args(argv)
    selected = (
        {args.notebook: NOTEBOOK_DEFINITIONS[args.notebook]}
        if args.notebook
        else NOTEBOOK_DEFINITIONS
    )
    for name, cells in selected.items():
        build(name, cells, timeout=args.timeout)
    print(f"Built {len(selected)} notebook(s).")
    return 0


def _positive_int(value: str) -> int:
    """Return a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
