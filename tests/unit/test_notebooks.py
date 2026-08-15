"""Consistency checks for generated research notebooks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS = {
    "01_data_quality.ipynb": "NB_01_DATA_QUALITY",
    "02_momentum_research.ipynb": "NB_02_MOMENTUM_RESEARCH",
    "03_mean_reversion_research.ipynb": "NB_03_MEAN_REVERSION_RESEARCH",
    "04_pairs_trading_research.ipynb": "NB_04_PAIRS_TRADING_RESEARCH",
    "05_robustness_analysis.ipynb": "NB_05_ROBUSTNESS_ANALYSIS",
}
NOTEBOOK_CONFIGS = {
    "01_data_quality.ipynb": ("configs/momentum_sp500.yaml",),
    "02_momentum_research.ipynb": ("configs/momentum_sp500.yaml",),
    "03_mean_reversion_research.ipynb": ("configs/mean_reversion_etfs.yaml",),
    "04_pairs_trading_research.ipynb": ("configs/pairs_trading.yaml",),
    "05_robustness_analysis.ipynb": ("configs/momentum_sp500.yaml",),
}


def _cell_definitions() -> ModuleType:
    path = ROOT / "scripts" / "notebook_cells.py"
    spec = importlib.util.spec_from_file_location("quantlab_notebook_cells", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load notebook definitions from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_builder() -> ModuleType:
    path = ROOT / "scripts" / "build_notebooks.py"
    spec = importlib.util.spec_from_file_location("quantlab_notebook_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load notebook builder from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return "".join(cast(list[str], value))
    raise AssertionError(f"Unsupported notebook cell source: {type(value).__name__}")


@pytest.mark.parametrize(("filename", "definition_name"), NOTEBOOKS.items())
def test_generated_notebook_matches_its_cell_definition(
    filename: str, definition_name: str
) -> None:
    """Committed notebooks must match their single source of truth."""
    module = _cell_definitions()
    definitions = cast(list[tuple[str, str]], getattr(module, definition_name))
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    cells = notebook["cells"]

    assert len(cells) == len(definitions)
    for cell, (kind, source) in zip(cells, definitions, strict=True):
        expected_type = "markdown" if kind == "md" else "code"
        assert cell["cell_type"] == expected_type
        assert _source_text(cell["source"]) == source


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_generated_notebook_has_no_execution_errors(filename: str) -> None:
    """Every generated code cell must have completed successfully."""
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def test_notebook_templates_do_not_suppress_all_warnings() -> None:
    """Research notebooks must not hide every warning process-wide."""
    module = _cell_definitions()
    source = "\n".join(
        cell_source
        for definition_name in NOTEBOOKS.values()
        for _, cell_source in cast(
            list[tuple[str, str]], getattr(module, definition_name)
        )
    )

    assert 'warnings.filterwarnings("ignore")' not in source
    assert "import warnings" not in source


def test_notebook_plot_inputs_use_explicit_numpy_arrays() -> None:
    """Matplotlib inputs should not rely on pandas' ambiguous ``.values`` type."""
    module = _cell_definitions()
    source = "\n".join(
        cell_source
        for definition_name in NOTEBOOKS.values()
        for _, cell_source in cast(
            list[tuple[str, str]], getattr(module, definition_name)
        )
    )

    assert ".values" not in source


def test_notebook_builder_threads_the_configured_cell_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow research cells must be able to exceed the default timeout."""
    module = _notebook_builder()
    calls: list[tuple[str, int]] = []

    def fake_build(
        name: str, cells: list[tuple[str, str]], *, timeout: int = 3600
    ) -> None:
        assert cells
        calls.append((name, timeout))

    monkeypatch.setattr(module, "build", fake_build)
    assert module.main(["--notebook", "02_momentum_research.ipynb"]) == 0
    assert calls == [("02_momentum_research.ipynb", 3600)]

    calls.clear()
    assert (
        module.main(
            [
                "--notebook",
                "02_momentum_research.ipynb",
                "--timeout",
                "4200",
            ]
        )
        == 0
    )
    assert calls == [("02_momentum_research.ipynb", 4200)]


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_notebook_builder_rejects_invalid_cell_timeouts(value: str) -> None:
    """A non-positive or malformed timeout must fail before execution."""
    module = _notebook_builder()
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--timeout", value])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("filename", NOTEBOOKS)
def test_generated_notebook_outputs_match_current_quantlab_code(filename: str) -> None:
    """Stored outputs must come from the current QuantLab source tree."""
    from quantlab.backtesting.engine import _source_hash

    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    metadata = notebook.get("metadata", {}).get("quantlab", {})

    assert metadata.get("generator") == "scripts/build_notebooks.py"
    assert metadata.get("code_hash") == _source_hash()
    assert metadata.get("config_hashes") == {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in NOTEBOOK_CONFIGS[filename]
    }
