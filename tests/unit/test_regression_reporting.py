"""Regression tests for reporting behavior."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv
from tests.regression_helpers import (
    _FakeResult,
    _holdout_config,
    _import_script,
    _rf_test_setup,
    _wf_experiment_config,
    _write_wf_artifacts,
)

from quantlab.config import ExperimentConfig


def test_report_command_snapshot_survives_two_consecutive_runs(tmp_path: Path) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    # First "report" run: reads the original artefacts, then would call
    # result.save() — simulate that by writing metadata.json from exactly
    # what ends up in `result.metadata`.
    first_result = _FakeResult(config)
    first_robustness = load_previous_walk_forward_robustness(
        exp_dir,
        first_result,  # type: ignore[arg-type]
    )
    assert first_robustness is not None
    (exp_dir / "metadata.json").write_text(
        json.dumps(first_result.metadata), encoding="utf-8"
    )

    # Second "report" run: same config, reloading from the metadata.json the
    # first run just wrote.
    second_result = _FakeResult(config)
    second_robustness = load_previous_walk_forward_robustness(
        exp_dir,
        second_result,  # type: ignore[arg-type]
    )
    assert second_robustness is not None
    assert "walk_forward_oos_metrics" in second_result.metadata


def test_report_command_is_noop_for_plain_backtest_without_prior_metadata(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "plain_experiment"
    exp_dir.mkdir()  # no metadata.json at all yet

    fake_result = _FakeResult(_wf_experiment_config())
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_refuses_reuse_while_bundle_save_is_incomplete(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)
    (exp_dir / ".quantlab-save-in-progress").write_text(
        "interrupted\n", encoding="utf-8"
    )

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )

    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_skips_reinjection_when_config_differs(tmp_path: Path) -> None:
    """Metadata's embedded ``walk_forward_config_snapshot`` must not be
    reused for a report regenerated with a DIFFERENT config."""
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    _write_wf_artifacts(exp_dir, _wf_experiment_config())

    different_config = ExperimentConfig.from_dict(
        {
            "experiment_name": "wf_experiment",
            "data": {
                "instruments": [
                    {"symbol": "C", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "D", "source": "csv", "calendar": "XNYS"},
                ],  # different universe
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    fake_result = _FakeResult(different_config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_missing_checksums_blocks_reuse(
    tmp_path: Path,
) -> None:
    """Missing checksums can't verify artefact integrity, so they must not
    be treated as a free pass to reuse anyway -- checksums are recorded
    unconditionally at every save (see BacktestResult.save()), so their
    absence here is itself a red flag (an incomplete or tampered
    metadata.json), not proof of "nothing to check"."""
    import json as _json

    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    metadata_path = exp_dir / "metadata.json"
    metadata = _json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["walk_forward_csv_checksums"]
    metadata_path.write_text(_json.dumps(metadata), encoding="utf-8")

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_refuses_reuse_for_malformed_metadata_json(
    tmp_path: Path,
) -> None:
    """A hand-edited or partially-written metadata.json must be refused
    gracefully, the same as a missing file or a mismatched hash -- not
    crash reuse detection with a raw json.JSONDecodeError."""
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    (exp_dir / "metadata.json").write_text("{not valid json", encoding="utf-8")

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_load_previous_robustness_artifacts_refuses_reuse_for_malformed_metadata_json(
    tmp_path: Path,
) -> None:
    """Same guarantee as load_previous_walk_forward_robustness, for the
    holdout/plain-backtest robustness-artefact reuse path: a malformed
    metadata.json must be refused gracefully, not crash with a raw
    json.JSONDecodeError."""
    from quantlab.backtesting.result import load_previous_robustness_artifacts

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    (exp_dir / "metadata.json").write_text("{not valid json", encoding="utf-8")

    fake_result = _FakeResult(config)
    artifacts = load_previous_robustness_artifacts(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert artifacts == {}


def test_config_dir_prefers_a_package_bundled_copy_when_present(
    tmp_path: Path,
) -> None:
    import shutil
    import sys

    from quantlab import constants as real_constants

    package_dir = Path(real_constants.__file__).resolve().parent
    fake_site_packages = tmp_path / "site-packages"
    fake_site_packages.mkdir()
    fake_package_dir = fake_site_packages / "quantlab"
    shutil.copytree(package_dir, fake_package_dir)
    bundled_configs = fake_package_dir / "configs"
    bundled_configs.mkdir()
    (bundled_configs / "example.yaml").write_text("experiment_name: x\n")

    sys.path.insert(0, str(fake_site_packages))
    saved_modules = {
        name: mod
        for name, mod in list(sys.modules.items())
        if name == "quantlab" or name.startswith("quantlab.")
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        import quantlab.constants as fake_constants

        assert bundled_configs == fake_constants.CONFIGS_DIR
    finally:
        sys.path.remove(str(fake_site_packages))
        for name in list(sys.modules):
            if name == "quantlab" or name.startswith("quantlab."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def test_dockerfile_builds_a_locked_non_editable_runtime_image() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    lines = dockerfile.splitlines()
    configs_copy_idx = next(
        i for i, line in enumerate(lines) if line.strip().startswith("COPY configs")
    )
    sync_idx = next(
        i for i, line in enumerate(lines) if line.strip().startswith("RUN uv sync")
    )
    assert configs_copy_idx < sync_idx
    assert "COPY pyproject.toml uv.lock" in dockerfile
    assert "uv sync --locked" in dockerfile
    assert "--no-editable" in dockerfile
    assert "python:3.12-slim@sha256:" in dockerfile
    assert " AS builder" in dockerfile
    assert " AS runtime" in dockerfile
    assert "USER quantlab" in dockerfile


def test_cli_shipped_config_resolves_a_bundled_config_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.cli as cli_mod
    import quantlab.constants as constants_mod

    bundled_dir = tmp_path / "bundled_configs"
    bundled_dir.mkdir()
    (bundled_dir / "demo_offline.yaml").write_text(
        "experiment_name: bundled_copy\n"
        "data:\n"
        "  instruments:\n"
        "    - symbol: AAA\n"
        "      source: csv\n"
        "      calendar: XNYS\n"
        "  start_date: '2020-01-01'\n"
        "  end_date: '2021-01-01'\n"
        "strategy:\n"
        "  name: buy_and_hold\n",
        encoding="utf-8",
    )
    (bundled_dir / "alternate.yml").write_text("experiment_name: alternate\n")
    monkeypatch.setattr(constants_mod, "CONFIGS_DIR", bundled_dir)

    with_suffix = cli_mod._resolve_config_path(None, "demo_offline.yaml")
    without_suffix = cli_mod._resolve_config_path(None, "demo_offline")
    assert with_suffix == without_suffix == bundled_dir / "demo_offline.yaml"
    assert cli_mod._resolve_config_path(None, "alternate.yml") == (
        bundled_dir / "alternate.yml"
    )
    cfg = cli_mod._load_config(with_suffix)
    assert cfg.experiment_name == "bundled_copy"


def test_cli_shipped_config_unknown_name_lists_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognised `--shipped-config` name must raise with a clear,
    actionable message listing what *is* available, not a bare file-not-found."""
    import quantlab.cli as cli_mod
    import quantlab.constants as constants_mod
    from quantlab.exceptions import QuantLabError

    bundled_dir = tmp_path / "bundled_configs"
    bundled_dir.mkdir()
    (bundled_dir / "demo_offline.yaml").write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setattr(constants_mod, "CONFIGS_DIR", bundled_dir)

    with pytest.raises(QuantLabError, match="No shipped config named"):
        cli_mod._resolve_config_path(None, "does_not_exist")


@pytest.mark.parametrize(
    "name",
    [
        "../demo_offline",
        r"..\demo_offline",
        "/tmp/demo_offline.yaml",
        r"C:\temp\demo_offline.yaml",
        "demo_offline.txt",
        "",
        " demo_offline",
    ],
)
def test_cli_shipped_config_rejects_paths_and_non_yaml_names(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.cli as cli_mod
    import quantlab.constants as constants_mod
    from quantlab.exceptions import QuantLabError

    bundled_dir = tmp_path / "bundled_configs"
    bundled_dir.mkdir()
    monkeypatch.setattr(constants_mod, "CONFIGS_DIR", bundled_dir)

    with pytest.raises(QuantLabError, match="Invalid shipped config name"):
        cli_mod._resolve_config_path(None, name)


def test_cli_shipped_config_rejects_symlink_outside_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.cli as cli_mod
    import quantlab.constants as constants_mod
    from quantlab.exceptions import QuantLabError

    bundled_dir = tmp_path / "bundled_configs"
    bundled_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("experiment_name: outside\n", encoding="utf-8")
    link = bundled_dir / "linked.yaml"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable on this platform: {exc}")
    monkeypatch.setattr(constants_mod, "CONFIGS_DIR", bundled_dir)

    with pytest.raises(QuantLabError, match="resolved path"):
        cli_mod._resolve_config_path(None, "linked.yaml")


def test_cli_requires_exactly_one_of_config_or_shipped_config() -> None:
    """Neither `--config`/`--shipped-config` given, or both given at once,
    must raise a clear error rather than silently preferring one."""
    import quantlab.cli as cli_mod
    from quantlab.exceptions import QuantLabError

    with pytest.raises(QuantLabError, match="Exactly one of"):
        cli_mod._resolve_config_path(None, None)
    with pytest.raises(QuantLabError, match="Exactly one of"):
        cli_mod._resolve_config_path(Path("configs/default.yaml"), "demo_offline")


def test_run_walk_forward_wrapper_passes_explicit_cli_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    wrapper: Any = _import_script("run_walk_forward")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment_name: unused\n", encoding="utf-8")
    received: dict[str, object] = {}

    def fake_walk_forward(**kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(wrapper, "walk_forward", fake_walk_forward)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_walk_forward.py", "--config", str(config_path)],
    )

    assert wrapper.main() == 0
    assert received == {"config": config_path, "shipped_config": None}


def test_buy_and_hold_empty_grid_message_describes_one_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typer

    from quantlab.cli import _echo_parameter_grid

    messages: list[str] = []
    monkeypatch.setattr(typer, "echo", lambda message: messages.append(str(message)))

    _echo_parameter_grid("buy_and_hold", {})

    assert messages == [
        "  parameter grid: 1 combination (buy_and_hold has no parameters to "
        "optimize; using the configured strategy as-is)"
    ]


def test_cli_walk_forward_delegates_all_validation_csvs_to_result_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import quantlab.backtesting.runner as runner_module
    import quantlab.cli as cli_module
    import quantlab.data.loader as loader_module
    import quantlab.validation.robustness as robustness_module
    import quantlab.validation.walk_forward as walk_forward_module

    data, cfg = _holdout_config()
    returns = pd.Series(
        [0.01, -0.005],
        index=pd.date_range("2020-01-01", periods=2),
        name="return",
    )
    equity = (1.0 + returns).cumprod().rename("equity")
    summary = pd.DataFrame({"fold": [0], "test_sharpe": [0.42]})
    stress = pd.DataFrame({"scenario": ["baseline"], "sharpe": [0.42]})
    wf = SimpleNamespace(
        folds=[object()],
        oos_returns=returns,
        oos_equity=equity,
        # None here (not a full BacktestResult): exercises cli.py's
        # documented fallback to oos_metrics() when oos_result is absent,
        # same as this test's own intent -- it isn't testing OOS-metrics
        # completeness (see test_saved_metadata_records_an_explicit_
        # result_scope / the walk-forward metrics-reuse tests for that),
        # only that the CLI delegates validation CSVs to result.save().
        oos_result=None,
        summary_table=lambda: summary,
        oos_metrics=lambda periods_per_year, risk_free_rate: {
            "sharpe_ratio": 0.42,
            "cagr": 0.05,
        },
    )
    saved: dict[str, object] = {}

    class FakeResult:
        def __init__(self) -> None:
            self.save_warnings: list[str] = []
            self.metadata: dict[str, Any] = {"run_timestamp": "2026-01-01T00:00:00Z"}

        def save(self, output: Path, **kwargs: object) -> Path:
            saved["output"] = output
            saved.update(kwargs)
            return output

    class FakeValidator:
        def __init__(self, config: ExperimentConfig) -> None:
            assert config is cfg

        def run(self, loaded_data: pd.DataFrame, **kwargs: object) -> object:
            assert loaded_data is data
            saved["validator_kwargs"] = kwargs
            return wf

    monkeypatch.setattr(cli_module, "GENERATED_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli_module, "_load_config", lambda _path: cfg)
    monkeypatch.setattr(cli_module, "_default_grid", lambda _cfg: {})
    monkeypatch.setattr(
        loader_module.DataLoader,
        "load",
        lambda _self, _cfg: (data, SimpleNamespace(warnings=[])),
    )
    monkeypatch.setattr(walk_forward_module, "WalkForwardValidator", FakeValidator)
    monkeypatch.setattr(
        robustness_module,
        "run_stress_tests",
        lambda _data, _cfg, **_kwargs: stress,
    )
    monkeypatch.setattr(
        runner_module,
        "run_backtest_from_config",
        lambda _data, _cfg, *, data_quality_report: FakeResult(),
    )

    cli_module.walk_forward(config=tmp_path / "config.yaml", shipped_config=None)

    artifacts = saved["validation_artifacts"]
    assert isinstance(artifacts, dict)
    assert set(artifacts) == {
        "walk_forward_results.csv",
        "walk_forward_oos_returns.csv",
        "walk_forward_oos_equity.csv",
        "stress_tests.csv",
    }
    pd.testing.assert_frame_equal(artifacts["walk_forward_results.csv"], summary)
    pd.testing.assert_series_equal(artifacts["walk_forward_oos_returns.csv"], returns)
    pd.testing.assert_series_equal(artifacts["walk_forward_oos_equity.csv"], equity)
    pd.testing.assert_frame_equal(artifacts["stress_tests.csv"], stress)


@pytest.mark.parametrize(
    ("source", "expected", "unexpected"),
    [
        ("csv", "Loaded 2 rows", "Cached 2 rows"),
        ("yahoo", "Cached 2 rows", "Loaded 2 rows"),
    ],
)
def test_download_data_wrapper_reports_source_aware_persistence(
    source: str,
    expected: str,
    unexpected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    wrapper: Any = _import_script("download_data")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    config = type("Config", (), {"data_source": source})()
    frame = pd.DataFrame({"symbol": ["AAA", "AAA"], "close": [1.0, 2.0]})

    monkeypatch.setattr(
        wrapper.ExperimentConfig,
        "from_yaml",
        lambda _path: config,
    )
    monkeypatch.setattr(
        wrapper.DataLoader,
        "download",
        lambda _self, _config, *, force: frame,
    )
    monkeypatch.setattr(wrapper, "configure_logging", lambda: None)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "-c", str(config_path)])

    assert wrapper.main() == 0
    output = capsys.readouterr().out
    assert expected in output
    assert unexpected not in output
    if source == "csv":
        assert "not written to the Parquet cache" in output


def test_report_fallback_discovers_a_bundled_yml_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import quantlab.backtesting.result as result_module
    import quantlab.backtesting.runner as runner_module
    import quantlab.cli as cli_module
    import quantlab.constants as constants_module
    import quantlab.data.loader as loader_module

    data, cfg = _holdout_config()
    experiment = cfg.experiment_name
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    cfg.to_yaml(bundled / "only-yml.yml")
    reports = tmp_path / "reports"
    saved: dict[str, object] = {}
    fake_result = SimpleNamespace(save_warnings=[])

    monkeypatch.setattr(cli_module, "GENERATED_REPORTS_DIR", reports)
    monkeypatch.setattr(constants_module, "CONFIGS_DIR", bundled)
    monkeypatch.setattr(
        loader_module.DataLoader,
        "load",
        lambda _self, loaded_cfg: (
            data,
            SimpleNamespace(warnings=[]),
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "run_backtest_from_config",
        lambda loaded_data, loaded_cfg, *, data_quality_report: fake_result,
    )

    def fake_save(result: object, exp_dir: Path) -> Path:
        saved["result"] = result
        saved["exp_dir"] = exp_dir
        return exp_dir

    monkeypatch.setattr(result_module, "save_with_walk_forward_reuse", fake_save)

    cli_module.report(experiment=experiment)

    assert saved["result"] is fake_result
    assert saved["exp_dir"] == (reports / experiment).resolve()


def test_code_hash_changes_when_quantlab_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from quantlab.backtesting import engine

    real_root = Path(engine.__file__).resolve().parents[1]
    fake_root = tmp_path / "quantlab"
    shutil.copytree(real_root, fake_root)

    monkeypatch.setattr(
        engine, "__file__", str(fake_root / "backtesting" / "engine.py")
    )
    original = engine._source_hash()

    # Editing an arbitrary module's content (no version bump, no Git repo
    # involved at all) must change the hash. No explicit cache-clearing
    # call needed (unlike a plain `@lru_cache`, which would have no way to
    # notice this on its own within the same process): `_source_hash`
    # re-`stat()`s every file on each call and invalidates itself
    # automatically once one has actually changed.
    target = fake_root / "constants.py"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8"
    )
    mutated = engine._source_hash()

    assert original != mutated


def test_generator_hash_changes_when_generate_report_script_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scripts/generate_report.py lives outside src/quantlab/ entirely, so
    _hash_source_tree's own rglob scan can never reach it -- but it makes
    the exact same walk-forward save/reuse decision as the CLI
    (save_with_walk_forward_reuse), so its own content must still be
    covered by _generator_hash(), the hash everything gating artifact
    reuse is keyed on."""
    from quantlab.backtesting import engine

    fake_script = tmp_path / "generate_report.py"
    fake_script.write_text("# v1\n", encoding="utf-8")
    monkeypatch.setattr(engine, "_GENERATOR_SCRIPT", fake_script)

    original = engine._generator_hash()
    fake_script.write_text("# v2 -- edited\n", encoding="utf-8")
    mutated = engine._generator_hash()

    assert original != mutated


def test_dependency_versions_tracks_statsmodels() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    versions = result.metadata["dependency_versions"]
    assert isinstance(versions, dict)
    assert "statsmodels" in versions


def test_dependency_versions_tracks_pandas_market_calendars() -> None:
    """pandas-market-calendars directly drives calendar/session/settlement
    results (holidays, sessions, closures) -- a version bump can change
    backtest output the same way a pandas/numpy bump can, so it must be
    part of provenance too, not silently missing from it."""
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    versions = result.metadata["dependency_versions"]
    assert "pandas-market-calendars" in versions


def test_git_dirty_state_is_recorded_alongside_commit_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import functools
    import shutil
    import subprocess

    from quantlab.backtesting import engine

    if shutil.which("git") is None:
        pytest.skip("git is not installed in this environment")

    repo = tmp_path
    run = functools.partial(
        subprocess.run, cwd=repo, capture_output=True, text=True, check=True
    )
    run(["git", "init"])
    run(["git", "config", "user.email", "test@example.com"])
    run(["git", "config", "user.name", "Test"])
    tracked = repo / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    run(["git", "add", "tracked.txt"])
    run(["git", "commit", "-m", "initial"])

    # `_git_is_dirty` hardcodes `cwd=Path(__file__)...` (engine.py's own
    # directory) rather than accepting one — intercept `subprocess.run` to
    # redirect that `cwd` to our disposable repo instead.
    real_run = subprocess.run

    def _run_in_repo(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        kwargs["cwd"] = repo
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(engine.subprocess, "run", _run_in_repo)
    assert engine._git_is_dirty() is False

    # Modifying a tracked file must flip it to dirty.
    tracked.write_text("modified\n", encoding="utf-8")
    assert engine._git_is_dirty() is True

    # An untracked file must NOT count as dirty (--untracked-files=no): it
    # doesn't change what any already-imported code actually does.
    run(["git", "checkout", "--", "tracked.txt"])
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert engine._git_is_dirty() is False


def test_save_records_figure_failures_instead_of_masking_them(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    assert result.save_warnings == []  # nothing recorded before any save

    with patch(
        "quantlab.reporting.charts.save_figures", side_effect=RuntimeError("boom")
    ):
        out = result.save(tmp_path / "out")

    assert len(result.save_warnings) == 1
    assert "boom" in result.save_warnings[0]
    # The numeric artefacts must still be written despite the figure failure.
    assert (out / "metrics.json").is_file()
    assert (out / "equity_curve.csv").is_file()


def test_save_clears_managed_stale_figures_but_preserves_foreign_files(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = tmp_path / "out"
    result.save(out)
    assert any((out / "figures").iterdir())  # sanity: a clean save has PNGs

    stale_managed = out / "figures" / "equity_curve.png"
    foreign = out / "figures" / "user_chart.png"
    foreign.write_bytes(b"user-owned")
    foreign_directory = out / "figures" / "custom"
    foreign_directory.mkdir()
    foreign_note = foreign_directory / "notes.txt"
    foreign_note.write_text("preserve me", encoding="utf-8")

    with patch("quantlab.reporting.charts.report_figures", return_value={}):
        result.save(out)

    assert not stale_managed.exists()
    assert foreign.read_bytes() == b"user-owned"
    assert foreign_note.read_text(encoding="utf-8") == "preserve me"


def test_save_refuses_a_symlinked_figures_directory(tmp_path: Path) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.exceptions import BacktestError

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (out / "figures").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable on this platform: {exc}")

    with pytest.raises(BacktestError, match="symbolic link"):
        result.save(out)


def test_interrupted_save_marker_blocks_walk_forward_reuse(tmp_path: Path) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    exp_dir = tmp_path / "out"
    exp_dir.mkdir()
    (exp_dir / ".quantlab-save-in-progress").write_text("interrupted\n")
    (exp_dir / "metadata.json").write_text(
        json.dumps({"walk_forward_oos_metrics": {"sharpe_ratio": 99.0}}),
        encoding="utf-8",
    )

    assert load_previous_walk_forward_robustness(exp_dir, result) is None


def test_successful_save_removes_in_progress_marker(tmp_path: Path) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    out = result.save(tmp_path / "out")

    assert not (out / ".quantlab-save-in-progress").exists()


def test_save_uses_atomic_csv_replacement_and_marks_an_interrupted_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.backtesting.result as result_module
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = result.save(tmp_path / "out")
    equity_path = out / "equity_curve.csv"
    previous_equity = equity_path.read_bytes()
    original_write_csv = result_module._write_csv_atomic

    def failing_write_csv(
        value: pd.Series | pd.DataFrame,
        path: Path,
        *,
        index: bool = True,
    ) -> None:
        if path.name == "equity_curve.csv":

            def write_partial_then_fail(temporary: Path) -> None:
                temporary.write_bytes(b"partial replacement")
                raise RuntimeError("simulated interrupted CSV write")

            result_module._write_path_atomic(path, write_partial_then_fail)
            return
        original_write_csv(value, path, index=index)

    monkeypatch.setattr(result_module, "_write_csv_atomic", failing_write_csv)

    with pytest.raises(RuntimeError, match="interrupted CSV"):
        result.save(out)

    assert equity_path.read_bytes() == previous_equity
    assert (out / ".quantlab-save-in-progress").is_file()
    assert not list(out.glob(".equity_curve.csv.*.tmp"))


def test_save_writes_validation_csvs_and_checksums_inside_the_bundle(
    tmp_path: Path,
) -> None:
    import hashlib

    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    artifacts: dict[str, pd.Series | pd.DataFrame] = {
        "walk_forward_results.csv": pd.DataFrame({"fold": [0], "test_sharpe": [0.42]}),
        "walk_forward_oos_returns.csv": pd.Series([0.01], name="return"),
        "walk_forward_oos_equity.csv": pd.Series([101.0], name="equity"),
        "stress_tests.csv": pd.DataFrame({"scenario": ["baseline"], "sharpe": [0.42]}),
    }

    out = result.save(tmp_path / "out", validation_artifacts=artifacts)

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in artifacts
    }
    assert metadata["walk_forward_csv_checksums"] == expected
    assert not (out / ".quantlab-save-in-progress").exists()


def test_save_times_out_when_another_process_holds_the_bundle_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from filelock import FileLock

    import quantlab.backtesting.result as result_module
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.exceptions import BacktestError

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = tmp_path / "out"
    out.mkdir()
    lock = FileLock(str(result_module._bundle_lock_path(out)), timeout=0)

    monkeypatch.setattr(result_module, "_SAVE_LOCK_TIMEOUT_SECONDS", 0.01)
    with lock.acquire(), pytest.raises(BacktestError, match="another process"):
        result.save(out)

    assert not (out / ".quantlab-save-in-progress").exists()


def test_save_clears_stale_report_html_after_a_render_failure(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = tmp_path / "out"
    result.save(out)
    assert (out / "report.html").is_file()  # sanity: a clean save has one

    def failing_to_html(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom-html")

    result.to_html = failing_to_html  # type: ignore[assignment]
    result.save(out)

    assert not (out / "report.html").exists()


def test_save_figures_includes_cumulative_costs_chart(tmp_path: Path) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = result.save(tmp_path / "out")

    assert (out / "figures" / "cumulative_costs.png").is_file()
    assert 'alt="Cumulative costs"' in (out / "report.html").read_text(encoding="utf-8")


def test_cli_step_and_outcome_messages_are_cp1252_encodable(tmp_path: Path) -> None:
    from unittest.mock import patch

    import typer

    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.cli import _echo_save_outcome, _echo_step

    printed: list[str] = []
    with patch.object(
        typer, "secho", side_effect=lambda msg, **kw: printed.append(msg)
    ):
        _echo_step("Loading config configs/demo_offline.yaml")
        data, cfg = _holdout_config()
        result = run_backtest_from_config(data, cfg)
        result.save(tmp_path / "out")
        _echo_save_outcome(result, "Saved to somewhere")
        result.save_warnings = ["Could not render HTML report: boom"]
        _echo_save_outcome(result, "Saved to somewhere else")

    assert printed  # sanity: the patched calls actually happened
    for msg in printed:
        msg.encode("cp1252")  # raises UnicodeEncodeError if not representable


def test_cli_backtest_reports_partial_save_failure(tmp_path: Path) -> None:
    """`quantlab backtest`'s success banner must not print unconditionally
    when the figures/HTML report failed to render — a partial failure must
    be distinguishable from a clean save."""
    from unittest.mock import patch

    import typer

    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.cli import _echo_save_outcome

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    result.save(tmp_path / "out")
    assert result.save_warnings == []

    result.save_warnings = ["Could not render HTML report: boom"]
    printed: list[str] = []
    with patch.object(
        typer, "secho", side_effect=lambda msg, **kw: printed.append(msg)
    ):
        _echo_save_outcome(result, "Saved to somewhere")
    assert any("warning" in msg.lower() for msg in printed)
    assert not any(msg.startswith("[OK]") for msg in printed)


def test_rolling_sharpe_chart_subtracts_risk_free_rate() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from quantlab.reporting.charts import rolling_sharpe_chart

    data, cfg = _rf_test_setup(risk_free_rate=0.05)
    from quantlab.backtesting.runner import run_backtest_from_config

    result = run_backtest_from_config(data, cfg)
    fig = rolling_sharpe_chart(result, window=5)
    values = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)

    ppy = cfg.periods_per_year
    rf_period = cfg.risk_free_rate / ppy
    expected = (
        (result.returns - rf_period).rolling(5).mean()
        / result.returns.rolling(5).std(ddof=1)
    ) * np.sqrt(ppy)
    np.testing.assert_allclose(
        values[~np.isnan(values)],
        expected.to_numpy()[~expected.isna().to_numpy()],
        rtol=1e-9,
    )


def test_report_command_rejects_experiment_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    import quantlab.cli as cli_module

    messages: list[str] = []
    monkeypatch.setattr(cli_module, "GENERATED_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        typer,
        "secho",
        lambda message, **_kwargs: messages.append(str(message)),
    )
    monkeypatch.setattr(
        cli_module,
        "_load_config",
        lambda _path: pytest.fail("path traversal reached config loading"),
    )

    with pytest.raises(typer.Exit) as raised:
        cli_module.report(experiment="../../../etc")

    assert raised.value.exit_code == 1
    assert any(
        "Invalid --experiment" in message and "must not escape" in message
        for message in messages
    )


def test_report_figures_logs_a_warning_instead_of_silent_swallow(
    caplog: Any,
) -> None:
    import logging
    from unittest.mock import patch

    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.reporting.charts import report_figures

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "quantlab.reporting.charts.rolling_sharpe_chart",
            side_effect=RuntimeError("boom"),
        ),
    ):
        figures = report_figures(result)

    assert "rolling_sharpe" not in figures
    assert len(figures) >= 5  # every other chart still rendered
    assert any(
        "rolling_sharpe" in rec.getMessage() and "boom" in rec.getMessage()
        for rec in caplog.records
    )


def test_report_and_dashboard_rolling_sharpe_share_one_implementation() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from quantlab.reporting.charts import rolling_sharpe_chart
    from quantlab.risk.metrics import rolling_sharpe_ratio

    data, cfg = _rf_test_setup(risk_free_rate=0.04)
    from quantlab.backtesting.runner import run_backtest_from_config

    result = run_backtest_from_config(data, cfg)
    fig = rolling_sharpe_chart(result, window=10)
    chart_values = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)

    shared = rolling_sharpe_ratio(
        result.returns, 10, cfg.risk_free_rate, cfg.periods_per_year
    )
    np.testing.assert_allclose(
        chart_values[~np.isnan(chart_values)],
        shared.to_numpy()[~shared.isna().to_numpy()],
        rtol=1e-12,
    )


def test_mypy_scripts_do_not_require_the_notebooks_extra_to_resolve() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    overrides = pyproject["tool"]["mypy"]["overrides"]
    ignored_modules: set[str] = set()
    for override in overrides:
        if override.get("ignore_missing_imports"):
            ignored_modules.update(override["module"])
    assert "nbformat.*" in ignored_modules
    assert "nbclient.*" in ignored_modules


def test_run_backtest_script_flags_save_warnings_instead_of_plain_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    from unittest.mock import patch

    import quantlab.data.loader as loader_mod

    raw = tmp_path / "raw"
    raw.mkdir()
    make_ohlcv("AAA", np.linspace(100.0, 101.0, 60), start="2020-01-01").to_csv(
        raw / "AAA.csv", index=False
    )
    config = {
        "experiment_name": "run_backtest_script_save_warning_test",
        "data": {
            "instruments": [
                {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
            ],
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
        },
        "strategy": {"name": "buy_and_hold"},
    }
    import yaml

    config_path = tmp_path / "run_backtest_script_save_warning_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    run_backtest = _import_script("run_backtest")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest.py",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with patch(
        "quantlab.reporting.charts.save_figures", side_effect=RuntimeError("boom")
    ):
        run_backtest.main()  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "warning" in captured.out
    assert "boom" in captured.out
    assert "\nSaved to" not in captured.out  # the plain, unqualified message


def test_generate_report_script_flags_save_warnings_instead_of_plain_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same distinction, for `scripts/generate_report.py`."""
    import sys
    from unittest.mock import patch

    import quantlab.data.loader as loader_mod

    raw = tmp_path / "raw"
    raw.mkdir()
    make_ohlcv("AAA", np.linspace(100.0, 101.0, 60), start="2020-01-01").to_csv(
        raw / "AAA.csv", index=False
    )
    config = {
        "experiment_name": "generate_report_script_save_warning_test",
        "data": {
            "instruments": [
                {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
            ],
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
        },
        "strategy": {"name": "buy_and_hold"},
    }
    import yaml

    config_path = tmp_path / "generate_report_script_save_warning_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    generate_report = _import_script("generate_report")
    monkeypatch.setattr(generate_report, "GENERATED_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        sys, "argv", ["generate_report.py", "--config", str(config_path)]
    )
    with patch(
        "quantlab.reporting.charts.save_figures", side_effect=RuntimeError("boom")
    ):
        generate_report.main()  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "warning" in captured.out
    assert "boom" in captured.out
    assert not captured.out.startswith("Report written to")


def test_saved_metrics_json_has_no_nan_or_infinity_tokens(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "nan_json_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-01-30",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    result = run_backtest_from_config(frame, cfg)
    # A flat price series (zero variance) is exactly the realistic case that
    # produces NaN skewness/kurtosis -- confirm the premise before asserting
    # the invariant, so this test can't pass vacuously if that ever stops
    # being true (e.g. a metrics-formula change).
    assert math.isnan(result.metrics["skewness"])
    numpy_float32_nan: Any = np.float32(np.nan)
    result.metrics["numpy_float32_nan"] = numpy_float32_nan
    result.metadata["numpy_float32_infinity"] = np.float32(np.inf)

    out = result.save(tmp_path)
    raw_text = (out / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    parsed = json.loads(raw_text)
    assert parsed["skewness"] is None
    assert parsed["numpy_float32_nan"] is None
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["numpy_float32_infinity"] is None


def test_cumulative_costs_chart_actually_appears_in_html_report() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data = make_ohlcv(
        "AAA", geometric_series(120, mu=0.0005, sigma=0.01, s0=100.0, seed=1)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "cumulative_costs_html_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-04-30",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    result = run_backtest_from_config(data, cfg)
    html = result.to_html()
    assert 'alt="Cumulative costs"' in html


def test_save_warnings_surface_a_failed_report_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.reporting.charts as charts_mod
    from quantlab.backtesting.runner import run_backtest_from_config

    data = make_ohlcv(
        "AAA", geometric_series(120, mu=0.0005, sigma=0.01, s0=100.0, seed=1)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "chart_failure_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-04-30",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    result = run_backtest_from_config(data, cfg)

    def boom(_result: Any) -> Any:
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr(charts_mod, "equity_curve_chart", boom)

    out = result.save(tmp_path / "out")
    assert any(
        "equity_curve" in w and "simulated chart failure" in w
        for w in result.save_warnings
    )
    html_text = (out / "report.html").read_text(encoding="utf-8")
    assert "drawdown" in html_text.lower()  # the other charts still rendered


def test_saved_metadata_records_an_explicit_result_scope(tmp_path: Path) -> None:
    """Different CLI commands in walk-forward mode save fundamentally
    different result objects to the same experiment directory (a
    full-sample result with OOS evidence only attached as metadata, vs the
    OOS-stitched result itself) -- metadata.json must say explicitly which
    one `self.metrics` is, so a bundle read later isn't ambiguous about
    which methodology produced it."""
    from quantlab.backtesting.runner import run_backtest_from_config

    data = make_ohlcv(
        "AAA", geometric_series(120, mu=0.0005, sigma=0.01, s0=100.0, seed=1)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "result_scope_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-04-30",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )

    full_sample = run_backtest_from_config(data, cfg)
    out = full_sample.save(tmp_path / "full_sample")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result_scope"] == "full_sample"

    # Simulate a genuine wf.oos_result (see _attach_walk_forward_evidence /
    # walk_forward.py's own construction): its metrics ARE the OOS series.
    oos_scoped = run_backtest_from_config(data, cfg)
    oos_scoped.metadata["walk_forward_oos_metrics"] = dict(oos_scoped.metrics)
    out = oos_scoped.save(tmp_path / "oos_scoped")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result_scope"] == "out-of-sample (walk-forward test folds only)"
