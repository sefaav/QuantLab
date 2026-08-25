"""Integration tests for the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from tests.conftest import geometric_series, make_ohlcv
from typer.testing import CliRunner

from quantlab.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_reports_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate every CLI integration test's report output under its own
    tmp_path, never the real GENERATED_REPORTS_DIR -- several tests below
    share the experiment name "cli_test" and would otherwise leak
    checkpoints/results between each other whenever the full suite runs in
    one process (each still passes fine in isolation), a known source of
    full-suite-only flakiness."""
    path = tmp_path / "reports"
    _patch_reports_dir(monkeypatch, path)
    return path


def _write_offline_experiment(
    tmp_path: Path, *, extra: dict | None = None
) -> tuple[Path, Path]:
    """Create CSV data + a config using the csv source under tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for sym, seed in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
        prices = geometric_series(500, mu=0.0005, sigma=0.012, s0=100.0, seed=seed)
        make_ohlcv(sym, prices, start="2019-01-01").to_csv(
            raw / f"{sym}.csv", index=False
        )
    config = {
        "experiment_name": "cli_test",
        "data": {
            "instruments": [
                {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
            ],
            "start_date": "2019-01-01",
            "end_date": "2020-12-31",
        },
        "strategy": {
            "name": "cross_sectional_momentum",
            "parameters": {
                "lookback_period": 100,
                "skip_period": 5,
                "top_fraction": 0.5,
            },
        },
        "portfolio": {"allocator": "inverse_volatility", "maximum_weight": 0.6},
        "execution": {"commission_bps": 2.0, "spread_bps": 3.0, "slippage_bps": 2.0},
        "backtest": {
            "initial_capital": 100000,
            "benchmark": {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
        },
    }
    if extra:
        config.update(extra)
    config_path = tmp_path / "cli_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, raw


def _write_offline_walk_forward_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """A walk-forward-mode config with windows small enough for a fast fold."""
    return _write_offline_experiment(
        tmp_path,
        extra={
            "validation": {
                "method": "walk_forward",
                "train_window": 150,
                "validation_window": 60,
                "test_window": 60,
            }
        },
    )


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "download",
        "backtest",
        "walk-forward",
        "stress-test",
        "bootstrap",
        "permutation-test",
        "sensitivity",
        "robustness",
        "report",
        "dashboard",
    ]:
        assert command in result.stdout


def test_cli_backtest_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    # Point the DataLoader's default raw dir at our temp CSVs.
    import quantlab.data.loader as loader_mod

    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    out = tmp_path / "out"
    result = runner.invoke(
        app, ["backtest", "--config", str(config_path), "--output", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "report.html").is_file()
    assert (out / "metrics.json").is_file()


def test_cli_backtest_bad_config_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a: valid: config", encoding="utf-8")
    result = runner.invoke(app, ["backtest", "--config", str(bad)])
    assert result.exit_code != 0


def test_cli_missing_config_exits_nonzero() -> None:
    result = runner.invoke(app, ["backtest", "--config", "does_not_exist.yaml"])
    assert result.exit_code != 0


def _patch_raw_dir(monkeypatch: pytest.MonkeyPatch, raw: Path) -> None:
    import quantlab.data.loader as loader_mod

    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)


def _patch_reports_dir(monkeypatch: pytest.MonkeyPatch, reports_dir: Path) -> None:
    import quantlab.cli as cli_module

    monkeypatch.setattr(cli_module, "GENERATED_REPORTS_DIR", reports_dir)


def test_cli_stress_test_holdout_mode_saves_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["stress-test", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    assert (exp_dir / "stress_tests.csv").is_file()
    assert "commission x2" in result.stdout


@pytest.mark.slow
def test_cli_stress_test_walk_forward_mode_reruns_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """Walk-forward mode must save walk-forward fold artefacts alongside the
    stress table, proving the whole process re-ran (not a plain backtest)."""
    config_path, raw = _write_offline_walk_forward_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["stress-test", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    assert (exp_dir / "stress_tests.csv").is_file()
    assert (exp_dir / "walk_forward_results.csv").is_file()
    assert (exp_dir / "walk_forward_oos_returns.csv").is_file()
    metadata = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "walk_forward_oos_metrics" in metadata


@pytest.mark.slow
def test_cli_walk_forward_resumes_from_a_checkpoint_after_an_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """An interrupted `walk-forward` invocation must leave a checkpoint that
    a later, ordinary re-invocation of the same command picks up — fewer
    folds get (re-)selected on than a fresh run would need, proving it
    actually resumed instead of starting over."""
    config_path, raw = _write_offline_walk_forward_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)
    from quantlab.validation.walk_forward import WalkForwardValidator

    exp_dir = isolated_reports_dir / "cli_test"
    checkpoint_path = exp_dir / ".checkpoint_walk_forward.pkl"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    result = runner.invoke(app, ["walk-forward", "--config", str(config_path)])
    assert result.exit_code != 0
    assert checkpoint_path.is_file()
    monkeypatch.undo()
    # undo() also reverted these two patches (same monkeypatch instance).
    _patch_raw_dir(monkeypatch, raw)
    _patch_reports_dir(monkeypatch, isolated_reports_dir)

    # `walk-forward` also runs walk-forward-OOS-aware stress tests after
    # fold selection, which re-invoke _select_on_validation of their own for
    # scenarios that change signals/weights -- counted per call to
    # WalkForwardValidator.run() (the fold-selection step) rather than
    # globally, so those later, unrelated re-selections don't dilute what
    # this test is actually checking: that the *first* run() call (the one
    # resuming from the fold checkpoint) needed fewer than a fresh run's 4.
    real_run = WalkForwardValidator.run
    calls_per_run: list[int] = []

    def _counting_select(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls_per_run[-1] += 1
        return real_select2(self, *args, **kwargs)

    def _tracking_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls_per_run.append(0)
        return real_run(self, *args, **kwargs)

    real_select2 = WalkForwardValidator._select_on_validation
    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _counting_select)
    monkeypatch.setattr(WalkForwardValidator, "run", _tracking_run)
    result = runner.invoke(app, ["walk-forward", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert not checkpoint_path.is_file()
    assert calls_per_run[0] < 4  # 4 folds total; at least the 1st was already cached
    assert (exp_dir / "walk_forward_results.csv").is_file()


@pytest.mark.slow
def test_cli_walk_forward_fresh_flag_discards_an_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    config_path, raw = _write_offline_walk_forward_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)
    from quantlab.validation.walk_forward import WalkForwardValidator

    exp_dir = isolated_reports_dir / "cli_test"
    checkpoint_path = exp_dir / ".checkpoint_walk_forward.pkl"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    result = runner.invoke(app, ["walk-forward", "--config", str(config_path)])
    assert result.exit_code != 0
    assert checkpoint_path.is_file()
    monkeypatch.undo()
    # undo() also reverted these two patches (same monkeypatch instance).
    _patch_raw_dir(monkeypatch, raw)
    _patch_reports_dir(monkeypatch, isolated_reports_dir)

    # See test_cli_walk_forward_resumes_from_a_checkpoint_after_an_interruption
    # for why this counts per WalkForwardValidator.run() call rather than
    # globally: the walk-forward command's own stress-test step also
    # re-invokes _select_on_validation for scenarios that change
    # signals/weights, unrelated to whether --fresh discarded the fold
    # checkpoint.
    real_run = WalkForwardValidator.run
    calls_per_run: list[int] = []

    def _counting_select(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls_per_run[-1] += 1
        return real_select2(self, *args, **kwargs)

    def _tracking_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls_per_run.append(0)
        return real_run(self, *args, **kwargs)

    real_select2 = WalkForwardValidator._select_on_validation
    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _counting_select)
    monkeypatch.setattr(WalkForwardValidator, "run", _tracking_run)
    result = runner.invoke(
        app, ["walk-forward", "--config", str(config_path), "--fresh"]
    )
    assert result.exit_code == 0, result.stdout
    # Every fold recomputed, the stale checkpoint was discarded.
    assert calls_per_run[0] == 4


@pytest.mark.slow
def test_cli_walk_forward_fresh_flag_discards_the_nested_stress_cache_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """run_walk_forward_stress_tests() writes a second, nested checkpoint
    file for its weight-cache build (.checkpoint_stress_test_cache.pkl)
    alongside the main .checkpoint_stress_test.pkl -- --fresh must discard
    both, or a stale nested cache from an earlier interrupted run could
    silently be reused despite --fresh, contradicting its own documented
    "discard any existing checkpoint" guarantee."""
    from quantlab.validation.checkpoint import save_checkpoint
    from quantlab.validation.robustness import stress_test_checkpoint_paths

    config_path, raw = _write_offline_walk_forward_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    exp_dir = isolated_reports_dir / "cli_test"
    exp_dir.mkdir(parents=True, exist_ok=True)
    stress_checkpoint, nested_cache_checkpoint = stress_test_checkpoint_paths(
        exp_dir / ".checkpoint_stress_test.pkl"
    )
    # A leftover nested cache from some earlier interrupted stress-test run
    # -- content doesn't matter, only that --fresh must remove the file
    # unconditionally rather than leave it to be silently picked back up.
    save_checkpoint(nested_cache_checkpoint, {"dummy": "provenance"}, [], 0)
    assert nested_cache_checkpoint.is_file()

    result = runner.invoke(
        app, ["walk-forward", "--config", str(config_path), "--fresh"]
    )
    assert result.exit_code == 0, result.stdout
    assert not stress_checkpoint.is_file()
    assert not nested_cache_checkpoint.is_file()


def test_cli_bootstrap_cli_flag_overrides_yaml_n_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """CLI override > YAML > default: --n-iterations must win over
    robustness.bootstrap.n_iterations from the config."""
    config_path, raw = _write_offline_experiment(
        tmp_path, extra={"robustness": {"bootstrap": {"n_iterations": 250}}}
    )
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(
        app,
        ["bootstrap", "--config", str(config_path), "--n-iterations", "37"],
    )

    assert result.exit_code == 0, result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    summary = (exp_dir / "bootstrap_summary.csv").read_text(encoding="utf-8")
    assert summary  # a real file was written

    # The saved metadata must record the CLI-overridden value (37), not the
    # YAML default (250) -- otherwise a custom run is indistinguishable from
    # a standard one when the saved bundle is inspected later.
    metadata = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bootstrap_run_params"]["n_iterations"] == 37


@pytest.mark.parametrize(
    "option",
    ["--n-iterations", "--block-size"],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_bootstrap_rejects_non_positive_option_values_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_reports_dir: Path,
    option: str,
    value: str,
) -> None:
    """--n-iterations/--block-size must be rejected by Typer's own min=1
    bound at CLI-parsing time -- a bare ValueError from the underlying
    positive_int() check is not a QuantLabError, so bootstrap()'s own
    `except QuantLabError` would leave it as an unhandled traceback, and
    only after data was already loaded and a backtest already run."""
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(
        app, ["bootstrap", "--config", str(config_path), option, value]
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    assert not (exp_dir / "bootstrap_summary.csv").is_file()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_permutation_test_rejects_non_positive_n_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(
        app,
        ["permutation-test", "--config", str(config_path), "--n-iterations", value],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.stdout


def test_cli_sensitivity_records_the_axes_actually_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """CLI-supplied sensitivity axes must be recorded in the saved metadata,
    not just used to run the sweep -- otherwise a custom-axis run can't be
    distinguished from a robustness.sensitivity.parameters-driven one when
    the saved bundle is inspected later."""
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(
        app,
        [
            "sensitivity",
            "--config",
            str(config_path),
            "--param-x",
            "lookback_period",
            "--values-x",
            "60,100",
            "--param-y",
            "skip_period",
            "--values-y",
            "5,10",
        ],
    )

    assert result.exit_code == 0, result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    metadata = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    run_params = metadata["sensitivity_run_params"]
    assert run_params["parameter_x"] == "lookback_period"
    assert run_params["values_x"] == [60, 100]
    assert run_params["parameter_y"] == "skip_period"
    assert run_params["values_y"] == [5, 10]


def test_cli_bootstrap_without_cli_flag_uses_yaml_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --n-iterations, robustness.bootstrap.n_iterations from the
    YAML must be the one actually used (not the Pydantic 1000 default)."""
    config_path, raw = _write_offline_experiment(
        tmp_path, extra={"robustness": {"bootstrap": {"n_iterations": 17}}}
    )
    _patch_raw_dir(monkeypatch, raw)

    captured: dict[str, object] = {}
    import quantlab.cli as cli_module

    original = cli_module._compute_bootstrap

    def spy(cfg: object, result: object, **kwargs: object):  # type: ignore[no-untyped-def]
        captured["n_iterations"] = kwargs.get("n_iterations")
        return original(cfg, result, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_module, "_compute_bootstrap", spy)

    result = runner.invoke(app, ["bootstrap", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert captured["n_iterations"] is None  # CLI flag omitted


def test_cli_permutation_test_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["permutation-test", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "p-value" in result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    assert (exp_dir / "permutation_test.csv").is_file()


def _pin_provenance_for_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze generator_hash and git state so a reuse check depends only on
    what the test itself varies (e.g. the config).

    `load_previous_robustness_artifacts` checks generator_hash (not git
    state — see its docstring), computed from actual current file contents,
    including `cli.py` (unlike the narrower `code_hash`, which deliberately
    excludes it). A concurrent edit to any `src/quantlab` file while this
    test runs would otherwise change `_generator_hash()` between the two
    CLI invocations under test and fail it for a reason unrelated to the
    reuse logic being tested. `load_previous_walk_forward_robustness`
    (`quantlab.backtesting.result`) additionally refuses reuse whenever
    either side reports `git_dirty` -- this checkout's own working tree is
    routinely dirty during development, which would otherwise fail any
    walk-forward-reuse test for a reason that has nothing to do with the
    reuse logic under test either.

    Patches both `quantlab.backtesting.engine`'s own module-level functions
    (resolved as bare names by `engine.py`'s own `_build_metadata`, so
    patching the module's attributes is enough there) and
    `quantlab.validation.walk_forward`'s separately *imported* references
    to the same functions (`from ...engine import ...` binds its own names
    in that module's namespace -- patching only the origin module would
    leave a walk-forward OOS result's own metadata, built by
    `_build_oos_result`, using the real, unpinned values).
    """
    import quantlab.backtesting.engine as engine_module
    import quantlab.validation.walk_forward as walk_forward_module

    for module in (engine_module, walk_forward_module):
        monkeypatch.setattr(module, "_generator_hash", lambda: "test-generator-hash")
        monkeypatch.setattr(module, "_git_is_dirty", lambda: False)
        monkeypatch.setattr(module, "_git_commit_hash", lambda: "test-commit")


@pytest.mark.slow
def test_cli_stress_test_then_bootstrap_preserves_both_in_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """Running stress-test then bootstrap — two separate CLI invocations —
    against the same experiment directory must not delete the first
    command's evidence. `result.save()`'s pre-save cleanup removes any
    `_OPTIONAL_ARTIFACTS` file the specific call in progress doesn't
    re-supply, so without `save_with_robustness_reuse` the second command
    would silently wipe out the first's CSV and its report section."""
    config_path, raw = _write_offline_experiment(
        tmp_path, extra={"experiment_name": "cli_test_robustness_reuse"}
    )
    _patch_raw_dir(monkeypatch, raw)
    _pin_provenance_for_reuse(monkeypatch)

    exp_dir = isolated_reports_dir / "cli_test_robustness_reuse"

    stress_result = runner.invoke(app, ["stress-test", "--config", str(config_path)])
    assert stress_result.exit_code == 0, stress_result.stdout
    assert (exp_dir / "stress_tests.csv").is_file()

    bootstrap_result = runner.invoke(app, ["bootstrap", "--config", str(config_path)])
    assert bootstrap_result.exit_code == 0, bootstrap_result.stdout

    assert (exp_dir / "bootstrap_summary.csv").is_file()
    assert (exp_dir / "stress_tests.csv").is_file()
    report_html = (exp_dir / "report.html").read_text(encoding="utf-8")
    assert "Stress Tests" in report_html
    assert "Bootstrap" in report_html


@pytest.mark.slow
def test_cli_report_after_walk_forward_bootstrap_preserves_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """`bootstrap` in walk-forward mode, then `report`: the walk-forward CSVs
    and bootstrap's own CSV/run-params must all survive `report`'s save.

    `report` always saves via `save_with_walk_forward_reuse` (never
    `save_with_robustness_reuse`, which only the on-demand robustness
    commands use), so this exercises a genuinely different code path than
    test_cli_stress_test_then_bootstrap_preserves_both_in_report above.
    Two invariants must both hold for this to work: `save_with_walk_forward_
    reuse` has to know about bootstrap/permutation/sensitivity CSVs (not
    only the walk-forward ones), and the walk-forward OOS result bootstrap
    saves has to record walk_forward_config_snapshot/walk_forward_run_
    timestamp -- without which `report` couldn't recognise its own
    walk-forward CSVs as reusable and would delete them alongside
    everything else it doesn't recognise."""
    config_path, raw = _write_offline_walk_forward_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)
    _pin_provenance_for_reuse(monkeypatch)

    exp_dir = isolated_reports_dir / "cli_test"

    bootstrap_result = runner.invoke(
        app, ["bootstrap", "--config", str(config_path), "--n-iterations", "10"]
    )
    assert bootstrap_result.exit_code == 0, bootstrap_result.stdout
    assert (exp_dir / "bootstrap_summary.csv").is_file()
    assert (exp_dir / "walk_forward_results.csv").is_file()
    metadata_after_bootstrap = json.loads(
        (exp_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata_after_bootstrap.get("bootstrap_run_params") == {
        "n_iterations": 10,
        "block_size": metadata_after_bootstrap["bootstrap_run_params"]["block_size"],
    }
    assert "walk_forward_config_snapshot" in metadata_after_bootstrap
    assert "walk_forward_run_timestamp" in metadata_after_bootstrap

    report_result = runner.invoke(app, ["report", "--experiment", "cli_test"])
    assert report_result.exit_code == 0, report_result.stdout

    # Nothing bootstrap saved must have been deleted by report's own save.
    assert (exp_dir / "bootstrap_summary.csv").is_file()
    assert (exp_dir / "walk_forward_results.csv").is_file()
    assert (exp_dir / "walk_forward_oos_returns.csv").is_file()
    assert (exp_dir / "walk_forward_oos_equity.csv").is_file()
    metadata_after_report = json.loads(
        (exp_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert (
        metadata_after_report.get("bootstrap_run_params")
        == (metadata_after_bootstrap["bootstrap_run_params"])
    )
    assert "walk_forward_oos_metrics" in metadata_after_report
    report_html = (exp_dir / "report.html").read_text(encoding="utf-8")
    assert "Bootstrap" in report_html


@pytest.mark.slow
def test_cli_bootstrap_does_not_reuse_stress_test_from_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """The reuse in the test above must be provenance-checked, not
    unconditional: a stress_tests.csv left by a run under a *different*
    config must not silently survive into a bootstrap run under a changed
    one. Isolated from this checkout's own git state and source tree (see
    `_pin_provenance_for_reuse`) so this specifically exercises the config
    check, not an incidental dirty-tree or concurrent-edit refusal."""
    config_path, raw = _write_offline_experiment(
        tmp_path,
        extra={
            "experiment_name": "cli_test_robustness_reuse_mismatch",
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
        },
    )
    _patch_raw_dir(monkeypatch, raw)
    _pin_provenance_for_reuse(monkeypatch)

    exp_dir = isolated_reports_dir / "cli_test_robustness_reuse_mismatch"

    stress_result = runner.invoke(app, ["stress-test", "--config", str(config_path)])
    assert stress_result.exit_code == 0, stress_result.stdout
    assert (exp_dir / "stress_tests.csv").is_file()

    # Change the config in place (same experiment_name/data) rather than
    # calling _write_offline_experiment a second time, which would try to
    # recreate the same tmp_path/raw directory.
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["execution"]["commission_bps"] = 25.0
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    bootstrap_result = runner.invoke(app, ["bootstrap", "--config", str(config_path)])
    assert bootstrap_result.exit_code == 0, bootstrap_result.stdout

    assert (exp_dir / "bootstrap_summary.csv").is_file()
    assert not (exp_dir / "stress_tests.csv").is_file()


@pytest.mark.slow
def test_cli_bootstrap_does_not_reuse_a_tampered_stress_test_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    """A stress_tests.csv edited (or corrupted) on disk after it was saved
    must never be silently reused just because it's still valid CSV --
    load_previous_robustness_artifacts must verify its checksum, not only
    that provenance (config/data/code) still matches."""
    config_path, raw = _write_offline_experiment(
        tmp_path, extra={"experiment_name": "cli_test_tampered_reuse"}
    )
    _patch_raw_dir(monkeypatch, raw)
    _pin_provenance_for_reuse(monkeypatch)

    exp_dir = isolated_reports_dir / "cli_test_tampered_reuse"

    stress_result = runner.invoke(app, ["stress-test", "--config", str(config_path)])
    assert stress_result.exit_code == 0, stress_result.stdout
    assert (exp_dir / "stress_tests.csv").is_file()

    # Tamper with the file directly, bypassing the checksum that was
    # recorded when it was originally saved.
    tampered = pd.read_csv(exp_dir / "stress_tests.csv")
    tampered.loc[0, "sharpe"] = 999.0
    tampered.to_csv(exp_dir / "stress_tests.csv", index=False)

    bootstrap_result = runner.invoke(app, ["bootstrap", "--config", str(config_path)])
    assert bootstrap_result.exit_code == 0, bootstrap_result.stdout

    assert (exp_dir / "bootstrap_summary.csv").is_file()
    # The tampered CSV must not have been reused into this save (it's an
    # optional artefact the pre-save cleanup removes when the current call
    # doesn't re-supply or successfully recover it).
    assert not (exp_dir / "stress_tests.csv").is_file()
    report_html = (exp_dir / "report.html").read_text(encoding="utf-8")
    assert "Stress Tests" not in report_html


def test_cli_sensitivity_requires_axes_from_somewhere(tmp_path: Path) -> None:
    config_path, _raw = _write_offline_experiment(tmp_path)

    result = runner.invoke(app, ["sensitivity", "--config", str(config_path)])

    assert result.exit_code != 0
    # The [ERROR] message goes to stderr; result.output merges both streams.
    assert "No sensitivity axes given" in result.output


def test_cli_sensitivity_rejects_partial_cli_axes(tmp_path: Path) -> None:
    config_path, _raw = _write_offline_experiment(tmp_path)

    result = runner.invoke(
        app,
        [
            "sensitivity",
            "--config",
            str(config_path),
            "--param-x",
            "lookback_period",
        ],
    )

    assert result.exit_code != 0
    assert "must all be" in result.output


def test_cli_sensitivity_runs_with_cli_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(
        app,
        [
            "sensitivity",
            "--config",
            str(config_path),
            "--param-x",
            "lookback_period",
            "--values-x",
            "60,100",
            "--param-y",
            "skip_period",
            "--values-y",
            "5,10",
        ],
    )

    assert result.exit_code == 0, result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    assert (exp_dir / "sensitivity.csv").is_file()


def test_cli_robustness_orchestrator_runs_only_enabled_techniques(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_reports_dir: Path
) -> None:
    config_path, raw = _write_offline_experiment(
        tmp_path,
        extra={
            "robustness": {
                "bootstrap": {"enabled": True, "n_iterations": 30},
                "permutation_test": {"enabled": True, "n_iterations": 30},
            }
        },
    )
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["robustness", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    exp_dir = isolated_reports_dir / "cli_test"
    assert (exp_dir / "bootstrap_summary.csv").is_file()
    assert (exp_dir / "permutation_test.csv").is_file()
    # Neither stress tests nor sensitivity were enabled in this config.
    assert not (exp_dir / "stress_tests.csv").is_file()
    assert not (exp_dir / "sensitivity.csv").is_file()


def test_cli_robustness_orchestrator_warns_when_nothing_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["robustness", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "no robustness.* technique is enabled" in result.stdout


def test_make_cli_progress_callback_returns_none_when_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Piped/redirected output (e.g. CI logs) must not get a
    carriage-return-redrawn line — it only makes sense on a real terminal,
    and every caller already treats on_progress=None as "don't report"."""
    import sys

    import quantlab.cli as cli_module

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert cli_module._make_cli_progress_callback("Walk-forward") is None


def test_make_cli_progress_callback_writes_a_redrawn_line_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import quantlab.cli as cli_module

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    on_progress = cli_module._make_cli_progress_callback("Walk-forward")
    assert on_progress is not None

    on_progress(0, 10)
    on_progress(5, 10)
    out = capsys.readouterr().out
    assert out.count("\r") == 2
    assert "Walk-forward: 5/10" in out
    assert not out.endswith("\n")  # still in progress, no trailing newline yet


def test_make_cli_progress_callback_ends_with_a_newline_on_completion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import quantlab.cli as cli_module

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    on_progress = cli_module._make_cli_progress_callback("Walk-forward")
    assert on_progress is not None

    on_progress(0, 4)
    on_progress(4, 4)
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert "finishing" in out


def test_make_cli_progress_callback_flags_a_resumed_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A first tick with done > 0 can only mean a checkpoint was resumed —
    surfaced in the terminal since nothing else would tell the user."""
    import sys

    import quantlab.cli as cli_module

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    on_progress = cli_module._make_cli_progress_callback("Walk-forward")
    assert on_progress is not None

    on_progress(3, 10)
    out = capsys.readouterr().out
    assert "resumed from a previous checkpoint" in out
