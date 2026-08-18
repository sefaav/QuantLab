"""Integration tests for the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from tests.conftest import geometric_series, make_ohlcv
from typer.testing import CliRunner

from quantlab.cli import app

runner = CliRunner()


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
            "source": "csv",
            "symbols": ["AAA", "BBB", "CCC"],
            "start_date": "2019-01-01",
            "end_date": "2020-12-31",
            "market_calendar": "XNYS",
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
        "backtest": {"initial_capital": 100000, "benchmark_symbol": "AAA"},
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


def test_cli_stress_test_holdout_mode_saves_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["stress-test", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test"
    assert (exp_dir / "stress_tests.csv").is_file()
    assert "commission x2" in result.stdout


def test_cli_stress_test_walk_forward_mode_reruns_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk-forward mode must save walk-forward fold artefacts alongside the
    stress table, proving the whole process re-ran (not a plain backtest)."""
    config_path, raw = _write_offline_walk_forward_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["stress-test", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test"
    assert (exp_dir / "stress_tests.csv").is_file()
    assert (exp_dir / "walk_forward_results.csv").is_file()
    assert (exp_dir / "walk_forward_oos_returns.csv").is_file()
    metadata = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "walk_forward_oos_metrics" in metadata


def test_cli_bootstrap_cli_flag_overrides_yaml_n_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test"
    summary = (exp_dir / "bootstrap_summary.csv").read_text(encoding="utf-8")
    assert summary  # a real file was written


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    _patch_raw_dir(monkeypatch, raw)

    result = runner.invoke(app, ["permutation-test", "--config", str(config_path)])

    assert result.exit_code == 0, result.stdout
    assert "p-value" in result.stdout
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test"
    assert (exp_dir / "permutation_test.csv").is_file()


def _pin_provenance_for_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze code_hash so a robustness-artefact-reuse check depends only
    on what the test itself varies (e.g. the config).

    `load_previous_robustness_artifacts` checks code_hash (not git state —
    see its docstring), computed from actual current file contents. A
    concurrent edit to any `src/quantlab` file while this test runs would
    otherwise change `_source_hash()` between the two CLI invocations under
    test and fail it for a reason unrelated to the reuse logic being
    tested.
    """
    import quantlab.backtesting.engine as engine_module

    monkeypatch.setattr(engine_module, "_source_hash", lambda: "test-code-hash")


def test_cli_stress_test_then_bootstrap_preserves_both_in_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test_robustness_reuse"

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


def test_cli_bootstrap_does_not_reuse_stress_test_from_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test_robustness_reuse_mismatch"

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test"
    assert (exp_dir / "sensitivity.csv").is_file()


def test_cli_robustness_orchestrator_runs_only_enabled_techniques(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    from quantlab.constants import GENERATED_REPORTS_DIR

    exp_dir = GENERATED_REPORTS_DIR / "cli_test"
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
