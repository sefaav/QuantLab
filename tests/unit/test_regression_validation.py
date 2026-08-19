"""Regression tests for validation behavior."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv
from tests.regression_helpers import (
    _FAKE_WF_RUN_TIMESTAMP,
    _benchmark_contamination_frame,
    _FakeResult,
    _holdout_config,
    _rf_test_setup,
    _wf_experiment_config,
    _write_wf_artifacts,
)

from quantlab.config import ExperimentConfig, ValidationMethod
from quantlab.exceptions import InvalidConfigurationError


def test_holdout_config_produces_attached_oos_metrics() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv(sym, geometric_series(500, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [
            ("AAA", 1, 0.0007),
            ("BBB", 2, 0.0003),
            ("CCC", 3, 0.0005),
        ]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "holdout",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA", "BBB", "CCC"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 60,
                    "skip_period": 5,
                    "top_fraction": 0.5,
                },
            },
            "portfolio": {"allocator": "inverse_volatility"},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {"initial_capital": 100_000},
            "validation": {
                "method": "holdout",
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
            },
        }
    )
    result = run_backtest_from_config(data, cfg)
    assert "holdout_oos_metrics" in result.metadata
    assert "sharpe_ratio" in result.metadata["holdout_oos_metrics"]


def test_conclusion_never_claims_oos_without_artifact() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.reporting.research_summary import conclusion, methodology

    frames = [
        make_ohlcv(sym, geometric_series(300, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [("AAA", 1, 0.0007), ("BBB", 2, 0.0003)]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "plain",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA", "BBB"],
                "start_date": "2020-01-01",
                "end_date": "2020-11-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "backtest": {"initial_capital": 100_000},
        }
    )
    result = run_backtest_from_config(data, cfg)
    assert "holdout_oos_metrics" not in result.metadata
    assert "walk_forward_oos_metrics" not in result.metadata
    text = conclusion(result) + methodology(result)
    assert "out-of-sample" not in text.lower() or "full-sample" in text.lower()
    assert "full-sample" in conclusion(result)


def test_holdout_report_has_train_validation_test_breakdown() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    report = result.metadata["holdout_report"]
    for block in ("train", "validation", "test"):
        assert f"{block}_metrics" in report
        assert "sharpe_ratio" in report[f"{block}_metrics"]
        assert f"{block}_period" in report
        start, end = report[f"{block}_period"]
        assert start < end

    # Blocks must be chronologically ordered and non-overlapping.
    train_end = pd.Timestamp(report["train_period"][1])
    validation_start = pd.Timestamp(report["validation_period"][0])
    validation_end = pd.Timestamp(report["validation_period"][1])
    test_start = pd.Timestamp(report["test_period"][0])
    assert train_end <= validation_start
    assert validation_end <= test_start


def test_holdout_test_returns_and_equity_are_persisted_on_result() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    assert result.holdout_test_returns is not None
    assert result.holdout_test_equity is not None
    assert len(result.holdout_test_returns) > 0
    # Equity must have one more point than returns: an explicit baseline
    # point at the start of the holdout test block.
    assert len(result.holdout_test_equity) == len(result.holdout_test_returns) + 1


def test_holdout_report_saved_to_disk(tmp_path: Path) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    out = result.save(tmp_path / "out")

    assert (out / "holdout_test_returns.csv").is_file()
    assert (out / "holdout_test_equity.csv").is_file()
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "holdout_report" in metadata
    assert "train_metrics" in metadata["holdout_report"]


def test_holdout_split_table_appears_in_html_report_without_manual_wiring() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    html = result.to_html()
    assert "Test (out-of-sample)" in html


def test_conclusion_cross_references_full_sample_and_oos_sharpe_by_name() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.reporting.research_summary import conclusion

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    text = conclusion(result)
    assert "full-sample" in text.lower()
    assert "out-of-sample" in text.lower()
    # The two Sharpe values must both be printed, not just one.
    full_sharpe = result.metrics["sharpe_ratio"]
    oos_sharpe = result.metadata["holdout_oos_metrics"]["sharpe_ratio"]
    assert f"{full_sharpe:.2f}" in text
    assert f"{oos_sharpe:.2f}" in text


def test_conclusion_does_not_claim_separate_full_sample_for_walk_forward() -> None:
    """A walk-forward OOS result's `metrics` *is* the stitched OOS series —
    the conclusion must not present it a second time as a separate
    "full-sample" figure alongside an identical "out-of-sample" one."""
    from quantlab.reporting.research_summary import conclusion
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    text = conclusion(wf.oos_result)
    assert "separate full-sample" not in text
    oos_sharpe = wf.oos_result.metadata["walk_forward_oos_metrics"]["sharpe_ratio"]
    formatted = f"{oos_sharpe:.2f}"
    assert formatted in text
    # Printed once, not once as "out-of-sample" and again as a duplicate
    # "full-sample" figure that happens to be numerically identical.
    assert text.count(formatted) == 1


def test_out_of_sample_scope_distinguishes_walk_forward_from_holdout() -> None:
    """Only a walk-forward OOS result's `metrics` themselves *are* the
    out-of-sample series — a holdout result's `metrics` remain a genuine
    full-sample fit even though OOS evidence is also attached separately,
    so it must not be flagged the same way."""
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.reporting.research_summary import out_of_sample_scope
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    assert (
        out_of_sample_scope(wf.oos_result)
        == "out-of-sample (walk-forward test folds only)"
    )

    holdout_data, holdout_cfg = _holdout_config()
    holdout_result = run_backtest_from_config(holdout_data, holdout_cfg)
    assert "holdout_oos_metrics" in holdout_result.metadata
    assert out_of_sample_scope(holdout_result) is None

    plain_result = run_backtest_from_config(data, cfg)
    assert out_of_sample_scope(plain_result) is None


def test_subperiod_table_labels_walk_forward_aggregate_as_out_of_sample() -> None:
    """`subperiod_table`'s aggregate row must not call a walk-forward OOS
    series "Full sample" — the opposite of what it is."""
    from quantlab.reporting.tables import subperiod_table
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    table = subperiod_table(wf.oos_result)
    periods = set(table["Period"])
    assert "Out-of-sample" in periods
    assert "Full sample" not in periods


def test_html_report_labels_walk_forward_results_as_out_of_sample() -> None:
    """The Results section headings and executive summary must say
    out-of-sample (walk-forward), never the "Full-sample" wording that's
    only accurate for a genuine full-sample fit."""
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    html = wf.oos_result.to_html()
    assert "Out-of-sample (walk-forward) headline metrics" in html
    assert "Full-sample" not in html
    assert "These are out-of-sample (walk-forward test folds only) results" in html


def test_holdout_test_ratio_without_validation_ratio_does_not_crash() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv(sym, geometric_series(500, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [("AAA", 1, 0.0007), ("BBB", 2, 0.0003)]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "holdout_no_validation_block",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA", "BBB"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "validation": {"method": "holdout", "test_ratio": 0.2},
        }
    )
    result = run_backtest_from_config(data, cfg)
    report = result.metadata["holdout_report"]
    # A genuinely absent validation block is represented by omitting the key
    # entirely, not a fake empty dict — a real validation block always has
    # at least a few metrics in it, so `{}` would be ambiguous between "no
    # block" and "a block that happened to compute no metrics".
    assert "validation_metrics" not in report
    assert "validation_period" not in report
    assert "sharpe_ratio" in report["test_metrics"]
    # Must still render without crashing, and must NOT show a misleading
    # "Validation" row for a zero-width synthesised validation period as if
    # it were a real row with metrics.
    html = result.to_html()
    assert "Test (out-of-sample)" in html
    assert "<td>Validation</td>" not in html


def test_holdout_empty_train_block_does_not_crash() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frame = make_ohlcv("AAA", [100.0, 101.0], start="2020-01-01")
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "holdout_empty_train",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
            },
            "strategy": {"name": "buy_and_hold"},
            "validation": {"method": "holdout", "test_ratio": 0.9},
        }
    )
    result = run_backtest_from_config(frame, cfg)
    # Too small to produce a usable holdout at all — must degrade to "no
    # holdout attached", not crash.
    assert "holdout_report" not in result.metadata


def test_report_command_preserves_walk_forward_metadata(tmp_path: Path) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )

    assert fake_result.metadata["walk_forward_oos_metrics"] == {
        "sharpe_ratio": 0.42,
        "cagr": 0.05,
    }
    assert robustness is not None
    assert "walk_forward" in robustness
    assert "stress_tests" in robustness
    # The snapshot must be carried forward into `result.metadata`, not just
    # checked: `result.save()` writes metadata.json from this dict alone, so
    # a second consecutive `report` run depends on finding it already
    # attached there, not merely validated in passing.
    assert fake_result.metadata["walk_forward_config_snapshot"] == config.model_dump(
        mode="json"
    )


def test_report_command_detects_config_yaml_edited_after_walk_forward_run(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    old_config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, old_config)

    edited_config = ExperimentConfig.from_dict(
        {
            "experiment_name": "wf_experiment",
            "data": {
                "symbols": ["NEW1", "NEW2"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    edited_config.to_yaml(exp_dir / "config.yaml")  # simulates a hand edit
    # `report` would reload config.yaml (now edited) and pass it as
    # result.config — that is what we simulate here.
    reloaded_config = ExperimentConfig.from_yaml(exp_dir / "config.yaml")

    fake_result = _FakeResult(reloaded_config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_stale_walk_forward_when_data_hash_differs(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)  # old metadata.json has _FAKE_DATA_HASH

    fake_result = _FakeResult(config)
    fake_result.metadata["data_hash"] = "a-completely-different-hash"
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_stale_walk_forward_when_code_hash_differs(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)  # old metadata.json has _FAKE_CODE_HASH

    fake_result = _FakeResult(config)
    fake_result.metadata["code_hash"] = "a-completely-different-code-hash"
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_stale_walk_forward_when_code_hash_missing(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    fake_result.metadata["code_hash"] = None
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_stale_walk_forward_when_git_commit_differs(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)  # old metadata.json has _FAKE_GIT_COMMIT

    fake_result = _FakeResult(config)
    fake_result.metadata["git_commit"] = "a-completely-different-commit"
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_reuses_walk_forward_when_git_commit_unavailable(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    fake_result.metadata["git_commit"] = None
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is not None
    assert fake_result.metadata["walk_forward_oos_metrics"] == {
        "sharpe_ratio": 0.42,
        "cagr": 0.05,
    }


def test_report_command_rejects_stale_walk_forward_when_tree_is_dirty(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    fake_result.metadata["git_dirty"] = True
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_stale_walk_forward_when_dependencies_differ(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    fake_result.metadata["dependency_versions"] = {
        "quantlab": "0.1.0",
        "pandas": "9.9.9",
    }
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_walk_forward_csv_modified_on_disk(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    # Corrupt one CSV's bytes after its checksum was already recorded --
    # nothing about the config/data/code identity changed.
    (exp_dir / "walk_forward_oos_returns.csv").write_text(
        "timestamp,return\n2020-01-01,999.0\n", encoding="utf-8"
    )

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_corrupted_stress_tests_csv(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    (exp_dir / "stress_tests.csv").write_text("CORRUPTED,999", encoding="utf-8")

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_a_deleted_stress_tests_csv(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    (exp_dir / "stress_tests.csv").unlink()

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_preserves_walk_forward_run_timestamp(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    fake_result.metadata["run_timestamp"] = "2026-07-29T00:00:00+00:00"
    load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert fake_result.metadata["walk_forward_run_timestamp"] == _FAKE_WF_RUN_TIMESTAMP
    # The regeneration's own fresh run_timestamp must be left alone -- it
    # correctly describes when *this* plain-backtest recomputation happened.
    assert fake_result.metadata["run_timestamp"] == "2026-07-29T00:00:00+00:00"


def test_generate_report_preserves_walk_forward_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins git_dirty=False: `load_previous_walk_forward_robustness` refuses
    reuse whenever either save saw a dirty tree (git_commit can't be trusted
    otherwise), which is real, intentional behaviour — but leaving it to
    whatever the ambient repo state happens to be during a test run makes
    this test's pass/fail depend on something it doesn't control (an
    ordinarily-uncommitted working session, or an unrelated CI build step,
    would fail it for a reason that has nothing to do with the reuse logic
    actually under test here)."""
    import quantlab.backtesting.engine as engine_module

    monkeypatch.setattr(engine_module, "_git_is_dirty", lambda: False)
    from quantlab.backtesting.result import save_with_walk_forward_reuse
    from quantlab.backtesting.runner import run_backtest_from_config

    frame = make_ohlcv(
        "AAA", geometric_series(300, mu=0.0004, sigma=0.01, s0=100.0, seed=7)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "wf_generate_report_experiment",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-10-27",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    exp_dir = tmp_path / cfg.experiment_name
    exp_dir.mkdir()

    # Simulate the `quantlab walk-forward` command: attach its metadata and
    # let the result writer publish/checksum all validation CSVs in one save.
    wf_result = run_backtest_from_config(frame, cfg)
    wf_result.metadata["walk_forward_oos_metrics"] = {"sharpe_ratio": 0.42}
    wf_result.metadata["walk_forward_config_snapshot"] = cfg.model_dump(mode="json")
    wf_result.save(
        exp_dir,
        validation_artifacts={
            "walk_forward_results.csv": pd.DataFrame(
                {"fold": [0], "test_sharpe": [0.42]}
            ),
            "walk_forward_oos_returns.csv": pd.Series([0.01], name="return"),
            "walk_forward_oos_equity.csv": pd.Series([100.0], name="equity"),
        },
    )

    # `generate_report.py` re-running a plain backtest on the same config and
    # the same data (identical `data_hash`) over that same directory.
    report_result = run_backtest_from_config(frame, cfg)
    out = save_with_walk_forward_reuse(report_result, exp_dir)

    assert (out / "walk_forward_results.csv").is_file()
    assert (out / "walk_forward_oos_returns.csv").is_file()
    assert (out / "walk_forward_oos_equity.csv").is_file()
    saved_metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["walk_forward_oos_metrics"] == {"sharpe_ratio": 0.42}


def test_walk_forward_oos_metrics_use_configured_risk_free_rate() -> None:
    """`oos_metrics()` must use `config.backtest.risk_free_rate`, not
    always default to 0% — otherwise its Sharpe silently diverges from the
    full-sample Sharpe shown next to it in reports."""
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup(risk_free_rate=0.05)
    wf = WalkForwardValidator(cfg)
    wf_result = wf.run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    with_rf = wf_result.oos_metrics(cfg.periods_per_year, cfg.risk_free_rate)
    without_rf = wf_result.oos_metrics(cfg.periods_per_year, 0.0)
    assert with_rf["sharpe_ratio"] != pytest.approx(without_rf["sharpe_ratio"])


def test_walk_forward_charges_entry_cost_at_the_first_fold_start() -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    def run_with_commission(commission_bps: float) -> list[pd.Series]:
        data, cfg = _rf_test_setup()
        cfg = cfg.revalidated_copy(
            update={
                "execution": cfg.execution.revalidated_copy(
                    update={
                        "commission_bps": commission_bps,
                        "spread_bps": 0.0,
                        "slippage_bps": 0.0,
                    }
                )
            }
        )
        wf = WalkForwardValidator(cfg).run(
            data,
            parameter_grid={},
            train_window=100,
            validation_window=50,
            test_window=50,
        )
        return [f.test_returns for f in wf.folds]

    with_commission = run_with_commission(100.0)  # 1%
    without_commission = run_with_commission(0.0)
    assert len(with_commission) == len(without_commission) >= 3
    for i, (taxed, untaxed) in enumerate(
        zip(with_commission, without_commission, strict=True)
    ):
        # Per-fold reporting starts on the first bar where this fold's target
        # is executed, so the initial entry cost is on row 0.
        entry_cost = untaxed.iloc[0] - taxed.iloc[0]
        if i == 0:
            assert entry_cost == pytest.approx(0.01, abs=1e-6), (
                f"expected ~1% entry cost at the walk-forward run's first "
                f"fold start, got {entry_cost}"
            )
        else:
            assert entry_cost == pytest.approx(0.0, abs=1e-6), (
                f"fold {i}: expected ~0 extra cost (position carried over "
                f"from the previous fold, no re-entry needed), got "
                f"{entry_cost}"
            )


def test_walk_forward_chains_accounting_state_across_fold_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    # A perfectly flat (zero-drift, zero-volatility) price series isolates
    # the transaction-cost effect this test cares about from any actual
    # market P&L a real, non-constant price would also contribute.
    data = make_ohlcv("AAA", geometric_series(400, mu=0.0, sigma=0.0, s0=100.0, seed=1))
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "chain_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "commission_bps": 100.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100.0},
            # Explicit daily rebalancing: `run()` re-applies the rebalance
            # schedule to the concatenated *target* series, and the default
            # monthly schedule would otherwise mask this test's forced
            # fold-1 flip whenever both folds' test windows land in the
            # same calendar month (a monthly cadence legitimately doesn't
            # trade again until its next scheduled date, so the
            # rebalance-then-ffill step would keep holding fold 0's value
            # straight through fold 1's dates) — daily makes every date a
            # rebalance date, so the forced per-fold values this test
            # injects always take effect exactly where asserted, isolating
            # the turnover-chaining behaviour this test cares about from
            # the separate rebalance-schedule question.
            "portfolio": {"rebalance_frequency": "daily"},
        }
    )
    validator = WalkForwardValidator(cfg)

    from quantlab.constants import SYMBOL
    from quantlab.data.base import price_matrix
    from quantlab.validation.splits import WalkForwardWindow, walk_forward_windows

    tradable = data[data[SYMBOL].isin(set(cfg.symbols))]
    index = pd.DatetimeIndex(price_matrix(tradable, adjusted=True).index)
    windows = walk_forward_windows(
        index, train_window=100, validation_window=20, test_window=5, expanding=True
    )
    assert len(windows) >= 2

    # Force fold 0's test-period weights fully long, fold 1's fully short --
    # bypassing the real strategy/parameter-grid machinery entirely, so this
    # test exercises exactly the concatenation-then-account mechanism, not
    # any particular strategy's behaviour.
    forced = {0: 1.0, 1: -1.0}

    def fake_weights_on_test(
        self: WalkForwardValidator,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        value = forced.get(window.fold, 1.0)
        return pd.DataFrame({"AAA": [value] * len(window.test)}, index=window.test)

    monkeypatch.setattr(WalkForwardValidator, "_weights_on_test", fake_weights_on_test)

    result = validator.run(
        data, parameter_grid={}, train_window=100, validation_window=20, test_window=5
    )
    assert len(result.folds) >= 2

    fold0_end, fold1_end = windows[0].test[-1], windows[1].test[-1]
    chained_equity = result.oos_equity.loc[fold1_end]
    assert chained_equity == pytest.approx(97.02, abs=0.05), (
        f"expected the continuously-chained flip to cost ~2 units of "
        f"turnover (final equity ~97.02), got {chained_equity}"
    )

    # An independent-per-fold computation would instead land here --
    # confirms the assertion above is checking a value distinctly
    # different from that alternative, not vacuously satisfied by some
    # other unrelated equity value.
    independent_final = 98.01
    assert chained_equity != pytest.approx(independent_final, abs=0.05)
    assert fold0_end < fold1_end  # sanity: folds are in chronological order


def test_walk_forward_turnover_cap_chains_across_fold_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv("AAA", geometric_series(400, mu=0.0, sigma=0.0, s0=100.0, seed=1))
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "turnover_chain_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "portfolio": {"rebalance_frequency": "daily", "maximum_turnover": 0.1},
        }
    )
    validator = WalkForwardValidator(cfg)

    from quantlab.constants import SYMBOL
    from quantlab.data.base import price_matrix
    from quantlab.validation.splits import WalkForwardWindow, walk_forward_windows

    tradable = data[data[SYMBOL].isin(set(cfg.symbols))]
    index = pd.DatetimeIndex(price_matrix(tradable, adjusted=True).index)
    windows = walk_forward_windows(
        index, train_window=100, validation_window=20, test_window=5, expanding=True
    )
    assert len(windows) >= 2

    forced = {0: 1.0, 1: -1.0}

    def fake_weights_on_test(
        self: WalkForwardValidator,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        value = forced.get(window.fold, 1.0)
        return pd.DataFrame({"AAA": [value] * len(window.test)}, index=window.test)

    monkeypatch.setattr(WalkForwardValidator, "_weights_on_test", fake_weights_on_test)

    import quantlab.validation.walk_forward as wf_mod

    captured: dict[str, pd.DataFrame] = {}
    orig_run_accounting = wf_mod.run_accounting

    def spy_run_accounting(all_weights: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        captured["weights"] = all_weights.copy()
        return orig_run_accounting(all_weights, *args, **kwargs)

    monkeypatch.setattr(wf_mod, "run_accounting", spy_run_accounting)

    validator.run(
        data, parameter_grid={}, train_window=100, validation_window=20, test_window=5
    )

    weights = captured["weights"]
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    assert turnover.max() == pytest.approx(0.1, abs=1e-6), (
        f"maximum_turnover=0.1 must bound every rebalance, including across "
        f"a fold boundary; observed max realised turnover {turnover.max()}"
    )


def test_walk_forward_oos_curve_starts_flat_at_the_very_first_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv("AAA", geometric_series(400, mu=0.0, sigma=0.0, s0=100.0, seed=1))
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "first_bar_flat_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "commission_bps": 100.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "portfolio": {"rebalance_frequency": "daily"},
        }
    )
    validator = WalkForwardValidator(cfg)

    from quantlab.constants import SYMBOL
    from quantlab.data.base import price_matrix
    from quantlab.validation.splits import WalkForwardWindow, walk_forward_windows

    tradable = data[data[SYMBOL].isin(set(cfg.symbols))]
    index = pd.DatetimeIndex(price_matrix(tradable, adjusted=True).index)
    windows = walk_forward_windows(
        index, train_window=100, validation_window=20, test_window=5, expanding=True
    )
    assert len(windows) >= 1

    def fake_weights_on_test(
        self: WalkForwardValidator,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        # Fully long from the very first OOS date onward -- if the first bar
        # weren't forced flat, this would show a nonzero return/cost right
        # away instead of on the *second* date.
        return pd.DataFrame({"AAA": [1.0] * len(window.test)}, index=window.test)

    monkeypatch.setattr(WalkForwardValidator, "_weights_on_test", fake_weights_on_test)

    result = validator.run(
        data, parameter_grid={}, train_window=100, validation_window=20, test_window=5
    )
    first_date = result.oos_returns.index[0]
    assert result.oos_returns.iloc[0] == pytest.approx(0.0, abs=1e-12), (
        f"the very first OOS bar ({first_date}) must be flat (no prior "
        f"position to have entered from), got return {result.oos_returns.iloc[0]}"
    )
    assert result.oos_equity.iloc[0] == pytest.approx(cfg.initial_capital, abs=1e-9)
    # The *second* date is where the actual entry cost shows up.
    assert result.oos_returns.iloc[1] < 0.0, "the entry cost must show up on day 2"


def test_walk_forward_rejects_test_window_shorter_than_a_rebalance_cycle() -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv(
        "AAA", geometric_series(400, mu=0.0005, sigma=0.01, s0=100.0, seed=1)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "too_short_fold_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            # Default rebalance_frequency is monthly (~21 trading days) --
            # a 5-bar test_window can never reach a second rebalance date.
            "portfolio": {},
        }
    )
    validator = WalkForwardValidator(cfg)
    with pytest.raises(InvalidConfigurationError, match="rebalance"):
        validator.run(
            data,
            parameter_grid={},
            train_window=100,
            validation_window=20,
            test_window=5,
        )


def test_walk_forward_fold_metrics_reflect_only_the_deployed_period() -> None:
    from quantlab.constants import SYMBOL
    from quantlab.data.base import price_matrix
    from quantlab.validation.splits import walk_forward_windows
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv(
        "AAA", geometric_series(400, mu=0.0005, sigma=0.01, s0=100.0, seed=1)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "aligned_fold_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {},  # default rebalance_frequency: monthly
        }
    )
    validator = WalkForwardValidator(cfg)
    result = validator.run(
        data, parameter_grid={}, train_window=100, validation_window=20, test_window=40
    )
    assert len(result.folds) >= 2

    tradable = data[data[SYMBOL].isin(set(cfg.symbols))]
    index = pd.DatetimeIndex(price_matrix(tradable, adjusted=True).index)
    windows = walk_forward_windows(
        index, train_window=100, validation_window=20, test_window=40
    )
    reported_starts = [f.test_returns.index[0] for f in result.folds]
    raw_starts = [w.test[0] for w in windows]
    assert any(r != raw for r, raw in zip(reported_starts, raw_starts, strict=True)), (
        "expected at least one fold's reported start to be pushed forward "
        "off the raw bar-count boundary onto its own rebalance date"
    )

    oos_index = result.oos_returns.index
    for prev_fold, next_fold in zip(result.folds, result.folds[1:], strict=False):
        prev_end_loc = oos_index.get_loc(prev_fold.test_returns.index[-1])
        next_start_loc = oos_index.get_loc(next_fold.test_returns.index[0])
        assert isinstance(prev_end_loc, int)
        assert isinstance(next_start_loc, int)
        assert next_start_loc == prev_end_loc + 1, (
            "consecutive folds' reported ranges must be contiguous: "
            f"{prev_fold.fold} ends at position {prev_end_loc}, "
            f"{next_fold.fold} starts at position {next_start_loc}"
        )


def test_walk_forward_validation_scoring_charges_entry_cost_too() -> None:
    from quantlab.validation.walk_forward import _evaluate_fresh_from_window_start

    data, cfg = _rf_test_setup()
    cfg = cfg.revalidated_copy(
        update={
            "execution": cfg.execution.revalidated_copy(
                update={"commission_bps": 100.0, "spread_bps": 0.0, "slippage_bps": 0.0}
            )
        }
    )
    idx = pd.DatetimeIndex(sorted(data["timestamp"].unique()))
    train_start = idx[0]
    val_start = idx[100]
    val_end = idx[149]
    returns = _evaluate_fresh_from_window_start(
        data, cfg, train_start, val_start, val_end
    )
    # The entry trade (row 1, since row 0 is always forced flat) must show
    # the ~1% commission cost, not a silently-free continuation of a
    # position already built up during train.
    assert returns.iloc[1] < -0.005


def test_stress_test_baseline_sharpe_matches_engine_sharpe() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.validation.robustness import run_stress_tests

    data, cfg = _rf_test_setup()
    result = run_backtest_from_config(data, cfg)
    stress = run_stress_tests(data, cfg)
    baseline_sharpe = stress.loc[stress["scenario"] == "baseline", "sharpe"].iloc[0]
    assert baseline_sharpe == pytest.approx(result.metrics["sharpe_ratio"])


def test_bootstrap_accepts_risk_free_rate() -> None:
    from quantlab.validation.bootstrap import bootstrap_returns

    _, cfg = _rf_test_setup()
    returns = pd.Series(
        np.random.default_rng(0).normal(0.0005, 0.01, 300),
        index=pd.date_range("2020-01-01", periods=300, freq="B"),
    )
    with_rf = bootstrap_returns(
        returns, n_iterations=50, seed=1, risk_free_rate=cfg.risk_free_rate
    )
    without_rf = bootstrap_returns(returns, n_iterations=50, seed=1, risk_free_rate=0.0)
    assert with_rf.samples["sharpe"].mean() != pytest.approx(
        without_rf.samples["sharpe"].mean()
    )


def test_holdout_split_ignores_benchmark_calendar() -> None:
    from quantlab.data.base import price_matrix
    from quantlab.validation.holdout import compute_holdout_split
    from quantlab.validation.splits import chronological_split

    data = _benchmark_contamination_frame()
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-11",
                "end_date": "2020-01-20",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {"benchmark_symbol": "BENCH"},
            "validation": {"method": "holdout", "test_ratio": 0.4},
            "reproducibility": {"random_seed": 42},
        }
    )
    split = compute_holdout_split(data, cfg)
    assert split is not None

    tradable_only = data[data["symbol"] == "AAA"].reset_index(drop=True)
    tradable_index = pd.DatetimeIndex(price_matrix(tradable_only).index)
    expected = chronological_split(tradable_index, 0.6, 0.0, 0.4)
    assert split.test[0] == expected.test[0]


def test_walk_forward_windows_ignore_benchmark_calendar() -> None:
    from quantlab.data.base import price_matrix
    from quantlab.validation.splits import walk_forward_windows
    from quantlab.validation.walk_forward import WalkForwardValidator

    tradable_dates = pd.date_range("2020-01-11", periods=30, freq="D")
    benchmark_dates = pd.date_range("2020-01-01", periods=60, freq="D")
    frames = [
        make_ohlcv(
            "AAA",
            geometric_series(len(tradable_dates), 0.0005, 0.01, 100.0, seed=1),
            start="2020-01-11",
        ),
        make_ohlcv(
            "BENCH",
            geometric_series(len(benchmark_dates), 0.0005, 0.01, 100.0, seed=2),
            start="2020-01-01",
        ),
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-11",
                "end_date": "2020-02-09",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            # Daily rebalancing: this test is about benchmark-calendar
            # exclusion from fold boundaries, not about rebalance-cycle
            # alignment — daily makes every bar a rebalance date,
            # so a 5-bar test_window trivially contains one.
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {},
            "backtest": {"benchmark_symbol": "BENCH"},
            "validation": {"method": "walk_forward"},
            "reproducibility": {"random_seed": 42},
        }
    )
    validator = WalkForwardValidator(cfg)
    wf = validator.run(
        data, parameter_grid={}, train_window=10, validation_window=5, test_window=5
    )

    tradable_only = data[data["symbol"] == "AAA"].reset_index(drop=True)
    tradable_index = pd.DatetimeIndex(price_matrix(tradable_only).index)
    expected_windows = walk_forward_windows(tradable_index, 10, 5, 5, expanding=True)
    # A target chosen on test[0] is executed on the following bar.
    assert wf.folds[0].test_returns.index[0] == expected_windows[0].test[1]


def test_stale_benchmark_and_holdout_artifacts_cleaned_up_on_resave(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    out_dir = tmp_path / "exp"
    data, cfg_with = _holdout_config()
    cfg_with = cfg_with.revalidated_copy(
        update={
            "backtest": cfg_with.backtest.revalidated_copy(
                update={"benchmark_symbol": "AAA"}
            )
        }
    )
    result_with = run_backtest_from_config(data, cfg_with)
    result_with.save(out_dir)
    assert (out_dir / "benchmark.csv").is_file()
    assert (out_dir / "holdout_test_returns.csv").is_file()
    assert (out_dir / "holdout_test_equity.csv").is_file()

    cfg_without = cfg_with.revalidated_copy(
        update={
            "backtest": cfg_with.backtest.revalidated_copy(
                update={"benchmark_symbol": None}
            ),
            "validation": cfg_with.validation.revalidated_copy(
                update={
                    "method": ValidationMethod.WALK_FORWARD,
                    "validation_ratio": None,
                    "test_ratio": None,
                }
            ),
        }
    )
    result_without = run_backtest_from_config(data, cfg_without)
    result_without.save(out_dir)
    assert not (out_dir / "benchmark.csv").is_file()
    assert not (out_dir / "holdout_test_returns.csv").is_file()
    assert not (out_dir / "holdout_test_equity.csv").is_file()


def test_stale_walk_forward_csvs_cleaned_up_by_plain_backtest_resave(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)
    assert (exp_dir / "walk_forward_results.csv").is_file()
    assert (exp_dir / "stress_tests.csv").is_file()

    data = pd.concat(
        [
            make_ohlcv("A", geometric_series(300, 0.0004, 0.01, 100.0, seed=1)),
            make_ohlcv("B", geometric_series(300, 0.0002, 0.012, 100.0, seed=2)),
        ],
        ignore_index=True,
    )
    result = run_backtest_from_config(data, config)
    result.save(exp_dir)

    for name in (
        "walk_forward_results.csv",
        "walk_forward_oos_returns.csv",
        "walk_forward_oos_equity.csv",
        "stress_tests.csv",
    ):
        assert not (exp_dir / name).is_file(), f"{name} should have been cleaned up"


def test_walk_forward_windows_rejects_non_positive_test_window() -> None:
    from quantlab.exceptions import InvalidConfigurationError
    from quantlab.validation.splits import walk_forward_windows

    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    with pytest.raises(InvalidConfigurationError, match="test_window"):
        walk_forward_windows(idx, train_window=100, validation_window=10, test_window=0)
    with pytest.raises(InvalidConfigurationError, match="test_window"):
        walk_forward_windows(
            idx, train_window=100, validation_window=10, test_window=-5
        )


def test_walk_forward_windows_rejects_non_positive_train_window() -> None:
    from quantlab.exceptions import InvalidConfigurationError
    from quantlab.validation.splits import walk_forward_windows

    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    with pytest.raises(InvalidConfigurationError, match="train_window"):
        walk_forward_windows(idx, train_window=0, validation_window=10, test_window=10)


def test_bootstrap_rejects_non_positive_iterations() -> None:
    from quantlab.validation.bootstrap import bootstrap_returns

    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    with pytest.raises(ValueError, match="n_iterations"):
        bootstrap_returns(returns, n_iterations=0)
    with pytest.raises(ValueError, match="n_iterations"):
        bootstrap_returns(returns, n_iterations=-1)


def test_monte_carlo_permutation_rejects_non_positive_iterations() -> None:
    """`n_iterations <= 0` must be rejected explicitly, not silently
    produce a degenerate p-value of 1.0 — indistinguishable from
    "definitely not significant" rather than "nothing was tested"."""
    from quantlab.validation.robustness import monte_carlo_permutation

    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    with pytest.raises(ValueError, match="n_iterations"):
        monte_carlo_permutation(returns, n_iterations=0)


def test_holdout_test_equity_keeps_dates_like_holdout_test_returns() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)
    test_returns = result.holdout_test_returns
    test_equity = result.holdout_test_equity
    assert test_returns is not None
    assert test_equity is not None

    assert isinstance(test_returns.index, pd.DatetimeIndex)
    assert isinstance(test_equity.index, pd.DatetimeIndex)
    # The equity curve has one extra (baseline) point before the first return.
    assert test_equity.index[1:].equals(test_returns.index)


def test_walk_forward_windows_rejects_non_positive_validation_window() -> None:
    from quantlab.exceptions import InvalidConfigurationError
    from quantlab.validation.splits import walk_forward_windows

    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    with pytest.raises(InvalidConfigurationError, match="validation_window"):
        walk_forward_windows(
            idx, train_window=500, validation_window=0, test_window=100
        )
    with pytest.raises(InvalidConfigurationError, match="validation_window"):
        walk_forward_windows(
            idx, train_window=500, validation_window=-1, test_window=100
        )


def test_bootstrap_rejects_negative_block_size() -> None:
    """`_resample_indices` treats any `block_size <= 1` as the plain i.i.d.
    bootstrap — a negative value must be rejected explicitly as the
    nonsensical input it is, not silently reinterpreted as `1`."""
    from quantlab.validation.bootstrap import bootstrap_returns

    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_returns(returns, block_size=-3)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_returns(returns, block_size=0)
    # block_size=1 (the documented i.i.d. default) must still work.
    result = bootstrap_returns(returns, block_size=1, n_iterations=10)
    assert len(result.samples) == 10


def test_notebook_walk_forward_cell_passes_risk_free_rate() -> None:
    notebook_namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / "notebook_cells.py")
    )
    notebook_cells = cast(
        list[tuple[str, str]], notebook_namespace["NB_02_MOMENTUM_RESEARCH"]
    )

    oos_metrics_cells = [
        code
        for cell_type, code in notebook_cells
        if cell_type == "code" and "wf.oos_metrics(" in code
    ]
    assert oos_metrics_cells, "expected a wf.oos_metrics(...) cell in the notebook"
    for code in oos_metrics_cells:
        assert "wf.oos_metrics(config.periods_per_year, config.risk_free_rate)" in code
