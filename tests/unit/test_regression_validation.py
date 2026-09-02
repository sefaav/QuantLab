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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
                ],
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
    assert "holdout_chronological_metrics" in result.metadata
    assert "sharpe_ratio" in result.metadata["holdout_chronological_metrics"]


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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-11-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "backtest": {"initial_capital": 100_000},
        }
    )
    result = run_backtest_from_config(data, cfg)
    assert "holdout_chronological_metrics" not in result.metadata
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
    assert "Holdout Split" in html
    assert "<td>Test</td>" in html


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
    oos_sharpe = result.metadata["holdout_chronological_metrics"]["sharpe_ratio"]
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
    assert "holdout_chronological_metrics" in holdout_result.metadata
    assert out_of_sample_scope(holdout_result) is None

    plain_result = run_backtest_from_config(data, cfg)
    assert out_of_sample_scope(plain_result) is None


def test_limitations_caveats_the_holdout_blocks_out_of_sample_label() -> None:
    """A holdout split is only "out-of-sample" if strategy/parameter choices
    were genuinely frozen before it was inspected -- QuantLab has no way to
    verify that discipline was followed, so the report must say so rather
    than silently presenting the label as an established fact. Walk-forward
    doesn't need this caveat: its rolling-window discipline is baked into
    the validation loop itself, not a promise from the user."""
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.reporting.research_summary import limitations
    from quantlab.validation.walk_forward import WalkForwardValidator

    holdout_data, holdout_cfg = _holdout_config()
    holdout_result = run_backtest_from_config(holdout_data, holdout_cfg)
    holdout_items = limitations(holdout_result)
    assert any("out-of-sample" in item and "frozen" in item for item in holdout_items)

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    wf_items = limitations(wf.oos_result)
    assert not any("frozen" in item for item in wf_items)

    plain_result = run_backtest_from_config(data, cfg)
    plain_items = limitations(plain_result)
    assert not any("frozen" in item for item in plain_items)


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


def test_walk_forward_oos_result_reports_config_yaml_reflects_everything() -> None:
    """WalkForwardValidator never accepts a custom strategy/allocator/
    execution-model instance the way BacktestEngine.run() does (docs/api.md)
    -- every component is always built from active_config alone, so all
    three config_yaml_reflects_* flags are unconditional facts here, not a
    best-effort verification. Without them present and True, the HTML
    report's footer (which requires all three) would treat every
    walk-forward result as unverified and never claim reproducibility, even
    though a walk-forward result always is."""
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    assert wf.oos_result.metadata["config_yaml_reflects_strategy"] is True
    assert wf.oos_result.metadata["config_yaml_reflects_allocator"] is True
    assert wf.oos_result.metadata["config_yaml_reflects_execution"] is True
    html = wf.oos_result.to_html()
    assert "reproducible from config.yaml given the same code" in html


def test_walk_forward_trade_log_has_no_reason_attribution() -> None:
    """The stitched out-of-sample trade log has no trigger/adjustment/
    position_strategy_origin provenance -- not a crash, not a fabricated
    value, and not the `unknown` safety-net code either (that is reserved
    for the *active* attribution path failing to identify a real cause,
    never for an attribution path that was never run). Each fold reruns
    the pipeline independently with its own warmup/fit; the diagnostic
    frames a single engine run keeps for attribution do not survive the
    cut/restitch across folds, so `build_trade_log` is called here without
    any of the provenance kwargs (see the comment at that call site in
    `walk_forward.py`), leaving the reason columns `None`/`NaT`
    everywhere -- a real architectural fact about walk-forward, not a
    negligence bug."""
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    wf = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=200, validation_window=50, test_window=50
    )
    assert wf.oos_result is not None
    trades = wf.oos_result.trades
    assert len(trades) > 0  # sanity: the fixture must actually produce trades to check

    reason_columns = [
        "trigger_reason_code",
        "trigger_reason_detail_code",
        "trigger_reason_details",
        "adjustment_reason_codes",
        "adjustment_reason_details",
        "position_strategy_origin_code",
        "position_strategy_origin_details",
    ]
    for column in reason_columns:
        assert trades[column].isna().all(), column
    assert trades["position_strategy_origin_timestamp"].isna().all()
    assert (trades["trigger_reason_code"] == "unknown").sum() == 0


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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
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
    assert "<td>Test</td>" in html
    assert "<td>Validation</td>" not in html


def test_holdout_empty_train_block_does_not_crash() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frame = make_ohlcv("AAA", [100.0, 101.0], start="2020-01-01")
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "holdout_empty_train",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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
                "instruments": [
                    {"symbol": "NEW1", "source": "yahoo", "calendar": "XNYS"},
                    {"symbol": "NEW2", "source": "yahoo", "calendar": "XNYS"},
                ],
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


def test_report_command_rejects_stale_walk_forward_when_generator_hash_differs(
    tmp_path: Path,
) -> None:
    """generator_hash, not the narrower code_hash, gates reuse of a saved
    bundle: it also covers cli.py's own orchestration of how the bundle
    gets assembled, which code_hash deliberately excludes."""
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)  # old metadata.json has _FAKE_GENERATOR_HASH

    fake_result = _FakeResult(config)
    fake_result.metadata["generator_hash"] = "a-completely-different-generator-hash"
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_rejects_stale_walk_forward_when_generator_hash_missing(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    fake_result = _FakeResult(config)
    fake_result.metadata["generator_hash"] = None
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_report_command_reuses_walk_forward_when_git_commit_differs(
    tmp_path: Path,
) -> None:
    """generator_hash already hashes current file contents (uncommitted
    changes included), so it alone gives the guarantee needed here --
    a different git_commit (e.g. a rebase, or two checkouts of the exact
    same tree state under different commit objects) must never refuse
    reuse on its own once generator_hash matches."""
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
    assert robustness is not None
    assert fake_result.metadata["walk_forward_oos_metrics"] == {
        "sharpe_ratio": 0.42,
        "cagr": 0.05,
    }


def test_report_command_reuses_walk_forward_when_tree_is_dirty(
    tmp_path: Path,
) -> None:
    """The exact bug this test guards against: `git status`/git_dirty cover
    the *whole* repository, not just the files generator_hash is scoped
    to, so an ordinarily-uncommitted development session (or an unrelated
    change elsewhere in the repo) must never refuse a `report`
    regeneration when config/data/generator_hash/dependencies are all
    still identical -- generator_hash alone already gives that guarantee,
    uncommitted changes included."""
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
    assert robustness is not None
    assert fake_result.metadata["walk_forward_oos_metrics"] == {
        "sharpe_ratio": 0.42,
        "cagr": 0.05,
    }


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
    tmp_path: Path,
) -> None:
    """`load_previous_walk_forward_robustness` reuse depends only on
    config/data/generator_hash/dependency provenance, not on the ambient
    repo's git-dirty state (see test_report_command_reuses_walk_forward_
    when_tree_is_dirty) -- this test needs no git-state pinning to be
    deterministic regardless of whatever the checkout's own working tree
    looks like during a test run."""
    from quantlab.backtesting.result import save_with_walk_forward_reuse
    from quantlab.backtesting.runner import run_backtest_from_config

    frame = make_ohlcv(
        "AAA", geometric_series(300, mu=0.0004, sigma=0.01, s0=100.0, seed=7)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "wf_generate_report_experiment",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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


def test_walk_forward_step_defaults_to_test_window() -> None:
    """Omitting ``step`` must reproduce the original non-overlapping-folds
    behaviour exactly: the same fold count/dates as an explicit
    ``step=test_window``, and the metadata must record that resolved value."""
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    default_run = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=150, validation_window=30, test_window=40
    )
    explicit_run = WalkForwardValidator(cfg).run(
        data,
        parameter_grid={},
        train_window=150,
        validation_window=30,
        test_window=40,
        step=40,
    )
    assert len(default_run.folds) == len(explicit_run.folds)
    assert default_run.oos_returns.equals(explicit_run.oos_returns)
    assert default_run.oos_result is not None
    assert default_run.oos_result.metadata["walk_forward_windows"]["step"] == 40


def test_walk_forward_step_smaller_than_test_window_overlaps_folds() -> None:
    """A step smaller than test_window must produce MORE folds than the
    default (overlapping test blocks) and must not raise the "test blocks
    overlap" error a genuine bug would trigger -- the stitched OOS series
    must still come out with a unique, sorted date index."""
    from quantlab.config import RebalanceFrequency
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    # Daily rebalancing so each fold's first execution date is distinct even
    # with heavily overlapping test blocks -- with the default monthly
    # cadence, a step this much smaller than test_window can make two folds'
    # test blocks share the same first rebalance date, which is a distinct,
    # separately covered failure mode (see
    # test_walk_forward_folds_are_rejected_when_execution_dates_collide).
    cfg = cfg.revalidated_copy(
        update={
            "portfolio": cfg.portfolio.revalidated_copy(
                update={"rebalance_frequency": RebalanceFrequency.DAILY}
            )
        }
    )
    default_run = WalkForwardValidator(cfg).run(
        data, parameter_grid={}, train_window=150, validation_window=30, test_window=40
    )
    overlapping_run = WalkForwardValidator(cfg).run(
        data,
        parameter_grid={},
        train_window=150,
        validation_window=30,
        test_window=40,
        step=20,
    )
    assert len(overlapping_run.folds) > len(default_run.folds)
    assert overlapping_run.oos_returns.index.is_unique
    assert overlapping_run.oos_returns.index.is_monotonic_increasing


def test_walk_forward_folds_are_rejected_when_execution_dates_collide() -> None:
    """A step small enough to overlap test blocks, combined with rebalancing
    too infrequent to distinguish them, can make two folds resolve to the
    SAME first execution date -- silently attributing zero (or the wrong)
    observations to one of them via FoldResult.test_returns slicing. This
    must be rejected outright rather than reported silently."""
    from quantlab.exceptions import InvalidConfigurationError
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    with pytest.raises(InvalidConfigurationError, match="execution date"):
        WalkForwardValidator(cfg).run(
            data,
            parameter_grid={},
            train_window=150,
            validation_window=30,
            test_window=40,
            step=20,
        )


def test_walk_forward_step_larger_than_test_window_is_rejected() -> None:
    """A step larger than test_window would skip observations between folds,
    leaving the stitched OOS curve with gaps that CAGR/annualisation (which
    assume regularly spaced observations) cannot account for -- rejected
    outright rather than silently misreporting elapsed time."""
    from quantlab.exceptions import InvalidConfigurationError
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    with pytest.raises(InvalidConfigurationError, match="step"):
        WalkForwardValidator(cfg).run(
            data,
            parameter_grid={},
            train_window=150,
            validation_window=30,
            test_window=40,
            step=80,
        )


def test_walk_forward_overlapping_step_keeps_the_latest_folds_target() -> None:
    """On a date shared by two overlapping folds' test blocks, the stitched
    series must keep the LATER fold's own target -- verified directly on
    ``_finalize()``'s own dedup step with two fabricated, deliberately
    conflicting target pieces, sidestepping any dependency on a real
    strategy/allocator pipeline actually producing distinguishable values
    for two overlapping folds (not guaranteed for every strategy)."""
    from quantlab.config import RebalanceFrequency
    from quantlab.validation.splits import WalkForwardWindow
    from quantlab.validation.walk_forward import WalkForwardValidator

    data, cfg = _rf_test_setup()
    # Daily rebalancing so every short, synthetic fold test-window below
    # has at least one rebalance date -- irrelevant to what this test
    # actually verifies (the dedup step), just a precondition _finalize()
    # enforces.
    cfg = cfg.revalidated_copy(
        update={
            "portfolio": cfg.portfolio.revalidated_copy(
                update={"rebalance_frequency": RebalanceFrequency.DAILY}
            )
        }
    )
    validator = WalkForwardValidator(cfg)
    idx = pd.bdate_range("2020-06-01", periods=10)
    shared_dates = idx[3:7]
    piece_a = pd.DataFrame(0.1, index=idx[0:7], columns=["AAA", "BBB"])
    piece_b = pd.DataFrame(0.9, index=idx[3:10], columns=["AAA", "BBB"])
    fold_a = WalkForwardWindow(fold=0, train=idx[:1], validation=idx[:1], test=idx[0:7])
    fold_b = WalkForwardWindow(
        fold=1, train=idx[:1], validation=idx[:1], test=idx[3:10]
    )

    stitched = validator._finalize(
        [fold_a, fold_b],
        [{}, {}],
        [1.0, 1.0],
        [piece_a, piece_b],
        data[data["symbol"].isin({"AAA", "BBB"})],
        data,
        cfg,
        252,
        0.0,
        0,
        {},
        7,
        1,
        7,
        4,  # step=4 < test_window=7 -> overlap, dedup path
        True,
        0.0,
    )
    assert stitched.oos_result is not None
    assert stitched.oos_result.target_weights is not None
    for date in shared_dates:
        target = stitched.oos_result.target_weights.at[date, "AAA"]
        assert target == pytest.approx(0.9)


def test_walk_forward_charges_entry_cost_at_the_first_fold_start() -> None:
    """This test is about the OOS-stitching mechanism, not weight drift: it
    checks that per-fold reporting doesn't spuriously double-charge an
    entry cost at a fold boundary the position was actually carried
    through. `model_weight_drift` is pinned `False` here so a genuine,
    correct periodic rebalance-driven cost (buy_and_hold's constant
    target snapping back from organic price drift on a scheduled
    rebalance -- see test_weight_drift.py) can't be mistaken for a
    stitching bug."""
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
                ),
                "portfolio": cfg.portfolio.revalidated_copy(
                    update={"model_weight_drift": False}
                ),
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


@pytest.mark.slow
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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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


@pytest.mark.slow
def test_walk_forward_turnover_cap_chains_across_fold_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv("AAA", geometric_series(400, mu=0.0, sigma=0.0, s0=100.0, seed=1))
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "turnover_chain_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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

    result = validator.run(
        data, parameter_grid={}, train_window=100, validation_window=20, test_window=5
    )

    # With `model_weight_drift` at its default (True), `apply_weight_drift`
    # (inside `run_accounting`) is the SOLE place `maximum_turnover` is
    # enforced -- the decision-level `all_weights` handed to accounting is
    # deliberately left uncapped (see `decision_portfolio_config` in
    # walk_forward.py's own OOS-stitching call site), so the cap must be
    # checked on the final realised turnover, not intercepted upstream.
    assert result.oos_result is not None
    turnover = result.oos_result.turnover
    assert turnover is not None
    assert turnover.max() == pytest.approx(0.1, abs=1e-6), (
        f"maximum_turnover=0.1 must bound every rebalance, including across "
        f"a fold boundary; observed max realised turnover {turnover.max()}"
    )


@pytest.mark.slow
def test_walk_forward_oos_curve_starts_flat_at_the_very_first_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv("AAA", geometric_series(400, mu=0.0, sigma=0.0, s0=100.0, seed=1))
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "first_bar_flat_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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


@pytest.mark.slow
def test_walk_forward_rejects_test_window_shorter_than_a_rebalance_cycle() -> None:
    from quantlab.validation.walk_forward import WalkForwardValidator

    data = make_ohlcv(
        "AAA", geometric_series(400, mu=0.0005, sigma=0.01, s0=100.0, seed=1)
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "too_short_fold_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-11",
                "end_date": "2020-01-20",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {
                "benchmark": {"symbol": "BENCH", "source": "csv", "calendar": "XNYS"}
            },
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
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                ],
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
            "backtest": {
                "benchmark": {"symbol": "BENCH", "source": "csv", "calendar": "XNYS"}
            },
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
                update={
                    "benchmark": {
                        "symbol": "AAA",
                        "source": "csv",
                        "calendar": "XNYS",
                    }
                }
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
            "backtest": cfg_with.backtest.revalidated_copy(update={"benchmark": None}),
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


def test_evaluate_window_never_double_applies_execution_delay_to_rebalance_date() -> (
    None
):
    """`window_weights` (from `_weights_for_window`, via
    `run_backtest_from_config(..., execution_delay=execution_delay)`) is
    itself already `execution_delay`-shifted -- it IS
    `BacktestResult.weights`. Passing `execution_delay` a second time into
    `_rebalance_date_for_run_accounting` would shift the rebalance-date
    flag an EXTRA `execution_delay` rows past where `window_weights`
    itself already sits, misaligning candidate-scoring's schedule-anchor
    detection from the actual execution model it's supposed to score.
    `rebalance_date` must be built with `delay=0`, matching the sibling
    call site in this module's own candidate-scoring loop."""
    from tests.regression_helpers import _rf_test_setup

    import quantlab.validation.walk_forward as wf_mod

    data, cfg = _rf_test_setup()
    cfg = cfg.revalidated_copy(
        update={
            "portfolio": cfg.portfolio.revalidated_copy(
                update={"rebalance_frequency": "daily"}
            )
        }
    )
    lookback_start = pd.Timestamp("2020-01-01")
    window_start = pd.Timestamp("2020-03-01")
    window_end = pd.Timestamp("2020-06-01")

    captured: dict[str, pd.DataFrame] = {}
    orig_run_accounting = wf_mod.run_accounting

    def spy_run_accounting(*args: Any, **kwargs: Any) -> Any:
        captured["rebalance_date"] = kwargs["rebalance_date"].copy()
        captured["tradable"] = kwargs["tradable"]
        return orig_run_accounting(*args, **kwargs)

    wf_mod.run_accounting = spy_run_accounting
    try:
        window_weights, _ = wf_mod._weights_and_returns_for_validation(
            data, cfg, lookback_start, window_start, window_end, execution_delay=2
        )
    finally:
        wf_mod.run_accounting = orig_run_accounting

    expected = wf_mod._rebalance_date_for_run_accounting(
        window_weights,
        cfg.portfolio.rebalance_frequency,
        None,
        captured["tradable"],
        0,
    )
    double_delayed = wf_mod._rebalance_date_for_run_accounting(
        window_weights,
        cfg.portfolio.rebalance_frequency,
        None,
        captured["tradable"],
        2,
    )
    # Sanity: the two would genuinely differ, so this test is not vacuous.
    assert not expected.equals(double_delayed)
    pd.testing.assert_frame_equal(captured["rebalance_date"], expected)


def test_rebalance_date_for_run_accounting_never_true_on_a_closed_row() -> None:
    """Regression test: `compute_executed_weights` is built for *weights*,
    where a closed row correctly repeats the last tradable row's frozen
    value. `_rebalance_date_for_run_accounting` reused it to align a
    boolean flag -- applied to a flag, that same repetition kept it True
    for every row a column stayed closed right after a landing, which
    `apply_weight_drift`'s own documented precondition explicitly forbids
    (it re-anchors ordinary debt to a stale target and fires an
    unscheduled trade the moment the column reopens). A `daily` schedule
    flags every row True, so a landing on the last tradable row before a
    closure is guaranteed, not scenario-dependent."""
    from quantlab.validation.walk_forward import _rebalance_date_for_run_accounting

    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    decision_weights = pd.DataFrame({"A": [0.5] * 8, "B": [0.5] * 8}, index=dates)
    tradable = pd.DataFrame(
        {
            "A": [True, True, True, False, False, True, True, True],
            "B": [True] * 8,
        },
        index=dates,
    )

    result = _rebalance_date_for_run_accounting(
        decision_weights, "daily", None, tradable, 0
    )

    violation = result & ~tradable
    assert not violation.to_numpy().any(), (
        f"rebalance_date is True on a closed row: {violation[violation.any(axis=1)]}"
    )
    # Sanity: the closure itself is genuinely exercised, not vacuously
    # passing because A never lands True around it.
    assert result.loc[dates[2], "A"]  # lands True right before the closure
    assert not result.loc[dates[3], "A"]
    assert not result.loc[dates[4], "A"]
