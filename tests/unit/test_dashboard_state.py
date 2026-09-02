"""Tests for the Streamlit-independent dashboard configuration helpers."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from quantlab.dashboard.state import (
    build_config_from_inputs,
    estimate_walk_forward_backtest_count,
    run_dashboard_backtest_with_data,
)
from quantlab.validation.parameter_grid import parse_parameter_grid_values


def _base_inputs(**overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "instruments": [
            {"symbol": "SPY", "source": "csv", "calendar": "XNYS"},
            {"symbol": "QQQ", "source": "csv", "calendar": "XNYS"},
        ],
        "start_date": datetime.date(2019, 1, 1),
        "end_date": datetime.date(2020, 1, 1),
        "strategy_name": "buy_and_hold",
        "strategy_parameters": {},
        "allocator": "equal_weight",
        "rebalance_frequency": "monthly",
    }
    inputs.update(overrides)
    return inputs


def test_build_config_from_inputs_defaults_to_holdout_validation() -> None:
    config = build_config_from_inputs(_base_inputs())
    assert config.validation.method == "holdout"


def test_build_config_from_inputs_builds_walk_forward_validation_block() -> None:
    config = build_config_from_inputs(
        _base_inputs(
            strategy_name="time_series_momentum",
            strategy_parameters={"lookback_period": 120, "skip_period": 5},
            validation_method="walk_forward",
            train_window=300,
            validation_window=60,
            test_window=60,
            expanding=False,
            optimization_metric="sortino",
            parameter_grid={"lookback_period": [60, 120]},
        )
    )
    assert config.validation.method == "walk_forward"
    assert config.validation.train_window == 300
    assert config.validation.validation_window == 60
    assert config.validation.test_window == 60
    assert config.validation.expanding is False
    assert config.validation.optimization_metric == "sortino"
    assert config.validation.parameter_grid == {"lookback_period": [60, 120]}


def test_build_config_from_inputs_walk_forward_without_grid_is_none() -> None:
    """An empty/missing grid must become `None` (fall back to the strategy's
    default grid), not an empty dict rejected as "no candidate values"."""
    config = build_config_from_inputs(
        _base_inputs(
            validation_method="walk_forward",
            train_window=300,
            validation_window=60,
            test_window=60,
            parameter_grid={},
        )
    )
    assert config.validation.parameter_grid is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("60, 120, 252", [60, 120, 252]),
        ("0.1, 0.25, 0.5", [0.1, 0.25, 0.5]),
        ("true, false", [True, False]),
        ("binary, continuous", ["binary", "continuous"]),
        (" 60 ,, 120 ", [60, 120]),
        ("", []),
    ],
)
def test_parse_parameter_grid_values(raw: str, expected: list[object]) -> None:
    assert parse_parameter_grid_values(raw) == expected


def test_estimate_walk_forward_backtest_count_matches_fold_times_combinations() -> None:
    from quantlab.validation.splits import walk_forward_windows

    start = datetime.date(2018, 1, 1)
    end = datetime.date(2021, 12, 31)
    index = pd.DatetimeIndex(pd.bdate_range(start, end))
    windows = walk_forward_windows(index, 300, 120, 120, expanding=True)

    estimate = estimate_walk_forward_backtest_count(
        start_date=start,
        end_date=end,
        is_247_market=False,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        parameter_grid={"lookback_period": [60, 120], "skip_period": [0, 21]},
    )

    assert estimate == len(windows) * (2 * 2 + 1)


def test_estimate_walk_forward_backtest_count_empty_grid_counts_one_combination() -> (
    None
):
    estimate = estimate_walk_forward_backtest_count(
        start_date=datetime.date(2018, 1, 1),
        end_date=datetime.date(2021, 12, 31),
        is_247_market=False,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        parameter_grid={},
    )
    assert estimate > 0


def test_estimate_walk_forward_backtest_count_is_zero_for_too_short_a_range() -> None:
    estimate = estimate_walk_forward_backtest_count(
        start_date=datetime.date(2020, 1, 1),
        end_date=datetime.date(2020, 1, 10),
        is_247_market=False,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        parameter_grid={},
    )
    assert estimate == 0


def test_checkpoint_path_matches_the_cli_convention() -> None:
    """Same GENERATED_REPORTS_DIR / experiment_name / '.checkpoint_<technique>.pkl'
    convention the CLI uses (src/quantlab/cli.py), so a dashboard run and a
    same-named CLI run can resume each other's progress."""
    from quantlab.constants import GENERATED_REPORTS_DIR
    from quantlab.dashboard.state import _checkpoint_path

    config = build_config_from_inputs(_base_inputs(experiment_name="my_experiment"))
    path = _checkpoint_path(config, "walk_forward")
    assert path == (
        GENERATED_REPORTS_DIR / "my_experiment" / ".checkpoint_walk_forward.pkl"
    )


def test_run_dashboard_walk_forward_passes_its_checkpoint_path_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring check: run_dashboard_walk_forward must forward the same path
    _checkpoint_path() computes to WalkForwardValidator.run(), not silently
    drop it — the actual resume mechanism is exercised end-to-end elsewhere
    (tests/unit/test_validation.py, tests/integration/test_cli.py)."""
    from types import SimpleNamespace

    import quantlab.dashboard.state as state_module
    from quantlab.dashboard.state import _checkpoint_path, run_dashboard_walk_forward
    from quantlab.validation.walk_forward import WalkForwardResult, WalkForwardValidator

    config = build_config_from_inputs(_base_inputs(experiment_name="wiring_check"))
    monkeypatch.setattr(
        state_module.DataLoader,
        "load",
        lambda self, cfg: (pd.DataFrame(), SimpleNamespace(warnings=[])),
    )
    captured: dict[str, object] = {}

    def _fake_run(self, data, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return WalkForwardResult()

    monkeypatch.setattr(WalkForwardValidator, "run", _fake_run)

    run_dashboard_walk_forward(config)

    assert captured["checkpoint_path"] == _checkpoint_path(config, "walk_forward")


def test_run_dashboard_backtest_with_data_returns_the_exact_frame_it_ran_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The returned frame must be the SAME object the backtest itself ran
    on (identity, not just equality) -- a Strategy Explorer results
    diagnostic reusing anything else (e.g. a fresh, independent reload)
    could silently observe different data than the displayed result for a
    remote source that changed, or a cache that refreshed, between the two
    loads."""
    from types import SimpleNamespace

    import quantlab.dashboard.state as state_module

    config = build_config_from_inputs(_base_inputs(experiment_name="identity_check"))
    the_frame = pd.DataFrame({"marker": [1, 2, 3]})
    monkeypatch.setattr(
        state_module.DataLoader,
        "load",
        lambda self, cfg: (the_frame, SimpleNamespace(warnings=["w"])),
    )
    captured: dict[str, object] = {}

    def fake_run_backtest_from_config(data, cfg, *, data_quality_report=None):  # type: ignore[no-untyped-def]
        captured["data_seen_by_backtest"] = data
        return "fake-result"

    monkeypatch.setattr(
        state_module, "run_backtest_from_config", fake_run_backtest_from_config
    )

    result, warnings, returned_data = run_dashboard_backtest_with_data(config)

    assert result == "fake-result"
    assert warnings == ["w"]
    assert returned_data is the_frame
    assert captured["data_seen_by_backtest"] is the_frame
