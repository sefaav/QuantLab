"""Tests for the Streamlit-independent dashboard configuration helpers."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from quantlab.dashboard.state import (
    ProgressPacer,
    build_config_from_inputs,
    estimate_walk_forward_backtest_count,
)
from quantlab.validation.parameter_grid import parse_parameter_grid_values


def _base_inputs(**overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "source": "csv",
        "market_calendar": "XNYS",
        "symbols": ["SPY", "QQQ"],
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


def test_progress_pacer_has_no_estimate_before_any_pace_is_known() -> None:
    pacer = ProgressPacer()
    assert pacer.remaining(0, 10) is None
    pacer.update(0, 0.0)  # the initial done=0 tick establishes no rate yet
    assert pacer.remaining(0, 10) is None


def test_progress_pacer_matches_a_constant_pace() -> None:
    """1 unit per second, 10 units total, 4 done at t=4s -> 6s left."""
    pacer = ProgressPacer()
    for done in range(1, 5):
        pacer.update(done, float(done))
    assert pacer.remaining(4, 10) == pytest.approx(6.0)


def test_progress_pacer_reacts_partially_to_a_single_slow_tick() -> None:
    """A single tick implying a slower pace is nudged towards, not fully
    adopted (``rising_smoothing=0.4``) — a fully-adopting earlier version
    let one noisy tick (parameter-grid candidates genuinely cost different
    amounts, so per-tick rate oscillates with no real trend) ratchet the
    whole estimate up to that single outlier, observed as the displayed
    estimate jumping upward mid-run instead of trending down."""
    pacer = ProgressPacer()
    for done in range(1, 6):
        pacer.update(done, float(done))  # a steady 1s/unit so far
    assert pacer.remaining(5, 10) == pytest.approx(5.0)

    pacer.update(6, 15.0)  # this one unit took 10s, not 1s
    # rate = 0.4 * 10.0 + 0.6 * 1.0 = 4.6 -> 4 units left * 4.6 = 18.4,
    # nowhere near the 40.0 a full jump to the outlier's rate would give.
    assert pacer.remaining(6, 10) == pytest.approx(18.4)


def test_progress_pacer_still_catches_up_to_a_sustained_slowdown() -> None:
    """Several *consecutive* slower ticks (a genuine trend, not one noisy
    outlier) must still converge close to the new pace within a handful of
    ticks — the property that motivated asymmetric smoothing in the first
    place, preserved even after damping the single-tick reaction above."""
    pacer = ProgressPacer()
    for done in range(1, 6):
        pacer.update(done, float(done))  # a steady 1s/unit so far
    for done in range(6, 11):
        pacer.update(done, 5.0 + (done - 5) * 10.0)  # now a sustained 10s/unit
    # rate after 5 consecutive 10s/unit ticks, starting from 1.0:
    # 4.6 -> 6.76 -> 8.056 -> 8.8336 -> 9.30016 (within 7% of the true 10.0
    # pace, from a standing start, in only 5 ticks).
    assert pacer.remaining(10, 12) == pytest.approx(18.60032, rel=1e-3)


def test_progress_pacer_reacts_gradually_to_a_speedup() -> None:
    """A tick implying a faster pace than currently tracked is only
    partially trusted (the default ``falling_smoothing=0.2``) — one
    unusually quick tick shouldn't swing the estimate down only to be
    contradicted by the next."""
    pacer = ProgressPacer()
    for done in range(1, 6):
        pacer.update(done, float(done) * 10.0)  # a steady 10s/unit so far
    assert pacer.remaining(5, 10) == pytest.approx(50.0)

    pacer.update(6, 51.0)  # this one unit took just 1s, not 10s
    # rate = 0.2 * 1.0 + 0.8 * 10.0 = 8.2 -> 4 units left * 8.2 = 32.8.
    assert pacer.remaining(6, 10) == pytest.approx(32.8)


def test_progress_pacer_ignores_non_increasing_updates() -> None:
    """A caller reporting the same or an earlier `done` (e.g. a duplicate
    tick) must not corrupt the tracked pace or divide by zero."""
    pacer = ProgressPacer()
    pacer.update(3, 3.0)
    pacer.update(3, 4.0)  # same done again, time still passed
    pacer.update(2, 5.0)  # done went backwards
    # No exception, and the pace from the one genuine forward step (3 units
    # in 3 seconds) is still what's tracked.
    assert pacer.remaining(3, 10) == pytest.approx(7.0)
