"""Tests for allocation, constraints and rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.config import RebalanceFrequency
from quantlab.portfolio.allocator import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    SignalProportionalAllocator,
    build_allocator,
)
from quantlab.portfolio.constraints import ConstraintSet
from quantlab.portfolio.position_sizing import gross_exposure
from quantlab.portfolio.rebalancing import (
    apply_rebalancing,
    compute_turnover,
    rebalance_dates,
)


def _signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}, index=index)


# --------------------------------------------------------------------------- #
# Allocators
# --------------------------------------------------------------------------- #
def test_equal_weight_25_percent(synthetic_panel: pd.DataFrame) -> None:
    """Four active longs → 25% each."""
    idx = pd.date_range("2020-01-01", periods=3)
    signals = _signals(idx)
    weights = EqualWeightAllocator().allocate(signals, synthetic_panel)
    assert np.allclose(weights.iloc[0].to_numpy(), 0.25)
    assert gross_exposure(weights).iloc[0] == pytest.approx(1.0)


def test_signal_proportional_normalises(synthetic_panel: pd.DataFrame) -> None:
    idx = pd.date_range("2020-01-01", periods=1)
    signals = pd.DataFrame({"A": [1.0], "B": [0.5], "C": [0.0], "D": [-0.5]}, index=idx)
    weights = SignalProportionalAllocator().allocate(signals, synthetic_panel)
    assert gross_exposure(weights).iloc[0] == pytest.approx(1.0)
    # Sign preserved.
    assert float(weights["A"].loc[idx[0]]) > 0
    assert float(weights["D"].loc[idx[0]]) < 0


def test_inverse_volatility_favours_low_vol(synthetic_panel: pd.DataFrame) -> None:
    # BBB (seed 2) is calmer than CCC (seed 3, sigma 0.02) — expect higher weight.
    idx = synthetic_panel["timestamp"].drop_duplicates().sort_values()
    signals = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB", "CCC"])
    alloc = InverseVolatilityAllocator(volatility_window=63)
    weights = alloc.allocate(signals, synthetic_panel).dropna()
    last = weights.iloc[-1]
    assert gross_exposure(weights).iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert last["BBB"] > last["CCC"]


def test_build_allocator_by_name(synthetic_panel: pd.DataFrame) -> None:
    alloc = build_allocator("equal_weight")
    assert isinstance(alloc, EqualWeightAllocator)


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
def test_constraint_max_weight() -> None:
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=idx)
    out = ConstraintSet(maximum_weight=0.3).apply(weights)
    assert out.abs().to_numpy().max() <= 0.3 + 1e-9


def test_constraint_long_only() -> None:
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=idx)
    out = ConstraintSet(long_only=True).apply(weights)
    assert (out.to_numpy() >= 0).all()
    # Negative weights are clipped to zero, not flipped or redistributed.
    assert float(out["A"].loc[idx[0]]) == pytest.approx(0.5)
    assert float(out["B"].loc[idx[0]]) == pytest.approx(0.0)


def test_constraint_max_positions() -> None:
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.4], "B": [0.3], "C": [0.2], "D": [0.1]}, index=idx)
    out = ConstraintSet(maximum_positions=2).apply(weights)
    assert (out.iloc[0].abs() > 0).sum() == 2
    # The two largest survived.
    assert float(out["A"].loc[idx[0]]) > 0
    assert float(out["B"].loc[idx[0]]) > 0


def test_constraint_gross_cap() -> None:
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=idx)  # gross 2.0
    out = ConstraintSet(maximum_gross_exposure=1.0).apply(weights)
    assert gross_exposure(out).iloc[0] == pytest.approx(1.0)


def test_weights_have_no_nan_or_inf(synthetic_panel: pd.DataFrame) -> None:
    idx = synthetic_panel["timestamp"].drop_duplicates().sort_values()
    signals = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB", "CCC"])
    weights = InverseVolatilityAllocator().allocate(signals, synthetic_panel)
    assert np.isfinite(weights.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Rebalancing
# --------------------------------------------------------------------------- #
def test_rebalance_dates_monthly() -> None:
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    dates = rebalance_dates(idx, RebalanceFrequency.MONTHLY)
    # Jan/Feb/Mar first-of-month (approx) → 3 rebalance dates.
    assert len(dates) == 3
    assert dates[0] == idx[0]


def test_rebalance_dates_weekly_does_not_split_a_non_western_trading_week() -> None:
    """XSAU trades Sunday-Thursday. Grouping by a fixed Monday-Sunday ISO
    week (`.to_period("W")`) would put XSAU's Sunday session in the
    *previous* ISO week from its own Monday-Thursday sessions, splitting one
    real trading week into two rebalances instead of one -- calendar-aware
    grouping must use the calendar's own trading week instead (mirrors the
    equivalent resampler fix, see quantlab.data.resampler._resample_by_session)."""
    idx = pd.DatetimeIndex(
        [
            "2024-01-07",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",  # week 1: Sun-Thu
            "2024-01-14",
            "2024-01-15",
            "2024-01-16",
            "2024-01-17",
            "2024-01-18",  # week 2: Sun-Thu
        ]
    )
    dates = rebalance_dates(idx, RebalanceFrequency.WEEKLY, calendar="XSAU")
    assert list(dates) == [idx[0], idx[5]]


def test_apply_rebalancing_holds_between_dates() -> None:
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    target = pd.DataFrame(np.linspace(0.1, 0.9, 60), index=idx, columns=["A"])
    held = apply_rebalancing(target, RebalanceFrequency.MONTHLY)
    # Within January the held weight is constant and equal to January's
    # first-day target, not just any single value.
    jan = held.loc["2020-01-01":"2020-01-31", "A"]
    assert jan.nunique() == 1
    assert jan.iloc[0] == pytest.approx(target["A"].iloc[0])


def test_turnover_definition() -> None:
    idx = pd.date_range("2020-01-01", periods=3)
    held = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [0.0, 0.0, 0.5]}, index=idx)
    turnover = compute_turnover(held)
    # t0: |0.5|+|0| = 0.5 (entry); t1: 0; t2: |−0.5|+|0.5| = 1.0
    assert turnover.iloc[0] == pytest.approx(0.5)
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert turnover.iloc[2] == pytest.approx(1.0)
