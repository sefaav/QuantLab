"""Tests for allocation, constraints and rebalancing."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantlab.config import PortfolioConfig, RebalanceFrequency
from quantlab.portfolio.allocator import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    SignalProportionalAllocator,
    build_allocator,
)
from quantlab.portfolio.constraints import ConstraintSet, ConstraintTouch, _mark_touched
from quantlab.portfolio.position_sizing import gross_exposure
from quantlab.portfolio.rebalancing import (
    _rebalance_tradability_aware,
    apply_rebalancing,
    cap_turnover,
    compute_turnover,
    rebalance_and_cap_turnover,
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


# --------------------------------------------------------------------------- #
# Constraint provenance (apply_with_provenance)
# --------------------------------------------------------------------------- #
def test_apply_with_provenance_matches_apply_exactly() -> None:
    """apply() and apply_with_provenance()[0] must be byte-identical --
    provenance tracking is pure instrumentation, never a second, possibly
    diverging computation."""
    idx = pd.date_range("2020-01-01", periods=5)
    rng = np.random.default_rng(0)
    weights = pd.DataFrame(
        rng.normal(scale=0.5, size=(5, 4)), index=idx, columns=["A", "B", "C", "D"]
    )
    constraints = ConstraintSet(
        maximum_weight=0.3,
        minimum_weight=0.02,
        maximum_gross_exposure=1.0,
        maximum_leverage=1.0,
        maximum_net_exposure=0.5,
        maximum_positions=3,
        long_only=False,
    )

    direct = constraints.apply(weights)
    via_provenance, _ = constraints.apply_with_provenance(weights)

    pd.testing.assert_frame_equal(direct, via_provenance)


def test_apply_with_provenance_marks_only_the_constraint_that_fired() -> None:
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=idx)

    _, touches = ConstraintSet(maximum_weight=0.3).apply_with_provenance(weights)

    assert set(touches) == {"maximum_weight"}
    assert bool(touches["maximum_weight"].touched.loc[idx[0], "A"])
    assert touches["maximum_weight"].before.loc[idx[0], "A"] == pytest.approx(0.6)
    assert touches["maximum_weight"].after.loc[idx[0], "A"] == pytest.approx(0.3)


def test_apply_with_provenance_untriggered_constraint_has_an_all_false_mask() -> None:
    """A configured constraint that never actually binds must still appear
    in the provenance dict (so callers can tell "configured but inert"
    from "not configured"), with an all-False touched mask."""
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.2], "B": [0.2]}, index=idx)  # already <= 0.3

    _, touches = ConstraintSet(maximum_weight=0.3).apply_with_provenance(weights)

    assert "maximum_weight" in touches
    assert not touches["maximum_weight"].touched.to_numpy().any()


def test_apply_with_provenance_marks_every_constraint_that_binds_on_the_same_cell() -> (
    None
):
    """maximum_weight trims first, then maximum_gross_exposure rescales the
    whole row further -- both must be recorded as touching the cell, not
    just the last one to run (the user's explicit "multiple causes"
    requirement)."""
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.9], "B": [0.9]}, index=idx)  # gross 1.8

    _, touches = ConstraintSet(
        maximum_weight=0.5, maximum_gross_exposure=0.6
    ).apply_with_provenance(weights)

    assert set(touches) == {"maximum_weight", "maximum_gross_exposure"}
    assert bool(touches["maximum_weight"].touched.loc[idx[0], "A"])
    assert bool(touches["maximum_gross_exposure"].touched.loc[idx[0], "A"])


def test_apply_with_provenance_maximum_weight_direct_vs_redistribution() -> None:
    """A cell directly clipped by maximum_weight (A) vs a cell only
    redimensioned by the water-filling redistribution that follows (B,
    which never itself exceeded the cap) must be distinguishable via
    `direct` -- confirmed against `renormalize_within_cap`'s own two-step
    clip-then-water-fill behaviour (position_sizing.py)."""
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.6], "B": [0.1]}, index=idx)

    _, touches = ConstraintSet(maximum_weight=0.3).apply_with_provenance(weights)
    touch = touches["maximum_weight"]

    assert bool(touch.direct.loc[idx[0], "A"])
    assert bool(touch.touched.loc[idx[0], "B"])
    assert not bool(touch.direct.loc[idx[0], "B"])
    # B was genuinely redistributed upward (never itself over the cap).
    assert cast(float, touch.after.at[idx[0], "B"]) > cast(
        float, touch.before.at[idx[0], "B"]
    )


def test_apply_with_provenance_minimum_weight_direct_vs_redistribution() -> None:
    """A is dropped as dust (direct); B, a genuine survivor well above the
    minimum, is only redimensioned by the redistribution back to the
    pre-drop gross target."""
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.05], "B": [0.3]}, index=idx)

    _, touches = ConstraintSet(minimum_weight=0.1).apply_with_provenance(weights)
    touch = touches["minimum_weight"]

    assert bool(touch.direct.loc[idx[0], "A"])
    assert touch.after.loc[idx[0], "A"] == pytest.approx(0.0)
    assert bool(touch.touched.loc[idx[0], "B"])
    assert not bool(touch.direct.loc[idx[0], "B"])
    assert cast(float, touch.after.at[idx[0], "B"]) > cast(
        float, touch.before.at[idx[0], "B"]
    )


def test_apply_with_provenance_maximum_positions_direct_vs_redistribution() -> None:
    """C (the smallest) is directly dropped by the cardinality cut; A/B
    (the survivors) are only redimensioned by the redistribution back to
    the pre-drop gross target."""
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.5], "B": [0.3], "C": [0.1]}, index=idx)

    _, touches = ConstraintSet(maximum_positions=2).apply_with_provenance(weights)
    touch = touches["maximum_positions"]

    assert bool(touch.direct.loc[idx[0], "C"])
    assert touch.after.loc[idx[0], "C"] == pytest.approx(0.0)
    for survivor in ("A", "B"):
        assert bool(touch.touched.loc[idx[0], survivor])
        assert not bool(touch.direct.loc[idx[0], survivor])
        after_value = cast(float, touch.after.at[idx[0], survivor])
        before_value = cast(float, touch.before.at[idx[0], survivor])
        assert after_value > before_value


def test_apply_with_provenance_direct_equals_touched_for_uniform_rescales() -> None:
    """maximum_gross_exposure/maximum_leverage/maximum_net_exposure/
    long_only are uniform whole-row rescales with no cell-level direct-vs-
    indirect distinction -- `direct` must always equal `touched`."""
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.9], "B": [0.9]}, index=idx)

    _, touches = ConstraintSet(maximum_gross_exposure=1.0).apply_with_provenance(
        weights
    )
    touch = touches["maximum_gross_exposure"]

    pd.testing.assert_frame_equal(touch.direct, touch.touched)


def test_apply_with_provenance_direct_is_always_a_subset_of_touched() -> None:
    """Invariant that must hold for every constraint, every run: a cell
    can never be `direct` without also being `touched`."""
    idx = pd.date_range("2020-01-01", periods=5)
    rng = np.random.default_rng(1)
    weights = pd.DataFrame(
        rng.normal(scale=0.5, size=(5, 4)), index=idx, columns=["A", "B", "C", "D"]
    )
    constraints = ConstraintSet(
        maximum_weight=0.3,
        minimum_weight=0.02,
        maximum_gross_exposure=1.0,
        maximum_leverage=1.0,
        maximum_net_exposure=0.5,
        maximum_positions=3,
        long_only=False,
    )

    _, touches = constraints.apply_with_provenance(weights)

    for touch in touches.values():
        violation = touch.direct & ~touch.touched
        assert not bool(violation.to_numpy().any())


def test_mark_touched_after_only_updates_on_a_pass_that_actually_retouches() -> None:
    """Point 2/9: a cell whose value changes between two passes of THIS
    constraint, but NOT because of this constraint's own operation on the
    intervening pass, must not have that unrelated change attributed to
    it -- `after` only moves on a pass where `before != after` for THIS
    call. `before` stays the very first value across every pass."""
    idx = pd.date_range("2020-01-01", periods=1)
    touched: dict[str, ConstraintTouch] = {}

    # Pass 1: X changes A from 0.9 to 0.7.
    _mark_touched(
        touched,
        "X",
        pd.DataFrame({"A": [0.9]}, index=idx),
        pd.DataFrame({"A": [0.7]}, index=idx),
    )
    assert touched["X"].before.loc[idx[0], "A"] == pytest.approx(0.9)
    assert touched["X"].after.loc[idx[0], "A"] == pytest.approx(0.7)

    # Between passes, an UNRELATED operation moved A to 0.75. Pass 2: X's
    # own before/after this call are equal (0.75 -> 0.75) -- it did not
    # retouch A.
    _mark_touched(
        touched,
        "X",
        pd.DataFrame({"A": [0.75]}, index=idx),
        pd.DataFrame({"A": [0.75]}, index=idx),
    )
    assert touched["X"].after.loc[idx[0], "A"] == pytest.approx(0.7)
    assert touched["X"].before.loc[idx[0], "A"] == pytest.approx(0.9)

    # Pass 3: X retouches A for real (0.75 -> 0.6).
    _mark_touched(
        touched,
        "X",
        pd.DataFrame({"A": [0.75]}, index=idx),
        pd.DataFrame({"A": [0.6]}, index=idx),
    )
    assert touched["X"].before.loc[idx[0], "A"] == pytest.approx(0.9)
    assert touched["X"].after.loc[idx[0], "A"] == pytest.approx(0.6)


def test_maximum_weight_direct_predicate_matches_the_real_clip_exactly() -> None:
    """No epsilon reconstructed for `direct`: at exactly the cap, clip()
    is a no-op (not direct); a hair above it, clip() DOES change the
    value (direct) -- boundary values chosen to straddle EPSILON."""
    cap = 0.3
    idx = pd.date_range("2020-01-01", periods=1)
    for offset, expect_direct in ((0.0, False), (1e-13, False), (1e-8, True)):
        weights = pd.DataFrame({"A": [cap + offset], "B": [0.05]}, index=idx)
        _, touches = ConstraintSet(maximum_weight=cap).apply_with_provenance(weights)
        assert bool(touches["maximum_weight"].direct.loc[idx[0], "A"]) is expect_direct


def test_weights_have_no_nan_or_inf(synthetic_panel: pd.DataFrame) -> None:
    idx = synthetic_panel["timestamp"].drop_duplicates().sort_values()
    signals = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB", "CCC"])
    weights = InverseVolatilityAllocator().allocate(signals, synthetic_panel)
    assert np.isfinite(weights.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Turnover-cap provenance (cell-level, episode-scoped)
# --------------------------------------------------------------------------- #
def test_cap_turnover_provenance_does_not_change_the_computed_weights() -> None:
    """Provenance tracking is pure instrumentation -- requesting it must
    never change the numeric result."""
    idx = pd.date_range("2020-01-01", periods=4)
    held = pd.DataFrame({"A": [1.0, 1.0, 0.3, 0.3]}, index=idx)
    episode_id = pd.DataFrame({"A": [1, 1, 2, 2]}, index=idx)

    without = cap_turnover(held, maximum_turnover=0.4)
    with_provenance, _ = cap_turnover(
        held, maximum_turnover=0.4, episode_id=episode_id, return_provenance=True
    )

    pd.testing.assert_frame_equal(without, with_provenance)


def test_cap_turnover_touched_is_cell_level_not_a_row_broadcast() -> None:
    """Two columns, only one of which actually has a requested delta this
    row -- the untouched one must never be marked, even though the row as
    a whole was turnover-limited."""
    idx = pd.date_range("2020-01-01", periods=1)
    held = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=idx)
    episode_id = pd.DataFrame({"A": [1], "B": [0]}, index=idx)

    _, provenance = cap_turnover(
        held, maximum_turnover=0.4, episode_id=episode_id, return_provenance=True
    )

    assert bool(provenance.turnover_touched.loc[idx[0], "A"])
    assert not bool(provenance.turnover_touched.loc[idx[0], "B"])


def test_cap_turnover_touched_persists_across_the_same_episode() -> None:
    """The exact scenario from the redesign's central example: a target
    held constant across 3 turnover-limited rebalances -- the LAST fill
    (no longer actively binding) must still carry the real turnover_cap
    provenance, as a catch-up of the same still-unresolved episode."""
    idx = pd.date_range("2020-01-01", periods=3)
    held = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    episode_id = pd.DataFrame({"A": [1, 1, 1]}, index=idx)

    output, provenance = cap_turnover(
        held, maximum_turnover=0.4, episode_id=episode_id, return_provenance=True
    )

    assert output["A"].tolist() == pytest.approx([0.4, 0.8, 1.0])
    assert provenance.turnover_actively_limited["A"].tolist() == [True, True, False]
    assert provenance.turnover_touched["A"].tolist() == [True, True, True]


def test_cap_turnover_new_episode_does_not_inherit_debt_even_same_sign() -> None:
    """Core fix: a NEW upstream decision (different episode_id) must NOT
    inherit an old episode's turnover debt, even when the new target
    happens to continue moving in the same direction."""
    idx = pd.date_range("2020-01-01", periods=2)
    held = pd.DataFrame({"A": [1.0, 0.6]}, index=idx)
    episode_id = pd.DataFrame({"A": [1, 2]}, index=idx)

    output, provenance = cap_turnover(
        held, maximum_turnover=0.4, episode_id=episode_id, return_provenance=True
    )

    assert output["A"].tolist() == pytest.approx([0.4, 0.6])
    assert provenance.turnover_actively_limited["A"].tolist() == [True, False]
    assert provenance.turnover_touched["A"].tolist() == [True, False]


def test_cap_turnover_new_episode_abandoning_the_debt() -> None:
    idx = pd.date_range("2020-01-01", periods=2)
    held = pd.DataFrame({"A": [1.0, 0.2]}, index=idx)
    episode_id = pd.DataFrame({"A": [1, 2]}, index=idx)

    output, provenance = cap_turnover(
        held, maximum_turnover=0.4, episode_id=episode_id, return_provenance=True
    )

    assert output["A"].tolist() == pytest.approx([0.4, 0.2])
    assert provenance.turnover_touched["A"].tolist() == [True, False]


def test_cap_turnover_debt_identity_comes_from_episode_id_not_target_value() -> None:
    """The adversarial case demanded explicitly: an IDENTICAL weight path
    (so the numeric result is unaffected), but the second row is tagged
    as a DIFFERENT episode than the first even though the target value
    coincidentally repeats -- the catch-up must not be attributed to the
    first episode's debt. A same-episode control confirms the debt IS
    correctly inherited when it's genuinely the same decision."""
    idx = pd.date_range("2020-01-01", periods=2)
    held = pd.DataFrame({"A": [0.5, 0.5]}, index=idx)

    same_episode = pd.DataFrame({"A": [1, 1]}, index=idx)
    same_output, same_provenance = cap_turnover(
        held, maximum_turnover=0.3, episode_id=same_episode, return_provenance=True
    )
    assert same_provenance.turnover_touched["A"].tolist() == [True, True]

    different_episode = pd.DataFrame({"A": [1, 2]}, index=idx)
    different_output, different_provenance = cap_turnover(
        held,
        maximum_turnover=0.3,
        episode_id=different_episode,
        return_provenance=True,
    )
    assert different_provenance.turnover_touched["A"].tolist() == [True, False]

    # Same weight path in both cases -- provenance never affects the
    # computed numbers.
    pd.testing.assert_series_equal(same_output["A"], different_output["A"])


def test_rebalance_tradability_aware_tradability_touched_on_reopen_catchup() -> None:
    """A target changes while the symbol is closed; no trade happens while
    it stays closed; on reopening, the catch-up trade must carry
    tradability_touched=True even though `tradable` is True again that
    exact day (the current-row boolean alone cannot explain a real
    executed trade -- see _rebalance_tradability_aware's own change==0
    guarantee while ineligible)."""
    idx = pd.date_range("2020-01-01", periods=3)
    target = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)
    tradable = pd.DataFrame({"A": [True, False, True]}, index=idx)
    episode_id = pd.DataFrame({"A": [0, 1, 1]}, index=idx)
    portfolio_config = PortfolioConfig(rebalance_frequency=RebalanceFrequency.DAILY)

    output, provenance = _rebalance_tradability_aware(
        target,
        portfolio_config,
        tradable,
        episode_id=episode_id,
        return_provenance=True,
    )

    # No trade at all while closed (row 1): held at the prior value.
    assert output["A"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert not bool(provenance.tradability_touched.loc[idx[1], "A"])
    assert bool(provenance.tradability_touched.loc[idx[2], "A"])


def test_rebalance_and_cap_turnover_returns_provenance_when_requested() -> None:
    """The public dispatcher (used by engine.py) must forward episode_id/
    return_provenance correctly on both the tradable and non-tradable
    paths, and on the no-turnover-cap-configured path."""
    idx = pd.date_range("2020-01-01", periods=2)
    target = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
    episode_id = pd.DataFrame({"A": [1, 1]}, index=idx)
    portfolio_config = PortfolioConfig(
        rebalance_frequency=RebalanceFrequency.DAILY, maximum_turnover=0.4
    )

    output, provenance = rebalance_and_cap_turnover(
        target, portfolio_config, episode_id=episode_id, return_provenance=True
    )
    assert output["A"].tolist() == pytest.approx([0.4, 0.8])
    assert provenance.turnover_touched["A"].tolist() == [True, True]

    # No turnover cap configured at all -- provenance must still come
    # back, all-False (nothing can ever be turnover-limited).
    no_cap_config = PortfolioConfig(rebalance_frequency=RebalanceFrequency.DAILY)
    no_cap_output, no_cap_provenance = rebalance_and_cap_turnover(
        target, no_cap_config, episode_id=episode_id, return_provenance=True
    )
    assert no_cap_output["A"].tolist() == pytest.approx([1.0, 1.0])
    assert not bool(no_cap_provenance.turnover_touched.to_numpy().any())


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
