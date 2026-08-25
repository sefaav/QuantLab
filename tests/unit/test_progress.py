"""Tests for the CLI/dashboard-shared progress pacer."""

from __future__ import annotations

import pytest

from quantlab.progress import ProgressPacer


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
