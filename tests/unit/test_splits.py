"""Tests for chronological splits and walk-forward windows."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from quantlab.exceptions import InvalidConfigurationError
from quantlab.validation.splits import (
    chronological_split,
    formation_test_windows,
    walk_forward_windows,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_chronological_split_is_contiguous_and_ordered() -> None:
    idx = _index(1000)
    split = chronological_split(idx, 0.6, 0.2, 0.2)
    assert len(split.train) == 600
    assert len(split.validation) == 200
    assert len(split.test) == 200
    # No leakage: train entirely before validation, before test.
    assert split.train.max() < split.validation.min()
    assert split.validation.max() < split.test.min()


def test_split_covers_all_observations() -> None:
    idx = _index(1000)
    split = chronological_split(idx, 0.6, 0.2, 0.2)
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(idx)


def test_split_rejects_bad_ratios() -> None:
    with pytest.raises(InvalidConfigurationError):
        chronological_split(_index(100), 0.5, 0.3, 0.3)


def test_split_rejects_unsorted_index() -> None:
    idx = _index(100)[::-1]
    with pytest.raises(InvalidConfigurationError):
        chronological_split(idx, 0.6, 0.2, 0.2)


def test_walk_forward_windows_no_overlap_in_test() -> None:
    idx = _index(2000)
    windows = walk_forward_windows(
        idx, train_window=1000, validation_window=252, test_window=126, expanding=True
    )
    assert len(windows) >= 1
    # Test blocks must be disjoint and advancing.
    prev_end = None
    for w in windows:
        assert w.validation.max() < w.test.min()  # test strictly after validation
        if prev_end is not None:
            assert w.test.min() > prev_end
        prev_end = w.test.max()


def test_walk_forward_expanding_train_grows() -> None:
    idx = _index(2000)
    windows = walk_forward_windows(
        idx, train_window=800, validation_window=200, test_window=100, expanding=True
    )
    if len(windows) >= 2:
        assert len(windows[1].train) > len(windows[0].train)


def test_walk_forward_rolling_train_constant() -> None:
    idx = _index(2000)
    windows = walk_forward_windows(
        idx, train_window=800, validation_window=200, test_window=100, expanding=False
    )
    if len(windows) >= 2:
        assert len(windows[1].train) == len(windows[0].train)


def test_split_as_dict_matches_the_named_attributes() -> None:
    idx = _index(100)
    split = chronological_split(idx, 0.6, 0.2, 0.2)
    as_dict = split.as_dict()
    assert as_dict["train"].equals(split.train)
    assert as_dict["validation"].equals(split.validation)
    assert as_dict["test"].equals(split.test)


def test_split_rejects_zero_train_or_test_ratio() -> None:
    with pytest.raises(InvalidConfigurationError, match="greater than zero"):
        chronological_split(_index(100), 0.0, 0.5, 0.5)
    with pytest.raises(InvalidConfigurationError, match="greater than zero"):
        chronological_split(_index(100), 0.5, 0.5, 0.0)


def test_split_rejects_validation_ratio_that_rounds_to_an_empty_block() -> None:
    # 10 observations x 0.05 validation_ratio truncates to zero rows, even
    # though the ratio itself is positive and the three ratios sum to 1.0.
    with pytest.raises(InvalidConfigurationError, match="empty validation block"):
        chronological_split(_index(10), 0.8, 0.05, 0.15)


def test_walk_forward_windows_rejects_a_non_integer_window() -> None:
    idx = _index(50)
    with pytest.raises(InvalidConfigurationError, match="integer"):
        walk_forward_windows(idx, "10", 5, 5)  # type: ignore[arg-type]


def test_formation_test_windows_never_overlap_formation_and_test() -> None:
    dates = pd.bdate_range("2010-01-01", periods=1000)
    windows = formation_test_windows(dates, formation_window=252, test_window=126)
    assert len(windows) == (1000 - 252) // 126
    for window in windows:
        assert window.formation.stop == window.test.start
        assert window.formation_end < window.test_start
        assert window.test.stop - window.test.start == 126
    # Test windows are contiguous and non-overlapping.
    for previous, current in itertools.pairwise(windows):
        assert previous.test.stop == current.test.start


def test_formation_test_windows_fold_ids_are_sequential() -> None:
    dates = pd.bdate_range("2010-01-01", periods=600)
    windows = formation_test_windows(dates, formation_window=126, test_window=63)
    assert [w.fold for w in windows] == list(range(len(windows)))


def test_formation_test_windows_custom_step_overlaps_test_windows() -> None:
    dates = pd.bdate_range("2010-01-01", periods=600)
    windows = formation_test_windows(
        dates, formation_window=126, test_window=63, step=21
    )
    assert len(windows) > (600 - 126) // 63
    for previous, current in itertools.pairwise(windows):
        assert current.test.start - previous.test.start == 21


def test_formation_test_windows_step_exceeding_test_window_leaves_a_gap() -> None:
    """Unlike walk_forward_windows (whose folds stitch into one continuous
    OOS curve, so a step > test_window would leave a CAGR-breaking gap),
    formation_test_windows evaluates each fold independently -- a step
    larger than test_window is a legitimate sparser design (e.g.
    re-forming quarterly with a short test window), not an error."""
    dates = pd.bdate_range("2010-01-01", periods=600)
    windows = formation_test_windows(
        dates, formation_window=126, test_window=63, step=100
    )
    assert len(windows) >= 2
    for previous, current in itertools.pairwise(windows):
        assert current.test.start - previous.test.start == 100
        # A genuine gap between consecutive TEST windows: dates in between
        # are evaluated by neither fold (though they still feed the next
        # fold's own formation window as historical data).
        assert current.test.start > previous.test.stop


def test_formation_test_windows_rejects_insufficient_history() -> None:
    dates = pd.bdate_range("2010-01-01", periods=100)
    with pytest.raises(InvalidConfigurationError, match="Not enough history"):
        formation_test_windows(dates, formation_window=252, test_window=126)


def test_formation_test_windows_exact_boundary_count() -> None:
    """The exact-minimum history (formation_window + test_window
    observations) must produce exactly one fold, never zero and never
    raise; one observation short must raise instead of silently
    producing zero folds."""
    formation_window, test_window = 60, 20
    exact = pd.bdate_range("2010-01-01", periods=formation_window + test_window)
    windows = formation_test_windows(
        exact, formation_window=formation_window, test_window=test_window
    )
    assert len(windows) == 1
    assert windows[0].test_end == exact[-1]

    one_short = exact[:-1]
    with pytest.raises(InvalidConfigurationError, match="Not enough history"):
        formation_test_windows(
            one_short, formation_window=formation_window, test_window=test_window
        )


def test_formation_test_windows_start_end_timestamps_match_index() -> None:
    dates = pd.bdate_range("2010-01-01", periods=400)
    windows = formation_test_windows(dates, formation_window=100, test_window=50)
    first = windows[0]
    assert first.formation_start == dates[0]
    assert first.formation_end == dates[99]
    assert first.test_start == dates[100]
    assert first.test_end == dates[149]
