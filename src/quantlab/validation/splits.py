"""Chronological train, validation and test partitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from quantlab.exceptions import InvalidConfigurationError


@dataclass(frozen=True)
class ChronologicalSplit:
    """A chronological train/validation/test partition of a datetime index."""

    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def as_dict(self) -> dict[str, pd.DatetimeIndex]:
        """Return the three blocks keyed by name."""
        return {"train": self.train, "validation": self.validation, "test": self.test}


def chronological_split(
    index: pd.DatetimeIndex,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> ChronologicalSplit:
    """Split a canonical datetime index into contiguous chronological blocks."""
    _validate_datetime_index(index)
    ratios = {
        "train_ratio": _validate_ratio(train_ratio, name="train_ratio"),
        "validation_ratio": _validate_ratio(validation_ratio, name="validation_ratio"),
        "test_ratio": _validate_ratio(test_ratio, name="test_ratio"),
    }
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise InvalidConfigurationError(
            f"Split ratios must sum to exactly 1.0, got {total:.12g}."
        )
    if ratios["train_ratio"] == 0.0 or ratios["test_ratio"] == 0.0:
        raise InvalidConfigurationError(
            "train_ratio and test_ratio must both be greater than zero."
        )

    n = len(index)
    n_train = int(n * ratios["train_ratio"])
    n_validation = int(n * ratios["validation_ratio"])
    n_test = n - n_train - n_validation
    if n_train == 0 or n_test == 0:
        raise InvalidConfigurationError(
            "Split ratios leave an empty train or test block for the available "
            f"{n} observations."
        )
    if ratios["validation_ratio"] > 0.0 and n_validation == 0:
        raise InvalidConfigurationError(
            "validation_ratio is positive but rounds to an empty validation block "
            f"for the available {n} observations."
        )

    return ChronologicalSplit(
        train=index[:n_train],
        validation=index[n_train : n_train + n_validation],
        test=index[n_train + n_validation :],
    )


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward fold with train, validation and test date blocks."""

    fold: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward_windows(
    index: pd.DatetimeIndex,
    train_window: int,
    validation_window: int,
    test_window: int,
    *,
    expanding: bool = True,
    step: int | None = None,
) -> list[WalkForwardWindow]:
    """Generate contiguous rolling or expanding walk-forward folds.

    Each fold's train window starts ``step`` periods after the previous
    fold's (default: ``step = test_window``, the original "test blocks
    advance by test_window and never overlap" behaviour). A ``step``
    smaller than ``test_window`` makes consecutive test blocks overlap
    (denser evaluation). ``step`` must not exceed ``test_window``: a larger
    step would skip observations between folds, leaving the stitched OOS
    curve with gaps that CAGR/annualisation (which assume regularly spaced
    observations) would silently misread as a shorter elapsed time.
    """
    _validate_datetime_index(index)
    train_window = _validate_window(train_window, name="train_window")
    validation_window = _validate_window(validation_window, name="validation_window")
    test_window = _validate_window(test_window, name="test_window")
    step = test_window if step is None else _validate_window(step, name="step")
    if step > test_window:
        raise InvalidConfigurationError(
            f"step ({step}) must not exceed test_window ({test_window}) -- a "
            "larger step would skip observations between folds, which "
            "CAGR/annualisation cannot account for."
        )
    if not isinstance(expanding, (bool, np.bool_)):
        raise InvalidConfigurationError("expanding must be a boolean.")
    expanding = bool(expanding)

    n = len(index)
    folds: list[WalkForwardWindow] = []
    fold = 0
    start = 0
    while True:
        train_end = start + train_window
        validation_end = train_end + validation_window
        test_end = validation_end + test_window
        if test_end > n:
            break
        train_start = 0 if expanding else start
        folds.append(
            WalkForwardWindow(
                fold=fold,
                train=index[train_start:train_end],
                validation=index[train_end:validation_end],
                test=index[validation_end:test_end],
            )
        )
        fold += 1
        start += step
    return folds


@dataclass(frozen=True)
class FormationTestWindow:
    """One formation-then-test fold: positional slices plus start/end timestamps.

    Unlike :class:`WalkForwardWindow` (train/validation/test), this has no
    validation block -- for a Gatev, Goetzmann & Rouwenhorst (2006, "Pairs
    Trading: Performance of a Relative-Value Arbitrage Rule")-style design
    where quantities are estimated once on the formation window and a
    fixed rule is then applied out of sample, a middle validation block
    has no role and would only eat into the test window for no purpose.

    Deliberately carries both ``formation``/``test`` (integer-position
    ``slice`` objects, for ``.iloc[]``-style indexing directly against a
    prices matrix -- the natural operation for a study evaluating many
    entities over the same window) and the four ``*_start``/``*_end``
    timestamps (for display/logging without re-deriving them from
    ``index[slice]`` at every call site). Unlike :class:`WalkForwardWindow`,
    which stores each block as its own :class:`pandas.DatetimeIndex` and
    has no positional-slice form at all, unifying these into one
    representation here would drop one capability call sites need -- the
    four timestamps are computed once, in this function alone, from the
    same ``index`` the slices came from, so there is no risk of the two
    forms drifting apart independently.
    """

    fold: int
    formation: slice
    test: slice
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def formation_test_windows(
    index: pd.DatetimeIndex,
    formation_window: int,
    test_window: int,
    *,
    step: int | None = None,
) -> list[FormationTestWindow]:
    """Generate contiguous formation-then-test folds with no validation block.

    Fold ``i``'s formation window is ``index[start-formation_window:start]``
    immediately followed by its test window ``index[start:start+test_window]``
    -- formation always strictly precedes test, with no gap and no overlap.
    ``step`` (default ``test_window``, giving contiguous, non-overlapping
    test windows; smaller overlaps them for denser evaluation) may exceed
    ``test_window`` here, UNLIKE :func:`walk_forward_windows`'s own
    identically-named parameter: each fold here is evaluated on its own
    (see :class:`FormationTestWindow`'s own docstring on the Gatev/
    Goetzmann/Rouwenhorst design), never stitched into one continuous
    curve, so a gap between consecutive test windows -- e.g. re-forming
    quarterly with a one-month test window -- is a legitimate sparser
    design, not a CAGR/annualisation hazard the way it is for a stitched
    OOS curve.

    Args:
        index: Sorted, unique datetime index covering both formation and
            test data.
        formation_window: Number of leading periods used to estimate
            whatever a caller freezes before trading (e.g. a hedge ratio,
            a spread's mean/std, an entry threshold).
        test_window: Number of periods immediately following formation
            that are then evaluated/traded with those frozen estimates.
        step: Periods between consecutive folds' formation starts.
            Defaults to ``test_window``.

    Returns:
        At least one fold. Raises :class:`InvalidConfigurationError` if
        ``index`` has too few observations for even one.
    """
    _validate_datetime_index(index)
    formation_window = _validate_window(formation_window, name="formation_window")
    test_window = _validate_window(test_window, name="test_window")
    step = test_window if step is None else _validate_window(step, name="step")

    n = len(index)
    windows: list[FormationTestWindow] = []
    start = formation_window
    while start + test_window <= n:
        windows.append(
            FormationTestWindow(
                fold=len(windows),
                formation=slice(start - formation_window, start),
                test=slice(start, start + test_window),
                formation_start=index[start - formation_window],
                formation_end=index[start - 1],
                test_start=index[start],
                test_end=index[start + test_window - 1],
            )
        )
        start += step
    if not windows:
        raise InvalidConfigurationError(
            "Not enough history for a single formation/test fold: need at "
            f"least {formation_window + test_window} observations, got {n}."
        )
    return windows


def _validate_datetime_index(index: object) -> None:
    """Validate the chronology required by every split helper."""
    if not isinstance(index, pd.DatetimeIndex):
        raise InvalidConfigurationError("index must be a pandas DatetimeIndex.")
    if not index.is_unique:
        raise InvalidConfigurationError("index must not contain duplicate dates.")
    if not index.is_monotonic_increasing:
        raise InvalidConfigurationError("index must be sorted in increasing order.")


def _validate_ratio(value: object, *, name: str) -> float:
    """Return a finite ratio in the closed interval [0, 1]."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InvalidConfigurationError(f"{name} must be a real number.")
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise InvalidConfigurationError(f"{name} must be between 0 and 1.")
    return ratio


def _validate_window(value: object, *, name: str) -> int:
    """Return a strictly positive window length."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InvalidConfigurationError(f"{name} must be an integer.")
    window = int(value)
    if window <= 0:
        raise InvalidConfigurationError(f"{name} must be greater than zero.")
    return window
