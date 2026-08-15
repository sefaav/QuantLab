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
) -> list[WalkForwardWindow]:
    """Generate contiguous rolling or expanding walk-forward folds.

    Test blocks advance by ``test_window`` and therefore never overlap.
    """
    _validate_datetime_index(index)
    train_window = _validate_window(train_window, name="train_window")
    validation_window = _validate_window(validation_window, name="validation_window")
    test_window = _validate_window(test_window, name="test_window")
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
        start += test_window
    return folds


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
