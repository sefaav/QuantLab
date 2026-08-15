"""Validation shared by public portfolio-construction functions."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from quantlab.exceptions import InvalidConfigurationError


def finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    """Return a finite real value satisfying an optional lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InvalidConfigurationError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidConfigurationError(f"{name} must be a finite number.")
    if minimum is not None:
        invalid = result <= minimum if strict else result < minimum
        if invalid:
            operator = ">" if strict else ">="
            raise InvalidConfigurationError(f"{name} must be {operator} {minimum}.")
    return result


def positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    """Return an integer greater than or equal to ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InvalidConfigurationError(f"{name} must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise InvalidConfigurationError(f"{name} must be an integer >= {minimum}.")
    return result


def boolean(value: object, *, name: str) -> bool:
    """Return a genuine Python or NumPy boolean."""
    if not isinstance(value, (bool, np.bool_)):
        raise InvalidConfigurationError(f"{name} must be a boolean.")
    return bool(value)


def validate_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    allow_missing: bool = False,
    require_datetime_index: bool = False,
) -> pd.DataFrame:
    """Validate a numeric matrix without changing its values or axes."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if not frame.index.is_unique:
        raise InvalidConfigurationError(f"{name} index must be unique.")
    if not frame.index.is_monotonic_increasing:
        raise InvalidConfigurationError(
            f"{name} index must be sorted in increasing order."
        )
    if require_datetime_index and not isinstance(frame.index, pd.DatetimeIndex):
        raise InvalidConfigurationError(f"{name} must use a DatetimeIndex.")
    if not frame.columns.is_unique:
        raise InvalidConfigurationError(f"{name} columns must be unique.")
    if any(
        is_bool_dtype(dtype) or not is_numeric_dtype(dtype) for dtype in frame.dtypes
    ):
        raise InvalidConfigurationError(
            f"{name} must contain only numeric, non-boolean values."
        )
    try:
        values = frame.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(
            f"{name} must contain only numeric values."
        ) from exc
    if np.isinf(values).any():
        raise InvalidConfigurationError(f"{name} must not contain Infinity.")
    if not allow_missing and np.isnan(values).any():
        raise InvalidConfigurationError(f"{name} must not contain missing values.")
    return frame


def validate_datetime_index(index: pd.DatetimeIndex, *, name: str) -> pd.DatetimeIndex:
    """Validate a chronological, unique datetime index."""
    if not isinstance(index, pd.DatetimeIndex):
        raise InvalidConfigurationError(f"{name} must be a DatetimeIndex.")
    if not index.is_unique:
        raise InvalidConfigurationError(f"{name} must be unique.")
    if not index.is_monotonic_increasing:
        raise InvalidConfigurationError(f"{name} must be sorted in increasing order.")
    if index.hasnans:
        raise InvalidConfigurationError(f"{name} must not contain NaT.")
    return index


def require_same_axes(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    reference_name: str,
    other_name: str,
) -> None:
    """Require two matrices to use identical index and columns."""
    if not other.index.equals(reference.index) or not other.columns.equals(
        reference.columns
    ):
        raise InvalidConfigurationError(
            f"{other_name} must have the same index and columns as {reference_name}."
        )
