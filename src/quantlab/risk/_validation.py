"""Shared input validation for risk analytics."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd


def finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def positive_int(value: object, *, name: str) -> int:
    """Return a strictly positive integer, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def nonnegative_int(value: object, *, name: str) -> int:
    """Return a non-negative integer, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def boolean(value: object, *, name: str) -> bool:
    """Return a genuine boolean value."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def numeric_series(
    value: object,
    *,
    name: str,
    allow_nan: bool,
    require_unique_index: bool = False,
    require_sorted_index: bool = False,
) -> pd.Series:
    """Validate and return a float copy of a numeric Series."""
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if (
        not pd.api.types.is_numeric_dtype(value.dtype)
        or pd.api.types.is_bool_dtype(value.dtype)
        or pd.api.types.is_complex_dtype(value.dtype)
    ):
        raise TypeError(f"{name} must contain real numeric values.")
    if require_unique_index and not value.index.is_unique:
        raise ValueError(f"{name} must have a unique index.")
    if require_sorted_index and not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order.")

    result = value.astype(float).copy()
    values = result.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinite values.")
    if not allow_nan and np.isnan(values).any():
        raise ValueError(f"{name} must not contain missing values.")
    return result


def equity_series(
    value: object,
    *,
    name: str = "equity",
    require_nonnegative: bool = True,
    prevent_resurrection: bool = True,
) -> pd.Series:
    """Validate a chronological equity curve."""
    result = numeric_series(
        value,
        name=name,
        allow_nan=False,
        require_unique_index=True,
        require_sorted_index=True,
    )
    if result.empty:
        return result
    if result.iloc[0] <= 0.0:
        raise ValueError(f"{name} must start with a strictly positive value.")
    if require_nonnegative and (result < 0.0).any():
        raise ValueError(f"{name} must not contain negative values.")
    if prevent_resurrection:
        ruined = result <= 0.0
        if (
            ruined.any()
            and (result.iloc[int(np.flatnonzero(ruined.to_numpy())[0]) :] > 0).any()
        ):
            raise ValueError(
                f"{name} cannot become positive again after reaching zero."
            )
    return result


def numeric_frame(value: object, *, name: str) -> pd.DataFrame:
    """Validate and return a finite float copy of a numeric DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if not value.index.is_unique:
        raise ValueError(f"{name} must have a unique index.")
    if not value.columns.is_unique:
        raise ValueError(f"{name} must have unique columns.")
    if not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order.")
    invalid_columns = [
        column
        for column in value.columns
        if not pd.api.types.is_numeric_dtype(value[column].dtype)
        or pd.api.types.is_bool_dtype(value[column].dtype)
        or pd.api.types.is_complex_dtype(value[column].dtype)
    ]
    if invalid_columns:
        raise TypeError(f"{name} must contain only real numeric columns.")
    result = value.astype(float).copy()
    if result.isna().to_numpy().any():
        raise ValueError(f"{name} must not contain missing values.")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} must not contain infinite values.")
    return result
