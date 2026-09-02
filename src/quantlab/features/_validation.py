"""Shared validation for public feature functions."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TypeVar

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    """Return an integer greater than or equal to ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return result


def non_negative_int(value: object, *, name: str) -> int:
    """Return a non-negative integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return result


def finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    """Return a finite real number satisfying an optional lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if minimum is not None:
        invalid = result <= minimum if strict else result < minimum
        if invalid:
            operator = ">" if strict else ">="
            raise ValueError(f"{name} must be {operator} {minimum}, got {value!r}.")
    return result


def boolean(value: object, *, name: str) -> bool:
    """Return a genuine Python or NumPy boolean."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return bool(value)


def choice(value: object, *, name: str, options: frozenset[str]) -> str:
    """Return a string restricted to a fixed set of accepted values."""
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {sorted(options)}, got {value!r}.")
    return value


def numeric_pandas(
    data: PandasT,
    *,
    name: str,
    strictly_positive: bool = False,
) -> PandasT:
    """Validate a numeric Series/DataFrame while preserving missing values."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a pandas Series or DataFrame.")
    if not data.index.is_unique:
        raise ValueError(f"{name} index must not contain duplicate labels.")
    if not data.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order.")
    dtypes = [data.dtype] if isinstance(data, pd.Series) else list(data.dtypes)
    if any(is_bool_dtype(dtype) or not is_numeric_dtype(dtype) for dtype in dtypes):
        raise TypeError(f"{name} must contain only numeric, non-boolean values.")
    if isinstance(data, pd.DataFrame) and not data.columns.is_unique:
        raise ValueError(f"{name} columns must not contain duplicate labels.")
    try:
        values = data.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numeric values.") from exc
    present = ~np.isnan(values)
    if not np.isfinite(values[present]).all():
        raise ValueError(f"{name} must not contain Infinity.")
    if strictly_positive and (values[present] <= 0.0).any():
        raise ValueError(f"{name} must contain only positive values where defined.")
    return data


def same_axes(reference: PandasT, *others: PandasT, names: tuple[str, ...]) -> None:
    """Require several pandas objects to have identical axes and container type."""
    if len(others) != len(names):
        raise ValueError("An axis-validation name is required for every object.")
    for other, name in zip(others, names, strict=True):
        if type(other) is not type(reference) or not other.index.equals(
            reference.index
        ):
            raise ValueError(
                f"{name} must have the same type and index as the reference."
            )
        if isinstance(reference, pd.DataFrame) and not other.columns.equals(
            reference.columns
        ):
            raise ValueError(f"{name} must have the same columns as the reference.")
