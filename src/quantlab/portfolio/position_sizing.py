"""Pure helpers for portfolio-weight sizing and exposure measurement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from quantlab.constants import EPSILON
from quantlab.exceptions import InvalidConfigurationError
from quantlab.portfolio._validation import (
    finite_real,
    positive_int,
    require_same_axes,
    validate_frame,
)


def normalize_gross(weights: pd.DataFrame, target_gross: float = 1.0) -> pd.DataFrame:
    """Scale each non-flat row to a non-negative gross-exposure target."""
    validated = validate_frame(weights, name="weights")
    target = finite_real(target_gross, name="target_gross", minimum=0.0)
    gross = validated.abs().sum(axis=1)
    scale = (target / gross.where(gross > EPSILON)).fillna(0.0)
    return validated.mul(scale, axis=0).astype(float)


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Return row-wise gross exposure ``sum(abs(weight))``."""
    validated = validate_frame(weights, name="weights")
    return validated.abs().sum(axis=1)


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Return row-wise signed net exposure ``sum(weight)``."""
    validated = validate_frame(weights, name="weights")
    return validated.sum(axis=1)


def active_positions(weights: pd.DataFrame) -> pd.Series:
    """Return the row-wise number of economically non-zero positions."""
    validated = validate_frame(weights, name="weights")
    return (validated.abs() > EPSILON).sum(axis=1)


def inverse_volatility_weights(
    signals: pd.DataFrame, volatility: pd.DataFrame
) -> pd.DataFrame:
    """Return ``signal / volatility`` with unavailable risk estimates flat.

    A missing, zero or near-zero volatility does not provide a usable sizing
    denominator and therefore receives a zero raw weight.
    """
    validated_signals = validate_frame(signals, name="signals")
    validated_volatility = validate_frame(
        volatility, name="volatility", allow_missing=True
    )
    require_same_axes(
        validated_signals,
        validated_volatility,
        reference_name="signals",
        other_name="volatility",
    )
    denominator = validated_volatility.abs().where(validated_volatility.abs() > EPSILON)
    return (validated_signals / denominator).fillna(0.0).astype(float)


def renormalize_within_cap(
    weights: pd.DataFrame,
    target_gross: float | pd.Series,
    cap: float,
    max_iterations: int | None = None,
) -> pd.DataFrame:
    """Move each row toward ``target_gross`` without exceeding ``cap``.

    Scaling down is uniform. Scaling up uses water-filling over the existing
    non-zero support. If that support cannot reach the requested gross target,
    the maximum feasible exposure is returned without creating new positions.
    """
    validated = validate_frame(weights, name="weights")
    weight_cap = finite_real(cap, name="cap", minimum=0.0, strict=True)
    iterations = (
        max(validated.shape[1], 1)
        if max_iterations is None
        else positive_int(max_iterations, name="max_iterations")
    )
    targets = _gross_targets(validated, target_gross)
    targets_by_label = targets.to_dict()

    def _row(row: pd.Series) -> pd.Series:
        requested = float(targets_by_label[row.name])
        sized = row.astype(float).clip(-weight_cap, weight_cap)
        # Preserve every mathematically non-zero input direction. Very small
        # seed weights may still be the only route to a feasible capped target.
        support = sized.ne(0.0)
        feasible_target = min(requested, int(support.sum()) * weight_cap)
        current = float(sized.abs().sum())
        if current <= EPSILON or feasible_target <= EPSILON:
            return sized * 0.0
        if current > feasible_target:
            return sized * (feasible_target / current)

        for _ in range(iterations):
            current = float(sized.abs().sum())
            if abs(current - feasible_target) <= EPSILON:
                break
            free = support & (sized.abs() < weight_cap - EPSILON)
            if not bool(free.any()):
                break
            fixed_gross = float(sized[~free].abs().sum())
            free_gross = float(sized[free].abs().sum())
            remaining = feasible_target - fixed_gross
            if free_gross <= EPSILON or remaining <= EPSILON:
                break
            sized.loc[free] *= remaining / free_gross
            sized = sized.clip(-weight_cap, weight_cap)

        achieved = float(sized.abs().sum())
        if abs(achieved - feasible_target) > 1e-9:
            raise InvalidConfigurationError(
                "renormalize_within_cap did not converge within max_iterations."
            )
        return sized

    return validated.apply(_row, axis=1)


def _gross_targets(weights: pd.DataFrame, target_gross: float | pd.Series) -> pd.Series:
    if not isinstance(target_gross, pd.Series):
        target = finite_real(target_gross, name="target_gross", minimum=0.0)
        return pd.Series(target, index=weights.index, dtype=float)
    if is_bool_dtype(target_gross.dtype) or not is_numeric_dtype(target_gross.dtype):
        raise InvalidConfigurationError("target_gross must contain numeric values.")
    if not target_gross.index.is_unique or not target_gross.index.equals(weights.index):
        raise InvalidConfigurationError(
            "target_gross must have the same unique index as weights."
        )
    values = target_gross.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise InvalidConfigurationError(
            "target_gross must contain finite, non-negative values."
        )
    return target_gross.astype(float)
