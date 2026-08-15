"""Rebalancing schedules and stateful turnover limits.

Targets are sampled on rebalance dates and represented as constant portfolio
weights between them. This vectorised approximation does not model weight
drift caused by relative asset-price moves between rebalances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import PortfolioConfig, RebalanceFrequency
from quantlab.constants import EPSILON
from quantlab.exceptions import InvalidConfigurationError
from quantlab.portfolio._validation import (
    boolean,
    finite_real,
    validate_datetime_index,
    validate_frame,
)

_PERIOD_ALIAS = {
    RebalanceFrequency.WEEKLY: "W",
    RebalanceFrequency.MONTHLY: "M",
    RebalanceFrequency.QUARTERLY: "Q",
}


def rebalance_dates(
    index: pd.DatetimeIndex, frequency: RebalanceFrequency | str
) -> pd.DatetimeIndex:
    """Return the first available observation in each rebalance period."""
    validated_index = validate_datetime_index(index, name="index")
    rebalance_frequency = _parse_frequency(frequency)
    if len(validated_index) == 0 or rebalance_frequency is RebalanceFrequency.DAILY:
        return validated_index
    if rebalance_frequency is RebalanceFrequency.CUSTOM:
        raise InvalidConfigurationError(
            "rebalance_frequency 'custom' is not implemented because no custom "
            "date schedule is defined in PortfolioConfig."
        )

    grouping_index = (
        validated_index.tz_localize(None)
        if validated_index.tz is not None
        else validated_index
    )
    periods = grouping_index.to_period(_PERIOD_ALIAS[rebalance_frequency])
    is_first = ~pd.Series(periods, index=validated_index).duplicated().to_numpy()
    return pd.DatetimeIndex(validated_index[is_first])


def apply_rebalancing(
    target_weights: pd.DataFrame,
    frequency: RebalanceFrequency | str,
) -> pd.DataFrame:
    """Sample finite targets on rebalance dates and hold them between dates."""
    validated = validate_frame(
        target_weights, name="target_weights", require_datetime_index=True
    )
    dates = rebalance_dates(pd.DatetimeIndex(validated.index), frequency)
    on_dates = validated.loc[validated.index.isin(dates)]
    return on_dates.reindex(validated.index).ffill().astype(float)


def compute_turnover(held_weights: pd.DataFrame) -> pd.Series:
    """Return L1 weight change with an all-cash position before the first row."""
    validated = validate_frame(held_weights, name="held_weights")
    previous = validated.shift(1).fillna(0.0)
    return (validated - previous).abs().sum(axis=1)


def rebalance_and_cap_turnover(
    target_weights: pd.DataFrame, portfolio_config: PortfolioConfig
) -> pd.DataFrame:
    """Apply the stateful schedule and turnover cap over one continuous index.

    Minimum-weight and position-count constraints apply to targets upstream.
    A turnover-limited interpolation may temporarily cross those non-convex
    target-only boundaries while remaining within all exposure constraints.
    """
    held = apply_rebalancing(target_weights, portfolio_config.rebalance_frequency)
    if portfolio_config.maximum_turnover is None:
        return held

    gross_caps = [portfolio_config.maximum_leverage]
    if portfolio_config.maximum_gross_exposure is not None:
        gross_caps.append(portfolio_config.maximum_gross_exposure)
    effective_gross_cap = min(gross_caps)
    dates = rebalance_dates(
        pd.DatetimeIndex(held.index), portfolio_config.rebalance_frequency
    )
    return cap_turnover(
        held,
        portfolio_config.maximum_turnover,
        rebalance_index=dates,
        maximum_weight=portfolio_config.maximum_weight,
        maximum_gross_exposure=effective_gross_cap,
        maximum_net_exposure=portfolio_config.maximum_net_exposure,
        long_only=portfolio_config.long_only,
    )


def cap_turnover(
    held_weights: pd.DataFrame,
    maximum_turnover: float,
    *,
    rebalance_index: pd.DatetimeIndex | None = None,
    maximum_weight: float | None = None,
    maximum_gross_exposure: float | None = None,
    maximum_net_exposure: float | None = None,
    long_only: bool = False,
) -> pd.DataFrame:
    """Partially move toward each scheduled target within an L1 budget.

    The result is a straight-line interpolation from the previous holding to
    an already-compliant target. Per-asset, gross, net and long-only bounds are
    convex, so compliant endpoints keep every intermediate point compliant.
    Cardinality and minimum-position-size constraints remain target-only.
    """
    validated = validate_frame(held_weights, name="held_weights")
    turnover_cap = finite_real(maximum_turnover, name="maximum_turnover", minimum=0.0)
    weight_cap = _optional_non_negative(maximum_weight, name="maximum_weight")
    gross_cap = _optional_non_negative(
        maximum_gross_exposure, name="maximum_gross_exposure"
    )
    net_cap = _optional_non_negative(maximum_net_exposure, name="maximum_net_exposure")
    require_long_only = boolean(long_only, name="long_only")
    is_rebalance_date = _rebalance_mask(validated.index, rebalance_index)

    targets = validated.to_numpy(dtype=float)
    row_count, column_count = targets.shape
    output = np.zeros((row_count, column_count), dtype=float)
    previous = np.zeros(column_count, dtype=float)
    for row_number in range(row_count):
        if not is_rebalance_date[row_number]:
            output[row_number] = previous
            continue
        target = targets[row_number]
        _validate_target_row_compliant(
            target,
            maximum_weight=weight_cap,
            maximum_gross_exposure=gross_cap,
            maximum_net_exposure=net_cap,
            long_only=require_long_only,
            row_label=validated.index[row_number],
        )
        change = target - previous
        requested_turnover = float(np.abs(change).sum())
        if requested_turnover <= turnover_cap + EPSILON:
            current = target
        else:
            current = previous + (turnover_cap / requested_turnover) * change
        output[row_number] = current
        previous = current
    return pd.DataFrame(output, index=validated.index, columns=validated.columns)


def _validate_target_row_compliant(
    target: np.ndarray,
    *,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
    row_label: object,
) -> None:
    """Require each target endpoint to satisfy the convex constraints."""
    violations: list[str] = []
    if long_only and np.any(target < -EPSILON):
        violations.append("long_only")
    if maximum_weight is not None and np.any(np.abs(target) > maximum_weight + EPSILON):
        violations.append("maximum_weight")
    gross = float(np.abs(target).sum())
    if maximum_gross_exposure is not None and gross > maximum_gross_exposure + EPSILON:
        violations.append("maximum_gross_exposure")
    net = float(abs(target.sum()))
    if maximum_net_exposure is not None and net > maximum_net_exposure + EPSILON:
        violations.append("maximum_net_exposure")
    if violations:
        raise InvalidConfigurationError(
            f"Target row at {row_label!r} violates constraints enforced "
            f"upstream: {', '.join(violations)}."
        )


def _rebalance_mask(
    index: pd.Index, rebalance_index: pd.DatetimeIndex | None
) -> np.ndarray:
    if rebalance_index is None:
        return np.ones(len(index), dtype=bool)
    validated_dates = validate_datetime_index(rebalance_index, name="rebalance_index")
    if not isinstance(index, pd.DatetimeIndex):
        raise InvalidConfigurationError(
            "held_weights must use a DatetimeIndex when rebalance_index is given."
        )
    outside = validated_dates.difference(index)
    if len(outside):
        raise InvalidConfigurationError(
            "rebalance_index must be a subset of held_weights.index."
        )
    return np.asarray(index.isin(validated_dates), dtype=bool)


def _optional_non_negative(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return finite_real(value, name=name, minimum=0.0)


def _parse_frequency(value: RebalanceFrequency | str) -> RebalanceFrequency:
    try:
        return RebalanceFrequency(value)
    except (TypeError, ValueError) as exc:
        valid = [frequency.value for frequency in RebalanceFrequency]
        raise InvalidConfigurationError(
            f"Unknown rebalance frequency {value!r}; expected one of {valid}."
        ) from exc
