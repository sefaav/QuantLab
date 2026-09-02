"""Public portfolio-package functions reject malformed inputs with clear,
catchable errors: non-finite weights, invalid types (e.g. bool where a float
is expected), out-of-range configuration values, and axis mismatches.
Constraint objects are also verified immutable once constructed.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import quantlab.portfolio as portfolio
from quantlab.config import PortfolioConfig
from quantlab.exceptions import InvalidConfigurationError
from quantlab.portfolio.allocator import (
    InverseVolatilityAllocator,
    PortfolioAllocator,
    VolatilityTargetingAllocator,
    register_allocator,
)
from quantlab.portfolio.constraints import ConstraintSet
from quantlab.portfolio.position_sizing import (
    gross_exposure,
    inverse_volatility_weights,
    normalize_gross,
    renormalize_within_cap,
)
from quantlab.portfolio.rebalancing import (
    apply_rebalancing,
    cap_turnover,
    compute_turnover,
    rebalance_and_cap_turnover,
    rebalance_dates,
)
from quantlab.portfolio.volatility_targeting import (
    apply_volatility_target,
    estimated_portfolio_volatility,
    volatility_target_leverage,
)


def test_portfolio_package_exports_complete_public_api() -> None:
    expected = {
        "active_positions",
        "estimated_portfolio_volatility",
        "gross_exposure",
        "inverse_volatility_weights",
        "net_exposure",
        "normalize_gross",
        "rebalance_and_cap_turnover",
        "renormalize_within_cap",
        "volatility_target_leverage",
    }
    assert expected <= set(portfolio.__all__)
    assert all(hasattr(portfolio, name) for name in expected)


@pytest.mark.parametrize("target", [-1.0, float("nan"), float("inf"), True])
def test_normalize_gross_rejects_invalid_targets(target: object) -> None:
    weights = pd.DataFrame({"A": [0.6], "B": [0.4]})
    with pytest.raises(InvalidConfigurationError):
        normalize_gross(weights, target_gross=target)  # type: ignore[arg-type]


def test_weight_helpers_reject_non_finite_values() -> None:
    with pytest.raises(InvalidConfigurationError, match="missing"):
        gross_exposure(pd.DataFrame({"A": [np.nan]}))
    with pytest.raises(InvalidConfigurationError, match="Infinity"):
        normalize_gross(pd.DataFrame({"A": [np.inf]}))


def test_inverse_volatility_flattens_unusable_estimates() -> None:
    signals = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]})
    volatility = pd.DataFrame({"A": [0.0], "B": [np.nan], "C": [0.2]})
    result = inverse_volatility_weights(signals, volatility)
    assert result.iloc[0].tolist() == pytest.approx([0.0, 0.0, 5.0])


def test_inverse_volatility_requires_identical_axes() -> None:
    signals = pd.DataFrame({"A": [1.0]})
    volatility = pd.DataFrame({"B": [0.2]})
    with pytest.raises(InvalidConfigurationError, match="same index and columns"):
        inverse_volatility_weights(signals, volatility)


def test_renormalize_within_cap_scales_down_and_caps_input() -> None:
    weights = pd.DataFrame({"A": [0.8], "B": [0.1]})
    result = renormalize_within_cap(weights, target_gross=0.3, cap=0.5)
    assert result.abs().sum(axis=1).iloc[0] == pytest.approx(0.3)
    assert result.abs().max(axis=1).iloc[0] <= 0.5


def test_renormalize_within_cap_reports_iteration_exhaustion() -> None:
    weights = pd.DataFrame({"A": [0.9], "B": [0.09], "C": [0.01]})
    with pytest.raises(InvalidConfigurationError, match="did not converge"):
        renormalize_within_cap(weights, target_gross=1.0, cap=0.4, max_iterations=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_weight": -0.1},
        {"minimum_weight": float("nan")},
        {"maximum_gross_exposure": -1.0},
        {"maximum_positions": 0},
        {"long_only": 1},
        {"minimum_weight": 0.6, "maximum_weight": 0.5},
    ],
)
def test_constraint_set_validates_direct_construction(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(InvalidConfigurationError):
        ConstraintSet(**kwargs)  # type: ignore[arg-type]


def test_constraint_set_is_immutable_and_rejects_missing_weights() -> None:
    constraints = ConstraintSet(maximum_weight=0.5)
    with pytest.raises(FrozenInstanceError):
        constraints.maximum_weight = 0.4  # type: ignore[misc]
    with pytest.raises(InvalidConfigurationError, match="missing"):
        constraints.apply(pd.DataFrame({"A": [np.nan]}))


def test_rebalancing_rejects_unsorted_or_missing_targets() -> None:
    unsorted = pd.DataFrame(
        {"A": [0.1, 0.2]},
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
    )
    with pytest.raises(InvalidConfigurationError, match="sorted"):
        apply_rebalancing(unsorted, "monthly")
    missing = pd.DataFrame(
        {"A": [0.1, np.nan]}, index=pd.date_range("2024-01-01", periods=2)
    )
    with pytest.raises(InvalidConfigurationError, match="missing"):
        apply_rebalancing(missing, "daily")


def test_rebalance_dates_rejects_invalid_frequency_and_duplicate_index() -> None:
    duplicate = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    with pytest.raises(InvalidConfigurationError, match="unique"):
        rebalance_dates(duplicate, "daily")
    with pytest.raises(InvalidConfigurationError, match="Unknown"):
        rebalance_dates(pd.date_range("2024-01-01", periods=2), "yearly")


def test_rebalance_and_cap_turnover_rejects_a_non_boolean_tradable_mask() -> None:
    """A `tradable` column carrying object-dtype values (e.g. the literal
    string 'False') must be rejected explicitly -- a raw
    `.to_numpy(dtype=bool)` conversion would otherwise silently coerce any
    non-empty string, including 'False' itself, to True. Mirrors
    quantlab.execution.orders.validate_execution_frame's identical guard."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    targets = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, index=idx)
    tradable = pd.DataFrame(
        {"A": [True, True, True], "B": ["False", "False", "False"]},
        index=idx,
        dtype=object,
    )
    with pytest.raises(InvalidConfigurationError, match="boolean"):
        rebalance_and_cap_turnover(
            targets, PortfolioConfig(allocator="equal_weight"), tradable=tradable
        )


def test_turnover_functions_reject_non_finite_weights() -> None:
    bad = pd.DataFrame({"A": [0.5, np.nan, 0.0]})
    with pytest.raises(InvalidConfigurationError, match="missing"):
        compute_turnover(bad)
    with pytest.raises(InvalidConfigurationError, match="missing"):
        cap_turnover(bad, maximum_turnover=0.5)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1])
def test_cap_turnover_rejects_invalid_direct_budget(bad: object) -> None:
    targets = pd.DataFrame({"A": [0.5]})
    with pytest.raises(InvalidConfigurationError):
        cap_turnover(targets, maximum_turnover=bad)  # type: ignore[call-overload]


def test_volatility_estimator_excludes_missing_observations() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    weights = pd.DataFrame({"A": [1.0] * 4}, index=index)
    returns = pd.DataFrame({"A": [0.1, np.nan, -0.1, 0.1]}, index=index)
    estimated = estimated_portfolio_volatility(
        weights, returns, window=4, periods_per_year=1
    )
    assert estimated.iloc[2] == pytest.approx(np.std([0.1, -0.1], ddof=1))


def test_volatility_leverage_respects_cap_during_warmup() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    weights = pd.DataFrame({"A": [1.0] * 3}, index=index)
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=index)
    leverage = volatility_target_leverage(
        weights,
        returns,
        target_volatility=0.1,
        window=3,
        maximum_leverage=0.5,
    )
    assert (leverage <= 0.5).all()
    assert leverage.iloc[0] == pytest.approx(0.5)


def test_missing_risk_estimate_after_warmup_gets_zero_leverage() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    weights = pd.DataFrame({"A": [1.0] * 4}, index=index)
    returns = pd.DataFrame({"A": [np.nan] * 4}, index=index)
    leverage = volatility_target_leverage(
        weights, returns, target_volatility=0.1, window=4
    )
    assert leverage.iloc[0] == pytest.approx(1.0)
    assert leverage.iloc[1:].eq(0.0).all()


def test_volatility_target_requires_identical_axes() -> None:
    weights = pd.DataFrame({"A": [1.0]})
    returns = pd.DataFrame({"B": [0.01]})
    with pytest.raises(InvalidConfigurationError, match="same index and columns"):
        apply_volatility_target(weights, returns, target_volatility=0.1, window=2)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InverseVolatilityAllocator(volatility_window=1),
        lambda: InverseVolatilityAllocator(maximum_weight=1.1),
        lambda: InverseVolatilityAllocator(periods_per_year=True),
        lambda: VolatilityTargetingAllocator(target_volatility=-0.1),
        lambda: VolatilityTargetingAllocator(maximum_leverage=0.0),
    ],
)
def test_allocator_constructors_validate_direct_usage(factory: object) -> None:
    with pytest.raises(InvalidConfigurationError):
        factory()  # type: ignore[operator]


def test_allocator_registry_rejects_blank_and_duplicate_names() -> None:
    class CustomAllocator(PortfolioAllocator):
        def allocate(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
            return signals

    with pytest.raises(InvalidConfigurationError, match="non-empty"):
        register_allocator("")
    with pytest.raises(InvalidConfigurationError, match="already registered"):
        register_allocator("equal_weight")(CustomAllocator)
