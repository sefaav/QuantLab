"""Rebalancing schedules and stateful turnover limits.

Targets are sampled on rebalance dates and represented as constant portfolio
weights between them. This vectorised approximation does not model weight
drift caused by relative asset-price moves between rebalances.

Timing convention: every function in this module produces *decided* weights,
not executed ones -- including a row where a closed symbol's pending target
first resolves after reopening (see :func:`rebalance_and_cap_turnover`'s
``tradable`` parameter). ``held_weights[t]`` is "what the strategy decided
using information available through t," never "what the portfolio actually
holds at t." :mod:`quantlab.backtesting.accounting` always applies one further
(tradability-respecting) shift before any weight can affect returns, turnover
or costs -- uniformly, with no special case for a reopening row. A target
that first appears in ``held_weights`` on a symbol's reopening day therefore
does not affect the portfolio until that symbol's *next* tradable session,
exactly as an ordinary scheduled rebalance decided on day T only takes effect
on day T+1 -- never on day T itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from quantlab.config import PortfolioConfig, RebalanceFrequency
from quantlab.constants import EPSILON
from quantlab.data.calendar import is_247, session_labels, weekly_bucket_start
from quantlab.exceptions import BacktestError, InvalidConfigurationError
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
    index: pd.DatetimeIndex,
    frequency: RebalanceFrequency | str,
    *,
    calendar: str | None = None,
) -> pd.DatetimeIndex:
    """Return the first available observation in each rebalance period.

    ``calendar``, when given (and not ``"24/7"``), groups by each
    timestamp's real trading-session date instead of its raw UTC value --
    otherwise a calendar whose local session crosses UTC midnight (e.g.
    XASX, UTC+10/+11) could have one session's own bars split across two
    different weekly/monthly periods right at a period boundary, and
    every bar frequency shares the same calendar-aware grouping. Omit it
    (the default) for a portfolio with no single shared calendar -- a raw
    UTC boundary is the same documented approximation already used
    elsewhere for a mixed-calendar universe.
    """
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
    session_calendar = (
        calendar if calendar is not None and not is_247(calendar) else None
    )
    if session_calendar is not None:
        grouping_index = pd.DatetimeIndex(
            session_labels(session_calendar, pd.Series(grouping_index)).to_numpy()
        )
    is_weekly = rebalance_frequency is RebalanceFrequency.WEEKLY
    if session_calendar is not None and is_weekly:
        # A calendar's trading week isn't always Monday-Sunday (e.g. XSAU
        # trades Sunday-Thursday) -- `.to_period("W")` always bins by the
        # fixed ISO week, which would split such a week's own sessions
        # across two different periods right at its own boundary (its
        # Sunday session falls in the *previous* ISO week from its
        # Monday-Thursday sessions). `weekly_bucket_start` groups by the
        # calendar's own trading week instead, mirroring the resampler's
        # identical fix (see quantlab.data.resampler._resample_by_session).
        group_keys: pd.Index = pd.DatetimeIndex(
            [
                weekly_bucket_start(timestamp, calendar=session_calendar)
                for timestamp in grouping_index
            ]
        )
    else:
        group_keys = grouping_index.to_period(_PERIOD_ALIAS[rebalance_frequency])
    is_first = ~pd.Series(group_keys, index=validated_index).duplicated().to_numpy()
    return pd.DatetimeIndex(validated_index[is_first])


def apply_rebalancing(
    target_weights: pd.DataFrame,
    frequency: RebalanceFrequency | str,
    *,
    calendar: str | None = None,
) -> pd.DataFrame:
    """Sample finite targets on rebalance dates and hold them between dates."""
    validated = validate_frame(
        target_weights, name="target_weights", require_datetime_index=True
    )
    dates = rebalance_dates(
        pd.DatetimeIndex(validated.index), frequency, calendar=calendar
    )
    on_dates = validated.loc[validated.index.isin(dates)]
    return on_dates.reindex(validated.index).ffill().astype(float)


def compute_turnover(held_weights: pd.DataFrame) -> pd.Series:
    """Return L1 weight change with an all-cash position before the first row."""
    validated = validate_frame(held_weights, name="held_weights")
    previous = validated.shift(1).fillna(0.0)
    return (validated - previous).abs().sum(axis=1)


def rebalance_and_cap_turnover(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    *,
    tradable: pd.DataFrame | None = None,
    calendar: str | None = None,
) -> pd.DataFrame:
    """Apply the stateful schedule and turnover cap over one continuous index.

    Minimum-weight and position-count constraints apply to targets upstream.
    A turnover-limited interpolation may temporarily cross those non-convex
    target-only boundaries while remaining within all exposure constraints.

    Args:
        target_weights: Raw target weights, one row per date.
        portfolio_config: Rebalance frequency, turnover cap and exposure caps.
        tradable: Optional ``dates x symbols`` bool frame (True = that symbol
            is tradable that date). When given, a closed symbol never trades
            that date — its rebalance is deferred as a pending target and
            caught up (even off-schedule, possibly over several sessions if
            ``maximum_turnover`` limits the rate) as soon as it reopens. Note
            that a pending target resolving in the returned frame on a
            symbol's reopening day is still only a *decision* dated that
            day — see the module docstring's timing convention — it does not
            reach the accounting layer's executed weights, turnover or costs
            until that symbol's next tradable session. When ``None`` (the
            default), behaviour is exactly the pre-existing schedule/
            turnover-cap path below, unchanged.
        calendar: Optional shared calendar name, forwarded to
            :func:`rebalance_dates` so a weekly/monthly schedule groups by
            real trading sessions instead of raw UTC dates. Only meaningful
            when every instrument shares one calendar; omit for a mixed
            universe (see :func:`rebalance_dates`).
    """
    if tradable is not None:
        return _rebalance_tradability_aware(
            target_weights, portfolio_config, tradable, calendar=calendar
        )

    held = apply_rebalancing(
        target_weights, portfolio_config.rebalance_frequency, calendar=calendar
    )
    if portfolio_config.maximum_turnover is None:
        return held

    gross_caps = [portfolio_config.maximum_leverage]
    if portfolio_config.maximum_gross_exposure is not None:
        gross_caps.append(portfolio_config.maximum_gross_exposure)
    effective_gross_cap = min(gross_caps)
    dates = rebalance_dates(
        pd.DatetimeIndex(held.index),
        portfolio_config.rebalance_frequency,
        calendar=calendar,
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


def _compliance_violations(
    weights: np.ndarray,
    *,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
) -> list[str]:
    """Return the names of every convex constraint ``weights`` violates."""
    violations: list[str] = []
    if long_only and np.any(weights < -EPSILON):
        violations.append("long_only")
    if maximum_weight is not None and np.any(
        np.abs(weights) > maximum_weight + EPSILON
    ):
        violations.append("maximum_weight")
    gross = float(np.abs(weights).sum())
    if maximum_gross_exposure is not None and gross > maximum_gross_exposure + EPSILON:
        violations.append("maximum_gross_exposure")
    net = float(abs(weights.sum()))
    if maximum_net_exposure is not None and net > maximum_net_exposure + EPSILON:
        violations.append("maximum_net_exposure")
    return violations


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
    violations = _compliance_violations(
        target,
        maximum_weight=maximum_weight,
        maximum_gross_exposure=maximum_gross_exposure,
        maximum_net_exposure=maximum_net_exposure,
        long_only=long_only,
    )
    if violations:
        raise InvalidConfigurationError(
            f"Target row at {row_label!r} violates constraints enforced "
            f"upstream: {', '.join(violations)}."
        )


def _max_feasible_fraction(
    previous: np.ndarray,
    change: np.ndarray,
    *,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
    upper_bound: float,
) -> float:
    """Return the largest feasible fraction of ``change`` to apply.

    Finds ``f`` in ``[0, upper_bound]`` keeping ``previous + f * change``
    compliant. ``f=0`` (no movement) is always feasible because ``previous`` is itself
    compliant by construction (every row this module produces is). Each
    constraint is convex and affine in ``f`` along this segment, so its
    feasible set is a prefix ``[0, f_max]`` — bisection is well-founded and
    converges to the exact boundary.

    This is needed because freezing some columns (a closed symbol) while
    others move toward their own target changes the direction of travel from
    the straight ``previous -> target`` line that :func:`cap_turnover` relies
    on for its convexity argument — the new endpoint is not guaranteed
    feasible even though both ``previous`` and the full ``target`` are.
    """

    def compliant(fraction: float) -> bool:
        candidate = previous + fraction * change
        return not _compliance_violations(
            candidate,
            maximum_weight=maximum_weight,
            maximum_gross_exposure=maximum_gross_exposure,
            maximum_net_exposure=maximum_net_exposure,
            long_only=long_only,
        )

    if compliant(upper_bound):
        return upper_bound
    lo, hi = 0.0, upper_bound
    for _ in range(60):  # ~1e-18 relative precision, far below EPSILON
        mid = (lo + hi) / 2
        if compliant(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _assert_holdings_compliant(
    current: np.ndarray,
    *,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_net_exposure: float | None,
    long_only: bool,
    row_label: object,
) -> None:
    """Defensive check: every row this module actually outputs must comply.

    :func:`_max_feasible_fraction` already guarantees this by construction —
    a violation here would mean a bug in that search, not a bad input — but
    a silent violation would be an expensive, hard-to-diagnose out-of-mandate
    position, so this fails loudly rather than trusting the invariant blindly.
    """
    violations = _compliance_violations(
        current,
        maximum_weight=maximum_weight,
        maximum_gross_exposure=maximum_gross_exposure,
        maximum_net_exposure=maximum_net_exposure,
        long_only=long_only,
    )
    if violations:
        raise BacktestError(
            f"Tradability-aware rebalancing produced holdings at {row_label!r} "
            f"that violate: {', '.join(violations)}. This indicates a bug in "
            "the rebalancing algorithm, not a configuration problem."
        )


def _rebalance_tradability_aware(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    tradable: pd.DataFrame,
    *,
    calendar: str | None = None,
) -> pd.DataFrame:
    """Rebalance while respecting per-symbol tradability.

    A symbol that is closed on a rebalance date never trades that date; its
    new target becomes a pending debt that is retried on every subsequent
    date it is tradable (not just the next scheduled rebalance) until fully
    executed — even if that takes several sessions under a turnover cap. A
    symbol that is always tradable is completely unaffected: its cadence
    (rebalance-date-only catch-up) is byte-identical to :func:`cap_turnover`.
    """
    validated = validate_frame(
        target_weights, name="target_weights", require_datetime_index=True
    )
    dates = rebalance_dates(
        pd.DatetimeIndex(validated.index),
        portfolio_config.rebalance_frequency,
        calendar=calendar,
    )
    is_rebalance_date = np.asarray(validated.index.isin(dates), dtype=bool)
    turnover_cap = (
        finite_real(
            portfolio_config.maximum_turnover, name="maximum_turnover", minimum=0.0
        )
        if portfolio_config.maximum_turnover is not None
        else float("inf")
    )
    gross_caps = [portfolio_config.maximum_leverage]
    if portfolio_config.maximum_gross_exposure is not None:
        gross_caps.append(portfolio_config.maximum_gross_exposure)
    gross_cap = min(gross_caps)
    weight_cap = _optional_non_negative(
        portfolio_config.maximum_weight, name="maximum_weight"
    )
    net_cap = _optional_non_negative(
        portfolio_config.maximum_net_exposure, name="maximum_net_exposure"
    )
    long_only = boolean(portfolio_config.long_only, name="long_only")

    # Exact same *set* of dates and symbols, no missing values -- a mismatched
    # set always means an upstream wiring bug (`tradable` is only ever built
    # internally from the same data as `target_weights`, never user input),
    # so it must raise loudly rather than silently default an unrecognized
    # cell to "tradable" and risk trading a symbol that should have stayed
    # closed. Column *order* alone is not a mismatch, though: a caller may
    # build `tradable` from a declared symbol list while `target_weights`
    # comes from an alphabetically-pivoted price matrix -- reindexing onto
    # `validated`'s order is always safe once the sets are confirmed equal.
    if set(tradable.index) != set(validated.index) or set(tradable.columns) != set(
        validated.columns
    ):
        raise InvalidConfigurationError(
            "tradable must have the same dates and symbols as target_weights."
        )
    tradable = tradable.reindex(index=validated.index, columns=validated.columns)
    if tradable.isna().to_numpy().any():
        raise InvalidConfigurationError("tradable must not contain missing values.")
    non_bool_columns = [
        column for column, dtype in tradable.dtypes.items() if not is_bool_dtype(dtype)
    ]
    if non_bool_columns:
        raise InvalidConfigurationError(
            f"tradable must contain only boolean values; column(s) "
            f"{non_bool_columns} are not boolean dtype (e.g. a string "
            "'False' would otherwise silently coerce to True)."
        )
    tradable_np = tradable.to_numpy(dtype=bool)
    targets = validated.to_numpy(dtype=float)
    row_count, column_count = targets.shape
    output = np.zeros((row_count, column_count), dtype=float)
    previous = np.zeros(column_count, dtype=float)
    pending_target = np.zeros(column_count, dtype=float)
    pending_due_to_closure = np.zeros(column_count, dtype=bool)

    for row_number in range(row_count):
        row_tradable = tradable_np[row_number]
        if is_rebalance_date[row_number]:
            target_row = targets[row_number]
            _validate_target_row_compliant(
                target_row,
                maximum_weight=weight_cap,
                maximum_gross_exposure=gross_cap,
                maximum_net_exposure=net_cap,
                long_only=long_only,
                row_label=validated.index[row_number],
            )
            newly_blocked = (~row_tradable) & (np.abs(target_row - previous) > EPSILON)
            pending_target = target_row
            pending_due_to_closure = pending_due_to_closure | newly_blocked

        eligible = row_tradable & (
            is_rebalance_date[row_number] | pending_due_to_closure
        )
        change = np.where(eligible, pending_target - previous, 0.0)
        requested_turnover = float(np.abs(change).sum())
        fraction_from_turnover = (
            1.0
            if requested_turnover <= turnover_cap + EPSILON
            else turnover_cap / requested_turnover
        )
        fraction = _max_feasible_fraction(
            previous,
            change,
            maximum_weight=weight_cap,
            maximum_gross_exposure=gross_cap,
            maximum_net_exposure=net_cap,
            long_only=long_only,
            upper_bound=fraction_from_turnover,
        )
        current = previous + fraction * change
        _assert_holdings_compliant(
            current,
            maximum_weight=weight_cap,
            maximum_gross_exposure=gross_cap,
            maximum_net_exposure=net_cap,
            long_only=long_only,
            row_label=validated.index[row_number],
        )

        unresolved = np.abs(current - pending_target) > EPSILON
        # Only possible when some column is currently untradable (see
        # _max_feasible_fraction docstring): with every symbol tradable, this
        # row's change/target endpoints are both convex-compliant, so the
        # bisection above never binds below the turnover-derived fraction.
        compliance_limited = fraction < fraction_from_turnover - EPSILON
        pending_due_to_closure = (
            pending_due_to_closure | (eligible & unresolved & compliance_limited)
        ) & unresolved

        output[row_number] = current
        previous = current
    return pd.DataFrame(output, index=validated.index, columns=validated.columns)


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
