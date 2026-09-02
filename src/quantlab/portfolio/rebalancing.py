"""Rebalancing schedules and stateful turnover limits.

Targets are sampled on rebalance dates and represented as constant portfolio
weights between them -- this module's own output is a decision-timeline
step function. Real, price-driven weight drift between genuine trades is
modeled separately and downstream, on the EXECUTED timeline, by
:func:`quantlab.backtesting.accounting.apply_weight_drift` (gated by
``PortfolioConfig.model_weight_drift``); this module's own output is
identical regardless of whether that gate is on or off.

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

from dataclasses import dataclass
from typing import Literal, overload

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


@dataclass(frozen=True)
class TurnoverProvenance:
    """Cell-level, real provenance from a turnover-capped rebalance.

    ``turnover_actively_limited`` is True where the turnover budget itself
    bound this row's move for that cell. ``turnover_touched`` is the
    broader, *episode-scoped* provenance -- also True on a later row that
    is still catching up a debt created by an earlier turnover-limited
    move toward the SAME upstream decision (see ``episode_id`` on
    :func:`cap_turnover`/:func:`rebalance_and_cap_turnover`), even when
    that later row is no longer itself actively binding. ``tradability_
    touched``/``tradability_compliance_limited`` are always all-``False``
    for :func:`cap_turnover` (no tradability concept); populated for
    :func:`_rebalance_tradability_aware`.
    """

    turnover_actively_limited: pd.DataFrame
    turnover_touched: pd.DataFrame
    tradability_touched: pd.DataFrame
    tradability_compliance_limited: pd.DataFrame


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


@overload
def rebalance_and_cap_turnover(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    *,
    tradable: pd.DataFrame | None = None,
    calendar: str | None = None,
    episode_id: pd.DataFrame | None = None,
    return_provenance: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def rebalance_and_cap_turnover(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    *,
    tradable: pd.DataFrame | None = None,
    calendar: str | None = None,
    episode_id: pd.DataFrame | None = None,
    return_provenance: Literal[True],
) -> tuple[pd.DataFrame, TurnoverProvenance]: ...


def rebalance_and_cap_turnover(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    *,
    tradable: pd.DataFrame | None = None,
    calendar: str | None = None,
    episode_id: pd.DataFrame | None = None,
    return_provenance: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, TurnoverProvenance]:
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
        episode_id: Forwarded to :func:`cap_turnover`/the tradability-aware
            path -- see :func:`cap_turnover`'s own docstring.
        return_provenance: Forwarded the same way -- see :func:`cap_turnover`.
    """
    if tradable is not None:
        # Branched (rather than forwarding the plain `bool` variable
        # directly) so mypy can select the correct @overload -- a
        # non-literal bool cannot match either `Literal[True]`/
        # `Literal[False]` overload variant.
        if return_provenance:
            return _rebalance_tradability_aware(
                target_weights,
                portfolio_config,
                tradable,
                calendar=calendar,
                episode_id=episode_id,
                return_provenance=True,
            )
        return _rebalance_tradability_aware(
            target_weights,
            portfolio_config,
            tradable,
            calendar=calendar,
            episode_id=episode_id,
            return_provenance=False,
        )

    held = apply_rebalancing(
        target_weights, portfolio_config.rebalance_frequency, calendar=calendar
    )
    if portfolio_config.maximum_turnover is None:
        if not return_provenance:
            return held
        return held, _no_provenance(held)

    gross_caps = [portfolio_config.maximum_leverage]
    if portfolio_config.maximum_gross_exposure is not None:
        gross_caps.append(portfolio_config.maximum_gross_exposure)
    effective_gross_cap = min(gross_caps)
    dates = rebalance_dates(
        pd.DatetimeIndex(held.index),
        portfolio_config.rebalance_frequency,
        calendar=calendar,
    )
    if return_provenance:
        return cap_turnover(
            held,
            portfolio_config.maximum_turnover,
            rebalance_index=dates,
            maximum_weight=portfolio_config.maximum_weight,
            maximum_gross_exposure=effective_gross_cap,
            maximum_net_exposure=portfolio_config.maximum_net_exposure,
            long_only=portfolio_config.long_only,
            episode_id=episode_id,
            return_provenance=True,
        )
    return cap_turnover(
        held,
        portfolio_config.maximum_turnover,
        rebalance_index=dates,
        maximum_weight=portfolio_config.maximum_weight,
        maximum_gross_exposure=effective_gross_cap,
        maximum_net_exposure=portfolio_config.maximum_net_exposure,
        long_only=portfolio_config.long_only,
        episode_id=episode_id,
        return_provenance=False,
    )


@overload
def cap_turnover(
    held_weights: pd.DataFrame,
    maximum_turnover: float,
    *,
    rebalance_index: pd.DatetimeIndex | None = None,
    maximum_weight: float | None = None,
    maximum_gross_exposure: float | None = None,
    maximum_net_exposure: float | None = None,
    long_only: bool = False,
    episode_id: pd.DataFrame | None = None,
    return_provenance: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def cap_turnover(
    held_weights: pd.DataFrame,
    maximum_turnover: float,
    *,
    rebalance_index: pd.DatetimeIndex | None = None,
    maximum_weight: float | None = None,
    maximum_gross_exposure: float | None = None,
    maximum_net_exposure: float | None = None,
    long_only: bool = False,
    episode_id: pd.DataFrame | None = None,
    return_provenance: Literal[True],
) -> tuple[pd.DataFrame, TurnoverProvenance]: ...


def cap_turnover(
    held_weights: pd.DataFrame,
    maximum_turnover: float,
    *,
    rebalance_index: pd.DatetimeIndex | None = None,
    maximum_weight: float | None = None,
    maximum_gross_exposure: float | None = None,
    maximum_net_exposure: float | None = None,
    long_only: bool = False,
    episode_id: pd.DataFrame | None = None,
    return_provenance: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, TurnoverProvenance]:
    """Partially move toward each scheduled target within an L1 budget.

    The result is a straight-line interpolation from the previous holding to
    an already-compliant target. Per-asset, gross, net and long-only bounds are
    convex, so compliant endpoints keep every intermediate point compliant.
    Cardinality and minimum-position-size constraints remain target-only.

    Args:
        held_weights: Scheduled targets to move toward, one row per date
            (constant between rebalance dates -- see :func:`apply_rebalancing`).
        maximum_turnover: Maximum L1 weight change allowed on any single row
            (a rebalance whose full target exceeds this lands partially and
            keeps closing the gap over subsequent rows).
        rebalance_index: Dates on which a new target is actually chased;
            every other date holds the previous value. ``None`` treats
            every date as a rebalance date.
        maximum_weight: Optional per-asset cap, enforced on the target
            (upstream) and therefore on every intermediate point.
        maximum_gross_exposure: Optional gross exposure cap, same convexity
            argument.
        maximum_net_exposure: Optional net exposure cap, same convexity
            argument.
        long_only: When True, rejects a target with any negative weight.
        episode_id: Required when ``return_provenance`` is True. A ``dates
            x symbols`` integer frame identifying, per cell, which upstream
            decision produced the target currently being chased -- two
            cells sharing the same value are the SAME still-unresolved
            decision, even if the target happens to repeat a prior numeric
            value; a different value always means a genuinely different
            upstream decision. Built by the caller (see ``engine.py``),
            never reconstructed here from the target's own numeric value
            (which cannot tell two decisions with the same target apart).
        return_provenance: When True, also return a :class:`TurnoverProvenance`
            with real, cell-level attribution of which trades were caused
            (directly or as an episode-scoped catch-up) by the turnover
            cap. Does not affect the computed weights in any way -- the
            numeric branch below is identical whether or not this is set.
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

    # Always bound with cheap placeholders, even though they are only ever
    # read (below and by the caller) under `if return_provenance:` -- the
    # same unchanged flag that guards their real assignment just below. A
    # static analyzer cannot follow "guarded by the same boolean flag"
    # across the loop in between; this changes no behaviour.
    episode_values = np.empty((row_count, column_count), dtype=float)
    pending_episode_id = np.full(column_count, -1.0)
    actively_limited_out = np.zeros((row_count, column_count), dtype=bool)
    touched_out = np.zeros((row_count, column_count), dtype=bool)
    if return_provenance:
        if episode_id is None:
            raise BacktestError(
                "episode_id is required when return_provenance is True."
            )
        episode_values = (
            validate_frame(episode_id, name="episode_id")
            .reindex(index=validated.index, columns=validated.columns)
            .to_numpy(dtype=float)
        )

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
        if return_provenance:
            row_actively_limited = requested_turnover > turnover_cap + EPSILON
            changed_this_row = np.abs(change) > EPSILON
            generation = episode_values[row_number]
            debt_still_relevant = (pending_episode_id != -1.0) & (
                pending_episode_id == generation
            )
            actively_limited_out[row_number] = changed_this_row & row_actively_limited
            touched_out[row_number] = changed_this_row & (
                row_actively_limited | debt_still_relevant
            )
            still_outstanding = np.abs(current - target) > EPSILON
            pending_episode_id = np.where(still_outstanding, generation, -1.0)
        output[row_number] = current
        previous = current
    result = pd.DataFrame(output, index=validated.index, columns=validated.columns)
    if not return_provenance:
        return result
    provenance = TurnoverProvenance(
        turnover_actively_limited=pd.DataFrame(
            actively_limited_out, index=validated.index, columns=validated.columns
        ),
        turnover_touched=pd.DataFrame(
            touched_out, index=validated.index, columns=validated.columns
        ),
        tradability_touched=pd.DataFrame(
            False, index=validated.index, columns=validated.columns
        ),
        tradability_compliance_limited=pd.DataFrame(
            False, index=validated.index, columns=validated.columns
        ),
    )
    return result, provenance


def _no_provenance(frame: pd.DataFrame) -> TurnoverProvenance:
    """All-``False`` provenance for a path where nothing can be attributed."""
    empty = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    return TurnoverProvenance(
        turnover_actively_limited=empty,
        turnover_touched=empty,
        tradability_touched=empty,
        tradability_compliance_limited=empty,
    )


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


@overload
def _rebalance_tradability_aware(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    tradable: pd.DataFrame,
    *,
    calendar: str | None = None,
    episode_id: pd.DataFrame | None = None,
    return_provenance: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def _rebalance_tradability_aware(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    tradable: pd.DataFrame,
    *,
    calendar: str | None = None,
    episode_id: pd.DataFrame | None = None,
    return_provenance: Literal[True],
) -> tuple[pd.DataFrame, TurnoverProvenance]: ...


def _rebalance_tradability_aware(
    target_weights: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    tradable: pd.DataFrame,
    *,
    calendar: str | None = None,
    episode_id: pd.DataFrame | None = None,
    return_provenance: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, TurnoverProvenance]:
    """Rebalance while respecting per-symbol tradability.

    A symbol that is closed on a rebalance date never trades that date; its
    new target becomes a pending debt that is retried on every subsequent
    date it is tradable (not just the next scheduled rebalance) until fully
    executed — even if that takes several sessions under a turnover cap. A
    symbol that is always tradable is completely unaffected: its cadence
    (rebalance-date-only catch-up) is byte-identical to :func:`cap_turnover`.

    See :func:`cap_turnover` for ``episode_id``/``return_provenance``.
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

    # Always bound with cheap placeholders, even though they are only ever
    # read (below and by the caller) under `if return_provenance:` -- the
    # same unchanged flag that guards their real assignment. A static
    # analyzer cannot follow "guarded by the same boolean flag" across the
    # loop in between; this changes no behaviour.
    episode_values = np.empty((row_count, column_count), dtype=float)
    pending_turnover_episode_id = np.full(column_count, -1.0)
    actively_limited_out = np.zeros((row_count, column_count), dtype=bool)
    turnover_touched_out = np.zeros((row_count, column_count), dtype=bool)
    tradability_touched_out = np.zeros((row_count, column_count), dtype=bool)
    compliance_limited_out = np.zeros((row_count, column_count), dtype=bool)
    pending_before = np.zeros(column_count, dtype=bool)
    if return_provenance:
        if episode_id is None:
            raise BacktestError(
                "episode_id is required when return_provenance is True."
            )
        episode_values = (
            validate_frame(episode_id, name="episode_id")
            .reindex(index=validated.index, columns=validated.columns)
            .to_numpy(dtype=float)
        )

    for row_number in range(row_count):
        row_tradable = tradable_np[row_number]
        if return_provenance:
            pending_before = pending_due_to_closure.copy()
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

        if return_provenance:
            changed_this_row = np.abs(change) > EPSILON
            # tradability: this row's move is (at least partly) a catch-up
            # of a delta previously blocked by a closure (pending_before),
            # or a feasibility limit reached only because another column
            # stayed frozen by a closure (compliance_limited, proven in
            # _max_feasible_fraction's own docstring to be a tradability
            # artifact, never a turnover-budget one).
            tradability_touched_out[row_number] = changed_this_row & (
                pending_before | compliance_limited
            )
            compliance_limited_out[row_number] = changed_this_row & compliance_limited
            # turnover: independent of tradability, scoped to the same
            # episode-id convention as cap_turnover. `~pending_before`
            # avoids double-counting a cell whose shortfall this row is
            # already explained by tradability's own domain.
            generation = episode_values[row_number]
            debt_still_relevant = (
                (pending_turnover_episode_id != -1.0)
                & (pending_turnover_episode_id == generation)
                & ~pending_before
            )
            row_turnover_limited = fraction_from_turnover < 1.0 - EPSILON
            actively_limited_out[row_number] = changed_this_row & row_turnover_limited
            turnover_touched_out[row_number] = changed_this_row & (
                row_turnover_limited | debt_still_relevant
            )
            still_outstanding_turnover = eligible & unresolved
            pending_turnover_episode_id = np.where(
                still_outstanding_turnover, generation, -1.0
            )

        output[row_number] = current
        previous = current
    result = pd.DataFrame(output, index=validated.index, columns=validated.columns)
    if not return_provenance:
        return result
    provenance = TurnoverProvenance(
        turnover_actively_limited=pd.DataFrame(
            actively_limited_out, index=validated.index, columns=validated.columns
        ),
        turnover_touched=pd.DataFrame(
            turnover_touched_out, index=validated.index, columns=validated.columns
        ),
        tradability_touched=pd.DataFrame(
            tradability_touched_out, index=validated.index, columns=validated.columns
        ),
        tradability_compliance_limited=pd.DataFrame(
            compliance_limited_out, index=validated.index, columns=validated.columns
        ),
    )
    return result, provenance


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
