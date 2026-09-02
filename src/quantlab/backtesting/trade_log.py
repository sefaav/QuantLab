"""Build synthetic fills from executed portfolio-weight changes.

Each row records a non-zero change, its estimated notional and modelled costs.
It does not simulate share quantities or partial order execution.

Reason attribution uses four independent concepts, which may legitimately
differ on the same row without contradiction:

- ``action`` -- what actually happened to the executed weight
  (``_classify_action``).
- ``trigger`` -- the single most-upstream event CURRENTLY
  consumed that initiated the target change (``strategy_signal`` >
  ``portfolio_rebalance`` > ``volatility_target_adjustment`` > none). Not
  an exhaustive list of every transformed layer -- a downstream
  recomputation triggered by the same upstream event is not a separate
  cause.
- ``adjustment(s)`` -- the downstream execution/portfolio layer(s) that
  materially modified, delayed, redistributed, constrained or forced the
  path from the upstream target to the currently executed position,
  including a known execution debt whose catch-up this row represents
  (turnover cap, a prior symbol closure) -- computed INDEPENDENTLY of
  ``trigger`` and always from each layer's own real provenance, never
  deduced from ``new != desired``. ``position_rescaling``/
  ``deferred_catchup`` are a strict last-resort fallback, used only when
  no real adjustment layer and no trigger explain the row.
- ``position_strategy_origin`` -- the origin of the currently active
  strategic regime/stance relevant to this row (this symbol/leg), driven
  purely by the strategy's own decision state (see ``engine.py``'s
  ``decision_proxy``) -- NOT necessarily the origin of an executed-weight
  episode (a downstream layer can hold the executed weight flat while the
  strategic stance stays active), and NOT an execution timestamp.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from quantlab.constants import BPS_TO_FRACTION, EPSILON
from quantlab.exceptions import BacktestError
from quantlab.execution.orders import (
    equity_before_period,
    traded_notional,
    validate_execution_frame,
)
from quantlab.execution.slippage import (
    SlippageModel,
    validate_slippage_cost_frame,
)
from quantlab.portfolio.constraints import ConstraintTouch

#: Schema version of :data:`TRADE_LOG_COLUMNS`. A bare 12-column layout with
#: a single ``reason_code`` and none of the trigger/adjustment/position_
#: strategy_origin columns is version 1. Recorded in report/export metadata
#: (never as a column -- a bare CSV stays self-describing via its column
#: names instead, see docs on ``render_trade_table``/report generation).
TRADE_LOG_SCHEMA_VERSION = 2

#: Column order of the trade log.
TRADE_LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "previous_weight",
    "new_weight",
    "weight_change",
    "side",
    "action",
    "trigger_reason_code",
    "trigger_reason_detail_code",
    "trigger_reason_details",
    "adjustment_reason_codes",
    "adjustment_reason_details",
    "position_strategy_origin_timestamp",
    "position_strategy_origin_code",
    "position_strategy_origin_details",
    "reference_price",
    "traded_notional",
    "commission",
    "spread_cost",
    "slippage_cost",
    "total_cost",
]
assert len(TRADE_LOG_COLUMNS) == 21

#: `reason_code`/`reason_detail_code` frames -- all-or-nothing kwargs of
#: `build_trade_log` that enable reason attribution. Keyed by the kwarg
#: name so validation and array-extraction can iterate them uniformly.
_REASON_FRAME_KWARGS = (
    "executed_desired",
    "executed_constrained",
    "executed_signal_diag",
    "executed_allocated_diag",
    "executed_desired_diag",
)

#: Canonical, exhaustive order of every value `adjustment_reason_codes`
#: can ever contain -- constraint names (with their `_redistribution`
#: variant right after the base name, for the 3 constraints that have a
#: redistribution concept), the execution-layer causes, and the two
#: last-resort fallbacks. The single source of truth for how multiple
#: codes are combined into one `adjustment_reason_codes` string -- nothing
#: else in the codebase (including the dashboard) should build or parse
#: that convention on its own; use serialize_adjustment_codes/
#: parse_adjustment_codes instead. Every code `_classify_reason` can ever
#: emit MUST appear here, or `serialize_adjustment_codes` raises.
ADJUSTMENT_ORDER = (
    "long_only",
    "maximum_positions",
    "maximum_positions_redistribution",
    "minimum_weight",
    "minimum_weight_redistribution",
    "maximum_weight",
    "maximum_weight_redistribution",
    "maximum_gross_exposure",
    "maximum_leverage",
    "maximum_net_exposure",
    "tradability",
    "turnover_cap",
    "drift_compliance",
    "drift_compliance_pending",
    "stop_loss",
    "take_profit",
    "forced_liquidation",
    "position_rescaling",
    "deferred_catchup",
)

#: Redistribution-specific detail text, keyed by the constraint's BASE
#: name (not the `_redistribution`-suffixed code) -- a single generic
#: "another position was capped" sentence is wrong for minimum_weight
#: (dust removal) and maximum_positions (cardinality drop), so each gets
#: its own honest wording.
_REDISTRIBUTION_DETAIL_TEXT = {
    "maximum_weight": "redistribution after another position was capped",
    "minimum_weight": "redistribution after dust/small positions were removed",
    "maximum_positions": (
        "redistribution after positions were dropped to satisfy maximum_positions"
    ),
}


def serialize_adjustment_codes(codes: Iterable[str]) -> str:
    """Join adjustment names into adjustment_reason_codes' stable format.

    Names are deduplicated and ordered per ADJUSTMENT_ORDER (the real
    pipeline's own order), not by input/call order, so the result is
    deterministic and causally meaningful (e.g.
    "maximum_weight+turnover_cap"). Raises ``BacktestError`` on a code
    absent from ADJUSTMENT_ORDER -- a wiring bug between `_classify_
    reason` and this canonical list, never dropped silently.
    """
    present = set(codes)
    unknown = present - set(ADJUSTMENT_ORDER)
    if unknown:
        raise BacktestError(
            f"Unknown adjustment code(s) {sorted(unknown)}; not present in "
            "ADJUSTMENT_ORDER."
        )
    return "+".join(name for name in ADJUSTMENT_ORDER if name in present)


def parse_adjustment_codes(value: str, *, strict: bool = True) -> list[str]:
    """Inverse of serialize_adjustment_codes.

    ``strict=True`` (the default, used everywhere internally and in
    tests) raises ``BacktestError`` on a code absent from
    ADJUSTMENT_ORDER. ``strict=False`` preserves an unrecognized code
    as-is instead of raising -- reserved for displaying an artifact
    potentially produced by a future schema version; a code is never
    silently dropped in either mode.
    """
    codes = value.split("+")
    if strict:
        unknown = set(codes) - set(ADJUSTMENT_ORDER)
        if unknown:
            raise BacktestError(
                f"Unknown adjustment code(s) {sorted(unknown)}; not present in "
                "ADJUSTMENT_ORDER."
            )
    return codes


def stop_loss_take_profit_trigger_counts(trade_log: pd.DataFrame) -> dict[str, int]:
    """Count trade-log rows carrying the ``stop_loss``/``take_profit`` code.

    Generic across every strategy: whenever ``stop_loss_pct``/
    ``take_profit_pct`` are configured (any of the 5 directional
    strategies), a triggered force-flatten shows up here via the SAME
    ``adjustment_reason_codes`` column ``build_trade_log`` already
    populates -- never a second, strategy-specific recomputation. Returns
    ``{"stop_loss": 0, "take_profit": 0}`` on an empty log or one with no
    ``adjustment_reason_codes`` column (e.g. neither threshold was
    configured).
    """
    if trade_log.empty or "adjustment_reason_codes" not in trade_log.columns:
        return {"stop_loss": 0, "take_profit": 0}
    parsed = (
        trade_log["adjustment_reason_codes"]
        .dropna()
        .apply(lambda value: parse_adjustment_codes(value, strict=False))
    )
    return {
        "stop_loss": int(parsed.apply(lambda codes: "stop_loss" in codes).sum()),
        "take_profit": int(parsed.apply(lambda codes: "take_profit" in codes).sum()),
    }


def _classify_action(previous: float, new: float) -> str:
    """Classify what happened to a position from its weight before/after.

    Long/flat/short thresholds reuse the project-wide EPSILON, matching
    every other "is this effectively zero" check in the codebase. Total
    over every reachable input -- including the sub-epsilon corner case
    where ``previous`` and ``new`` individually read as flat yet differ by
    more than EPSILON (e.g. +0.6e-12 / -0.6e-12): noise-level floating
    point residue, not a real position, named explicitly as
    ``"flat_to_flat"`` rather than folded into an entry/exit label that
    would misrepresent a pair of weights indistinguishable from zero.
    """
    was_flat = abs(previous) <= EPSILON
    was_long = previous > EPSILON
    was_short = previous < -EPSILON
    now_flat = abs(new) <= EPSILON
    now_long = new > EPSILON
    now_short = new < -EPSILON

    if was_flat and now_long:
        return "entry_long"
    if was_flat and now_short:
        return "entry_short"
    if was_long and now_flat:
        return "exit_long"
    if was_short and now_flat:
        return "exit_short"
    if was_long and now_short:
        return "reverse_long_to_short"
    if was_short and now_long:
        return "reverse_short_to_long"
    if was_long and now_long:
        return "increase_long" if new > previous else "reduce_long"
    if was_short and now_short:
        return "increase_short" if new < previous else "reduce_short"
    return "flat_to_flat"


def _compose_details(generic: str, specific: str | None) -> str:
    """Enrich the generic pipeline-level explanation with a business-specific one.

    The specific detail never REPLACES the generic text, it only adds to
    it (explicit product rule: a more precise piece of information must
    never remove a correct one already available).
    """
    if specific is None:
        return generic
    return f"{generic}; {specific}"


@dataclass(frozen=True)
class TradeReason:
    """Trigger + adjustment attribution for one fill (see module docstring)."""

    trigger_code: str | None
    trigger_detail_code: str | None
    trigger_details: str | None
    adjustment_codes: str | None
    adjustment_details: str | None


def _classify_reason(
    *,
    new: float,
    previous: float,
    executed_desired: float,
    executed_desired_prev: float,
    executed_constrained: float,
    signal_now: float,
    signal_prev: float,
    allocated_now: float,
    allocated_prev: float,
    desired_diag_now: float,
    desired_diag_prev: float,
    strategy_detail_code: str | None = None,
    strategy_details: str | None = None,
    contributing_constraints: Sequence[str] = (),
    constraint_before: Mapping[str, float] | None = None,
    constraint_after: Mapping[str, float] | None = None,
    tradability_touched: bool = False,
    tradability_compliance_limited: bool = False,
    turnover_touched: bool = False,
    turnover_actively_limited: bool = False,
    stop_loss_triggered: bool = False,
    take_profit_triggered: bool = False,
    forced_liquidation: bool = False,
    drift_compliance_forced: bool = False,
    drift_compliance_pending: bool = False,
) -> TradeReason:
    """Return the trigger + adjustment attribution for one fill.

    Both are assigned from real, per-layer provenance signals only, never
    deduced from `new != desired`. Full trigger/adjustment priority order:
    see docs/backtesting.md#trade-log-reason-attribution. ``signal_now``/
    ``signal_prev`` must already come from the strategy's own diagnostic
    decision proxy (``decision_signal()`` when provided, else the raw
    signal) -- see ``engine.py`` -- never the raw signal directly for a
    strategy whose raw signal mixes decision state with mechanical
    rescaling.
    """
    # TRIGGER
    if abs(signal_now - signal_prev) > EPSILON:
        generic = f"signal {signal_prev:.4f} -> {signal_now:.4f} since last rebalance"
        trigger_code: str | None = "strategy_signal"
        trigger_detail_code = strategy_detail_code
        trigger_details: str | None = _compose_details(generic, strategy_details)
    elif abs(allocated_now - allocated_prev) > EPSILON:
        trigger_code = "portfolio_rebalance"
        trigger_detail_code = None
        trigger_details = (
            f"allocator output {allocated_prev:.4f} -> {allocated_now:.4f} "
            "since last rebalance"
        )
    elif abs(desired_diag_now - desired_diag_prev) > EPSILON:
        trigger_code = "volatility_target_adjustment"
        trigger_detail_code = None
        trigger_details = (
            f"target {desired_diag_prev:.4f} -> {desired_diag_now:.4f} "
            "since last rebalance"
        )
    else:
        trigger_code, trigger_detail_code, trigger_details = None, None, None

    # ADJUSTMENT(S)
    adjustment_codes_list: list[str] = []
    adjustment_clauses: list[str] = []
    if forced_liquidation:
        # Overrides every other adjustment: once ruined, no other layer's
        # specific clip value still explains the executed weight.
        adjustment_codes_list = ["forced_liquidation"]
        adjustment_clauses = [
            "portfolio equity reached zero -- position forcibly flattened, "
            "no margin call modeled"
        ]
    elif stop_loss_triggered or take_profit_triggered:
        # Overrides every ordinary constraint adjustment (a real
        # stop-loss/take-profit force-flatten fully explains the executed
        # weight regardless of what a constraint would otherwise have
        # clipped it to), but is itself overridden above by
        # forced_liquidation -- portfolio ruin is more severe than a
        # single position's own risk control.
        adjustment_codes_list = []
        adjustment_clauses = []
        if stop_loss_triggered:
            adjustment_codes_list.append("stop_loss")
            adjustment_clauses.append(
                "cumulative return since entry breached the configured "
                "stop_loss_pct -- position forcibly closed"
            )
        if take_profit_triggered:
            adjustment_codes_list.append("take_profit")
            adjustment_clauses.append(
                "cumulative return since entry reached the configured "
                "take_profit_pct -- position forcibly closed"
            )
    elif drift_compliance_forced or drift_compliance_pending:
        # Overrides every ordinary constraint/tradability/turnover_cap
        # adjustment: this row's magnitude comes from the drift-compliance
        # LP restoring a hard risk limit organic price drift breached
        # off-schedule, not from the ordinary decision pipeline at all --
        # but is itself overridden above by forced_liquidation/stop_loss/
        # take_profit, each a still more specific or more severe cause.
        adjustment_codes_list = []
        adjustment_clauses = []
        if drift_compliance_forced:
            adjustment_codes_list.append("drift_compliance")
            adjustment_clauses.append(
                "organic price drift breached a hard portfolio risk limit "
                "between rebalances -- position corrected back toward "
                "compliance by the drift-compliance linear program"
            )
        if drift_compliance_pending:
            adjustment_codes_list.append("drift_compliance_pending")
            adjustment_clauses.append(
                "drift-caused breach not yet fully resolved -- the "
                "responsible symbol/group is still untradable, best "
                "achievable correction applied, retried each day it "
                "remains blocked"
            )
    else:
        for name in contributing_constraints:
            adjustment_codes_list.append(name)
            assert constraint_before is not None
            assert constraint_after is not None
            before_value = constraint_before[name]
            after_value = constraint_after[name]
            if name.endswith("_redistribution"):
                base_name = name.removesuffix("_redistribution")
                cause_text = _REDISTRIBUTION_DETAIL_TEXT.get(
                    base_name, "redistribution after another position was adjusted"
                )
                adjustment_clauses.append(
                    f"{name}: {before_value:.4f} -> {after_value:.4f} ({cause_text})"
                )
            else:
                adjustment_clauses.append(
                    f"{name}: {before_value:.4f} -> {after_value:.4f}"
                )

        # tradability/turnover_cap may share the same executed magnitude
        # (they can both apply to the same cell in the same pass, without
        # a clean per-mechanism split) -- only the FIRST cause to fire
        # (ADJUSTMENT_ORDER: tradability before turnover_cap) carries the
        # magnitude, the other stays purely causal rather than repeating a
        # possibly-misleading shared number.
        magnitude_already_shown = False
        if tradability_touched:
            adjustment_codes_list.append("tradability")
            cause_text = (
                "rebalancing feasibility limit reached while another "
                "symbol remained closed"
                if tradability_compliance_limited
                else "catching up a delta previously blocked while the "
                "symbol was closed"
            )
            if not magnitude_already_shown:
                adjustment_clauses.append(
                    f"tradability: desired {executed_constrained:.4f}, "
                    f"executed {new:.4f} ({cause_text})"
                )
                magnitude_already_shown = True
            else:
                adjustment_clauses.append(f"tradability: {cause_text}")
        if turnover_touched:
            adjustment_codes_list.append("turnover_cap")
            cause_text = (
                "turnover-capped this period"
                if turnover_actively_limited
                else "catching up a target previously deferred by turnover cap"
            )
            if not magnitude_already_shown:
                adjustment_clauses.append(
                    f"turnover_cap: desired {executed_constrained:.4f}, "
                    f"executed {new:.4f} ({cause_text})"
                )
                magnitude_already_shown = True
            else:
                adjustment_clauses.append(
                    f"turnover_cap: {cause_text} "
                    "(see above for the shared executed magnitude)"
                )

        # Strict fallback: reached only when NOTHING real above explains
        # this row's movement, and no trigger explains it either.
        if (
            not adjustment_codes_list
            and trigger_code is None
            and abs(new - previous) > EPSILON
        ):
            if abs(executed_desired - executed_desired_prev) > EPSILON:
                adjustment_codes_list = ["position_rescaling"]
                adjustment_clauses = [
                    f"target continued drifting {executed_desired_prev:.4f} -> "
                    f"{executed_desired:.4f} with no new upstream decision"
                ]
            else:
                adjustment_codes_list = ["deferred_catchup"]
                adjustment_clauses = [
                    f"{previous:.4f} -> {new:.4f} with no new upstream driver"
                ]

    if adjustment_codes_list:
        adjustment_codes: str | None = serialize_adjustment_codes(adjustment_codes_list)
        adjustment_details: str | None = "; ".join(adjustment_clauses)
    else:
        adjustment_codes, adjustment_details = None, None

    if trigger_code is None and adjustment_codes is None:
        # Safety net, should not normally happen.
        trigger_code, trigger_detail_code, trigger_details = (
            "unknown",
            None,
            "no upstream driver identified",
        )

    return TradeReason(
        trigger_code=trigger_code,
        trigger_detail_code=trigger_detail_code,
        trigger_details=trigger_details,
        adjustment_codes=adjustment_codes,
        adjustment_details=adjustment_details,
    )


def _non_negative_rate(value: object, name: str) -> float:
    """Return a finite, non-negative numeric rate or raise ``BacktestError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise BacktestError(f"{name} must be a finite non-negative number.")
    rate = float(value)
    if not np.isfinite(rate) or rate < 0.0:
        raise BacktestError(f"{name} must be a finite non-negative number.")
    return rate


def _validate_unique_axes(frame: pd.DataFrame, name: str) -> None:
    """Reject duplicate labels that would make scalar fills ambiguous."""
    if not frame.index.is_unique:
        raise BacktestError(f"{name} index must not contain duplicate labels.")
    if not frame.columns.is_unique:
        raise BacktestError(f"{name} columns must not contain duplicate labels.")


def _validate_matching_frame(
    frame: object, name: str, reference: pd.DataFrame
) -> pd.DataFrame:
    """Validate a frame is a DataFrame sharing ``reference``'s exact axes."""
    if not isinstance(frame, pd.DataFrame):
        raise BacktestError(f"{name} must be a pandas DataFrame.")
    _validate_unique_axes(frame, name)
    if not frame.index.equals(reference.index) or not frame.columns.equals(
        reference.columns
    ):
        raise BacktestError(
            f"{name} must have exactly the same index and columns as executed_weights."
        )
    return frame


def _validate_optional_group(
    names: tuple[str, ...],
    frames: Mapping[str, pd.DataFrame | None],
    *,
    reference: pd.DataFrame,
    requires: bool,
    requires_label: str,
) -> bool:
    """Validate an all-or-nothing optional kwarg group; return whether supplied.

    Every frame in ``names`` must be supplied together or not at all. When
    supplied, ``requires`` (typically ``attribute_reasons``) must already
    be true, or the group is rejected as meaningless on its own.
    """
    supplied = [name for name in names if frames[name] is not None]
    if not supplied:
        return False
    if len(supplied) != len(names):
        missing = sorted(set(names) - set(supplied))
        raise BacktestError(
            f"{'/'.join(names)} must be supplied all together or not at "
            f"all; missing: {missing}."
        )
    if not requires:
        raise BacktestError(f"{'/'.join(names)} requires {requires_label}.")
    for name in names:
        frame = frames[name]
        assert frame is not None
        _validate_matching_frame(frame, name, reference)
    return True


def build_trade_log(
    executed_weights: pd.DataFrame,
    weight_changes: pd.DataFrame,
    equity: pd.Series,
    reference_prices: pd.DataFrame,
    *,
    commission_bps: float,
    spread_bps: float,
    slippage_model: SlippageModel,
    slippage_equity: pd.Series | None = None,
    executed_desired: pd.DataFrame | None = None,
    executed_constrained: pd.DataFrame | None = None,
    executed_signal_diag: pd.DataFrame | None = None,
    executed_allocated_diag: pd.DataFrame | None = None,
    executed_desired_diag: pd.DataFrame | None = None,
    tradable: pd.DataFrame | None = None,
    executed_strategy_reason_code: pd.DataFrame | None = None,
    executed_strategy_reason_details: pd.DataFrame | None = None,
    constraint_provenance: dict[str, ConstraintTouch] | None = None,
    executed_turnover_actively_limited: pd.DataFrame | None = None,
    executed_turnover_touched: pd.DataFrame | None = None,
    executed_tradability_touched: pd.DataFrame | None = None,
    executed_tradability_compliance_limited: pd.DataFrame | None = None,
    executed_forced_liquidation: pd.DataFrame | None = None,
    executed_stop_loss_triggered: pd.DataFrame | None = None,
    executed_take_profit_triggered: pd.DataFrame | None = None,
    executed_drift_compliance_forced: pd.DataFrame | None = None,
    executed_drift_compliance_pending: pd.DataFrame | None = None,
    executed_position_strategy_origin_timestamp: pd.DataFrame | None = None,
    executed_position_strategy_origin_code: pd.DataFrame | None = None,
    executed_position_strategy_origin_details: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the trade log from executed weight changes.

    Args:
        executed_weights: Weights actually in force each period.
        weight_changes: Per-symbol change in the executed book.
        equity: Net equity curve (used to size notional off prior equity).
        reference_prices: Price matrix for the reference price of each fill.
        commission_bps: Commission in bps.
        spread_bps: Full spread in bps (half applied).
        slippage_model: The same per-symbol model used by accounting.
        slippage_equity: Per-date equity passed to the slippage model's
            ``equity`` argument. Pass ``AccountingResult.equity_for_costs``
            to reproduce volume-based accounting costs.
        executed_desired: The fully-desired, pre-constraint target weights
            (post-allocator, post-volatility-targeting), aligned to
            ``executed_weights``' own index via the same
            ``executed_weights()`` shift -- a *real* pipeline frame, not a
            reconstruction. Required (with the 4 arguments below) to
            populate the trigger/adjustment columns; when omitted, those
            columns are ``None``/``NaT`` for every row (e.g. walk-forward's
            own call site, which rebuilds trades from a stitched
            out-of-sample weight series with no per-fold diagnostic frames
            surviving the stitch -- attribution is genuinely unavailable
            there, not merely unwired) while ``action`` is still always
            computed.
        executed_constrained: The post-``ConstraintSet``, pre-turnover-cap
            target weights, aligned the same way as ``executed_desired``.
        executed_signal_diag: The strategy's decision proxy (``decision_
            signal()`` when the strategy provides one, else the raw
            signal), resampled to rebalance dates and aligned to
            ``executed_weights``' index -- a *diagnostic* frame (see
            ``engine.py``'s ``_rebalance_diagnostic_frame``), used only to
            detect "did the strategy's decision change since the last
            rebalance", never to recompute an executed weight.
        executed_allocated_diag: The allocator's raw output, diagnostic-
            sampled the same way as ``executed_signal_diag``.
        executed_desired_diag: ``executed_desired``, diagnostic-sampled the
            same way, used to detect a volatility-targeting-driven change
            since the last rebalance (as opposed to ``executed_desired``
            itself, which is used unsampled here to detect a still-drifting
            target for the position_rescaling/deferred_catchup fallback).
        tradable: Per-symbol tradability mask, real and already computed
            by the engine. ``None`` means every symbol was always tradable
            (no per-symbol calendar closures modeled) -- distinct from a
            per-cell ``False``.
        executed_strategy_reason_code: The strategy's own per-cell
            ``trigger_reason_detail_code`` (``str | None``), aligned to
            ``executed_weights``' index the same way as the other reason
            frames. Requires the 5 arguments above; both this and
            ``executed_strategy_reason_details`` must be supplied
            together or not at all. When present and a row resolves to
            trigger ``"strategy_signal"``, overrides that branch's
            generic text with the strategy's own attribution.
        executed_strategy_reason_details: The strategy's own per-cell
            free-text explanation, paired with
            ``executed_strategy_reason_code``.
        constraint_provenance: Per-constraint provenance from
            :meth:`ConstraintSet.apply_with_provenance`, keyed by
            constraint name (including a ``"*_redistribution"`` entry for
            a constraint split into direct/redistribution -- both keys
            carry the SAME ``before``/``after`` as the base constraint,
            only ``touched`` differs), with each ``ConstraintTouch``'s
            frames already aligned to ``executed_weights``' index the same
            way as the other reason frames. Requires the 5 arguments
            above. Every constraint whose ``touched`` mask is set for a
            given cell contributes its own entry (with its own
            before/after text) to ``adjustment_reason_codes`` -- never a
            single winning constraint.
        executed_turnover_actively_limited: Real, cell-level provenance
            from ``rebalancing.py``'s turnover-cap tracking (see
            ``cap_turnover``/``rebalance_and_cap_turnover``'s
            ``return_provenance``): True where the turnover budget itself
            bound this row's move for that cell. Requires the 5 arguments
            above; must be supplied together with
            ``executed_turnover_touched`` or not at all.
        executed_turnover_touched: The broader, episode-scoped turnover
            provenance from the same source -- also True on a later row
            still catching up a debt from an earlier turnover-limited move
            toward the SAME still-unresolved upstream decision.
        executed_tradability_touched: Real, cell-level provenance from
            ``rebalancing.py``'s closure-catchup tracking: True where this
            row's move is (at least partly) catching up a delta previously
            blocked by a closure, or a feasibility limit reached only
            because another symbol stayed closed. Must be supplied
            together with ``executed_tradability_compliance_limited`` or
            not at all; requires the 5 arguments above.
        executed_tradability_compliance_limited: Sub-case of the above,
            distinguishing the feasibility-limit case for detail text.
        executed_forced_liquidation: Real, row-broadcast provenance from
            ``AccountingResult.ruined``: True on every date the portfolio
            was ruined and positions were forcibly flattened. Requires the
            5 arguments above. Highest-priority adjustment -- see
            docs/backtesting.md#trade-log-reason-attribution for the full
            override order. Currently unreachable via the real pipeline
            (``run_accounting`` zeroes ``weight_changes`` on every ruined
            date), wired anyway so `_classify_reason` never falls back to
            `unknown`/`deferred_catchup` should that ever change.
        executed_stop_loss_triggered: Real, row-broadcast provenance from
            ``AccountingResult.stop_loss_triggered``: True on every cell
            whose position was force-flattened by a stop-loss breach on
            the real executed position (see ``quantlab.backtesting.
            accounting._detect_stop_loss_take_profit``). Unlike
            ``executed_forced_liquidation``, this branch IS reachable in
            practice -- a stop-loss-forced exit is a real, non-zero
            weight change. Requires the 5 reason-attribution arguments
            above.
        executed_take_profit_triggered: Same as
            ``executed_stop_loss_triggered``, for the favorable-side
            threshold.
        executed_drift_compliance_forced: Real, row-broadcast provenance
            from ``AccountingResult.drift_compliance_forced``: True on
            every cell whose executed weight was set by the drift-
            compliance linear program landing a correction for a hard
            risk limit organic price drift breached off-schedule (see
            ``quantlab.backtesting.accounting.apply_weight_drift``).
            Requires the 5 reason-attribution arguments above.
        executed_drift_compliance_pending: Same as ``executed_drift_
            compliance_forced``, for a still-unresolved breach (the
            responsible symbol/group remains untradable) -- the best
            achievable correction, retried every day it stays blocked.
        executed_position_strategy_origin_timestamp: The timestamp of the
            most recent strategic regime transition (per the strategy's
            own decision proxy) still active for this cell -- ``NaT`` when
            no strategic position is currently active (the decision proxy
            is flat). Must be supplied together with the 2 arguments below
            or not at all; requires the 5 arguments above.
        executed_position_strategy_origin_code: The strategy's own
            ``explain_signals()`` detail code at that origin transition,
            or ``None`` when unavailable -- a temporally correct origin is
            tracked independently of whether a specific code exists for it.
        executed_position_strategy_origin_details: Free text paired with
            the above.

    Returns:
        A DataFrame with :data:`TRADE_LOG_COLUMNS`, one row per non-zero fill.
    """
    if not isinstance(executed_weights, pd.DataFrame):
        raise BacktestError("executed_weights must be a pandas DataFrame.")
    if not isinstance(weight_changes, pd.DataFrame):
        raise BacktestError("weight_changes must be a pandas DataFrame.")
    if not isinstance(reference_prices, pd.DataFrame):
        raise BacktestError("reference_prices must be a pandas DataFrame.")
    if not isinstance(equity, pd.Series):
        raise BacktestError("equity must be a pandas Series.")
    if slippage_equity is not None and not isinstance(slippage_equity, pd.Series):
        raise BacktestError("slippage_equity must be a pandas Series.")
    if not isinstance(slippage_model, SlippageModel):
        raise BacktestError("slippage_model must implement SlippageModel.")

    reason_frames = {
        "executed_desired": executed_desired,
        "executed_constrained": executed_constrained,
        "executed_signal_diag": executed_signal_diag,
        "executed_allocated_diag": executed_allocated_diag,
        "executed_desired_diag": executed_desired_diag,
    }
    supplied_reason_frames = [
        name for name in _REASON_FRAME_KWARGS if reason_frames[name] is not None
    ]
    if supplied_reason_frames and len(supplied_reason_frames) != len(
        _REASON_FRAME_KWARGS
    ):
        missing = sorted(set(_REASON_FRAME_KWARGS) - set(supplied_reason_frames))
        raise BacktestError(
            "executed_desired/executed_constrained/executed_signal_diag/"
            "executed_allocated_diag/executed_desired_diag must be supplied "
            f"all together or not at all; missing: {missing}."
        )
    attribute_reasons = bool(supplied_reason_frames)
    if attribute_reasons:
        for name in _REASON_FRAME_KWARGS:
            frame = reason_frames[name]
            assert frame is not None  # narrowed by attribute_reasons above
            _validate_matching_frame(frame, name, executed_weights)
    if tradable is not None:
        _validate_matching_frame(tradable, "tradable", executed_weights)

    attribute_strategy_reasons = _validate_optional_group(
        ("executed_strategy_reason_code", "executed_strategy_reason_details"),
        {
            "executed_strategy_reason_code": executed_strategy_reason_code,
            "executed_strategy_reason_details": executed_strategy_reason_details,
        },
        reference=executed_weights,
        requires=attribute_reasons,
        requires_label="the reason-attribution frames (executed_desired etc.)",
    )

    if constraint_provenance is not None:
        if not attribute_reasons:
            raise BacktestError(
                "constraint_provenance requires the reason-attribution frames "
                "(executed_desired etc.) to also be supplied."
            )
        if not isinstance(constraint_provenance, dict):
            raise BacktestError("constraint_provenance must be a dict.")
        for constraint_name, touch in constraint_provenance.items():
            if not isinstance(touch, ConstraintTouch):
                raise BacktestError(
                    f"constraint_provenance[{constraint_name!r}] must be a "
                    "ConstraintTouch."
                )
            for field_name, frame in (
                ("touched", touch.touched),
                ("before", touch.before),
                ("after", touch.after),
                ("direct", touch.direct),
            ):
                label = f"constraint_provenance[{constraint_name!r}].{field_name}"
                _validate_matching_frame(frame, label, executed_weights)

    attribute_turnover = _validate_optional_group(
        ("executed_turnover_actively_limited", "executed_turnover_touched"),
        {
            "executed_turnover_actively_limited": executed_turnover_actively_limited,
            "executed_turnover_touched": executed_turnover_touched,
        },
        reference=executed_weights,
        requires=attribute_reasons,
        requires_label="the reason-attribution frames (executed_desired etc.)",
    )
    attribute_tradability = _validate_optional_group(
        (
            "executed_tradability_touched",
            "executed_tradability_compliance_limited",
        ),
        {
            "executed_tradability_touched": executed_tradability_touched,
            "executed_tradability_compliance_limited": (
                executed_tradability_compliance_limited
            ),
        },
        reference=executed_weights,
        requires=attribute_reasons,
        requires_label="the reason-attribution frames (executed_desired etc.)",
    )
    attribute_forced_liquidation = False
    if executed_forced_liquidation is not None:
        if not attribute_reasons:
            raise BacktestError(
                "executed_forced_liquidation requires the reason-attribution "
                "frames (executed_desired etc.) to also be supplied."
            )
        _validate_matching_frame(
            executed_forced_liquidation, "executed_forced_liquidation", executed_weights
        )
        attribute_forced_liquidation = True
    attribute_stop_loss = False
    if executed_stop_loss_triggered is not None:
        if not attribute_reasons:
            raise BacktestError(
                "executed_stop_loss_triggered requires the reason-attribution "
                "frames (executed_desired etc.) to also be supplied."
            )
        _validate_matching_frame(
            executed_stop_loss_triggered,
            "executed_stop_loss_triggered",
            executed_weights,
        )
        attribute_stop_loss = True
    attribute_take_profit = False
    if executed_take_profit_triggered is not None:
        if not attribute_reasons:
            raise BacktestError(
                "executed_take_profit_triggered requires the reason-attribution "
                "frames (executed_desired etc.) to also be supplied."
            )
        _validate_matching_frame(
            executed_take_profit_triggered,
            "executed_take_profit_triggered",
            executed_weights,
        )
        attribute_take_profit = True
    attribute_drift_compliance_forced = False
    if executed_drift_compliance_forced is not None:
        if not attribute_reasons:
            raise BacktestError(
                "executed_drift_compliance_forced requires the reason-"
                "attribution frames (executed_desired etc.) to also be supplied."
            )
        _validate_matching_frame(
            executed_drift_compliance_forced,
            "executed_drift_compliance_forced",
            executed_weights,
        )
        attribute_drift_compliance_forced = True
    attribute_drift_compliance_pending = False
    if executed_drift_compliance_pending is not None:
        if not attribute_reasons:
            raise BacktestError(
                "executed_drift_compliance_pending requires the reason-"
                "attribution frames (executed_desired etc.) to also be supplied."
            )
        _validate_matching_frame(
            executed_drift_compliance_pending,
            "executed_drift_compliance_pending",
            executed_weights,
        )
        attribute_drift_compliance_pending = True
    attribute_position_origin = _validate_optional_group(
        (
            "executed_position_strategy_origin_timestamp",
            "executed_position_strategy_origin_code",
            "executed_position_strategy_origin_details",
        ),
        {
            "executed_position_strategy_origin_timestamp": (
                executed_position_strategy_origin_timestamp
            ),
            "executed_position_strategy_origin_code": (
                executed_position_strategy_origin_code
            ),
            "executed_position_strategy_origin_details": (
                executed_position_strategy_origin_details
            ),
        },
        reference=executed_weights,
        requires=attribute_reasons,
        requires_label="the reason-attribution frames (executed_desired etc.)",
    )

    commission_rate = _non_negative_rate(commission_bps, "commission_bps")
    spread_rate = _non_negative_rate(spread_bps, "spread_bps")

    for name, frame in (
        ("executed_weights", executed_weights),
        ("weight_changes", weight_changes),
        ("reference_prices", reference_prices),
    ):
        _validate_unique_axes(frame, name)
    if not equity.index.is_unique:
        raise BacktestError("equity index must not contain duplicate labels.")
    if slippage_equity is not None and not slippage_equity.index.is_unique:
        raise BacktestError("slippage_equity index must not contain duplicate labels.")
    if not weight_changes.index.is_monotonic_increasing:
        raise BacktestError("weight_changes index must be sorted in increasing order.")
    if not executed_weights.index.equals(weight_changes.index) or not (
        executed_weights.columns.equals(weight_changes.columns)
    ):
        raise BacktestError(
            "executed_weights and weight_changes must have identical axes."
        )

    if weight_changes.empty:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    executed = validate_execution_frame(executed_weights, name="executed_weights")
    changes = validate_execution_frame(weight_changes, name="weight_changes")
    executed_values = executed.to_numpy()
    change_values = changes.to_numpy()
    # NOT the previous ROW's own executed value (`executed_values[:-1]`,
    # shifted) -- that only coincides with the value actually held right
    # before THIS row's trade when `weight_changes` is a plain row-to-row
    # diff of `executed` (true when model_weight_drift=False). With drift
    # active, `weight_changes` is `apply_weight_drift`'s own real per-row
    # TRADE delta (zero on a pure-drift row, the landed trade's true size
    # otherwise) -- the position can have drifted organically between the
    # previous row's own reported value and this row's trade, so the two
    # diverge. `executed - change` is correct in BOTH cases: it is, by
    # construction, exactly the value this row's own trade delta was
    # computed against (`new - previous == change` holds by definition,
    # not just as a property to verify), and reduces to the identical
    # previous-row-shifted value whenever `weight_changes` genuinely is a
    # plain diff (the non-drift path, unchanged from before this fix).
    previous_values = executed_values - change_values

    # Placeholder defaults for every reason-attribution lookup array, always
    # bound regardless of which `attribute_*` flags are set. Each one is
    # only ever read far below, inside the per-row/per-column loop, under
    # the exact same `attribute_*` flag that gates its real assignment here
    # -- so a placeholder value is never actually consulted at runtime. This
    # is for static analysis only (a linter cannot follow "guarded by the
    # same unchanged boolean flag" across the loop boundary in between);
    # it changes no behaviour.
    empty_float = np.empty(executed_values.shape, dtype=float)
    empty_object = np.empty(executed_values.shape, dtype=object)
    empty_bool = np.empty(executed_values.shape, dtype=bool)
    desired_values = empty_float
    constrained_values = empty_float
    signal_diag_values = empty_float
    allocated_diag_values = empty_float
    desired_diag_values = empty_float
    desired_prev_values = empty_float
    signal_diag_prev_values = empty_float
    allocated_diag_prev_values = empty_float
    desired_diag_prev_values = empty_float
    strategy_detail_code_values = empty_object
    strategy_details_values = empty_object
    constraint_names: list[str] = []
    constraint_touched_values: dict[str, np.ndarray] = {}
    constraint_before_values: dict[str, np.ndarray] = {}
    constraint_after_values: dict[str, np.ndarray] = {}
    tradability_touched_values = empty_bool
    tradability_compliance_limited_values = empty_bool
    turnover_touched_values = empty_bool
    turnover_actively_limited_values = empty_bool
    forced_liquidation_values = empty_bool
    stop_loss_triggered_values = empty_bool
    take_profit_triggered_values = empty_bool
    drift_compliance_forced_values = empty_bool
    drift_compliance_pending_values = empty_bool
    position_origin_timestamp_values = empty_object
    position_origin_code_values = empty_object
    position_origin_details_values = empty_object

    if attribute_reasons:
        assert executed_desired is not None
        assert executed_constrained is not None
        assert executed_signal_diag is not None
        assert executed_allocated_diag is not None
        assert executed_desired_diag is not None
        # Axes were already validated exactly equal to executed_weights'
        # (same order too) above -- validate_execution_frame doesn't reorder
        # anything, so a plain .to_numpy() lines up with executed_values
        # cell-for-cell, no reindex needed.
        desired_values = executed_desired.to_numpy()
        constrained_values = executed_constrained.to_numpy()
        signal_diag_values = executed_signal_diag.to_numpy()
        allocated_diag_values = executed_allocated_diag.to_numpy()
        desired_diag_values = executed_desired_diag.to_numpy()
        # Same "no prior rebalance -> flat" convention as previous_values:
        # np.vstack, never .shift(1) (which would put NaN in row 0 and
        # silently break the very first trade's comparison).
        zeros_row = np.zeros((1, executed_values.shape[1]))
        desired_prev_values = np.vstack([zeros_row, desired_values[:-1]])
        signal_diag_prev_values = np.vstack([zeros_row, signal_diag_values[:-1]])
        allocated_diag_prev_values = np.vstack([zeros_row, allocated_diag_values[:-1]])
        desired_diag_prev_values = np.vstack([zeros_row, desired_diag_values[:-1]])

        if attribute_strategy_reasons:
            assert executed_strategy_reason_code is not None
            assert executed_strategy_reason_details is not None
            strategy_detail_code_values = executed_strategy_reason_code.to_numpy(
                dtype=object
            )
            strategy_details_values = executed_strategy_reason_details.to_numpy(
                dtype=object
            )

        if constraint_provenance is not None:
            constraint_names = [
                name for name in ADJUSTMENT_ORDER if name in constraint_provenance
            ]
            constraint_touched_values = {
                name: constraint_provenance[name].touched.to_numpy(dtype=bool)
                for name in constraint_names
            }
            constraint_before_values = {
                name: constraint_provenance[name].before.to_numpy()
                for name in constraint_names
            }
            constraint_after_values = {
                name: constraint_provenance[name].after.to_numpy()
                for name in constraint_names
            }

        if attribute_turnover:
            assert executed_turnover_actively_limited is not None
            assert executed_turnover_touched is not None
            turnover_actively_limited_values = (
                executed_turnover_actively_limited.to_numpy(dtype=bool)
            )
            turnover_touched_values = executed_turnover_touched.to_numpy(dtype=bool)
        if attribute_tradability:
            assert executed_tradability_touched is not None
            assert executed_tradability_compliance_limited is not None
            tradability_touched_values = executed_tradability_touched.to_numpy(
                dtype=bool
            )
            tradability_compliance_limited_values = (
                executed_tradability_compliance_limited.to_numpy(dtype=bool)
            )
        if attribute_forced_liquidation:
            assert executed_forced_liquidation is not None
            forced_liquidation_values = executed_forced_liquidation.to_numpy(dtype=bool)
        if attribute_stop_loss:
            assert executed_stop_loss_triggered is not None
            stop_loss_triggered_values = executed_stop_loss_triggered.to_numpy(
                dtype=bool
            )
        if attribute_take_profit:
            assert executed_take_profit_triggered is not None
            take_profit_triggered_values = executed_take_profit_triggered.to_numpy(
                dtype=bool
            )
        if attribute_drift_compliance_forced:
            assert executed_drift_compliance_forced is not None
            drift_compliance_forced_values = executed_drift_compliance_forced.to_numpy(
                dtype=bool
            )
        if attribute_drift_compliance_pending:
            assert executed_drift_compliance_pending is not None
            drift_compliance_pending_values = (
                executed_drift_compliance_pending.to_numpy(dtype=bool)
            )
        if attribute_position_origin:
            assert executed_position_strategy_origin_timestamp is not None
            assert executed_position_strategy_origin_code is not None
            assert executed_position_strategy_origin_details is not None
            position_origin_timestamp_values = (
                executed_position_strategy_origin_timestamp.to_numpy(dtype=object)
            )
            position_origin_code_values = (
                executed_position_strategy_origin_code.to_numpy(dtype=object)
            )
            position_origin_details_values = (
                executed_position_strategy_origin_details.to_numpy(dtype=object)
            )

    previous_equity_series = equity_before_period(equity, changes.index)
    previous_equity = previous_equity_series.to_numpy()
    notional_values = traded_notional(changes, equity).to_numpy()

    slippage_ref = slippage_equity if slippage_equity is not None else equity
    previous_slippage_series = equity_before_period(
        slippage_ref,
        changes.index,
        name="slippage_equity",
    )

    slippage_fraction = slippage_model.per_symbol_cost(
        changes, previous_slippage_series
    )
    slippage_values = validate_slippage_cost_frame(
        slippage_fraction, changes
    ).to_numpy()
    slippage_currency = slippage_values * previous_equity[:, None]

    reference_aligned = reference_prices.reindex(
        index=weight_changes.index, columns=weight_changes.columns
    )
    try:
        reference_values = reference_aligned.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("reference_prices must contain numeric values.") from exc
    lagged_reference_values = np.vstack(
        [np.full((1, reference_values.shape[1]), np.nan), reference_values[:-1]]
    )

    records: list[dict[str, object]] = []
    # Iterate only over non-zero changes to keep the log compact.
    nonzero = np.abs(change_values) > EPSILON
    for row_number, timestamp in enumerate(weight_changes.index):
        changed_columns = np.flatnonzero(nonzero[row_number])
        if not len(changed_columns):
            continue
        for column_number in changed_columns:
            column_index = int(column_number)
            symbol = weight_changes.columns[column_index]
            delta = float(change_values[row_number, column_index])
            notional = float(notional_values[row_number, column_index])
            commission = notional * commission_rate * BPS_TO_FRACTION
            spread = notional * spread_rate * BPS_TO_FRACTION / 2.0
            slippage = float(slippage_currency[row_number, column_index])
            price = float(lagged_reference_values[row_number, column_index])
            if not np.isfinite(price) or price <= 0.0:
                raise BacktestError(
                    "A positive finite prior-period reference price is required "
                    f"for {symbol!r} on {timestamp!r}."
                )
            previous = float(previous_values[row_number, column_index])
            new = float(executed_values[row_number, column_index])
            if attribute_reasons:
                contributing_constraints = [
                    name
                    for name in constraint_names
                    if constraint_touched_values[name][row_number, column_index]
                ]
                constraint_before = {
                    name: float(
                        constraint_before_values[name][row_number, column_index]
                    )
                    for name in contributing_constraints
                }
                constraint_after = {
                    name: float(constraint_after_values[name][row_number, column_index])
                    for name in contributing_constraints
                }
                if attribute_strategy_reasons:
                    strategy_detail_code = strategy_detail_code_values[
                        row_number, column_index
                    ]
                    strategy_details = strategy_details_values[row_number, column_index]
                else:
                    strategy_detail_code, strategy_details = None, None
                reason = _classify_reason(
                    new=new,
                    previous=previous,
                    executed_desired=float(desired_values[row_number, column_index]),
                    executed_desired_prev=float(
                        desired_prev_values[row_number, column_index]
                    ),
                    executed_constrained=float(
                        constrained_values[row_number, column_index]
                    ),
                    signal_now=float(signal_diag_values[row_number, column_index]),
                    signal_prev=float(
                        signal_diag_prev_values[row_number, column_index]
                    ),
                    allocated_now=float(
                        allocated_diag_values[row_number, column_index]
                    ),
                    allocated_prev=float(
                        allocated_diag_prev_values[row_number, column_index]
                    ),
                    desired_diag_now=float(
                        desired_diag_values[row_number, column_index]
                    ),
                    desired_diag_prev=float(
                        desired_diag_prev_values[row_number, column_index]
                    ),
                    strategy_detail_code=strategy_detail_code,
                    strategy_details=strategy_details,
                    contributing_constraints=contributing_constraints,
                    constraint_before=constraint_before,
                    constraint_after=constraint_after,
                    tradability_touched=(
                        bool(tradability_touched_values[row_number, column_index])
                        if attribute_tradability
                        else False
                    ),
                    tradability_compliance_limited=(
                        bool(
                            tradability_compliance_limited_values[
                                row_number, column_index
                            ]
                        )
                        if attribute_tradability
                        else False
                    ),
                    turnover_touched=(
                        bool(turnover_touched_values[row_number, column_index])
                        if attribute_turnover
                        else False
                    ),
                    turnover_actively_limited=(
                        bool(turnover_actively_limited_values[row_number, column_index])
                        if attribute_turnover
                        else False
                    ),
                    stop_loss_triggered=(
                        bool(stop_loss_triggered_values[row_number, column_index])
                        if attribute_stop_loss
                        else False
                    ),
                    take_profit_triggered=(
                        bool(take_profit_triggered_values[row_number, column_index])
                        if attribute_take_profit
                        else False
                    ),
                    forced_liquidation=(
                        bool(forced_liquidation_values[row_number, column_index])
                        if attribute_forced_liquidation
                        else False
                    ),
                    drift_compliance_forced=(
                        bool(drift_compliance_forced_values[row_number, column_index])
                        if attribute_drift_compliance_forced
                        else False
                    ),
                    drift_compliance_pending=(
                        bool(drift_compliance_pending_values[row_number, column_index])
                        if attribute_drift_compliance_pending
                        else False
                    ),
                )
                trigger_reason_code = reason.trigger_code
                trigger_reason_detail_code = reason.trigger_detail_code
                trigger_reason_details = reason.trigger_details
                adjustment_reason_codes = reason.adjustment_codes
                adjustment_reason_details = reason.adjustment_details
                if attribute_position_origin:
                    position_strategy_origin_timestamp = (
                        position_origin_timestamp_values[row_number, column_index]
                    )
                    position_strategy_origin_code = position_origin_code_values[
                        row_number, column_index
                    ]
                    position_strategy_origin_details = position_origin_details_values[
                        row_number, column_index
                    ]
                else:
                    position_strategy_origin_timestamp = pd.NaT
                    position_strategy_origin_code = None
                    position_strategy_origin_details = None
            else:
                trigger_reason_code = None
                trigger_reason_detail_code = None
                trigger_reason_details = None
                adjustment_reason_codes = None
                adjustment_reason_details = None
                position_strategy_origin_timestamp = pd.NaT
                position_strategy_origin_code = None
                position_strategy_origin_details = None
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "previous_weight": previous,
                    "new_weight": new,
                    "weight_change": delta,
                    "side": "buy" if delta > 0 else "sell",
                    "action": _classify_action(previous, new),
                    "trigger_reason_code": trigger_reason_code,
                    "trigger_reason_detail_code": trigger_reason_detail_code,
                    "trigger_reason_details": trigger_reason_details,
                    "adjustment_reason_codes": adjustment_reason_codes,
                    "adjustment_reason_details": adjustment_reason_details,
                    "position_strategy_origin_timestamp": (
                        position_strategy_origin_timestamp
                    ),
                    "position_strategy_origin_code": position_strategy_origin_code,
                    "position_strategy_origin_details": (
                        position_strategy_origin_details
                    ),
                    "reference_price": price,
                    "traded_notional": notional,
                    "commission": commission,
                    "spread_cost": spread,
                    "slippage_cost": slippage,
                    "total_cost": commission + spread + slippage,
                }
            )
    if not records:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    result = pd.DataFrame.from_records(records)[TRADE_LOG_COLUMNS]
    # Cast every str|None reason column to plain "object" dtype explicitly:
    # pandas' default string-dtype inference can otherwise silently take
    # over a column that mixes real strings with `None` across rows
    # (missing values become NaN instead of None), which every `is None`
    # check elsewhere in this pipeline relies on not happening.
    for column in (
        "trigger_reason_code",
        "trigger_reason_detail_code",
        "trigger_reason_details",
        "adjustment_reason_codes",
        "adjustment_reason_details",
        "position_strategy_origin_code",
        "position_strategy_origin_details",
    ):
        column_values = result[column].astype(object)
        result[column] = column_values.where(column_values.notna(), None)
    return result
