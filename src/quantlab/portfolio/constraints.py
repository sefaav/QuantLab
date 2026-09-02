"""Pointwise portfolio constraints applied to target weights."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import PortfolioConfig
from quantlab.constants import EPSILON
from quantlab.exceptions import InvalidConfigurationError
from quantlab.portfolio._validation import (
    boolean,
    finite_real,
    positive_int,
    validate_frame,
)
from quantlab.portfolio.position_sizing import (
    gross_exposure,
    net_exposure,
    renormalize_within_cap,
)


@dataclass(frozen=True)
class ConstraintTouch:
    """Per-constraint provenance from :meth:`ConstraintSet.apply_with_provenance`.

    ``touched`` is True at every ``(date, symbol)`` cell this constraint
    changed by more than ``EPSILON`` at ANY point during constraint
    resolution -- including repeated passes inside the dust-cleanup
    fixed-point loop (a cumulative OR across every application).

    ``before`` holds the weight immediately before the FIRST pass that
    ever changed a given cell; ``after`` holds the weight immediately
    after the LAST pass that actually changed it -- never a snapshot from
    a later pass that left the cell untouched (its value may have moved
    for an unrelated reason between two passes of THIS constraint, and
    attributing that movement to this constraint would be wrong). Both
    are used only to build human-readable reason text, never to
    redetermine ``touched`` itself.

    ``direct`` is a cumulative OR (same convention as ``touched``)
    restricted to cells whose OWN value triggered this constraint's
    clip/drop decision at some point, as opposed to a cell only
    redimensioned as a downstream consequence (redistribution/rescaling
    of the survivors). For constraints with no redistribution concept
    (``maximum_gross_exposure``, ``maximum_leverage``,
    ``maximum_net_exposure``, ``long_only`` -- uniform whole-row
    rescales), ``direct == touched`` always.
    """

    touched: pd.DataFrame
    before: pd.DataFrame
    after: pd.DataFrame
    direct: pd.DataFrame


def _mark_touched(
    touched: dict[str, ConstraintTouch] | None,
    name: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    direct_this_pass: pd.DataFrame | None = None,
) -> None:
    """Record one constraint's effect, cumulatively, when tracking is on.

    ``touched`` (the mask) is a cumulative OR across every application of
    this constraint during the call. ``before`` is refreshed only the
    first time a cell is ever touched; ``after`` is refreshed only on a
    pass that actually retouches the cell -- a later no-op pass (e.g.
    once the dust-cleanup loop has converged, or a pass where some OTHER
    constraint moved this cell instead) must not overwrite an earlier,
    informative before/after pair with an unrelated snapshot.

    ``direct_this_pass``, when given, is the real (peek-based, no
    reconstructed threshold) predicate for which cells THIS constraint's
    own clip/drop decision fired on, at exactly this pass -- it is
    combined with ``changed`` before being OR-ed into the cumulative
    ``direct`` mask. ``None`` (the default) means every changed cell is
    direct (constraints with no redistribution concept).
    """
    if touched is None:
        return
    changed = (after - before).abs() > EPSILON
    direct_this_pass_mask = (
        changed if direct_this_pass is None else (direct_this_pass & changed)
    )
    if name in touched:
        existing = touched[name]
        first_touch_this_pass = changed & ~existing.touched
        touched[name] = ConstraintTouch(
            touched=existing.touched | changed,
            before=existing.before.where(~first_touch_this_pass, before),
            after=existing.after.where(~changed, after),
            direct=existing.direct | direct_this_pass_mask,
        )
    else:
        touched[name] = ConstraintTouch(
            touched=changed, before=before, after=after, direct=direct_this_pass_mask
        )


@dataclass(frozen=True)
class ConstraintSet:
    """Immutable collection of optional target-portfolio constraints."""

    maximum_weight: float | None = None
    minimum_weight: float | None = None
    maximum_gross_exposure: float | None = None
    maximum_net_exposure: float | None = None
    maximum_leverage: float | None = None
    maximum_positions: int | None = None
    long_only: bool = False

    def __post_init__(self) -> None:
        """Validate direct construction independently of Pydantic config."""
        for name in ("maximum_weight", "maximum_gross_exposure", "maximum_leverage"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    finite_real(value, name=name, minimum=0.0, strict=True),
                )
        for name in ("minimum_weight", "maximum_net_exposure"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, finite_real(value, name=name, minimum=0.0)
                )
        if self.maximum_positions is not None:
            object.__setattr__(
                self,
                "maximum_positions",
                positive_int(self.maximum_positions, name="maximum_positions"),
            )
        object.__setattr__(self, "long_only", boolean(self.long_only, name="long_only"))
        if (
            self.minimum_weight is not None
            and self.maximum_weight is not None
            and self.minimum_weight > self.maximum_weight
        ):
            raise InvalidConfigurationError(
                "minimum_weight must not exceed maximum_weight."
            )

    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Return finite weights satisfying every configured constraint."""
        return self._apply_impl(weights, None)

    def apply_with_provenance(
        self, weights: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, ConstraintTouch]]:
        """Same computation as :meth:`apply`, plus per-constraint provenance.

        For each configured constraint, records which cells it actually
        changed. The weight computation itself is identical to
        :meth:`apply` -- both delegate to the same ``_apply_impl``, which
        only records provenance when asked to. This is real provenance
        captured directly from the actual execution, not a parallel
        reconstruction, so it can never diverge from what ``apply()``
        itself would have produced.
        """
        touched: dict[str, ConstraintTouch] = {}
        result = self._apply_impl(weights, touched)
        return result, touched

    def _apply_impl(
        self, weights: pd.DataFrame, touched: dict[str, ConstraintTouch] | None
    ) -> pd.DataFrame:
        out = validate_frame(weights, name="weights").copy().astype(float)
        if self.long_only:
            before = out
            out = out.clip(lower=0.0)
            _mark_touched(touched, "long_only", before, out)
        if self.maximum_positions is not None:
            before = out
            pre_drop_gross = gross_exposure(out)
            after_cap = _cap_positions(out, self.maximum_positions)
            direct_this_pass = after_cap.ne(before)
            out = _redistribute_to_target(after_cap, pre_drop_gross)
            _mark_touched(
                touched,
                "maximum_positions",
                before,
                out,
                direct_this_pass=direct_this_pass,
            )
        if self.minimum_weight is not None:
            before = out
            pre_drop_gross = gross_exposure(out)
            after_drop = out.where(out.abs() >= self.minimum_weight, 0.0)
            direct_this_pass = after_drop.ne(before)
            out = _redistribute_to_target(after_drop, pre_drop_gross)
            _mark_touched(
                touched,
                "minimum_weight",
                before,
                out,
                direct_this_pass=direct_this_pass,
            )
        if self.maximum_weight is not None:
            before = out
            pre_cap_gross = gross_exposure(out)
            after_clip = out.astype(float).clip(
                -self.maximum_weight, self.maximum_weight
            )
            direct_this_pass = after_clip.ne(before)
            out = renormalize_within_cap(
                out, target_gross=pre_cap_gross, cap=self.maximum_weight
            )
            _mark_touched(
                touched,
                "maximum_weight",
                before,
                out,
                direct_this_pass=direct_this_pass,
            )

        pre_exposure_cap = out.copy()
        out = self._apply_exposure_caps(out, touched)
        if self.minimum_weight is not None:
            out = self._clean_dust_to_fixed_point(out, touched)
            before = out
            out = _rescue_needless_full_liquidation(
                out,
                pre_exposure_cap,
                minimum_weight=self.minimum_weight,
                maximum_weight=self.maximum_weight,
                maximum_gross_exposure=self.maximum_gross_exposure,
                maximum_leverage=self.maximum_leverage,
                maximum_net_exposure=self.maximum_net_exposure,
            )
            _mark_touched(touched, "minimum_weight", before, out)
        self._assert_satisfied(out)
        return out

    def _apply_exposure_caps(
        self,
        weights: pd.DataFrame,
        touched: dict[str, ConstraintTouch] | None = None,
    ) -> pd.DataFrame:
        out = weights
        if self.maximum_gross_exposure is not None:
            before = out
            out = _cap_gross(out, self.maximum_gross_exposure)
            _mark_touched(touched, "maximum_gross_exposure", before, out)
        if self.maximum_leverage is not None:
            before = out
            out = _cap_gross(out, self.maximum_leverage)
            _mark_touched(touched, "maximum_leverage", before, out)
        if self.maximum_net_exposure is not None:
            before = out
            out = _cap_net(out, self.maximum_net_exposure)
            _mark_touched(touched, "maximum_net_exposure", before, out)
        return out

    def _clean_dust_to_fixed_point(
        self,
        weights: pd.DataFrame,
        touched: dict[str, ConstraintTouch] | None = None,
    ) -> pd.DataFrame:
        """Repeat dust removal because exposure caps can create new dust."""
        assert self.minimum_weight is not None
        out = weights
        for _ in range(max(out.shape[1] + 1, 1)):
            before = out.copy()
            step_before = out
            pre_drop_gross = gross_exposure(out)
            after_drop = out.where(out.abs() >= self.minimum_weight, 0.0)
            direct_this_pass = after_drop.ne(step_before)
            out = _redistribute_to_target(after_drop, pre_drop_gross)
            _mark_touched(
                touched,
                "minimum_weight",
                step_before,
                out,
                direct_this_pass=direct_this_pass,
            )
            if self.maximum_weight is not None:
                step_before = out
                pre_cap_gross = gross_exposure(out)
                after_clip = out.astype(float).clip(
                    -self.maximum_weight, self.maximum_weight
                )
                direct_this_pass = after_clip.ne(step_before)
                out = renormalize_within_cap(
                    out, target_gross=pre_cap_gross, cap=self.maximum_weight
                )
                _mark_touched(
                    touched,
                    "maximum_weight",
                    step_before,
                    out,
                    direct_this_pass=direct_this_pass,
                )
            out = self._apply_exposure_caps(out, touched)
            if np.allclose(out.to_numpy(), before.to_numpy(), atol=1e-10, rtol=0.0):
                return out
        raise InvalidConfigurationError(
            "Portfolio constraints did not converge to a fixed point."
        )

    def _assert_satisfied(self, weights: pd.DataFrame) -> None:
        """Verify that interactions between constraints preserved invariants."""
        tolerance = 1e-7
        violations: list[str] = []
        if self.long_only and (weights < -tolerance).to_numpy().any():
            violations.append("long_only")
        if (
            self.maximum_positions is not None
            and ((weights.abs() > EPSILON).sum(axis=1) > self.maximum_positions).any()
        ):
            violations.append("maximum_positions")
        if self.minimum_weight is not None:
            dust = (weights.abs() > EPSILON) & (
                weights.abs() < self.minimum_weight - tolerance
            )
            if dust.to_numpy().any():
                violations.append("minimum_weight")
        if (
            self.maximum_weight is not None
            and (weights.abs() > self.maximum_weight + tolerance).to_numpy().any()
        ):
            violations.append("maximum_weight")
        gross = weights.abs().sum(axis=1)
        for name, cap in (
            ("maximum_gross_exposure", self.maximum_gross_exposure),
            ("maximum_leverage", self.maximum_leverage),
        ):
            if cap is not None and (gross > cap + tolerance).any():
                violations.append(name)
        if (
            self.maximum_net_exposure is not None
            and (
                weights.sum(axis=1).abs() > self.maximum_net_exposure + tolerance
            ).any()
        ):
            violations.append("maximum_net_exposure")
        if violations:
            raise InvalidConfigurationError(
                "Could not satisfy portfolio constraints: "
                + ", ".join(sorted(set(violations)))
                + "."
            )


def _rescue_needless_full_liquidation(
    out: pd.DataFrame,
    entering: pd.DataFrame,
    *,
    minimum_weight: float,
    maximum_weight: float | None,
    maximum_gross_exposure: float | None,
    maximum_leverage: float | None,
    maximum_net_exposure: float | None,
) -> pd.DataFrame:
    """Recover a compliant non-empty candidate after full dust liquidation.

    The heuristic first tries the largest single position, then the best
    equal-sized opposite-sign pair. It intentionally does not solve a general
    subset-optimisation problem.
    """
    all_zero = (out.abs() <= EPSILON).all(axis=1)
    entering_nonzero = (entering.abs() > EPSILON).any(axis=1)
    rescue_rows = out.index[all_zero & entering_nonzero]
    if rescue_rows.empty:
        return out

    single_caps = [
        cap
        for cap in (
            maximum_weight,
            maximum_gross_exposure,
            maximum_leverage,
            maximum_net_exposure,
        )
        if cap is not None
    ]
    single_asset_cap = min(single_caps) if single_caps else None
    pair_caps = [
        cap for cap in (maximum_gross_exposure, maximum_leverage) if cap is not None
    ]
    pair_cap = min(pair_caps) / 2.0 if pair_caps else None
    if maximum_weight is not None:
        pair_cap = maximum_weight if pair_cap is None else min(pair_cap, maximum_weight)

    rescued = out.copy()
    for index_label in rescue_rows:
        row = entering.loc[index_label]
        column = row.abs().idxmax()
        single_magnitude = abs(row[column])
        if single_asset_cap is not None:
            single_magnitude = min(single_magnitude, single_asset_cap)
        if single_asset_cap is None or single_magnitude >= minimum_weight:
            rescued.loc[index_label, column] = single_magnitude * np.sign(row[column])
            continue

        positive = row[row > EPSILON].sort_values(ascending=False)
        negative = row[row < -EPSILON].sort_values()
        best: tuple[Hashable, Hashable, float] | None = None
        for positive_column, positive_value in positive.items():
            for negative_column, negative_value in negative.items():
                magnitude = min(positive_value, -negative_value)
                if pair_cap is not None:
                    magnitude = min(magnitude, pair_cap)
                if magnitude < minimum_weight:
                    continue
                if best is None or magnitude > best[2]:
                    best = (
                        positive_column,
                        negative_column,
                        float(magnitude),
                    )
        if best is not None:
            positive_column, negative_column, magnitude = best
            rescued.loc[index_label, positive_column] = magnitude
            rescued.loc[index_label, negative_column] = -magnitude
    return rescued


def _redistribute_to_target(
    weights: pd.DataFrame, target_gross: pd.Series
) -> pd.DataFrame:
    """Rescale surviving positions toward each row's prior gross exposure."""
    current = gross_exposure(weights)
    scale = (target_gross / current.where(current > EPSILON)).fillna(1.0)
    return weights.mul(scale, axis=0)


def _cap_positions(weights: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """Keep only the largest absolute weights in each row."""

    def _row(row: pd.Series) -> pd.Series:
        nonzero = row[row.abs() > EPSILON]
        if len(nonzero) <= max_positions:
            return row
        keep = nonzero.abs().nlargest(max_positions).index
        capped = row.copy()
        capped.loc[~capped.index.isin(keep)] = 0.0
        return capped

    return weights.apply(_row, axis=1)


def _cap_gross(weights: pd.DataFrame, maximum: float) -> pd.DataFrame:
    """Uniformly scale rows whose gross exposure exceeds ``maximum``."""
    gross = gross_exposure(weights)
    scale = (maximum / gross.where(gross > EPSILON)).clip(upper=1.0).fillna(1.0)
    return weights.mul(scale, axis=0)


def _cap_net(weights: pd.DataFrame, maximum: float) -> pd.DataFrame:
    """Uniformly scale rows whose absolute net exposure exceeds ``maximum``."""
    net = net_exposure(weights).abs()
    scale = (maximum / net.where(net > EPSILON)).clip(upper=1.0).fillna(1.0)
    return weights.mul(scale, axis=0)


def constraints_from_config(portfolio_config: PortfolioConfig) -> ConstraintSet:
    """Build a validated constraint set from portfolio configuration."""
    return ConstraintSet(
        maximum_weight=portfolio_config.maximum_weight,
        minimum_weight=portfolio_config.target_minimum_weight,
        maximum_gross_exposure=portfolio_config.maximum_gross_exposure,
        maximum_net_exposure=portfolio_config.maximum_net_exposure,
        maximum_leverage=portfolio_config.maximum_leverage,
        maximum_positions=portfolio_config.target_maximum_positions,
        long_only=portfolio_config.long_only,
    )
