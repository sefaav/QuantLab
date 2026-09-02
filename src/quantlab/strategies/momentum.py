"""Trailing time-series and cross-sectional momentum strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features._validation import (
    boolean,
    choice,
    finite_real,
    non_negative_int,
    positive_int,
)
from quantlab.features.cross_sectional import select_top_bottom
from quantlab.features.momentum import momentum, volatility_adjusted_momentum
from quantlab.strategies.base import (
    PRICE_TYPES,
    BaseStrategy,
    SignalReasons,
    register_strategy,
    validate_risk_control_parameters,
)

_TIME_SERIES_SCALINGS = frozenset({"binary", "continuous", "volatility_adjusted"})


def _scaling(value: object, *, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"signal_scaling must be one of {sorted(allowed)}, got {value!r}."
        )
    return value


@register_strategy("time_series_momentum")
class TimeSeriesMomentumStrategy(BaseStrategy):
    """Map each asset's trailing momentum to a directional signal.

    ``binary`` uses only direction, ``continuous`` standardises momentum by
    its trailing dispersion, and ``volatility_adjusted`` divides momentum by
    trailing annualised volatility.
    """

    def __init__(
        self,
        lookback_period: int = 252,
        skip_period: int = 21,
        long_only: bool = True,
        signal_scaling: str = "binary",
        volatility_window: int = 63,
        periods_per_year: int = 252,
        price_type: str = "adjusted_close",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        values = self.validate_parameters(
            {
                "lookback_period": lookback_period,
                "skip_period": skip_period,
                "long_only": long_only,
                "signal_scaling": signal_scaling,
                "volatility_window": volatility_window,
                "periods_per_year": periods_per_year,
                "price_type": price_type,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        )
        self.lookback_period = values["lookback_period"]
        self.skip_period = values["skip_period"]
        self.long_only = values["long_only"]
        self.signal_scaling = values["signal_scaling"]
        self.volatility_window = values["volatility_window"]
        self.periods_per_year = values["periods_per_year"]
        self.price_type = values["price_type"]
        self.stop_loss_pct = values["stop_loss_pct"]
        self.take_profit_pct = values["take_profit_pct"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate momentum windows, scaling and annualisation."""
        values = dict(parameters)
        values["lookback_period"] = positive_int(
            values["lookback_period"], name="lookback_period"
        )
        values["skip_period"] = non_negative_int(
            values["skip_period"], name="skip_period"
        )
        if values["skip_period"] >= values["lookback_period"]:
            raise ValueError("skip_period must be smaller than lookback_period.")
        values["long_only"] = boolean(values["long_only"], name="long_only")
        values["signal_scaling"] = _scaling(
            values["signal_scaling"], allowed=_TIME_SERIES_SCALINGS
        )
        if values["signal_scaling"] == "continuous" and values["lookback_period"] < 2:
            raise ValueError(
                "lookback_period must be at least 2 for continuous scaling."
            )
        values["volatility_window"] = positive_int(
            values["volatility_window"], name="volatility_window", minimum=2
        )
        values["periods_per_year"] = positive_int(
            values["periods_per_year"], name="periods_per_year"
        )
        values["price_type"] = choice(
            values["price_type"], name="price_type", options=PRICE_TYPES
        )
        values["stop_loss_pct"], values["take_profit_pct"] = (
            validate_risk_control_parameters(
                values["stop_loss_pct"], values["take_profit_pct"]
            )
        )
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return per-asset trailing-momentum signals."""
        prices = self._prices(data)
        score = self._native_feature(
            prices, lambda p: momentum(p, self.lookback_period, self.skip_period)
        )

        if self.signal_scaling == "binary":
            signal = pd.DataFrame(
                np.sign(score), index=score.index, columns=score.columns
            )
        elif self.signal_scaling == "continuous":
            dispersion = self._native_feature(
                score,
                lambda s: s.rolling(
                    self.lookback_period, min_periods=min(20, self.lookback_period)
                ).std(ddof=1),
            )
            signal = (score / dispersion).clip(-1.0, 1.0)
        elif self.signal_scaling == "volatility_adjusted":
            # Delegates to the public helper (rather than recomputing
            # momentum/volatility inline) so a zero-volatility window is
            # masked to NaN -- never silently divided into +-inf, which
            # `.clip(-1, 1)` would otherwise turn into a false +-1.0 full
            # -conviction signal instead of the "no reliable read" it is.
            signal = self._native_feature(
                prices,
                lambda p: volatility_adjusted_momentum(
                    p,
                    self.lookback_period,
                    self.skip_period,
                    self.volatility_window,
                    self.periods_per_year,
                ),
            ).clip(-1.0, 1.0)
        else:  # pragma: no cover - constructor invariant
            raise RuntimeError(f"Unsupported signal scaling: {self.signal_scaling!r}.")

        if self.long_only:
            signal = signal.clip(lower=0.0)
        return self._validate_signals(signal, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons | None:
        """Explain each transition of a ``binary``-scaled momentum signal.

        Only ``signal_scaling == "binary"`` gets a specific attribution:
        that mode is a clean, discrete ``sign(momentum_score)`` (like
        ``trend_following``'s crossover), so "the score crossed zero" is
        a real, nameable event. ``"continuous"``/``"volatility_adjusted"``
        produce a continuously varying value that changes on almost every
        rebalance date -- the generic ``"signal X -> Y since last
        rebalance"`` text already IS the complete explanation there (the
        magnitude itself is the story); inventing a label repeated on
        nearly every row would be exactly the kind of generic-reason-
        dressed-as-specific this feature exists to avoid, so this
        deliberately returns ``None`` for those two modes.
        """
        if self.signal_scaling != "binary":
            return None
        prices = self._prices(data)
        score = self._native_feature(
            prices, lambda p: momentum(p, self.lookback_period, self.skip_period)
        )
        signal = pd.DataFrame(np.sign(score), index=score.index, columns=score.columns)
        if self.long_only:
            signal = signal.clip(lower=0.0)
        final = signal.fillna(0.0).to_numpy()
        previous = np.vstack([np.zeros((1, final.shape[1])), final[:-1]])
        score_values = score.to_numpy()

        detail_code = np.empty(final.shape, dtype=object)
        details = np.empty(final.shape, dtype=object)
        for row in range(final.shape[0]):
            for col in range(final.shape[1]):
                delta = final[row, col] - previous[row, col]
                if abs(delta) <= EPSILON:
                    continue
                score_value = score_values[row, col]
                if final[row, col] > EPSILON:
                    detail_code[row, col] = "positive_momentum_entry"
                    details[row, col] = (
                        f"momentum score {score_value:.4f} turned positive"
                    )
                elif final[row, col] < -EPSILON:
                    detail_code[row, col] = "negative_momentum_entry"
                    details[row, col] = (
                        f"momentum score {score_value:.4f} turned negative"
                    )
                else:
                    detail_code[row, col] = "momentum_exit"
                    details[row, col] = f"momentum score {score_value:.4f} crossed zero"

        return self._validate_signal_reasons(
            pd.DataFrame(
                detail_code, index=prices.index, columns=prices.columns, dtype=object
            ),
            pd.DataFrame(
                details, index=prices.index, columns=prices.columns, dtype=object
            ),
            prices,
        )


_CROSS_SECTIONAL_SCALINGS = frozenset({"binary", "continuous"})


def _cross_sectional_magnitude(
    score: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    """Per-date, per-leg rank-based magnitude, never zero for a selected asset.

    Computed separately within each of the two SELECTED legs (long:
    ``selection > 0``, short: ``selection < 0``), never across the whole
    cross-section. For the long leg: rank ascending by score (1 = weakest
    selected long, N = strongest), divided by the leg's own selected count
    N, so magnitude is non-decreasing in score and always in ``(0, 1]``
    (never exactly 0 for a selected asset, since the minimum attainable
    rank is 1, not 0). The short leg mirrors this: rank DESCENDING by
    score (the most negative -- the best short candidate -- gets the top
    rank N), so magnitude is non-increasing in score. Tied scores get the
    IDENTICAL rank -- the top of their shared tie group (pandas'
    ``rank(method="max")``) -- so magnitude depends only on each asset's
    own score, never on column/symbol order, and a leg with a single
    selected asset (or every selected score tied) resolves to a magnitude
    of exactly 1.0 for every tied member.
    """
    long_mask = selection.gt(0.0)
    short_mask = selection.lt(0.0)
    long_ranks = score.where(long_mask).rank(axis=1, method="max")
    short_ranks = score.where(short_mask).rank(axis=1, method="max", ascending=False)
    long_count = long_mask.sum(axis=1)
    short_count = short_mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        long_magnitude = long_ranks.div(long_count, axis=0)
        short_magnitude = short_ranks.div(short_count, axis=0)
    magnitude = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    magnitude = magnitude.where(~long_mask, long_magnitude)
    magnitude = magnitude.where(~short_mask, short_magnitude)
    return magnitude


@register_strategy("cross_sectional_momentum")
class CrossSectionalMomentumStrategy(BaseStrategy):
    """Select the strongest assets and optionally short the weakest.

    ``binary`` (default) gives every selected asset an identical signal
    magnitude (+-1). ``continuous`` scales each selected asset's SIGNAL
    magnitude by its RANK within its own selected leg (see
    :func:`_cross_sectional_magnitude`): within the long leg, the weakest
    selected name gets the smallest magnitude and the strongest gets the
    full +1; within the short leg (mirrored), the least-negative selected
    name gets the smallest magnitude and the most negative gets the full
    -1 -- guaranteed monotone in score by construction and never zero for
    a selected asset, while WHICH assets are selected (the top/bottom
    fraction cutoff itself) is unchanged. This is a SIGNAL magnitude, not
    a portfolio weight: the allocator downstream still determines the
    actual target weights. Only the ``signal_proportional`` allocator
    actually uses this magnitude -- the default ``equal_weight`` allocator
    discards it (``np.sign`` of the signal), making ``continuous`` behave
    identically to ``binary``.
    """

    def __init__(
        self,
        lookback_period: int = 252,
        skip_period: int = 21,
        top_fraction: float = 0.25,
        bottom_fraction: float = 0.25,
        long_short: bool = False,
        signal_scaling: str = "binary",
        price_type: str = "adjusted_close",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        values = self.validate_parameters(
            {
                "lookback_period": lookback_period,
                "skip_period": skip_period,
                "top_fraction": top_fraction,
                "bottom_fraction": bottom_fraction,
                "long_short": long_short,
                "signal_scaling": signal_scaling,
                "price_type": price_type,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        )
        self.lookback_period = values["lookback_period"]
        self.skip_period = values["skip_period"]
        self.top_fraction = values["top_fraction"]
        self.bottom_fraction = values["bottom_fraction"]
        self.long_short = values["long_short"]
        self.signal_scaling = values["signal_scaling"]
        self.price_type = values["price_type"]
        self.stop_loss_pct = values["stop_loss_pct"]
        self.take_profit_pct = values["take_profit_pct"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate momentum windows and disjoint selection fractions."""
        values = dict(parameters)
        values["lookback_period"] = positive_int(
            values["lookback_period"], name="lookback_period"
        )
        values["skip_period"] = non_negative_int(
            values["skip_period"], name="skip_period"
        )
        if values["skip_period"] >= values["lookback_period"]:
            raise ValueError("skip_period must be smaller than lookback_period.")
        values["top_fraction"] = finite_real(
            values["top_fraction"], name="top_fraction", minimum=0.0
        )
        values["bottom_fraction"] = finite_real(
            values["bottom_fraction"], name="bottom_fraction", minimum=0.0
        )
        if values["top_fraction"] > 1.0 or values["bottom_fraction"] > 1.0:
            raise ValueError("top_fraction and bottom_fraction must not exceed 1.")
        values["long_short"] = boolean(values["long_short"], name="long_short")
        if (
            values["long_short"]
            and values["top_fraction"] + values["bottom_fraction"] > 1.0
        ):
            raise ValueError(
                "top_fraction + bottom_fraction must not exceed 1 when "
                "long_short is enabled."
            )
        values["signal_scaling"] = _scaling(
            values["signal_scaling"], allowed=_CROSS_SECTIONAL_SCALINGS
        )
        values["price_type"] = choice(
            values["price_type"], name="price_type", options=PRICE_TYPES
        )
        values["stop_loss_pct"], values["take_profit_pct"] = (
            validate_risk_control_parameters(
                values["stop_loss_pct"], values["take_profit_pct"]
            )
        )
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return disjoint cross-sectional selections."""
        prices = self._prices(data)
        score = self._native_feature(
            prices, lambda p: momentum(p, self.lookback_period, self.skip_period)
        )
        bottom = self.bottom_fraction if self.long_short else 0.0
        selection = select_top_bottom(
            score, top_fraction=self.top_fraction, bottom_fraction=bottom
        )
        if self.signal_scaling == "continuous":
            magnitude = _cross_sectional_magnitude(score, selection)
            selection_values = selection.to_numpy(dtype=float)
            scaled = np.where(
                selection_values != 0.0,
                selection_values * magnitude.to_numpy(dtype=float),
                0.0,
            )
            signal = pd.DataFrame(
                scaled, index=selection.index, columns=selection.columns
            )
        else:
            signal = selection
        return self._validate_signals(signal, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons | None:
        """Explain each entry into/exit from the top or bottom selection.

        There is no persistent membership state at all (see
        ``select_top_bottom``'s own docstring: it recomputes the disjoint
        top/bottom groups from scratch at every date) -- a transition is
        simply "this symbol's selection changed since yesterday", read
        directly off the SAME ``select_top_bottom(...)`` call
        ``generate_signals()`` itself makes (the real function, not a
        reimplementation, so it can never diverge). ``select_top_bottom``
        does not expose each symbol's rank or the cutoff score, so those
        are not included here -- only the momentum score, which is real
        and directly available.

        Only ``signal_scaling == "binary"`` gets this attribution, same
        rationale as ``TimeSeriesMomentumStrategy``: under ``"continuous"``
        the executed WEIGHT still varies within an unchanged selection on
        almost every rebalance (the cross-sectional magnitude), so the
        generic "signal X -> Y since last rebalance" text already is the
        complete explanation there.
        """
        if self.signal_scaling != "binary":
            return None
        prices = self._prices(data)
        score = self._native_feature(
            prices, lambda p: momentum(p, self.lookback_period, self.skip_period)
        )
        bottom = self.bottom_fraction if self.long_short else 0.0
        selection = select_top_bottom(
            score, top_fraction=self.top_fraction, bottom_fraction=bottom
        )
        final = selection.fillna(0.0).to_numpy()
        previous = np.vstack([np.zeros((1, final.shape[1])), final[:-1]])
        score_values = score.to_numpy()

        detail_code = np.empty(final.shape, dtype=object)
        details = np.empty(final.shape, dtype=object)
        for row in range(final.shape[0]):
            for col in range(final.shape[1]):
                delta = final[row, col] - previous[row, col]
                if abs(delta) <= EPSILON:
                    continue
                score_value = score_values[row, col]
                if final[row, col] > EPSILON:
                    detail_code[row, col] = "entered_top_selection"
                    details[row, col] = f"momentum score {score_value:.4f}"
                elif final[row, col] < -EPSILON:
                    detail_code[row, col] = "entered_bottom_selection"
                    details[row, col] = f"momentum score {score_value:.4f}"
                elif previous[row, col] > EPSILON:
                    detail_code[row, col] = "left_top_selection"
                    details[row, col] = f"momentum score {score_value:.4f}"
                else:
                    detail_code[row, col] = "left_bottom_selection"
                    details[row, col] = f"momentum score {score_value:.4f}"

        return self._validate_signal_reasons(
            pd.DataFrame(
                detail_code, index=prices.index, columns=prices.columns, dtype=object
            ),
            pd.DataFrame(
                details, index=prices.index, columns=prices.columns, dtype=object
            ),
            prices,
        )
