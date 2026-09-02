"""Stateful mean reversion around a trailing, zero-centered indicator.

Five interchangeable indicators (``indicator=``) can drive the same
entry/exit/stop state machine: a rolling z-score (the default), a
Bollinger-band-relative deviation (NOT the traditional [0, 1]-ranged %B --
see ``bollinger``'s own branch in :func:`_centered_indicator`), RSI,
distance to a moving average, or a rolling percentile rank -- see
:data:`INDICATORS`. Only three (``zscore``, ``rsi``, ``percentile``) are
offered as a primary choice in the dashboard/lab UI (see
:data:`UI_INDICATORS`); ``bollinger``/``distance_ma`` remain fully valid,
tested constructor arguments for internal/research use. Every indicator is
first converted to a zero-centered series (negative = oversold/long
candidate, positive = overbought/short candidate) by
:func:`_centered_indicator`; the state machine itself
(:func:`_walk_positions_with_reasons`) only ever compares a threshold
against ``abs(value)`` on that centered series, so it is completely
indicator-agnostic.

``stop_threshold`` is genuinely optional: pass an explicit ``None`` to
disable it entirely, or leave it unset to use the chosen indicator's
default (see :data:`INDICATOR_DEFAULT_THRESHOLDS`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features._validation import boolean, choice, finite_real, positive_int
from quantlab.features.mean_reversion import (
    normalized_distance_to_mean,
    rolling_percentile_rank,
    rolling_zscore,
    rsi,
)
from quantlab.features.native_calendar import compute_native_then_align
from quantlab.logging_config import get_logger
from quantlab.strategies.base import (
    PRICE_TYPES,
    UNSET,
    BaseStrategy,
    SignalReasons,
    UnsetType,
    register_strategy,
    validate_risk_control_parameters,
)

logger = get_logger(__name__)

#: Every indicator `MeanReversionStrategy` can drive its state machine from.
#: `bollinger`/`distance_ma` remain fully supported here (validated,
#: tested, computable) for internal/research use -- see `UI_INDICATORS` for
#: the narrower set actually offered as a primary choice in the dashboard
#: and Strategy Explorer lab.
INDICATORS = frozenset({"zscore", "bollinger", "rsi", "distance_ma", "percentile"})

#: Indicators offered as the primary choice in the dashboard sidebar and
#: the Strategy Explorer lab's indicator selector. `bollinger` is excluded
#: here -- it is `(price - rolling_mean) / (num_std * rolling_std)`, the
#: SAME rolling mean/std construction as `zscore` merely rescaled by
#: `bollinger_num_std`, so it rarely produces a meaningfully different
#: backtest. `distance_ma` is excluded for a DIFFERENT reason: it is
#: `(price - rolling_mean) / rolling_mean` -- normalized by the mean's own
#: level, with no volatility term at all -- so unlike `bollinger` it is
#: not simply a rescaled `zscore` and can diverge from it materially
#: whenever the asset's volatility regime shifts (a fixed % move away
#: from the mean reads as a smaller z-score in a high-volatility period
#: than in a low-volatility one, but reads as the same `distance_ma`
#: either way). Both stay valid, documented `INDICATORS` members and can
#: still be selected programmatically (YAML config, Python, robustness
#: sweeps) -- only the two main UI selectors are narrowed.
UI_INDICATORS: tuple[str, ...] = ("zscore", "rsi", "percentile")

#: Sensible (entry, exit, stop) defaults per indicator, used only when the
#: caller leaves `entry_threshold`/`exit_threshold` unset, or leaves
#: `stop_threshold` at its own sentinel default (see `MeanReversionStrategy
#: .__init__`) -- each indicator's centered series has a different natural
#: scale (a z-score's few units vs. a fractional distance vs. RSI's +/-50
#: range), so one shared default would be meaningless for at least four of
#: the five indicators. Public: the dashboard sidebar and the Strategy
#: Explorer lab both read these same numbers for their own widgets'
#: default values, rather than each hardcoding a second copy that could
#: silently drift from this one.
INDICATOR_DEFAULT_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "zscore": (2.0, 0.5, 4.0),
    "bollinger": (1.0, 0.2, 1.5),
    "rsi": (20.0, 10.0, 45.0),
    "distance_ma": (0.05, 0.01, 0.15),
    "percentile": (0.45, 0.10, 0.49),
}


def _centered_indicator(
    prices: pd.DataFrame,
    indicator: str,
    lookback_period: int,
    bollinger_num_std: float,
    symbol_calendars: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute ``indicator``'s zero-centered series for the state machine.

    Zero-centered: negative means below "normal" (a long/oversold
    candidate), positive means above "normal" (a short/overbought
    candidate) -- the same sign convention `rolling_zscore` already has,
    so the entry/exit/stop machine below never needs to know which
    indicator produced the series it is walking.

    ``symbol_calendars`` (the strategy's own, see `BaseStrategy.symbol_
    calendars`) routes every rolling computation through `compute_native_
    then_align` so each symbol's window is computed on its own native
    session dates rather than a closure-padded combined timeline --
    `None` short-circuits to a single vectorized computation.
    """
    combined_index = pd.DatetimeIndex(prices.index)
    if indicator == "zscore":
        return compute_native_then_align(
            lambda p: rolling_zscore(p, lookback_period),
            prices,
            symbol_calendars,
            combined_index,
        )
    if indicator == "bollinger":

        def _bollinger(p: pd.DataFrame) -> pd.DataFrame:
            mean = p.rolling(lookback_period, min_periods=lookback_period).mean()
            std = p.rolling(lookback_period, min_periods=lookback_period).std(ddof=1)
            return (p - mean) / (bollinger_num_std * std + EPSILON)

        return compute_native_then_align(
            _bollinger, prices, symbol_calendars, combined_index
        )
    if indicator == "rsi":
        return (
            compute_native_then_align(
                lambda p: rsi(p, lookback_period),
                prices,
                symbol_calendars,
                combined_index,
            )
            - 50.0
        )
    if indicator == "distance_ma":
        return compute_native_then_align(
            lambda p: normalized_distance_to_mean(p, lookback_period),
            prices,
            symbol_calendars,
            combined_index,
        )
    if indicator == "percentile":
        return (
            compute_native_then_align(
                lambda p: rolling_percentile_rank(p, lookback_period),
                prices,
                symbol_calendars,
                combined_index,
            )
            - 0.5
        )
    raise ValueError(f"Unknown indicator {indicator!r}.")  # unreachable after choice()


def _walk_positions_with_reasons(
    z: np.ndarray,
    entry: float,
    exit_: float,
    stop: float | None,
    long_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a centered indicator into persistent positions in ``{-1, 0, 1}``.

    Also records, at every index where the state actually transitions, a
    closed-set ``reason_detail_code`` and a human-readable ``reason_
    details`` string naming exactly which branch below fired -- computed
    in the SAME pass as the position itself, so ``generate_signals()``
    and ``explain_signals()`` can never disagree about why a position
    changed (both call this one function; neither reconstructs the other
    separately). Indicator-agnostic: ``z`` is whatever
    :func:`_centered_indicator` produced for the configured ``indicator``,
    never assumed to specifically be a z-score.
    """
    positions = np.zeros_like(z, dtype=float)
    detail_code = np.full(z.shape, None, dtype=object)
    details = np.full(z.shape, None, dtype=object)
    state = 0.0
    for index, value in enumerate(z):
        previous_state = state
        if np.isnan(value):
            state = 0.0
            if previous_state != state:
                detail_code[index] = "data_unavailable_exit"
                details[index] = "indicator unavailable (insufficient trailing history)"
        elif stop is not None and abs(value) > stop:
            state = 0.0
            if previous_state != state:
                signed_stop = stop if value > 0 else -stop
                detail_code[index] = "stop_loss_exit"
                details[index] = (
                    f"indicator {value:.4f} breached stop threshold {signed_stop:.4f}"
                )
        elif state == 0.0:
            if value < -entry:
                state = 1.0
                detail_code[index] = "oversold_entry"
                details[index] = (
                    f"indicator {value:.4f} crossed entry threshold {-entry:.4f}"
                )
            elif value > entry and not long_only:
                state = -1.0
                detail_code[index] = "overbought_entry"
                details[index] = (
                    f"indicator {value:.4f} crossed entry threshold {entry:.4f}"
                )
        elif (state == 1.0 and value > -exit_) or (state == -1.0 and value < exit_):
            state = 0.0
            threshold = -exit_ if previous_state == 1.0 else exit_
            detail_code[index] = "mean_reversion_exit"
            details[index] = (
                f"indicator {value:.4f} crossed exit threshold {threshold:.4f}"
            )
        positions[index] = state
    return positions, detail_code, details


@register_strategy("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    """Trade deviations from a chosen indicator until exit or stop thresholds."""

    def __init__(
        self,
        lookback_period: int = 20,
        indicator: str = "zscore",
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
        stop_threshold: float | UnsetType | None = UNSET,
        bollinger_num_std: float = 2.0,
        long_only: bool = True,
        price_type: str = "adjusted_close",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        values = self.validate_parameters(
            {
                "lookback_period": lookback_period,
                "indicator": indicator,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "stop_threshold": stop_threshold,
                "bollinger_num_std": bollinger_num_std,
                "long_only": long_only,
                "price_type": price_type,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        )
        self.lookback_period = values["lookback_period"]
        self.indicator = values["indicator"]
        self.entry_threshold = values["entry_threshold"]
        self.exit_threshold = values["exit_threshold"]
        self.stop_threshold = values["stop_threshold"]
        self.bollinger_num_std = values["bollinger_num_std"]
        self.long_only = values["long_only"]
        self.price_type = values["price_type"]
        self.stop_loss_pct = values["stop_loss_pct"]
        self.take_profit_pct = values["take_profit_pct"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the indicator choice, thresholds and direction mode."""
        values = dict(parameters)
        values["lookback_period"] = positive_int(
            values["lookback_period"], name="lookback_period", minimum=2
        )
        values["indicator"] = choice(
            values["indicator"], name="indicator", options=INDICATORS
        )
        values["bollinger_num_std"] = finite_real(
            values["bollinger_num_std"],
            name="bollinger_num_std",
            minimum=0.0,
            strict=True,
        )
        values["long_only"] = boolean(values["long_only"], name="long_only")
        values["price_type"] = choice(
            values["price_type"], name="price_type", options=PRICE_TYPES
        )
        values["stop_loss_pct"], values["take_profit_pct"] = (
            validate_risk_control_parameters(
                values["stop_loss_pct"], values["take_profit_pct"]
            )
        )

        entry_threshold = values["entry_threshold"]
        exit_threshold = values["exit_threshold"]
        stop_threshold = values["stop_threshold"]
        default_entry, default_exit, default_stop = INDICATOR_DEFAULT_THRESHOLDS[
            values["indicator"]
        ]
        if entry_threshold is None:
            entry_threshold = default_entry
        if exit_threshold is None:
            exit_threshold = default_exit
        if isinstance(stop_threshold, UnsetType):
            # Not passed at all -> use this indicator's default stop.
            stop_threshold = default_stop
        # else: an explicit stop_threshold=None means "disabled" and is
        # left as None; an explicit float is used as-is.

        values["entry_threshold"] = finite_real(
            entry_threshold, name="entry_threshold", minimum=0.0, strict=True
        )
        values["exit_threshold"] = finite_real(
            exit_threshold, name="exit_threshold", minimum=0.0
        )
        if stop_threshold is not None:
            values["stop_threshold"] = finite_real(
                stop_threshold, name="stop_threshold", minimum=0.0
            )
        else:
            values["stop_threshold"] = None
        if values["entry_threshold"] <= values["exit_threshold"]:
            raise ValueError("entry_threshold must exceed exit_threshold.")
        if (
            values["stop_threshold"] is not None
            and values["stop_threshold"] <= values["entry_threshold"]
        ):
            raise ValueError("stop_threshold must exceed entry_threshold.")
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return the trailing state machine's position for every asset."""
        prices = self._prices(data)
        indicator = _centered_indicator(
            prices,
            self.indicator,
            self.lookback_period,
            self.bollinger_num_std,
            self.symbol_calendars,
        )
        signals = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns, dtype=float
        )
        for symbol in prices.columns:
            positions, _, _ = _walk_positions_with_reasons(
                indicator[symbol].to_numpy(dtype=float),
                entry=self.entry_threshold,
                exit_=self.exit_threshold,
                stop=self.stop_threshold,
                long_only=self.long_only,
            )
            signals[symbol] = positions
        return self._validate_signals(signals, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons:
        """Explain each position transition from the same indicator walk.

        Recomputes the centered indicator and re-walks the identical
        state machine used by ``generate_signals()`` (same function, same
        parameters, same ``data``) -- a pure, deterministic recomputation,
        never a cache of a prior call, so it can never look ahead or
        drift from the positions actually produced.
        """
        prices = self._prices(data)
        indicator = _centered_indicator(
            prices,
            self.indicator,
            self.lookback_period,
            self.bollinger_num_std,
            self.symbol_calendars,
        )
        # Built as plain numpy object arrays and handed to the DataFrame
        # constructor with an explicit dtype=object: assigning column by
        # column (`frame[symbol] = array`) lets pandas' string-dtype
        # inference silently promote a None/str object column to its new
        # StringDtype, which represents "missing" as NaN instead of the
        # None _validate_signal_reasons requires -- constructing the
        # whole frame at once with dtype=object forced avoids that.
        detail_code_values = np.empty(prices.shape, dtype=object)
        details_values = np.empty(prices.shape, dtype=object)
        for column_index, symbol in enumerate(prices.columns):
            _, symbol_detail_code, symbol_details = _walk_positions_with_reasons(
                indicator[symbol].to_numpy(dtype=float),
                entry=self.entry_threshold,
                exit_=self.exit_threshold,
                stop=self.stop_threshold,
                long_only=self.long_only,
            )
            detail_code_values[:, column_index] = symbol_detail_code
            details_values[:, column_index] = symbol_details
        detail_code = pd.DataFrame(
            detail_code_values, index=prices.index, columns=prices.columns, dtype=object
        )
        details = pd.DataFrame(
            details_values, index=prices.index, columns=prices.columns, dtype=object
        )
        return self._validate_signal_reasons(detail_code, details, prices)
