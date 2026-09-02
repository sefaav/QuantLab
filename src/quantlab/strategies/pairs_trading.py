"""ADF-gated pairs trading on a trailing price-level relationship."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab.constants import EPSILON
from quantlab.data.calendar import is_session_day
from quantlab.exceptions import StrategyError
from quantlab.features._validation import (
    boolean,
    choice,
    finite_real,
    numeric_pandas,
    positive_int,
    same_axes,
)
from quantlab.features.mean_reversion import rolling_percentile_rank, rolling_zscore
from quantlab.features.mean_reversion import rsi as _price_rsi
from quantlab.features.pairs_diagnostics import spread as compute_spread
from quantlab.features.stationarity import adf_test
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
from quantlab.strategies.mean_reversion import INDICATOR_DEFAULT_THRESHOLDS

logger = get_logger(__name__)

#: Indicators `PairsTradingStrategy` can drive its state machine from --
#: mirrors mean_reversion's `UI_INDICATORS` (the 3 indicators that can
#: meaningfully diverge), applied to the spread residual instead of a raw
#: price. No `bollinger`/`distance_ma` here: pairs trading never offered
#: them, and mean_reversion's own analysis (they rarely diverge from
#: `zscore`) applies just as much to a spread.
INDICATORS = ("zscore", "rsi", "percentile")


def _centered_spread_indicator(
    spread: pd.Series, indicator: str, window: int
) -> pd.Series:
    """Compute ``indicator``'s zero-centered series for the spread residual.

    Mirrors `quantlab.strategies.mean_reversion._centered_indicator`'s
    dispatch and sign convention (negative = below normal, positive =
    above normal), but for a single spread Series that can legitimately be
    zero or negative (unlike a price) -- `rsi`/`rolling_percentile_rank`
    are called with `strictly_positive=False` for exactly that reason.
    """
    if indicator == "zscore":
        return rolling_zscore(spread, window)
    if indicator == "rsi":
        return _price_rsi(spread, window, strictly_positive=False) - 50.0
    if indicator == "percentile":
        return rolling_percentile_rank(spread, window, strictly_positive=False) - 0.5
    raise ValueError(f"Unknown indicator {indicator!r}.")  # unreachable after choice()


def adf_pvalue(series: pd.Series) -> float | None:
    """Return an ADF p-value, or ``None`` when the test is inconclusive.

    Thin convenience wrapper over :func:`quantlab.features.stationarity.
    adf_test`, kept here since the stationarity gate below only ever needs
    the raw p-value, not the full structured result.
    """
    result = adf_test(series)
    return result.pvalue if result is not None else None


def _ols_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return intercept and slope for ``y = intercept + slope * x``."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have identical shapes.")
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2:
        return np.nan, np.nan
    clean_x = x_values[mask]
    clean_y = y_values[mask]
    if float(np.ptp(clean_x)) <= EPSILON:
        return np.nan, np.nan
    design = np.column_stack([np.ones(len(clean_x)), clean_x])
    coefficients, *_ = np.linalg.lstsq(design, clean_y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the OLS slope of y on x with an intercept."""
    return _ols_coefficients(x, y)[1]


def rolling_hedge_parameters(
    a: pd.Series, b: pd.Series, window: int, dynamic: bool
) -> tuple[pd.Series, pd.Series]:
    """Return trailing OLS intercept and slope without pre-formation values."""
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise TypeError("a and b must be pandas Series.")
    validated_a = numeric_pandas(a, name="a", strictly_positive=True)
    validated_b = numeric_pandas(b, name="b", strictly_positive=True)
    same_axes(validated_a, validated_b, names=("b",))
    length = positive_int(window, name="window", minimum=2)
    use_dynamic = boolean(dynamic, name="dynamic")

    intercept = pd.Series(np.nan, index=validated_a.index, dtype=float)
    beta = pd.Series(np.nan, index=validated_a.index, dtype=float)
    if len(validated_a) <= length:
        return intercept, beta

    if not use_dynamic:
        alpha_value, beta_value = _ols_coefficients(
            validated_b.iloc[:length].to_numpy(dtype=float),
            validated_a.iloc[:length].to_numpy(dtype=float),
        )
        intercept.iloc[length:] = alpha_value
        beta.iloc[length:] = beta_value
        return intercept, beta

    a_values = validated_a.to_numpy(dtype=float)
    b_values = validated_b.to_numpy(dtype=float)
    for position in range(length, len(validated_a)):
        start = position - length
        alpha_value, beta_value = _ols_coefficients(
            b_values[start:position], a_values[start:position]
        )
        intercept.iloc[position] = alpha_value
        beta.iloc[position] = beta_value
    return intercept, beta


def _rolling_hedge_ratio(
    a: pd.Series, b: pd.Series, window: int, dynamic: bool
) -> pd.Series:
    """Return the trailing OLS hedge slope."""
    return rolling_hedge_parameters(a, b, window, dynamic)[1]


def _walk_pairs_positions(
    indicator: np.ndarray,
    tradable: np.ndarray,
    entry: float,
    exit_: float,
    stop: float | None,
) -> np.ndarray:
    """Convert the spread's centered indicator into positions in ``{-1, 0, 1}``."""
    positions, _, _ = _walk_pairs_positions_with_reasons(
        indicator, tradable, entry, exit_, stop
    )
    return positions


def _walk_pairs_positions_with_reasons(
    indicator: np.ndarray,
    tradable: np.ndarray,
    entry: float,
    exit_: float,
    stop: float | None,
    *,
    adf_gate_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the spread's centered indicator into positions in ``{-1, 0, 1}``.

    Also records, at every index where the state actually transitions, a
    closed-set ``reason_detail_code`` and a human-readable ``reason_
    details`` string naming exactly which branch fired -- computed in the
    SAME pass as the position itself (mirrors mean_reversion's
    ``_walk_positions_with_reasons``), so ``generate_signals()`` and
    ``explain_signals()`` can never disagree. The stationarity (ADF) gate
    is consulted ONLY inside the ``state == 0.0`` branch, exactly as in
    the original state machine -- it can silently block an entry but can
    never, by itself, force an exit, so it never gets its own
    ``reason_detail_code``; a successful entry mentions the gate as
    context in ``reason_details`` only.
    """
    if len(indicator) != len(tradable):
        raise ValueError("indicator and tradable must have the same length.")
    positions = np.zeros_like(indicator, dtype=float)
    detail_code = np.full(indicator.shape, None, dtype=object)
    details = np.full(indicator.shape, None, dtype=object)
    state = 0.0
    for index, value in enumerate(indicator):
        previous_state = state
        if not np.isfinite(value):
            state = 0.0
            if previous_state != state:
                detail_code[index] = "data_unavailable_exit"
                details[index] = (
                    "spread indicator unavailable (insufficient trailing history)"
                )
        elif stop is not None and abs(value) > stop:
            state = 0.0
            if previous_state != state:
                signed_stop = stop if value > 0 else -stop
                detail_code[index] = "stop_loss_exit"
                details[index] = (
                    f"spread indicator {value:.4f} breached stop threshold "
                    f"{signed_stop:.4f}"
                )
        elif state == 0.0:
            if bool(tradable[index]):
                gate_clause = (
                    "stationarity gate open" if adf_gate_enabled else "gate disabled"
                )
                if value < -entry:
                    state = 1.0
                    detail_code[index] = "spread_oversold_entry"
                    details[index] = (
                        f"spread indicator {value:.4f} crossed entry threshold "
                        f"{-entry:.4f} ({gate_clause})"
                    )
                elif value > entry:
                    state = -1.0
                    detail_code[index] = "spread_overbought_entry"
                    details[index] = (
                        f"spread indicator {value:.4f} crossed entry threshold "
                        f"{entry:.4f} ({gate_clause})"
                    )
        elif (state == 1.0 and value > -exit_) or (state == -1.0 and value < exit_):
            state = 0.0
            threshold = -exit_ if previous_state == 1.0 else exit_
            detail_code[index] = "mean_reversion_exit"
            details[index] = (
                f"spread indicator {value:.4f} crossed exit threshold {threshold:.4f}"
            )
        positions[index] = state
    return positions, detail_code, details


@register_strategy("pairs_trading")
class PairsTradingStrategy(BaseStrategy):
    """Trade the residual of a trailing price-level regression between two assets.

    The ADF test gates new entries (unless ``adf_pvalue_threshold=None``
    disables it). Open positions still follow their own exit/stop rules
    (on the chosen ``indicator``'s centered series of the spread), and any
    undefined indicator value forces the pair flat.
    """

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        formation_window: int = 252,
        indicator_window: int = 63,
        indicator: str = "zscore",
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
        stop_threshold: float | UnsetType | None = UNSET,
        dynamic_hedge_ratio: bool = True,
        adf_pvalue_threshold: float | None = 0.10,
        price_type: str = "adjusted_close",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        values = self.validate_parameters(
            {
                "symbol_a": symbol_a,
                "symbol_b": symbol_b,
                "formation_window": formation_window,
                "indicator_window": indicator_window,
                "indicator": indicator,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "stop_threshold": stop_threshold,
                "dynamic_hedge_ratio": dynamic_hedge_ratio,
                "adf_pvalue_threshold": adf_pvalue_threshold,
                "price_type": price_type,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
            }
        )
        self.symbol_a = values["symbol_a"]
        self.symbol_b = values["symbol_b"]
        self.formation_window = values["formation_window"]
        self.indicator_window = values["indicator_window"]
        self.indicator = values["indicator"]
        self.entry_threshold = values["entry_threshold"]
        self.exit_threshold = values["exit_threshold"]
        self.stop_threshold = values["stop_threshold"]
        self.dynamic_hedge_ratio = values["dynamic_hedge_ratio"]
        self.adf_pvalue_threshold = values["adf_pvalue_threshold"]
        self.price_type = values["price_type"]
        self.stop_loss_pct = values["stop_loss_pct"]
        self.take_profit_pct = values["take_profit_pct"]
        self._freeze_parameters()

    def position_groups(self) -> tuple[tuple[str, ...], ...] | None:
        """The two legs form one economic position for stop-loss/take-profit.

        See `BaseStrategy.position_groups()` -- a per-leg check would
        evaluate the wrong thing (e.g. treat a hedge leg's own gain,
        which OFFSETS the pair's real loss, as if it were an independent
        position).
        """
        return ((self.symbol_a, self.symbol_b),)

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate symbols, windows, thresholds and ADF confidence."""
        values = dict(parameters)
        for key in ("symbol_a", "symbol_b"):
            value = values[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string.")
            values[key] = value.strip().upper()
        if values["symbol_a"] == values["symbol_b"]:
            raise ValueError("symbol_a and symbol_b must differ.")
        values["formation_window"] = positive_int(
            values["formation_window"], name="formation_window", minimum=20
        )
        values["indicator_window"] = positive_int(
            values["indicator_window"], name="indicator_window", minimum=2
        )
        values["indicator"] = choice(
            values["indicator"], name="indicator", options=frozenset(INDICATORS)
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
        values["dynamic_hedge_ratio"] = boolean(
            values["dynamic_hedge_ratio"], name="dynamic_hedge_ratio"
        )
        if values["adf_pvalue_threshold"] is not None:
            values["adf_pvalue_threshold"] = finite_real(
                values["adf_pvalue_threshold"],
                name="adf_pvalue_threshold",
                minimum=0.0,
                strict=True,
            )
            if values["adf_pvalue_threshold"] >= 1.0:
                raise ValueError("adf_pvalue_threshold must be strictly below 1.")
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
        """Return signals for the two configured legs.

        The two legs' relative weights are scaled by the fitted hedge ratio
        (``beta``), not necessarily dollar-neutral.
        """
        prices = self._prices(data)
        for symbol in (self.symbol_a, self.symbol_b):
            if symbol not in prices.columns:
                raise StrategyError(
                    f"Pairs trading needs symbol '{symbol}' in the data; "
                    f"available: {list(prices.columns)}."
                )
        a, b, indicator, beta, tradable = self._native_pair_context(prices)
        state = pd.Series(
            _walk_pairs_positions(
                indicator.to_numpy(dtype=float),
                tradable,
                entry=self.entry_threshold,
                exit_=self.exit_threshold,
                stop=self.stop_threshold,
            ),
            index=prices.index,
            dtype=float,
        )

        raw_legs = pd.DataFrame(
            {
                self.symbol_a: state * a,
                self.symbol_b: -state * beta * b,
            },
            index=prices.index,
        )
        largest_leg = raw_legs.abs().max(axis=1).where(lambda value: value > EPSILON)
        pair_signals = raw_legs.div(largest_leg, axis=0)
        signals = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns, dtype=float
        )
        signals.loc[:, [self.symbol_a, self.symbol_b]] = pair_signals
        return self._validate_signals(signals, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons:
        """Explain each transition of the pair's shared position state.

        Recomputes the spread/indicator/ADF-gate exactly like
        ``generate_signals()`` and re-walks the SAME state machine
        (shared helper function) so the two can never disagree. Both
        legs move together (one shared pair position), so they always
        carry the same reason on the same date; every other symbol in
        the universe stays ``None`` (this strategy never touches them).
        """
        prices = self._prices(data)
        for symbol in (self.symbol_a, self.symbol_b):
            if symbol not in prices.columns:
                raise StrategyError(
                    f"Pairs trading needs symbol '{symbol}' in the data; "
                    f"available: {list(prices.columns)}."
                )
        _, _, indicator, _, tradable = self._native_pair_context(prices)
        _, symbol_detail_code, symbol_details = _walk_pairs_positions_with_reasons(
            indicator.to_numpy(dtype=float),
            tradable,
            entry=self.entry_threshold,
            exit_=self.exit_threshold,
            stop=self.stop_threshold,
            adf_gate_enabled=self.adf_pvalue_threshold is not None,
        )

        detail_code = np.full(
            (len(prices.index), len(prices.columns)), None, dtype=object
        )
        details = np.full((len(prices.index), len(prices.columns)), None, dtype=object)
        a_index = prices.columns.get_loc(self.symbol_a)
        b_index = prices.columns.get_loc(self.symbol_b)
        detail_code[:, a_index] = symbol_detail_code
        detail_code[:, b_index] = symbol_detail_code
        details[:, a_index] = symbol_details
        details[:, b_index] = symbol_details

        return self._validate_signal_reasons(
            pd.DataFrame(
                detail_code, index=prices.index, columns=prices.columns, dtype=object
            ),
            pd.DataFrame(
                details, index=prices.index, columns=prices.columns, dtype=object
            ),
            prices,
        )

    def decision_signal(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Expose the discrete pair state, stripped of price/beta rescaling.

        ``generate_signals()``'s own output mixes the discrete decision
        (``state`` in ``{-1, 0, 1}``, which only changes on a real
        entry/exit/reversal) with purely mechanical rescaling from
        ``a``/``b`` (price) and ``beta`` (hedge ratio, recomputed daily
        when ``dynamic_hedge_ratio=True``) -- a plain "did the final
        signal change" comparison cannot tell a real new decision apart
        from pure drift. This recomputes ``state`` the same way
        ``explain_signals()`` does (shared helper, same pass, never
        diverges from ``generate_signals()``'s own internal ``state``) and
        rebroadcasts it onto ``symbol_a``/``symbol_b`` (0 elsewhere) --
        diagnostic only, see :meth:`BaseStrategy.decision_signal`.
        """
        prices = self._prices(data)
        for symbol in (self.symbol_a, self.symbol_b):
            if symbol not in prices.columns:
                raise StrategyError(
                    f"Pairs trading needs symbol '{symbol}' in the data; "
                    f"available: {list(prices.columns)}."
                )
        _, _, indicator, _, tradable = self._native_pair_context(prices)
        state, _, _ = _walk_pairs_positions_with_reasons(
            indicator.to_numpy(dtype=float),
            tradable,
            entry=self.entry_threshold,
            exit_=self.exit_threshold,
            stop=self.stop_threshold,
        )
        decision = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns, dtype=float
        )
        decision[self.symbol_a] = state
        decision[self.symbol_b] = state
        return self._validate_decision_signal(decision, prices)

    def _native_pair_context(
        self, prices: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, np.ndarray]:
        """Return ``(a, b, indicator, beta, tradable)`` for the shared state machine.

        The hedge-ratio fit, spread/indicator and periodic ADF re-check are
        all rolling-window computations over the two legs jointly -- per
        ``BaseStrategy.symbol_calendars``, they are computed on the
        INTERSECTION of both legs' own native session dates (never a date
        manufactured by one leg's own closure padding), then reindexed and
        forward-filled onto the combined timeline, exactly like a single-
        symbol native feature (see ``quantlab.features.native_calendar``).
        ``tradable`` (the state machine's own entry gate) is the AND of the
        ADF stationarity gate with "both legs open today" -- a closed leg
        must never allow a fresh entry on a stale price for that leg, even
        when the ADF gate alone would allow it; an already-open position's
        exit/stop rules are unaffected (the state machine only consults
        ``tradable`` for a NEW entry, never for an exit/stop).
        """
        a = prices[self.symbol_a]
        b = prices[self.symbol_b]
        calendars = self.symbol_calendars or {}
        calendar_a = calendars.get(self.symbol_a)
        calendar_b = calendars.get(self.symbol_b)
        combined_index = pd.DatetimeIndex(prices.index)
        open_a = (
            is_session_day(calendar_a, combined_index)
            if calendar_a is not None
            else np.ones(len(prices.index), dtype=bool)
        )
        open_b = (
            is_session_day(calendar_b, combined_index)
            if calendar_b is not None
            else np.ones(len(prices.index), dtype=bool)
        )
        both_open = open_a & open_b

        if bool(both_open.all()):
            intercept, beta = rolling_hedge_parameters(
                a, b, self.formation_window, self.dynamic_hedge_ratio
            )
            spread = compute_spread(a, b, intercept, beta)
            indicator = _centered_spread_indicator(
                spread, self.indicator, self.indicator_window
            )
            adf_gate = self._stationarity_gate(a, b)
            return a, b, indicator, beta, both_open & adf_gate

        native_index = prices.index[both_open]
        native_a, native_b = a.loc[native_index], b.loc[native_index]
        intercept, beta_native = rolling_hedge_parameters(
            native_a, native_b, self.formation_window, self.dynamic_hedge_ratio
        )
        spread_native = compute_spread(native_a, native_b, intercept, beta_native)
        indicator_native = _centered_spread_indicator(
            spread_native, self.indicator, self.indicator_window
        )
        adf_gate_native = self._stationarity_gate(native_a, native_b)

        fillable = pd.Series(~both_open, index=prices.index)
        beta = beta_native.reindex(prices.index)
        beta = beta.mask(fillable & beta.isna(), beta.ffill())
        indicator = indicator_native.reindex(prices.index)
        indicator = indicator.mask(fillable & indicator.isna(), indicator.ffill())
        adf_gate_series = pd.Series(adf_gate_native, index=native_index).reindex(
            prices.index
        )
        adf_gate_series = adf_gate_series.mask(
            fillable & adf_gate_series.isna(), adf_gate_series.ffill()
        )
        adf_gate = adf_gate_series.fillna(False).to_numpy(dtype=bool)
        return a, b, indicator, beta, both_open & adf_gate

    def _stationarity_gate(self, a: pd.Series, b: pd.Series) -> np.ndarray:
        """Test full trailing formation residuals at bounded intervals.

        Returns an all-``True`` gate (every date tradable) without running
        any ADF test when ``adf_pvalue_threshold is None`` -- the gate is
        disabled entirely, not merely widened.
        """
        if self.adf_pvalue_threshold is None:
            return np.ones(len(a), dtype=bool)
        pvalues = periodic_stationarity_pvalues(
            a,
            b,
            formation_window=self.formation_window,
            stride=self.indicator_window,
            dynamic_hedge_ratio=self.dynamic_hedge_ratio,
        )
        values = pvalues.to_numpy(dtype=float)
        gate = np.isfinite(values) & (values <= self.adf_pvalue_threshold)
        return cast(np.ndarray, gate)


def periodic_stationarity_pvalues(
    a: pd.Series,
    b: pd.Series,
    *,
    formation_window: int,
    stride: int,
    dynamic_hedge_ratio: bool,
) -> pd.Series:
    """ADF p-value of a single-window regression residual, rechecked periodically.

    Recomputed every ``stride`` positions starting at ``formation_window``
    and held constant between checkpoints (matching ``PairsTradingStrategy.
    _stationarity_gate``'s own periodic recheck, which this function IS --
    ``_stationarity_gate`` just thresholds it). With ``dynamic_hedge_ratio=
    True``, each checkpoint refits (intercept, beta) on its own trailing
    ``formation_window``-length window. With ``dynamic_hedge_ratio=False``,
    every checkpoint instead reuses the ONE (intercept, beta) fit once on
    the very first ``formation_window`` -- only the ADF test itself, not
    the regression, is redone at each checkpoint. Either way this is
    deliberately distinct from running ADF on a slice of ``spread(a, b,
    *rolling_hedge_parameters(...))``: with ``dynamic_hedge_ratio=True``
    that spread's hedge ratio is refit EVERY day (trailing
    ``formation_window``-length window ending at that day), so slicing it
    would test a residual built from a DIFFERENT regression than the
    checkpoint-window fit the strategy's own gate actually uses. Reused
    identically by :func:`quantlab.features.pairs_diagnostics.
    compute_pair_diagnostics` (its ``rolling_adf_pvalue`` field) so the
    Strategy Explorer's diagnostics agree with what the live strategy
    gates entries on whenever the data range, symbols, price type and
    parameters are identical.

    Returns ``NaN`` before the first checkpoint (``position <
    formation_window``) and at any checkpoint where the window has missing
    data or the regression is numerically degenerate.
    """
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise TypeError("a and b must be pandas Series.")
    same_axes(a, b, names=("b",))
    window_length = positive_int(formation_window, name="formation_window", minimum=2)
    check_stride = positive_int(stride, name="stride", minimum=1)
    use_dynamic = boolean(dynamic_hedge_ratio, name="dynamic_hedge_ratio")

    pvalue = pd.Series(np.nan, index=a.index, dtype=float)
    last_pvalue: float | None = None
    static_coefficients: tuple[float, float] | None = None
    if not use_dynamic and len(a) >= window_length:
        static_coefficients = _ols_coefficients(
            b.iloc[:window_length].to_numpy(dtype=float),
            a.iloc[:window_length].to_numpy(dtype=float),
        )
    for position in range(window_length, len(a)):
        if (position - window_length) % check_stride == 0:
            start = position - window_length
            window = pd.concat(
                {"a": a.iloc[start:position], "b": b.iloc[start:position]},
                axis=1,
            ).dropna()
            if len(window) != window_length:
                last_pvalue = None
            else:
                intercept, beta = (
                    _ols_coefficients(
                        window["b"].to_numpy(dtype=float),
                        window["a"].to_numpy(dtype=float),
                    )
                    if use_dynamic
                    else static_coefficients or (np.nan, np.nan)
                )
                if not np.isfinite(intercept) or not np.isfinite(beta):
                    last_pvalue = None
                else:
                    residual = window["a"] - intercept - beta * window["b"]
                    last_pvalue = adf_pvalue(residual)
        pvalue.iloc[position] = last_pvalue if last_pvalue is not None else np.nan
    return pvalue
