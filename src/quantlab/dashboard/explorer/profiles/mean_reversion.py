"""Strategy Explorer profile for ``mean_reversion``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from quantlab.dashboard.explorer.profile import (
    ParameterDoc,
    ResultsDiagnostics,
    StrategyProfile,
    register_profile,
)

if TYPE_CHECKING:
    from quantlab.config import ExperimentConfig
    from quantlab.reporting.sections import DiagnosticsSection

_OVERVIEW = """
Mean reversion bets that a price which has moved unusually far from its
own recent average will tend to move back toward it. Unlike pairs
trading (which trades a *relationship* between two assets), this is a
single-asset strategy: it watches one symbol's own trailing statistics
and trades against short-term overreaction in that symbol alone.

Typical horizon: several bars to a few dozen bars (often days to a few weeks
on daily data), set mostly by `lookback_period` (how "recent" is measured)
and how quickly a given instrument tends to snap back. Data needed: prices
for the traded symbol, long enough to build a stable rolling mean/std before
trading starts.
"""

_ECONOMIC_INTUITION = """
Some short-term price moves may be driven by order flow, liquidity
conditions or overreaction rather than a permanent repricing. The strategy
bets that sufficiently unusual deviations will move back toward a recent
statistical reference level. This premise is more plausible
on liquid, range-bound-ish instruments; it works poorly on an asset that
is genuinely re-rating to a new regime (see Limitations).
"""

_MATH = """
`generate_signals()`'s pipeline, in order:

1. **Centered indicator** -- one of three primary indicators
   (`indicator`), each turned into a zero-centered series where negative
   means "below normal" (a long candidate) and positive means "above
   normal" (a short candidate):
   - `zscore`: `(price - rolling_mean) / rolling_std` over `lookback_period`.
   - `rsi`: Wilder's RSI over `lookback_period`, minus `50` (RSI's own
     neutral midpoint) so oversold/overbought reads as negative/positive.
   - `percentile`: the price's trailing percentile rank within
     `lookback_period` (0 = lowest, 1 = highest), minus `0.5`.

   Two further indicators, `bollinger` and `distance_ma`, are implemented
   and fully usable (Python/YAML, robustness sweeps) but not offered in
   this UI: `bollinger` is `(price - rolling_mean) / (bollinger_num_std *
   rolling_std)` -- the same rolling mean/std construction as `zscore`,
   merely rescaled, so it rarely produces a materially different
   backtest. `distance_ma` is `(price - rolling_mean) / rolling_mean` --
   normalized by the mean's own level rather than volatility, so unlike
   `bollinger` it CAN diverge from `zscore` materially when the
   volatility regime shifts, even though it is excluded from this UI for
   the same "not offered as a primary choice" reason.
2. **State machine**, walked one bar at a time per symbol, identical
   regardless of which indicator fed it (it only ever compares a
   threshold against the indicator's absolute value):
   - If the indicator is undefined (insufficient trailing history), force
     flat.
   - If `|indicator| > stop_threshold` (when set), force flat regardless
     of current state.
   - Flat state: enter long when `indicator < -entry_threshold`; enter
     short when `indicator > entry_threshold` AND `long_only=False` --
     `long_only` gates the short-entry branch directly inside this state
     machine, so a short position is never entered in the first place
     when it is `True`.
   - In a position: exit when the indicator crosses back through
     `-exit_threshold` (long) or `exit_threshold` (short).

`entry_threshold`/`exit_threshold`/`stop_threshold` are on the CHOSEN
indicator's own scale -- a threshold of `2.0` means very different things
for `zscore` (2 standard deviations) vs. `rsi` (would mean RSI 48-52,
barely oversold at all) vs. `percentile` (meaningless above `0.5`). Left
unset, each defaults to a value sized for that specific indicator (see
each parameter's own doc below); switching `indicator` on an otherwise-
unchanged config silently keeps whatever thresholds were explicitly set,
which may no longer make sense on the new indicator's scale -- always
re-check thresholds after changing `indicator`.

The state is a strategy signal, not a final portfolio weight: allocation,
constraints, rebalancing and execution still act downstream. Unlike pairs
trading, this single-asset mean-reversion strategy has no hedge ratio or second
instrument, and it does not apply a stationarity test before opening a
position. It therefore relies on the selected indicator to identify potential
mean-reversion opportunities without first verifying that the underlying
series is stationary.
"""

_ASSUMPTIONS = """
**Economic**: the price's short-term deviations are assumed to be
temporary around a comparatively stable recent level, rather than the start of a
sustained re-rating. **Statistical**: the price series (or at least its
short-term behaviour) is closer to mean-reverting than to a random walk
or a trend -- see the lab's own ADF/Hurst diagnostics for whether that
actually holds on the chosen instrument and period. **Implementation**:
`lookback_period` is long enough to give a stable rolling mean/std, but
short enough that "recent average" still means something economically
(a 5-year lookback on a stock that re-rated 2 years ago is not a useful
reference level).
"""

_DIAGNOSTICS = """
The lab below compares the three primary indicators (RSI, rolling z-score,
rolling percentile rank) on the same price series so their differences on
identical data are directly visible -- Bollinger Bands and distance to
moving average are also implemented (see Mathematical definition) but not
shown in this comparison: Bollinger rarely diverges materially from
zscore (the same mean/std construction, merely rescaled), while distance
to moving average is left out for a different reason -- it is not
volatility-normalized, so it can diverge from zscore materially when the
volatility regime shifts -- plus the actual backtestable state machine
overlaid on the currently selected `indicator` and thresholds, plus
stationarity diagnostics (ADF,
half-life, Hurst exponent) on the actual instrument and period being
considered -- asking whether this sample is consistent with mean
reversion before looking at trade-level performance. These diagnostics
are sensitive to their estimator and test specification; they do not
validate the strategy.
"""

_INTERPRETATION = """
With QuantLab's default constant-only ADF regression and AIC lag selection,
a low p-value is evidence against a unit root on the tested sample; it is
not proof of stationarity. A Hurst estimate below 0.5 is a separate,
descriptive indication of anti-persistence, not a hypothesis test. A finite
half-life much longer than `lookback_period` suggests that the indicator's
reference window may be short relative to the estimated speed of reversion.
The entry/exit thresholds do not impose a maximum holding period: a position
can remain open until its exit, stop or missing-data condition is reached.
"""

_LIMITATIONS = """
**Regime shift**: an asset can permanently re-rate (a real fundamental
change) rather than mean-revert -- indicator-based entries have no way to
tell "overreaction" apart from "the mean has genuinely moved", and
`stop_threshold` only limits the indicator's own deviation tolerated
before an exit is requested when that happens, not the realized monetary
loss (gaps, execution delay, and a moving mean/volatility or costs can
still produce a larger loss than the indicator distance alone would
suggest).
**Trending markets**: mean reversion structurally underperforms during
sustained trends, since every "extreme" reading keeps getting more
extreme instead of reverting -- see the Hurst/ADF diagnostics to assess
whether the instrument is currently more likely to be mean-reverting or
trending.
**Transaction costs**: frequent small round trips (a natural consequence
of tight `entry_threshold`/`exit_threshold` gaps) are especially
vulnerable to transaction costs, which can quickly erode the typically
modest edge per trade.
**Parameter instability**: the "right" lookback/thresholds can drift over
time as an instrument's own volatility regime changes, and thresholds
tuned for one `indicator` are not portable to another (see Mathematical
definition).
"""

_REFERENCES = (
    "The [statsmodels ADF documentation](https://www.statsmodels.org/stable/"
    "generated/statsmodels.tsa.stattools.adfuller.html) specifies the unit-"
    "root null, constant/trend choices and AIC lag selection used by "
    "QuantLab's wrapper. It supports interpretation of the test, not the "
    "claim that a price-level mean-reversion strategy is profitable.\n\n"
    "Ernest P. Chan's [*Algorithmic Trading: Winning Strategies and Their "
    "Rationale*](https://onlinelibrary.wiley.com/doi/book/10.1002/"
    "9781118676998) (Wiley, 2013) gives a practical treatment of mean-"
    "reversion research and implementation, including stationarity, "
    "half-life and trading-rule considerations. It is practical guidance, "
    "not evidence that a particular configuration will remain profitable."
)

_PARAMETERS = [
    ParameterDoc(
        name="lookback_period",
        what="Trailing window (periods) used for the rolling mean/std (or "
        "RSI/percentile window) that defines the centered indicator.",
        where="Step 1 -- every subsequent decision is a function of this indicator.",
        why="Defines what 'normal' means: too short and it chases noise, "
        "too long and it stops describing the current regime.",
        default="20",
        typical_range="10-60 periods.",
        effect_increase="A smoother, more stable reference level, but "
        "slower to adapt if the instrument's own typical range has "
        "genuinely shifted.",
        effect_decrease="Faster adaptation to a shifting regime, but a "
        "noisier indicator more prone to false signals.",
        tradeoffs="Stability of the reference level vs. responsiveness to "
        "genuine regime change.",
        interactions="Sets the scale the lab's ADF/half-life diagnostics "
        "should be compared against -- a half-life much longer than "
        "lookback_period means the window is too short to capture a full "
        "reversion cycle.",
    ),
    ParameterDoc(
        name="indicator",
        what="Which zero-centered indicator drives the state machine. "
        "Primary choices: 'zscore', 'rsi' or 'percentile' (see "
        "Mathematical definition for each formula). 'bollinger' and "
        "'distance_ma' are also accepted (Python/YAML) but not offered in "
        "this UI: 'bollinger' is a close variant of zscore (the same "
        "rolling mean/std construction, merely rescaled), so it rarely "
        "diverges from it materially, but 'distance_ma' normalizes by the "
        "rolling mean's own level rather than volatility and so CAN "
        "diverge from zscore materially when the volatility regime shifts.",
        where="Step 1 -- determines what feeds every subsequent decision.",
        why="Different indicators make different bets about what 'unusual' "
        "means: a z-score is scale-free relative to recent volatility, RSI "
        "is a bounded oscillator based on the ratio of recent average "
        "gains to average losses, and percentile rank is a purely "
        "non-parametric 'how extreme relative to recent history'.",
        default="zscore",
        typical_range="One of 'zscore', 'rsi', 'percentile'.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="z-score standardizes deviations using the rolling mean "
        "and volatility, so its interpretation is most natural when the "
        "local distribution and volatility regime are reasonably stable. "
        "RSI and percentile are bounded and less sensitive to the asset's "
        "absolute price scale, but they compress information about the "
        "magnitude of deviations; percentile, in particular, captures "
        "rank rather than how far beyond an extreme the observation lies.",
        interactions="entry_threshold/exit_threshold/stop_threshold are on "
        "THIS indicator's own scale -- changing indicator without "
        "re-checking thresholds can silently produce a state machine that "
        "almost never trades (thresholds too wide for the new scale) or "
        "trades constantly (too narrow). bollinger_num_std only applies "
        "when indicator='bollinger'.",
    ),
    ParameterDoc(
        name="bollinger_num_std",
        what="Number of standard deviations the Bollinger bands extend "
        "from the rolling mean -- only used when indicator='bollinger', "
        "which is available (Python/YAML) but not offered in this UI's "
        "Indicator choice; see that parameter's own doc.",
        where="Step 1, bollinger branch only -- rescales the centered "
        "indicator so a threshold of 1.0 means 'price outside the bands'.",
        why="Sets how wide 'the bands' are, independent of the entry/exit/"
        "stop thresholds themselves.",
        default="2.0",
        typical_range="1.5-3.0.",
        effect_increase="Wider bands -- a given entry_threshold now "
        "requires a larger absolute price move to trigger.",
        effect_decrease="Narrower bands -- more sensitive entries for the "
        "same entry_threshold.",
        tradeoffs="Conventional Bollinger practice (2.0) vs. a "
        "deliberately wider/narrower band for this instrument's own "
        "volatility character.",
        interactions="Has no effect at all unless indicator='bollinger'.",
    ),
    ParameterDoc(
        name="entry_threshold",
        what="Indicator magnitude (on the chosen indicator's own scale) "
        "that opens a new position.",
        where="State machine, flat-state entry condition.",
        why="Sets how unusual a deviation has to be before it's worth trading.",
        default="Indicator-specific: 2.0 (zscore), 1.0 (bollinger), 20.0 "
        "(rsi, i.e. RSI below 30 or above 70), 0.05 (distance_ma, a 5% "
        "move), 0.45 (percentile, i.e. below the 5th or above the 95th "
        "percentile) -- applied only when left unset (None).",
        typical_range="Depends on indicator; see default above.",
        effect_increase="Fewer, more extreme entries -- higher conviction "
        "per trade, lower turnover.",
        effect_decrease="More frequent entries on smaller deviations -- "
        "more trades, more exposure to noise.",
        tradeoffs="Trade frequency vs. conviction per trade.",
        interactions="Must exceed exit_threshold; must be below "
        "stop_threshold when set. Its practical meaning changes entirely "
        "with indicator -- see that parameter's own doc.",
    ),
    ParameterDoc(
        name="exit_threshold",
        what="Indicator magnitude (crossed on the way back toward zero) "
        "that closes an open position.",
        where="State machine, in-position exit condition.",
        why="Decides how much of the reversion to capture before closing.",
        default="Indicator-specific: 0.5 (zscore), 0.2 (bollinger), 10.0 "
        "(rsi), 0.01 (distance_ma), 0.10 (percentile) -- applied only "
        "when left unset (None).",
        typical_range="Depends on indicator; see default above.",
        effect_increase="Exits earlier, leaving more of a full reversion "
        "uncaptured but reducing time-in-trade.",
        effect_decrease="Holds for a more complete reversion, at the cost "
        "of more time exposed to a reversal.",
        tradeoffs="Captured reversion vs. time-in-trade risk.",
        interactions="Must be strictly below entry_threshold.",
    ),
    ParameterDoc(
        name="stop_threshold",
        what="Indicator magnitude that force-closes a position regardless "
        "of direction -- protection against a deviation that keeps "
        "widening instead of reverting (may indicate a regime shift or "
        "model breakdown rather than an ordinary fluctuation).",
        where="State machine, after unavailable-data handling and before "
        "entry or normal exit logic.",
        why="Limits the indicator's own deviation tolerated before "
        "requesting an exit when the mean-reversion premise itself has "
        "broken down for this instrument -- it does not cap the realized "
        "monetary loss (gaps, execution delay, and a moving mean/"
        "volatility or costs can still produce a larger loss than the "
        "indicator distance alone would suggest).",
        default="Indicator-specific: 4.0 (zscore), 1.5 (bollinger), 45.0 "
        "(rsi), 0.15 (distance_ma), 0.49 (percentile) -- applied when this "
        "parameter is left out entirely. Pass an explicit stop_threshold="
        "None to disable the stop altogether (the state machine then never "
        "force-closes on indicator magnitude, only on the ordinary exit "
        "condition or missing data).",
        typical_range="Typically 1.5-2x entry_threshold; omit the "
        "parameter to use the indicator-specific default, or pass None to "
        "disable it.",
        effect_increase="More room before an exit is requested -- fewer "
        "stop-outs on noise, larger potential loss per trade.",
        effect_decrease="Tighter risk control, more prone to being stopped "
        "out by a temporary overshoot that would otherwise have reverted.",
        tradeoffs="Downside protection vs. premature stop-outs.",
        interactions="Must be strictly greater than entry_threshold.",
    ),
    ParameterDoc(
        name="long_only",
        what="Whether short entries are structurally disabled.",
        where="Gates the short-entry branch directly inside the state "
        "machine (step 2) -- a short position is never entered in the "
        "first place when True, there is no separate final clip.",
        why="Many portfolios/mandates cannot or should not short. "
        "Enabling long_only also changes turnover and exposure because "
        "only the negative-indicator entry side can generate positions.",
        default="True",
        typical_range="Boolean.",
        effect_increase="N/A (boolean).",
        effect_decrease="N/A (boolean).",
        tradeoffs="True removes short-side opportunities but avoids "
        "short-specific costs/constraints (borrow, uptick rules); False "
        "represents reversion in both directions but adds short-side risk.",
        interactions="With False, entry_threshold/exit_threshold/"
        "stop_threshold apply symmetrically to both the long and short side.",
    ),
    ParameterDoc(
        name="stop_loss_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens this symbol's REAL executed position -- an "
        "additional, independent risk control on top of stop_threshold "
        "(which limits the INDICATOR's own deviation, not a monetary "
        "loss).",
        where="Applied after the backtest allocator/constraints/"
        "rebalancing/execution -- on the position actually held, not on "
        "this strategy's raw signal (a signal is not necessarily a "
        "realized position). See `quantlab.backtesting.accounting."
        "_detect_stop_loss_take_profit`.",
        why="stop_threshold protects against the mean-reversion premise "
        "itself breaking down (the indicator keeps widening); stop_loss_pct "
        "protects against realized monetary loss regardless of what the "
        "indicator says, e.g. from gaps, execution delay or costs.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="More room before a forced exit -- fewer stop-outs "
        "on ordinary volatility, larger potential realized loss per trade.",
        effect_decrease="Tighter monetary risk control, more prone to "
        "being stopped out by a temporary adverse move.",
        tradeoffs="Realized-loss protection vs. premature exits. Evaluated "
        "on GROSS (pre-cost) return -- QuantLab's execution cost model is "
        "portfolio-level only, so an exact net-of-cost trigger is not "
        "presently computable; this is a disclosed design convention, not "
        "a universal definition.",
        interactions="Independent of entry_threshold/exit_threshold/"
        "stop_threshold -- both mechanisms can be active at once, or "
        "either alone. Once triggered, no immediate re-entry at a rebased "
        "price -- flat until the position's next real entry.",
    ),
    ParameterDoc(
        name="take_profit_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens this symbol's REAL executed position on the "
        "favorable side -- locks in a gain rather than waiting for "
        "exit_threshold's ordinary mean-reversion exit.",
        where="Same mechanism as stop_loss_pct, opposite direction.",
        why="Realizes a gain directly once a target is reached, instead "
        "of depending on the indicator reverting all the way back through "
        "exit_threshold (which may give back some of the gain first).",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="Lets more of a favorable move run before locking it in.",
        effect_decrease="Locks in gains earlier, potentially forfeiting "
        "further upside.",
        tradeoffs="Locking in gains early vs. capturing a larger reversion.",
        interactions="Independent of stop_loss_pct and the entry/exit/"
        "stop_threshold family; see stop_loss_pct's own doc for the "
        "shared gross-return/re-entry conventions.",
    ),
    ParameterDoc(
        name="price_type",
        what="Which price series ('adjusted_close' or 'close') "
        "generate_signals() reads. Execution/costs always use the raw "
        "close regardless.",
        where="Feeds the centered indicator computation in step 1.",
        why="A split or large dividend shows up as a price jump in raw "
        "close but not in adjusted close -- unadjusted, it would look "
        "exactly like an extreme z-score deviation.",
        default="adjusted_close",
        typical_range="adjusted_close (recommended) or close.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="adjusted_close avoids false signals from corporate "
        "actions; close matches what was literally quoted, useful mainly "
        "for auditing.",
        interactions="A dividend/split near the current date would "
        "otherwise register as a large, entirely spurious entry signal.",
    ),
]


def _lab(st: Any) -> None:
    from quantlab.dashboard.explorer.labs.mean_reversion import render

    render(st)


@dataclass(frozen=True)
class MeanReversionDiagnostics:
    """State-machine activity + stationarity, one row per traded symbol.

    ``indicators``/``states`` carry the full per-symbol series for the
    Results-tab chart -- the report's own table (``_report_section``) is a
    snapshot only, mirroring ``PairDiagnostics``'s own table-vs-chart split.
    """

    indicator: str
    summary: pd.DataFrame
    indicators: dict[str, pd.Series]
    states: dict[str, pd.Series]
    entry_threshold: float
    exit_threshold: float
    stop_threshold: float | None


def _compute_diagnostics(
    data: pd.DataFrame, cfg: ExperimentConfig
) -> MeanReversionDiagnostics:
    from quantlab.data.base import price_matrix
    from quantlab.features.mean_reversion import half_life as compute_half_life
    from quantlab.features.stationarity import adf_test, hurst_exponent
    from quantlab.strategies.mean_reversion import (
        MeanReversionStrategy,
        _centered_indicator,
    )

    # Built from the SAME constructor call the real backtest made (not a
    # re-derivation of defaults/threshold-resolution here) -- see
    # MeanReversionStrategy's own None-vs-unset stop_threshold semantics,
    # easy to get subtly wrong by hand.
    strategy = MeanReversionStrategy(**cfg.strategy_parameters)
    # Engine-injected context (see BaseStrategy.symbol_calendars's own
    # docstring) -- set here too, exactly as BacktestEngine.run() does,
    # so this diagnostic's indicator/signals are computed on each symbol's
    # own native calendar rather than silently falling back to the
    # closure-padded combined timeline under a mixed-calendar universe.
    strategy.symbol_calendars = {
        instrument.symbol: instrument.calendar for instrument in cfg.data.instruments
    }
    price_type = cfg.strategy.signal_price_type
    prices = price_matrix(data, adjusted=price_type != "close")
    indicator = _centered_indicator(
        prices,
        strategy.indicator,
        strategy.lookback_period,
        strategy.bollinger_num_std,
        strategy.symbol_calendars,
    )
    state = strategy.generate_signals(data)
    reasons = strategy.explain_signals(data)

    rows = []
    indicators: dict[str, pd.Series] = {}
    states: dict[str, pd.Series] = {}
    for symbol in prices.columns:
        symbol_state = state[symbol]
        symbol_detail = reasons.detail_code[symbol]
        tested = prices[symbol].dropna()
        adf = adf_test(tested) if len(tested) >= 2 else None
        rows.append(
            {
                "Symbol": symbol,
                "Time in position": float((symbol_state != 0.0).mean()),
                "Entries": int(
                    symbol_detail.isin(["oversold_entry", "overbought_entry"]).sum()
                ),
                "Stop exits": int((symbol_detail == "stop_loss_exit").sum()),
                "ADF p-value": adf.pvalue if adf is not None else float("nan"),
                # Same wording/threshold as render_stationarity_card's own
                # verdict, using ADFResult's own reject_null (never a second,
                # independently-chosen significance level).
                "Verdict": (
                    ("Reject H0" if adf.reject_null else "Cannot reject H0")
                    if adf is not None
                    else "n/a"
                ),
                "Half-life": compute_half_life(tested)
                if len(tested) >= 2
                else float("inf"),
                "Hurst": hurst_exponent(tested) if len(tested) >= 2 else float("nan"),
            }
        )
        indicators[symbol] = indicator[symbol]
        states[symbol] = symbol_state
    summary = pd.DataFrame(rows).set_index("Symbol")
    return MeanReversionDiagnostics(
        indicator=strategy.indicator,
        summary=summary,
        indicators=indicators,
        states=states,
        entry_threshold=strategy.entry_threshold,
        exit_threshold=strategy.exit_threshold,
        stop_threshold=strategy.stop_threshold,
    )


def _render_diagnostics(st: Any, result: MeanReversionDiagnostics) -> None:
    from quantlab.dashboard.explorer.shared_components import (
        centered_indicator_threshold_overlay,
        render_price_chart,
    )

    st.subheader("Stationarity diagnostics & State machine")
    st.caption(
        f"indicator = **{result.indicator}**. Full-sample ADF/half-life/Hurst "
        "per symbol -- descriptive, not a validated backtest result on "
        "their own. Verdict uses the same H0 (unit root) rejection rule as "
        "the interactive lab's own stationarity card."
    )
    st.dataframe(result.summary, width="stretch")
    symbol = st.selectbox(
        "Symbol", list(result.indicators), key="mr_results_diag_symbol"
    )
    threshold_series, line_colors = centered_indicator_threshold_overlay(
        result.indicators[symbol],
        f"{result.indicator} indicator",
        entry_threshold=result.entry_threshold,
        exit_threshold=result.exit_threshold,
        stop_threshold=result.stop_threshold,
    )
    render_price_chart(
        st,
        threshold_series,
        title=f"{symbol}: Centered '{result.indicator}' indicator with "
        "entry/exit/stop thresholds",
        yaxis_title="Centered indicator",
        colors=line_colors,
    )
    render_price_chart(
        st,
        {"Position (state)": result.states[symbol]},
        title=f"{symbol}: state signal",
        yaxis_title="Signal state",
    )


def _report_section(result: MeanReversionDiagnostics) -> DiagnosticsSection:
    from quantlab.reporting.sections import DiagnosticsSection

    table = result.summary.reset_index()
    return DiagnosticsSection(
        table=table,
        note=(
            f"Mean reversion state-machine activity (indicator={result.indicator}) "
            "and full-sample stationarity diagnostics, one row per traded "
            "symbol. Descriptive, not a validation of profitability."
        ),
    )


register_profile(
    StrategyProfile(
        strategy_name="mean_reversion",
        display_name="Mean Reversion",
        category="Mean reversion",
        overview_md=_OVERVIEW,
        economic_intuition_md=_ECONOMIC_INTUITION,
        mathematical_definition_md=_MATH,
        assumptions_md=_ASSUMPTIONS,
        diagnostics_md=_DIAGNOSTICS,
        interpretation_md=_INTERPRETATION,
        limitations_md=_LIMITATIONS,
        references_md=_REFERENCES,
        parameters=_PARAMETERS,
        lab=_lab,
        results_diagnostics=ResultsDiagnostics(
            key="mean_reversion_diagnostics",
            compute=_compute_diagnostics,
            render=_render_diagnostics,
            report_section=_report_section,
        ),
    )
)
