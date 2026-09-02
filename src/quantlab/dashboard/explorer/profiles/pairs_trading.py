"""Strategy Explorer profile for ``pairs_trading``."""

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
    from quantlab.features.pairs_diagnostics import PairDiagnostics
    from quantlab.reporting.sections import DiagnosticsSection

_OVERVIEW = """
Pairs trading trades the *relationship* between two related assets, not
either asset's own direction. When the price of A drifts away from what
B's own movement would predict, the strategy bets on that gap ("the
spread") closing again. For the usual positive fitted beta this creates
opposite-signed legs; with a negative beta, the leg signs need not be one
long and one short. The signal construction encodes the fitted OLS slope
between its legs; it is NOT necessarily dollar-neutral, and QuantLab's
weight-based accounting does not simulate literal share quantities -- see
Mathematical definition. It seeks statistical
arbitrage / mean reversion in a relationship, the classic example being
two economically linked instruments (two banks, two miners, an ETF and
its underlying index) whose prices tend to move together.

Typical horizon: several bars to a few dozen bars per trade (often days to
weeks on daily data), depending on `indicator_window` and how quickly the
spread mean-reverts. Data needed: aligned prices
for both legs over a long enough history to fit a reliable relationship
(`formation_window` periods) before trading starts.
"""

_ECONOMIC_INTUITION = """
Two assets exposed to the same underlying economic driver (an industry,
a currency, an index) should move together over time. When one
temporarily overreacts -- to flow, sentiment, a stock-specific headline --
while the underlying driver hasn't actually changed, the gap between them
is expected to close as both assets re-anchor to that shared driver. This
is intended as a relative-value bet: it does not directly forecast whether the
*market* goes up or down, only that this specific pair's relationship is
temporarily distorted and will normalize.
"""

_MATH = """
`generate_signals()`'s pipeline, in order:

1. **Hedge ratio** -- `rolling_hedge_parameters(a, b, formation_window,
   dynamic_hedge_ratio)` fits `a = intercept + beta * b` by trailing OLS.
   If `dynamic_hedge_ratio=False`, this fit happens once on the first
   `formation_window` observations and is held constant afterward; if
   `True`, it is refit on every trailing `formation_window`-length window
   (so `beta`/`intercept` can drift as the relationship itself drifts).
2. **Spread** -- `spread = a - intercept - beta * b`, the residual of that
   fit: how far A actually sits from what the fitted relationship predicts.
3. **Centered indicator** -- one of three indicators (`indicator`, same
   choice and defaults as `mean_reversion`), applied to the spread residual
   instead of a raw price: `zscore` (`rolling_zscore(spread,
   indicator_window)`, the default), `rsi`, or `percentile`. Each is
   zero-centered the same way mean_reversion's indicator is (negative =
   spread below normal, positive = above normal).
4. **Stationarity gate** -- every `indicator_window` bars, the full
   trailing `formation_window` residual is ADF-tested; new entries are
   only allowed while the resulting p-value stays `<= adf_pvalue_threshold`
   (open positions are unaffected -- the gate blocks new entries only).
   Set `adf_pvalue_threshold=None` to disable the gate entirely (every
   date becomes tradable, subject only to the entry/exit/stop thresholds
   below).
5. **State machine** -- flat: enter long when `indicator < -entry_threshold`
   (and the gate is open), enter short when `indicator > entry_threshold`.
   In a position: exit when the indicator crosses back through
   `-exit_threshold` (long) / `exit_threshold` (short); force-flat if
   `|indicator| > stop_threshold` or the indicator becomes undefined.
6. **OLS-scaled legs** -- the discrete state (`{-1, 0, 1}`) is applied as
   `symbol_a: state * a`, `symbol_b: -state * beta * b`, then BOTH legs
   are divided by whichever one is larger in absolute value, so EACH leg
   individually is bounded to `[-1, 1]` -- this bounds each leg's own
   signal magnitude, it does NOT force the two legs' dollar exposures to
   be equal and opposite (that would require intercept == 0, generally
   false); combined gross exposure before portfolio-level allocation can
   run up to roughly 2.

This formula encodes the fitted share ratio in signal space, but QuantLab
accounts for portfolio weights and does not create or round share orders.
These are still strategy signals. The required `signal_proportional`
allocator converts them to target weights; constraints, volatility
targeting, rebalancing and execution then determine the weights actually
traded.
"""

_ASSUMPTIONS = """
**Economic**: the strategy assumes the two assets share a durable common
driver rather than only coincidental historical correlation. **Statistical**:
the fitted residual is assumed to be stationary and that property is
assumed to persist beyond each formation sample. **Implementation**: the hedge ratio
estimated over `formation_window` (or refit on every bar, if dynamic)
remains a useful description of the relationship;
transaction costs on both legs are small relative to the typical spread
move being captured.
"""

_DIAGNOSTICS = """
The lab below (and the Results tab, for an actual backtest) reports: return
correlation and a rolling version of it (a screening signal, not proof of
tradability); the hedge ratio series and its own stability (std of beta --
a relationship whose slope keeps changing makes the spread harder to
interpret, because both the spread itself and the hedge ratio used to
construct it are varying over time; this raw standard deviation is in
beta's own units, which depend on the pair's price scales and which symbol
is A vs. B -- compare it across different
formation_window/dynamic_hedge_ratio settings for the SAME pair, not
across different pairs); an exploratory full-sample ADF on the adaptively
constructed spread and Engle-Granger cointegration between the raw series
(two related but distinct questions); a *rolling* ADF
p-value so stationarity is checked throughout the sample, not only once
over the full history; and the spread's mean-reversion half-life.
"""

_INTERPRETATION = """
A low p-value is sample evidence against the relevant null, not a guarantee
that the relationship will persist. The full-sample ADF is exploratory and,
when `dynamic_hedge_ratio=True`, tests one series assembled from many rolling
regressions; that adaptive construction is not the same as a standard
single-regression residual test and can make the result look more stable.
The rolling ADF series uses the same formulas as the strategy's periodic
entry gate, but only matches its RESULT exactly for a single-calendar
pair -- under a mixed-calendar universe the live gate evaluates on the
intersection of both legs' own native session dates while this diagnostic
uses the full combined timeline, so the two can genuinely differ (see
docs/limitations.md). A shorter finite half-life relative to
`indicator_window` is more compatible with completing threshold round
trips, but does not guarantee that they occur or survive costs.
"""

_LIMITATIONS = """
**Structural break**: a merger, a regulatory change, or an index
reconstitution can permanently sever a relationship that looked stable for
years -- the strategy has no way to distinguish "temporarily wide spread"
from "the relationship is gone" except waiting for the ADF gate to close
new entries (already-open positions still follow their own exit/stop --
`stop_threshold` limits the indicator's own deviation tolerated before an
exit is requested, it does not cap the realized monetary loss: gaps,
execution delay, and a moving mean/volatility/hedge-ratio or costs can
still produce a larger loss than the indicator distance alone would
suggest).
**Crowding**: a well-known, liquid pair attracts other pairs traders,
which can compress the very edge the spread is supposed to capture.
**Costs**: two legs can mean two sets of transaction costs. QuantLab models
its configured commission, spread and slippage, but does not separately
model stock-borrow fees, financing rates or locate availability; those
short-side costs and constraints remain outside the result unless the user
approximates them in the configured costs.
**Weight-based execution**: the OLS-scaled leg formula is converted to target
weights; QuantLab does not maintain literal share counts or guarantee exact
share neutrality, particularly when adjusted signal prices differ from raw
execution reference prices.
**Unstable hedge ratio**: with `dynamic_hedge_ratio=True`, a beta that
swings a lot between refits makes the spread itself a moving target,
undermining the whole premise of trading a *stable* residual.
"""

_REFERENCES = (
    'Engle & Granger (1987), ["Co-integration and Error Correction: '
    'Representation, Estimation, and Testing"](https://doi.org/10.2307/'
    "1913236), *Econometrica* 55(2), 251-276, develops the cointegration and "
    "error-correction framework. QuantLab specifically calls the augmented "
    "Engle-Granger test documented by [statsmodels](https://www.statsmodels."
    "org/stable/generated/statsmodels.tsa.stattools.coint.html). These sources "
    "support the statistical tests, not QuantLab's thresholds or the "
    "profitability of a pair.\n\n"
    "Ernest P. Chan's [*Algorithmic Trading: Winning Strategies and Their "
    "Rationale*](https://onlinelibrary.wiley.com/doi/book/10.1002/"
    "9781118676998) (Wiley, 2013) provides a practical discussion of mean-"
    "reverting spreads, stationarity, cointegration and hedge-ratio "
    "construction. It complements the statistical sources above but does "
    "not establish that a particular pair or configuration is profitable."
)

_PARAMETERS = [
    ParameterDoc(
        name="symbol_a",
        what="The first leg of the pair.",
        where="Defines `a` in every equation above.",
        why="Pairs trading needs two named instruments to relate.",
        default="(required)",
        typical_range="A symbol present in the configured universe, with "
        "timestamps compatible with symbol_b.",
        effect_increase="N/A -- a selection, not a magnitude.",
        effect_decrease="N/A -- a selection, not a magnitude.",
        tradeoffs="Choosing a genuinely economically-related pair matters far "
        "more than any other parameter here.",
        interactions="Must differ from symbol_b; both must be present in the "
        "configured universe.",
    ),
    ParameterDoc(
        name="symbol_b",
        what="The second leg of the pair.",
        where="Defines `b` in every equation above.",
        why="See symbol_a.",
        default="(required)",
        typical_range="A symbol present in the configured universe, with "
        "timestamps compatible with symbol_a.",
        effect_increase="N/A -- a selection, not a magnitude.",
        effect_decrease="N/A -- a selection, not a magnitude.",
        tradeoffs="See symbol_a.",
        interactions="Must differ from symbol_a.",
    ),
    ParameterDoc(
        name="formation_window",
        what="Trailing window (periods) used to fit the hedge ratio and "
        "run the periodic ADF stationarity test.",
        where="Step 1 (hedge ratio fit) and step 4 (stationarity gate).",
        why="A relationship needs enough history to estimate reliably, but "
        "not so much that it stops describing the CURRENT relationship.",
        default="252",
        typical_range="~60-500 periods (roughly 3 months to 2 years of daily data).",
        effect_increase="A smoother, more stable hedge-ratio estimate, but "
        "slower to adapt if the relationship is genuinely changing; more "
        "data required before the strategy can trade at all.",
        effect_decrease="Faster adaptation to a changing relationship, but a "
        "noisier hedge-ratio estimate and a less powerful stationarity test "
        "(fewer observations per ADF run).",
        tradeoffs="Stability of the estimate vs. responsiveness to genuine "
        "regime change.",
        interactions="Must be smaller than the available history; interacts "
        "with dynamic_hedge_ratio (a short window refit every period reacts "
        "fast but noisily).",
    ),
    ParameterDoc(
        name="indicator_window",
        what="Trailing window (periods) used to compute the spread's "
        "centered indicator, and the cadence of the stationarity gate "
        "re-check.",
        where="Step 3 (centered indicator) and step 4 (gate re-check interval).",
        why="The indicator needs its own recent window to be a meaningful "
        "'how unusual is this right now' measure.",
        default="63",
        typical_range="~10-90 periods.",
        effect_increase="A smoother indicator, less sensitive to short-lived "
        "noise, but slower to flag a genuine new dislocation.",
        effect_decrease="A twitchier indicator that reacts fast to a fresh "
        "dislocation, but more prone to false entries from noise.",
        tradeoffs="Signal smoothness vs. responsiveness.",
        interactions="Should generally be smaller than formation_window (the "
        "indicator describes short-run deviation from a longer-run "
        "relationship, not the other way around).",
    ),
    ParameterDoc(
        name="indicator",
        what="Which zero-centered indicator of the spread residual drives "
        "the state machine: 'zscore' (default), 'rsi' or 'percentile' -- "
        "same three choices and defaults as mean_reversion's own "
        "`indicator`, applied here to the spread instead of a raw price.",
        where="Step 3 -- determines what feeds every subsequent decision.",
        why="Same rationale as mean_reversion's `indicator`: a z-score is "
        "scale-free relative to the spread's own recent volatility, RSI is "
        "a bounded oscillator, percentile rank is purely non-parametric.",
        default="zscore",
        typical_range="One of 'zscore', 'rsi', 'percentile'.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="See mean_reversion's `indicator` doc for the full "
        "zscore/rsi/percentile tradeoff discussion -- it applies "
        "identically here.",
        interactions="entry_threshold/exit_threshold/stop_threshold are on "
        "THIS indicator's own scale -- changing indicator without "
        "re-checking thresholds can silently produce a state machine that "
        "rarely or constantly trades.",
    ),
    ParameterDoc(
        name="entry_threshold",
        what="Indicator magnitude (on the chosen indicator's own scale) "
        "that opens a new position.",
        where="Step 5, flat-state entry condition.",
        why="Defines how large a dislocation has to be before it's worth "
        "trading (net of costs and estimation noise).",
        default="Indicator-specific: 2.0 (zscore), 20.0 (rsi), 0.45 "
        "(percentile) -- applied only when left unset (None).",
        typical_range="Depends on indicator; see default above.",
        effect_increase="Fewer, larger, more extreme entries -- higher "
        "conviction per trade, fewer trades overall (lower turnover/costs).",
        effect_decrease="More frequent entries on smaller dislocations -- "
        "more trades, more exposure to noise being mistaken for a genuine "
        "opportunity.",
        tradeoffs="Trade frequency and turnover vs. conviction per trade.",
        interactions="Must exceed exit_threshold (validated); if "
        "stop_threshold is set, must be below it.",
    ),
    ParameterDoc(
        name="exit_threshold",
        what="Indicator magnitude that closes an open position as the spread reverts.",
        where="Step 5, in-position exit condition.",
        why="Decides how much of the reversion to actually capture before "
        "closing, versus how long to stay exposed hoping for more.",
        default="Indicator-specific: 0.5 (zscore), 10.0 (rsi), 0.10 "
        "(percentile) -- applied only when left unset (None).",
        typical_range="Depends on indicator; see default above.",
        effect_increase="Exits earlier, leaving more of a full reversion "
        "uncaptured but reducing time-in-trade and reversal risk.",
        effect_decrease="Holds longer for a more complete reversion, at the "
        "cost of more time exposed to the spread reversing direction again "
        "before exit.",
        tradeoffs="Captured reversion vs. time-in-trade risk.",
        interactions="Must be strictly below entry_threshold.",
    ),
    ParameterDoc(
        name="stop_threshold",
        what="Indicator magnitude that force-closes a position regardless "
        "of direction -- a circuit breaker for a spread that keeps widening "
        "instead of reverting.",
        where="Step 5, checked before every other branch.",
        why="Limits the indicator's own deviation tolerated before "
        "requesting an exit on a relationship that may have broken down "
        "rather than merely dislocated; it does not cap the realized "
        "monetary loss -- gaps, execution delay, and a moving mean/"
        "volatility/hedge-ratio or costs can still produce a larger loss "
        "than the indicator distance alone would suggest.",
        default="Indicator-specific: 4.0 (zscore), 45.0 (rsi), 0.49 "
        "(percentile) -- applied when this parameter is left out entirely. "
        "Pass an explicit stop_threshold=None to disable the stop "
        "altogether.",
        typical_range="Typically 1.5-2x entry_threshold; omit the "
        "parameter to use the indicator-specific default, or pass None to "
        "disable it.",
        effect_increase="More room for the spread to widen before an exit "
        "is requested -- fewer stop-outs on noise, but larger potential "
        "loss per trade.",
        effect_decrease="Tighter risk control, but more prone to being "
        "stopped out by a temporary overshoot that would otherwise have "
        "reverted.",
        tradeoffs="Downside protection vs. premature stop-outs.",
        interactions="Must exceed entry_threshold when set; an explicit "
        "None disables the stop entirely (positions then only exit via "
        "exit_threshold).",
    ),
    ParameterDoc(
        name="dynamic_hedge_ratio",
        what="Whether the hedge ratio is refit every period (True) or fit "
        "once at formation and held constant (False).",
        where="Step 1.",
        why="A relationship's slope can itself drift over time; this "
        "decides whether the strategy tracks that drift or assumes it "
        "away.",
        default="True",
        typical_range="Boolean.",
        effect_increase="N/A (boolean).",
        effect_decrease="N/A (boolean).",
        tradeoffs="True adapts to a genuinely drifting relationship but "
        "makes the spread noisier (a moving hedge ratio adds its own "
        "variance); False is a simpler, more stable spread definition but "
        "can go stale if the true relationship shifts materially after "
        "formation.",
        interactions="The stationarity gate keeps the same recheck cadence; "
        "this parameter changes whether its coefficients are refit or held static.",
    ),
    ParameterDoc(
        name="adf_pvalue_threshold",
        what="Maximum ADF p-value (from the periodic stationarity gate) at "
        "which a NEW entry is still allowed -- optional: pass None to "
        "disable the gate entirely (every date becomes tradable, subject "
        "only to the entry/exit/stop thresholds).",
        where="Step 4, compared against the gate's own p-value.",
        why="Refuses to open a fresh position on a relationship the data no "
        "longer supports as stationary, even if the indicator looks "
        "attractive.",
        default="0.10 (gate enabled by default).",
        typical_range="0.05-0.10, or None to disable.",
        effect_increase="Looser gate -- more candidate entries pass, "
        "including weaker statistical evidence of stationarity.",
        effect_decrease="Stricter gate -- fewer entries pass, but each one "
        "clears a higher statistical bar.",
        tradeoffs="Opportunity (more entries) vs. statistical rigor (fewer, "
        "better-supported entries). Never affects an already-open "
        "position's own exit/stop.",
        interactions="Interacts with formation_window (more observations can "
        "improve test precision but do not guarantee greater power) and "
        "indicator_window (sets how often the gate is re-evaluated).",
    ),
    ParameterDoc(
        name="price_type",
        what="Which price series ('adjusted_close' or 'close') "
        "generate_signals() reads to compute the hedge ratio, spread and "
        "z-score. Execution/costs always use the raw close regardless.",
        where="Every step above operates on whichever price series this selects.",
        why="A split or large dividend on either leg shows up as a price "
        "jump in raw close but not in adjusted close -- unadjusted, it "
        "would look exactly like a spread dislocation.",
        default="adjusted_close",
        typical_range="adjusted_close (recommended) or close.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="adjusted_close avoids false dislocations from corporate "
        "actions; close matches what was literally quoted at the time, "
        "useful mainly for auditing against raw market data.",
        interactions="A split on only one leg while using close would "
        "corrupt the hedge ratio and spread for a long stretch after the "
        "split -- adjusted_close is the safer default for exactly this "
        "reason.",
    ),
    ParameterDoc(
        name="stop_loss_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens BOTH legs together, based on the PAIR's combined "
        "economic P&L -- not either leg's own return in isolation.",
        where="Applied after the backtest allocator/constraints/"
        "rebalancing/execution, on the position actually held -- see "
        "`PairsTradingStrategy.position_groups()` (declares the two legs "
        "as one group) and `quantlab.backtesting.accounting.`"
        "`_detect_stop_loss_take_profit`.",
        why="A hedge leg's own gain can OFFSET the pair's real loss (or "
        "vice versa) -- a per-leg stop would misjudge risk entirely; the "
        "pair's own combined return, per unit of ITS OWN gross exposure "
        "at each date, is correct regardless of a static or dynamic "
        "hedge ratio, rebalancing, or partial fills.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="More room before the pair is forced flat -- "
        "fewer stop-outs on ordinary spread noise, larger potential "
        "realized loss.",
        effect_decrease="Tighter monetary risk control on the pair as a "
        "whole, more prone to being stopped out by a temporary spread move.",
        tradeoffs="Realized-loss protection vs. premature exits. Evaluated "
        "on GROSS (pre-cost) return -- QuantLab's execution cost model is "
        "portfolio-level only, so an exact net-of-cost trigger is not "
        "presently computable; this is a disclosed design convention, not "
        "a universal definition.",
        interactions="Independent of entry_threshold/exit_threshold/"
        "stop_threshold (the spread's own indicator-based stop). Once "
        "triggered, no immediate re-entry at a rebased price -- both legs "
        "stay flat until the pair's next real entry.",
    ),
    ParameterDoc(
        name="take_profit_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens both legs together on the favorable side, based "
        "on the pair's combined P&L.",
        where="Same mechanism as stop_loss_pct, opposite direction.",
        why="Realizes a gain directly once the pair's own combined return "
        "target is reached, instead of depending on the spread reverting "
        "all the way back through exit_threshold.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="Lets more of a favorable spread move run before "
        "locking it in.",
        effect_decrease="Locks in gains earlier, potentially forfeiting "
        "further convergence.",
        tradeoffs="Locking in gains early vs. capturing a larger reversion.",
        interactions="Independent of stop_loss_pct and the entry/exit/"
        "stop indicator-threshold family; see stop_loss_pct's own doc for "
        "the shared combined-P&L/gross-return/re-entry conventions.",
    ),
]


@dataclass(frozen=True)
class PairsTradingDiagnostics:
    """Pair relationship diagnostics, plus trading-threshold breach counts.

    ``diagnostics`` is the existing correlation/hedge-ratio/ADF/
    cointegration/half-life view (unchanged). The trading-threshold fields
    add the "which BARS satisfy the entry condition under the ADF filter"
    view from the interactive lab's own Trading thresholds section,
    computed on the SAME centered indicator the real strategy trades
    (`indicator`, resolved exactly as `PairsTradingStrategy` itself
    resolves it). ``viable_bars`` counts every bar the threshold+gate
    condition holds, NOT distinct trade-entry events -- the live
    strategy's own state machine (`_walk_pairs_positions_with_reasons`)
    only opens a position on the FIRST such bar after being flat, so this
    count is generally larger than the real number of entries a backtest
    would make.
    """

    diagnostics: PairDiagnostics
    indicator: str
    entry_threshold: float
    exit_threshold: float
    stop_threshold: float | None
    adf_pvalue_threshold: float | None
    entry_breaches: int
    viable_bars: int
    stop_breaches: int


def _compute_diagnostics(
    data: pd.DataFrame, cfg: ExperimentConfig
) -> PairsTradingDiagnostics:
    from quantlab.data.base import price_matrix
    from quantlab.features.pairs_diagnostics import compute_pair_diagnostics
    from quantlab.strategies.pairs_trading import PairsTradingStrategy

    params = cfg.strategy_parameters
    # `strategy.parameters.price_type` is rejected at config validation
    # (see `StrategyConfig._reject_price_type_in_parameters`) -- the
    # strategy's own price series is always `strategy.signal_price_type`.
    # Diagnostics computed on the wrong price series would show a
    # different hedge ratio/spread than the one actually traded.
    price_type = cfg.strategy.signal_price_type
    prices = price_matrix(data, adjusted=price_type != "close")
    # Built from the SAME constructor call the real backtest made -- see
    # MeanReversionStrategy's identical rationale for stop_threshold's
    # None-vs-unset semantics, shared by this strategy.
    strategy = PairsTradingStrategy(**params)
    diagnostics = compute_pair_diagnostics(
        prices,
        strategy.symbol_a,
        strategy.symbol_b,
        formation_window=strategy.formation_window,
        indicator_window=strategy.indicator_window,
        dynamic_hedge_ratio=strategy.dynamic_hedge_ratio,
        indicator=strategy.indicator,
    )
    indicator = diagnostics.spread_indicator
    entry = strategy.entry_threshold
    stop = strategy.stop_threshold
    adf_threshold = strategy.adf_pvalue_threshold
    crosses_entry = (indicator > entry) | (indicator < -entry)
    if adf_threshold is not None:
        gate_open = (
            diagnostics.rolling_adf_pvalue.reindex(indicator.index) <= adf_threshold
        ).fillna(False)
    else:
        gate_open = pd.Series(True, index=indicator.index)
    viable = crosses_entry & gate_open
    stop_breaches = (
        int(((indicator > stop) | (indicator < -stop)).sum()) if stop is not None else 0
    )
    return PairsTradingDiagnostics(
        diagnostics=diagnostics,
        indicator=strategy.indicator,
        entry_threshold=entry,
        exit_threshold=strategy.exit_threshold,
        stop_threshold=stop,
        adf_pvalue_threshold=adf_threshold,
        entry_breaches=int(crosses_entry.sum()),
        viable_bars=int(viable.sum()),
        stop_breaches=stop_breaches,
    )


def _render_diagnostics(st: Any, result: PairsTradingDiagnostics) -> None:
    from quantlab.dashboard.components import render_pair_diagnostics

    render_pair_diagnostics(
        st,
        result.diagnostics,
        entry_threshold=result.entry_threshold,
        exit_threshold=result.exit_threshold,
        stop_threshold=result.stop_threshold,
        adf_pvalue_threshold=result.adf_pvalue_threshold,
    )


def _report_section(result: PairsTradingDiagnostics) -> DiagnosticsSection:
    from quantlab.reporting.charts import fig_to_base64, pair_spread_chart
    from quantlab.reporting.sections import DiagnosticsSection
    from quantlab.reporting.tables import pair_diagnostics_summary_table

    table = pair_diagnostics_summary_table(result.diagnostics)
    threshold_rows = pd.DataFrame(
        [
            ("Indicator", result.indicator),
            ("Entry threshold", result.entry_threshold),
            ("Exit threshold", result.exit_threshold),
            (
                "Stop threshold",
                result.stop_threshold
                if result.stop_threshold is not None
                else "disabled",
            ),
            (
                "ADF p-value threshold",
                result.adf_pvalue_threshold
                if result.adf_pvalue_threshold is not None
                else "disabled",
            ),
            ("Entry threshold breaches (bar count)", result.entry_breaches),
            ("Viable bars (threshold + ADF gate)", result.viable_bars),
            ("Stop threshold breaches", result.stop_breaches),
        ],
        columns=["Metric", "Value"],
    )
    table = pd.concat([table, threshold_rows], ignore_index=True)
    return DiagnosticsSection(
        table=table,
        chart_data_uri=fig_to_base64(pair_spread_chart(result.diagnostics)),
        note=(
            "Pair relationship diagnostics (correlation, hedge ratio, "
            "spread stationarity, cointegration) plus trading-threshold "
            "breach counts on the configured indicator's centered series, "
            "including how many crossings were also viable under the ADF "
            "stationarity gate."
        ),
    )


def _lab(st: Any) -> None:
    from quantlab.dashboard.explorer.labs.pairs_trading import render

    render(st)


register_profile(
    StrategyProfile(
        strategy_name="pairs_trading",
        display_name="Pairs Trading",
        category="Relative value",
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
            key="pair_diagnostics",
            compute=_compute_diagnostics,
            render=_render_diagnostics,
            report_section=_report_section,
        ),
    )
)
