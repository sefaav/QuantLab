"""Strategy Explorer profile for ``time_series_momentum``."""

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
Time-series momentum emits a directional signal from each asset's own
trend: positive momentum produces a long signal; negative momentum
produces a flat or short signal. Unlike cross-sectional momentum, there is
no universe ranking -- each symbol is judged purely against its own past,
independent of how any other asset is doing. Also called absolute momentum or trend
following on returns.

Typical horizon: tens to hundreds of bars (often weeks to months on daily
data), set by `lookback_period`. Data needed: one asset's own price history,
long enough to build a stable score before trading starts.
"""

_ECONOMIC_INTUITION = """
Some asset-price trends have historically persisted for a while before
reversing. Proposed explanations include gradual information diffusion,
institutional flows that take time to execute, and behavioural
trend-following or herding. Trading in the direction of an established trend is
a bet that whatever is driving it (an improving/deteriorating
fundamental picture, sustained buying/selling pressure) has not yet fully
played out.
"""

_MATH = """
`generate_signals()`'s pipeline, in order:

1. **Score** -- `score = momentum(prices, lookback_period, skip_period)` =
   `P_{t-skip} / P_{t-lookback} - 1`, exactly as in cross-sectional
   momentum, but evaluated for one asset in isolation (no ranking against
   others).
2. **Scaling** (`signal_scaling`) turns that score into a signal in
   `[-1, 1]`:
   - `binary`: `sign(score)` -- emits `+1` or `-1` (or `0` if exactly
     zero), regardless of how strong the trend is.
   - `continuous`: `clip(score / rolling_std(score, lookback_period), -1,
     1)` -- scales the signal by how unusual the current score is relative
     to its own recent dispersion.
   - `volatility_adjusted`: `clip(score / realized_volatility(returns,
     volatility_window, periods_per_year), -1, 1)` -- scales down in
     high-volatility regimes and up in low-volatility ones, for a given
     raw score.
3. **`long_only`** clips the result to `>= 0` when set, removing short
   signals entirely.

These values are strategy signals, not final portfolio weights. The
allocator decides how signal magnitude is translated into target weights;
portfolio constraints, volatility targeting, rebalancing and execution can
then modify or delay those targets further.
"""

_ASSUMPTIONS = """
**Economic**: this specific asset's own recent trend is assumed to contain
information about its near-term future direction (trend persistence), not merely
backward-looking noise. **Statistical**: the chosen `signal_scaling` mode
matches how the underlying trend actually behaves -- e.g.
`volatility_adjusted` assumes recent realised volatility is a reasonable
guide to near-term risk, which can fail sharply around a volatility
regime change. **Implementation**: `lookback_period` should be long enough
to reduce noise but short enough to react to a change in direction; the
data cannot guarantee either property.
"""

_DIAGNOSTICS = """
The lab below plots the raw momentum score, then all three
`signal_scaling` modes side by side on that SAME score (the clearest way
to see what changing this one parameter actually does), the realised
volatility series that drives the `volatility_adjusted` mode specifically,
and a past-score-vs-future-return persistence scatter for the chosen
asset.
"""

_INTERPRETATION = """
Compare the three scaling-mode lines: `binary` is a step function, while
`continuous` and `volatility_adjusted` vary the signal magnitude with the
score (and, for the latter, with trailing volatility). The latter mode is a
heuristic signal-scaling rule, not a portfolio-level volatility target: it
divides a lookback return by an annualised trailing volatility estimate and
clips the result. If the modes diverge,
`signal_scaling` would have changed the input supplied to the allocator;
the final exposure also depends on all downstream portfolio and execution
settings. A flat or negative persistence correlation on the chosen
asset is evidence this strategy's core premise does not hold well for it,
regardless of what a specific historical backtest shows.
"""

_LIMITATIONS = """
**Whipsaws**: in a choppy, range-bound market with no sustained
direction, trend-following signals can repeatedly flip. Those flips create
costs only when they change executed weights at rebalance dates -- see the
Trend Following strategy's own `efficiency_ratio` diagnostic for a direct
measure of this failure mode, equally applicable here. **Sharp reversals**:
a fast trend reversal (a market correction or a shock) can hurt before the
signal has time to catch up, since it is inherently backward-looking over
`lookback_period`.
**Regime dependence for `volatility_adjusted`**: trailing-volatility
sizing can react slowly to sudden volatility spikes, leaving positions
temporarily sized for a calmer regime than the one in which risk is
actually realized.
"""

_REFERENCES = (
    'Moskowitz, Ooi & Pedersen (2012), ["Time Series Momentum"]('
    "https://doi.org/10.1016/j.jfineco.2011.11.003), *Journal of Financial "
    "Economics* 104(2), 228-250, documents time-series return predictability "
    "across 58 liquid futures contracts. It supports the broad "
    "time-series-momentum premise, not QuantLab's specific asset universe, "
    "parameter choices, or signal-scaling implementations."
)

_PARAMETERS = [
    ParameterDoc(
        name="lookback_period",
        what="Total look-back window (periods) the momentum score is measured over.",
        where="Step 1.",
        why="Sets the horizon over which 'the trend' is defined for this one asset.",
        default="252",
        typical_range="63-252 periods.",
        effect_increase="Captures a longer, smoother trend; slower to "
        "react to a genuine new trend.",
        effect_decrease="More responsive to a recent shift, but noisier.",
        tradeoffs="Horizon length vs. responsiveness.",
        interactions="Must exceed skip_period; also the window "
        "`continuous` scaling uses for the score's own rolling dispersion.",
    ),
    ParameterDoc(
        name="skip_period",
        what="Most recent periods excluded from the lookback window.",
        where="Step 1.",
        why="Very recent short-term returns have sometimes shown reversal "
        "rather than continuation.",
        default="21",
        typical_range="0-21 periods.",
        effect_increase="Cleaner separation from short-term reversal, "
        "slower reaction to a genuinely very recent shift.",
        effect_decrease="More responsive, more exposed to short-term reversal.",
        tradeoffs="Signal purity vs. responsiveness.",
        interactions="Must be strictly less than lookback_period.",
    ),
    ParameterDoc(
        name="signal_scaling",
        what="How the raw momentum score is mapped to a signal magnitude "
        "-- binary/continuous/volatility_adjusted.",
        where="Step 2 -- determines how the value supplied to the allocator "
        "varies with signal strength.",
        why="Different scaling modes make very different bets: full signal "
        "magnitude on any nonzero score (binary) vs. graded sizing by "
        "conviction (continuous) vs. graded sizing by conviction AND "
        "current risk (volatility_adjusted).",
        default="binary",
        typical_range="One of the three modes.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="binary is simplest and ignores trend magnitude; "
        "continuous scales with the score but needs a stable "
        "rolling dispersion estimate; volatility_adjusted additionally "
        "adjusts the signal using a risk estimate but depends on volatility_window/"
        "periods_per_year being well-chosen.",
        interactions="continuous and volatility_adjusted both need "
        "reliable rolling statistics -- noisy with too little history. "
        "The latter is separate from any portfolio-level "
        "target_volatility setting.",
    ),
    ParameterDoc(
        name="volatility_window",
        what="Trailing window (periods) for realised volatility, used "
        "only by the volatility_adjusted scaling mode.",
        where="Step 2, volatility_adjusted branch only.",
        why="Defines what 'current risk' means for sizing purposes.",
        default="63",
        typical_range="21-126 periods.",
        effect_increase="Smoother, slower-changing risk estimate.",
        effect_decrease="Faster-reacting risk estimate, noisier.",
        tradeoffs="Stability vs. responsiveness of the risk estimate.",
        interactions="Only matters when signal_scaling='volatility_"
        "adjusted'; interacts with periods_per_year (annualisation).",
    ),
    ParameterDoc(
        name="long_only",
        what="Whether short signals are structurally disabled.",
        where="Applied as a final clip to >= 0.",
        why="Many portfolios/mandates cannot or should not short.",
        default="True",
        typical_range="Boolean.",
        effect_increase="N/A (boolean).",
        effect_decrease="N/A (boolean).",
        tradeoffs="True avoids short-specific costs/constraints but "
        "forfeits potential gains from downtrends; False represents both "
        "directions but adds short-side and whipsaw risk.",
        interactions="When False the signal is symmetric around zero, but "
        "the allocator and constraints still determine final long/short weights.",
    ),
    ParameterDoc(
        name="periods_per_year",
        what="Annualisation factor used to convert per-period volatility "
        "into annualised volatility for the volatility_adjusted scaling "
        "mode.",
        where="Step 2, volatility_adjusted branch only (via realized_volatility).",
        why="Volatility is naturally a per-period quantity; annualising "
        "it makes the number comparable across different bar frequencies "
        "and to conventional risk figures.",
        default="252 (injected from the experiment's own data frequency; "
        "not usually set explicitly per-strategy).",
        typical_range="252 for daily equities, 365 for daily crypto (24/7 "
        "markets), or the bars-per-year implied by the configured "
        "frequency.",
        effect_increase="Scales the reported/used volatility level up for "
        "the same raw return dispersion -- shifts how aggressively "
        "volatility_adjusted sizes down in a given regime.",
        effect_decrease="Scales it down -- less aggressive de-risking for "
        "the same raw dispersion.",
        tradeoffs="Getting this wrong for the data's actual frequency "
        "silently mis-scales every volatility_adjusted signal; it should "
        "match the experiment's own annualisation, not be tuned as a free "
        "parameter.",
        interactions="Has no effect at all unless signal_scaling="
        "'volatility_adjusted'.",
    ),
    ParameterDoc(
        name="price_type",
        what="Which price series ('adjusted_close' or 'close') "
        "generate_signals() reads. Execution/costs always use the raw "
        "close regardless.",
        where="Feeds the momentum score in step 1.",
        why="A split or large dividend shows up as a price jump in raw "
        "close but not in adjusted close -- unadjusted, it would look "
        "exactly like a genuine trend move.",
        default="adjusted_close",
        typical_range="adjusted_close (recommended) or close.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="adjusted_close avoids false trend signals from "
        "corporate actions; close matches what was literally quoted.",
        interactions="A split near the current date would otherwise "
        "register as a large, entirely spurious trend signal.",
    ),
    ParameterDoc(
        name="stop_loss_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens this symbol's REAL executed position -- unlike "
        "this strategy's own momentum score, which is memoryless (no "
        "notion of 'since entry'), this operates on the actual position "
        "held after the allocator/constraints/rebalancing/execution.",
        where="Applied downstream of generate_signals() entirely -- see "
        "`quantlab.backtesting.accounting._detect_stop_loss_take_profit`. "
        "generate_signals() itself is unchanged by this parameter.",
        why="A trend can reverse sharply before the trailing momentum "
        "score itself catches up (it is backward-looking over "
        "lookback_period) -- this bounds the realized loss directly, "
        "independent of the score.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="More room before a forced exit -- fewer stop-outs "
        "on ordinary volatility, larger potential realized loss per trade.",
        effect_decrease="Tighter monetary risk control, more prone to "
        "being stopped out by a temporary adverse move before the trend "
        "score itself reverses.",
        tradeoffs="Realized-loss protection vs. premature exits. Evaluated "
        "on GROSS (pre-cost) return -- QuantLab's execution cost model is "
        "portfolio-level only, so an exact net-of-cost trigger is not "
        "presently computable; this is a disclosed design convention, not "
        "a universal definition.",
        interactions="Once triggered, no immediate re-entry at a rebased "
        "price -- flat until the position's next real entry (a fresh "
        "flat-to-non-flat transition of the executed weight).",
    ),
    ParameterDoc(
        name="take_profit_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens this symbol's REAL executed position on the "
        "favorable side -- locks in a gain directly rather than waiting "
        "for the momentum score to fade.",
        where="Same mechanism as stop_loss_pct, opposite direction.",
        why="Realizes a gain directly once a target is reached, instead "
        "of depending on the trend persisting (and then reversing) before "
        "the score itself signals an exit.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="Lets more of a favorable trend run before locking it in.",
        effect_decrease="Locks in gains earlier, potentially forfeiting "
        "further trend continuation.",
        tradeoffs="Locking in gains early vs. capturing a longer trend.",
        interactions="Independent of stop_loss_pct; see its own doc for "
        "the shared gross-return/re-entry conventions.",
    ),
]


def _lab(st: Any) -> None:
    from quantlab.dashboard.explorer.labs.time_series_momentum import render

    render(st)


#: Fixed default forward-return horizon for this diagnostic -- matching
#: the interactive lab's own default. Deliberately INDEPENDENT of
#: ``skip_period`` (a strategy parameter meaning "how much of the recent
#: past to exclude from the score", not "how long to hold looking
#: forward"): ``skip_period=0`` is a perfectly valid strategy config, but
#: ``holding_period=0`` is rejected by ``momentum_persistence`` (must be
#: >= 1), and a `skip_period` above 252 (also valid -- only constrained to
#: be < lookback_period) would put the Results-tab slider's default value
#: outside its own 1-252 range. Never reuse ``skip_period`` here again.
_DEFAULT_DIAGNOSTIC_HOLDING_PERIOD = 21


@dataclass(frozen=True)
class TimeSeriesMomentumDiagnostics:
    """Past-momentum-vs-future-return persistence, one row per symbol.

    ``summary``/``paired`` are computed at ``holding_period`` (fixed to
    ``_DEFAULT_DIAGNOSTIC_HOLDING_PERIOD`` -- a stable, documented default
    independent of any UI widget AND of ``skip_period``, see that
    constant's own docstring) for the exported HTML report, which has no
    interactivity. ``prices``/``lookback_period``/``skip_period`` are
    carried alongside so the Results tab can recompute this SAME
    diagnostic at a user-chosen holding_period on demand -- a cheap,
    purely local recomputation, not a backtest re-run (see
    ``_render_diagnostics``).
    """

    holding_period: int
    lookback_period: int
    skip_period: int
    prices: dict[str, pd.Series]
    summary: pd.DataFrame
    paired: dict[str, pd.DataFrame]


def _persistence_tables(
    prices: dict[str, pd.Series],
    lookback_period: int,
    skip_period: int,
    holding_period: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    from quantlab.features.momentum import momentum_persistence

    rows = []
    paired_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, series in prices.items():
        paired = momentum_persistence(
            series, lookback_period, skip_period, holding_period
        )
        correlation = (
            float(paired["past_momentum"].corr(paired["future_return"]))
            if not paired.empty
            else float("nan")
        )
        rows.append(
            {"Symbol": symbol, "Correlation": correlation, "Observations": len(paired)}
        )
        paired_by_symbol[symbol] = paired
    summary = pd.DataFrame(rows).set_index("Symbol")
    return summary, paired_by_symbol


def _compute_diagnostics(
    data: pd.DataFrame, cfg: ExperimentConfig
) -> TimeSeriesMomentumDiagnostics:
    from quantlab.data.base import price_matrix

    params = cfg.strategy_parameters
    lookback = int(params.get("lookback_period", 252))
    skip = int(params.get("skip_period", 21))
    price_type = cfg.strategy.signal_price_type
    price_frame = price_matrix(data, adjusted=price_type != "close")
    prices = {symbol: price_frame[symbol] for symbol in price_frame.columns}
    # Fixed, skip_period-INDEPENDENT default (see
    # _DEFAULT_DIAGNOSTIC_HOLDING_PERIOD's own docstring for why). The
    # Results tab lets the user override it independently (see
    # _render_diagnostics); the exported HTML report always uses this
    # fixed value so the report stays stable and reproducible.
    holding_period = _DEFAULT_DIAGNOSTIC_HOLDING_PERIOD
    summary, paired = _persistence_tables(prices, lookback, skip, holding_period)
    return TimeSeriesMomentumDiagnostics(
        holding_period=holding_period,
        lookback_period=lookback,
        skip_period=skip,
        prices=prices,
        summary=summary,
        paired=paired,
    )


def _render_diagnostics(st: Any, result: TimeSeriesMomentumDiagnostics) -> None:
    st.subheader("Momentum persistence diagnostics")
    holding_period = st.slider(
        "Forward-return horizon (periods) for this diagnostic",
        1,
        252,
        result.holding_period,
        key="tsmom_results_diag_holding_period",
        help=(
            "Diagnostic setting only -- changing this does not rerun or "
            "alter the backtest, only how many periods ahead this "
            "persistence check looks."
        ),
    )
    st.caption(
        "Forward-return horizon used only for this diagnostic; it does not "
        "change the strategy or its backtest. Longer horizons produce "
        "overlapping forward-return windows across consecutive dates, so "
        "the apparent number of observations overstates the independent "
        "information actually available -- treat this as descriptive "
        "sample evidence, not a hypothesis test."
    )
    if holding_period == result.holding_period:
        summary, paired_by_symbol = result.summary, result.paired
    else:
        summary, paired_by_symbol = _persistence_tables(
            result.prices, result.lookback_period, result.skip_period, holding_period
        )
    st.caption(
        f"Does past momentum score predict the subsequent "
        f"{holding_period}-period return, per symbol?"
    )
    st.dataframe(summary, width="stretch")
    symbol = st.selectbox(
        "Symbol", list(paired_by_symbol), key="tsmom_results_diag_symbol"
    )
    paired = paired_by_symbol[symbol]
    if paired.empty:
        st.info("Not enough history to pair momentum with a future return yet.")
        return
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=paired["past_momentum"],
            y=paired["future_return"],
            mode="markers",
            marker={"size": 5, "opacity": 0.5},
        )
    )
    fig.update_layout(
        title=f"{symbol}: past momentum vs. subsequent {holding_period}-period return",
        xaxis_title="Past momentum score",
        yaxis_title="Future return",
        height=380,
    )
    st.plotly_chart(fig, width="stretch")


def _report_section(result: TimeSeriesMomentumDiagnostics) -> DiagnosticsSection:
    from quantlab.dashboard.explorer.shared_components import live_widget_value
    from quantlab.reporting.sections import DiagnosticsSection

    # Reflects the user's own live Results-tab slider choice (see
    # _render_diagnostics), not always result.holding_period -- falls back
    # to it when the dashboard isn't running at all (e.g. the CLI's own
    # report generation) or that slider was never rendered this session.
    holding_period = live_widget_value(
        "tsmom_results_diag_holding_period", result.holding_period
    )
    if holding_period == result.holding_period:
        summary = result.summary
    else:
        summary, _ = _persistence_tables(
            result.prices, result.lookback_period, result.skip_period, holding_period
        )
    table = summary.reset_index()
    return DiagnosticsSection(
        table=table,
        note=(
            "Momentum persistence: correlation between each symbol's past "
            f"momentum score and its subsequent {holding_period}-"
            "period return. Descriptive sample evidence, not a hypothesis "
            "test -- overlapping holding periods are not independent."
        ),
    )


register_profile(
    StrategyProfile(
        strategy_name="time_series_momentum",
        display_name="Time-Series Momentum",
        category="Trend / momentum",
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
            key="time_series_momentum_diagnostics",
            compute=_compute_diagnostics,
            render=_render_diagnostics,
            report_section=_report_section,
        ),
    )
)
