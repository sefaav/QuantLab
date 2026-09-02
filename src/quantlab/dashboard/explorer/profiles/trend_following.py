"""Strategy Explorer profile for ``trend_following``."""

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
The simplest possible trend signal: two moving averages of the same
price, a fast one and a slow one. When the fast average is above the
slow one, the recent price action is running "hot" relative to the
longer-term level -- treated as an uptrend, and vice versa. This is a
classic technical trend-following construction, distinct from
time-series momentum's return-based score (though both aim to capture
the same underlying trend-persistence effect via a different lens).

Typical horizon: set by `slow_window` -- tens to hundreds of bars (often
weeks to months on daily data) for conventional 20/100 or 50/200-style
pairings. Data needed: one asset's own price history, at least
`slow_window` long before a signal exists.
"""

_ECONOMIC_INTUITION = """
Same underlying premise as time-series momentum -- some price trends may
persist for a while, potentially because of gradual information diffusion
and trend-following flows --
expressed through a moving-average crossover instead of a raw return
score. A crossover is a simpler, more interpretable trend detector: it
directly asks "is the recent average price above or below the
longer-term average", which is easy to reason about and has a long
history in technical analysis.
"""

_MATH = """
`generate_signals()`'s pipeline:

1. **Two moving averages** -- `fast = moving_average(prices,
   fast_window)`, `slow = moving_average(prices, slow_window)`, each a
   simple trailing mean.
2. **Crossover sign** -- `signal = sign(fast - slow)`: `+1` when the fast
   average sits above the slow one, `-1` when below, `0` on an exact tie,
   `NaN` during warm-up (before `slow_window` observations exist) --
   `_validate_signals()` then converts that warm-up `NaN` into an actual
   flat (`0.0`) position before this becomes the strategy's real output,
   exactly like every other built-in strategy's warm-up period.
3. **`long_only`** clips the result to `>= 0` when set, so a downtrend
   simply goes flat instead of short.

That's the entire strategy signal -- no smoothing beyond the two moving
averages themselves and no separate confirmation step. It is not yet a
portfolio weight: the allocator, constraints, rebalancing schedule and
execution model determine the weight that is ultimately traded.
"""

_ASSUMPTIONS = """
**Economic**: the asset is assumed to exhibit trend persistence that a
moving-average crossover can detect, rather than only noise around a stable level.
**Statistical**: the price series has enough directional
persistence (see the lab's Efficiency Ratio diagnostic) that a crossover
signal isn't dominated by whipsaws. **Implementation**: `fast_window`/
`slow_window` are set to a horizon where a real trend, once established,
lasts noticeably longer than the lag the crossover itself introduces
(a slow-moving average by construction reacts to a trend change well
after it has already started).
"""

_DIAGNOSTICS = """
The lab below shows the crossover itself, a whipsaw diagnostic (how many
times the raw crossover direction changes within a trailing window -- an
upper-bound source of potential turnover when `long_only=True`, and a source
of actual turnover only when sampled targets change at rebalance dates),
Kaufman's Efficiency Ratio (a 0-1 measure of how
"clean" vs. "choppy" the recent price path has been, independent of
direction), and a side-by-side comparison of a few conventional
fast/slow window pairings. A perfectly flat window makes the usual ratio
mathematically undefined (`0 / 0`); QuantLab displays `0.5` for that special
case, so interpret it as a neutral implementation convention rather than
evidence of a moderately efficient trend.
"""

_INTERPRETATION = """
A high, stable Efficiency Ratio alongside few raw crossover changes describes a
market this strategy is well-suited to (clean, sustained trends). A low
Efficiency Ratio alongside frequent raw crossover changes describes a choppy,
range-bound
market -- exactly where trend following can struggle. Signal changes only
create trades and transaction costs when they alter the executed target at
a rebalance after downstream allocation, constraints and execution. Compare
these two diagnostics across different periods on the same instrument to
see how much the strategy's own suitability changes over time, not just
across instruments.
"""

_LIMITATIONS = """
**Whipsaws**: the strategy's single biggest failure mode -- a choppy,
range-bound market repeatedly triggers crossovers in both directions
without ever capturing a sustained move. Crossovers that change executed
weights can generate turnover and transaction costs. **Lag**: a
moving-average crossover only confirms a trend change after it has partly
already happened (more so for a slower `slow_window`)
-- some of the early, most profitable part of a new trend is structurally
missed. **No magnitude information**: the signal only knows "above" or
"below", not "by how much" -- a fast average barely above the slow one
and a fast average far above it produce the identical `+1` signal.
"""

_REFERENCES = (
    "Perry J. Kaufman, [*Trading Systems and Methods*, 5th ed.](https://doi."
    "org/10.1002/9781119202561), is the specific source used here for "
    "Kaufman's Efficiency Ratio and broader trend-system context. Zakamulin "
    '& Giner (2023), ["Optimal trend-following with transaction costs"]('
    "https://doi.org/10.1016/j.irfa.2023.102928), studies the relationship "
    "between trend models, transaction costs and simple moving-average "
    "crossover rules. Neither source establishes that QuantLab's specific "
    "parameter choices or lookback windows will remain profitable."
)

_PARAMETERS = [
    ParameterDoc(
        name="fast_window",
        what="Trailing window (periods) for the fast moving average.",
        where="Step 1.",
        why="Sets how quickly the 'current' side of the crossover reacts "
        "to new prices.",
        default="20",
        typical_range="5-50 periods.",
        effect_increase="A smoother fast average, closer to the slow one "
        "-- fewer, later crossovers.",
        effect_decrease="A twitchier fast average -- more, earlier "
        "crossovers, more whipsaw risk in a choppy market.",
        tradeoffs="Responsiveness vs. whipsaw frequency.",
        interactions="Must be strictly less than slow_window; the gap "
        "between the two mainly sets the pair's relative smoothing/lag, "
        "not a price-move-size threshold -- both a small sustained drift "
        "and a single sharp move can flip the crossover, depending on how "
        "the two averages evolve.",
    ),
    ParameterDoc(
        name="slow_window",
        what="Trailing window (periods) for the slow moving average.",
        where="Step 1.",
        why="Defines the longer-term reference level the fast average is "
        "compared against.",
        default="100",
        typical_range="50-200 periods.",
        effect_increase="A more stable long-term reference, but a slower, "
        "later-confirming signal -- more of an established trend is "
        "missed before the crossover fires.",
        effect_decrease="A faster-reacting reference, closer to "
        "fast_window -- less lag, but a noisier, whipsaw-prone signal.",
        tradeoffs="Confirmation lag vs. responsiveness.",
        interactions="Must be strictly greater than fast_window.",
    ),
    ParameterDoc(
        name="long_only",
        what="Whether a downtrend (fast below slow) emits a flat signal "
        "(True) or a short signal (False).",
        where="Final clip to >= 0.",
        why="Many portfolios/mandates cannot or should not short.",
        default="True",
        typical_range="Boolean.",
        effect_increase="N/A (boolean).",
        effect_decrease="N/A (boolean).",
        tradeoffs="True avoids short-specific costs/constraints but "
        "forfeits potential gains from downtrends; False represents both "
        "directions but introduces short-side risk and additional ways for "
        "a noisy crossover to create an adverse target.",
        interactions="The allocator, portfolio constraints and rebalance "
        "schedule decide whether that short signal becomes an executed short.",
    ),
    ParameterDoc(
        name="price_type",
        what="Which price series ('adjusted_close' or 'close') "
        "generate_signals() reads. Execution/costs always use the raw "
        "close regardless.",
        where="Feeds both moving averages in step 1.",
        why="A split or large dividend shows up as a price jump in raw "
        "close but not in adjusted close -- unadjusted, it would look "
        "exactly like a real crossover.",
        default="adjusted_close",
        typical_range="adjusted_close (recommended) or close.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="adjusted_close avoids false crossovers from corporate "
        "actions; close matches what was literally quoted.",
        interactions="A split near the current date would otherwise "
        "trigger a spurious crossover in both moving averages at once.",
    ),
    ParameterDoc(
        name="stop_loss_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens this symbol's REAL executed position -- unlike "
        "the crossover itself, which is memoryless (no notion of 'since "
        "entry'), this operates on the actual position held after the "
        "allocator/constraints/rebalancing/execution.",
        where="Applied downstream of generate_signals() entirely -- see "
        "`quantlab.backtesting.accounting._detect_stop_loss_take_profit`. "
        "generate_signals() itself is unchanged by this parameter.",
        why="A crossover only confirms a trend change after it has partly "
        "already happened (more so for a slower slow_window) -- this "
        "bounds the realized loss directly while waiting for the slower "
        "crossover to catch up.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="More room before a forced exit -- fewer stop-outs "
        "on ordinary whipsaw, larger potential realized loss per trade.",
        effect_decrease="Tighter monetary risk control, more prone to "
        "being stopped out by a temporary adverse move before the "
        "crossover itself reverses.",
        tradeoffs="Realized-loss protection vs. premature exits. Evaluated "
        "on GROSS (pre-cost) return -- QuantLab's execution cost model is "
        "portfolio-level only, so an exact net-of-cost trigger is not "
        "presently computable; this is a disclosed design convention, not "
        "a universal definition.",
        interactions="Once triggered, no immediate re-entry at a rebased "
        "price -- flat until the position's next real entry (a fresh "
        "flat-to-non-flat transition of the executed weight), even if the "
        "raw crossover itself has not flipped back.",
    ),
    ParameterDoc(
        name="take_profit_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens this symbol's REAL executed position on the "
        "favorable side -- locks in a gain directly rather than waiting "
        "for the crossover to reverse.",
        where="Same mechanism as stop_loss_pct, opposite direction.",
        why="Realizes a gain directly once a target is reached, instead "
        "of depending on the trend persisting (and then reversing) before "
        "the crossover itself signals an exit.",
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
    from quantlab.dashboard.explorer.labs.trend_following import render

    render(st)


#: Fixed windows for the two diagnostics below, matching the interactive
#: lab's own defaults -- see time_series_momentum's identically-motivated
#: constants for why these are fixed rather than strategy parameters.
_WHIPSAW_WINDOW = 126
_EFFICIENCY_RATIO_WINDOW = 200
#: The Results-tab Efficiency Ratio slider's own declared bounds -- must
#: match the literal min/max passed to st.slider() below exactly.
_ER_SLIDER_MIN = 5
_ER_SLIDER_MAX = _EFFICIENCY_RATIO_WINDOW


def _default_er_window(slow_window: int) -> int:
    """Clamp the ER-window default into the slider's own declared range.

    `slow_window` can legitimately be as low as 2 (`fast_window=1,
    slow_window=2` is a valid strategy config -- only `fast_window <
    slow_window` and both `>= 1` are enforced), which would otherwise put
    `min(slow_window, _EFFICIENCY_RATIO_WINDOW)` below the slider's
    declared minimum of 5.
    """
    return min(_ER_SLIDER_MAX, max(_ER_SLIDER_MIN, slow_window))


@dataclass(frozen=True)
class TrendFollowingDiagnostics:
    """Crossover/whipsaw/trend-strength diagnostics, one row per symbol.

    ``summary`` is computed at the default whipsaw/Efficiency Ratio
    windows (``_WHIPSAW_WINDOW``, ``min(slow_window,
    _EFFICIENCY_RATIO_WINDOW)``). ``prices``/``signal``/``slow_window``
    are carried alongside so the Results tab -- and the exported HTML
    report, which reflects the same live widget choice (see
    ``_report_section``) -- can recompute both diagnostics at a
    user-chosen window on demand, a cheap, purely local recomputation,
    not a backtest re-run (see ``_render_diagnostics``).
    """

    summary: pd.DataFrame
    prices: dict[str, pd.Series]
    fast_ma: dict[str, pd.Series]
    slow_ma: dict[str, pd.Series]
    signal: dict[str, pd.Series]
    slow_window: int


def _whipsaw_and_efficiency_ratio_table(
    prices: dict[str, pd.Series],
    signal: dict[str, pd.Series],
    whipsaw_window: int,
    er_window: int,
) -> pd.DataFrame:
    import numpy as np

    from quantlab.features.technical import efficiency_ratio

    rows = []
    for symbol, series in prices.items():
        flips = signal[symbol].diff().fillna(0.0).ne(0.0)
        rolling_flips = flips.rolling(whipsaw_window, min_periods=1).sum()
        er = efficiency_ratio(series, er_window)
        rows.append(
            {
                "Symbol": symbol,
                f"Whipsaw (flips / {whipsaw_window}p, latest)": float(
                    rolling_flips.iloc[-1]
                )
                if len(rolling_flips)
                else float("nan"),
                "Median Efficiency Ratio": float(np.nanmedian(er.to_numpy()))
                if len(er)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("Symbol")


def _compute_diagnostics(
    data: pd.DataFrame, cfg: ExperimentConfig
) -> TrendFollowingDiagnostics:
    from quantlab.data.base import price_matrix
    from quantlab.features.momentum import ma_crossover_signal, moving_average

    params = cfg.strategy_parameters
    fast_window = int(params.get("fast_window", 20))
    slow_window = int(params.get("slow_window", 100))
    er_window = _default_er_window(slow_window)
    price_type = cfg.strategy.signal_price_type
    price_frame = price_matrix(data, adjusted=price_type != "close")

    prices: dict[str, pd.Series] = {}
    fast_ma: dict[str, pd.Series] = {}
    slow_ma: dict[str, pd.Series] = {}
    signal: dict[str, pd.Series] = {}
    for symbol in price_frame.columns:
        series = price_frame[symbol]
        prices[symbol] = series
        fast_ma[symbol] = moving_average(series, fast_window)
        slow_ma[symbol] = moving_average(series, slow_window)
        signal[symbol] = ma_crossover_signal(series, fast_window, slow_window)
    summary = _whipsaw_and_efficiency_ratio_table(
        prices, signal, _WHIPSAW_WINDOW, er_window
    )
    return TrendFollowingDiagnostics(
        summary=summary,
        prices=prices,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        signal=signal,
        slow_window=slow_window,
    )


def _render_diagnostics(st: Any, result: TrendFollowingDiagnostics) -> None:
    from quantlab.dashboard.explorer.shared_components import (
        ENTRY_LINE_COLOR,
        EXIT_LINE_COLOR,
        render_price_chart,
    )

    st.subheader("Crossover / whipsaw / trend-strength diagnostics")
    st.caption(
        "Near 1 Efficiency Ratio: a clean, sustained trend (favourable for "
        "this strategy). Near 0: a choppy path (noise dominating). A high "
        "whipsaw count alongside a low Efficiency Ratio describes a market "
        "this strategy struggles with."
    )
    col_w, col_er = st.columns(2)
    whipsaw_window = col_w.slider(
        "Count flips over the trailing N periods",
        20,
        504,
        _WHIPSAW_WINDOW,
        key="tf_results_whipsaw_window",
    )
    default_er_window = _default_er_window(result.slow_window)
    er_window = col_er.slider(
        "Efficiency Ratio window",
        _ER_SLIDER_MIN,
        _ER_SLIDER_MAX,
        default_er_window,
        key="tf_results_er_window",
    )
    st.caption(
        "Diagnostic settings only -- changing these values does not rerun "
        "or alter the backtest; they only change how the already-computed "
        "signal is analyzed. These are NOT strategy parameters: they affect "
        "neither the signal nor the executed trades."
    )
    if whipsaw_window == _WHIPSAW_WINDOW and er_window == default_er_window:
        summary = result.summary
    else:
        summary = _whipsaw_and_efficiency_ratio_table(
            result.prices, result.signal, whipsaw_window, er_window
        )
    st.dataframe(summary, width="stretch")
    symbol = st.selectbox("Symbol", list(result.signal), key="tf_results_diag_symbol")
    render_price_chart(
        st,
        {
            "Price": result.prices[symbol],
            "Fast MA": result.fast_ma[symbol],
            "Slow MA": result.slow_ma[symbol],
        },
        title=f"{symbol}: fast/slow moving-average crossover",
        # Price is left at Plotly's own default first-trace color; Fast/
        # Slow MA get explicit, visibly distinct colors so neither is ever
        # mistaken for the price line itself.
        colors={"Fast MA": ENTRY_LINE_COLOR, "Slow MA": EXIT_LINE_COLOR},
    )
    render_price_chart(
        st,
        {"Crossover signal": result.signal[symbol]},
        title=f"{symbol}: raw crossover signal",
        yaxis_title="Signal",
    )
    flips = result.signal[symbol].diff().fillna(0.0).ne(0.0)
    rolling_flips = flips.rolling(whipsaw_window, min_periods=1).sum()
    render_price_chart(
        st,
        {f"Raw crossover changes in trailing {whipsaw_window} periods": rolling_flips},
        title=f"{symbol}: raw crossover-change frequency over time",
        yaxis_title="Flip count",
    )
    from quantlab.features.technical import efficiency_ratio

    er_series = efficiency_ratio(result.prices[symbol], er_window)
    render_price_chart(
        st,
        {"Efficiency Ratio": er_series},
        title=f"{symbol}: Kaufman's Efficiency Ratio",
        yaxis_title="ER",
    )


def _report_section(result: TrendFollowingDiagnostics) -> DiagnosticsSection:
    from quantlab.dashboard.explorer.shared_components import live_widget_value
    from quantlab.reporting.sections import DiagnosticsSection

    default_er_window = _default_er_window(result.slow_window)
    # Reflects the user's own live Results-tab slider choices (see
    # _render_diagnostics), not always the fixed defaults -- falls back to
    # them when the dashboard isn't running at all (e.g. the CLI's own
    # report generation) or those sliders were never rendered this session.
    whipsaw_window = live_widget_value("tf_results_whipsaw_window", _WHIPSAW_WINDOW)
    er_window = live_widget_value("tf_results_er_window", default_er_window)
    if whipsaw_window == _WHIPSAW_WINDOW and er_window == default_er_window:
        summary = result.summary
    else:
        summary = _whipsaw_and_efficiency_ratio_table(
            result.prices, result.signal, whipsaw_window, er_window
        )
    table = summary.reset_index()
    return DiagnosticsSection(
        table=table,
        note=(
            "Trend-following diagnostics per symbol: the most recent "
            f"whipsaw (raw crossover flip) count over a trailing "
            f"{whipsaw_window}-period window, and the median Kaufman "
            f"Efficiency Ratio over a trailing {er_window}-period window "
            "(near 1 = clean trend, near 0 = choppy noise)."
        ),
    )


register_profile(
    StrategyProfile(
        strategy_name="trend_following",
        display_name="Trend Following",
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
            key="trend_following_diagnostics",
            compute=_compute_diagnostics,
            render=_render_diagnostics,
            report_section=_report_section,
        ),
    )
)
