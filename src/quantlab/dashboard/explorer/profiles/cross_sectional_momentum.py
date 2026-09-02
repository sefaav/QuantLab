"""Strategy Explorer profile for ``cross_sectional_momentum``."""

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
Cross-sectional momentum ranks a universe of assets by their recent
performance and emits long signals for the strongest performers
(optionally short signals for the weakest). Unlike time-series momentum --
which asks "is THIS asset's own trend up or down" -- cross-sectional
momentum only ever asks a relative
question: "is this asset outperforming the OTHERS in the universe right
now". An asset can have a positive score in a falling universe, or a
negative score in a rising one.

The score horizon is set by `lookback_period`; the configured rebalance
frequency controls when rankings can change executed targets. Data needed:
a reasonably broad universe of comparable assets (a sector, an asset
class, a country set) with a shared history at least `lookback_period`
long.
"""

_ECONOMIC_INTUITION = """
Relative winner-minus-loser momentum has been documented historically over
medium-term horizons (roughly 3-12 months) in several markets. Proposed drivers
include investor underreaction to new information (prices adjust slowly,
not instantly, to genuinely good/bad news) and herding/trend-following
behaviour among market participants. This is a *relative*, not an
absolute, bet: the strategy buys the best performers in the universe,
whatever the universe as a whole is doing.
"""

_MATH = """
`generate_signals()`'s pipeline, in order:

1. **Score** -- `score = momentum(prices, lookback_period, skip_period)` =
   `P_{t-skip} / P_{t-lookback} - 1` for every symbol, at every date.
2. **Selection** -- `select_top_bottom(score, top_fraction, bottom_fraction
   if long_short else 0.0)` picks the top `top_fraction` of the universe
   (by score, that date) as `+1`, and -- only when `long_short=True` -- the
   bottom `bottom_fraction` as `-1`. Every other asset is `0`. Selections
   are disjoint: an asset is never both top and bottom. The ranking is
   computed on every bar, but only values sampled by the portfolio's
   rebalance schedule can alter executed targets.
3. **`skip_period`** excludes the most recent periods from the lookback
   window -- the classic "12-1" convention (12-month lookback, skip the
   most recent month) exists because very recent short-term returns have
   sometimes *reversed* rather than continued, which would otherwise partly cancel
   out the momentum effect being captured.

4. **`signal_scaling`** -- `binary` (default) emits the discrete
   `{-1, 0, +1}` selection unchanged; `continuous` additionally scales
   each selected asset's signal by its RANK within its own selected leg
   (see the `signal_scaling` parameter below), still guaranteed monotone
   in score and never zero for a selected asset. Either way, how much
   capital each selected asset actually gets still also depends on the
   portfolio allocator's own job downstream -- only the
   `signal_proportional` allocator actually consumes a `continuous`
   signal's magnitude; `equal_weight` discards it via `np.sign`.
"""

_ASSUMPTIONS = """
**Economic**: the strategy assumes recent relative winners are more likely
to outperform over its horizon in this universe; that may be absent or
sample-specific.
**Statistical**: momentum scores computed the same way across the whole
universe are comparable (broadly similar volatility/liquidity regimes --
comparing a mega-cap ETF's momentum score directly against a thinly-traded
micro-cap's is less meaningful). **Implementation**: `top_fraction` (and
`bottom_fraction`) leave enough assets in the selection to diversify
idiosyncratic risk, given the configured universe size.
"""

_DIAGNOSTICS = """
The lab below shows the momentum score itself for every universe member,
a full ranking + selection snapshot on any chosen date, the actual
cross-sectional test this strategy depends on (does a higher-RANKED
asset go on to earn a higher subsequent return than a lower-ranked one,
across the universe, at each date -- via a Spearman rank correlation and
the realized top-minus-bottom spread return), a single-asset time-series
diagnostic shown only for comparison, and a side-by-side comparison of
the score at several different lookback windows.
"""

_INTERPRETATION = """
A positive, reasonably stable rank correlation (and a positive top-minus-
bottom spread) across most dates is descriptive support for the strategy's
core premise on this data; a value near zero or negative weighs against it,
regardless of how a particular backtest's aggregate performance looks. This
is descriptive sample evidence, not a hypothesis test (overlapping holding periods are
not independent observations). The single-asset time-series scatter
answering a DIFFERENT question (does this one asset's own past predict
its own future) can look positive even when the cross-sectional ranking
signal is weak, or vice versa -- they are not substitutes for each other.
In long-only mode the displayed bottom group is a diagnostic comparison,
not a short book the strategy actually trades.
The ranking snapshot is useful for sanity-checking `top_fraction`/
`bottom_fraction` against the actual universe size -- e.g.
`top_fraction=0.25` on a 4-symbol universe selects exactly one asset,
which is a very different portfolio than the same fraction on 50 symbols.
"""

_LIMITATIONS = """
**Momentum crashes**: momentum has historically suffered sharp, sudden
reversals -- most notoriously around the 2009 market bottom -- when
previously beaten-down assets rebound violently, hurting exactly the
long-winners/short-losers positioning momentum takes. **Turnover**:
changes in the top/bottom selection can generate meaningful turnover and
transaction costs when they alter weights at rebalance dates, especially
with a short `skip_period` or a volatile universe where rankings shuffle
often. **Crowding**: momentum
is one of the most widely traded factors; crowded positioning can amplify
the crash risk above. **Small universes**: with few symbols,
`top_fraction`/`bottom_fraction` select very few assets, concentrating
idiosyncratic risk that a "diversified factor" framing usually assumes
away.
"""

_REFERENCES = (
    'Jegadeesh & Titman (1993), ["Returns to Buying Winners and Selling '
    'Losers: Implications for Stock Market Efficiency"]('
    "https://doi.org/10.1111/j.1540-6261.1993.tb04702.x), *Journal of "
    "Finance* 48(1), is the foundational academic study of cross-sectional/"
    "relative momentum -- it examines 3-12 month formation/holding periods "
    '(and a 1-week-skip variant), not a literal "12-1" specification; the '
    "specific 12-month-lookback/skip-one-month convention used by default "
    "here is a common later variant in the literature/practice, not a direct "
    "reproduction of the paper's methodology or evidence for this particular "
    "universe."
)

_PARAMETERS = [
    ParameterDoc(
        name="lookback_period",
        what="Total look-back window (periods) the momentum score is measured over.",
        where="Step 1.",
        why="Sets the horizon over which 'recent performance' is defined.",
        default="252",
        typical_range="126-252 periods (roughly 6-12 months of daily data).",
        effect_increase="Captures a longer-horizon trend, less sensitive to "
        "short-term noise, but slower to pick up a genuinely new trend.",
        effect_decrease="More responsive to a recent shift in relative "
        "performance, but noisier and more exposed to short-term reversal.",
        tradeoffs="Horizon length vs. responsiveness -- 3-12 month "
        "formation/holding horizons are common in the cited foundational study.",
        interactions="Must exceed skip_period; interacts with the "
        "portfolio's rebalance frequency (a lookback much shorter than the "
        "rebalance interval largely resets the ranking every rebalance).",
    ),
    ParameterDoc(
        name="skip_period",
        what="Most recent periods excluded from the lookback window.",
        where="Step 1, subtracted from the window used for the score.",
        why="Very recent short-term returns have sometimes shown reversal; "
        "skipping them reduces their influence on the momentum score.",
        default="21",
        typical_range="0-21 periods (0 to about 1 month of daily data).",
        effect_increase="Excludes more recent history from the score -- "
        "cleaner separation from short-term reversal, but the score reacts "
        "more slowly to a genuine, very recent shift.",
        effect_decrease="Includes more recent history -- more responsive, "
        "but more exposed to the short-term-reversal effect canceling out "
        "part of the momentum signal.",
        tradeoffs="Purity of the momentum signal vs. responsiveness.",
        interactions="Must be strictly less than lookback_period.",
    ),
    ParameterDoc(
        name="top_fraction",
        what="Fraction of the universe (by score, each date) selected as "
        "the long side. The count is rounded down, with at least one asset "
        "selected whenever the fraction is positive and data is available.",
        where="Step 2.",
        why="Controls concentration: how many of the best-ranked assets "
        "actually get a position.",
        default="0.25",
        typical_range="0.1-0.5.",
        effect_increase="More assets held -- more diversified, closer to "
        "the whole universe's own behaviour, weaker tilt toward the very "
        "best performers.",
        effect_decrease="Fewer assets held -- more concentrated, a purer "
        "bet on the top performers specifically, but more idiosyncratic "
        "risk per position.",
        tradeoffs="Diversification vs. concentration in the strongest signal.",
        interactions="With long_short=True, top_fraction + bottom_fraction "
        "must not exceed 1; with a small universe, a large fraction can "
        "select nearly everyone, diluting the selection to almost nothing.",
    ),
    ParameterDoc(
        name="bottom_fraction",
        what="Fraction of the universe selected as the short side -- only "
        "used when long_short=True, with the same floor/minimum-one count "
        "rule as top_fraction.",
        where="Step 2.",
        why="Symmetric counterpart to top_fraction for the short leg.",
        default="0.25",
        typical_range="0.1-0.5.",
        effect_increase="More short positions -- more diversified short "
        "book, weaker conviction per short.",
        effect_decrease="Fewer, higher-conviction short positions.",
        tradeoffs="Same as top_fraction, applied to the short side.",
        interactions="Ignored entirely when long_short=False; combined "
        "with top_fraction must not exceed 1.",
    ),
    ParameterDoc(
        name="long_short",
        what="Whether the bottom-ranked assets are actively shorted "
        "(True) or simply not held (False, long-only).",
        where="Step 2 -- gates whether bottom_fraction has any effect at all.",
        why="Many portfolios/mandates cannot or should not short; shorting "
        "also adds financing, borrow and short-side risk.",
        default="False",
        typical_range="Boolean.",
        effect_increase="N/A (boolean).",
        effect_decrease="N/A (boolean).",
        tradeoffs="True represents the classic long-winners/short-losers "
        "construction but adds short-specific costs/risks (borrow and "
        "unbounded theoretical loss on a runaway short); False is simpler "
        "and avoids those, but no longer captures the short leg.",
        interactions="bottom_fraction only matters when this is True.",
    ),
    ParameterDoc(
        name="signal_scaling",
        what="How the discrete top/bottom selection is expressed as a "
        "signal magnitude: 'binary' (every selected asset gets identical "
        "+1/-1 weight) or 'continuous' (each selected asset's weight is "
        "its RANK within its own selected leg, divided by that leg's own "
        "selected count -- e.g. the weakest of 4 selected longs gets "
        "0.25, the strongest gets 1.0; the short leg mirrors this on the "
        "most-negative-score side).",
        where="Final signal output, after the Step 2 selection above.",
        why="binary treats every selected name as an equally-strong bet; "
        "continuous instead lets the strongest-ranked name in each leg "
        "carry a larger weight than one that just barely qualified -- "
        "WHICH assets are selected is unchanged either way, only their "
        "relative size. Ranked within the leg rather than standardized "
        "against the whole cross-section's mean/dispersion, which is NOT "
        "guaranteed monotone in score when a leg straddles the "
        "cross-sectional mean.",
        default="binary",
        typical_range="One of 'binary', 'continuous'.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="binary is simplest and treats every selected name "
        "identically; continuous adds conviction-weighting within the "
        "selection at the cost of a slightly less interpretable weight.",
        interactions="Position SIZE within the selection is otherwise "
        "entirely the portfolio allocator's responsibility, not this "
        "parameter's -- continuous only reshapes the SIGNAL handed to it, "
        "and only actually changes sizing under an allocator that reads "
        "signal magnitude (e.g. 'signal_proportional'); config validation "
        "rejects pairing non-binary scaling with 'equal_weight', which "
        "would otherwise silently discard it back down to binary sizing.",
    ),
    ParameterDoc(
        name="price_type",
        what="Which price series ('adjusted_close' or 'close') "
        "generate_signals() reads. Execution/costs always use the raw "
        "close regardless.",
        where="Feeds the momentum score in step 1, for every universe member.",
        why="A split or large dividend on any one symbol shows up as a "
        "price jump in raw close but not in adjusted close -- unadjusted, "
        "it would distort that symbol's momentum score and its ranking "
        "against the rest of the universe.",
        default="adjusted_close",
        typical_range="adjusted_close (recommended) or close.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="adjusted_close keeps corporate actions from distorting "
        "relative rankings across the universe; close matches what was "
        "literally quoted, useful mainly for auditing.",
        interactions="Matters more here than for a single-asset strategy: "
        "one mis-adjusted symbol distorts not just its own score but its "
        "relative RANK against every other universe member.",
    ),
    ParameterDoc(
        name="stop_loss_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens ONE symbol's REAL executed position -- ranking "
        "membership itself has no persistent state, but this operates on "
        "the actual position held after the allocator/constraints/"
        "rebalancing/execution, which can span several rebalances.",
        where="Applied downstream of generate_signals() entirely -- see "
        "`quantlab.backtesting.accounting._detect_stop_loss_take_profit`. "
        "generate_signals() itself is unchanged by this parameter.",
        why="A symbol can stay selected across several rebalances while "
        "its own price moves sharply against the position -- this bounds "
        "the realized loss on that specific holding, independent of "
        "whether it is still ranked in the top/bottom fraction.",
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
        interactions="Applies independently per symbol (no position_"
        "groups declared). Once triggered, no immediate re-entry at a "
        "rebased price -- flat until that symbol's next real entry (a "
        "fresh flat-to-non-flat transition of the executed weight), even "
        "if it re-qualifies for the top/bottom fraction sooner.",
    ),
    ParameterDoc(
        name="take_profit_pct",
        what="Fractional (e.g. 0.10 = 10%) gross-return threshold that "
        "force-flattens one symbol's REAL executed position on the "
        "favorable side -- locks in a gain directly rather than waiting "
        "for it to drop out of the ranking.",
        where="Same mechanism as stop_loss_pct, opposite direction.",
        why="Realizes a gain directly once a target is reached, instead "
        "of depending on the symbol eventually falling out of the top/"
        "bottom fraction.",
        default="None (disabled) -- enabling it changes no existing "
        "behavior unless explicitly set.",
        typical_range="0.05-0.20, or None to disable.",
        effect_increase="Lets more of a favorable move run before locking it in.",
        effect_decrease="Locks in gains earlier, potentially forfeiting "
        "further outperformance.",
        tradeoffs="Locking in gains early vs. capturing more outperformance.",
        interactions="Independent of stop_loss_pct; see its own doc for "
        "the shared gross-return/re-entry conventions.",
    ),
]


def _lab(st: Any) -> None:
    from quantlab.dashboard.explorer.labs.cross_sectional_momentum import render

    render(st)


#: Fixed default forward-return horizon for this diagnostic -- matching
#: the interactive lab's own default. Deliberately INDEPENDENT of
#: ``skip_period`` (a strategy parameter meaning "how much of the recent
#: past to exclude from the score", not "how long to hold looking
#: forward"): ``skip_period=0`` is a perfectly valid strategy config, but
#: ``holding_period=0`` is rejected by ``cross_sectional_momentum_
#: persistence`` (must be >= 1), and a `skip_period` above 252 (also
#: valid -- only constrained to be < lookback_period) would put the
#: Results-tab slider's default value outside its own 1-252 range. Never
#: reuse ``skip_period`` here again.
_DEFAULT_DIAGNOSTIC_HOLDING_PERIOD = 21


@dataclass(frozen=True)
class CrossSectionalMomentumDiagnostics:
    """Cross-sectional rank-correlation/spread persistence over the sample.

    ``persistence``/``mean_rank_correlation``/``mean_top_minus_bottom`` are
    computed at ``holding_period`` (fixed to
    ``_DEFAULT_DIAGNOSTIC_HOLDING_PERIOD`` -- see that constant's own
    docstring for why it is independent of ``skip_period``) for the
    exported HTML report, which has no interactivity. ``prices`` and the
    other resolved parameters are carried alongside so the Results tab can
    recompute this SAME diagnostic at a user-chosen holding_period on
    demand -- a cheap, purely local recomputation, not a backtest re-run
    (see ``_render_diagnostics``).
    """

    holding_period: int
    lookback_period: int
    skip_period: int
    top_fraction: float
    effective_bottom_fraction: float
    long_short: bool
    prices: pd.DataFrame
    mean_rank_correlation: float
    mean_top_minus_bottom: float
    persistence: pd.DataFrame


def _persistence_table(
    prices: pd.DataFrame,
    lookback_period: int,
    skip_period: int,
    holding_period: int,
    *,
    top_fraction: float,
    bottom_fraction: float,
) -> tuple[pd.DataFrame, float, float]:
    from quantlab.features.momentum import cross_sectional_momentum_persistence

    persistence = cross_sectional_momentum_persistence(
        prices,
        lookback_period,
        skip_period,
        holding_period,
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    mean_corr = (
        float(persistence["rank_correlation"].mean())
        if not persistence.empty
        else float("nan")
    )
    mean_spread = (
        float(persistence["top_minus_bottom"].mean())
        if not persistence.empty
        else float("nan")
    )
    return persistence, mean_corr, mean_spread


def _compute_diagnostics(
    data: pd.DataFrame, cfg: ExperimentConfig
) -> CrossSectionalMomentumDiagnostics:
    from quantlab.data.base import price_matrix

    params = cfg.strategy_parameters
    lookback = int(params.get("lookback_period", 252))
    skip = int(params.get("skip_period", 21))
    top_fraction = float(params.get("top_fraction", 0.25))
    long_short = bool(params.get("long_short", False))
    bottom_fraction = float(params.get("bottom_fraction", 0.25))
    # When long_short=False there is no traded short book -- a comparison
    # bottom fraction (capped to what fits alongside top_fraction) is used
    # purely to compute the diagnostic, exactly like the interactive lab.
    comparison_bottom_fraction = min(top_fraction, max(0.0, 1.0 - top_fraction))
    effective_bottom = bottom_fraction if long_short else comparison_bottom_fraction

    price_type = cfg.strategy.signal_price_type
    prices = price_matrix(data, adjusted=price_type != "close")
    # Fixed, skip_period-INDEPENDENT default (see
    # _DEFAULT_DIAGNOSTIC_HOLDING_PERIOD's own docstring for why). The
    # Results tab lets the user override it independently (see
    # _render_diagnostics); the exported HTML report always uses this
    # fixed value so the report stays stable and reproducible.
    holding_period = _DEFAULT_DIAGNOSTIC_HOLDING_PERIOD
    persistence, mean_corr, mean_spread = _persistence_table(
        prices,
        lookback,
        skip,
        holding_period,
        top_fraction=top_fraction,
        bottom_fraction=effective_bottom,
    )
    return CrossSectionalMomentumDiagnostics(
        holding_period=holding_period,
        lookback_period=lookback,
        skip_period=skip,
        top_fraction=top_fraction,
        effective_bottom_fraction=effective_bottom,
        long_short=long_short,
        prices=prices,
        mean_rank_correlation=mean_corr,
        mean_top_minus_bottom=mean_spread,
        persistence=persistence,
    )


def _render_diagnostics(st: Any, result: CrossSectionalMomentumDiagnostics) -> None:
    from quantlab.dashboard.explorer.shared_components import (
        render_price_chart,
        strong,
    )

    st.subheader("Cross-sectional momentum persistence")
    holding_period = st.slider(
        "Forward-return horizon (periods) for this diagnostic",
        1,
        252,
        result.holding_period,
        key="csmom_results_diag_holding_period",
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
        persistence = result.persistence
        mean_corr = result.mean_rank_correlation
        mean_spread = result.mean_top_minus_bottom
    else:
        persistence, mean_corr, mean_spread = _persistence_table(
            result.prices,
            result.lookback_period,
            result.skip_period,
            holding_period,
            top_fraction=result.top_fraction,
            bottom_fraction=result.effective_bottom_fraction,
        )
    st.caption(
        "The question this strategy actually trades: do higher-ranked "
        "assets go on to earn higher subsequent returns, RELATIVE TO EACH "
        f"OTHER, over a {holding_period}-period horizon? Mean "
        f"rank correlation: {strong(f'{mean_corr:.3f}')}. Mean "
        f"top-minus-bottom spread: {strong(f'{mean_spread:.3%}')}."
        + (
            ""
            if result.long_short
            else " (long_short is disabled -- the bottom group here is a "
            "research comparison only, not a short book this backtest "
            "actually held.)"
        ),
        unsafe_allow_html=True,
    )
    if persistence.empty:
        st.info("Not enough dates with at least 3 scored assets in this result.")
        return
    render_price_chart(
        st,
        {"Rank correlation": persistence["rank_correlation"]},
        title="Spearman rank correlation: momentum score vs. subsequent return",
        yaxis_title="Rank correlation",
    )
    render_price_chart(
        st,
        {"Top - bottom spread return": persistence["top_minus_bottom"]},
        title=f"Realized top-minus-bottom {holding_period}-period return",
        yaxis_title="Return",
    )


def _report_section(result: CrossSectionalMomentumDiagnostics) -> DiagnosticsSection:
    from quantlab.dashboard.explorer.shared_components import live_widget_value
    from quantlab.reporting.sections import DiagnosticsSection

    # Reflects the user's own live Results-tab slider choice (see
    # _render_diagnostics), not always result.holding_period -- falls back
    # to it when the dashboard isn't running at all (e.g. the CLI's own
    # report generation) or that slider was never rendered this session.
    holding_period = live_widget_value(
        "csmom_results_diag_holding_period", result.holding_period
    )
    if holding_period == result.holding_period:
        mean_corr = result.mean_rank_correlation
        mean_spread = result.mean_top_minus_bottom
    else:
        _persistence, mean_corr, mean_spread = _persistence_table(
            result.prices,
            result.lookback_period,
            result.skip_period,
            holding_period,
            top_fraction=result.top_fraction,
            bottom_fraction=result.effective_bottom_fraction,
        )
    table = pd.DataFrame(
        [
            ("Holding period (periods)", holding_period),
            ("Long/short", result.long_short),
            ("Mean rank correlation", mean_corr),
            ("Mean top-minus-bottom spread", mean_spread),
        ],
        columns=["Metric", "Value"],
    )
    return DiagnosticsSection(
        table=table,
        note=(
            "Cross-sectional momentum persistence: does a higher-ranked "
            "asset earn a higher subsequent return than a lower-ranked one, "
            "across the universe? Descriptive sample evidence, not a "
            "hypothesis test -- overlapping holding periods are not "
            "independent."
        ),
    )


register_profile(
    StrategyProfile(
        strategy_name="cross_sectional_momentum",
        display_name="Cross-Sectional Momentum",
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
            key="cross_sectional_momentum_diagnostics",
            compute=_compute_diagnostics,
            render=_render_diagnostics,
            report_section=_report_section,
        ),
    )
)
