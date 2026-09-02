"""Strategy Explorer profile for ``buy_and_hold``."""

from __future__ import annotations

from typing import Any

from quantlab.dashboard.explorer.profile import (
    ParameterDoc,
    StrategyProfile,
    register_profile,
)

_OVERVIEW = """
The simplest strategy in QuantLab, and the essential baseline every other
strategy should be compared against: emit a long eligibility signal for
every configured instrument while its price is available, with no timing
decision at all. It has no view on direction, no entry/exit logic, and
(with more than one asset) is not literally "hold forever at fixed
weights" the way the name might suggest -- see the lab below for why.

Typical horizon: the entire backtest period. Data needed: whatever the
configured instruments have.
"""

_ECONOMIC_INTUITION = """
Buy and hold exists as a strategy for two reasons: it can be a long-term
investment approach in its own right (broad, diversified market exposure
seeks to capture a long-run risk premium without the additional turnover
associated with active timing), and it serves as the essential passive
baseline against which every actively timed strategy in this project
should be evaluated.
An active strategy should demonstrate an objective benefit -- for example,
higher risk-adjusted performance or lower drawdown -- after accounting for
its extra transaction costs, turnover, execution risk and model risk.
"""

_MATH = """
`generate_signals()` is a single line: `signal = prices.notna().astype
(float)` -- exactly `1.0` wherever a valid price is available for a
symbol and `0.0` otherwise. There is no lookback window, threshold,
ranking rule, or strategy state to track.

Crucially, these signals are not portfolio weights. They only indicate
which assets are eligible to be held. The actual realized portfolio
weights are determined downstream by the configured allocator and the
`rebalance_frequency`.

With a single asset, an eligibility signal that remains at `1.0` asks the
allocator to hold that asset. It produces full exposure only in the
absence of downstream scaling or constraints; volatility targeting,
weight/exposure limits, turnover limits and execution timing can all
reduce or delay the final weight. With multiple
assets, however, an eligibility signal of 1.0 for every asset does not
imply a weight of 1.0 in every asset; the allocator determines how
capital is distributed among the eligible assets (see Assumptions / lab
below).
"""

_ASSUMPTIONS = """
**Economic**: the configured instrument(s) are assumed to be suitable for
long-term exposure with no active timing view -- more plausible for a broad,
long-term-appropriate holding, less appropriate for something
mean-reverting, cyclical, or otherwise unsuited to indefinite exposure.

**Portfolio construction**: with more than one instrument, "buy and hold" is
implicitly also a statement about the *rebalance schedule* -- the
strategy itself is silent on target weights, so the configured allocator and
`rebalance_frequency` entirely determine how closely realized weights stay
to any intended allocation.
"""

_DIAGNOSTICS = """
The lab below is mostly a price explorer -- there is little to diagnose
about a strategy with no timing parameters. Its main
diagnostic, with two or more symbols selected, illustrates how much a
real portfolio's weights drift away from an equal split purely
because of differences in each asset's return between rebalances -- the
concrete illustration of why "always invested" is not the same claim as
"static weights". This matches what a real QuantLab backtest itself now
does by default (`portfolio.model_weight_drift=True` -- see
docs/backtesting.md's Weight drift section); it stays a standalone,
theoretical illustration only in that it never runs the actual
accounting/cost/compliance-correction pipeline, so it can still diverge
in the details from a genuine backtest of this strategy.
"""

_INTERPRETATION = """
As a baseline: any actively timed strategy backtested over the same
instruments and period should be compared against this one's Sharpe,
CAGR and drawdown -- underperforming buy and hold net of costs is a
meaningful signal that the added complexity is not earning its keep on
this data. As a strategy in its own right with multiple assets: the
weight-drift chart shows why `rebalance_frequency` matters even here,
despite this strategy having no parameters of its own -- QuantLab's own
accounting now models this same intra-period drift by default (see
docs/backtesting.md's Weight drift section), so `rebalance_frequency`
genuinely shapes the realized backtest, not only a real portfolio it is
meant to approximate.
"""

_LIMITATIONS = """
**No strategy-level risk management**: the signal itself remains long
through a sustained decline -- it has no stop or de-risking rule. The
allocator, volatility target and portfolio constraints may still scale
the final exposure. **No timing skill claim**: by construction it cannot
outperform through market timing because there is none -- its reported
performance instead reflects the returns of the selected instruments
together with the configured allocation, constraints, rebalancing and
execution rules. **Weight drift is modeled by default**
(`portfolio.model_weight_drift=True`) -- each asset's own price move
drifts its executed weight between real trades, matching a real
portfolio held at fixed share counts (illustrated in the lab above);
`model_weight_drift=False` remains available as an explicit legacy/
reproducibility flag that reproduces the constant-weight step function
instead (see docs/backtesting.md's Weight drift section).
"""

_PARAMETERS = [
    ParameterDoc(
        name="price_type",
        what="Which price series ('adjusted_close' or 'close') "
        "generate_signals() reads to decide whether a symbol is eligible "
        "to be 'held' on a given date. Execution/costs always use the "
        "raw close regardless.",
        where="The sole input to the strategy's one-line signal.",
        why="Determines only WHETHER a price exists on a date (affecting "
        "the earliest date a delayed listing becomes eligible), not the "
        "position's magnitude -- both price types are non-missing on the "
        "same dates for a normal listing, so this rarely changes anything "
        "in practice.",
        default="adjusted_close",
        typical_range="adjusted_close (recommended) or close.",
        effect_increase="N/A -- a choice, not a magnitude.",
        effect_decrease="N/A -- a choice, not a magnitude.",
        tradeoffs="Essentially none for this strategy specifically; kept "
        "for consistency with every other strategy's own signal_price_"
        "type configuration.",
        interactions="None -- this strategy has no other parameters to interact with.",
    ),
]


def _lab(st: Any) -> None:
    from quantlab.dashboard.explorer.labs.buy_and_hold import render

    render(st)


register_profile(
    StrategyProfile(
        strategy_name="buy_and_hold",
        display_name="Buy & Hold",
        category="Baseline",
        overview_md=_OVERVIEW,
        economic_intuition_md=_ECONOMIC_INTUITION,
        mathematical_definition_md=_MATH,
        assumptions_md=_ASSUMPTIONS,
        diagnostics_md=_DIAGNOSTICS,
        interpretation_md=_INTERPRETATION,
        limitations_md=_LIMITATIONS,
        parameters=_PARAMETERS,
        lab=_lab,
    )
)
