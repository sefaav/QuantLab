"""Interactive Cross-Sectional Momentum lab.

Momentum formation -> asset ranking on a chosen date -> momentum
persistence -> parameter comparison. Uses the exact same
``quantlab.features.momentum``/``cross_sectional`` functions
``CrossSectionalMomentumStrategy`` itself calls.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def render(st: Any) -> None:
    """Render the Cross-Sectional Momentum interactive lab."""
    from quantlab.dashboard.explorer.shared_components import (
        load_explorer_prices_cached,
        render_price_chart,
        render_symbol_and_source_picker,
        strong,
    )
    from quantlab.dashboard.state import default_end_date
    from quantlab.data.base import price_matrix
    from quantlab.features.cross_sectional import select_top_bottom
    from quantlab.features.momentum import (
        cross_sectional_momentum_persistence,
        momentum,
        momentum_persistence,
    )

    st.markdown("#### Universe and momentum formation")
    picker_result = render_symbol_and_source_picker(
        st,
        key_prefix="explorer_csmom",
        default_symbols=("SPY", "QQQ", "TLT", "GLD"),
    )
    if picker_result is None:
        st.info("Pick at least two symbols above to continue.")
        return
    symbols, source, calendar, use_bundled_demo_data = picker_result
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Start date", value=date(2019, 1, 1), key="explorer_csmom_start"
    )
    end_date = col_end.date_input(
        "End date", value=default_end_date(), key="explorer_csmom_end"
    )
    if len(symbols) < 2:
        st.info("Pick at least two symbols above to continue.")
        return

    try:
        data = load_explorer_prices_cached(
            st,
            symbols,
            source=source,
            calendar=calendar,
            start_date=start_date,
            end_date=end_date,
            use_bundled_demo_data=use_bundled_demo_data,
        )
    except Exception as exc:
        logger.exception("Cross-Sectional Momentum lab: could not load data")
        st.error(f"Could not load data: {exc}")
        return
    prices = price_matrix(data)
    available = [symbol for symbol in symbols if symbol in prices.columns]
    if len(available) < 2:
        st.warning("Fewer than two selected symbols have data in this range.")
        return
    prices = prices[available]

    col_look, col_skip = st.columns(2)
    lookback = col_look.slider(
        "lookback_period", 21, 504, 252, key="explorer_csmom_lookback"
    )
    skip = col_skip.slider(
        "skip_period", 0, min(63, lookback - 1), 21, key="explorer_csmom_skip"
    )
    scores = momentum(prices, lookback, skip)
    render_price_chart(
        st,
        {symbol: scores[symbol] for symbol in available},
        title="Momentum score per symbol",
        yaxis_title="Momentum score",
    )

    st.markdown("#### Asset ranking on a chosen date")
    valid_dates = scores.dropna(how="all").index
    if len(valid_dates) == 0:
        st.info("No date has a defined momentum score yet -- widen the range.")
        return
    chosen_date = st.select_slider(
        "Date",
        options=list(valid_dates),
        value=valid_dates[-1],
        key="explorer_csmom_date",
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )
    col_top, col_short, col_bottom = st.columns(3)
    top_fraction = col_top.slider(
        "top_fraction", 0.1, 1.0, 0.25, 0.05, key="explorer_csmom_top"
    )
    long_short = col_short.checkbox(
        "long_short (also short the bottom fraction)",
        value=False,
        key="explorer_csmom_long_short",
    )
    bottom_fraction = (
        col_bottom.slider(
            "bottom_fraction",
            0.0,
            1.0 - top_fraction,
            min(0.25, 1.0 - top_fraction),
            0.05,
            key="explorer_csmom_bottom",
        )
        if long_short
        else 0.0
    )
    signal_scaling = st.selectbox(
        "signal_scaling",
        ["binary", "continuous"],
        key="explorer_csmom_signal_scaling",
        help="binary weights every selected asset identically. continuous "
        "weights each selected asset by its RANK within its own selected "
        "leg, divided by that leg's own selected count -- illustrated on "
        "the stop-loss/take-profit chart below via the strategy's real "
        "generate_signals().",
    )
    row = scores.loc[[chosen_date]]
    selection = select_top_bottom(row, top_fraction, bottom_fraction)
    ranking = pd.DataFrame(
        {
            "Momentum score": row.iloc[0],
            "Rank": row.iloc[0].rank(ascending=False),
            "Selected": selection.iloc[0].map({1.0: "top", 0.0: "no", -1.0: "bottom"}),
        }
    ).sort_values("Rank")
    st.dataframe(ranking, width="stretch")

    st.markdown("#### Cross-sectional momentum persistence")
    st.caption(
        "The question this strategy actually trades: on each date, do "
        "assets ranked higher on momentum go on to earn higher subsequent "
        "returns than assets ranked lower, RELATIVE TO EACH OTHER? A "
        "single asset's own serial correlation (see the time-series "
        "diagnostic below) is neither necessary nor sufficient for this. "
        "When long_short is disabled, the bottom group below is a research "
        "comparison only -- it is not a short book held by the strategy."
    )
    holding_period = st.slider(
        "holding_period (for the future return)",
        1,
        126,
        21,
        key="explorer_csmom_holding",
    )
    # When long_short is disabled, top_fraction alone can still legitimately
    # reach 1.0 (a valid strategy configuration -- select the whole universe
    # as "top"). Reusing it verbatim as the comparison bottom fraction would
    # then push top_fraction + bottom_fraction past 1 and make
    # select_top_bottom() raise. comparison_bottom_fraction is capped to
    # what actually fits, and exists purely for this diagnostic comparison
    # -- it never governs a real short book (see the caption above).
    comparison_bottom_fraction = min(top_fraction, max(0.0, 1.0 - top_fraction))
    effective_bottom = bottom_fraction if long_short else comparison_bottom_fraction
    persistence = cross_sectional_momentum_persistence(
        prices,
        lookback,
        skip,
        holding_period,
        top_fraction=top_fraction,
        bottom_fraction=effective_bottom,
    )
    if persistence.empty:
        st.info(
            "Not enough dates with at least 3 scored assets to compute "
            "cross-sectional persistence yet -- widen the date range, "
            "shorten lookback_period, or add more symbols to the universe."
        )
    else:
        render_price_chart(
            st,
            {"Rank correlation": persistence["rank_correlation"]},
            title="Spearman rank correlation: momentum score vs. subsequent "
            "return, across the universe",
            yaxis_title="Rank correlation",
        )
        render_price_chart(
            st,
            {"Top - bottom spread return": persistence["top_minus_bottom"]},
            title=f"Realized top({top_fraction:.0%}) minus bottom"
            f"({effective_bottom:.0%}) {holding_period}-period return",
            yaxis_title="Return",
        )
        mean_corr = persistence["rank_correlation"].mean()
        mean_spread = persistence["top_minus_bottom"].mean()
        st.caption(
            f"Mean rank correlation over this sample: {strong(f'{mean_corr:.3f}')}. "
            f"Mean top-minus-bottom spread: {strong(f'{mean_spread:.3%}')}. "
            "Descriptive sample evidence, not a hypothesis test -- overlapping "
            "holding periods across consecutive dates are not independent "
            "observations.",
            unsafe_allow_html=True,
        )

    st.markdown("#### Time-series diagnostic (for comparison)")
    st.caption(
        "A single asset's own past-momentum-vs-future-return relationship "
        "-- this is the TIME-SERIES momentum question (see the Time-Series "
        "Momentum strategy page), not what cross-sectional momentum "
        "actually trades. Shown here only as a point of comparison."
    )
    persistence_symbol = st.selectbox(
        "Symbol", available, key="explorer_csmom_persist_symbol"
    )
    paired = momentum_persistence(
        prices[persistence_symbol], lookback, skip, holding_period
    )
    if paired.empty:
        st.info("Not enough history to pair momentum with a future return yet.")
    else:
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
            title=f"{persistence_symbol}: past momentum vs. subsequent "
            f"{holding_period}-period return (time-series diagnostic)",
            xaxis_title="Past momentum score",
            yaxis_title="Future return",
            height=380,
        )
        st.plotly_chart(fig, width="stretch")
        correlation = paired["past_momentum"].corr(paired["future_return"])
        st.caption(
            f"Correlation between the two columns above: "
            f"{strong(f'{correlation:.3f}')}.",
            unsafe_allow_html=True,
        )

    st.markdown("#### Parameter comparison")
    compare_symbol = st.selectbox(
        "Symbol", available, key="explorer_csmom_compare_symbol"
    )
    lookbacks_to_compare = st.multiselect(
        "lookback_period values to compare",
        options=[63, 126, 189, 252, 378, 504],
        default=[126, 252, 504],
        key="explorer_csmom_compare_lookbacks",
    )
    if lookbacks_to_compare:
        series = {
            f"lookback={lb}": momentum(prices[compare_symbol], lb, skip)
            for lb in lookbacks_to_compare
        }
        render_price_chart(
            st,
            series,
            title=f"{compare_symbol}: momentum score at different lookbacks",
            yaxis_title="Momentum score",
        )

    try:
        from quantlab.dashboard.explorer.shared_components import (
            render_stop_loss_take_profit_illustration,
        )
        from quantlab.strategies.momentum import CrossSectionalMomentumStrategy

        cs_strategy = CrossSectionalMomentumStrategy(
            lookback_period=lookback,
            skip_period=skip,
            top_fraction=top_fraction,
            bottom_fraction=bottom_fraction,
            long_short=long_short,
            signal_scaling=signal_scaling,
        )
        cs_signals = cs_strategy.generate_signals(data)
    except Exception as exc:
        logger.exception(
            "Cross-Sectional Momentum lab: could not compute signals for "
            "the stop-loss/take-profit illustration"
        )
        st.error(f"Could not compute the selection signal for this illustration: {exc}")
    else:
        render_stop_loss_take_profit_illustration(
            st,
            cs_signals[compare_symbol],
            prices[compare_symbol],
            key_prefix="explorer_csmom",
        )
