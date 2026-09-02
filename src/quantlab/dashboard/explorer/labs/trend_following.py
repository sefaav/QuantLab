"""Interactive Trend Following lab.

Fast/slow moving-average crossover -> whipsaw diagnostic (how often the
signal flips) -> trend-strength diagnostic (Efficiency Ratio) ->
parameter comparison.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def render(st: Any) -> None:
    """Render the Trend Following interactive lab."""
    from quantlab.dashboard.explorer.shared_components import (
        ENTRY_LINE_COLOR,
        EXIT_LINE_COLOR,
        load_explorer_prices_cached,
        render_price_chart,
        render_stop_loss_take_profit_illustration,
        render_symbol_and_source_picker,
        strong,
    )
    from quantlab.dashboard.state import default_end_date
    from quantlab.data.base import price_matrix
    from quantlab.features.momentum import ma_crossover_signal, moving_average
    from quantlab.features.technical import efficiency_ratio

    st.markdown("#### Fast/slow moving-average crossover")
    picker_result = render_symbol_and_source_picker(
        st,
        key_prefix="explorer_trend",
        default_symbols=("SPY", "QQQ", "TLT", "GLD"),
    )
    if picker_result is None:
        st.info("Pick at least one symbol above to load price data.")
        return
    symbols, source, calendar, use_bundled_demo_data = picker_result
    symbol = st.selectbox("Symbol to analyze", symbols, key="explorer_trend_symbol")
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Start date", value=date(2019, 1, 1), key="explorer_trend_start"
    )
    end_date = col_end.date_input(
        "End date", value=default_end_date(), key="explorer_trend_end"
    )

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
        logger.exception("Trend Following lab: could not load data for %s", symbol)
        st.error(f"Could not load data for {symbol}: {exc}")
        return
    prices_frame = price_matrix(data)
    if symbol not in prices_frame.columns:
        st.warning(f"No data for {symbol} in this range.")
        return
    prices = prices_frame[symbol]

    col_fast, col_slow = st.columns(2)
    fast_window = col_fast.slider("fast_window", 2, 100, 20, key="explorer_trend_fast")
    slow_window = col_slow.slider(
        "slow_window",
        fast_window + 1,
        300,
        max(fast_window + 1, 100),
        key="explorer_trend_slow",
    )
    fast_ma = moving_average(prices, fast_window)
    slow_ma = moving_average(prices, slow_window)
    render_price_chart(
        st,
        {
            "Price": prices,
            f"Fast MA ({fast_window})": fast_ma,
            f"Slow MA ({slow_window})": slow_ma,
        },
        title=f"{symbol}: fast/slow moving-average crossover",
        # Price is left at Plotly's own default first-trace color; Fast/
        # Slow MA get explicit, visibly distinct colors (matching Results)
        # so neither is ever mistaken for the price line itself.
        colors={
            f"Fast MA ({fast_window})": ENTRY_LINE_COLOR,
            f"Slow MA ({slow_window})": EXIT_LINE_COLOR,
        },
    )
    signal = ma_crossover_signal(prices, fast_window, slow_window)
    long_only = st.checkbox(
        "long_only (a downtrend goes flat instead of short)",
        value=True,
        key="explorer_trend_long_only",
    )
    executable_signal = signal.clip(lower=0.0) if long_only else signal
    render_price_chart(
        st,
        {
            "Executable signal (post warm-up fill)": executable_signal.fillna(0.0),
        },
        title="Crossover signal -- what generate_signals() actually returns "
        f"({'long_only' if long_only else 'long/short'})",
        yaxis_title="Signal",
    )
    render_stop_loss_take_profit_illustration(
        st, executable_signal.fillna(0.0), prices, key_prefix="explorer_trend"
    )

    st.markdown("#### Whipsaw diagnostic")
    st.caption(
        "How often does the raw crossover direction change? Frequent changes "
        "indicate whipsaw pressure, but they create trades and costs only if "
        "they change a target sampled at a rebalance date. With long_only=True, "
        "this raw diagnostic can also count movements within the clipped-flat "
        "region, so it is an upper-bound indicator rather than executed turnover."
    )
    flips = signal.diff().fillna(0.0).ne(0.0)
    window = st.slider(
        "Count flips over the trailing N periods",
        20,
        504,
        126,
        key="explorer_trend_flip_window",
    )
    rolling_flips = flips.rolling(window, min_periods=1).sum()
    render_price_chart(
        st,
        {f"Raw crossover changes in trailing {window} periods": rolling_flips},
        title="Raw crossover-change frequency over time",
        yaxis_title="Flip count",
    )

    st.markdown("#### Trend-strength diagnostic (Efficiency Ratio)")
    st.caption(
        "Near 1: price moved efficiently in one direction (a clean trend, "
        "favourable for this strategy). Near 0: the same net move took a "
        "much choppier path (noise dominating -- unfavourable). A perfectly "
        "flat window has an undefined 0/0 ratio; QuantLab displays 0.5 for "
        "that special case as a neutral convention, not as trend evidence."
    )
    er_window = st.slider(
        "Efficiency Ratio window",
        5,
        200,
        min(slow_window, 200),
        key="explorer_trend_er_window",
    )
    er = efficiency_ratio(prices, er_window)
    render_price_chart(
        st,
        {"Efficiency Ratio": er},
        title="Kaufman's Efficiency Ratio",
        yaxis_title="ER",
    )
    st.caption(
        f"Median Efficiency Ratio over this sample: "
        f"{strong(f'{np.nanmedian(er.to_numpy()):.2f}')}.",
        unsafe_allow_html=True,
    )

    st.markdown("#### Parameter comparison")
    combos = st.multiselect(
        "(fast_window, slow_window) combinations to compare",
        options=["(10, 50)", "(20, 100)", "(50, 200)"],
        default=["(10, 50)", "(20, 100)", "(50, 200)"],
        key="explorer_trend_compare",
    )
    if combos:
        series = {}
        for combo in combos:
            fast_str, slow_str = combo.strip("()").split(",")
            fast, slow = int(fast_str), int(slow_str)
            series[combo] = ma_crossover_signal(prices, fast, slow)
        render_price_chart(
            st,
            series,
            title=f"{symbol}: crossover signal at different windows",
            yaxis_title="Signal",
        )
