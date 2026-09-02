"""Interactive Mean Reversion lab.

Price explorer -> indicator comparison (the three primary indicators
`MeanReversionStrategy` offers in the main UI, on the SAME data) -> the
real backtestable state machine (calling `MeanReversionStrategy` directly,
for whichever `indicator` is selected) -> stationarity diagnostics (ADF,
half-life, Hurst). Every indicator call is the exact function
`MeanReversionStrategy` itself uses -- nothing here is a second
implementation.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def render(st: Any) -> None:
    """Render the Mean Reversion interactive lab (see module docstring)."""
    from quantlab.dashboard.explorer.shared_components import (
        centered_indicator_threshold_overlay,
        load_explorer_prices_cached,
        render_price_chart,
        render_stationarity_card,
        render_stop_loss_take_profit_illustration,
        render_symbol_and_source_picker,
        strong,
    )
    from quantlab.dashboard.state import default_end_date
    from quantlab.data.base import price_matrix
    from quantlab.features.mean_reversion import (
        half_life,
        rolling_percentile_rank,
        rolling_zscore,
        rsi,
    )
    from quantlab.features.stationarity import adf_test, hurst_exponent

    st.markdown("#### Price explorer")
    picker_result = render_symbol_and_source_picker(
        st,
        key_prefix="explorer_mr",
        default_symbols=("SPY", "QQQ", "TLT", "GLD"),
    )
    if picker_result is None:
        st.info("Pick at least one symbol above to load price data.")
        return
    symbols, source, calendar, use_bundled_demo_data = picker_result
    symbol = st.selectbox("Symbol to analyze", symbols, key="explorer_mr_symbol")
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Start date", value=date(2019, 1, 1), key="explorer_mr_start"
    )
    end_date = col_end.date_input(
        "End date", value=default_end_date(), key="explorer_mr_end"
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
        logger.exception("Mean Reversion lab: could not load data for %s", symbol)
        st.error(f"Could not load data for {symbol}: {exc}")
        return
    prices_frame = price_matrix(data)
    if symbol not in prices_frame.columns:
        st.warning(f"No data for {symbol} in this range.")
        return
    prices = prices_frame[symbol]
    render_price_chart(st, {symbol: prices}, title=f"{symbol} price")

    st.markdown("#### Indicator comparison")
    st.caption(
        "Three different ways of asking 'how far from normal is this price "
        "right now' -- the three primary indicators `MeanReversionStrategy` "
        "offers via `indicator` -- compared on the same data. (Two further "
        "indicators, Bollinger Bands and distance-to-MA, are also "
        "implemented and usable programmatically -- see the Mathematical "
        "definition below -- and not shown here: Bollinger's %B is a close "
        "affine variant of the rolling z-score, while distance-to-MA "
        "normalizes by price level rather than volatility and can diverge "
        "from the z-score materially.)"
    )
    col_rsi, col_z, col_pct = st.columns(3)
    use_rsi = col_rsi.checkbox("RSI", value=True, key="explorer_mr_use_rsi")
    use_zscore = col_z.checkbox(
        "Rolling z-score", value=True, key="explorer_mr_use_zscore"
    )
    use_percentile = col_pct.checkbox(
        "Percentile rank", value=True, key="explorer_mr_use_percentile"
    )

    if use_rsi:
        rsi_window = st.slider("RSI window", 2, 60, 14, key="explorer_mr_rsi_window")
        render_price_chart(
            st,
            {"RSI": rsi(prices, rsi_window)},
            title="RSI (30/70 conventionally mark oversold/overbought)",
            yaxis_title="RSI",
        )

    if use_zscore:
        z_window = st.slider(
            "Z-score window", 2, 200, 20, key="explorer_mr_zscore_window"
        )
        zscore = rolling_zscore(prices, z_window)
        render_price_chart(
            st,
            {"Z-score": zscore},
            title="Rolling z-score (this is exactly what MeanReversionStrategy "
            "trades when indicator='zscore', its default -- see the State "
            "machine section below for entry/exit/stop thresholds overlaid "
            "on whichever indicator is selected there)",
            yaxis_title="Z-score",
        )

    if use_percentile:
        pct_window = st.slider(
            "Percentile window", 5, 200, 20, key="explorer_mr_pct_window"
        )
        percentile = rolling_percentile_rank(prices, pct_window)
        render_price_chart(
            st,
            {"Percentile rank": percentile},
            title="Trailing percentile rank (0 = lowest in window, "
            "1 = highest, 0.5 = middle)",
            yaxis_title="Percentile",
        )

    st.markdown("#### State machine (indicator / entry / exit / stop / long_only)")
    st.caption(
        "The strategy state emitted by `MeanReversionStrategy` on this "
        "data for the parameters below -- computed by calling the real "
        "strategy class directly, for whichever indicator is selected. It "
        "is still a signal: allocator, constraints, rebalancing and "
        "execution determine the final portfolio weight."
    )
    from quantlab.dashboard.components import (
        entry_threshold_bounds,
        exit_threshold_bounds,
        mean_reversion_slider_bounds,
    )
    from quantlab.strategies.mean_reversion import UI_INDICATORS

    sm_indicator = st.selectbox(
        "indicator",
        list(UI_INDICATORS),
        key="explorer_mr_sm_indicator",
        help="Selecting a different indicator resets the thresholds below "
        "to that indicator's own defaults -- a threshold tuned for one "
        "indicator's scale is not meaningful on another's.",
    )
    (
        entry_min,
        entry_max,
        entry_default,
        exit_default,
        stop_max,
        stop_default,
        step,
    ) = mean_reversion_slider_bounds(sm_indicator)
    # Asked BEFORE the entry slider (not after) so entry's own bounds can
    # already know whether the stop slider will even be rendered -- see
    # entry_threshold_bounds's own docstring.
    sm_use_stop = st.checkbox(
        "stop_threshold enabled",
        value=True,
        key=f"explorer_mr_sm_use_stop_{sm_indicator}",
    )
    entry_min, entry_max = entry_threshold_bounds(
        entry_min, entry_max, stop_max, step, stop_enabled=sm_use_stop
    )
    entry_default = min(max(entry_default, entry_min), entry_max)

    col_entry, col_exit = st.columns(2)
    sm_entry = col_entry.slider(
        "entry_threshold",
        entry_min,
        entry_max,
        entry_default,
        step,
        key=f"explorer_mr_sm_entry_{sm_indicator}",
    )
    sm_exit_bounds = exit_threshold_bounds(sm_entry, step)
    if sm_exit_bounds is None:
        col_exit.caption(
            "exit_threshold: 0.0 (the only value possible this close to zero)"
        )
        sm_exit = 0.0
    else:
        sm_exit_min, sm_exit_max = sm_exit_bounds
        sm_exit = col_exit.slider(
            "exit_threshold",
            sm_exit_min,
            sm_exit_max,
            min(exit_default, sm_exit_max),
            step,
            key=f"explorer_mr_sm_exit_{sm_indicator}",
        )
    sm_stop = (
        st.slider(
            "stop_threshold (limits the indicator's own deviation "
            "tolerated, not the realized monetary loss)",
            sm_entry + step,
            stop_max,
            max(stop_default, sm_entry + step),
            step,
            key=f"explorer_mr_sm_stop_{sm_indicator}",
        )
        if sm_use_stop
        else None
    )
    sm_long_only = st.checkbox(
        "long_only (short entries never trigger when True)",
        value=True,
        key="explorer_mr_sm_long_only",
    )
    sm_lookback = st.slider(
        "lookback_period (for this state machine)",
        2,
        200,
        20,
        key="explorer_mr_sm_lookback",
    )
    try:
        from quantlab.strategies.mean_reversion import (
            MeanReversionStrategy,
            _centered_indicator,
        )

        strategy = MeanReversionStrategy(
            lookback_period=sm_lookback,
            indicator=sm_indicator,
            entry_threshold=sm_entry,
            exit_threshold=sm_exit,
            stop_threshold=sm_stop,
            long_only=sm_long_only,
        )
        state = strategy.generate_signals(data)[symbol]
        indicator = _centered_indicator(
            prices_frame[[symbol]], sm_indicator, sm_lookback, 2.0
        )[symbol]
    except Exception as exc:
        logger.exception("Mean Reversion lab: could not compute the state machine")
        st.error(f"Could not compute the state machine for these parameters: {exc}")
    else:
        threshold_series, line_colors = centered_indicator_threshold_overlay(
            indicator,
            f"{sm_indicator} indicator",
            entry_threshold=sm_entry,
            exit_threshold=sm_exit,
            stop_threshold=sm_stop,
        )
        render_price_chart(
            st,
            threshold_series,
            title=f"Centered '{sm_indicator}' indicator with entry/exit/stop "
            "thresholds",
            yaxis_title="Centered indicator",
            colors=line_colors,
        )
        render_price_chart(
            st,
            {"Position (state)": state},
            title="MeanReversionStrategy state signal for these parameters",
            yaxis_title="Signal state",
        )
        time_in_position = float((state != 0.0).mean())
        stop_text = "disabled" if sm_stop is None else f"at {sm_stop:g}"
        st.caption(
            f"Time in position: {strong(f'{time_in_position:.1%}')} of bars. "
            f"{'Long-only' if sm_long_only else 'Long/short'} -- "
            f"stop_threshold {strong(stop_text)}.",
            unsafe_allow_html=True,
        )
        render_stop_loss_take_profit_illustration(
            st, state, prices, key_prefix="explorer_mr"
        )

    st.markdown("#### Stationarity tests")
    st.caption(
        "Is this sample consistent with mean reversion, or does the selected "
        "test fail to reject a unit-root model? The answer depends on the "
        "sample and test specification; it does not validate profitability."
    )
    adf_window = st.slider(
        "Test on the trailing N periods", 30, 1000, 252, key="explorer_mr_adf_window"
    )
    tested_series = prices.dropna().iloc[-adf_window:]
    render_stationarity_card(
        st, adf_test(tested_series), label=f"ADF (last {adf_window} periods)"
    )
    hl = half_life(tested_series)
    hl_text = f"{hl:.1f} periods" if math.isfinite(hl) else "no finite estimate"
    hurst = hurst_exponent(tested_series)
    hurst_text = f"{hurst:.3f}" if math.isfinite(hurst) else "n/a (too little data)"
    col_hl, col_hurst = st.columns(2)
    col_hl.metric("Half-life", hl_text)
    col_hurst.metric("Hurst exponent", hurst_text)
    st.caption(
        "H < 0.5 indicates anti-persistence under this estimator and "
        "sample (~0.5 a random walk, > 0.5 a trending/persistent series). "
        "It is neither a stationarity test nor proof of exploitable mean "
        "reversion -- a descriptive estimate on this sample, not a "
        "hypothesis test."
    )
