"""Interactive Pairs Trading lab.

Workflow: universe selection -> correlation screening -> pair inspection ->
hedge ratio -> spread -> stationarity/cointegration -> mean-reversion
characteristics -> trading thresholds. Every number here comes from
``quantlab.features.pairs_diagnostics.compute_pair_diagnostics`` -- the
exact function the Results tab and the HTML report also use, so this lab
shows the same hedge ratio and ADF p-value as a real backtest of the same
pair whenever the data range, symbols, price type and parameters match
(this lab's own controls let a user explore different ones on purpose).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def render(st: Any) -> None:
    """Render the Pairs Trading interactive lab (see module docstring)."""
    from quantlab.dashboard.explorer.shared_components import (
        VIABLE_ENTRY_MARKER_COLOR,
        centered_indicator_threshold_overlay,
        load_explorer_prices_cached,
        render_correlation_matrix,
        render_price_chart,
        render_stationarity_card,
        render_symbol_and_source_picker,
        strong,
    )
    from quantlab.dashboard.state import default_end_date
    from quantlab.data.base import price_matrix
    from quantlab.features.correlation import correlation_matrix
    from quantlab.features.pairs_diagnostics import compute_pair_diagnostics

    st.markdown("#### 1. Universe selection")
    picker_result = render_symbol_and_source_picker(
        st,
        key_prefix="explorer_pairs",
        default_symbols=("SPY", "QQQ", "TLT", "GLD"),
    )
    if picker_result is None:
        st.info("Pick at least two symbols above to continue.")
        return
    symbols, source, calendar, use_bundled_demo_data = picker_result
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Start date", value=date(2019, 1, 1), key="explorer_pairs_start"
    )
    end_date = col_end.date_input(
        "End date", value=default_end_date(), key="explorer_pairs_end"
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
        logger.exception("Pairs Trading lab: could not load data for %s", symbols)
        st.error(f"Could not load data for {symbols}: {exc}")
        return
    prices = price_matrix(data)
    available = [symbol for symbol in symbols if symbol in prices.columns]
    if len(available) < 2:
        st.warning(
            "Fewer than two of the selected symbols have data in this range "
            "-- widen the date range or pick different symbols."
        )
        return

    st.markdown("#### 2. Correlation analysis")
    st.caption(
        "Correlation alone is not sufficient for pairs trading: two assets "
        "can be highly correlated in returns yet never form a stable, "
        "tradable spread. Use this to screen candidates, not to pick a pair."
    )
    render_correlation_matrix(st, correlation_matrix(prices[available]))

    st.markdown("#### 3. Pair inspection")
    col_a, col_b = st.columns(2)
    symbol_a = col_a.selectbox("Symbol A", available, key="explorer_pairs_a")
    remaining = [symbol for symbol in available if symbol != symbol_a] or available
    symbol_b = col_b.selectbox("Symbol B", remaining, key="explorer_pairs_b")
    normalized = prices[[symbol_a, symbol_b]] / prices[[symbol_a, symbol_b]].iloc[0]
    render_price_chart(
        st,
        {symbol_a: normalized[symbol_a], symbol_b: normalized[symbol_b]},
        title="Normalized prices (both start at 1.0)",
        yaxis_title="Normalized level",
    )

    st.markdown("#### 4-5. Hedge ratio and spread")
    st.caption(
        "The hedge ratio (beta) comes from a trailing OLS fit of A on B "
        "over `formation_window`; the spread is A minus that fitted line. "
        "The sliders below default smaller than PairsTradingStrategy's own "
        "defaults (formation_window=252, indicator_window=63) so this lab "
        "produces a usable spread on the shorter bundled offline demo "
        "date ranges -- widen them to match a real backtest's config."
    )
    formation_window = st.slider(
        "formation_window (periods)", 20, 500, 100, key="explorer_pairs_formation"
    )
    indicator_window = st.slider(
        "indicator_window (periods)", 5, 200, 20, key="explorer_pairs_indicator_window"
    )
    dynamic = st.checkbox(
        "dynamic_hedge_ratio (refit the OLS every period vs. once at formation)",
        value=True,
        key="explorer_pairs_dynamic",
    )

    try:
        diagnostics = compute_pair_diagnostics(
            prices,
            symbol_a,
            symbol_b,
            formation_window=formation_window,
            indicator_window=indicator_window,
            dynamic_hedge_ratio=dynamic,
        )
    except Exception as exc:
        logger.exception(
            "Pairs Trading lab: could not compute diagnostics for %s/%s",
            symbol_a,
            symbol_b,
        )
        st.error(f"Could not compute diagnostics for {symbol_a}/{symbol_b}: {exc}")
        return

    render_price_chart(
        st,
        {"Hedge ratio (beta)": diagnostics.hedge_ratio},
        title="Rolling hedge ratio",
        yaxis_title="Beta",
    )
    render_price_chart(
        st,
        {"Spread": diagnostics.spread},
        title=f"{symbol_a}/{symbol_b} spread (A - intercept - beta*B)",
        yaxis_title="Spread",
    )

    st.markdown("#### 6. Stationarity and cointegration")
    st.caption(
        "Distinct questions: is the spread itself stationary (ADF), and are "
        "A and B cointegrated as a pair (Engle-Granger)? A pair can pass one "
        "and not the other. The displayed full-sample ADF is exploratory; "
        "with a dynamic hedge it tests an adaptively assembled rolling spread."
    )
    render_stationarity_card(st, diagnostics.adf_result, label="ADF (spread)")
    render_stationarity_card(
        st, diagnostics.cointegration_result, label="Engle-Granger cointegration"
    )
    if diagnostics.rolling_adf_pvalue.notna().any():
        render_price_chart(
            st,
            {"Rolling ADF p-value": diagnostics.rolling_adf_pvalue},
            title="Causal periodic ADF gate used for new entries",
            yaxis_title="p-value",
        )
    else:
        st.info(
            "Not enough history yet for a rolling stationarity check "
            "-- widen the date range or shrink formation_window."
        )

    st.markdown("#### 7. Mean-reversion characteristics")
    col_hl, col_stab = st.columns(2)
    half_life_text = (
        f"{diagnostics.half_life:.1f} periods"
        if math.isfinite(diagnostics.half_life)
        else "no finite estimate for this sample"
    )
    col_hl.metric("Half-life", half_life_text)
    col_stab.metric(
        "Hedge-ratio stability (std of beta)",
        f"{diagnostics.hedge_ratio_stability:.4f}"
        if math.isfinite(diagnostics.hedge_ratio_stability)
        else "n/a",
    )
    st.caption(
        "A larger hedge-ratio std means the fitted slope changes more in "
        "this sample. It is measured in beta's scale-dependent units, so "
        "compare it across settings for the same ordered pair, not across pairs."
    )

    st.markdown("#### 8. Trading thresholds")
    st.caption(
        "The same three-indicator choice as Mean Reversion (see that "
        "strategy's own page), applied to this pair's spread instead of a "
        "raw price."
    )
    from quantlab.dashboard.components import (
        entry_threshold_bounds,
        exit_threshold_bounds,
        mean_reversion_slider_bounds,
    )
    from quantlab.strategies.pairs_trading import INDICATORS as PAIRS_INDICATORS
    from quantlab.strategies.pairs_trading import _centered_spread_indicator

    indicator_choice = st.selectbox(
        "indicator", list(PAIRS_INDICATORS), key="explorer_pairs_indicator"
    )
    (
        entry_min,
        entry_max,
        entry_default,
        exit_default,
        stop_max,
        stop_default,
        step,
    ) = mean_reversion_slider_bounds(indicator_choice)
    # Asked BEFORE the entry slider (not after) so entry's own bounds can
    # already know whether the stop slider will even be rendered -- see
    # entry_threshold_bounds's own docstring.
    use_stop = st.checkbox(
        "stop_threshold enabled",
        value=True,
        key=f"explorer_pairs_use_stop_{indicator_choice}",
    )
    entry_min, entry_max = entry_threshold_bounds(
        entry_min, entry_max, stop_max, step, stop_enabled=use_stop
    )
    entry_default = min(max(entry_default, entry_min), entry_max)
    entry = st.slider(
        "entry_threshold",
        entry_min,
        entry_max,
        entry_default,
        step,
        key=f"explorer_pairs_entry_{indicator_choice}",
    )
    pairs_exit_bounds = exit_threshold_bounds(entry, step)
    if pairs_exit_bounds is None:
        st.caption("exit_threshold: 0.0 (the only value possible this close to zero)")
        exit_ = 0.0
    else:
        pairs_exit_min, pairs_exit_max = pairs_exit_bounds
        exit_ = st.slider(
            "exit_threshold",
            pairs_exit_min,
            pairs_exit_max,
            min(exit_default, pairs_exit_max),
            step,
            key=f"explorer_pairs_exit_{indicator_choice}",
        )
    stop = (
        st.slider(
            "stop_threshold (force-closes a position regardless of direction "
            "-- limits the indicator's own deviation tolerated, not the "
            "realized monetary loss)",
            entry + step,
            stop_max,
            max(stop_default, entry + step),
            step,
            key=f"explorer_pairs_stop_{indicator_choice}",
        )
        if use_stop
        else None
    )
    use_adf_gate = st.checkbox(
        "Require the ADF stationarity gate for a viable entry",
        value=True,
        key="explorer_pairs_use_adf_gate",
    )
    adf_threshold = (
        st.slider(
            "adf_pvalue_threshold",
            0.01,
            0.50,
            0.10,
            0.01,
            key="explorer_pairs_adf_threshold",
        )
        if use_adf_gate
        else None
    )

    indicator = _centered_spread_indicator(
        diagnostics.spread, indicator_choice, indicator_window
    )
    threshold_series, line_colors = centered_indicator_threshold_overlay(
        indicator,
        f"{indicator_choice} indicator",
        entry_threshold=entry,
        exit_threshold=exit_,
        stop_threshold=stop,
    )

    # A date is a VIABLE entry only when the indicator actually crosses the
    # entry threshold AND (if the gate is enabled) the rolling ADF p-value
    # at that date is <= adf_threshold -- mirrors PairsTradingStrategy.
    # _stationarity_gate's own condition exactly, using the same rolling
    # ADF series already displayed in step 6 above (never a second,
    # potentially diverging ADF computation).
    crosses_entry = (indicator > entry) | (indicator < -entry)
    if adf_threshold is not None:
        gate_open = diagnostics.rolling_adf_pvalue.reindex(indicator.index) <= (
            adf_threshold
        )
        gate_open = gate_open.fillna(False)
    else:
        gate_open = pd.Series(True, index=indicator.index)
    viable = crosses_entry & gate_open
    viable_marker = indicator.where(viable)
    markers = (
        {"Viable entry (threshold crossed + ADF gate open)": viable_marker}
        if viable.any()
        else None
    )
    marker_colors = (
        {"Viable entry (threshold crossed + ADF gate open)": VIABLE_ENTRY_MARKER_COLOR}
        if markers
        else {}
    )
    render_price_chart(
        st,
        threshold_series,
        title=f"'{indicator_choice}' indicator of the spread, with threshold "
        "overlays (not simulated trades)",
        yaxis_title="Centered indicator",
        markers=markers,
        colors={**line_colors, **marker_colors},
    )
    n_viable = int(viable.sum())
    st.caption(
        f"{strong(str(n_viable))} bar(s) in this sample are a viable entry: "
        "the entry threshold is crossed AND"
        + (
            " the ADF gate is open."
            if adf_threshold is not None
            else " (gate disabled)."
        )
        + " Not an entry count: current state, rebalancing and execution "
        "can still prevent or delay a trade.",
        unsafe_allow_html=True,
    )
    if stop is not None:
        stop_breaches = int(((indicator > stop) | (indicator < -stop)).sum())
        st.caption(
            f"{strong(str(stop_breaches))} bar(s) in this sample cross the "
            "stop threshold. If a position were open, the strategy state "
            "would request flat; rebalancing and execution determine when "
            "the weight changes. This limits further indicator-distance "
            "exposure but NOT the realized monetary loss on the "
            "way there (gaps/execution delay/costs can still exceed what "
            "the indicator distance alone suggests).",
            unsafe_allow_html=True,
        )
    breaches = int(crosses_entry.sum())
    st.caption(
        f"{strong(str(breaches))} bar(s) in this sample cross the entry "
        "threshold (before the ADF gate). A higher entry_threshold means "
        "fewer threshold breaches; a lower exit_threshold means each trade is "
        "held closer to full mean reversion before closing (more time in "
        "the trade, less residual edge left uncaptured).",
        unsafe_allow_html=True,
    )

    try:
        from quantlab.dashboard.explorer.shared_components import (
            render_stop_loss_take_profit_illustration,
        )
        from quantlab.strategies.pairs_trading import PairsTradingStrategy

        pair_strategy = PairsTradingStrategy(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            formation_window=formation_window,
            indicator_window=indicator_window,
            indicator=indicator_choice,
            entry_threshold=entry,
            exit_threshold=exit_,
            stop_threshold=stop,
            dynamic_hedge_ratio=dynamic,
            adf_pvalue_threshold=adf_threshold,
        )
        pair_signals = pair_strategy.generate_signals(data)
    except Exception as exc:
        logger.exception(
            "Pairs Trading lab: could not compute signals for the "
            "stop-loss/take-profit illustration"
        )
        st.error(f"Could not compute the pair's signal for this illustration: {exc}")
    else:
        render_stop_loss_take_profit_illustration(
            st,
            {symbol_a: pair_signals[symbol_a], symbol_b: pair_signals[symbol_b]},
            {symbol_a: prices[symbol_a], symbol_b: prices[symbol_b]},
            key_prefix="explorer_pairs",
            position_groups=((symbol_a, symbol_b),),
        )
