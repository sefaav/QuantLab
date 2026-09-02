"""Interactive Time-Series Momentum lab.

Momentum formation for a single asset -> comparing the three
`signal_scaling` modes on the SAME score -> momentum persistence ->
volatility diagnostic (for the volatility_adjusted mode).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def _periods_per_year_for_calendar(calendar: str) -> int:
    """Annualisation factor matching the real strategy's own convention.

    ``TimeSeriesMomentumStrategy``'s own ``periods_per_year`` is injected
    from the experiment's own data frequency, not fixed: 365 for a 24/7
    market, 252 for a session-bound one (see its ``ParameterDoc``'s
    ``typical_range``). Extracted as its own function so this derivation
    is directly testable without needing a Streamlit runtime.
    """
    from quantlab.data.calendar import is_247

    return 365 if is_247(calendar) else 252


def render(st: Any) -> None:
    """Render the Time-Series Momentum interactive lab."""
    from quantlab.dashboard.explorer.shared_components import (
        load_explorer_prices_cached,
        render_price_chart,
        render_stop_loss_take_profit_illustration,
        render_symbol_and_source_picker,
        strong,
    )
    from quantlab.dashboard.state import default_end_date
    from quantlab.data.base import price_matrix
    from quantlab.features.momentum import (
        momentum,
        momentum_persistence,
        volatility_adjusted_momentum,
    )
    from quantlab.features.returns import simple_returns
    from quantlab.features.volatility import realized_volatility

    st.markdown("#### Momentum formation")
    picker_result = render_symbol_and_source_picker(
        st,
        key_prefix="explorer_tsmom",
        default_symbols=("SPY", "QQQ", "TLT", "GLD"),
    )
    if picker_result is None:
        st.info("Pick at least one symbol above to load price data.")
        return
    symbols, source, calendar, use_bundled_demo_data = picker_result
    symbol = st.selectbox("Symbol to analyze", symbols, key="explorer_tsmom_symbol")
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Start date", value=date(2019, 1, 1), key="explorer_tsmom_start"
    )
    end_date = col_end.date_input(
        "End date", value=default_end_date(), key="explorer_tsmom_end"
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
        logger.exception("Time-Series Momentum lab: could not load data for %s", symbol)
        st.error(f"Could not load data for {symbol}: {exc}")
        return
    prices_frame = price_matrix(data)
    if symbol not in prices_frame.columns:
        st.warning(f"No data for {symbol} in this range.")
        return
    prices = prices_frame[symbol]

    col_look, col_skip = st.columns(2)
    lookback = col_look.slider(
        "lookback_period", 21, 504, 252, key="explorer_tsmom_lookback"
    )
    skip = col_skip.slider(
        "skip_period", 0, min(63, lookback - 1), 21, key="explorer_tsmom_skip"
    )
    score = momentum(prices, lookback, skip)
    render_price_chart(
        st, {"Momentum score": score}, title=f"{symbol}: raw momentum score"
    )

    st.markdown("#### Comparing signal_scaling modes")
    st.caption(
        "The same underlying score, mapped to a strategy signal three "
        "different ways. These are pre-long_only, pre-allocator values: "
        "portfolio construction and execution still determine the final weight."
    )
    binary_signal = np.sign(score)
    vol_window = st.slider(
        "volatility_window (for the volatility_adjusted mode only -- "
        "continuous uses its own rolling dispersion of the score, over "
        "lookback_period, not this window)",
        5,
        200,
        63,
        key="explorer_tsmom_vol_window",
    )
    dispersion = score.rolling(lookback, min_periods=min(20, lookback)).std(ddof=1)
    continuous_signal = (score / dispersion).clip(-1.0, 1.0)
    # Derived from the selected calendar -- not a tunable knob exposed
    # here, but must still match the real strategy's own convention (see
    # _periods_per_year_for_calendar's own docstring). Fixing this at 252
    # unconditionally used to silently mis-annualise the volatility_
    # adjusted panel below for a 24/7 (e.g. Binance) selection.
    periods_per_year = _periods_per_year_for_calendar(calendar)
    vol = realized_volatility(
        simple_returns(prices), window=vol_window, periods_per_year=periods_per_year
    )
    # Calls the exact same public helper TimeSeriesMomentumStrategy itself
    # delegates to for this mode, rather than reimplementing the division
    # -- a zero-volatility window is masked to NaN there, never silently
    # producing a false +-1.0 signal via a stray inf/-inf before `.clip()`.
    vol_adjusted_signal = volatility_adjusted_momentum(
        prices, lookback, skip, vol_window, periods_per_year
    ).clip(-1.0, 1.0)
    render_price_chart(
        st,
        {
            "binary": binary_signal,
            "continuous": continuous_signal,
            "volatility_adjusted": vol_adjusted_signal,
        },
        title=f"{symbol}: the same score under each signal_scaling mode",
        yaxis_title="Signal",
    )
    illustration_mode = st.selectbox(
        "Illustrate stop-loss/take-profit on which signal_scaling mode",
        ["binary", "continuous", "volatility_adjusted"],
        key="explorer_tsmom_illustration_mode",
    )
    illustration_signal = {
        "binary": binary_signal,
        "continuous": continuous_signal,
        "volatility_adjusted": vol_adjusted_signal,
    }[illustration_mode]
    render_stop_loss_take_profit_illustration(
        st, illustration_signal.fillna(0.0), prices, key_prefix="explorer_tsmom"
    )
    render_price_chart(
        st,
        {"Annualized volatility": vol},
        title="Volatility used by the volatility_adjusted mode "
        f"(annualised at {periods_per_year} periods/year)",
        yaxis_title="Volatility",
    )

    st.markdown("#### Momentum persistence")
    st.caption(
        "Does a high past momentum score actually predict a higher "
        "subsequent return for THIS asset, on this data?"
    )
    holding_period = st.slider(
        "holding_period (for the future return)",
        1,
        126,
        21,
        key="explorer_tsmom_holding",
    )
    paired = momentum_persistence(prices, lookback, skip, holding_period)
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
    correlation = paired["past_momentum"].corr(paired["future_return"])
    st.caption(
        f"Correlation between the two columns above: "
        f"{strong(f'{correlation:.3f}')}. Descriptive sample evidence, not a "
        "hypothesis test -- overlapping holding periods across consecutive "
        "dates are not independent observations.",
        unsafe_allow_html=True,
    )
