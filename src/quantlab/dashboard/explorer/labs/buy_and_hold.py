"""Interactive Buy & Hold lab.

Price explorer, plus the strategy's own counter-intuitive point: a
"buy and hold" *signal* does not, in a real portfolio held at fixed share
counts, mean *static weights* once more than one asset is involved -- this
lab illustrates that theoretical drift. QuantLab's own accounting engine
does NOT currently reproduce it (see the caption near the chart below and
docs/limitations.md's "Rebalancing is a step function" note): it holds
weights constant between rebalance dates by construction, a deliberate
vectorised-backtest simplification, not (yet) a price-driven recomputation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from quantlab.logging_config import get_logger

logger = get_logger(__name__)


def render(st: Any) -> None:
    """Render the Buy & Hold interactive lab."""
    from quantlab.dashboard.explorer.shared_components import (
        load_explorer_prices_cached,
        render_price_chart,
        render_symbol_and_source_picker,
    )
    from quantlab.dashboard.state import default_end_date
    from quantlab.data.base import price_matrix

    st.markdown("#### Price explorer")
    picker_result = render_symbol_and_source_picker(
        st,
        key_prefix="explorer_bh",
        default_symbols=("SPY", "QQQ"),
    )
    if picker_result is None:
        st.info("Pick at least one symbol above to continue.")
        return
    symbols, source, calendar, use_bundled_demo_data = picker_result
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Start date", value=date(2019, 1, 1), key="explorer_bh_start"
    )
    end_date = col_end.date_input(
        "End date", value=default_end_date(), key="explorer_bh_end"
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
        logger.exception("Buy & Hold lab: could not load data")
        st.error(f"Could not load data: {exc}")
        return
    prices = price_matrix(data)
    available = [symbol for symbol in symbols if symbol in prices.columns]
    if not available:
        st.warning("None of the selected symbols have data in this range.")
        return
    render_price_chart(
        st, {symbol: prices[symbol] for symbol in available}, title="Price"
    )

    if len(available) < 2:
        st.info(
            "Pick a second symbol below to see why 'always invested' does "
            "not mean 'static weights' once more than one asset is held."
        )
        return

    st.markdown("#### Why weights would drift even though the signal never changes")
    st.caption(
        "buy_and_hold's own signal is simply 'invested wherever price data "
        "exists' -- it never rebalances by itself. In a REAL portfolio held "
        "at fixed share counts, each asset's OWN return would move its "
        "share of the total value, so realized weights would drift away "
        "from equal (or whatever the initial split was) purely from price "
        "divergence, well before any portfolio-level rebalance schedule "
        "intervenes."
    )
    initial_weight = 1.0 / len(available)
    normalized = prices[available] / prices[available].iloc[0]
    drifted_value = normalized * initial_weight
    drifted_weights = drifted_value.div(drifted_value.sum(axis=1), axis=0)
    render_price_chart(
        st,
        {symbol: drifted_weights[symbol] for symbol in available},
        title="Theoretical weight drift with no rebalancing (starting "
        f"equal at {initial_weight:.0%} each) -- illustrative, not what "
        "QuantLab's accounting currently reproduces",
        yaxis_title="Weight",
    )
    st.caption(
        "**This chart is a theoretical illustration of share-count drift, "
        "not a preview of a QuantLab backtest.** QuantLab's own accounting "
        "engine currently holds weights CONSTANT between rebalance dates "
        "by construction (a deliberate vectorised-backtest simplification "
        "-- see docs/limitations.md, 'Rebalancing is a step function'), so "
        "it does not (yet) reproduce the drift shown above. In a real "
        "portfolio, the configured `rebalance_frequency` would periodically "
        "reset this drift back toward target -- less often means more "
        "drift between resets, more often means closer to the target split "
        "but more turnover/costs."
    )
