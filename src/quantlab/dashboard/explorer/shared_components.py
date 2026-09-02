"""Reusable Strategy Explorer UI components -- presentation only, no calculation.

Every function takes ``st`` (the Streamlit module, or a compatible fake for
tests) as its first argument, mirroring ``quantlab.dashboard.components``'s
own convention. The actual numbers always come from ``quantlab.features.*``
or a lab's own recomputation -- nothing here fits a statistic or builds a
signal; it only draws what it is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, cast

import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features.stationarity import ADFResult, CointegrationResult
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

#: Caps the shared lab price cache below -- a long dashboard session
#: trying many symbol/date/source combinations across every lab must not
#: grow it without bound.
_PRICE_CACHE_MAX_ENTRIES = 32

#: Shared "one color per role" palette for entry/exit/stop threshold lines
#: across labs -- both the positive and negative side of a threshold get
#: the SAME color (they are the same concept, mirrored), and different
#: roles get visibly different colors. Deliberately avoids blue: the
#: underlying price/indicator series these overlay is typically blue
#: (Plotly's own default first trace color).
ENTRY_LINE_COLOR = "#FF8C00"  # orange
EXIT_LINE_COLOR = "#2CA02C"  # green
STOP_LINE_COLOR = "#D62728"  # red
VIABLE_ENTRY_MARKER_COLOR = "#9467BD"  # purple, distinct from the 3 roles above


def live_widget_value(key: str, default: Any) -> Any:
    """Read a widget's CURRENT value from ``st.session_state``, if any.

    Used so a Results-tab report (generated separately from the
    interactive render pass -- see ``quantlab.dashboard.explorer.profile.
    ResultsDiagnostics``) can reflect a user's live diagnostic-only widget
    choice (e.g. a forward-return horizon slider) without threading that
    value through the ``compute``/``render``/``report_section`` pipeline
    explicitly. Falls back to ``default`` whenever that widget's value
    isn't available -- not just when Streamlit itself isn't installed, but
    also the ordinary case of the CLI's own report generation (``quantlab.
    cli``), which calls ``report_section`` directly with no Streamlit
    runtime/session at all, and the case where this particular widget was
    never rendered this session (e.g. its section was never opened).

    ``streamlit`` missing entirely is the one case narrowed to a silent
    fallback (documented, expected -- e.g. a stripped-down test
    environment). Reading ``st.session_state`` itself failing for any
    OTHER reason is unexpected (empirically, it degrades gracefully with
    no exception even with no active script run context) -- logged rather
    than swallowed silently, so an unrelated bug degrading a dashboard
    value still leaves a trace instead of vanishing without one.
    """
    try:
        import streamlit as st
    except ImportError:
        return default
    try:
        return st.session_state.get(key, default)
    except Exception:
        logger.exception(
            "live_widget_value(%r) could not read st.session_state -- "
            "falling back to default %r",
            key,
            default,
        )
        return default


def strong(value: str) -> str:
    """Wrap a standout value in a heavier-than-markdown-bold inline span.

    For use inside an ``st.caption``/``st.markdown`` string passed with
    ``unsafe_allow_html=True``. Plain markdown ``**bold**`` renders at a
    single fixed weight that a caption's own small, muted styling can
    wash out -- this pushes the weight further so a key computed number
    still reads as emphasized against the surrounding caption text.
    """
    return f'<strong style="font-weight:800">{value}</strong>'


def centered_indicator_threshold_overlay(
    indicator: pd.Series,
    indicator_name: str,
    *,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float | None,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Build the named +/-entry/+/-exit/+/-stop series and their line colors.

    Shared by every "centered indicator with entry/exit/stop thresholds"
    chart (mean_reversion, pairs_trading -- both their interactive labs and
    their Results-tab diagnostics), so the naming/coloring convention stays
    identical everywhere this chart appears. Returns ``(series, colors)``,
    both keyed by the same threshold labels, ready to pass straight into
    :func:`render_price_chart` (``series`` merged with the caller's own
    indicator entry, ``colors`` merged with any of the caller's own, e.g.
    a viable-entry marker color).
    """
    series: dict[str, pd.Series] = {
        indicator_name: indicator,
        f"+entry {entry_threshold:g}": pd.Series(
            entry_threshold, index=indicator.index
        ),
        f"-entry {entry_threshold:g}": pd.Series(
            -entry_threshold, index=indicator.index
        ),
        f"+exit {exit_threshold:g}": pd.Series(exit_threshold, index=indicator.index),
        f"-exit {exit_threshold:g}": pd.Series(-exit_threshold, index=indicator.index),
    }
    colors: dict[str, str] = {
        f"+entry {entry_threshold:g}": ENTRY_LINE_COLOR,
        f"-entry {entry_threshold:g}": ENTRY_LINE_COLOR,
        f"+exit {exit_threshold:g}": EXIT_LINE_COLOR,
        f"-exit {exit_threshold:g}": EXIT_LINE_COLOR,
    }
    if stop_threshold is not None:
        series[f"+stop {stop_threshold:g}"] = pd.Series(
            stop_threshold, index=indicator.index
        )
        series[f"-stop {stop_threshold:g}"] = pd.Series(
            -stop_threshold, index=indicator.index
        )
        colors[f"+stop {stop_threshold:g}"] = STOP_LINE_COLOR
        colors[f"-stop {stop_threshold:g}"] = STOP_LINE_COLOR
    return series, colors


def load_explorer_prices_cached(
    st: Any,
    symbols: Sequence[str],
    *,
    source: str,
    calendar: str,
    start_date: date,
    end_date: date,
    use_bundled_demo_data: bool = False,
) -> pd.DataFrame:
    """Load Strategy Explorer lab price data through one shared, bounded cache.

    Every lab shares this one ``@st.cache_data`` cache, keyed the same way
    and bounded with ``max_entries`` so a long session trying many symbol/
    date/source combinations cannot grow it without limit. ``source``/
    ``calendar``/``use_bundled_demo_data`` are real cache-key arguments
    (not closed-over constants) so two calls that differ only in one of
    them can never collide.
    """
    from quantlab.dashboard.state import load_explorer_prices

    @st.cache_data(
        show_spinner="Loading price data...", max_entries=_PRICE_CACHE_MAX_ENTRIES
    )
    def _load(
        symbols: tuple[str, ...],
        source: str,
        calendar: str,
        start: date,
        end: date,
        use_bundled_demo_data: bool,
    ) -> pd.DataFrame:
        return load_explorer_prices(
            list(symbols),
            source=source,
            calendar=calendar,
            start_date=start,
            end_date=end,
            use_bundled_demo_data=use_bundled_demo_data,
        )

    return cast(
        pd.DataFrame,
        _load(
            tuple(sorted(symbols)),
            source,
            calendar,
            start_date,
            end_date,
            use_bundled_demo_data,
        ),
    )


def render_symbol_and_source_picker(
    st: Any,
    *,
    key_prefix: str,
    default_symbols: Sequence[str],
    default_calendar: str = "XNYS",
) -> tuple[list[str], str, str, bool] | None:
    """Let a lab pick real data via the same data sources backtest/walk-forward use.

    Unlike the main dashboard's own per-instrument ``InstrumentConfig``
    table, every symbol here shares ONE calendar, since a lab computes on
    a single flat price matrix, not individually-configured instruments.

    Returns ``(symbols, source, calendar, use_bundled_demo_data)``, or
    ``None`` when nothing is selected yet (the caller should show a message
    rather than proceed) OR when the selected symbols would need more than
    one calendar (a lab cannot represent that -- the caller shows an error
    instead of silently picking one calendar for all of them). Every widget
    key is prefixed by ``key_prefix`` so two labs -- or a lab and the main
    sidebar -- never collide over the same ``session_state`` entry, notably
    the Binance "Load Binance symbols" gate, whose flag would otherwise be a
    single dashboard-wide switch shared by everything that uses it.
    """
    from quantlab.config import DataSourceName
    from quantlab.dashboard.components import (
        binance_universe_labels,
        parse_symbols,
        symbols_picker,
        yahoo_universe_labels,
    )
    from quantlab.data.resolution import detect_calendar

    source = st.radio(
        "Data source",
        ["csv", "yahoo", "binance"],
        key=f"{key_prefix}_source",
        horizontal=True,
        help=(
            "csv: local files under data/raw (bundled synthetic demo "
            "data available as an offline fallback below). yahoo/"
            "binance: the same data sources backtest/walk-forward use."
        ),
    )
    if source == "yahoo":
        symbols = symbols_picker(
            st,
            yahoo_universe_labels(),
            f"{key_prefix}_yahoo_symbols",
            tuple(default_symbols),
            accept_new_options=True,
        )
        if not symbols:
            return None
        # Best-effort per-symbol guess (e.g. "1211.HK" -> XHKG) -- a bare
        # US ticker with no recognized suffix falls back to XNYS, exactly
        # like the main dashboard's own per-instrument table default.
        detected = {
            detect_calendar(symbol, DataSourceName.YAHOO) or default_calendar
            for symbol in symbols
        }
        if len(detected) > 1:
            st.error(
                "Selected symbols need different calendars "
                f"({', '.join(sorted(detected))}) -- this lab computes on a "
                "single shared price matrix and cannot represent more than "
                "one calendar at once. Pick symbols on the same market, or "
                "use the main dashboard's Backtest mode (each instrument "
                "gets its own calendar there)."
            )
            return None
        # A keyed st.text_input only honours its `value=` argument the
        # FIRST time that key is created -- once session_state holds a
        # value for it, passing a freshly-detected default on a later
        # rerun is silently ignored by Streamlit itself, leaving the field
        # stuck on a stale guess after the symbol selection changes (e.g.
        # AAPL -> XNYS auto-filled, then swapped for 1211.HK, still
        # showing XNYS). Detected here explicitly: only when the symbol
        # SET actually changed since the guess was last made is the
        # session_state value overwritten -- an unrelated rerun (a
        # different widget elsewhere) never clobbers the user's own
        # manual edit.
        calendar_key = f"{key_prefix}_yahoo_calendar"
        symbols_for_key = f"{calendar_key}_for_symbols"
        symbols_tuple = tuple(symbols)
        if st.session_state.get(symbols_for_key) != symbols_tuple:
            st.session_state[calendar_key] = next(iter(detected))
            st.session_state[symbols_for_key] = symbols_tuple
        calendar = st.text_input(
            "Calendar",
            key=calendar_key,
            help="Auto-detected from the symbol suffix where possible "
            "(e.g. '.HK' -> XHKG) -- edit if the guess is wrong. '24/7' "
            "for a continuous market, or a pandas_market_calendars name "
            "such as XNYS, XHKG, XLON.",
        ).strip()
        if not calendar:
            return None
        return symbols, source, calendar, False
    if source == "binance":
        load_flag_key = f"{key_prefix}_binance_universe_load_requested"
        if not st.session_state.get(load_flag_key, False):
            st.caption(
                "Loading the tradable symbol list calls Binance's public "
                "API. Click below to fetch it (cached for an hour after "
                "that)."
            )
            if st.button(
                "Load Binance symbols",
                key=f"{key_prefix}_binance_universe_load_button",
            ):
                # See app.py's `_binance_symbols_picker` for why this
                # deliberately does NOT call st.rerun(): that would abort
                # this run right here, before finishing, dropping this
                # widget's own keyed session-state value back to its
                # default on the very next rerun.
                st.session_state[load_flag_key] = True
            else:
                return None
        symbols = symbols_picker(
            st, binance_universe_labels(st), f"{key_prefix}_binance_symbols", ()
        )
        if not symbols:
            return None
        return symbols, source, "24/7", False
    # source == "csv"
    raw = st.text_input(
        "CSV symbols (comma-separated)",
        ", ".join(default_symbols),
        key=f"{key_prefix}_csv_symbols_input",
        help=(
            "Local files under data/raw, one CSV per symbol. Falls back "
            "to QuantLab's bundled synthetic demo data below when every "
            "requested local file is absent."
        ),
    )
    symbols = parse_symbols(raw)
    use_bundled_demo_data = st.toggle(
        "Allow bundled synthetic demo data",
        value=True,
        key=f"{key_prefix}_use_bundled_demo_data",
        help=(
            "On by default here (unlike the main dashboard) so this lab "
            "keeps working offline with no setup. Turn off once you have "
            "real local files under data/raw for these symbols."
        ),
    )
    # A bare local filename carries no calendar information at all --
    # defaults to XNYS (matching the main dashboard's own per-instrument
    # table default) but is always editable, since these could just as
    # well be futures, a non-XNYS index, or anything else.
    calendar = st.text_input(
        "Calendar",
        default_calendar,
        key=f"{key_prefix}_csv_calendar",
        help="'24/7' for a continuous market, or a pandas_market_calendars "
        "name such as XNYS, XHKG, XLON -- CSV data carries no calendar "
        "information, so this cannot be auto-detected.",
    ).strip()
    if not symbols or not calendar:
        return None
    return symbols, source, calendar, use_bundled_demo_data


def render_price_chart(
    st: Any,
    series: Mapping[str, pd.Series],
    *,
    title: str,
    height: int = 360,
    yaxis_title: str = "Price",
    markers: Mapping[str, pd.Series] | None = None,
    marker_size: int = 9,
    colors: Mapping[str, str] | None = None,
) -> None:
    """Plot one or more named series on a shared time axis.

    The generic building block every lab uses for "show me this price
    series plus whatever indicator/threshold/overlay is currently
    selected" -- when a widget changes a parameter, the lab recomputes
    the relevant series and calls this again with the new values, which
    is how Streamlit's own rerun-on-interaction model makes a parameter's
    effect immediately visible without a bespoke "impact" widget.

    ``markers``, if given, are drawn as discrete point markers (not
    connected lines) layered on top of ``series`` -- for a sparse,
    date-indexed callout (e.g. "entries actually viable under a filter")
    that would be unreadable as its own connected line. Each series may
    contain ``NaN``/be missing dates freely; Plotly simply skips them.

    ``colors``, if given, maps a series/marker name to an explicit CSS
    color, overriding Plotly's default per-trace color cycling -- lets a
    caller give two differently-named series (e.g. a positive and a
    negative threshold line) the SAME color deliberately, which the
    default cycling (assigns by trace order, not by intent) cannot express.
    """
    import plotly.graph_objects as go

    color_map = colors or {}
    fig = go.Figure()
    for name, values in series.items():
        line = {"color": color_map[name]} if name in color_map else None
        fig.add_trace(
            go.Scatter(x=values.index, y=values, mode="lines", name=name, line=line)
        )
    for name, values in (markers or {}).items():
        marker = {"size": marker_size, "symbol": "circle"}
        if name in color_map:
            marker["color"] = color_map[name]
        fig.add_trace(
            go.Scatter(
                x=values.index, y=values, mode="markers", name=name, marker=marker
            )
        )
    fig.update_layout(
        title=title, height=height, xaxis_title="Date", yaxis_title=yaxis_title
    )
    st.plotly_chart(fig, width="stretch")


def render_price_explorer(st: Any, prices: pd.Series, symbol: str) -> None:
    """Plot one symbol's own price series."""
    render_price_chart(st, {symbol: prices}, title=f"{symbol} price")


def render_correlation_matrix(st: Any, matrix: pd.DataFrame) -> None:
    """Render a symbol x symbol correlation matrix as a heatmap + table.

    Shared by the Pairs Trading lab and the Results/report pair-diagnostics
    section -- never reimplemented separately in either place.
    """
    import plotly.graph_objects as go

    values = matrix.to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=[str(c) for c in matrix.columns],
            y=[str(r) for r in matrix.index],
            colorscale="RdBu",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "corr"},
            text=[[f"{v:.2f}" for v in row] for row in values],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Correlation matrix (of returns)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(matrix.round(3), width="stretch")


def render_stationarity_card(
    st: Any,
    result: ADFResult | CointegrationResult | None,
    *,
    label: str,
) -> None:
    """Display one stationarity/cointegration test result in plain language.

    ``None`` (an inconclusive test -- too little data, or a numerical
    failure) is rendered as an explicit message, never silently skipped,
    so a missing result always reads as "inconclusive", not as "the
    section vanished".
    """
    if result is None:
        st.info(f"{label}: inconclusive (too little data, or a numerical failure).")
        return
    verdict = "Reject H0" if result.reject_null else "Cannot reject H0"
    columns = st.columns(3)
    columns[0].metric("Statistic", f"{result.statistic:.4f}")
    columns[1].metric("p-value", f"{result.pvalue:.4f}")
    columns[2].metric("Verdict", verdict)
    critical = ", ".join(f"{k}={v:.3f}" for k, v in result.critical_values.items())
    st.caption(f"**{label}** -- critical values: {critical}. {result.interpretation}")


def render_stop_loss_take_profit_illustration(
    st: Any,
    positions: pd.Series | Mapping[str, pd.Series],
    prices: pd.Series | Mapping[str, pd.Series],
    *,
    key_prefix: str,
    position_groups: Sequence[tuple[str, ...]] | None = None,
) -> None:
    """Illustrate stop-loss/take-profit on the position(s) ALREADY shown above.

    A simplified illustration only, clearly labeled as such: QuantLab's
    real mechanism (``quantlab.backtesting.accounting.
    _detect_stop_loss_take_profit``) operates on the REAL executed
    position(s) after the allocator/constraints/rebalancing/execution --
    this lab has no access to that pipeline. It instead applies the EXACT
    SAME function directly to the position(s)/price(s) already displayed
    above, which is mathematically identical to the real formula's own
    reduction for that case -- not a second, approximate implementation.

    A single ``pd.Series`` pair (the common case: one symbol) is treated
    as one independent group. A ``Mapping[str, pd.Series]`` (e.g.
    pairs_trading's two legs) is treated as one COMBINED group unless
    ``position_groups`` says otherwise -- mirroring
    ``BaseStrategy.position_groups()``'s own default/override convention.
    """
    from quantlab.backtesting.accounting import _detect_stop_loss_take_profit

    if isinstance(positions, pd.Series):
        assert isinstance(prices, pd.Series)
        position_map = {"asset": positions}
        price_map: Mapping[str, pd.Series] = {"asset": prices}
        default_groups = None
    else:
        assert not isinstance(prices, pd.Series)
        position_map = dict(positions)
        price_map = prices
        default_groups = (tuple(position_map),) if len(position_map) > 1 else None

    st.markdown("#### Stop-loss / take-profit illustration")
    st.caption(
        "Simplified: applies QuantLab's exact stop-loss/take-profit "
        "formula directly to the position(s)/price(s) shown above. "
        "QuantLab's real backtest instead operates on the actual "
        "EXECUTED position after the allocator/constraints/rebalancing/"
        "execution, which this lab does not model -- treat this as "
        "illustrative, not a preview of real backtest numbers."
    )
    col_sl, col_tp = st.columns(2)
    stop_loss_pct = col_sl.slider(
        "stop_loss_pct (0 = disabled)",
        0.0,
        0.5,
        0.0,
        0.01,
        key=f"{key_prefix}_illustration_stop_loss",
    )
    take_profit_pct = col_tp.slider(
        "take_profit_pct (0 = disabled)",
        0.0,
        0.5,
        0.0,
        0.01,
        key=f"{key_prefix}_illustration_take_profit",
    )
    if stop_loss_pct <= 0.0 and take_profit_pct <= 0.0:
        return

    aligned_positions = {
        symbol: series.reindex(price_map[symbol].index).fillna(0.0)
        for symbol, series in position_map.items()
    }
    # `fill_method=None`: never let pandas' own version-dependent default
    # forward-fill a gap before computing the return -- a genuine internal
    # missing price must surface as NaN here, not silently vanish before
    # this function even sees it. Only the very FIRST observation (no
    # prior price to compare against at all -- not a "missing" price, a
    # structurally absent one) is explicitly zeroed; every OTHER NaN is a
    # real gap, reported below when it coincides with a held position
    # rather than silently treated as a flat 0% return.
    returns: dict[str, pd.Series] = {}
    missing_while_held: list[str] = []
    for symbol, series in price_map.items():
        pct = series.pct_change(fill_method=None)
        if len(pct):
            pct.iloc[0] = 0.0
        held = aligned_positions[symbol].abs() > EPSILON
        gap_dates = pct.index[pct.isna() & held]
        if len(gap_dates):
            shown = ", ".join(str(d.date()) for d in gap_dates[:5])
            if len(gap_dates) > 5:
                shown += f", +{len(gap_dates) - 5} more"
            missing_while_held.append(f"{symbol}: {shown}")
        returns[symbol] = pct.fillna(0.0)
    if missing_while_held:
        st.warning(
            "Missing price return(s) while a position was held -- treated "
            "as 0% for this illustration only, which can hide a real "
            "stop-loss/take-profit trigger on that date: "
            + "; ".join(missing_while_held)
        )
    stop_loss_result = _detect_stop_loss_take_profit(
        pd.DataFrame(aligned_positions),
        pd.DataFrame(returns),
        position_groups if position_groups is not None else default_groups,
        stop_loss_pct if stop_loss_pct > 0.0 else None,
        take_profit_pct if take_profit_pct > 0.0 else None,
    )
    gated, stop_loss_triggered, take_profit_triggered = stop_loss_result[:3]
    chart_series = {}
    for symbol in position_map:
        chart_series[f"{symbol}: position (as displayed above)"] = aligned_positions[
            symbol
        ]
        chart_series[f"{symbol}: position after stop-loss/take-profit"] = gated[symbol]
    render_price_chart(
        st,
        chart_series,
        title="Illustrative effect of stop-loss/take-profit on this position",
        yaxis_title="Position",
    )
    n_stop = int(stop_loss_triggered.any(axis=1).sum())
    n_take = int(take_profit_triggered.any(axis=1).sum())
    st.caption(f"Stop-loss fired on {n_stop} date(s); take-profit on {n_take} date(s).")
