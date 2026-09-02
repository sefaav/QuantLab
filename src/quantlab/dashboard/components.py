"""Reusable Streamlit UI components.

Streamlit and Plotly are imported lazily so importing this module (e.g. in
tests) does not require the dashboard extra to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from quantlab.backtesting.trade_log import parse_adjustment_codes
from quantlab.dashboard.state import binance_trading_symbols, yahoo_common_symbols
from quantlab.reporting.charts import (
    ACCENT,
    BENCHMARK,
    NEGATIVE,
    STRATEGY,
    adaptive_rolling_window,
    benchmark_legend_label,
)
from quantlab.risk import metrics as M
from quantlab.risk.drawdown import drawdown_series
from quantlab.risk.exposure import gross_exposure_series, net_exposure_series

if TYPE_CHECKING:
    from quantlab.backtesting.result import BacktestResult
    from quantlab.data.base import SymbolSuggestion
    from quantlab.features.pairs_diagnostics import PairDiagnostics

#: Shown in a symbol picker's help tooltip whenever its preloaded list isn't
#: a complete universe -- currently only Yahoo's, since Binance's list
#: genuinely is complete.
INCOMPLETE_LIST_NOTE = (
    "Not every symbol is suggested — if yours is missing, type its exact "
    "ticker and it'll still be accepted."
)


def parse_symbols(raw: str) -> list[str]:
    """Normalise symbols and remove duplicates while preserving their order."""
    return list(dict.fromkeys(s.strip().upper() for s in raw.split(",") if s.strip()))


def label_for(suggestion: SymbolSuggestion) -> str:
    """A symbol's display label -- ``"SYMBOL — description"`` when known."""
    if suggestion.description:
        return f"{suggestion.symbol} — {suggestion.description}"
    return suggestion.symbol


def cached_binance_universe(st: Any) -> list[SymbolSuggestion]:
    """Binance's full active spot-symbol universe, refreshed hourly.

    Fetched once per hour (per Streamlit cache entry, no arguments) rather
    than per keystroke: once loaded, picking symbols from it is instant,
    client-side dropdown filtering — no server round trip per character.
    """

    @st.cache_data(ttl=3600, show_spinner="Loading Binance's symbol list…")
    def _load() -> list[SymbolSuggestion]:
        return binance_trading_symbols()

    return cast("list[SymbolSuggestion]", _load())


def binance_universe_labels(st: Any) -> dict[str, str]:
    """Binance's cached universe as ``{symbol: display label}``."""
    return {s.symbol: label_for(s) for s in cached_binance_universe(st)}


def yahoo_universe_labels() -> dict[str, str]:
    """The bundled S&P 500 + major-ETF reference list as ``{symbol: label}``.

    Yahoo has no downloadable "every symbol" endpoint the way Binance does,
    so this static, bundled list stands in as an instant, offline universe —
    covering what most dashboard users will look for, not every symbol Yahoo
    can actually serve.
    """
    return {s.symbol: label_for(s) for s in yahoo_common_symbols()}


#: (entry_min, entry_max, stop_max, step) UI ranges per mean_reversion
#: `indicator` -- the entry/exit/stop DEFAULTS come from `quantlab.
#: strategies.mean_reversion.INDICATOR_DEFAULT_THRESHOLDS` (the strategy's
#: own single source of truth), not duplicated here.
_MEAN_REVERSION_INDICATOR_UI_RANGES: dict[str, tuple[float, float, float, float]] = {
    "zscore": (0.5, 4.0, 8.0, 0.1),
    "bollinger": (0.1, 3.0, 4.0, 0.05),
    "rsi": (5.0, 50.0, 50.0, 1.0),
    "distance_ma": (0.01, 0.3, 0.4, 0.01),
    "percentile": (0.01, 0.49, 0.49, 0.01),
}


def mean_reversion_slider_bounds(
    indicator: str,
) -> tuple[float, float, float, float, float, float, float]:
    """UI slider bounds for one `mean_reversion` `indicator`'s thresholds.

    Returns ``(entry_min, entry_max, entry_default, exit_default,
    stop_max, stop_default, step)``. The three defaults come from
    ``quantlab.strategies.mean_reversion.INDICATOR_DEFAULT_THRESHOLDS`` --
    shared by the main dashboard sidebar and the Strategy Explorer lab so
    neither can silently drift from what the strategy itself resolves to
    when a threshold is left unset.
    """
    from quantlab.strategies.mean_reversion import INDICATOR_DEFAULT_THRESHOLDS

    entry_min, entry_max, stop_max, step = _MEAN_REVERSION_INDICATOR_UI_RANGES[
        indicator
    ]
    entry_default, exit_default, stop_default = INDICATOR_DEFAULT_THRESHOLDS[indicator]
    return (
        entry_min,
        entry_max,
        entry_default,
        exit_default,
        stop_max,
        stop_default,
        step,
    )


def entry_threshold_bounds(
    entry_min: float,
    entry_max: float,
    stop_max: float,
    step: float,
    *,
    stop_enabled: bool,
) -> tuple[float, float]:
    """Entry threshold's own ``(min, max)``, narrowed only as needed.

    The low end keeps ``entry_min`` UNCHANGED -- entry never loses a
    reachable value (e.g. ``distance_ma``'s/``percentile``'s own
    step-sized minimum, ``0.01``) just to dodge the exit slider's own
    degenerate near-zero case; see :func:`exit_threshold_bounds`, which
    handles that case directly (fixing ``exit_threshold = 0.0`` instead of
    rendering a slider at all) rather than narrowing entry's domain to
    avoid it ever arising.

    The high end is narrowed only for the stop slider, and only when it
    actually exists: its own min is ``entry + step`` (its max is
    ``stop_max``), but it is only rendered when ``stop_enabled`` -- so
    entry only needs to stay 2 steps below ``stop_max`` while that slider
    actually exists. With stop disabled, entry keeps its FULL original
    upper range (e.g. RSI can still reach 50, percentile can still reach
    0.49) rather than silently losing reachable values for a slider that
    was never going to be built.
    """
    hi = min(entry_max, stop_max - 2.0 * step) if stop_enabled else entry_max
    return entry_min, hi


def exit_threshold_bounds(
    entry_threshold: float, step: float
) -> tuple[float, float] | None:
    """Exit threshold's own ``(min, max)``, or ``None`` when only 0.0 is valid.

    The exit slider's max is ``entry_threshold - step`` (its min is fixed
    at ``0.0``). When ``entry_threshold <= step``, that max is ``<= 0.0``
    -- Streamlit's slider rejects ``min == max``, so there is no slider
    to render at all: the only mathematically valid ``exit_threshold`` at
    that point is exactly ``0.0``, and the caller should show that
    directly (a caption or a disabled widget) instead of calling
    ``st.slider`` with a degenerate range.
    """
    exit_max = round(entry_threshold - step, 10)
    if exit_max <= 0.0:
        return None
    return 0.0, exit_max


def symbols_picker(
    st: Any,
    label_by_symbol: dict[str, str],
    key: str,
    default_symbols: tuple[str, ...],
    *,
    accept_new_options: bool = False,
    help_suffix: str = "",
) -> list[str]:
    """A single instant, client-side-filtered dropdown over a preloaded universe.

    Typing filters the already-loaded option list in the browser (like a
    search-engine dropdown) — no server round trip per character. When
    ``accept_new_options`` is set, a symbol absent from the preloaded list
    (Yahoo's bundled universe is large but not exhaustive — Yahoo has no
    downloadable "every symbol" list to preload the way Binance does) can
    still be typed and added directly. ``help_suffix`` lets a caller add
    context-specific guidance (e.g. the main sidebar's pairs_trading note)
    without this function assuming any particular caller's workflow.
    """
    help_text = "Tradable universe — start typing to filter"
    help_text += (
        ", or enter an exact symbol not in the list. " if accept_new_options else ". "
    )
    help_text += help_suffix
    if accept_new_options:
        help_text += " " + INCOMPLETE_LIST_NOTE

    if key not in st.session_state:
        st.session_state[key] = [
            label_by_symbol[s] for s in default_symbols if s in label_by_symbol
        ]
    # No `label_visibility="collapsed"` here: Streamlit hides the help
    # tooltip icon along with a collapsed label, and that icon is the only
    # place the market-calendar/incomplete-list notes above are surfaced.
    with st.container(border=True):
        picked_labels = st.multiselect(
            "Symbols",
            options=list(label_by_symbol.values()),
            key=key,
            placeholder=(
                "Type to find or add any symbol…"
                if accept_new_options
                else "Type to find a symbol…"
            ),
            accept_new_options=accept_new_options,
            help=help_text,
        )
    symbol_by_label = {label: symbol for symbol, label in label_by_symbol.items()}
    return [
        symbol_by_label.get(str(label), str(label).strip().upper())
        for label in picked_labels
    ]


def render_metric_cards(st: Any, result: BacktestResult) -> None:
    """Render the headline metric cards."""
    m = result.metrics

    def formatted_metric(key: str, spec: str) -> str:
        value = m.get(key)
        if value is None or not np.isfinite(value):
            return "n/a"
        return format(float(value), spec)

    total_costs = result.total_costs()
    formatted_costs = f"{total_costs:,.0f}" if np.isfinite(total_costs) else "n/a"
    cards = [
        ("Total return", formatted_metric("total_return", ".2%"), None),
        ("CAGR", formatted_metric("cagr", ".2%"), None),
        ("Sharpe", formatted_metric("sharpe_ratio", ".2f"), None),
        ("Sortino", formatted_metric("sortino_ratio", ".2f"), None),
        ("Max drawdown", formatted_metric("max_drawdown", ".2%"), None),
        ("Volatility", formatted_metric("annualized_volatility", ".2%"), None),
        (
            "Total costs",
            formatted_costs,
            "Cumulative modelled transaction costs, expressed in the same "
            "currency units as initial capital.",
        ),
        (
            "Number of fills",
            f"{result.number_of_trades()}",
            "Trade-log rows, one per symbol per executed order -- a "
            "declared multi-symbol position (e.g. a pairs_trading hedge) "
            "that trades both legs contributes one row per leg.",
        ),
    ]
    cols = st.columns(4)
    for i, (label, value, help_text) in enumerate(cards):
        cols[i % 4].metric(label, value, help=help_text)


def render_gross_net_comparison(st: Any, result: BacktestResult) -> None:
    """Render gross-vs-net performance and cost drag."""
    comparison = result.gross_net_comparison()

    def fmt_pct(key: str) -> str:
        value = comparison.get(key)
        return f"{value:.2%}" if value is not None and np.isfinite(value) else "n/a"

    def fmt_num(key: str) -> str:
        value = comparison.get(key)
        return f"{value:.2f}" if value is not None and np.isfinite(value) else "n/a"

    st.markdown("#### Gross vs Net")
    cards = [
        ("Net total return", fmt_pct("net_total_return"), None),
        ("Gross total return", fmt_pct("gross_total_return"), None),
        (
            "Cost drag",
            fmt_pct("cost_drag"),
            "Gross minus net total return — the cumulative performance "
            "given up to trading costs.",
        ),
        ("Net Sharpe", fmt_num("net_sharpe"), None),
        ("Gross Sharpe", fmt_num("gross_sharpe"), None),
    ]
    cols = st.columns(5)
    for i, (label, value, help_text) in enumerate(cards):
        cols[i].metric(label, value, help=help_text)


def render_sensitivity_heatmap(
    st: Any,
    sensitivity: pd.DataFrame,
    parameter_x: str,
    parameter_y: str,
    metric: str = "sharpe",
) -> None:
    """Render a parameter-sensitivity sweep as a Plotly heatmap.

    Shared by Backtest and Walk-forward mode: both feed this the same shape
    of ``sensitivity`` DataFrame (from ``run_parameter_sensitivity`` or its
    walk-forward-aware variant), so only the data source differs upstream.
    """
    import plotly.graph_objects as go

    from quantlab.validation.parameter_sensitivity import sensitivity_heatmap_data

    failed = int((sensitivity["status"] == "failed").sum()) if len(sensitivity) else 0
    if failed:
        st.caption(
            f"{failed} of {len(sensitivity)} combination(s) failed and are "
            "excluded from the heatmap below."
        )
    heatmap = sensitivity_heatmap_data(sensitivity, parameter_x, parameter_y, metric)
    fig = go.Figure(
        go.Heatmap(
            z=heatmap.to_numpy(),
            x=[str(v) for v in heatmap.columns],
            y=[str(v) for v in heatmap.index],
            colorscale="RdYlGn",
            colorbar={"title": metric},
        )
    )
    fig.update_layout(
        title=f"Sensitivity: {metric} by {parameter_x} / {parameter_y}",
        xaxis_title=parameter_x,
        yaxis_title=parameter_y,
        height=380,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sensitivity, width="stretch", hide_index=True)


def _line(x: Any, y: Any, name: str, color: str, dash: str | None = None) -> Any:
    import plotly.graph_objects as go

    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"color": color, "width": 1.6, "dash": dash},
    )


def _monthly_return_pivot(returns: pd.Series) -> pd.DataFrame:
    """Return a year-by-month matrix without inventing returns for empty months."""
    monthly = (1.0 + returns).resample("ME").prod(min_count=1) - 1.0
    monthly_index = pd.DatetimeIndex(monthly.index)
    frame = pd.DataFrame(
        {
            "year": monthly_index.year,
            "month": monthly_index.month,
            "ret": monthly.to_numpy(dtype=float),
        }
    )
    return frame.pivot(index="year", columns="month", values="ret")


def render_charts(st: Any, result: BacktestResult) -> None:
    """Render the dashboard chart grid."""
    import plotly.graph_objects as go

    ppy = result.config.periods_per_year
    equity = result.equity_curve
    rets = result.returns

    # Equity curve vs benchmark.
    fig = go.Figure()
    fig.add_trace(_line(equity.index, equity.to_numpy(), "Strategy", STRATEGY))
    if result.benchmark_returns is not None:
        bench_eq = float(equity.iloc[0]) * (1 + result.benchmark_returns).cumprod()
        fig.add_trace(
            _line(
                bench_eq.index,
                bench_eq.to_numpy(),
                benchmark_legend_label(result),
                BENCHMARK,
                "dash",
            )
        )
    fig.update_layout(title="Equity curve vs benchmark", height=380)
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    # Drawdown.
    dd = drawdown_series(equity)
    fig_dd = go.Figure(
        go.Scatter(
            x=dd.index, y=dd.to_numpy(), fill="tozeroy", line={"color": NEGATIVE}
        )
    )
    fig_dd.update_layout(title="Drawdown", height=300)
    col1.plotly_chart(fig_dd, width="stretch")

    # Monthly returns heatmap.
    pivot = _monthly_return_pivot(rets)
    fig_heat = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"tickformat": ".0%"},
        )
    )
    fig_heat.update_layout(title="Monthly returns", height=300)
    col2.plotly_chart(fig_heat, width="stretch")

    col3, col4 = st.columns(2)

    # Turnover.
    if result.turnover is not None:
        fig_to = go.Figure(
            go.Scatter(
                x=result.turnover.index,
                y=result.turnover.to_numpy(),
                line={"color": ACCENT},
            )
        )
        fig_to.update_layout(title="Turnover", height=300)
        col3.plotly_chart(fig_to, width="stretch")

    # Same adaptive window and formula as the HTML report's rolling charts.
    window = adaptive_rolling_window(len(rets))
    roll_sharpe = M.rolling_sharpe_ratio(
        rets, window, result.config.risk_free_rate, ppy
    )
    fig_rs = go.Figure(
        go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.to_numpy(), line={"color": STRATEGY}
        )
    )
    fig_rs.update_layout(title=f"Rolling Sharpe ({window}p)", height=300)
    col4.plotly_chart(fig_rs, width="stretch")

    col5, col6 = st.columns(2)

    # Rolling volatility.
    roll_vol = rets.rolling(window).std(ddof=1) * np.sqrt(ppy)
    fig_rv = go.Figure(
        go.Scatter(x=roll_vol.index, y=roll_vol.to_numpy(), line={"color": BENCHMARK})
    )
    fig_rv.update_layout(title=f"Rolling volatility ({window}p)", height=300)
    col5.plotly_chart(fig_rv, width="stretch")

    # Return distribution.
    fig_hist = go.Figure(
        go.Histogram(x=rets.dropna().to_numpy(), nbinsx=50, marker_color=STRATEGY)
    )
    fig_hist.update_layout(title="Return distribution", height=300)
    col6.plotly_chart(fig_hist, width="stretch")

    # Positions over time.
    fig_pos = go.Figure()
    for col in result.positions.columns:
        fig_pos.add_trace(
            go.Scatter(
                x=result.positions.index,
                y=result.positions[col].to_numpy(),
                name=col,
                mode="lines",
                stackgroup="one",
            )
        )
    fig_pos.update_layout(title="Positions over time", height=340)
    st.plotly_chart(fig_pos, width="stretch")


def render_exposure_and_cost_charts(st: Any, result: BacktestResult) -> None:
    """Render gross/net exposure and cumulative cost, shown after Gross vs Net."""
    import plotly.graph_objects as go

    col1, col2 = st.columns(2)

    gross = gross_exposure_series(result.positions)
    net = net_exposure_series(result.positions)
    fig_exp = go.Figure()
    fig_exp.add_trace(_line(gross.index, gross.to_numpy(), "Gross", STRATEGY))
    fig_exp.add_trace(_line(net.index, net.to_numpy(), "Net", ACCENT))
    fig_exp.update_layout(title="Gross / net exposure", height=300)
    col1.plotly_chart(fig_exp, width="stretch")

    if "total" in result.costs.columns:
        cum = result.costs["total"].cumsum()
        fig_cost = go.Figure(
            go.Scatter(x=cum.index, y=cum.to_numpy(), line={"color": NEGATIVE})
        )
        fig_cost.update_layout(title="Cumulative cost (fraction)", height=300)
        col2.plotly_chart(fig_cost, width="stretch")


def render_pair_diagnostics(
    st: Any,
    diagnostics: PairDiagnostics,
    *,
    entry_threshold: float,
    exit_threshold: float,
    stop_threshold: float | None,
    adf_pvalue_threshold: float | None,
) -> None:
    """Render a pairs-trading result's correlation/hedge-ratio/spread diagnostics.

    Shown only for a pairs_trading backtest -- the pairs_trading Strategy
    Explorer profile's own ``results_diagnostics`` is the single place
    that decides whether this section is computed at all (see
    ``quantlab.dashboard.explorer.profile.ResultsDiagnostics``); nothing
    here or in its caller checks the strategy's name.

    The centered-indicator chart mirrors the interactive lab's own
    "Trading thresholds" section exactly (same overlay helper, same viable-
    entry gating rule: threshold crossed AND, if configured, the causal
    rolling ADF p-value at that bar is <= ``adf_pvalue_threshold`` --
    :meth:`~quantlab.strategies.pairs_trading.PairsTradingStrategy.
    _stationarity_gate`'s own condition, using the SAME rolling ADF series
    already shown above, never a second, potentially diverging computation).
    """
    from quantlab.dashboard.explorer.shared_components import (
        VIABLE_ENTRY_MARKER_COLOR,
        centered_indicator_threshold_overlay,
        render_price_chart,
        render_stationarity_card,
    )

    st.subheader("Pair relationship diagnostics")
    st.caption(
        f"{diagnostics.symbol_a} / {diagnostics.symbol_b} -- return "
        f"correlation {diagnostics.correlation:.2f}."
    )
    render_price_chart(
        st,
        {"Hedge ratio (beta)": diagnostics.hedge_ratio},
        title="Rolling hedge ratio",
        yaxis_title="Beta",
    )
    render_price_chart(
        st,
        {"Spread": diagnostics.spread},
        title=f"{diagnostics.symbol_a}/{diagnostics.symbol_b} spread",
        yaxis_title="Spread",
    )

    indicator = diagnostics.spread_indicator
    threshold_series, line_colors = centered_indicator_threshold_overlay(
        indicator,
        f"{diagnostics.indicator} indicator",
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
    )
    crosses_entry = (indicator > entry_threshold) | (indicator < -entry_threshold)
    if adf_pvalue_threshold is not None:
        gate_open = (
            diagnostics.rolling_adf_pvalue.reindex(indicator.index)
            <= adf_pvalue_threshold
        ).fillna(False)
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
        title=f"Centered '{diagnostics.indicator}' indicator of the spread "
        "with entry/exit/stop thresholds",
        yaxis_title="Centered indicator",
        markers=markers,
        colors={**line_colors, **marker_colors},
    )
    gate_text = (
        f"ADF gate at p <= {adf_pvalue_threshold:g}"
        if adf_pvalue_threshold is not None
        else "ADF gate disabled"
    )
    n_viable = int(viable.sum())
    if stop_threshold is not None:
        stop_breaches = int(
            ((indicator > stop_threshold) | (indicator < -stop_threshold)).sum()
        )
        stop_clause = (
            f" **{stop_breaches}** bar(s) cross the stop threshold "
            f"({stop_threshold:g})."
        )
    else:
        stop_clause = " Stop threshold is disabled."
    st.caption(
        f"**{int(crosses_entry.sum())}** bar(s) cross the entry threshold; "
        f"**{n_viable}** of those are viable ({gate_text})."
        + stop_clause
        + " Not an entry count: current state, rebalancing and execution can "
        "still prevent or delay a trade."
    )

    render_stationarity_card(st, diagnostics.adf_result, label="ADF (spread)")
    render_stationarity_card(
        st, diagnostics.cointegration_result, label="Engle-Granger cointegration"
    )
    if diagnostics.rolling_adf_pvalue.notna().any():
        render_price_chart(
            st,
            {"Rolling ADF p-value": diagnostics.rolling_adf_pvalue},
            title="Stationarity stability over time",
            yaxis_title="p-value",
        )
    half_life_text = (
        f"{diagnostics.half_life:.1f} periods"
        if np.isfinite(diagnostics.half_life)
        else "not mean-reverting over this sample (no finite half-life)"
    )
    hedge_stability_text = (
        f"{diagnostics.hedge_ratio_stability:.4f}"
        if np.isfinite(diagnostics.hedge_ratio_stability)
        else "n/a"
    )
    col_hl, col_stab = st.columns(2)
    col_hl.metric("Half-life", half_life_text)
    col_stab.metric("Hedge-ratio stability (std of beta)", hedge_stability_text)
    st.caption(
        "A larger hedge-ratio std means the fitted slope changes more in "
        "this sample. It is measured in beta's scale-dependent units, so "
        "compare it across settings for the same ordered pair, not across pairs."
    )


#: Fixed display schema for `render_trade_table` -- the SAME 15 columns, in
#: this exact order, for every strategy. Only the values differ; a column is
#: never added, dropped or reordered based on which strategy produced the
#: result (`test_dashboard.py` asserts this across several strategies).
_TRADE_TABLE_DISPLAY_COLUMNS = [
    "Timestamp",
    "Symbol",
    "Action",
    "Previous weight",
    "New weight",
    "Weight change",
    "Trigger",
    "Trigger detail",
    "Adjustments",
    "Position origin",
    "Position origin date",
    "Details",
    "Reference price",
    "Traded notional",
    "Total cost",
]


def _is_missing(value: object) -> bool:
    """True for None/NaN/NaT/`pd.NA`, without raising on a non-scalar value.

    A plain ``isinstance(value, float) and pd.isna(value)`` check misses
    ``pd.NA`` (not a ``float`` subclass) and
    non-float-64 NaN scalars like ``numpy.float32('nan')``/
    ``numpy.float16('nan')`` (also not ``float`` subclasses -- only
    ``numpy.float64`` is, via CPython's numpy integration) -- either would
    then reach ``str(value)`` -> ``parse_adjustment_codes()`` as a bogus
    ``"<NA>"``/``"nan"`` code and raise ``BacktestError`` on an unknown
    adjustment code, instead of being treated as simply missing.
    """
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(cast(Any, value)))


def _format_adjustments(value: object) -> object:
    """Cosmetically re-space the "+"-joined machine codes for readability.

    Still the exact same codes, never a fusion of different columns'
    data. `None` (no adjustment layer acted on this trade) stays `None`,
    never a blanked-away real value.
    """
    if _is_missing(value):
        return None
    return " + ".join(parse_adjustment_codes(str(value)))


def _compose_details(trigger_details: object, adjustment_details: object) -> object:
    """Human-readable summary combining the trigger's and adjustment(s)' text.

    E.g. "Trigger: oversold entry; Adjustment: maximum_weight 0.62 ->
    0.50". Purely a reading aid -- the raw, machine-readable columns
    this is built from are untouched in `result.trades`/the CSV export
    below.
    """
    parts = []
    if not _is_missing(trigger_details):
        parts.append(f"Trigger: {trigger_details}")
    if not _is_missing(adjustment_details):
        parts.append(f"Adjustment: {adjustment_details}")
    return "; ".join(parts) if parts else None


def render_trade_table(st: Any, result: BacktestResult) -> None:
    """Render a uniform trade table with a CSV download.

    The same 15 display columns (`_TRADE_TABLE_DISPLAY_COLUMNS`), in the
    same order, are shown for every strategy -- only the VALUES differ.
    Missing information (no adjustment on this trade, no reason
    attribution available at all for this run, e.g. a walk-forward
    out-of-sample result) renders as a blank cell; the column itself
    always stays, it is never conditionally dropped.

    `Position origin`/`Position origin date` are always shown, on every
    row, even when they duplicate `Trigger` -- blanking them
    conditionally would make an empty cell ambiguous between "no origin"
    and "hidden because redundant", exactly the ambiguity this design
    avoids: blank means either that no strategic position is currently
    active (the decision proxy is flat) or that position-origin
    attribution was unavailable for this result -- notably for a stitched
    walk-forward out-of-sample result, which does not carry it -- never a
    value hidden for brevity.

    This is a read-only presentation view derived from `result.trades`.
    The CSV download button below always exports the raw, full
    21-column `result.trades` frame untouched, regardless of what is
    shown here -- the visible table's schema never drives the export's.
    """
    trades = result.trades
    if trades.empty:
        st.info("No trades were recorded for this configuration.")
        return
    from quantlab.backtesting.trade_log import stop_loss_take_profit_trigger_counts
    from quantlab.dashboard.explorer.shared_components import strong

    trigger_counts = stop_loss_take_profit_trigger_counts(trades)
    strategy_params = result.config.strategy_parameters
    stop_loss_enabled = strategy_params.get("stop_loss_pct") is not None
    take_profit_enabled = strategy_params.get("take_profit_pct") is not None
    # Gated on whether each was actually CONFIGURED, not on whether it
    # fired -- a configured-but-never-triggered stop/target (count 0) is a
    # meaningful fact worth showing, and independent of the other: enabling
    # only stop-loss must never imply take-profit was also active at 0.
    if stop_loss_enabled or take_profit_enabled:
        parts = []
        if stop_loss_enabled:
            parts.append(
                f"Stop-loss affected {strong(str(trigger_counts['stop_loss']))} "
                "symbol-position exit(s)"
            )
        if take_profit_enabled:
            label = "take-profit" if parts else "Take-profit"
            parts.append(
                f"{label} affected {strong(str(trigger_counts['take_profit']))} "
                "symbol-position exit(s)"
            )
        st.caption(
            "; ".join(parts) + " -- counted per trade-log row, so a "
            "declared multi-symbol position (e.g. a pairs_trading hedge) "
            "that force-flattens contributes one row per leg, not "
            "necessarily one distinct stop-loss/take-profit EVENT.",
            unsafe_allow_html=True,
        )
    details = [
        _compose_details(trigger_detail, adjustment_detail)
        for trigger_detail, adjustment_detail in zip(
            trades["trigger_reason_details"],
            trades["adjustment_reason_details"],
            strict=True,
        )
    ]
    display_trades = pd.DataFrame(
        {
            "Timestamp": trades["timestamp"],
            "Symbol": trades["symbol"],
            "Action": trades["action"],
            "Previous weight": trades["previous_weight"],
            "New weight": trades["new_weight"],
            "Weight change": trades["weight_change"],
            "Trigger": trades["trigger_reason_code"],
            "Trigger detail": trades["trigger_reason_detail_code"],
            "Adjustments": trades["adjustment_reason_codes"].map(_format_adjustments),
            "Position origin": trades["position_strategy_origin_code"],
            "Position origin date": trades["position_strategy_origin_timestamp"],
            "Details": details,
            "Reference price": trades["reference_price"],
            "Traded notional": trades["traded_notional"],
            "Total cost": trades["total_cost"],
        }
    )
    assert list(display_trades.columns) == _TRADE_TABLE_DISPLAY_COLUMNS
    st.dataframe(
        display_trades,
        width="stretch",
        height=320,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp", format="YYYY-MM-DD HH:mm"
            ),
            "Previous weight": st.column_config.NumberColumn(
                "Previous weight", format="percent"
            ),
            "New weight": st.column_config.NumberColumn("New weight", format="percent"),
            "Weight change": st.column_config.NumberColumn(
                "Weight change", format="percent"
            ),
            "Trigger": st.column_config.TextColumn(
                "Trigger",
                help="The single most-upstream event that initiated the "
                "target change. Blank when reason attribution wasn't "
                "available for this run (e.g. a walk-forward "
                "out-of-sample result) or when nothing upstream changed.",
            ),
            "Trigger detail": st.column_config.TextColumn(
                "Trigger detail",
                help="The strategy's own precise sub-cause for the "
                "trigger, when identifiable. Blank is not a bug -- "
                "portfolio_rebalance/volatility_target_adjustment "
                "legitimately have no sub-cause.",
            ),
            "Adjustments": st.column_config.TextColumn(
                "Adjustments",
                help="Every downstream layer that modified, delayed, "
                "redistributed, constrained or forced the executed size "
                "-- can combine several causes on the same trade (e.g. "
                "a constraint AND turnover_cap together). Blank means no "
                "adjustment layer acted on this trade.",
            ),
            "Position origin": st.column_config.TextColumn(
                "Position origin",
                help="Origin of the currently active strategic regime "
                "for this symbol/leg -- not necessarily the origin of "
                "the currently executed weight, nor an execution "
                "timestamp. Always shown, even when it duplicates "
                "Trigger. Blank means no strategic position is "
                "currently active (flat), never a value hidden for "
                "brevity.",
            ),
            "Position origin date": st.column_config.DatetimeColumn(
                "Position origin date",
                format="YYYY-MM-DD HH:mm",
                help="Date of the strategic transition that created the "
                "currently active regime -- can be much earlier than "
                "this trade's own date.",
            ),
            "Details": st.column_config.TextColumn(
                "Details",
                help="Trigger/adjustment free text combined for quick "
                "reading. The machine-readable columns this is built "
                "from remain intact in the CSV export below.",
            ),
            "Reference price": st.column_config.NumberColumn(
                "Reference price",
                format="%.2f",
                help="The prior-period price used by the backtest to "
                "value this trade and compute its notional/costs -- not "
                "a real fill price from an order book.",
            ),
            "Traded notional": st.column_config.NumberColumn(
                "Traded notional (currency units)", format="localized"
            ),
            "Total cost": st.column_config.NumberColumn(
                "Total cost (currency units)", format="localized"
            ),
        },
    )
    st.download_button(
        "Download trades (CSV)",
        trades.to_csv(index=False).encode("utf-8"),
        file_name=f"{result.config.experiment_name}_trades.csv",
        mime="text/csv",
    )
