"""Reusable Streamlit UI components.

Streamlit and Plotly are imported lazily so importing this module (e.g. in
tests) does not require the dashboard extra to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
        ("Number of trades", f"{result.number_of_trades()}", None),
    ]
    cols = st.columns(4)
    for i, (label, value, help_text) in enumerate(cards):
        cols[i % 4].metric(label, value, help=help_text)


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

    # Exposure.
    gross = gross_exposure_series(result.positions)
    net = net_exposure_series(result.positions)
    fig_exp = go.Figure()
    fig_exp.add_trace(_line(gross.index, gross.to_numpy(), "Gross", STRATEGY))
    fig_exp.add_trace(_line(net.index, net.to_numpy(), "Net", ACCENT))
    fig_exp.update_layout(title="Gross / net exposure", height=300)
    col3.plotly_chart(fig_exp, width="stretch")

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
        col4.plotly_chart(fig_to, width="stretch")

    col5, col6 = st.columns(2)

    # Cumulative costs.
    if "total" in result.costs.columns:
        cum = result.costs["total"].cumsum()
        fig_cost = go.Figure(
            go.Scatter(x=cum.index, y=cum.to_numpy(), line={"color": NEGATIVE})
        )
        fig_cost.update_layout(title="Cumulative cost (fraction)", height=300)
        col5.plotly_chart(fig_cost, width="stretch")

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
    col6.plotly_chart(fig_rs, width="stretch")

    col7, col8 = st.columns(2)

    # Rolling volatility.
    roll_vol = rets.rolling(window).std(ddof=1) * np.sqrt(ppy)
    fig_rv = go.Figure(
        go.Scatter(x=roll_vol.index, y=roll_vol.to_numpy(), line={"color": BENCHMARK})
    )
    fig_rv.update_layout(title=f"Rolling volatility ({window}p)", height=300)
    col7.plotly_chart(fig_rv, width="stretch")

    # Return distribution.
    fig_hist = go.Figure(
        go.Histogram(x=rets.dropna().to_numpy(), nbinsx=50, marker_color=STRATEGY)
    )
    fig_hist.update_layout(title="Return distribution", height=300)
    col8.plotly_chart(fig_hist, width="stretch")

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


def render_trade_table(st: Any, result: BacktestResult) -> None:
    """Render the trade table with a CSV download."""
    trades = result.trades
    if trades.empty:
        st.info("No trades were recorded for this configuration.")
        return
    display_cols = [
        "timestamp",
        "symbol",
        "side",
        "weight_change",
        "traded_notional",
        "total_cost",
    ]
    st.dataframe(
        trades[display_cols],
        width="stretch",
        height=320,
        hide_index=True,
        column_config={
            "weight_change": st.column_config.NumberColumn(
                "Weight change", format="percent"
            ),
            "traded_notional": st.column_config.NumberColumn(
                "Traded notional (currency units)", format="localized"
            ),
            "total_cost": st.column_config.NumberColumn(
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
