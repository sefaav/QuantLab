"""Static report charts and interactive dashboard figures."""

from __future__ import annotations

import base64
import calendar
import io
import os
import tempfile
from collections.abc import Callable, Mapping
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from quantlab.logging_config import get_logger
from quantlab.risk import metrics as M
from quantlab.risk.drawdown import drawdown_series
from quantlab.risk.exposure import gross_exposure_series, net_exposure_series

if TYPE_CHECKING:
    from quantlab.backtesting.result import BacktestResult
    from quantlab.features.pairs_diagnostics import PairDiagnostics

logger = get_logger(__name__)

STRATEGY = "#2563eb"
BENCHMARK = "#6b7280"
POSITIVE = "#16a34a"
NEGATIVE = "#dc2626"
ACCENT = "#f59e0b"
_GRID = "#e5e7eb"
_PNG_DATA_URI_PREFIX = "data:image/png;base64,"


def benchmark_legend_label(result: BacktestResult) -> str:
    """Return a legend label that identifies the configured benchmark."""
    label = result.config.benchmark_label
    return f"Benchmark ({label})" if label else "Benchmark"


def _style_axes(ax: Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Create a backend-local Agg figure without changing global Matplotlib state."""
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    return figure, axes


def fig_to_base64(fig: Figure) -> str:
    """Encode and close a figure as a self-contained PNG data URI."""
    try:
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        fig.clear()
    return f"{_PNG_DATA_URI_PREFIX}{encoded}"


def equity_curve_chart(result: BacktestResult) -> Figure:
    """Plot strategy and benchmark equity on the same starting capital."""
    fig, ax = _new_figure((9, 4))
    equity = result.equity_curve
    ax.plot(equity.index, equity.to_numpy(), color=STRATEGY, lw=1.6, label="Strategy")
    if result.benchmark_returns is not None:
        initial = float(equity.iloc[0])
        bench_equity = initial * (1.0 + result.benchmark_returns).cumprod()
        ax.plot(
            bench_equity.index,
            bench_equity.to_numpy(),
            color=BENCHMARK,
            lw=1.3,
            ls="--",
            label=benchmark_legend_label(result),
        )
    _style_axes(ax, "Equity curve", "Portfolio value")
    ax.legend(frameon=False, fontsize=8)
    return fig


def drawdown_chart(result: BacktestResult) -> Figure:
    """Plot the strategy's decline from its running equity peak."""
    drawdown = drawdown_series(result.equity_curve)
    fig, ax = _new_figure((9, 2.6))
    ax.fill_between(
        drawdown.index, drawdown.to_numpy(), 0.0, color=NEGATIVE, alpha=0.35
    )
    ax.plot(drawdown.index, drawdown.to_numpy(), color=NEGATIVE, lw=1.0)
    _style_axes(ax, "Drawdown", "Drawdown")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return fig


def monthly_returns_heatmap(result: BacktestResult) -> Figure:
    """Plot monthly returns while preserving months without observations."""
    monthly = (1.0 + result.returns).resample("ME").prod(min_count=1) - 1.0
    monthly_index = pd.DatetimeIndex(monthly.index)
    frame = pd.DataFrame(
        {
            "year": monthly_index.year,
            "month": monthly_index.month,
            "return": monthly.to_numpy(dtype=float),
        }
    )
    pivot = frame.pivot(index="year", columns="month", values="return")
    fig, ax = _new_figure((9, max(2.2, 0.4 * len(pivot) + 1)))
    if pivot.empty or len(pivot.columns) == 0:
        ax.text(
            0.5,
            0.5,
            "No monthly returns available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BENCHMARK,
        )
        ax.set_title("Monthly returns", fontsize=11, fontweight="bold")
        return fig

    values = pivot.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = max(float(finite.max()), 0.01) if finite.size else 0.01
    cmap = colormaps["BrBG"].with_extremes(bad="#e5e7eb")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([calendar.month_abbr[int(month)] for month in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly returns", fontsize=11, fontweight="bold")
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02,
        format=lambda value, _: f"{value:.0%}",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.1%}" if np.isfinite(value) else "n/a"
            colour = (
                "white"
                if np.isfinite(value) and abs(value) > vmax * 0.55
                else "#111827"
            )
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=colour,
            )
    return fig


def sensitivity_heatmap_chart(
    sensitivity: pd.DataFrame, metric: str = "sharpe"
) -> Figure:
    """Plot a 2-parameter sensitivity sweep as a metric heatmap.

    Infers which two columns are the swept parameters directly from
    `sensitivity` itself (see `infer_sensitivity_parameter_columns`),
    mirroring the dashboard's interactive Plotly heatmap
    (`components.render_sensitivity_heatmap`) in a static form for the HTML
    report.
    """
    from quantlab.validation.parameter_sensitivity import (
        infer_sensitivity_parameter_columns,
        sensitivity_heatmap_data,
    )

    parameter_x, parameter_y = infer_sensitivity_parameter_columns(sensitivity)
    pivot = sensitivity_heatmap_data(sensitivity, parameter_x, parameter_y, metric)

    fig, ax = _new_figure((7, max(2.5, 0.5 * len(pivot.index) + 1)))
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    title = f"Sensitivity: {metric} by {parameter_x} / {parameter_y}"
    if finite.size == 0:
        ax.text(
            0.5,
            0.5,
            "No successful combinations to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BENCHMARK,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        return fig

    vmin, vmax = float(finite.min()), float(finite.max())
    span = vmax - vmin if vmax > vmin else 1.0
    cmap = colormaps["RdYlGn"].with_extremes(bad="#e5e7eb")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel(parameter_x, fontsize=9)
    ax.set_ylabel(parameter_y, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.2f}" if np.isfinite(value) else "n/a"
            # RdYlGn is light (yellow) near the middle of the range and
            # darker toward both ends (red/green), so contrast text there.
            normalized = (value - vmin) / span
            colour = (
                "white"
                if np.isfinite(value) and abs(normalized - 0.5) > 0.3
                else "#111827"
            )
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=colour,
            )
    return fig


def correlation_heatmap_chart(matrix: pd.DataFrame) -> Figure:
    """Plot a symbol x symbol correlation matrix as a static heatmap.

    Mirrors the dashboard's interactive Plotly heatmap
    (``dashboard.explorer.shared_components.render_correlation_matrix``)
    in a static form for the HTML report.
    """
    width = max(4.0, 0.6 * len(matrix.columns) + 2.0)
    height = max(3.0, 0.6 * len(matrix.index) + 1.0)
    fig, ax = _new_figure((width, height))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(
        values, cmap=colormaps["RdBu"], vmin=-1.0, vmax=1.0, aspect="auto"
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(r) for r in matrix.index])
    ax.set_title("Correlation matrix (of returns)", fontsize=11, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            colour = "white" if abs(value) > 0.6 else "#111827"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=colour,
            )
    return fig


def pair_spread_chart(diagnostics: PairDiagnostics) -> Figure:
    """Plot a pair's spread, indicator and rolling stationarity p-value.

    The three panels answer, respectively: what does the residual look
    like, how far is it currently from its own recent behaviour (per
    ``diagnostics.indicator`` -- zscore, rsi or percentile, whichever the
    pair was actually diagnosed with), and has the relationship stayed
    stationary throughout the sample rather than only when tested once
    over the whole history (see ``PairDiagnostics.rolling_adf_pvalue``).
    """
    fig = Figure(figsize=(9, 7.5))
    FigureCanvasAgg(fig)
    ax_spread, ax_indicator, ax_pvalue = fig.subplots(3, 1, sharex=True)

    spread = diagnostics.spread
    ax_spread.plot(spread.index, spread.to_numpy(), color=STRATEGY, lw=1.2)
    ax_spread.axhline(0.0, color=_GRID, lw=1.0)
    _style_axes(
        ax_spread, f"{diagnostics.symbol_a}/{diagnostics.symbol_b} spread", "Spread"
    )

    indicator = diagnostics.spread_indicator
    indicator_label = f"{diagnostics.indicator} indicator"
    ax_indicator.plot(indicator.index, indicator.to_numpy(), color=ACCENT, lw=1.2)
    ax_indicator.axhline(0.0, color=_GRID, lw=1.0)
    _style_axes(ax_indicator, f"Spread {indicator_label}", indicator_label)

    pvalue = diagnostics.rolling_adf_pvalue.dropna()
    if len(pvalue):
        ax_pvalue.plot(
            pvalue.index, pvalue.to_numpy(), color=NEGATIVE, marker="o", ms=3, lw=1.0
        )
    else:
        ax_pvalue.text(
            0.5,
            0.5,
            "Not enough history for a rolling stationarity check.",
            ha="center",
            va="center",
            transform=ax_pvalue.transAxes,
            color=BENCHMARK,
        )
    ax_pvalue.axhline(0.05, color=_GRID, lw=1.0, ls="--")
    _style_axes(ax_pvalue, "Rolling ADF p-value (stability over time)", "p-value")
    fig.tight_layout()
    return fig


def adaptive_rolling_window(n_observations: int) -> int:
    """Shrink the rolling Sharpe/volatility window for short samples.

    Shared by the HTML report and the dashboard, for both the rolling-Sharpe
    and rolling-volatility charts, so all of them show a populated chart on
    the same backtest instead of going blank under 126 periods.
    """
    return min(126, max(20, n_observations // 4))


def rolling_sharpe_chart(result: BacktestResult, window: int = 126) -> Figure:
    """Plot the configured-risk-free trailing Sharpe over ``window`` periods."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral):
        raise ValueError("window must be an integer greater than 1.")
    window = int(window)
    if window < 2:
        raise ValueError("window must be an integer greater than 1.")

    fig, ax = _new_figure((9, 2.6))
    if len(result.returns) < window:
        ax.text(
            0.5,
            0.5,
            f"Requires {window} observations; {len(result.returns)} available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BENCHMARK,
        )
        _style_axes(ax, f"Rolling Sharpe ({window}p)", "Sharpe")
        return fig

    sharpe = M.rolling_sharpe_ratio(
        result.returns,
        window,
        result.config.risk_free_rate,
        result.config.periods_per_year,
    )
    ax.plot(sharpe.index, sharpe.to_numpy(), color=STRATEGY, lw=1.2)
    ax.axhline(0.0, color=BENCHMARK, lw=0.8, ls="--")
    _style_axes(ax, f"Rolling Sharpe ({window}p)", "Sharpe")
    return fig


def rolling_volatility_chart(result: BacktestResult, window: int = 126) -> Figure:
    """Plot trailing annualised realised volatility over ``window`` periods."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral):
        raise ValueError("window must be an integer greater than 1.")
    window = int(window)
    if window < 2:
        raise ValueError("window must be an integer greater than 1.")

    fig, ax = _new_figure((9, 2.6))
    if len(result.returns) < window:
        ax.text(
            0.5,
            0.5,
            f"Requires {window} observations; {len(result.returns)} available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BENCHMARK,
        )
        _style_axes(ax, f"Rolling volatility ({window}p)", "Volatility")
        return fig

    volatility = result.returns.rolling(window).std(ddof=1) * np.sqrt(
        result.config.periods_per_year
    )
    ax.plot(volatility.index, volatility.to_numpy(), color=BENCHMARK, lw=1.2)
    _style_axes(ax, f"Rolling volatility ({window}p)", "Volatility")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return fig


def returns_distribution_chart(result: BacktestResult) -> Figure:
    """Plot the distribution of finite period returns."""
    returns = result.returns.replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = _new_figure((5, 3))
    if returns.empty:
        ax.text(
            0.5,
            0.5,
            "No finite returns available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BENCHMARK,
        )
    else:
        ax.hist(returns.to_numpy(), bins=50, color=STRATEGY, alpha=0.8)
        ax.axvline(0.0, color=NEGATIVE, lw=1.0)
    _style_axes(ax, "Return distribution", "Observations")
    return fig


def exposure_chart(result: BacktestResult) -> Figure:
    """Plot gross and net exposure over time."""
    gross = gross_exposure_series(result.positions)
    net = net_exposure_series(result.positions)
    fig, ax = _new_figure((9, 2.6))
    ax.plot(gross.index, gross.to_numpy(), color=STRATEGY, lw=1.1, label="Gross")
    ax.plot(net.index, net.to_numpy(), color=ACCENT, lw=1.1, label="Net")
    _style_axes(ax, "Exposure", "Exposure (x)")
    ax.legend(frameon=False, fontsize=8)
    return fig


def cumulative_costs_chart(result: BacktestResult) -> Figure:
    """Plot the running sum of per-period cost fractions."""
    cumulative = (
        result.costs["total"].cumsum()
        if "total" in result.costs.columns
        else pd.Series(dtype=float)
    )
    fig, ax = _new_figure((9, 2.6))
    if cumulative.empty:
        ax.text(
            0.5,
            0.5,
            "No cost series available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BENCHMARK,
        )
    else:
        ax.plot(cumulative.index, cumulative.to_numpy(), color=NEGATIVE, lw=1.2)
    _style_axes(ax, "Cumulative transaction-cost fraction", "Cost fraction")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    return fig


def _rolling_sharpe_chart_adaptive(result: BacktestResult) -> Figure:
    """Render the rolling-Sharpe chart with a sample-adapted window."""
    window = adaptive_rolling_window(len(result.returns))
    return rolling_sharpe_chart(result, window=window)


def _rolling_volatility_chart_adaptive(result: BacktestResult) -> Figure:
    """Render the rolling-volatility chart with a sample-adapted window."""
    window = adaptive_rolling_window(len(result.returns))
    return rolling_volatility_chart(result, window=window)


def _chart_builders() -> dict[str, Callable[[BacktestResult], Figure]]:
    """Return the current builders shared by HTML and disk reports."""
    return {
        "equity_curve": equity_curve_chart,
        "drawdown": drawdown_chart,
        "monthly_returns": monthly_returns_heatmap,
        "rolling_sharpe": _rolling_sharpe_chart_adaptive,
        "rolling_volatility": _rolling_volatility_chart_adaptive,
        "exposure": exposure_chart,
        "cumulative_costs": cumulative_costs_chart,
        "returns_distribution": returns_distribution_chart,
    }


def managed_report_figure_filenames() -> tuple[str, ...]:
    """Return the figure filenames that QuantLab owns in a saved bundle."""
    return tuple(f"{name}.png" for name in _chart_builders())


def report_figures(
    result: BacktestResult, warnings: list[str] | None = None
) -> dict[str, str]:
    """Render charts independently and collect any per-chart failures."""
    figures: dict[str, str] = {}
    for name, builder in _chart_builders().items():
        try:
            figures[name] = fig_to_base64(builder(result))
        except Exception as exc:
            message = f"Could not render '{name}' chart for report: {exc}"
            logger.warning(message, exc_info=True)
            if warnings is not None:
                warnings.append(message)
    return figures


def _png_bytes(data_uri: str) -> bytes:
    if not data_uri.startswith(_PNG_DATA_URI_PREFIX):
        raise ValueError("Expected a base64 PNG data URI.")
    return base64.b64decode(
        data_uri[len(_PNG_DATA_URI_PREFIX) :],
        validate=True,
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Atomically replace one binary file in its destination directory."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_figures(
    result: BacktestResult,
    directory: str | Path,
    warnings: list[str] | None = None,
    *,
    rendered: Mapping[str, str] | None = None,
) -> list[Path]:
    """Save available charts and continue after individual write failures."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    figures = (
        dict(rendered) if rendered is not None else report_figures(result, warnings)
    )
    paths: list[Path] = []
    for name, data_uri in figures.items():
        path = destination / f"{name}.png"
        try:
            _write_bytes_atomic(path, _png_bytes(data_uri))
            paths.append(path)
        except Exception as exc:
            message = f"Could not save '{name}' chart to {path}: {exc}"
            logger.warning(message, exc_info=True)
            if warnings is not None:
                warnings.append(message)
    return paths


def equity_and_drawdown_figure(result: BacktestResult) -> Any:
    """Return an interactive Plotly equity and drawdown figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    equity = result.equity_curve
    drawdown = drawdown_series(equity)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("Equity curve", "Drawdown"),
    )
    figure.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.to_numpy(),
            name="Strategy",
            line={"color": STRATEGY, "width": 2},
        ),
        row=1,
        col=1,
    )
    if result.benchmark_returns is not None:
        benchmark_equity = (
            float(equity.iloc[0]) * (1.0 + result.benchmark_returns).cumprod()
        )
        figure.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity.to_numpy(),
                name=benchmark_legend_label(result),
                line={"color": BENCHMARK, "width": 1.5, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.to_numpy(),
            name="Drawdown",
            fill="tozeroy",
            line={"color": NEGATIVE, "width": 1},
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
    )
    return figure
