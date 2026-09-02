"""Summary and sub-period tables."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantlab.reporting.research_summary import out_of_sample_scope
from quantlab.risk.drawdown import max_drawdown
from quantlab.risk.metrics import (
    annualized_volatility,
    cagr,
    equity_from_returns,
    sharpe_ratio,
    total_return,
)

if TYPE_CHECKING:
    from quantlab.backtesting.result import BacktestResult
    from quantlab.features.pairs_diagnostics import PairDiagnostics


_METRIC_FORMAT = {
    "total_return": ("Total return", "pct"),
    "cagr": ("CAGR", "pct"),
    "annualized_volatility": ("Volatility (ann.)", "pct"),
    "sharpe_ratio": ("Sharpe", "num"),
    "sortino_ratio": ("Sortino", "num"),
    "calmar_ratio": ("Calmar", "num"),
    "max_drawdown": ("Max drawdown", "pct"),
    "average_drawdown": ("Average drawdown", "pct"),
    "hit_rate": ("Hit rate (non-zero periods)", "pct"),
    "var_95": ("VaR 95%", "pct"),
    "cvar_95": ("CVaR 95%", "pct"),
    "skewness": ("Skewness", "num"),
    "kurtosis": ("Kurtosis", "num"),
    "annual_turnover": ("Annual turnover (x/year)", "num"),
    "average_gross_exposure": ("Avg gross exposure (x)", "num"),
    "number_of_trades": ("Number of fills", "int"),
    "beta": ("Beta", "num"),
    "alpha": ("Alpha (ann.)", "pct"),
    "information_ratio": ("Information ratio", "num"),
    "tracking_error": ("Tracking error", "pct"),
}


def _fmt(value: object, kind: str) -> str:
    """Format a finite numeric value for a human-facing table."""
    if (
        value is None
        or isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        return "n/a"
    number = float(value)
    if not np.isfinite(number):
        return "n/a"
    if kind == "pct":
        return f"{number:.2%}"
    if kind == "currency":
        return f"{number:,.2f}"
    if kind == "int":
        return f"{int(number)}"
    return f"{number:.2f}"


def metrics_table(result: BacktestResult) -> pd.DataFrame:
    """Return a two-column (Metric, Value) table of headline metrics."""
    rows = []
    for key, (label, kind) in _METRIC_FORMAT.items():
        if key in result.metrics:
            rows.append({"Metric": label, "Value": _fmt(result.metrics[key], kind)})
    return pd.DataFrame(rows)


def gross_net_table(result: BacktestResult) -> pd.DataFrame:
    """Gross vs net comparison."""
    comp = result.gross_net_comparison()
    rows = [
        ("Gross total return", _fmt(comp.get("gross_total_return"), "pct")),
        ("Net total return", _fmt(comp.get("net_total_return"), "pct")),
        ("Cost drag", _fmt(comp.get("cost_drag"), "pct")),
        ("Total cost (currency units)", _fmt(comp.get("total_cost"), "currency")),
        ("Gross Sharpe", _fmt(comp.get("gross_sharpe"), "num")),
        ("Net Sharpe", _fmt(comp.get("net_sharpe"), "num")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _subperiod_row(
    label: str, returns: pd.Series, ppy: int, risk_free_rate: float = 0.0
) -> dict[str, object]:
    equity = equity_from_returns(returns)
    return {
        "Period": label,
        "Return": total_return(equity),
        "CAGR": cagr(equity, ppy),
        "Sharpe": sharpe_ratio(returns, risk_free_rate, ppy),
        "Max Drawdown": max_drawdown(equity),
        "Volatility": annualized_volatility(returns, ppy),
    }


def yearly_returns_table(result: BacktestResult) -> pd.DataFrame:
    """Per-year performance breakdown."""
    ppy = result.config.periods_per_year
    rf = result.config.backtest.risk_free_rate
    rets = result.returns
    rows = [
        _subperiod_row(str(year), grp, ppy, rf)
        for year, grp in rets.groupby(pd.DatetimeIndex(rets.index).year)
    ]
    return pd.DataFrame(rows)


def subperiod_table(result: BacktestResult) -> pd.DataFrame:
    """Aggregate and yearly performance, turnover and trade counts.

    The aggregate row is labelled "Out-of-sample" instead of "Full sample"
    when ``result.metrics`` are themselves a walk-forward OOS series (see
    ``research_summary.out_of_sample_scope``) — otherwise it would claim
    the opposite of what that series actually is. ``Turnover (x)`` is the
    cumulative L1 turnover multiple within each row's period; it is not
    annualised for the aggregate row.
    """
    ppy = result.config.periods_per_year
    rf = result.config.backtest.risk_free_rate
    rets = result.returns
    turnover = (
        result.turnover if result.turnover is not None else pd.Series(dtype=float)
    )

    aggregate_label = (
        "Out-of-sample" if out_of_sample_scope(result) is not None else "Full sample"
    )
    rows = [_subperiod_row(aggregate_label, rets, ppy, rf)]
    for year, grp in rets.groupby(pd.DatetimeIndex(rets.index).year):
        rows.append(_subperiod_row(str(year), grp, ppy, rf))

    table = pd.DataFrame(rows)
    turnovers = []
    trade_counts = []
    trades = result.trades
    for label in table["Period"]:
        if label == aggregate_label:
            mask = pd.Series(True, index=rets.index)
        else:
            year_match = pd.DatetimeIndex(rets.index).year == int(label)
            mask = pd.Series(year_match, index=rets.index)
        turnovers.append(float(turnover[mask].sum()) if len(turnover) else np.nan)
        if len(trades) and "timestamp" in trades.columns:
            ts = pd.to_datetime(trades["timestamp"])
            if label == aggregate_label:
                trade_counts.append(len(trades))
            else:
                trade_counts.append(int((ts.dt.year == int(label)).sum()))
        else:
            trade_counts.append(0)
    table["Turnover (x)"] = turnovers
    table["Number of Trades"] = trade_counts
    return table


def regime_table(
    result: BacktestResult, *, window: int = 21, min_periods: int = 5
) -> pd.DataFrame:
    """Split returns by a trailing benchmark-volatility regime.

    Undefined warm-up observations are excluded. Drawdown and CAGR describe the
    sequence of returns observed inside each regime, not one continuous market
    interval. ``window`` and ``min_periods`` are measured in observations.
    """
    if result.benchmark_returns is None:
        return pd.DataFrame()
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, Integral)
        or window < 2
    ):
        raise ValueError("window must be an integer greater than 1.")
    if (
        isinstance(min_periods, (bool, np.bool_))
        or not isinstance(min_periods, Integral)
        or min_periods < 2
        or min_periods > window
    ):
        raise ValueError("min_periods must be an integer in [2, window].")
    window = int(window)
    min_periods = int(min_periods)

    ppy = result.config.periods_per_year
    rf = result.config.backtest.risk_free_rate
    aligned = pd.concat(
        {
            "strategy": result.returns,
            "benchmark": result.benchmark_returns,
        },
        axis=1,
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    roll_vol = aligned["benchmark"].rolling(window, min_periods=min_periods).std(ddof=1)
    median = roll_vol.median()
    if not np.isfinite(median):
        return pd.DataFrame()

    valid = roll_vol.notna()
    regimes = {
        "Low volatility": valid & (roll_vol <= median),
        "High volatility": valid & (roll_vol > median),
    }
    rows: list[dict[str, object]] = []
    for label, mask in regimes.items():
        regime_returns = aligned.loc[mask, "strategy"]
        if regime_returns.empty:
            continue
        row = _subperiod_row(label, regime_returns, ppy, rf)
        row["Observations"] = len(regime_returns)
        rows.append(row)
    return pd.DataFrame(rows)


#: Formatting kind for each statistic name BootstrapResult.summary() reports.
_BOOTSTRAP_STATISTIC_FORMAT: dict[str, tuple[str, str]] = {
    "cagr": ("CAGR", "pct"),
    "sharpe": ("Sharpe", "num"),
    "max_drawdown": ("Max Drawdown", "pct"),
    "final_value": ("Final Value", "currency"),
}


def pair_diagnostics_summary_table(diagnostics: PairDiagnostics) -> pd.DataFrame:
    """Metric/Value summary table for a pairs-trading result's diagnostics.

    A snapshot only -- the full spread/indicator/rolling-stability history
    lives in the accompanying chart (``reporting.charts.pair_spread_
    chart``), not in this table.
    """
    adf = diagnostics.adf_result
    coint = diagnostics.cointegration_result
    rows = [
        ("Symbols", f"{diagnostics.symbol_a} / {diagnostics.symbol_b}"),
        ("Return correlation", _fmt(diagnostics.correlation, "num")),
        (
            "Hedge-ratio stability (std of beta)",
            _fmt(diagnostics.hedge_ratio_stability, "num"),
        ),
        ("Half-life (periods)", _fmt(diagnostics.half_life, "num")),
        ("ADF statistic (spread)", _fmt(adf.statistic, "num") if adf else "n/a"),
        ("ADF p-value (spread)", _fmt(adf.pvalue, "num") if adf else "n/a"),
        (
            "Engle-Granger statistic",
            _fmt(coint.statistic, "num") if coint else "n/a",
        ),
        ("Engle-Granger p-value", _fmt(coint.pvalue, "num") if coint else "n/a"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def format_bootstrap_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Format ``BootstrapResult.summary()`` for display.

    Its ``mean``/``median``/``std``/``p_lower``/``p_upper`` columns stack
    values of very different scale across rows -- a CAGR near 0.05 next to
    a final value near 100000 -- because each row is a different
    statistic sharing the same generic columns. Pandas' default float
    repr renders that mix inconsistently (scientific notation for some
    cells, fixed-point for others, depending on each cell's own
    magnitude). Formatting every cell in a row by its own statistic's
    kind -- the same percent/number/currency convention ``metrics_table``
    uses -- keeps the whole table in fixed-point notation regardless of
    what the other rows contain.
    """
    numeric_columns = [column for column in summary.columns if column != "statistic"]
    rows = []
    for _, row in summary.iterrows():
        label, kind = _BOOTSTRAP_STATISTIC_FORMAT.get(
            row["statistic"], (str(row["statistic"]), "num")
        )
        formatted_row: dict[str, object] = {"statistic": label}
        for column in numeric_columns:
            formatted_row[column] = _fmt(row[column], kind)
        rows.append(formatted_row)
    return pd.DataFrame(rows, columns=["statistic", *numeric_columns])
