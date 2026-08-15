"""Reporting: charts, tables, HTML report and research narrative."""

from __future__ import annotations

from quantlab.reporting.charts import (
    equity_and_drawdown_figure,
    report_figures,
    save_figures,
)
from quantlab.reporting.html_report import render_html_report
from quantlab.reporting.tables import (
    gross_net_table,
    metrics_table,
    regime_table,
    subperiod_table,
    yearly_returns_table,
)

__all__ = [
    "equity_and_drawdown_figure",
    "gross_net_table",
    "metrics_table",
    "regime_table",
    "render_html_report",
    "report_figures",
    "save_figures",
    "subperiod_table",
    "yearly_returns_table",
]
