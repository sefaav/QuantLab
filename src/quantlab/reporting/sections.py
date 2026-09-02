"""Generic, strategy-agnostic containers for an extra HTML report section.

A strategy profile that declares its own :class:`~quantlab.dashboard.
explorer.profile.ResultsDiagnostics` builds one of these to describe its
report section; ``html_report.py`` renders it by type, never by strategy
name, so a future strategy can add its own report section without any
change to the report renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DiagnosticsSection:
    """One report section: a table, an optional chart, an optional note.

    ``chart_data_uri`` is a ready-to-embed ``data:image/...;base64,...``
    string (see ``reporting.charts.fig_to_base64``). ``note`` is plain
    text (escaped at render time, never treated as markup).
    """

    table: pd.DataFrame
    chart_data_uri: str | None = None
    note: str | None = None
