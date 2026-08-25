"""Render a standalone HTML research report with embedded charts."""

from __future__ import annotations

import html
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantlab.logging_config import get_logger
from quantlab.reporting import research_summary as rs
from quantlab.reporting.charts import report_figures
from quantlab.reporting.tables import gross_net_table, metrics_table, subperiod_table

if TYPE_CHECKING:
    from quantlab.backtesting.result import BacktestResult

logger = get_logger(__name__)

_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  margin: 0; padding: 0 0 60px; color: #1f2937; background: #f9fafb; }
header { background: #111827; color: #f9fafb; padding: 28px 40px; }
header h1 { margin: 0 0 4px; font-size: 24px; }
header p { margin: 0; color: #9ca3af; font-size: 13px; }
.disclaimer { background: #fef3c7; border-left: 4px solid #f59e0b; color: #92400e;
  padding: 12px 40px; font-size: 13px; }
main { max-width: 980px; margin: 0 auto; padding: 24px 40px; }
section { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
  padding: 20px 24px; margin: 18px 0; }
h2 { font-size: 18px; margin-top: 0; border-bottom: 2px solid #2563eb;
  padding-bottom: 6px; color: #111827; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }
th { background: #f3f4f6; }
img { max-width: 100%; height: auto; margin: 8px 0; }
ul { font-size: 13px; line-height: 1.6; }
.blockquote { border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 16px;
  font-style: italic; color: #1e3a8a; }
.chart-unavailable { color: #92400e; background: #fef3c7; padding: 8px 12px;
  border-radius: 6px; font-size: 13px; }
.quality-ok { color: #166534; }
.quality-warning { color: #92400e; }
footer { text-align: center; color: #9ca3af; font-size: 12px; padding-top: 20px; }
"""

_PERCENT_COLUMNS = {
    "return",
    "totalreturn",
    "annualreturn",
    "annualizedreturn",
    "cagr",
    "maxdrawdown",
    "averagedrawdown",
    "volatility",
    "annualizedvolatility",
    "alpha",
    "trackingerror",
    "var95",
    "var99",
    "cvar95",
    "cvar99",
}
_NUMBER_COLUMNS = {
    "sharpe",
    "sharperatio",
    "sortino",
    "sortinoratio",
    "calmar",
    "calmarratio",
    "beta",
    "inforatio",
    "informationratio",
    "validationscore",
    "testsharpe",
    "turnover",
    "turnoverx",
}
_INTEGER_COLUMNS = {"observations", "numberoftrades", "trades", "fold"}


def _normalise_label(label: object) -> str:
    return "".join(character for character in str(label).lower() if character.isalnum())


def _finite_number(value: object) -> float | None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _format_cell(value: object, column: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return "n/a"
    if isinstance(value, Real) and _finite_number(value) is None:
        return "n/a"
    key = _normalise_label(column)
    if key not in _PERCENT_COLUMNS | _NUMBER_COLUMNS | _INTEGER_COLUMNS:
        return value
    number = _finite_number(value)
    if number is None:
        return "n/a"
    if key in _PERCENT_COLUMNS:
        return f"{number:.2%}"
    if key in _INTEGER_COLUMNS:
        return f"{int(number)}"
    return f"{number:.2f}"


def _format_report_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        output[column] = output[column].map(
            lambda value, col=column: _format_cell(value, col)
        )
    return output


def _table_html(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "<p><em>Not available.</em></p>"
    return frame.to_html(index=False, border=0, escape=True)


def _fmt_pct_cols(frame: pd.DataFrame) -> pd.DataFrame:
    """Format known metric columns for backward-compatible callers."""
    return _format_report_table(frame)


def _write_text_atomic(path: Path, document: str) -> None:
    """Replace one text file atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_html_report(
    result: BacktestResult,
    output_path: str | Path | None = None,
    *,
    robustness: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    figures: Mapping[str, str] | None = None,
) -> str:
    """Render the report and optionally replace ``output_path`` atomically.

    Pre-rendered ``figures`` let a saved report reuse the same images for its
    PNG directory and embedded HTML instead of executing every chart twice.
    """
    rendered_figures = (
        dict(figures) if figures is not None else report_figures(result, warnings)
    )
    config = result.config

    def image(name: str, alt: str) -> str:
        source = rendered_figures.get(name)
        if source is None:
            return (
                '<p class="chart-unavailable"><strong>Chart unavailable:</strong> '
                f"{html.escape(alt)}. See metadata.json save_warnings or the log."
                "</p>"
            )
        return f'<img src="{source}" alt="{html.escape(alt)}"/>'

    limitations_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in rs.limitations(result)
    )
    robustness_html = _render_robustness(robustness, warnings)
    data_quality_html = _render_data_quality(result.metadata.get("data_quality"))
    # A walk-forward OOS result's `metrics` *are* the stitched out-of-sample
    # series (see WalkForwardValidator._build_oos_result) — labelling the
    # Results headings "Full-sample" would claim the opposite of what they
    # actually are. Holdout/plain results keep the accurate "Full-sample"
    # label: their `metrics` remain a genuine full-sample fit even when
    # holdout OOS evidence is also attached separately.
    results_scope = (
        "Out-of-sample (walk-forward)"
        if rs.out_of_sample_scope(result) is not None
        else "Full-sample"
    )
    # Only claim reproducibility when the engine could positively verify the
    # actual strategy/allocator/execution objects against config.yaml's own
    # values -- a direct-API run (docs/api.md) can pass a custom object that
    # silently diverges from config.yaml on any of the three (see
    # BacktestEngine._build_metadata), which must never be asserted away by
    # an unconditional footer. All three must verify, not just any one.
    config_verified = (
        bool(result.metadata.get("config_yaml_reflects_strategy"))
        and bool(result.metadata.get("config_yaml_reflects_allocator"))
        and bool(result.metadata.get("config_yaml_reflects_execution"))
    )
    reproducibility_note = (
        "reproducible from config.yaml given the same code, data and "
        "dependency versions recorded in metadata.json"
        if config_verified
        else "config.yaml in this bundle may not exactly reflect the "
        "strategy/allocator/execution actually used for this run -- see "
        "metadata.json's config_yaml_reflects_strategy/"
        "config_yaml_reflects_allocator/config_yaml_reflects_execution"
    )

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>QuantLab Report — {html.escape(config.experiment_name)}</title>
<style>{_CSS}</style></head>
<body>
<header>
  <h1>QuantLab Research Report</h1>
  <p>{html.escape(config.experiment_name)} &middot;
     strategy: {html.escape(config.strategy_name)}</p>
</header>
<div class="disclaimer">
  Research tool for educational purposes only. Not investment advice.
  Historical performance does not guarantee future results.
</div>
<main>
  <section><h2>Executive summary</h2>
    <p>{html.escape(rs.executive_summary(result))}</p></section>

  <section><h2>Research question</h2>
    <div class="blockquote">{html.escape(rs.research_question(result))}</div></section>

  <section><h2>Hypothesis</h2><p>{html.escape(rs.hypothesis(result))}</p></section>

  <section><h2>Data</h2><p>{html.escape(rs.data_description(result))}</p>
    {data_quality_html}</section>

  <section><h2>Methodology</h2><p>{html.escape(rs.methodology(result))}</p></section>

  <section><h2>Results</h2>
    {image("equity_curve", "Equity curve")}
    {image("drawdown", "Drawdown")}
    <h3>{results_scope} headline metrics</h3>
    {_table_html(metrics_table(result))}
    <h3>{results_scope} gross vs net</h3>
    {_table_html(gross_net_table(result))}
    <h3>{results_scope} performance by year</h3>
    {_table_html(_format_report_table(subperiod_table(result)))}
    {image("monthly_returns", "Monthly returns heatmap")}
    {image("rolling_sharpe", "Rolling Sharpe")}
    {image("rolling_volatility", "Rolling volatility")}
    {image("exposure", "Exposure")}
    {image("cumulative_costs", "Cumulative costs")}
    {image("returns_distribution", "Return distribution")}
  </section>

  <section><h2>Robustness</h2>{robustness_html}</section>

  <section><h2>Limitations</h2><ul>{limitations_html}</ul></section>

  <section><h2>Conclusion</h2>
    <div class="blockquote">{html.escape(rs.conclusion(result))}</div></section>

  <footer>Generated by QuantLab &middot; {reproducibility_note}</footer>
</main>
</body></html>"""

    if output_path is not None:
        _write_text_atomic(Path(output_path), document)
    return document


def _mapping_table(value: Mapping[Any, Any]) -> pd.DataFrame:
    rows = [
        {"Metric": str(key).replace("_", " ").title(), "Value": _format_cell(item, key)}
        for key, item in value.items()
    ]
    return pd.DataFrame(rows)


def _render_robustness_value(value: object) -> str:
    if isinstance(value, pd.DataFrame):
        return _table_html(_format_report_table(value))
    if isinstance(value, pd.Series):
        return _table_html(_mapping_table(value.to_dict()))
    if isinstance(value, Mapping):
        return _table_html(_mapping_table(value))
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and value
        and all(isinstance(item, Mapping) for item in value)
    ):
        return _table_html(_format_report_table(pd.DataFrame(value)))
    return f"<p>{html.escape(str(value))}</p>"


def _render_sensitivity_heatmap(
    sensitivity: pd.DataFrame, warnings: list[str] | None
) -> str:
    """Render a 2-parameter sensitivity sweep as an embedded heatmap image."""
    from quantlab.reporting.charts import fig_to_base64, sensitivity_heatmap_chart

    try:
        data_uri = fig_to_base64(sensitivity_heatmap_chart(sensitivity))
    except Exception as exc:
        message = f"Could not render sensitivity heatmap for report: {exc}"
        logger.warning(message, exc_info=True)
        if warnings is not None:
            warnings.append(message)
        return (
            '<p class="chart-unavailable"><strong>Chart unavailable:</strong> '
            f"sensitivity heatmap. {html.escape(str(exc))}</p>"
        )
    return f'<img src="{data_uri}" alt="Parameter sensitivity heatmap"/>'


def _render_robustness(
    robustness: dict[str, Any] | None, warnings: list[str] | None = None
) -> str:
    """Render supplied validation artefacts or explain how to generate them."""
    if not robustness:
        return (
            "<p><em>No robustness evidence is attached to this run.</em></p>"
            "<ul>"
            "<li><code>quantlab walk-forward</code> — out-of-sample folds plus "
            "stress-test evidence.</li>"
            "<li><code>quantlab.validation.parameter_sensitivity."
            "run_parameter_sensitivity</code> — stability across parameter "
            "choices (Python API).</li>"
            "<li><code>quantlab.validation.bootstrap.bootstrap_returns</code> "
            "— resampled return-path uncertainty (Python API).</li>"
            "<li><code>quantlab.validation.robustness.monte_carlo_permutation"
            "</code> — significance against a random-sign null (Python API).</li>"
            "</ul>"
            "<p><em>See <code>notebooks/05_robustness_analysis.ipynb</code> for "
            "a worked example of all four.</em></p>"
        )
    parts: list[str] = []
    for key, value in robustness.items():
        heading = str(key).replace("_", " ").title()
        parts.append(f"<h3>{html.escape(heading)}</h3>")
        if key == "permutation_test":
            parts.append(
                "<p><em>Randomly flips the sign of excess returns to test "
                "the realised Sharpe against a no-edge random-sign null. A "
                "low p-value is evidence against that specific null, not a "
                "probability of future profitability.</em></p>"
            )
        elif key == "sensitivity" and isinstance(value, pd.DataFrame) and len(value):
            parts.append(_render_sensitivity_heatmap(value, warnings))
        parts.append(_render_robustness_value(value))
    return "".join(parts)


def _render_data_quality(data_quality: dict[str, Any] | None) -> str:
    """Render loader counts and warnings stored in result metadata."""
    if not data_quality:
        return ""

    missing_counts = data_quality.get("missing_value_count") or {}
    missing_total = (
        sum(int(value) for value in missing_counts.values())
        if isinstance(missing_counts, Mapping)
        else 0
    )
    rows = [
        {"Check": "Clean rows", "Value": data_quality.get("clean_row_count", "n/a")},
        {"Check": "Raw rows", "Value": data_quality.get("raw_row_count", "n/a")},
        {"Check": "Duplicates", "Value": data_quality.get("duplicate_count", 0)},
        {
            "Check": "Invalid prices",
            "Value": data_quality.get("invalid_price_count", 0),
        },
        {"Check": "Missing values", "Value": missing_total},
        {
            "Check": "Missing periods",
            "Value": len(data_quality.get("missing_periods") or []),
        },
    ]
    summary = _table_html(pd.DataFrame(rows))
    warning_list = data_quality.get("warnings") or []
    if not isinstance(warning_list, list):
        warning_list = [str(warning_list)]
    if not warning_list:
        return f'{summary}<p class="quality-ok"><em>No data-quality warnings.</em></p>'
    items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warning_list)
    return (
        f'{summary}<p class="quality-warning"><strong>'
        f"{len(warning_list)} data-quality warning(s):</strong></p><ul>{items}</ul>"
    )
