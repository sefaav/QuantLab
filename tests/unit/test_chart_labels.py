"""Tests for benchmark labels shared by report and dashboard charts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from quantlab.reporting.charts import (
    benchmark_legend_label,
    equity_and_drawdown_figure,
    equity_curve_chart,
)


def _result(label: str | None) -> Any:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return SimpleNamespace(
        config=SimpleNamespace(benchmark_label=label),
        equity_curve=pd.Series([100.0, 101.0, 102.0], index=index),
        benchmark_returns=pd.Series([0.0, 0.02, -0.01], index=index),
    )


def test_benchmark_label_identifies_configured_benchmark() -> None:
    result = _result("Equal weight")

    assert benchmark_legend_label(result) == "Benchmark (Equal weight)"


def test_static_and_interactive_report_charts_share_benchmark_label() -> None:
    result = _result("SPY")

    static = equity_curve_chart(result)
    interactive = equity_and_drawdown_figure(result)

    assert static.axes[0].get_legend_handles_labels()[1] == [
        "Strategy",
        "Benchmark (SPY)",
    ]
    assert [trace.name for trace in interactive.data[:2]] == [
        "Strategy",
        "Benchmark (SPY)",
    ]


def test_benchmark_label_has_safe_fallback_when_benchmark_is_disabled() -> None:
    assert benchmark_legend_label(_result(None)) == "Benchmark"
