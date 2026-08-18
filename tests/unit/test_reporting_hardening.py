"""Regression tests for reporting accuracy and rendering resilience."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from tests.conftest import geometric_series, make_ohlcv

from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.reporting import charts
from quantlab.reporting.html_report import (
    _render_data_quality,
    _render_robustness,
)
from quantlab.reporting.research_summary import (
    conclusion,
    data_description,
    methodology,
)
from quantlab.reporting.tables import _fmt, regime_table


def _config(
    *, volume_slippage: bool = False, demo_fallback: bool = False
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "reporting_hardening",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2019-01-01",
                "end_date": "2021-12-31",
                "use_bundled_demo_data": demo_fallback,
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {
                "maximum_weight": 0.8,
                "target_volatility": 0.1,
                "maximum_leverage": 1.2,
                "maximum_turnover": 0.3,
            },
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 4.0,
                "slippage_bps": 1.0,
                "slippage_model": "volume" if volume_slippage else "constant",
                "impact_coefficient": 0.2,
            },
        }
    )


def _result() -> BacktestResult:
    data = make_ohlcv(
        "AAA",
        geometric_series(180, mu=0.0004, sigma=0.01, s0=100.0, seed=7),
        start="2020-01-01",
    )
    return run_backtest_from_config(data, _config())


def test_metric_formatter_handles_numpy_non_finite_values() -> None:
    assert _fmt(np.float32(np.inf), "num") == "n/a"
    assert _fmt(np.float32(np.nan), "pct") == "n/a"
    assert _fmt(np.float32(np.nan), "int") == "n/a"


def test_regime_table_excludes_undefined_warmup_and_empty_regimes() -> None:
    short_index = pd.date_range("2024-01-01", periods=4, freq="D")
    config = SimpleNamespace(
        periods_per_year=252,
        backtest=SimpleNamespace(risk_free_rate=0.0),
    )
    short: Any = SimpleNamespace(
        config=config,
        benchmark_returns=pd.Series([0.01, -0.01, 0.01, -0.01], index=short_index),
        returns=pd.Series(0.01, index=short_index),
    )
    assert regime_table(short).empty

    index = pd.date_range("2024-01-01", periods=8, freq="D")
    stable: Any = SimpleNamespace(
        config=config,
        benchmark_returns=pd.Series(0.0, index=index),
        returns=pd.Series(0.01, index=index),
    )
    table = regime_table(stable)
    assert table["Period"].tolist() == ["Low volatility"]
    assert table["Observations"].tolist() == [4]


def test_monthly_heatmap_marks_empty_month_as_unavailable() -> None:
    returns = pd.Series(
        [0.10, -0.05],
        index=pd.to_datetime(["2024-01-31", "2024-03-31"]),
    )
    fake_result: Any = SimpleNamespace(returns=returns)
    figure = charts.monthly_returns_heatmap(fake_result)
    values = figure.axes[0].images[0].get_array()
    labels = {text.get_text() for text in figure.axes[0].texts}
    try:
        assert values is not None
        assert np.ma.is_masked(values[0, 1])
        assert "n/a" in labels
    finally:
        figure.clear()


def test_short_rolling_sharpe_chart_explains_missing_history() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    result: Any = SimpleNamespace(
        returns=pd.Series(0.0, index=index),
        config=SimpleNamespace(risk_free_rate=0.0, periods_per_year=252),
    )
    figure = charts.rolling_sharpe_chart(result, window=20)
    try:
        assert "20 observations" in figure.axes[0].texts[0].get_text()
        assert "10 available" in figure.axes[0].texts[0].get_text()
    finally:
        figure.clear()
    with pytest.raises(ValueError, match="greater than 1"):
        charts.rolling_sharpe_chart(result, window=1)


def test_robustness_tables_format_percentage_columns() -> None:
    frame = pd.DataFrame(
        {
            "scenario": ["test"],
            "cagr": [0.05],
            "Max Drawdown": [-0.20],
            "test_sharpe": [0.42],
        }
    )
    rendered = _render_robustness({"stress_tests": frame})
    assert "5.00%" in rendered
    assert "-20.00%" in rendered
    assert "0.42" in rendered


def _sensitivity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lookback_period": [60, 60, 120, 120],
            "top_fraction": [0.3, 0.5, 0.3, 0.5],
            "sharpe": [0.5, 0.8, 0.2, 0.6],
            "cagr": [0.1, 0.15, 0.05, 0.12],
            "max_drawdown": [-0.1, -0.12, -0.2, -0.15],
            "turnover": [1.0, 1.0, 1.0, 1.0],
            "num_trades": [10, 10, 10, 10],
            "status": ["ok"] * 4,
            "error": [None] * 4,
        }
    )


def test_sensitivity_heatmap_chart_infers_swept_parameters() -> None:
    """The two swept-parameter columns are whatever is left after excluding
    the fixed metric/status columns every sensitivity table carries."""
    figure = charts.sensitivity_heatmap_chart(_sensitivity_frame())
    try:
        assert figure.axes[0].get_xlabel() == "lookback_period"
        assert figure.axes[0].get_ylabel() == "top_fraction"
    finally:
        figure.clear()


def test_sensitivity_heatmap_chart_rejects_ambiguous_columns() -> None:
    sensitivity = pd.DataFrame(
        {"a": [1], "b": [2], "c": [3], "sharpe": [0.5], "status": ["ok"]}
    )
    with pytest.raises(ValueError, match="exactly two"):
        charts.sensitivity_heatmap_chart(sensitivity)


def test_render_robustness_embeds_sensitivity_heatmap() -> None:
    rendered = _render_robustness({"sensitivity": _sensitivity_frame()})
    assert '<img src="data:image/png;base64,' in rendered
    assert "Parameter sensitivity heatmap" in rendered


def test_render_robustness_sensitivity_heatmap_failure_does_not_crash_report() -> None:
    """A malformed sensitivity table (here, only one swept-parameter column)
    must not take down the whole report — same resilience as the other
    charts (see report_figures)."""
    malformed = pd.DataFrame(
        {"lookback_period": [60], "sharpe": [0.5], "status": ["ok"]}
    )
    warnings: list[str] = []
    rendered = _render_robustness({"sensitivity": malformed}, warnings)
    assert "Chart unavailable" in rendered
    assert warnings
    assert "60" in rendered  # the raw table must still render


def test_methodology_describes_volume_slippage_and_constraints() -> None:
    result: Any = SimpleNamespace(
        config=_config(volume_slippage=True),
        metadata={},
    )
    text = methodology(result)
    assert "volume-based slippage" in text
    assert "impact coefficient 0.2000" in text
    assert "maximum absolute weight 80.00%" in text
    assert "annual volatility target 10.00%" in text
    assert "maximum L1 turnover per rebalance 0.30" in text


def test_data_description_separates_requested_and_observed_periods() -> None:
    index = pd.date_range("2020-02-03", periods=3, freq="D")
    result: Any = SimpleNamespace(
        config=_config(demo_fallback=True),
        equity_curve=pd.Series([100.0, 101.0, 102.0], index=index),
    )
    text = data_description(result)
    assert "Requested period: 2019-01-01 to 2021-12-31" in text
    assert "Observed backtest period: 2020-02-03 to 2020-02-05" in text
    assert "Bundled synthetic CSV fallback was enabled" in text


def test_missing_oos_metric_is_reported_as_unavailable() -> None:
    result: Any = SimpleNamespace(
        metrics={"sharpe_ratio": 0.5, "max_drawdown": -0.1},
        metadata={"walk_forward_oos_metrics": {"sharpe_ratio": 0.2}},
    )
    text = conclusion(result)
    assert "out-of-sample maximum drawdown n/a" in text
    assert "out-of-sample maximum drawdown 0.00%" not in text


def test_save_figures_continues_after_one_invalid_image(tmp_path: Any) -> None:
    good = "data:image/png;base64," + "cG5n"
    warnings: list[str] = []
    paths = charts.save_figures(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path,
        warnings,
        rendered={"good": good, "bad": "not-a-data-uri"},
    )
    assert [path.name for path in paths] == ["good.png"]
    assert (tmp_path / "good.png").read_bytes() == b"png"
    assert any("'bad'" in warning for warning in warnings)


def test_result_save_renders_each_chart_only_once(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result()
    calls = 0

    def builder(_result: BacktestResult) -> Any:
        nonlocal calls
        calls += 1
        figure = Figure()
        axis = figure.subplots()
        axis.plot([0.0, 1.0])
        return figure

    monkeypatch.setattr(charts, "_chart_builders", lambda: {"equity_curve": builder})
    output = result.save(tmp_path / "report")
    assert calls == 1
    assert (output / "figures" / "equity_curve.png").is_file()
    assert 'alt="Equity curve"' in (output / "report.html").read_text(encoding="utf-8")


def test_html_report_shows_chart_placeholders() -> None:
    rendered = _result().to_html(figures={})
    assert "Chart unavailable:" in rendered
    assert "Full-sample headline metrics" in rendered


def test_data_quality_section_includes_counts_without_warnings() -> None:
    rendered = _render_data_quality(
        {
            "raw_row_count": 10,
            "clean_row_count": 9,
            "duplicate_count": 1,
            "invalid_price_count": 0,
            "missing_value_count": {"close": 1},
            "missing_periods": [],
            "warnings": [],
        }
    )
    assert "Clean rows" in rendered
    assert "Missing values" in rendered
    assert "No data-quality warnings" in rendered
