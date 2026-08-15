"""Dashboard behaviour tests.

Uses Streamlit's ``AppTest`` harness to run the real ``app.py`` script (not a
reimplementation of its logic), driving it through an actual "Run backtest"
click against locally cached, offline (CSV-source) SPY/QQQ data and
inspecting actual tab content — a widget merely *appearing* is not enough to
catch a broken backtest or an empty/placeholder-only tab.
"""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from quantlab.dashboard.components import _monthly_return_pivot, render_metric_cards

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP_PATH = "src/quantlab/dashboard/app.py"


def _sidebar_selectbox(at: AppTest, label: str) -> Any:
    return next(sb for sb in at.sidebar.selectbox if sb.label == label)


def _sidebar_date_input(at: AppTest, label: str) -> Any:
    return next(d for d in at.sidebar.date_input if d.label == label)


def _sidebar_number_input(at: AppTest, label: str) -> Any:
    return next(field for field in at.sidebar.number_input if field.label == label)


def _sidebar_toggle(at: AppTest, label: str) -> Any:
    return next(toggle for toggle in at.sidebar.toggle if toggle.label == label)


def _sidebar_checkbox(at: AppTest, label: str) -> Any:
    return next(box for box in at.sidebar.checkbox if box.label == label)


def _configure_offline_pairs_trade(at: AppTest) -> AppTest:
    """Point the dashboard at locally cached CSV data for SPY/QQQ."""
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    at.sidebar.text_input[0].set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()
    return at


def test_default_dates_are_visible_and_ordered() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    start = _sidebar_date_input(at, "Start date")
    end = _sidebar_date_input(at, "End date")
    assert start.value == datetime.date(2019, 1, 1)
    assert start.value < end.value


def test_sidebar_warns_that_one_calendar_applies_to_the_entire_universe() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert any(
        "One calendar applies to every symbol" in warning.value
        and "AAPL and 1211.HK" in warning.value
        for warning in at.sidebar.warning
    )


def test_monthly_return_pivot_preserves_a_month_without_observations() -> None:
    returns = pd.Series(
        [0.10, -0.05],
        index=pd.to_datetime(["2024-01-31", "2024-03-31"]),
    )

    pivot = _monthly_return_pivot(returns)

    assert pivot.loc[2024, 1] == pytest.approx(0.10)
    assert pd.isna(pivot.loc[2024, 2])
    assert pivot.loc[2024, 3] == pytest.approx(-0.05)


def test_metric_cards_use_a_four_column_grid_and_explain_cost_units() -> None:
    class FakeColumn:
        def __init__(self, sink: list[tuple[str, str, dict[str, object]]]) -> None:
            self._sink = sink

        def metric(self, label: str, value: str, **kwargs: object) -> None:
            self._sink.append((label, value, kwargs))

    class FakeStreamlit:
        def __init__(self) -> None:
            self.columns_requested: int | None = None
            self.metrics: list[tuple[str, str, dict[str, object]]] = []

        def columns(self, n: int) -> list[FakeColumn]:
            self.columns_requested = n
            return [FakeColumn(self.metrics) for _ in range(n)]

    result: Any = SimpleNamespace(
        metrics={},
        total_costs=lambda: 1234.0,
        number_of_trades=lambda: 7,
    )
    fake = FakeStreamlit()

    render_metric_cards(fake, result)

    # Eight cards over 4 columns wrap into two visual rows of four, matching
    # the platform's default two-row metric layout.
    assert fake.columns_requested == 4
    assert len(fake.metrics) == 8
    costs = next(metric for metric in fake.metrics if metric[0] == "Total costs")
    assert costs[1] == "1,234"
    assert "currency units" in str(costs[2]["help"])


@pytest.mark.parametrize(
    ("signal_scaling", "expected_options"),
    [
        (
            "binary",
            [
                "equal_weight",
                "signal_proportional",
                "inverse_volatility",
                "volatility_targeting",
            ],
        ),
        (
            "continuous",
            ["signal_proportional", "inverse_volatility", "volatility_targeting"],
        ),
        ("volatility_adjusted", ["signal_proportional"]),
    ],
)
def test_allocator_options_match_time_series_momentum_scaling(
    signal_scaling: str, expected_options: list[str]
) -> None:
    """The dashboard must never offer an allocator/scaling pair that
    ExperimentConfig's validators would reject at "Run backtest" time.

    Each scaling is checked against a fresh AppTest instance: driving one
    instance through several scaling changes in sequence hits an AppTest
    widget-identity quirk unrelated to the app itself (confirmed harmless in
    a real browser), so this test sidesteps it instead of working around it.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Strategy").set_value("time_series_momentum").run()
    _sidebar_selectbox(at, "Signal scaling").set_value(signal_scaling).run()
    assert not at.exception

    allocator = _sidebar_selectbox(at, "Allocator")
    assert allocator.options == expected_options
    assert allocator.value in expected_options


def test_pairs_trading_symbol_inputs_render_without_crash() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    at.sidebar.text_input[0].set_value("AAA, BBB").run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()
    assert not at.exception

    symbol_a = _sidebar_selectbox(at, "Symbol A")
    symbol_b = _sidebar_selectbox(at, "Symbol B")
    # Symbol B's choices must exclude whatever Symbol A holds — a pairs
    # trade needs two distinct legs, so {"AAA", "BBB"} membership alone
    # would miss a regression that let both selectors collapse onto AAA.
    assert symbol_a.value == "AAA"
    assert symbol_b.value == "BBB"


def test_alternative_benchmark_can_be_selected_without_symbol_input() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    _sidebar_selectbox(at, "Benchmark").set_value("cash").run()

    assert not at.exception
    assert not any(field.label == "Benchmark symbol" for field in at.sidebar.text_input)


def test_bundled_demo_csvs_require_an_explicit_dashboard_opt_in() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    demo_toggle = _sidebar_toggle(at, "Allow bundled synthetic demo data")
    assert demo_toggle.value is False

    _configure_offline_pairs_trade(at)
    _sidebar_toggle(at, "Allow bundled synthetic demo data").set_value(True).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    assert at.session_state["result"].config.data.use_bundled_demo_data is True


def test_bundled_demo_toggle_is_hidden_for_non_csv_sources() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    assert not any(
        toggle.label == "Allow bundled synthetic demo data"
        for toggle in at.sidebar.toggle
    )


def test_strategy_widget_dispatch_uses_exact_registered_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.base as strategy_base

    custom_name = "custom_mean_reversion_probe"
    monkeypatch.setattr(
        strategy_base,
        "available_strategies",
        lambda: ["buy_and_hold", custom_name],
    )
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    _sidebar_selectbox(at, "Strategy").set_value(custom_name).run()

    assert not at.exception
    assert not any(slider.label == "Entry z-score" for slider in at.sidebar.slider)


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
def test_reversion_exit_slider_stays_strictly_below_entry(
    strategy_name: str,
) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        at.sidebar.text_input[0].set_value("AAA, BBB").run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()

    entry = next(
        slider for slider in at.sidebar.slider if slider.label == "Entry z-score"
    )
    entry.set_value(1.0).run()
    exit_ = next(
        slider for slider in at.sidebar.slider if slider.label == "Exit z-score"
    )

    exit_max = exit_.proto.max
    entry_value = cast(float, entry.value)
    assert exit_max == pytest.approx(0.9)
    assert exit_max < entry_value


def test_holdout_controls_do_not_claim_automatic_tuning() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    holdout = _sidebar_checkbox(at, "Chronological holdout (train / validation / test)")
    assert "No fitting or parameter tuning happens here" in holdout.proto.help
    holdout.set_value(True).run()
    validation = next(
        slider for slider in at.sidebar.slider if slider.label == "Validation fraction"
    )
    assert "does not tune or select parameters automatically" in validation.proto.help


def test_pairs_trading_with_fewer_than_two_symbols_shows_error_not_crash() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value("AAA").run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()
    assert not at.exception
    assert any("at least two symbols" in e.value for e in at.sidebar.error)


def test_pairs_trading_requires_two_distinct_symbols() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value("AAA, AAA").run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()

    assert not at.exception
    assert any("at least two symbols" in e.value for e in at.sidebar.error)


def test_dashboard_can_disable_volatility_targeting_and_set_risk_free_rate() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _sidebar_toggle(at, "Enable volatility targeting").set_value(False).run()
    _sidebar_number_input(at, "Risk-free rate (annual %)").set_value(3.5).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    result = at.session_state["result"]
    assert result.config.portfolio.target_volatility is None
    assert result.config.risk_free_rate == pytest.approx(0.035)


def test_failed_backtest_invalidates_previous_result() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.sidebar.button[0].click().run()
    assert "result" in at.session_state

    at.sidebar.text_input[0].set_value("").run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert any("Backtest failed" in error.value for error in at.error)
    assert "result" not in at.session_state
    assert "stress_tests" not in at.session_state
    assert "report_html" not in at.session_state


def test_successful_pairs_backtest_creates_expected_tab_navigation() -> None:
    """A successful run stores its result and renders the default results tab."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    assert "result" in at.session_state
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["Results", "Trades", "Robustness", "Report"]
    assert len(at.tabs[0].metric) > 0


def test_robustness_tab_shows_holdout_table_when_enabled() -> None:
    """With holdout ticked, the Robustness tab must show a real data table,
    not just the "no holdout attached" placeholder."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _sidebar_checkbox(
        at, "Chronological holdout (train / validation / test)"
    ).set_value(True).run()
    assert not at.exception

    # Tab content now renders lazily (only the currently-"open" tab computes
    # its content) — `AppTest` has no way to simulate an actual frontend tab
    # click, so the tab is selected directly via the widget's own session
    # state key instead (see `app.py`'s `st.tabs(..., key=...)`).
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    robustness_tab = at.tabs[2]
    assert robustness_tab.label == "Robustness"
    assert len(robustness_tab.dataframe) == 1
    table = robustness_tab.dataframe[0].value
    assert list(table["Block"]) == ["Train", "Validation", "Test (out-of-sample)"]
    column_config = json.loads(robustness_tab.dataframe[0].proto.columns)
    assert column_config["CAGR"]["type_config"]["format"] == "percent"
    assert column_config["Max Drawdown"]["type_config"]["format"] == "percent"
    assert not robustness_tab.info  # placeholder message must be gone


def test_robustness_tab_shows_placeholder_without_holdout() -> None:
    """Without holdout ticked, the tab must say so explicitly rather than
    silently showing nothing or a stale table."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    robustness_tab = at.tabs[2]
    assert len(robustness_tab.dataframe) == 0
    assert any("No out-of-sample holdout" in i.value for i in robustness_tab.info)
    assert any(b.label == "Run stress tests" for b in robustness_tab.button)


def test_failed_stress_run_clears_previous_stress_evidence() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    result = at.session_state["result"]
    result.metadata["data_hash"] = "stale-data-hash"
    at.session_state["result"] = result
    at.session_state["stress_tests"] = pd.DataFrame({"scenario": ["old"]})

    robustness_tab = at.tabs[2]
    stress_button = next(
        button for button in robustness_tab.button if button.label == "Run stress tests"
    )
    stress_button.click().run()

    assert not at.exception
    assert any("Market data changed" in error.value for error in at.error)
    assert "stress_tests" not in at.session_state
    assert "report_html" not in at.session_state


def test_stale_result_warning_shown_after_sidebar_change() -> None:
    """Changing a control after a run must flag the displayed result as
    stale, instead of silently showing numbers that no longer match the
    sidebar."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.sidebar.button[0].click().run()
    assert not any("configuration has changed" in w.value for w in at.warning)

    _sidebar_selectbox(at, "Strategy").set_value("mean_reversion").run()
    assert not at.exception
    assert any("configuration has changed" in w.value for w in at.warning)


def test_frequency_mismatch_shown_as_prominent_error_not_small_caption() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    at.sidebar.text_input[0].set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Frequency").set_value("1h").run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert any("Frequency mismatch detected" in e.value for e in at.error)


def test_no_deprecated_streamlit_apis_used() -> None:
    with open(APP_PATH, encoding="utf-8") as f:
        text = f.read()
    with open("src/quantlab/dashboard/components.py", encoding="utf-8") as f:
        text += f.read()
    assert "use_container_width" not in text
    assert "components.v1.html" not in text
    assert "st.components" not in text


def test_only_the_open_tab_renders_its_content() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.sidebar.button[0].click().run()
    assert not at.exception

    results_tab = at.tabs[0]
    assert results_tab.label == "Results"
    assert len(results_tab.metric) > 0  # the *open* (default) tab did render

    trades_tab = at.tabs[1]
    assert trades_tab.label == "Trades"
    # Zero rendered elements at all (not merely an empty dataframe list) —
    # this config generates zero trades either way, so `render_trade_table`
    # would show an `st.info(...)` placeholder if it ran at all; the closed
    # tab must not even have that.
    assert len(trades_tab) == 0


def test_switching_the_open_tab_renders_its_content_instead() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Trades"
    at.sidebar.button[0].click().run()
    assert not at.exception

    trades_tab = at.tabs[1]
    assert trades_tab.label == "Trades"
    assert any("No trades were recorded" in i.value for i in trades_tab.info)


def test_report_tab_shows_a_warning_when_a_chart_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.reporting.charts as charts_mod

    def boom(_result: Any) -> Any:
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr(charts_mod, "equity_curve_chart", boom)

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Report"
    at.sidebar.button[0].click().run()
    assert not at.exception

    report_tab = at.tabs[3]
    assert report_tab.label == "Report"
    assert any(
        "equity_curve" in w.value and "simulated chart failure" in w.value
        for w in report_tab.warning
    )


def test_report_tab_includes_stress_tests_run_in_robustness_tab() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    robustness_tab = at.tabs[2]
    stress_button = next(
        b for b in robustness_tab.button if b.label == "Run stress tests"
    )
    stress_button.click().run()
    assert not at.exception
    assert "stress_tests" in at.session_state
    assert at.session_state["stress_tests"] is not None

    at.session_state["dashboard_active_tab"] = "Report"
    at.run()
    assert not at.exception

    report_tab = at.tabs[3]
    assert report_tab.label == "Report"
    _, (html, _warnings) = at.session_state["report_html"]
    assert "Stress Tests" in html
