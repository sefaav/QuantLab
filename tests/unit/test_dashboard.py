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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from quantlab.dashboard.components import (
    _monthly_return_pivot,
    render_gross_net_comparison,
    render_metric_cards,
)

pytest.importorskip("streamlit")

import streamlit as st
from streamlit.testing.v1 import AppTest

# Absolute, since AppTest.from_file() resolves a relative path against the
# *calling test file's* directory rather than the working directory.
APP_PATH = str(
    Path(__file__).resolve().parents[2] / "src" / "quantlab" / "dashboard" / "app.py"
)


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


def _sidebar_text_input(at: AppTest, label: str) -> Any:
    return next(field for field in at.sidebar.text_input if field.label == label)


def _configure_offline_pairs_trade(at: AppTest) -> AppTest:
    """Point the dashboard at locally cached CSV data for SPY/QQQ."""
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()
    return at


def _switch_to_walk_forward_mode(at: AppTest) -> AppTest:
    at.segmented_control[0].set_value("Walk-forward").run()
    return at


def _configure_offline_walk_forward(at: AppTest) -> AppTest:
    """Point the dashboard at locally cached CSV data with small fold windows.

    ``mean_reversion`` (default ``lookback_period=20``) needs far less
    warm-up than momentum strategies, but the dashboard's default allocator
    is ``inverse_volatility`` (``volatility_window=63``), so train+validation
    must still clear that — hence 70/20 rather than something tinier.
    """
    _switch_to_walk_forward_mode(at)
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 7, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("mean_reversion").run()
    # Weekly (not the default monthly) guarantees a rebalance date inside
    # every small test window below, regardless of where a fold happens to
    # fall in the calendar.
    _sidebar_selectbox(at, "Rebalance frequency").set_value("weekly").run()
    _sidebar_number_input(at, "Train window (periods)").set_value(70).run()
    _sidebar_number_input(at, "Validation window (periods)").set_value(20).run()
    _sidebar_number_input(at, "Test window (periods)").set_value(20).run()
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


def _sidebar_multiselect(at: AppTest, label: str) -> Any:
    return next(ms for ms in at.sidebar.multiselect if ms.label == label)


_MARKET_CALENDAR_NOTE = (
    "One market calendar applies to the entire universe, so every symbol "
    "must follow the same trading schedule."
)


def test_csv_symbols_and_benchmark_help_mention_market_calendar() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    symbols_field = _sidebar_text_input(at, "Symbols (comma-separated)")
    assert _MARKET_CALENDAR_NOTE in symbols_field.proto.help

    _sidebar_selectbox(at, "Benchmark").set_value("symbol").run()
    benchmark_field = _sidebar_text_input(at, "Benchmark symbol")
    assert _MARKET_CALENDAR_NOTE in benchmark_field.proto.help


def test_yahoo_symbols_and_benchmark_help_mention_market_calendar() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    picker = _sidebar_multiselect(at, "Symbols")
    assert _MARKET_CALENDAR_NOTE in picker.proto.help

    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert _MARKET_CALENDAR_NOTE in benchmark.proto.help


def test_symbols_picker_help_icon_is_not_hidden_by_a_collapsed_label() -> None:
    """Regression test: Streamlit hides a widget's help tooltip icon along
    with a `label_visibility="collapsed"` label, which silently swallowed
    the market-calendar/incomplete-list notes above. The label must stay
    visible so the help icon (and thus those notes) stays reachable."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    picker = _sidebar_multiselect(at, "Symbols")
    assert picker.proto.label_visibility.value == 0  # VISIBLE


def test_binance_symbols_and_benchmark_help_omit_market_calendar_note() -> None:
    """Every Binance pair already shares the same 24/7 calendar, so the
    cross-market mixing warning (relevant for csv/yahoo) doesn't apply."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("binance").run()

    picker = _sidebar_multiselect(at, "Symbols")
    assert _MARKET_CALENDAR_NOTE not in picker.proto.help

    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert _MARKET_CALENDAR_NOTE not in benchmark.proto.help


def test_csv_source_keeps_the_free_text_symbols_field() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert any(
        field.label == "Symbols (comma-separated)" for field in at.sidebar.text_input
    )
    assert not any(ms.label == "Symbols" for ms in at.sidebar.multiselect)


def test_binance_symbols_picker_is_an_instant_dropdown_over_the_full_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binance's whole universe is loaded once; picking needs no search box."""
    from quantlab.data.base import SymbolSuggestion
    from quantlab.data.binance import BinanceDataSource

    fetches: list[None] = []

    def fake_universe(self: BinanceDataSource) -> list[SymbolSuggestion]:
        fetches.append(None)
        return [
            SymbolSuggestion(symbol="BTCUSDT", description="BTC/USDT"),
            SymbolSuggestion(symbol="ETHUSDT", description="ETH/USDT"),
        ]

    monkeypatch.setattr(BinanceDataSource, "list_trading_symbols", fake_universe)
    st.cache_data.clear()  # avoid a real universe cached by an earlier test/run

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("binance").run()

    assert not at.exception
    assert not any(field.label == "Search" for field in at.sidebar.text_input)
    picker = _sidebar_multiselect(at, "Symbols")
    assert picker.options == ["BTCUSDT — BTC/USDT", "ETHUSDT — ETH/USDT"]
    assert picker.value == ["BTCUSDT — BTC/USDT", "ETHUSDT — ETH/USDT"]

    # Selecting from the already-loaded list makes no further fetch.
    picker.set_value(["ETHUSDT — ETH/USDT"]).run()
    assert not at.exception
    assert fetches == [None]


def test_yahoo_symbols_picker_is_an_instant_dropdown_over_the_bundled_universe() -> (
    None
):
    """Yahoo has no downloadable "every symbol" endpoint like Binance, so it
    uses a bundled S&P 500 + ETF + major-global-company list instead — same
    one-widget, no-search-box shape as Binance, just backed by a static file
    instead of a live call."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    assert not at.exception
    assert not any(field.label == "Search" for field in at.sidebar.text_input)
    picker = _sidebar_multiselect(at, "Symbols")
    # The bundled S&P 500 + ETF + global-company list, not just a handful.
    assert len(picker.options) > 650
    assert "AAPL — Apple Inc." in picker.options
    assert "MSFT — Microsoft" in picker.options
    assert set(picker.value) == {
        "SPY — SPDR S&P 500 ETF Trust",
        "QQQ — Invesco QQQ Trust (Nasdaq-100)",
        "TLT — iShares 20+ Year Treasury Bond ETF",
        "GLD — SPDR Gold Shares",
    }

    # Picking from the already-loaded list is a plain client-side selection.
    picker.set_value([*picker.value, "AAPL — Apple Inc."]).run()
    assert not at.exception
    assert "AAPL — Apple Inc." in picker.value


def test_yahoo_symbols_picker_includes_major_non_us_companies() -> None:
    """Regression test: BMW (a DAX constituent, not S&P 500) must be
    findable — the bundled list isn't limited to US large caps."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    picker = _sidebar_multiselect(at, "Symbols")
    assert any(option.startswith("BMW.DE") for option in picker.options)


def test_yahoo_symbols_picker_accepts_a_symbol_outside_the_bundled_list() -> None:
    """Yahoo's bundled list is large but not exhaustive (no such thing
    exists for Yahoo, unlike Binance's complete pair list), so an exact
    symbol not in it can still be typed and added directly."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    picker = _sidebar_multiselect(at, "Symbols")
    assert picker.proto.accept_new_options is True
    assert not any(o.startswith("NOVN.SW") for o in picker.options)

    picker.set_value([*picker.value, "novn.sw"]).run()

    # The chip shows exactly what was typed; _symbols_picker() normalises
    # (uppercases) it only in the returned list used to build the backtest
    # config, not in the widget's own displayed value.
    assert not at.exception
    assert "novn.sw" in picker.value


def test_yahoo_shows_a_note_that_suggestions_are_not_exhaustive() -> None:
    """The caveat lives only in the help tooltip, not as a permanent
    on-screen caption — it's a corner case, not something worth
    dedicating always-visible space to."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    assert not any(
        "Not every symbol is suggested" in c.value for c in at.sidebar.caption
    )
    picker = _sidebar_multiselect(at, "Symbols")
    assert "Not every symbol is suggested" in picker.proto.help
    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert "Not every symbol is suggested" in benchmark.proto.help


def test_binance_shows_no_incomplete_suggestions_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binance's preloaded list is genuinely complete, so the "not every
    symbol is suggested" caveat (which only applies to Yahoo) must not
    appear for it, in the help text or otherwise."""
    from quantlab.data.base import SymbolSuggestion
    from quantlab.data.binance import BinanceDataSource

    monkeypatch.setattr(
        BinanceDataSource,
        "list_trading_symbols",
        lambda self: [SymbolSuggestion(symbol="BTCUSDT", description="BTC/USDT")],
    )
    st.cache_data.clear()  # avoid a real universe cached by an earlier test/run

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("binance").run()

    assert not any(
        "Not every symbol is suggested" in c.value for c in at.sidebar.caption
    )
    picker = _sidebar_multiselect(at, "Symbols")
    assert "Not every symbol is suggested" not in picker.proto.help
    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert "Not every symbol is suggested" not in benchmark.proto.help


def test_yahoo_benchmark_symbol_accepts_new_options() -> None:
    """AppTest can't simulate typing a brand-new selectbox value (it looks
    the value up by index into the fixed options list, which a freshly
    typed value isn't part of), so this only checks the widget is
    configured to accept one — verified end-to-end manually instead."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert benchmark.proto.accept_new_options is True


def test_binance_symbols_picker_rejects_symbols_outside_the_universe() -> None:
    """Unlike Yahoo, Binance's preloaded list is genuinely complete, so
    there is no free-typing escape hatch for it."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("binance").run()

    picker = _sidebar_multiselect(at, "Symbols")
    assert picker.proto.accept_new_options is False


def test_csv_benchmark_symbol_stays_free_text() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert any(field.label == "Benchmark symbol" for field in at.sidebar.text_input)
    assert not any(sb.label == "Benchmark symbol" for sb in at.sidebar.selectbox)


def test_binance_benchmark_symbol_is_a_dropdown_over_the_same_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.data.base import SymbolSuggestion
    from quantlab.data.binance import BinanceDataSource

    monkeypatch.setattr(
        BinanceDataSource,
        "list_trading_symbols",
        lambda self: [
            SymbolSuggestion(symbol="BTCUSDT", description="BTC/USDT"),
            SymbolSuggestion(symbol="ETHUSDT", description="ETH/USDT"),
        ],
    )
    st.cache_data.clear()  # avoid a real universe cached by an earlier test/run

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("binance").run()

    assert not at.exception
    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert benchmark.options == ["BTCUSDT — BTC/USDT", "ETHUSDT — ETH/USDT"]
    assert benchmark.value == "BTCUSDT — BTC/USDT"


def test_yahoo_benchmark_symbol_offers_the_full_bundled_universe() -> None:
    """The benchmark dropdown isn't limited to the 4 starter symbols — it
    shares the same bundled S&P 500 + ETF list as the Symbols picker."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("yahoo").run()

    assert not at.exception
    benchmark = _sidebar_selectbox(at, "Benchmark symbol")
    assert len(benchmark.options) > 500
    assert "AAPL — Apple Inc." in benchmark.options
    assert benchmark.value == "SPY — SPDR S&P 500 ETF Trust"


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

    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("AAA, BBB").run()
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
        _sidebar_text_input(at, "Symbols (comma-separated)").set_value("AAA, BBB").run()
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
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("AAA").run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()
    assert not at.exception
    assert any("at least two symbols" in e.value for e in at.sidebar.error)


def test_pairs_trading_requires_two_distinct_symbols() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("AAA, AAA").run()
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

    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("").run()
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
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("SPY, QQQ").run()
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


def test_advanced_data_settings_reveal_forward_fill_limit_only_when_relevant() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    policy = _sidebar_selectbox(at, "Missing value policy")
    assert policy.options == ["drop", "forward_fill", "raise", "none"]
    assert not any(
        field.label == "Forward-fill limit (consecutive bars)"
        for field in at.sidebar.number_input
    )

    policy.set_value("forward_fill").run()
    assert not at.exception
    assert any(
        field.label == "Forward-fill limit (consecutive bars)"
        for field in at.sidebar.number_input
    )


def test_advanced_data_settings_are_accepted_by_the_config() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _sidebar_selectbox(at, "Missing value policy").set_value("forward_fill").run()
    _sidebar_number_input(at, "Forward-fill limit (consecutive bars)").set_value(
        3
    ).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    result = at.session_state["result"]
    assert result.config.data.missing_value_policy == "forward_fill"
    assert result.config.data.forward_fill_limit == 3


def test_advanced_execution_settings_reveal_impact_coefficient_only_for_volume() -> (
    None
):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    model = _sidebar_selectbox(at, "Slippage model")
    assert model.options == ["constant", "volume"]
    assert not any(
        field.label == "Volume impact coefficient" for field in at.sidebar.number_input
    )

    model.set_value("volume").run()
    assert not at.exception
    assert any(
        field.label == "Volume impact coefficient" for field in at.sidebar.number_input
    )


def test_advanced_execution_settings_are_accepted_by_the_config() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _sidebar_selectbox(at, "Slippage model").set_value("volume").run()
    _sidebar_number_input(at, "Volume impact coefficient").set_value(0.25).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    result = at.session_state["result"]
    assert result.config.execution.slippage_model == "volume"
    assert result.config.execution.impact_coefficient == pytest.approx(0.25)


def test_advanced_portfolio_constraints_hidden_for_pairs_trading() -> None:
    """A minimum weight, position cap, or exposure cap could drop one leg
    and break the pair hedge, so these constraints must not even be offered."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)

    assert not any(
        c.label == "Enable minimum position size" for c in at.sidebar.checkbox
    )
    assert any(
        "Advanced portfolio constraints are disabled for pairs_trading" in c.value
        for c in at.sidebar.caption
    )


def test_advanced_portfolio_constraints_are_accepted_by_the_config() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value(
        "SPY, QQQ, TLT, GLD"
    ).run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()

    _sidebar_checkbox(at, "Enable minimum position size").set_value(True).run()
    _sidebar_checkbox(at, "Cap gross exposure").set_value(True).run()
    _sidebar_checkbox(at, "Cap net exposure").set_value(True).run()
    _sidebar_checkbox(at, "Cap number of positions").set_value(True).run()
    _sidebar_checkbox(at, "Cap turnover per rebalance").set_value(True).run()
    assert not at.exception

    at.sidebar.button[0].click().run()
    assert not at.exception
    assert not at.error
    portfolio = at.session_state["result"].config.portfolio
    assert portfolio.target_minimum_weight is not None
    assert portfolio.maximum_gross_exposure is not None
    assert portfolio.maximum_net_exposure is not None
    assert portfolio.target_maximum_positions is not None
    assert portfolio.maximum_turnover is not None


def test_gross_vs_net_section_renders_with_real_values() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    metrics = {m.label: m.value for m in at.tabs[0].metric}
    for label in (
        "Net total return",
        "Gross total return",
        "Cost drag",
        "Net Sharpe",
        "Gross Sharpe",
    ):
        assert label in metrics
        assert metrics[label] != "n/a"


def test_gross_vs_net_comparison_uses_a_five_column_grid() -> None:
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

        def markdown(self, text: str) -> None:
            pass

    result: Any = SimpleNamespace(
        gross_net_comparison=lambda: {
            "net_total_return": 0.05,
            "net_sharpe": 1.2,
            "total_cost": 123.0,
            "gross_sharpe": 1.4,
            "gross_total_return": 0.07,
            "cost_drag": 0.02,
        }
    )
    fake = FakeStreamlit()

    render_gross_net_comparison(fake, result)

    assert fake.columns_requested == 5
    values = {label: value for label, value, _ in fake.metrics}
    assert values["Net total return"] == "5.00%"
    assert values["Gross total return"] == "7.00%"
    assert values["Cost drag"] == "2.00%"
    assert values["Net Sharpe"] == "1.20"
    assert values["Gross Sharpe"] == "1.40"


def test_gross_vs_net_comparison_shows_na_for_missing_gross_fields() -> None:
    class FakeColumn:
        def __init__(self, sink: list[tuple[str, str, dict[str, object]]]) -> None:
            self._sink = sink

        def metric(self, label: str, value: str, **kwargs: object) -> None:
            self._sink.append((label, value, kwargs))

    class FakeStreamlit:
        def __init__(self) -> None:
            self.metrics: list[tuple[str, str, dict[str, object]]] = []

        def columns(self, n: int) -> list[FakeColumn]:
            return [FakeColumn(self.metrics) for _ in range(n)]

        def markdown(self, text: str) -> None:
            pass

    result: Any = SimpleNamespace(
        gross_net_comparison=lambda: {
            "net_total_return": 0.05,
            "net_sharpe": 1.2,
            "total_cost": 0.0,
        }
    )
    fake = FakeStreamlit()

    render_gross_net_comparison(fake, result)

    values = {label: value for label, value, _ in fake.metrics}
    assert values["Gross total return"] == "n/a"
    assert values["Gross Sharpe"] == "n/a"


def test_walk_forward_mode_reveals_dedicated_sidebar_controls() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert any(b.label == "Run backtest" for b in at.sidebar.button)
    assert any(
        c.label == "Chronological holdout (train / validation / test)"
        for c in at.sidebar.checkbox
    )

    _switch_to_walk_forward_mode(at)

    assert not at.exception
    assert any(b.label == "Run walk-forward" for b in at.sidebar.button)
    assert not any(b.label == "Run backtest" for b in at.sidebar.button)
    assert not any(
        c.label == "Chronological holdout (train / validation / test)"
        for c in at.sidebar.checkbox
    )
    assert _sidebar_number_input(at, "Train window (periods)").value == 500
    assert _sidebar_number_input(at, "Validation window (periods)").value == 126
    assert _sidebar_number_input(at, "Test window (periods)").value == 126
    assert any(ms.label == "Grid parameters" for ms in at.sidebar.multiselect)


def test_walk_forward_mode_hides_buy_and_hold_strategy() -> None:
    """buy_and_hold has no parameters, so it has nothing for walk-forward's
    fold-by-fold validation-block selection to select — offering it there
    would look like optimization is happening when it is not."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    strategy = _sidebar_selectbox(at, "Strategy")
    assert "buy_and_hold" in strategy.options

    _switch_to_walk_forward_mode(at)

    strategy = _sidebar_selectbox(at, "Strategy")
    assert "buy_and_hold" not in strategy.options
    # cross_sectional_momentum (index 0 of the filtered, sorted list) still
    # has parameters to select, so it is a meaningful default here.
    assert strategy.value == "cross_sectional_momentum"


def test_switching_to_walk_forward_with_buy_and_hold_selected_does_not_crash() -> None:
    """The Strategy widget's persisted value (buy_and_hold, from Backtest
    mode) is no longer in Walk-forward mode's filtered options list —
    Streamlit must fall back cleanly, not raise."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Strategy").set_value("buy_and_hold").run()
    assert not at.exception

    _switch_to_walk_forward_mode(at)

    assert not at.exception
    strategy = _sidebar_selectbox(at, "Strategy")
    assert strategy.value in strategy.options


def test_walk_forward_mode_has_no_static_backtest_count_estimate() -> None:
    """The static "Estimated ~N backtests" caption was replaced by a live
    progress bar shown while a run is actually in flight (see
    test_walk_forward_run_reports_fold_progress in test_validation.py for
    the underlying on_progress contract) — the sidebar no longer guesses a
    duration upfront."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_walk_forward(at)

    assert not at.exception
    assert not any("Estimated ~" in c.value for c in at.sidebar.caption)


def test_walk_forward_mode_still_warns_when_no_fold_fits() -> None:
    """The zero-fold pre-flight warning is a validity check, not a duration
    estimate, and must survive removing the "Estimated ~" caption above."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_walk_forward_mode(at)
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_number_input(at, "Train window (periods)").set_value(10_000).run()

    assert not at.exception
    assert any("No walk-forward fold fits" in w.value for w in at.sidebar.warning)


def test_walk_forward_grid_parameter_selection_adds_a_values_field() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_walk_forward(at)

    grid_picker = _sidebar_multiselect(at, "Grid parameters")
    assert "lookback_period" in grid_picker.options
    grid_picker.set_value(["lookback_period"]).run()

    assert not at.exception
    assert any(
        field.label == "Candidate values for lookback_period (comma-separated)"
        for field in at.sidebar.text_input
    )


def test_walk_forward_run_populates_oos_result_and_tabs() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    assert "wf_result" in at.session_state
    wf = at.session_state["wf_result"]
    assert len(wf.folds) >= 1
    assert wf.oos_result is not None

    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["Results", "Trades", "Robustness", "Report"]
    metrics = {m.label: m.value for m in at.tabs[0].metric}
    for label in ("Total return", "Sharpe", "Net total return", "Gross total return"):
        assert label in metrics
        assert metrics[label] != "n/a"


def test_walk_forward_robustness_tab_shows_fold_table_and_stability() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    _sidebar_multiselect(at, "Grid parameters").set_value(["lookback_period"]).run()
    _sidebar_text_input(
        at, "Candidate values for lookback_period (comma-separated)"
    ).set_value("10, 20").run()
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()

    assert not at.exception
    robustness_tab = at.tabs[2]
    assert robustness_tab.label == "Robustness"
    fold_table = robustness_tab.dataframe[0].value
    assert "param_lookback_period" in fold_table.columns
    assert "test_sharpe" in fold_table.columns
    # Stress tests do appear here, but must be the walk-forward-aware
    # variant under the hood — verified at the backend level by
    # test_run_walk_forward_stress_tests_reselects_parameters_under_higher_costs
    # in test_validation.py, not by this dashboard-rendering test.
    assert any(b.label == "Run stress tests" for b in robustness_tab.button)


def test_walk_forward_report_tab_includes_fold_evidence() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    at.session_state["dashboard_active_tab"] = "Report"
    at.sidebar.button[0].click().run()

    assert not at.exception
    report_tab = at.tabs[3]
    assert report_tab.label == "Report"
    _, (html, _warnings) = at.session_state["wf_report_html"]
    assert "walk" in html.lower()


def test_walk_forward_no_fitting_fold_shows_error_not_crash() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_walk_forward_mode(at)
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_number_input(at, "Train window (periods)").set_value(10_000).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert any("No walk-forward fold fit" in e.value for e in at.error)
    assert "wf_result" not in at.session_state


def test_switching_to_backtest_mode_does_not_affect_a_stored_walk_forward_result() -> (
    None
):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert "wf_result" in at.session_state

    at.segmented_control[0].set_value("Backtest").run()

    assert not at.exception
    assert "wf_result" in at.session_state
    assert any(b.label == "Run backtest" for b in at.sidebar.button)
    assert any(i.value.startswith("Configure an experiment") for i in at.info)


def _configure_offline_mean_reversion(at: AppTest) -> AppTest:
    """A strategy with real grid-eligible parameters, for sensitivity tests."""
    _sidebar_selectbox(at, "Data source").set_value("csv").run()
    _sidebar_text_input(at, "Symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("mean_reversion").run()
    return at


def test_backtest_robustness_tab_bootstrap_runs_and_displays() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    n_iterations = next(
        n for n in at.tabs[2].number_input if n.label == "Bootstrap iterations"
    )
    n_iterations.set_value(100).run()
    bootstrap_button = next(b for b in at.tabs[2].button if b.label == "Run bootstrap")
    bootstrap_button.click().run()

    assert not at.exception
    assert "bootstrap_summary" in at.session_state
    summary = at.session_state["bootstrap_summary"]
    assert set(summary["statistic"]) == {
        "cagr",
        "sharpe",
        "max_drawdown",
        "final_value",
    }
    assert len(at.tabs[2].dataframe) >= 1


def test_backtest_robustness_tab_permutation_test_runs_and_displays() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    n_iterations = next(
        n for n in at.tabs[2].number_input if n.label == "Permutation iterations"
    )
    n_iterations.set_value(100).run()
    permutation_button = next(
        b for b in at.tabs[2].button if b.label == "Run permutation test"
    )
    permutation_button.click().run()

    assert not at.exception
    assert "permutation_test" in at.session_state
    outcome = at.session_state["permutation_test"]
    assert {"real_sharpe", "p_value", "n_iterations"} <= set(outcome)
    metric_labels = {m.label for m in at.tabs[2].metric}
    assert {"Real Sharpe", "p-value"} <= metric_labels


def test_backtest_robustness_tab_sensitivity_runs_and_displays_heatmap() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_mean_reversion(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    robustness_tab = at.tabs[2]
    x_select = next(
        sb for sb in robustness_tab.selectbox if sb.label == "Parameter (x-axis)"
    )
    x_select.set_value("lookback_period").run()
    robustness_tab = at.tabs[2]
    x_values = next(
        f
        for f in robustness_tab.text_input
        if f.label == "Candidate values (x, comma-separated)"
    )
    x_values.set_value("10, 20").run()
    robustness_tab = at.tabs[2]
    y_select = next(
        sb for sb in robustness_tab.selectbox if sb.label == "Parameter (y-axis)"
    )
    y_select.set_value("entry_zscore").run()
    robustness_tab = at.tabs[2]
    y_values = next(
        f
        for f in robustness_tab.text_input
        if f.label == "Candidate values (y, comma-separated)"
    )
    y_values.set_value("1.5, 2.5").run()

    robustness_tab = at.tabs[2]
    run_button = next(
        b for b in robustness_tab.button if b.label == "Run parameter sensitivity"
    )
    assert not run_button.proto.disabled
    run_button.click().run()

    assert not at.exception
    assert "sensitivity" in at.session_state
    sensitivity = at.session_state["sensitivity"]
    assert len(sensitivity) == 4
    # AppTest has no plotly_chart accessor; render_sensitivity_heatmap also
    # renders the raw sensitivity table right after the chart, so its
    # presence confirms the heatmap section actually rendered.
    assert any(len(df.value) == 4 for df in at.tabs[2].dataframe)


def test_walk_forward_robustness_tab_bootstrap_runs_and_displays() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    n_iterations = next(
        n for n in at.tabs[2].number_input if n.label == "Bootstrap iterations"
    )
    n_iterations.set_value(100).run()
    bootstrap_button = next(b for b in at.tabs[2].button if b.label == "Run bootstrap")
    bootstrap_button.click().run()

    assert not at.exception
    assert "wf_bootstrap_summary" in at.session_state
    summary = at.session_state["wf_bootstrap_summary"]
    assert set(summary["statistic"]) == {
        "cagr",
        "sharpe",
        "max_drawdown",
        "final_value",
    }


def test_walk_forward_robustness_tab_permutation_test_runs_and_displays() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    n_iterations = next(
        n for n in at.tabs[2].number_input if n.label == "Permutation iterations"
    )
    n_iterations.set_value(100).run()
    permutation_button = next(
        b for b in at.tabs[2].button if b.label == "Run permutation test"
    )
    permutation_button.click().run()

    assert not at.exception
    assert "wf_permutation_test" in at.session_state


def test_walk_forward_robustness_tab_stress_tests_reruns_selection() -> None:
    """Confirms the button in Walk-forward mode actually calls the
    walk-forward-aware state function (distinct session key, real scenarios)
    rather than Backtest mode's plain-backtest variant."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _configure_offline_walk_forward(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    stress_button = next(b for b in at.tabs[2].button if b.label == "Run stress tests")
    stress_button.click().run()

    assert not at.exception
    assert "wf_stress_tests" in at.session_state
    assert "stress_tests" not in at.session_state  # Backtest mode's own key
    stress = at.session_state["wf_stress_tests"]
    assert "commission x5" in set(stress["scenario"])


def test_backtest_run_all_robustness_tests_populates_every_technique() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_mean_reversion(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    at.sidebar.button[0].click().run()
    assert not at.exception

    robustness_tab = at.tabs[2]
    next(
        n for n in robustness_tab.number_input if n.label == "Bootstrap iterations"
    ).set_value(100).run()
    robustness_tab = at.tabs[2]
    next(
        n for n in robustness_tab.number_input if n.label == "Permutation iterations"
    ).set_value(100).run()
    robustness_tab = at.tabs[2]
    next(
        sb for sb in robustness_tab.selectbox if sb.label == "Parameter (x-axis)"
    ).set_value("lookback_period").run()
    robustness_tab = at.tabs[2]
    next(
        f
        for f in robustness_tab.text_input
        if f.label == "Candidate values (x, comma-separated)"
    ).set_value("10, 20").run()
    robustness_tab = at.tabs[2]
    next(
        sb for sb in robustness_tab.selectbox if sb.label == "Parameter (y-axis)"
    ).set_value("entry_zscore").run()
    robustness_tab = at.tabs[2]
    next(
        f
        for f in robustness_tab.text_input
        if f.label == "Candidate values (y, comma-separated)"
    ).set_value("1.5, 2.5").run()

    robustness_tab = at.tabs[2]
    run_all_button = next(
        b for b in robustness_tab.button if b.label == "Run all robustness tests"
    )
    run_all_button.click().run()

    assert not at.exception
    for key in ("stress_tests", "bootstrap_summary", "permutation_test", "sensitivity"):
        assert key in at.session_state, f"{key} was not populated by Run all"
