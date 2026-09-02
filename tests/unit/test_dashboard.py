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

import numpy as np
import pandas as pd
import pytest

from quantlab.dashboard.components import (
    _is_missing,
    _monthly_return_pivot,
    render_gross_net_comparison,
    render_metric_cards,
    render_pair_diagnostics,
    render_trade_table,
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


def _sidebar_multiselect(at: AppTest, label: str) -> Any:
    return next(ms for ms in at.sidebar.multiselect if ms.label == label)


def _sidebar_multiselect_by_key(at: AppTest, key: str) -> Any:
    """Yahoo's and Binance's symbol pickers share the label "Symbols" (both
    built from the shared ``_symbols_picker`` helper in app.py), so they
    must be told apart by widget key ("yahoo_symbols" / "binance_symbols")
    instead of by label."""
    return next(ms for ms in at.sidebar.multiselect if ms.key == key)


def _run_button(at: AppTest) -> Any:
    """The "Run backtest"/"Run walk-forward" button, found by its stable
    key rather than by position -- the sidebar can render OTHER buttons
    before it (e.g. "Load Binance symbols", gated behind its own explicit
    click -- see `_load_binance_universe`), so it is not reliably
    ``at.sidebar.button[0]``."""
    return next(b for b in at.sidebar.button if b.key == "run_button")


def _load_binance_universe(at: AppTest) -> AppTest:
    """Click "Load Binance symbols" so the ``binance_symbols`` multiselect
    actually exists -- fetching Binance's universe is gated behind this
    explicit button precisely so it is NOT called on every dashboard load
    (see `_binance_symbols_picker` in app.py)."""
    next(
        b for b in at.sidebar.button if b.key == "binance_universe_load_button"
    ).click().run()
    return at


def _configure_offline_pairs_trade(at: AppTest) -> AppTest:
    """Point the dashboard at locally cached CSV data for SPY/QQQ."""
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
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
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
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


def test_symbols_picker_help_icon_is_not_hidden_by_a_collapsed_label() -> None:
    """Regression test: Streamlit hides a widget's help tooltip icon along
    with a `label_visibility="collapsed"` label, which would silently
    swallow the incomplete-list note in `_symbols_picker`'s help text. The
    label must stay visible so the help icon (and thus that note) stays
    reachable."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    picker = _sidebar_multiselect_by_key(at, "yahoo_symbols")
    assert picker.proto.label_visibility.value == 0  # VISIBLE


def test_csv_symbols_field_is_free_text_not_a_dropdown() -> None:
    """CSV symbols are entered as free text (a local filename), unlike the
    always-visible Yahoo/Binance pickers, which are dropdowns over a
    preloaded universe."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert any(
        field.label == "CSV symbols (comma-separated)"
        for field in at.sidebar.text_input
    )
    assert not any(ms.key == "csv_symbols" for ms in at.sidebar.multiselect)


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

    assert not at.exception
    # Not fetched on initial load -- see _load_binance_universe's docstring
    # and test_dashboard_initial_load_never_calls_binance below.
    assert fetches == []
    _load_binance_universe(at)
    assert fetches == [None]

    assert not any(field.label == "Search" for field in at.sidebar.text_input)
    picker = _sidebar_multiselect_by_key(at, "binance_symbols")
    assert picker.options == ["BTCUSDT — BTC/USDT", "ETHUSDT — ETH/USDT"]
    # Empty by default: a non-empty default would immediately conflict with
    # CSV's own bundled-demo default (see `_combine_instrument_picks`).
    assert picker.value == []

    # Selecting from the already-loaded list makes no further fetch.
    picker.set_value(["ETHUSDT — ETH/USDT"]).run()
    assert not at.exception
    assert fetches == [None]


def test_dashboard_initial_load_never_calls_binance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: a plain `st.expander` still runs its body every
    rerun even while collapsed, so the "Binance" section previously called
    Binance's real API on every dashboard load -- including a load that
    immediately switches to Strategies mode, which never touches the
    (hidden) sidebar at all. Fetching Binance's universe must only ever
    happen after the explicit "Load Binance symbols" button is clicked."""
    from quantlab.data.base import SymbolSuggestion
    from quantlab.data.binance import BinanceDataSource

    fetches: list[None] = []

    def fake_universe(self: BinanceDataSource) -> list[SymbolSuggestion]:
        fetches.append(None)
        return [SymbolSuggestion(symbol="BTCUSDT", description="BTC/USDT")]

    monkeypatch.setattr(BinanceDataSource, "list_trading_symbols", fake_universe)
    st.cache_data.clear()  # avoid a real universe cached by an earlier test/run

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert fetches == []

    _switch_to_strategies_mode(at)
    assert not at.exception
    assert fetches == []


def test_yahoo_symbols_picker_is_an_instant_dropdown_over_the_bundled_universe() -> (
    None
):
    """Yahoo has no downloadable "every symbol" endpoint like Binance, so it
    uses a bundled S&P 500 + ETF + major-global-company list instead — same
    one-widget, no-search-box shape as Binance, just backed by a static file
    instead of a live call."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception
    assert not any(field.label == "Search" for field in at.sidebar.text_input)
    picker = _sidebar_multiselect_by_key(at, "yahoo_symbols")
    # The bundled S&P 500 + ETF + global-company list, not just a handful.
    assert len(picker.options) > 650
    assert "AAPL — Apple Inc." in picker.options
    assert "MSFT — Microsoft" in picker.options
    # Empty by default: a non-empty default would immediately conflict with
    # CSV's own bundled-demo default (see `_combine_instrument_picks`).
    assert picker.value == []

    # Picking from the already-loaded list is a plain client-side selection.
    picker.set_value(["AAPL — Apple Inc."]).run()
    assert not at.exception
    assert "AAPL — Apple Inc." in picker.value


def test_yahoo_symbols_picker_includes_major_non_us_companies() -> None:
    """Regression test: BMW (a DAX constituent, not S&P 500) must be
    findable — the bundled list isn't limited to US large caps."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    picker = _sidebar_multiselect_by_key(at, "yahoo_symbols")
    assert any(option.startswith("BMW.DE") for option in picker.options)


def test_yahoo_symbols_picker_accepts_a_symbol_outside_the_bundled_list() -> None:
    """Yahoo's bundled list is large but not exhaustive (no such thing
    exists for Yahoo, unlike Binance's complete pair list), so an exact
    symbol not in it can still be typed and added directly."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    picker = _sidebar_multiselect_by_key(at, "yahoo_symbols")
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

    assert not any(
        "Not every symbol is suggested" in c.value for c in at.sidebar.caption
    )
    picker = _sidebar_multiselect_by_key(at, "yahoo_symbols")
    assert "Not every symbol is suggested" in picker.proto.help


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

    assert not any(
        "Not every symbol is suggested" in c.value for c in at.sidebar.caption
    )
    _load_binance_universe(at)
    picker = _sidebar_multiselect_by_key(at, "binance_symbols")
    assert "Not every symbol is suggested" not in picker.proto.help


def test_binance_symbols_picker_rejects_symbols_outside_the_universe() -> None:
    """Unlike Yahoo, Binance's preloaded list is genuinely complete, so
    there is no free-typing escape hatch for it."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    _load_binance_universe(at)
    picker = _sidebar_multiselect_by_key(at, "binance_symbols")
    assert picker.proto.accept_new_options is False


def test_monthly_return_pivot_preserves_a_month_without_observations() -> None:
    returns = pd.Series(
        [0.10, -0.05],
        index=pd.to_datetime(["2024-01-31", "2024-03-31"]),
    )

    pivot = _monthly_return_pivot(returns)

    assert pivot.loc[2024, 1] == pytest.approx(0.10)
    assert pd.isna(pivot.loc[2024, 2])
    assert pivot.loc[2024, 3] == pytest.approx(-0.05)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        (pd.NaT, True),
        (pd.NA, True),
        (float("nan"), True),
        (np.float64("nan"), True),
        (np.float32("nan"), True),
        (np.float16("nan"), True),
        ("some string", False),
        ("trigger+exit", False),
        (0.0, False),
        (0, False),
        (["a", "b"], False),
    ],
)
def test_is_missing_handles_every_missing_scalar_type(
    value: object, expected: bool
) -> None:
    """A prior `isinstance(value, float) and pd.isna(value)` implementation
    missed `pd.NA` (not a `float` subclass) and non-float64 NaN scalars
    like `numpy.float32('nan')` (only `numpy.float64` is itself a `float`
    subclass, via CPython's numpy integration) -- either would then reach
    `parse_adjustment_codes(str(value))` as a bogus `"<NA>"`/`"nan"` code
    and raise `BacktestError` instead of being treated as missing."""
    assert _is_missing(value) is expected


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

    # 4 columns: eight cards divide evenly into two full, aligned rows --
    # any count that doesn't evenly divide 8 leaves a ragged last row (e.g.
    # 3 columns strands "Total costs"/"Number of fills" alone on a
    # half-empty row, visually detached from the grid above).
    assert fake.columns_requested == 4
    assert len(fake.metrics) == 8
    costs = next(metric for metric in fake.metrics if metric[0] == "Total costs")
    assert costs[1] == "1,234"
    assert "currency units" in str(costs[2]["help"])


class _FakeColumnConfig:
    # Names mirror st.column_config's actual (PascalCase) API.
    def TextColumn(self, *args: object, **kwargs: object) -> dict[str, object]:  # noqa: N802
        return {"args": args, "kwargs": kwargs}

    def NumberColumn(self, *args: object, **kwargs: object) -> dict[str, object]:  # noqa: N802
        return {"args": args, "kwargs": kwargs}

    def DateColumn(self, *args: object, **kwargs: object) -> dict[str, object]:  # noqa: N802
        return {"args": args, "kwargs": kwargs}

    def DatetimeColumn(self, *args: object, **kwargs: object) -> dict[str, object]:  # noqa: N802
        return {"args": args, "kwargs": kwargs}


class _FakeStreamlit:
    def __init__(self) -> None:
        self.dataframe_calls: list[tuple[pd.DataFrame, dict[str, object]]] = []
        self.caption_calls: list[tuple[object, dict[str, object]]] = []
        self.column_config = _FakeColumnConfig()

    def info(self, *args: object, **kwargs: object) -> None:
        pass

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.dataframe_calls.append((frame, kwargs))

    def caption(self, text: object, **kwargs: object) -> None:
        self.caption_calls.append((text, kwargs))

    def download_button(self, *args: object, **kwargs: object) -> None:
        pass


def _trade_row(**overrides: object) -> dict[str, object]:
    """One full 21-field trade-log record, defaulted to a plain,
    unattributed fill -- callers override only the fields their scenario
    cares about."""
    row: dict[str, object] = {
        "timestamp": pd.Timestamp("2024-01-02"),
        "symbol": "AAA",
        "previous_weight": 0.0,
        "new_weight": 0.5,
        "weight_change": 0.5,
        "side": "buy",
        "action": "entry_long",
        "trigger_reason_code": None,
        "trigger_reason_detail_code": None,
        "trigger_reason_details": None,
        "adjustment_reason_codes": None,
        "adjustment_reason_details": None,
        "position_strategy_origin_timestamp": pd.NaT,
        "position_strategy_origin_code": None,
        "position_strategy_origin_details": None,
        "reference_price": 100.0,
        "traded_notional": 1000.0,
        "commission": 0.5,
        "spread_cost": 0.3,
        "slippage_cost": 0.2,
        "total_cost": 1.0,
    }
    row.update(overrides)
    return row


def test_trade_table_shows_separate_columns_when_no_row_has_a_detail_code() -> None:
    """Regression test: when every trigger_reason_detail_code is None (no
    strategy sub-code fired on this run), pandas can infer that all-None
    column as plain "object" while trigger_reason_code (real string
    values) infers its new "str" extension dtype -- assigning into the
    detail-code column without first casting it to plain "object" used
    to raise/silently corrupt data. Every trigger/adjustment/position-
    origin value is shown separate and unmodified (no Python-side string
    concatenation into a single column, no cell blanked for
    "compactness") -- each must stay exactly what build_trade_log
    produced, just under its human-readable display column."""
    trades = pd.DataFrame.from_records(
        [
            _trade_row(
                trigger_reason_code="strategy_signal",
                trigger_reason_details="signal 0.0000 -> 1.0000 since last rebalance",
                position_strategy_origin_timestamp=pd.Timestamp("2024-01-02"),
            )
        ]
    )
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(experiment_name="test", strategy_parameters={}),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    displayed = fake.dataframe_calls[0][0]
    assert displayed["Trigger"].tolist() == ["strategy_signal"]
    assert displayed["Trigger detail"].tolist() == [None]
    assert displayed["Adjustments"].tolist() == [None]
    assert displayed["Details"].tolist() == [
        "Trigger: signal 0.0000 -> 1.0000 since last rebalance"
    ]
    # Position origin is always present with its raw value, never blanked.
    assert displayed["Position origin date"].tolist() == [pd.Timestamp("2024-01-02")]


def test_trade_table_adjustment_codes_display_spaces_out_the_plus_join() -> None:
    """adjustment_reason_codes keeps its compact "+" form in the
    underlying data (see trade_log.serialize_adjustment_codes) but is
    reformatted with spaces for on-screen readability in the Adjustments
    column -- every other value, including Trigger, must be untouched
    (no "code (detail)" fusion)."""
    trades = pd.DataFrame.from_records(
        [
            _trade_row(
                action="increase_long",
                adjustment_reason_codes="maximum_weight+maximum_gross_exposure",
                adjustment_reason_details=(
                    "maximum_weight: 0.7 -> 0.5; maximum_gross_exposure: 1.0 -> 0.8"
                ),
            )
        ]
    )
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(experiment_name="test", strategy_parameters={}),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    displayed = fake.dataframe_calls[0][0]
    assert displayed["Trigger"].tolist() == [None]
    assert displayed["Adjustments"].tolist() == [
        "maximum_weight + maximum_gross_exposure"
    ]
    assert displayed["Details"].tolist() == [
        "Adjustment: maximum_weight: 0.7 -> 0.5; maximum_gross_exposure: 1.0 -> 0.8"
    ]
    # The underlying result.trades DataFrame itself must stay untouched.
    assert trades["adjustment_reason_codes"].tolist() == [
        "maximum_weight+maximum_gross_exposure"
    ]


def test_trade_table_display_columns_are_identical_across_strategies() -> None:
    """The visible table's column set and order must never depend on
    which strategy produced the result -- only the VALUES may differ.
    Simulates two very different strategies' trade logs (one with a
    constraint adjustment and a real position origin, one bare) and
    asserts both render the exact same 15 columns, in the exact same
    order (`_TRADE_TABLE_DISPLAY_COLUMNS`)."""
    from quantlab.dashboard.components import _TRADE_TABLE_DISPLAY_COLUMNS

    trend_following_trades = pd.DataFrame.from_records(
        [
            _trade_row(
                symbol="SPY",
                action="entry_long",
                trigger_reason_code="strategy_signal",
                trigger_reason_detail_code="bullish_crossover",
                position_strategy_origin_timestamp=pd.Timestamp("2024-01-02"),
                position_strategy_origin_code="bullish_crossover",
            )
        ]
    )
    buy_and_hold_trades = pd.DataFrame.from_records(
        [
            _trade_row(
                symbol="QQQ",
                action="entry_long",
                trigger_reason_code="strategy_signal",
                trigger_reason_detail_code="price_became_available",
                adjustment_reason_codes="maximum_weight",
                adjustment_reason_details="maximum_weight: 0.9 -> 0.6",
                position_strategy_origin_timestamp=pd.Timestamp("2024-01-02"),
                position_strategy_origin_code="price_became_available",
            )
        ]
    )

    displayed_columns = []
    for trades in (trend_following_trades, buy_and_hold_trades):
        result: Any = SimpleNamespace(
            trades=trades,
            config=SimpleNamespace(experiment_name="test", strategy_parameters={}),
        )
        fake = _FakeStreamlit()
        render_trade_table(fake, result)
        displayed_columns.append(list(fake.dataframe_calls[0][0].columns))

    assert displayed_columns[0] == displayed_columns[1] == _TRADE_TABLE_DISPLAY_COLUMNS


def test_trade_table_empty_position_origin_means_no_origin_not_masking() -> None:
    """A blank `Position origin` cell must mean only one thing -- no
    strategic position is currently active (decision_proxy is flat) --
    never "hidden because it duplicates Trigger". A row whose Trigger
    equals a real value but whose position_strategy_origin_code is
    genuinely None (e.g. a walk-forward result with no attribution, or a
    real exit-to-flat trade) must still render the Position origin
    column, blank, never dropped."""
    trades = pd.DataFrame.from_records(
        [
            _trade_row(
                action="exit_long",
                trigger_reason_code="strategy_signal",
                trigger_reason_detail_code="mean_reversion_exit",
                position_strategy_origin_timestamp=pd.NaT,
                position_strategy_origin_code=None,
            )
        ]
    )
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(experiment_name="test", strategy_parameters={}),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    displayed = fake.dataframe_calls[0][0]
    assert "Position origin" in displayed.columns
    assert "Position origin date" in displayed.columns
    assert displayed["Position origin"].tolist() == [None]
    assert pd.isna(displayed["Position origin date"].iloc[0])
    # Trigger, on the same row, is a real value -- confirms the blank
    # Position origin is not a side effect of an otherwise-empty row.
    assert displayed["Trigger"].tolist() == ["strategy_signal"]


def test_trade_table_csv_export_always_has_the_full_21_column_schema() -> None:
    """The CSV download must always use the raw `result.trades` frame --
    its schema must never depend on, or be narrowed by, the display
    view above it."""
    from quantlab.backtesting.trade_log import TRADE_LOG_COLUMNS

    trades = pd.DataFrame.from_records([_trade_row()])
    assert list(trades.columns) == TRADE_LOG_COLUMNS
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(experiment_name="test", strategy_parameters={}),
    )
    fake = _FakeStreamlit()

    exported: dict[str, object] = {}

    def _capture_download_button(*args: object, **kwargs: object) -> None:
        exported["csv_bytes"] = args[1] if len(args) > 1 else kwargs.get("data")

    fake.download_button = _capture_download_button  # type: ignore[method-assign]

    render_trade_table(fake, result)

    assert "csv_bytes" in exported
    exported_csv = cast(bytes, exported["csv_bytes"]).decode("utf-8")
    header = exported_csv.splitlines()[0]
    assert header.split(",") == TRADE_LOG_COLUMNS


def test_trade_table_stop_loss_take_profit_caption_shown_even_with_zero_triggers() -> (
    None
):
    """Both thresholds configured but neither ever fired (0 triggers) must
    still show the caption -- a configured-but-never-triggered stop/target
    is meaningful information, not the same as "not configured at all"."""
    trades = pd.DataFrame.from_records([_trade_row()])
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(
            experiment_name="test",
            strategy_parameters={"stop_loss_pct": 0.1, "take_profit_pct": 0.2},
        ),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    assert len(fake.caption_calls) == 1
    text = str(fake.caption_calls[0][0])
    assert "Stop-loss affected" in text
    assert "take-profit affected" in text


def test_trade_table_caption_only_mentions_stop_loss_when_take_profit_disabled() -> (
    None
):
    """Enabling only stop_loss_pct must produce a caption about stop-loss
    alone -- it must never imply take-profit was also active at 0."""
    trades = pd.DataFrame.from_records([_trade_row()])
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(
            experiment_name="test",
            strategy_parameters={"stop_loss_pct": 0.1, "take_profit_pct": None},
        ),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    assert len(fake.caption_calls) == 1
    text = str(fake.caption_calls[0][0])
    # The trailing counting-convention caveat generically names both terms
    # regardless of which is configured -- only the leading clause (before
    # it) must be scoped to what's actually enabled.
    leading_clause = text.split(" -- counted", 1)[0]
    assert "Stop-loss affected" in leading_clause
    assert "take-profit" not in leading_clause.lower()


def test_trade_table_caption_only_mentions_take_profit_when_stop_loss_disabled() -> (
    None
):
    """Symmetric case: only take_profit_pct configured -- the caption must
    lead with "Take-profit" (capitalized, since it is now the first
    clause) and never mention stop-loss."""
    trades = pd.DataFrame.from_records([_trade_row()])
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(
            experiment_name="test",
            strategy_parameters={"stop_loss_pct": None, "take_profit_pct": 0.2},
        ),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    assert len(fake.caption_calls) == 1
    text = str(fake.caption_calls[0][0])
    leading_clause = text.split(" -- counted", 1)[0]
    assert text.startswith("Take-profit affected")
    assert "stop-loss" not in leading_clause.lower()


def test_trade_table_no_caption_when_neither_stop_loss_nor_take_profit_configured() -> (
    None
):
    """Neither threshold configured -- no caption at all, matching a
    strategy that never declared either parameter."""
    trades = pd.DataFrame.from_records([_trade_row()])
    result: Any = SimpleNamespace(
        trades=trades,
        config=SimpleNamespace(experiment_name="test", strategy_parameters={}),
    )
    fake = _FakeStreamlit()

    render_trade_table(fake, result)

    assert fake.caption_calls == []


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
    ],
)
def test_allocator_options_match_cross_sectional_momentum_scaling(
    signal_scaling: str, expected_options: list[str]
) -> None:
    """Regression test: the dashboard must mirror ExperimentConfig's own
    validator (config.py's "Non-binary cross_sectional_momentum signals
    require an allocator that preserves signal magnitude; equal_weight
    keeps only signs") -- equal_weight was newly reachable in the UI for
    'continuous' scaling even though the backend has always rejected it,
    failing only after "Run backtest" instead of narrowing the choice
    up front, exactly like the sibling time_series_momentum case already
    does."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_selectbox(at, "Strategy").set_value("cross_sectional_momentum").run()
    _sidebar_selectbox(at, "Signal scaling").set_value(signal_scaling).run()
    assert not at.exception

    allocator = _sidebar_selectbox(at, "Allocator")
    assert allocator.options == expected_options
    assert allocator.value in expected_options


def test_pairs_trading_symbol_inputs_render_without_crash() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("AAA, BBB").run()
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
    _run_button(at).click().run()

    assert not at.exception
    assert not at.error
    assert at.session_state["result"].config.data.use_bundled_demo_data is True


def test_bundled_demo_toggle_visible_only_while_a_csv_instrument_exists() -> None:
    """Shown whenever any instrument-table row's Source is "csv" — not
    gated by a "Data source" selectbox, since all three pickers are always
    visible now."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    # CSV's own default population (SPY, QQQ, TLT, GLD) means the toggle is
    # visible out of the box.
    assert any(
        toggle.label == "Allow bundled synthetic demo data"
        for toggle in at.sidebar.toggle
    )

    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("").run()
    yahoo_ms = _sidebar_multiselect_by_key(at, "yahoo_symbols")
    yahoo_ms.set_value([yahoo_ms.options[0]]).run()

    assert not at.exception
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
    assert not any(slider.label == "Entry threshold" for slider in at.sidebar.slider)


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
def test_reversion_exit_slider_stays_strictly_below_entry(
    strategy_name: str,
) -> None:
    """Both mean_reversion and pairs_trading use the generic 'Entry/Exit
    threshold' labels -- both now support the same zscore/rsi/percentile
    indicator choice, not only a z-score."""
    entry_label, exit_label = "Entry threshold", "Exit threshold"
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
            "AAA, BBB"
        ).run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()

    entry = next(slider for slider in at.sidebar.slider if slider.label == entry_label)
    entry.set_value(1.0).run()
    exit_ = next(slider for slider in at.sidebar.slider if slider.label == exit_label)

    exit_max = exit_.proto.max
    entry_value = cast(float, entry.value)
    assert exit_max == pytest.approx(0.9)
    assert exit_max < entry_value


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
@pytest.mark.parametrize("indicator", ["rsi", "percentile"])
def test_entry_at_max_never_crashes_the_stop_slider(
    strategy_name: str, indicator: str
) -> None:
    """For rsi/percentile, entry_threshold's own slider max equals
    stop_threshold's max -- dragging entry all the way up must not push the
    stop slider's min (entry + step) past its own max and crash it."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
            "AAA, BBB"
        ).run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()
    _sidebar_selectbox(at, "Indicator").set_value(indicator).run()

    entry = next(
        slider for slider in at.sidebar.slider if slider.label == "Entry threshold"
    )
    entry.set_value(entry.proto.max).run()

    assert not at.exception
    stop = next(
        slider for slider in at.sidebar.slider if slider.label == "Stop threshold"
    )
    # Streamlit's own slider requires min STRICTLY less than max (min ==
    # max raises too) -- the off-by-one this test originally missed.
    assert stop.proto.min < stop.proto.max


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
@pytest.mark.parametrize("indicator", ["rsi", "percentile"])
def test_entry_at_min_never_crashes_the_exit_slider(
    strategy_name: str, indicator: str
) -> None:
    """Dragging entry all the way down to its own min must never crash --
    for `percentile`, entry's own min equals the exit slider's step, which
    pushes the exit slider's own max (entry - step) down to exactly 0.0:
    a degenerate range Streamlit's slider would reject (min == max), so
    the dashboard shows a fixed 0.0 caption instead of rendering a slider
    at all. For `rsi`, entry's min stays comfortably above `step`, so an
    ordinary, non-degenerate exit slider still renders."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
            "AAA, BBB"
        ).run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()
    _sidebar_selectbox(at, "Indicator").set_value(indicator).run()

    entry = next(
        slider for slider in at.sidebar.slider if slider.label == "Entry threshold"
    )
    entry.set_value(entry.proto.min).run()

    assert not at.exception
    exit_sliders = [s for s in at.sidebar.slider if s.label == "Exit threshold"]
    if indicator == "percentile":
        # Degenerate case: entry.min == step, so exit_threshold's only
        # valid value (0.0) is shown as a caption, never a slider.
        assert exit_sliders == []
        assert any(
            "Exit threshold: 0.0" in caption.value for caption in at.sidebar.caption
        )
    else:
        assert len(exit_sliders) == 1
        assert exit_sliders[0].proto.min < exit_sliders[0].proto.max


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
def test_entry_threshold_reaches_its_own_step_sized_minimum(
    strategy_name: str,
) -> None:
    """`percentile`'s own YAML/Python-valid minimum, 0.01 (== its step),
    must be reachable through the dashboard slider -- an earlier version
    unconditionally floored entry's own min at `2 * step` (0.02) to dodge
    the exit slider's degenerate near-zero case, silently making a valid
    configuration value unreachable through the UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
            "AAA, BBB"
        ).run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()
    _sidebar_selectbox(at, "Indicator").set_value("percentile").run()

    entry = next(
        slider for slider in at.sidebar.slider if slider.label == "Entry threshold"
    )
    assert entry.proto.min == pytest.approx(0.01)


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
@pytest.mark.parametrize(
    ("indicator", "full_max"), [("rsi", 50.0), ("percentile", 0.49)]
)
def test_entry_domain_keeps_its_full_range_when_stop_is_disabled(
    strategy_name: str, indicator: str, full_max: float
) -> None:
    """Entry threshold's own max must only be narrowed to protect the stop
    slider (min == entry + step must stay < stop_max) WHILE that slider
    actually exists -- with "Enable stop threshold" off, entry must keep
    its full original range (rsi up to 50, percentile up to 0.49), not
    silently lose reachable values guarding a slider that isn't rendered."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
            "AAA, BBB"
        ).run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()
    _sidebar_selectbox(at, "Indicator").set_value(indicator).run()
    stop_checkbox = next(
        box
        for box in at.sidebar.checkbox
        if box.label in ("Enable stop threshold", "stop_threshold enabled")
    )
    stop_checkbox.set_value(False).run()

    entry = next(
        slider for slider in at.sidebar.slider if slider.label == "Entry threshold"
    )
    assert entry.proto.max == pytest.approx(full_max)


@pytest.mark.parametrize("strategy_name", ["mean_reversion", "pairs_trading"])
def test_entry_domain_is_narrowed_when_stop_is_enabled(strategy_name: str) -> None:
    """The mirror of the above: with "Enable stop threshold" on (the
    default), entry's max IS narrowed below rsi's full 50 -- proving the
    restriction is actually conditional on the checkbox, not just always
    off."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if strategy_name == "pairs_trading":
        _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
            "AAA, BBB"
        ).run()
    _sidebar_selectbox(at, "Strategy").set_value(strategy_name).run()
    _sidebar_selectbox(at, "Indicator").set_value("rsi").run()

    entry = next(
        slider for slider in at.sidebar.slider if slider.label == "Entry threshold"
    )
    assert entry.proto.max == pytest.approx(48.0)


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
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("AAA").run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()
    assert not at.exception
    assert any("at least two symbols" in e.value for e in at.sidebar.error)


def test_pairs_trading_requires_two_distinct_symbols() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("AAA, AAA").run()
    _sidebar_selectbox(at, "Strategy").set_value("pairs_trading").run()

    assert not at.exception
    assert any("at least two symbols" in e.value for e in at.sidebar.error)


def test_dashboard_can_disable_volatility_targeting_and_set_risk_free_rate() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _sidebar_toggle(at, "Enable volatility targeting").set_value(False).run()
    _sidebar_number_input(at, "Risk-free rate (annual %)").set_value(3.5).run()
    _run_button(at).click().run()

    assert not at.exception
    assert not at.error
    result = at.session_state["result"]
    assert result.config.portfolio.target_volatility is None
    assert result.config.risk_free_rate == pytest.approx(0.035)


def test_failed_backtest_invalidates_previous_result() -> None:
    """Dropping to one CSV symbol while pairs_trading is still selected
    leaves `strategy_parameters` without symbol_a/symbol_b, which
    ExperimentConfig rejects — a config-validation failure, not an empty
    universe (which the Run button now disables outright, so it can no
    longer be used to reach this path)."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _run_button(at).click().run()
    assert "result" in at.session_state

    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY").run()
    assert _run_button(at).proto.disabled is False
    _run_button(at).click().run()

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
    _run_button(at).click().run()

    assert not at.exception
    assert not at.error
    assert "result" in at.session_state
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["Results", "Trades", "Robustness", "Report"]
    assert len(at.tabs[0].metric) > 0


def test_trades_tab_shows_action_and_separate_reason_columns() -> None:
    """The Trades tab must show action/trigger/adjustment/position-origin
    as separate, human-labelled columns in the fixed display order --
    never a bare buy/sell-only table, and never a fused "code (detail)"
    string (that silently dropped a detail code from this table's own
    native CSV export icon -- a real regression a user hit)."""
    from quantlab.dashboard.components import _TRADE_TABLE_DISPLAY_COLUMNS

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    # pairs_trading on this bundled synthetic (independent-random-walk) CSV
    # data never finds a tradeable spread -- use mean_reversion instead,
    # over a wide enough window to actually generate fills, so this test
    # exercises a populated table, not the empty-state "no trades" message.
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2020, 6, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("mean_reversion").run()
    _run_button(at).click().run()
    assert not at.exception

    at.session_state["dashboard_active_tab"] = "Trades"
    _run_button(at).click().run()
    assert not at.exception

    trades_tab = at.tabs[1]
    assert trades_tab.label == "Trades"
    assert len(trades_tab.dataframe) == 1
    table = trades_tab.dataframe[0].value
    # Fixed schema: same 15 columns, same order, regardless of strategy.
    assert list(table.columns) == _TRADE_TABLE_DISPLAY_COLUMNS
    assert len(table) > 0
    # Display labels are the DataFrame's own column names for most
    # columns; the few with extra column_config (formatting/help) still
    # carry that same name as their label, never a different alias.
    column_config = json.loads(trades_tab.dataframe[0].proto.columns)
    assert column_config["Trigger"]["label"] == "Trigger"
    assert column_config["Trigger detail"]["label"] == "Trigger detail"
    assert column_config["Adjustments"]["label"] == "Adjustments"
    assert column_config["Position origin"]["label"] == "Position origin"
    assert column_config["Position origin date"]["label"] == "Position origin date"
    # Every real action must resolve to one of _classify_action's own
    # labels, never a leftover placeholder -- confirms the engine's new
    # frames actually reached build_trade_log, not just that a "side"
    # column silently carried the table alone.
    from quantlab.backtesting.trade_log import _classify_action

    valid_actions = {
        _classify_action(p, n)
        for p in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for n in (-1.0, -0.5, 0.0, 0.5, 1.0)
    }
    assert set(table["Action"]) <= valid_actions
    # Trigger is never fused with Trigger detail -- no row's Trigger
    # contains a literal "(" -- and at least one row has a precise
    # Trigger detail (mean_reversion's own explain_signals()) plus at
    # least one strategy_signal row's Details text keeps the generic
    # "signal X -> Y since last rebalance" text. Also confirms the core
    # fix: at least one entry keeps its strategy_signal trigger even
    # though a real constraint also fired as an adjustment on it.
    assert table["Trigger"].notna().any()
    assert not any("(" in str(value) for value in table["Trigger"])
    assert table["Trigger detail"].notna().any()
    strategy_signal_details = table.loc[
        table["Trigger"] == "strategy_signal", "Details"
    ]
    assert strategy_signal_details.notna().all()
    assert all(
        "since last rebalance" in str(value) for value in strategy_signal_details
    )
    entries = table[table["Action"] == "entry_long"]
    assert (entries["Trigger"] == "strategy_signal").any()


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
    _run_button(at).click().run()
    assert not at.exception

    robustness_tab = at.tabs[2]
    assert robustness_tab.label == "Robustness"
    assert len(robustness_tab.dataframe) == 1
    table = robustness_tab.dataframe[0].value
    assert list(table["Block"]) == ["Train", "Validation", "Test"]
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
    assert not any("configuration has changed" in w.value for w in at.warning)

    _sidebar_selectbox(at, "Strategy").set_value("mean_reversion").run()
    assert not at.exception
    assert any("configuration has changed" in w.value for w in at.warning)


def test_frequency_mismatch_shown_as_prominent_error_not_small_caption() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Frequency").set_value("1h").run()
    _run_button(at).click().run()

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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()

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
    _run_button(at).click().run()

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
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value(
        "SPY, QQQ, TLT, GLD"
    ).run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()

    _sidebar_checkbox(at, "Enable minimum position size").set_value(True).run()
    _sidebar_checkbox(at, "Cap gross exposure").set_value(True).run()
    _sidebar_checkbox(at, "Cap net exposure").set_value(True).run()
    _sidebar_checkbox(at, "Cap number of positions").set_value(True).run()
    _sidebar_checkbox(at, "Cap turnover per period").set_value(True).run()
    assert not at.exception

    _run_button(at).click().run()
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
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _run_button(at).click().run()

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

    # 5 columns: exactly one card per column, one full row -- no ragged
    # remainder the way a count that doesn't evenly divide 5 would leave.
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
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
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
    _run_button(at).click().run()

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
    _run_button(at).click().run()

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
    _run_button(at).click().run()

    assert not at.exception
    report_tab = at.tabs[3]
    assert report_tab.label == "Report"
    _, (html, _warnings) = at.session_state["wf_report_html"]
    assert "walk" in html.lower()


def test_walk_forward_no_fitting_fold_shows_error_not_crash() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_walk_forward_mode(at)
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_number_input(at, "Train window (periods)").set_value(10_000).run()
    _run_button(at).click().run()

    assert not at.exception
    assert any("No walk-forward fold fit" in e.value for e in at.error)
    assert "wf_result" not in at.session_state


def test_switching_to_backtest_mode_does_not_affect_a_stored_walk_forward_result() -> (
    None
):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    _configure_offline_walk_forward(at)
    _run_button(at).click().run()
    assert not at.exception
    assert "wf_result" in at.session_state

    at.segmented_control[0].set_value("Backtest").run()

    assert not at.exception
    assert "wf_result" in at.session_state
    assert any(b.label == "Run backtest" for b in at.sidebar.button)
    assert any(i.value.startswith("Configure an experiment") for i in at.info)


def _configure_offline_mean_reversion(at: AppTest) -> AppTest:
    """A strategy with real grid-eligible parameters, for sensitivity tests."""
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("mean_reversion").run()
    return at


def test_backtest_robustness_tab_bootstrap_runs_and_displays() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Robustness"
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    y_select.set_value("entry_threshold").run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    _run_button(at).click().run()
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
    ).set_value("entry_threshold").run()
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


# --------------------------------------------------------------------------- #
# Multi-instrument sidebar (Phase C): conflict detection, provenance-based
# source, mixed-calendar handling, frequency-compatibility filtering, and the
# rewritten benchmark lock-on-overlap behaviour.
# --------------------------------------------------------------------------- #


def test_conflicting_symbol_across_pickers_blocks_submission_with_error() -> None:
    """Picking the same symbol from two pickers makes its source/calendar
    ambiguous — never silently deduplicated (see `_combine_instrument_picks`
    in app.py)."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert _run_button(at).proto.disabled is False

    yahoo_ms = _sidebar_multiselect_by_key(at, "yahoo_symbols")
    # SPY is already in CSV's default population (SPY, QQQ, TLT, GLD).
    yahoo_ms.set_value(["SPY — SPDR S&P 500 ETF Trust"]).run()

    assert not at.exception
    assert any("ambiguous" in e.value and "SPY" in e.value for e in at.sidebar.error)
    assert _run_button(at).proto.disabled is True


def test_instrument_source_comes_from_picker_provenance_not_a_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The instrument table's Source column defaults to the picker a symbol
    was actually picked from (see `_combine_instrument_picks`'s docstring in
    app.py), never a symbol-shape heuristic. Checked against the *built
    config* returned by `build_config_from_inputs`, not just the sidebar's
    own table, so this exercises the real
    sidebar -> `_collect_inputs` -> `build_config_from_inputs` pipeline.

    `run_dashboard_backtest_with_data` is monkeypatched to capture the
    config and raise immediately (before any data loading happens), so
    this stays fully offline — real Binance OHLCV data is never fetched,
    only its (also monkeypatched) symbol-suggestion list."""
    import quantlab.dashboard.state as state_module
    from quantlab.config import DataSourceName
    from quantlab.data.base import SymbolSuggestion
    from quantlab.data.binance import BinanceDataSource

    monkeypatch.setattr(
        BinanceDataSource,
        "list_trading_symbols",
        lambda self: [SymbolSuggestion(symbol="BTCUSDT", description="BTC/USDT")],
    )
    st.cache_data.clear()  # avoid a real universe cached by an earlier test/run

    captured: dict[str, Any] = {}

    def fake_run_dashboard_backtest_with_data(config: Any) -> Any:
        captured["config"] = config
        raise RuntimeError("stop before real data loading")

    monkeypatch.setattr(
        state_module,
        "run_dashboard_backtest_with_data",
        fake_run_dashboard_backtest_with_data,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("").run()
    _load_binance_universe(at)
    binance_ms = _sidebar_multiselect_by_key(at, "binance_symbols")
    binance_ms.set_value([binance_ms.options[0]]).run()
    assert _run_button(at).proto.disabled is False
    _run_button(at).click().run()

    assert "config" in captured
    instruments = captured["config"].data.instruments
    assert len(instruments) == 1
    assert instruments[0].symbol == "BTCUSDT"
    assert instruments[0].source is DataSourceName.BINANCE
    assert instruments[0].calendar == "24/7"


def test_frequency_options_reflect_selected_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Frequency dropdown offers exactly
    `compatible_frequencies_for_sources()`'s answer for the currently
    selected instruments' sources — computed here directly rather than
    hardcoded, so this test can't silently drift from the real
    frequency-compatibility intersection logic in config.py."""
    from quantlab.config import DataSourceName, compatible_frequencies_for_sources
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
    freq = next(sb for sb in at.sidebar.selectbox if sb.label == "Frequency")
    expected_csv_only = sorted(
        f.value for f in compatible_frequencies_for_sources({DataSourceName.CSV})
    )
    assert freq.options == expected_csv_only

    _load_binance_universe(at)
    binance_ms = _sidebar_multiselect_by_key(at, "binance_symbols")
    binance_ms.set_value([binance_ms.options[0]]).run()

    freq = next(sb for sb in at.sidebar.selectbox if sb.label == "Frequency")
    # '1h' is additionally excluded here: adding BTCUSDT mixes calendars
    # (CSV's default population is XNYS, BTCUSDT is 24/7), and verified
    # closures only work at daily frequency -- see
    # test_mixed_calendar_universe_excludes_intraday_frequency below for a
    # dedicated check of that exclusion.
    expected_mixed = sorted(
        f.value
        for f in compatible_frequencies_for_sources(
            {DataSourceName.CSV, DataSourceName.BINANCE}
        )
        if f.value != "1h"
    )
    assert freq.options == expected_mixed
    assert freq.options != expected_csv_only


def test_mixed_calendar_warning_and_periods_per_year_field_appear_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No real Binance/Yahoo network access is available in CI (every other
    test in this file that runs an actual backtest stays on CSV data for
    exactly that reason), so this only exercises the sidebar's reaction to a
    mixed-calendar universe — warning plus field appearing together, and the
    Run button staying enabled — not a full multi-source backtest run."""
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
    # CSV's default population is all XNYS, so no mixed-calendar warning yet.
    assert not any(
        "Instruments span more than one calendar" in w.value for w in at.sidebar.warning
    )
    assert not any(
        f.label == "Periods per year (annualisation factor)"
        for f in at.sidebar.number_input
    )

    _load_binance_universe(at)
    binance_ms = _sidebar_multiselect_by_key(at, "binance_symbols")
    binance_ms.set_value([binance_ms.options[0]]).run()

    assert not at.exception
    assert any(
        "Instruments span more than one calendar" in w.value for w in at.sidebar.warning
    )
    periods_field = next(
        f
        for f in at.sidebar.number_input
        if f.label == "Periods per year (annualisation factor)"
    )
    # Defaults to 365, not 252: the mix includes a 24/7 instrument (BTCUSDT),
    # so the business-day equity convention would understate its real
    # trading frequency.
    assert periods_field.value == 365
    assert _run_button(at).proto.disabled is False


def test_mixed_calendar_universe_excludes_intraday_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'1h' must not be offered for a mixed-calendar universe: verified
    closures only work at daily frequency, so ExperimentConfig itself would
    reject it -- the picker must never offer something the config would
    then refuse (same principle as source-compatibility filtering)."""
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
    freq = next(sb for sb in at.sidebar.selectbox if sb.label == "Frequency")
    assert "1h" in freq.options  # CSV's default population is all XNYS

    _load_binance_universe(at)
    binance_ms = _sidebar_multiselect_by_key(at, "binance_symbols")
    binance_ms.set_value([binance_ms.options[0]]).run()

    freq = next(sb for sb in at.sidebar.selectbox if sb.label == "Frequency")
    assert "1h" not in freq.options
    assert any(
        "'1h' is unavailable for a mixed-calendar universe" in c.value
        for c in at.sidebar.caption
    )


def test_periods_per_year_value_flows_into_the_built_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complements the sidebar-only check above: an explicit
    "Periods per year" value, entered while the universe spans more than
    one calendar, actually reaches `BacktestConfig.periods_per_year` via
    `_collect_inputs` / `build_config_from_inputs`.
    `run_dashboard_backtest_with_data` is monkeypatched to capture the
    config before any data loading, keeping this offline."""
    import quantlab.dashboard.state as state_module
    from quantlab.data.base import SymbolSuggestion
    from quantlab.data.binance import BinanceDataSource

    monkeypatch.setattr(
        BinanceDataSource,
        "list_trading_symbols",
        lambda self: [SymbolSuggestion(symbol="BTCUSDT", description="BTC/USDT")],
    )
    st.cache_data.clear()  # avoid a real universe cached by an earlier test/run

    captured: dict[str, Any] = {}

    def fake_run_dashboard_backtest_with_data(config: Any) -> Any:
        captured["config"] = config
        raise RuntimeError("stop before real data loading")

    monkeypatch.setattr(
        state_module,
        "run_dashboard_backtest_with_data",
        fake_run_dashboard_backtest_with_data,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _load_binance_universe(at)
    binance_ms = _sidebar_multiselect_by_key(at, "binance_symbols")
    binance_ms.set_value([binance_ms.options[0]]).run()
    periods_field = next(
        f
        for f in at.sidebar.number_input
        if f.label == "Periods per year (annualisation factor)"
    )
    periods_field.set_value(365).run()
    _run_button(at).click().run()

    assert "config" in captured
    assert captured["config"].backtest.periods_per_year == 365


def test_benchmark_symbol_matching_an_instrument_locks_its_source_and_calendar() -> (
    None
):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    # SPY is already a tradable instrument via CSV's default population.
    benchmark = _sidebar_text_input(at, "Benchmark symbol")
    assert benchmark.value == "SPY"
    assert any(
        "SPY is already a tradable instrument" in c.value
        and "source (csv)" in c.value
        and "calendar (XNYS)" in c.value
        for c in at.sidebar.caption
    )
    assert not any(sb.label == "Benchmark source" for sb in at.sidebar.selectbox)
    assert not any(f.label == "Benchmark calendar" for f in at.sidebar.text_input)


def test_benchmark_symbol_not_matching_any_instrument_shows_source_and_calendar() -> (
    None
):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_text_input(at, "Benchmark symbol").set_value("NOTINLIST").run()

    assert not any(
        "is already a tradable instrument" in c.value for c in at.sidebar.caption
    )
    source = next(sb for sb in at.sidebar.selectbox if sb.label == "Benchmark source")
    assert source.key == "benchmark_source_select"
    assert source.options == ["yahoo", "binance", "csv"]
    assert source.value == "csv"  # detect_source() can't guess a shape-less string
    calendar = next(f for f in at.sidebar.text_input if f.label == "Benchmark calendar")
    assert calendar.key == "benchmark_calendar_input"
    assert calendar.value == "XNYS"


def _switch_to_strategies_mode(at: AppTest) -> AppTest:
    at.segmented_control[0].set_value("Strategies").run()
    return at


def test_strategies_mode_shows_gallery_with_one_card_per_strategy() -> None:
    from quantlab.strategies.base import available_strategies

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    assert not at.exception

    strategies = available_strategies()
    open_buttons = {
        b.key for b in at.button if b.key and b.key.startswith("explorer_open_")
    }
    assert open_buttons == {f"explorer_open_{name}" for name in strategies}


def test_strategies_mode_hides_the_sidebar() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.sidebar.children) > 0  # Backtest mode: sidebar populated
    _switch_to_strategies_mode(at)
    assert len(at.sidebar.children) == 0


@pytest.mark.parametrize(
    "strategy_name",
    [
        "buy_and_hold",
        "pairs_trading",
        "mean_reversion",
        "time_series_momentum",
        "cross_sectional_momentum",
        "trend_following",
    ],
)
def test_strategy_detail_page_opens_with_every_section(strategy_name: str) -> None:
    """Opening any registered strategy's detail page must not raise, and
    must show every common documented section. References / Further
    reading is optional -- required only when that strategy's profile
    actually sets ``references_md`` (e.g. Buy & Hold deliberately has none;
    see `test_buy_and_hold_detail_page_has_no_references_section` below).
    The interactive lab lives in a lazy expander (see `detail.py`) and is
    collapsed by default, so its own body does NOT run here -- see
    `test_strategy_lab_opens_and_runs_without_exception` below for that."""
    from quantlab.dashboard.explorer.profile import get_profile

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key=f"explorer_open_{strategy_name}").click().run()

    assert not at.exception
    assert at.session_state["explorer_strategy"] == strategy_name
    expander_labels = {e.label for e in at.expander}
    assert expander_labels >= {
        "Overview",
        "Economic intuition",
        "Mathematical definition & signals",
        "Assumptions",
        "Diagnostics",
        "Parameters",
        "Interactive laboratory",
        "Interpretation",
        "Limitations & failure modes",
    }
    profile = get_profile(strategy_name)
    assert profile is not None
    has_references_section = "References / Further reading" in expander_labels
    assert has_references_section == (profile.references_md is not None)


def test_buy_and_hold_detail_page_has_no_references_section() -> None:
    """Buy & Hold's profile deliberately sets ``references_md=None`` -- no
    strategy-specific literature was genuinely indispensable for the
    zero-skill baseline it describes -- so its detail page must not show a
    References / Further reading section at all."""
    from quantlab.dashboard.explorer.profile import get_profile

    profile = get_profile("buy_and_hold")
    assert profile is not None
    assert profile.references_md is None

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key="explorer_open_buy_and_hold").click().run()
    assert not at.exception
    expander_labels = {e.label for e in at.expander}
    assert "References / Further reading" not in expander_labels


@pytest.mark.parametrize(
    "strategy_name",
    [
        "buy_and_hold",
        "pairs_trading",
        "mean_reversion",
        "time_series_momentum",
        "cross_sectional_momentum",
        "trend_following",
    ],
)
def test_strategy_lab_opens_and_runs_without_exception(strategy_name: str) -> None:
    """Actually opening each strategy's Interactive laboratory expander (not
    just visiting the detail page -- see the lazy-expander note on the test
    above) must run its full default body without exception, on real
    bundled offline data."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key=f"explorer_open_{strategy_name}").click().run()
    assert not at.exception

    at.session_state[f"explorer_lab_expander_{strategy_name}"] = True
    at.run()
    assert not at.exception


def _open_lab(at: AppTest, strategy_name: str) -> AppTest:
    _switch_to_strategies_mode(at)
    at.button(key=f"explorer_open_{strategy_name}").click().run()
    at.session_state[f"explorer_lab_expander_{strategy_name}"] = True
    at.run()
    return at


def test_lab_symbol_picker_defaults_to_csv_with_bundled_demo_data_on() -> None:
    """Unlike the main sidebar (`use_bundled_demo_data` defaults to False),
    a lab defaults it to True so it keeps working fully offline with no
    setup -- see `render_symbol_and_source_picker`."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "mean_reversion")
    assert not at.exception

    source = next(r for r in at.radio if r.key == "explorer_mr_source")
    assert source.value == "csv"
    csv_input = next(
        f for f in at.text_input if f.key == "explorer_mr_csv_symbols_input"
    )
    assert csv_input.value == "SPY, QQQ, TLT, GLD"
    bundled_toggle = next(
        t for t in at.toggle if t.key == "explorer_mr_use_bundled_demo_data"
    )
    assert bundled_toggle.value is True


@pytest.mark.parametrize(
    ("calendar", "expected"),
    [("24/7", 365), ("XNYS", 252), ("XHKG", 252)],
)
def test_tsmom_lab_periods_per_year_derives_from_the_calendar(
    calendar: str, expected: int
) -> None:
    """Regression test: the Time-Series Momentum lab's volatility_adjusted
    panel used to annualise at a hardcoded 252 regardless of the selected
    calendar -- for a 24/7 market this silently mis-annualised the
    illustrative volatility, unlike the real strategy (`periods_per_year`
    is injected from the experiment's own data frequency: 365 for daily
    crypto). Tested directly (no Streamlit runtime needed) since AppTest
    cannot introspect a rendered Plotly chart's own title/values in this
    Streamlit version."""
    from quantlab.dashboard.explorer.labs.time_series_momentum import (
        _periods_per_year_for_calendar,
    )

    assert _periods_per_year_for_calendar(calendar) == expected


def test_tsmom_lab_accepts_an_overridden_24_7_calendar_without_exception() -> None:
    """End-to-end smoke check that the new editable "Calendar" field (see
    render_symbol_and_source_picker) actually reaches the lab and the
    24/7 branch renders without exception -- avoids depending on live
    Binance network access just to exercise this calendar."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "time_series_momentum")
    assert not at.exception

    calendar_input = next(
        f for f in at.text_input if f.key == "explorer_tsmom_csv_calendar"
    )
    calendar_input.set_value("24/7").run()
    assert not at.exception


def test_lab_symbol_picker_switching_to_yahoo_shows_the_yahoo_picker() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "mean_reversion")

    source = next(r for r in at.radio if r.key == "explorer_mr_source")
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    source.set_value("yahoo").run()
    assert not at.exception
    assert not any(f.key == "explorer_mr_csv_symbols_input" for f in at.text_input)
    assert any(ms.key == "explorer_mr_yahoo_symbols" for ms in at.multiselect)


def test_lab_csv_calendar_defaults_to_xnys_and_is_editable() -> None:
    """Regression test: a lab's CSV symbol picker used to always assume
    XNYS with no way to change it -- CSV data carries no calendar
    information at all, so a non-XNYS local instrument (futures, a
    non-US index) needs an explicit override, same as the main
    dashboard's own per-instrument table."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "mean_reversion")
    assert not at.exception

    calendar_input = next(
        f for f in at.text_input if f.key == "explorer_mr_csv_calendar"
    )
    assert calendar_input.value == "XNYS"


def test_lab_yahoo_symbol_auto_detects_a_non_xnys_calendar() -> None:
    """Regression test: a Yahoo symbol used to silently get XNYS regardless
    of its own suffix -- "1211.HK" must auto-detect XHKG (see
    `detect_calendar`), shown in an editable field the user can still
    correct if the guess is wrong."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "mean_reversion")

    source = next(r for r in at.radio if r.key == "explorer_mr_source")
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    source.set_value("yahoo").run()

    picker = next(ms for ms in at.multiselect if ms.key == "explorer_mr_yahoo_symbols")
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    picker.set_value(["1211.HK"]).run()
    assert not at.exception

    calendar_input = next(
        f for f in at.text_input if f.key == "explorer_mr_yahoo_calendar"
    )
    assert calendar_input.value == "XHKG"


def test_lab_yahoo_multi_calendar_selection_is_rejected_with_a_clear_error() -> None:
    """A lab computes on one flat price matrix and cannot represent more
    than one calendar at once -- selecting symbols that need different
    calendars (a US ticker and a Hong Kong one) must be rejected with a
    clear error rather than silently picking one for all of them."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "mean_reversion")

    source = next(r for r in at.radio if r.key == "explorer_mr_source")
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    source.set_value("yahoo").run()

    picker = next(ms for ms in at.multiselect if ms.key == "explorer_mr_yahoo_symbols")
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    picker.set_value(["AAPL", "1211.HK"]).run()
    assert not at.exception
    assert not any(f.key == "explorer_mr_yahoo_calendar" for f in at.text_input)
    assert any("different calendars" in e.value for e in at.error)


def test_lab_symbol_picker_binance_requires_an_explicit_load_click() -> None:
    """Mirrors the main sidebar's own Binance gate: fetching the universe
    is never triggered just by selecting "binance" as the source."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _open_lab(at, "mean_reversion")

    source = next(r for r in at.radio if r.key == "explorer_mr_source")
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    source.set_value("binance").run()
    assert not at.exception
    assert not any(ms.key == "explorer_mr_binance_symbols" for ms in at.multiselect)
    load_button = next(
        b for b in at.button if b.key == "explorer_mr_binance_universe_load_button"
    )

    at.session_state["explorer_lab_expander_mean_reversion"] = True
    load_button.click().run()
    assert not at.exception
    assert any(ms.key == "explorer_mr_binance_symbols" for ms in at.multiselect)


def test_every_lab_uses_a_symbol_picker_key_prefix_unique_to_itself() -> None:
    """Every lab's `render_symbol_and_source_picker(key_prefix=...)` must be
    unique -- a shared/copy-pasted prefix would make two labs silently
    read and write the same session_state entries (a picked Yahoo symbol
    in one lab leaking into another)."""
    import re
    from pathlib import Path

    import quantlab.dashboard.explorer.labs as labs_package

    source_dir = Path(labs_package.__file__).parent
    lab_files = (
        "buy_and_hold.py",
        "pairs_trading.py",
        "mean_reversion.py",
        "time_series_momentum.py",
        "cross_sectional_momentum.py",
        "trend_following.py",
    )
    found_prefixes = {}
    for filename in lab_files:
        text = (source_dir / filename).read_text(encoding="utf-8")
        match = re.search(r'key_prefix="(explorer_\w+)"', text)
        assert match is not None, f"{filename}: no key_prefix found"
        found_prefixes[filename] = match.group(1)
    assert len(set(found_prefixes.values())) == 6, found_prefixes


def test_cross_sectional_momentum_lab_handles_long_only_top_fraction_above_half() -> (
    None
):
    """Regression test: with long_short disabled and top_fraction=0.75, the
    lab previously reused top_fraction verbatim as the comparison bottom
    fraction, so 0.75 + 0.75 > 1 made select_top_bottom() raise inside
    cross_sectional_momentum_persistence()."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key="explorer_open_cross_sectional_momentum").click().run()
    at.session_state["explorer_lab_expander_cross_sectional_momentum"] = True
    at.run()
    assert not at.exception

    at.session_state["explorer_lab_expander_cross_sectional_momentum"] = True
    at.slider(key="explorer_csmom_top").set_value(0.75).run()
    assert not at.exception


def test_back_to_gallery_button_returns_to_the_gallery() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key="explorer_open_buy_and_hold").click().run()
    assert at.session_state["explorer_strategy"] == "buy_and_hold"

    at.button(key="explorer_back").click().run()
    assert not at.exception
    assert "explorer_strategy" not in at.session_state
    open_buttons = [
        b.key for b in at.button if b.key and b.key.startswith("explorer_open_")
    ]
    assert "explorer_open_buy_and_hold" in open_buttons


def test_pairs_trading_backtest_results_tab_shows_pair_diagnostics() -> None:
    """The generic Strategy Explorer results-diagnostics dispatch (declared
    only by the pairs_trading profile) surfaces its section in the Results
    tab -- and only there, never for a strategy without one."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    _run_button(at).click().run()
    assert not at.exception

    subheaders = {s.value for s in at.subheader}
    assert "Pair relationship diagnostics" in subheaders


def test_non_pairs_strategy_backtest_results_tab_has_no_pair_diagnostics() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _run_button(at).click().run()
    assert not at.exception

    subheaders = {s.value for s in at.subheader}
    assert "Pair relationship diagnostics" not in subheaders


def test_backtest_downloaded_report_includes_pair_diagnostics() -> None:
    """The HTML report downloaded from Backtest mode's Report tab must
    include the same pair-diagnostics section visible live in Results --
    `_collect_backtest_robustness_evidence` folds it in, keyed by the
    profile's own `results_diagnostics.key`, converted via
    `report_section()` exactly like the CLI does."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _configure_offline_pairs_trade(at)
    at.session_state["dashboard_active_tab"] = "Report"
    _run_button(at).click().run()
    assert not at.exception

    _, (report_tab_html, _warnings) = at.session_state["report_html"]
    assert "Pair Diagnostics" in report_tab_html
    assert "Strategy diagnostics" in report_tab_html
    assert "<h2>Robustness</h2>" in report_tab_html
    robustness_index = report_tab_html.index("<h2>Robustness</h2>")
    diagnostics_index = report_tab_html.index("Strategy diagnostics")
    assert diagnostics_index < robustness_index


def test_downloaded_report_updates_when_a_live_diagnostic_slider_moves() -> None:
    """Regression test: moving the Results tab's "Forward-return horizon"
    slider used to leave the downloaded HTML report stuck showing the
    fixed default (skip_period) value, because `_render_report_tab`'s
    cache key was keyed off `id(diagnostics)`, which never changes when
    only a live Results-tab widget choice changes -- `report_section()`
    reads that widget straight from session_state instead (see
    `_collect_backtest_robustness_evidence`). Reproduces with
    long_short=True, matching the user-reported scenario exactly."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _sidebar_text_input(at, "CSV symbols (comma-separated)").set_value("SPY, QQQ").run()
    _sidebar_date_input(at, "End date").set_value(datetime.date(2019, 6, 1)).run()
    _sidebar_selectbox(at, "Strategy").set_value("cross_sectional_momentum").run()
    _sidebar_checkbox(at, "Long/short").set_value(True).run()
    at.session_state["dashboard_active_tab"] = "Report"
    _run_button(at).click().run()
    assert not at.exception

    _, (report_before, _warnings_before) = at.session_state["report_html"]

    at.session_state["csmom_results_diag_holding_period"] = 5
    at.run()
    assert not at.exception

    _, (report_after, _warnings_after) = at.session_state["report_html"]
    assert report_after != report_before


def test_walk_forward_diagnostics_note_shown_for_a_strategy_with_diagnostics() -> None:
    """A strategy WITH declared `results_diagnostics` (every built-in
    strategy except buy_and_hold, which is itself excluded from Walk-
    forward mode entirely) gets an explanatory note instead of a silently
    missing section (see `_render_walk_forward_diagnostics_note`: each
    fold can select different parameters than the base config, so the
    diagnostics visible in Backtest mode are intentionally absent here).
    The `profile.results_diagnostics is None` branch itself (no note) is
    covered structurally by test_dashboard_explorer_profiles.py, not by a
    live strategy here -- every walk-forward-eligible strategy now
    declares diagnostics."""
    at_pairs = AppTest.from_file(APP_PATH, default_timeout=120)
    at_pairs.run()
    _switch_to_walk_forward_mode(at_pairs)
    _sidebar_text_input(at_pairs, "CSV symbols (comma-separated)").set_value(
        "SPY, QQQ"
    ).run()
    _sidebar_selectbox(at_pairs, "Strategy").set_value("pairs_trading").run()
    # Small windows and a short date range -- this only needs ONE completed
    # fold to reach the Results tab, not a realistic pairs-trading backtest
    # (an unbounded end date here previously produced dozens of folds times
    # a parameter grid search, timing out well past two minutes).
    # formation_window must fit inside train_window for any fold to produce
    # a weight at all.
    _sidebar_date_input(at_pairs, "End date").set_value(
        datetime.date(2019, 7, 15)
    ).run()
    formation_window = next(
        s for s in at_pairs.sidebar.slider if s.label == "Formation window (periods)"
    )
    formation_window.set_value(60).run()
    _sidebar_number_input(at_pairs, "Train window (periods)").set_value(90).run()
    _sidebar_number_input(at_pairs, "Validation window (periods)").set_value(20).run()
    _sidebar_number_input(at_pairs, "Test window (periods)").set_value(20).run()
    _run_button(at_pairs).click().run()
    assert not at_pairs.exception
    pairs_captions = [c.value for c in at_pairs.caption]
    assert any("results diagnostics" in c for c in pairs_captions)
    assert "wf_strategy_diagnostics" not in at_pairs.session_state


def test_pairs_trading_lab_widget_interaction_recomputes_without_exception() -> None:
    """Moving a lab parameter must trigger a real Streamlit rerun that
    recomputes and re-renders -- not just render once with default values."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key="explorer_open_pairs_trading").click().run()
    assert not at.exception
    # The lab lives in a lazy expander (see `detail.py`) -- collapsed by
    # default, so it must be opened before its widgets exist. AppTest does
    # not persist a directly-assigned session_state value for a tracked
    # container widget across an unrelated interaction's own rerun, so it
    # must be re-asserted before every subsequent `.run()` below.
    at.session_state["explorer_lab_expander_pairs_trading"] = True
    at.run()
    assert not at.exception

    at.session_state["explorer_lab_expander_pairs_trading"] = True
    at.slider(key="explorer_pairs_formation").set_value(250).run()
    assert not at.exception
    assert at.slider(key="explorer_pairs_formation").value == 250

    at.session_state["explorer_lab_expander_pairs_trading"] = True
    at.checkbox(key="explorer_pairs_dynamic").set_value(False).run()
    assert not at.exception


def test_mean_reversion_lab_widget_interaction_recomputes_without_exception() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key="explorer_open_mean_reversion").click().run()
    assert not at.exception
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    at.run()
    assert not at.exception

    at.session_state["explorer_lab_expander_mean_reversion"] = True
    at.checkbox(key="explorer_mr_use_zscore").set_value(False).run()
    assert not at.exception
    at.session_state["explorer_lab_expander_mean_reversion"] = True
    at.slider(key="explorer_mr_rsi_window").set_value(30).run()
    assert not at.exception


def test_strategy_lab_does_not_run_while_its_expander_is_collapsed() -> None:
    """Regression test: a plain `st.expander` still runs its body every
    rerun while collapsed. The Interactive laboratory expander must use
    the stateful/lazy variant so opening the detail page (or interacting
    with any OTHER widget on it) does not silently re-trigger the lab's
    full computation (data load, OLS fits, ADF/cointegration tests)."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    _switch_to_strategies_mode(at)
    at.button(key="explorer_open_pairs_trading").click().run()
    assert not at.exception
    # The lab's own widgets must not exist yet -- its body never ran.
    assert not any(s.key and s.key.startswith("explorer_pairs_") for s in at.slider)

    at.session_state["explorer_lab_expander_pairs_trading"] = True
    at.run()
    assert not at.exception
    assert any(s.key == "explorer_pairs_formation" for s in at.slider)


def test_render_pair_diagnostics_isolated_component(
    two_symbol_panel: pd.DataFrame,
) -> None:
    """Isolated (no real Streamlit, no dashboard) component test for the
    Results-tab pair-diagnostics renderer, driven by a real
    ``PairDiagnostics`` computed on the cointegrated-by-construction
    ``two_symbol_panel`` fixture -- complements the AppTest-based
    integration coverage above."""
    from quantlab.data.base import price_matrix
    from quantlab.features.pairs_diagnostics import compute_pair_diagnostics

    class FakeColumn:
        def __init__(self, sink: list[tuple[str, str]]) -> None:
            self._sink = sink

        def metric(self, label: str, value: str) -> None:
            self._sink.append((label, value))

    class FakeStreamlit:
        def __init__(self) -> None:
            self.subheaders: list[str] = []
            self.captions: list[str] = []
            self.infos: list[str] = []
            self.plotly_chart_calls = 0
            self.metrics: list[tuple[str, str]] = []

        def subheader(self, text: str) -> None:
            self.subheaders.append(text)

        def caption(self, text: str) -> None:
            self.captions.append(text)

        def info(self, text: str) -> None:
            self.infos.append(text)

        def plotly_chart(self, fig: object, **kwargs: object) -> None:
            self.plotly_chart_calls += 1

        def columns(self, n: int) -> list[FakeColumn]:
            return [FakeColumn(self.metrics) for _ in range(n)]

    diagnostics = compute_pair_diagnostics(
        price_matrix(two_symbol_panel),
        "EWA",
        "EWB",
        formation_window=100,
        indicator_window=20,
        dynamic_hedge_ratio=True,
    )
    fake = FakeStreamlit()

    render_pair_diagnostics(
        fake,
        diagnostics,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_threshold=4.0,
        adf_pvalue_threshold=0.10,
    )

    assert fake.subheaders == ["Pair relationship diagnostics"]
    assert any("EWA / EWB" in caption for caption in fake.captions)
    # Hedge ratio, spread, the centered-indicator threshold-overlay chart,
    # plus a rolling-ADF-p-value chart (the pair is cointegrated by
    # construction, so this last chart is present).
    assert fake.plotly_chart_calls == 4
    # Two stationarity cards (ADF, Engle-Granger cointegration), 3 metrics
    # each, plus the Half-life/Hedge-ratio-stability metric pair.
    assert len(fake.metrics) == 8
    labels = {label for label, _ in fake.metrics}
    assert labels == {
        "Statistic",
        "p-value",
        "Verdict",
        "Half-life",
        "Hedge-ratio stability (std of beta)",
    }
    assert not fake.infos  # both ADF and cointegration results are conclusive
