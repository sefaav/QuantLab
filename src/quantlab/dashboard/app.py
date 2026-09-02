"""Streamlit dashboard.

Run with:

    streamlit run src/quantlab/dashboard/app.py

Lets a user configure an experiment in the sidebar, run the backtest (its
delayed-execution barrier prevents common look-ahead leakage, though a custom
strategy remains responsible for its own causal construction), inspect
metrics/charts/trades, run robustness checks and download an HTML research
report.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import streamlit as st

import quantlab.dashboard.explorer.profiles  # noqa: F401  (registration side effect)
from quantlab.config import DataSourceName, compatible_frequencies_for_sources
from quantlab.dashboard.components import (
    binance_universe_labels,
    entry_threshold_bounds,
    exit_threshold_bounds,
    mean_reversion_slider_bounds,
    parse_symbols,
    render_charts,
    render_exposure_and_cost_charts,
    render_gross_net_comparison,
    render_metric_cards,
    render_sensitivity_heatmap,
    render_trade_table,
    symbols_picker,
    yahoo_universe_labels,
)
from quantlab.dashboard.explorer import detail as explorer_detail
from quantlab.dashboard.explorer import gallery as explorer_gallery
from quantlab.dashboard.explorer.profile import get_profile
from quantlab.dashboard.state import (
    build_config_from_inputs,
    default_end_date,
    detect_calendar,
    detect_source,
    estimate_walk_forward_backtest_count,
    run_dashboard_backtest_with_data,
    run_dashboard_bootstrap,
    run_dashboard_permutation_test,
    run_dashboard_sensitivity,
    run_dashboard_stress_tests,
    run_dashboard_walk_forward,
    run_dashboard_walk_forward_sensitivity,
    run_dashboard_walk_forward_stress_tests,
)
from quantlab.logging_config import configure_logging, get_logger
from quantlab.progress import ProgressReporter
from quantlab.reporting.sections import DiagnosticsSection
from quantlab.reporting.tables import format_bootstrap_summary
from quantlab.strategies.base import (
    available_strategies,
    strategy_parameter_names,
    strategy_sweepable_parameter_names,
)
from quantlab.strategies.mean_reversion import (
    UI_INDICATORS as MEAN_REVERSION_UI_INDICATORS,
)
from quantlab.validation.parameter_grid import parse_parameter_grid_values
from quantlab.validation.parameter_sensitivity import (
    infer_sensitivity_parameter_columns,
)

if TYPE_CHECKING:
    from quantlab.backtesting.result import BacktestResult
    from quantlab.config import ExperimentConfig
    from quantlab.validation.walk_forward import WalkForwardResult

# Streamlit runs this file in its own process (`quantlab dashboard` launches
# it via `subprocess.run`; a user can also run `streamlit run` on it
# directly), separate from any process that already called
# configure_logging() -- without this call here too, the dashboard's own
# logger has no handlers attached, so its exceptions/warnings are never
# written to logs/quantlab.log.
configure_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="QuantLab", page_icon="📈", layout="wide")


_PAIRS_TRADING_SYMBOLS_HELP = (
    "pairs_trading needs at least two symbols; its two legs are then picked below."
)


def _binance_symbols_picker() -> list[str]:
    # A plain `st.expander` still runs its body every rerun even while
    # collapsed, so this cannot simply live inside one -- that would
    # silently call Binance's API on every dashboard load, regardless of
    # whether the user ever opens this section (`cached_binance_universe`
    # is itself cached, but the FIRST network round trip per hour still
    # happens unconditionally). Gating the fetch behind an explicit button
    # avoids that, and (unlike a stateful/lazy expander) needs no
    # label<->symbol remapping to preserve an already-made selection across
    # reruns -- the underlying multiselect's own widget state persists
    # normally once it exists.
    if not st.session_state.get("binance_universe_load_requested", False):
        st.caption(
            "Loading the tradable symbol list calls Binance's public API. "
            "Click below to fetch it (cached for an hour after that)."
        )
        if st.button("Load Binance symbols", key="binance_universe_load_button"):
            # No `st.rerun()` here: the click already triggers Streamlit's
            # own full top-to-bottom rerun. An extra manual `st.rerun()`
            # would abort THIS run right here, before the CSV picker (and
            # any other widget further down the sidebar) ever executes --
            # and a widget that doesn't render on a given run loses its
            # keyed session-state value, silently resetting it back to
            # its default (this is exactly how the CSV symbols field used
            # to reset itself whenever this button was clicked). Setting
            # the flag and falling through to the picker below instead
            # renders it immediately, in this same run, with no abort.
            st.session_state["binance_universe_load_requested"] = True
        else:
            return []
    # Empty by default: all three pickers are visible simultaneously now, and
    # a non-empty default here would immediately conflict with CSV's bundled
    # demo default below (see `_combine_instrument_picks`).
    return symbols_picker(
        st,
        binance_universe_labels(st),
        "binance_symbols",
        (),
        help_suffix=_PAIRS_TRADING_SYMBOLS_HELP,
    )


def _yahoo_symbols_picker() -> list[str]:
    return symbols_picker(
        st,
        yahoo_universe_labels(),
        "yahoo_symbols",
        (),
        accept_new_options=True,
        help_suffix=_PAIRS_TRADING_SYMBOLS_HELP,
    )


def _csv_symbols_picker() -> list[str]:
    raw = st.text_input(
        "CSV symbols (comma-separated)",
        "SPY, QQQ, TLT, GLD",
        key="csv_symbols_input",
        help=(
            "Local files under data/raw, one CSV per symbol. When 'Allow "
            "bundled synthetic demo data' below is enabled, QuantLab falls "
            "back to its bundled SPY/QQQ/TLT/GLD demo files if every "
            "requested local file is absent."
        ),
    )
    return parse_symbols(raw)


def _combine_instrument_picks(
    yahoo_symbols: list[str],
    binance_symbols: list[str],
    csv_symbols: list[str],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Merge the three pickers into one ordered, deduplicated symbol list.

    Provenance is the picker a symbol actually came from — never a
    heuristic — so the instrument table's Source default is always exact.
    A symbol picked from two different pickers is a conflict: never
    silently deduplicated (which source/calendar would even apply is
    ambiguous), excluded from the returned symbol list and reported
    separately so the caller can block submission.
    """
    picks: list[tuple[list[str], str]] = [
        (yahoo_symbols, "yahoo"),
        (binance_symbols, "binance"),
        (csv_symbols, "csv"),
    ]
    seen_in: dict[str, list[str]] = {}
    order: list[str] = []
    for symbols, source_name in picks:
        for symbol in symbols:
            if symbol not in seen_in:
                order.append(symbol)
            seen_in.setdefault(symbol, []).append(source_name)
    conflicts = [symbol for symbol in order if len(seen_in[symbol]) > 1]
    conflict_set = set(conflicts)
    provenance = {
        symbol: seen_in[symbol][0] for symbol in order if symbol not in conflict_set
    }
    combined = [symbol for symbol in order if symbol not in conflict_set]
    return combined, provenance, conflicts


def _instrument_table(
    symbols: list[str], provenance: dict[str, str]
) -> list[dict[str, str]]:
    """Editable Source/Calendar table, one row per selected symbol.

    Source defaults to the picker the symbol came from (``provenance``) —
    never the ``detect_source`` heuristic. Calendar defaults to
    ``detect_calendar``'s best guess. Both are editable and rebuilt from
    ``symbols`` on every render, so removing a symbol from a picker above
    drops its row (and any prior edit) on the next run instead of leaving a
    stale entry behind.
    """
    overrides: dict[str, dict[str, str]] = st.session_state.get(
        "instrument_overrides", {}
    )
    rows = []
    for symbol in symbols:
        saved = overrides.get(symbol, {})
        default_source = saved.get("source") or provenance.get(symbol, "csv")
        default_calendar = saved.get("calendar") or (
            detect_calendar(symbol, DataSourceName(default_source)) or "XNYS"
        )
        rows.append(
            {
                "Instrument": symbol,
                "Source": default_source,
                "Calendar": default_calendar,
            }
        )
    edited = st.data_editor(
        pd.DataFrame(rows, columns=["Instrument", "Source", "Calendar"]),
        column_config={
            "Instrument": st.column_config.TextColumn(disabled=True),
            "Source": st.column_config.SelectboxColumn(
                options=["yahoo", "binance", "csv"], required=True
            ),
            "Calendar": st.column_config.TextColumn(
                required=True,
                help="'24/7' for a continuous market, or a pandas_market_calendars "
                "name such as XNYS, XHKG, XLON.",
            ),
        },
        hide_index=True,
        width="stretch",
        key="instrument_table_editor",
    )
    records = cast(list[dict[str, str]], edited.to_dict("records"))
    st.session_state["instrument_overrides"] = {
        row["Instrument"]: {"source": row["Source"], "calendar": row["Calendar"]}
        for row in records
    }
    return records


def _risk_control_pct_inputs(key_prefix: str) -> tuple[float | None, float | None]:
    """Render the shared stop_loss_pct/take_profit_pct widget pair.

    Operates on the REAL executed position (post-allocator/constraints/
    rebalancing/execution) -- see `quantlab.backtesting.accounting.
    _detect_stop_loss_take_profit`. Shared across every strategy that
    accepts these two parameters so the widget behavior/help text can
    never drift between strategies.
    """
    col_stop, col_take = st.columns(2)
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    with col_stop:
        if st.checkbox(
            "Enable stop-loss",
            value=False,
            key=f"{key_prefix}_enable_stop_loss_pct",
            help=(
                "Force-flatten the REAL executed position (or, for a "
                "strategy that declares a multi-asset group such as "
                "pairs_trading's two legs, the group's COMBINED position) "
                "once its gross (pre-cost) return since entry drops below "
                "-this value -- independent of any indicator-based stop "
                "the strategy itself may already have."
            ),
        ):
            stop_loss_pct = (
                st.slider(
                    "Stop-loss (%)",
                    1.0,
                    50.0,
                    10.0,
                    1.0,
                    key=f"{key_prefix}_stop_loss_pct",
                    help="Gross-return loss, since entry, that force-flattens "
                    "the position.",
                )
                / 100.0
            )
    with col_take:
        if st.checkbox(
            "Enable take-profit",
            value=False,
            key=f"{key_prefix}_enable_take_profit_pct",
            help=(
                "Force-flatten the REAL executed position (or, for a "
                "strategy that declares a multi-asset group such as "
                "pairs_trading's two legs, the group's COMBINED position) "
                "once its gross (pre-cost) return since entry rises above "
                "this value -- locks in a gain instead of waiting for the "
                "strategy's own ordinary exit condition."
            ),
        ):
            take_profit_pct = (
                st.slider(
                    "Take-profit (%)",
                    1.0,
                    50.0,
                    10.0,
                    1.0,
                    key=f"{key_prefix}_take_profit_pct",
                    help="Gross-return gain, since entry, that force-flattens "
                    "the position.",
                )
                / 100.0
            )
    return stop_loss_pct, take_profit_pct


def _strategy_param_inputs(strategy_name: str, symbols: list[str]) -> dict:
    """Render strategy-specific parameter widgets and return their values.

    Args:
        strategy_name: The selected strategy's registered name.
        symbols: The universe entered in the sidebar, used to default and
            populate the pairs-trading symbol pickers.
    """
    params: dict[str, object] = {}
    if strategy_name == "cross_sectional_momentum":
        lookback_period = st.slider(
            "Lookback (periods)",
            20,
            300,
            189,
            1,
            help=(
                "Trailing window used to rank assets by momentum, measured "
                "after skipping the most recent 'Skip' periods below."
            ),
        )
        params["lookback_period"] = lookback_period
        max_skip = min(42, lookback_period - 1)
        params["skip_period"] = st.slider(
            "Skip (periods)",
            0,
            max_skip,
            min(21, max_skip),
            1,
            help=(
                "Most-recent periods excluded from the lookback window, to "
                "avoid the well-documented short-term reversal effect right "
                "before formation."
            ),
        )
        params["long_short"] = st.checkbox(
            "Long/short",
            value=False,
            help=(
                "Off: hold only the top-fraction winners. On: also short the "
                "bottom-fraction losers, sized by 'Bottom fraction' below — a "
                "market-neutral-ish spread instead of a long-only tilt."
            ),
        )
        max_top_fraction = 0.75 if params["long_short"] else 1.0
        top_fraction = st.slider(
            "Top fraction",
            0.1,
            max_top_fraction,
            0.5,
            0.05,
            help=(
                "Fraction of the ranked universe held long — the strongest "
                "momentum names."
            ),
        )
        params["top_fraction"] = top_fraction
        if params["long_short"]:
            # top_fraction + bottom_fraction must not exceed 1.0 (disjoint groups).
            max_bottom_fraction = round(max(0.1, 1.0 - top_fraction), 2)
            params["bottom_fraction"] = st.slider(
                "Bottom fraction (shorted)",
                0.1,
                max_bottom_fraction,
                min(0.25, max_bottom_fraction),
                0.05,
                help=(
                    "Fraction of the ranked universe held short — the "
                    "weakest momentum names. Capped so it never overlaps "
                    "the top fraction above."
                ),
            )
        params["signal_scaling"] = st.selectbox(
            "Signal scaling",
            ["binary", "continuous"],
            index=0,
            help=(
                "binary weights every selected asset identically. continuous "
                "weights each selected asset by its RANK within its own "
                "selected leg, divided by that leg's own selected count "
                "(e.g. the weakest of 4 selected longs gets 0.25, the "
                "strongest gets 1.0) — which assets are selected is "
                "unchanged, only their relative size."
            ),
        )
        with st.expander("Advanced parameters"):
            params["stop_loss_pct"], params["take_profit_pct"] = (
                _risk_control_pct_inputs("csm")
            )
    elif strategy_name == "time_series_momentum":
        lookback_period = st.slider(
            "Lookback (periods)",
            20,
            300,
            200,
            1,
            help=(
                "Trailing window used to compute each asset's own momentum, "
                "measured after skipping the most recent 'Skip' periods below."
            ),
        )
        params["lookback_period"] = lookback_period
        max_skip = min(42, lookback_period - 1)
        params["skip_period"] = st.slider(
            "Skip (periods)",
            0,
            max_skip,
            min(21, max_skip),
            1,
            help=(
                "Most-recent periods excluded from the lookback window, to "
                "avoid the well-documented short-term reversal effect right "
                "before formation."
            ),
        )
        params["long_only"] = st.checkbox(
            "Long only",
            value=True,
            help=(
                "Clip the strategy's own signal to non-negative before "
                "allocation — separate from, and applied before, the "
                "portfolio-level 'Long only' constraint below."
            ),
        )
        signal_scaling = st.selectbox(
            "Signal scaling",
            ["binary", "continuous", "volatility_adjusted"],
            index=0,
            help=(
                "binary uses only direction; continuous standardises momentum "
                "by its trailing dispersion; volatility_adjusted divides "
                "momentum by trailing annualised volatility."
            ),
        )
        params["signal_scaling"] = signal_scaling
        if signal_scaling == "volatility_adjusted":
            params["volatility_window"] = st.slider(
                "Momentum volatility window (periods)",
                10,
                252,
                63,
                1,
                help=(
                    "Trailing window used to annualise each asset's own "
                    "volatility for this signal scaling — separate from the "
                    "portfolio-level volatility window in the Portfolio "
                    "section below."
                ),
            )
        with st.expander("Advanced parameters"):
            params["stop_loss_pct"], params["take_profit_pct"] = (
                _risk_control_pct_inputs("tsm")
            )
    elif strategy_name == "mean_reversion":
        params["lookback_period"] = st.slider(
            "Lookback (periods)",
            5,
            60,
            20,
            1,
            help=(
                "Trailing window used to compute the rolling mean/std (or "
                "RSI/percentile window) the chosen indicator is measured "
                "against."
            ),
        )
        indicator = st.selectbox(
            "Indicator",
            list(MEAN_REVERSION_UI_INDICATORS),
            key="mr_indicator",
            help=(
                "Which zero-centered indicator drives entries/exits/stop: "
                "a rolling z-score, RSI, or a trailing percentile rank. "
                "Switching indicator resets the thresholds below to that "
                "indicator's own defaults — a threshold tuned for one "
                "indicator's scale is not meaningful on another's."
            ),
        )
        params["indicator"] = indicator
        (
            entry_min,
            entry_max,
            entry_default,
            exit_default,
            stop_max,
            stop_default,
            step,
        ) = mean_reversion_slider_bounds(indicator)
        # Asked BEFORE the entry slider (not after) so entry's own bounds
        # can already know whether the stop slider will even be rendered
        # -- see entry_threshold_bounds's own docstring for why this
        # matters: entry must never lose reachable values (e.g. RSI's own
        # 50, percentile's own 0.49) to guard a slider that doesn't exist.
        enable_stop = st.checkbox(
            "Enable stop threshold",
            value=True,
            key="mr_enable_stop",
            help=(
                "Force the position flat when the indicator moves past this "
                "threshold, e.g. because the trailing mean itself has shifted "
                "and the entry threshold is no longer expected to revert."
            ),
        )
        # Reserved right here (directly below the checkbox it belongs to)
        # and filled in further down, once entry_threshold's own value is
        # known -- the stop slider's lower bound is entry_threshold + step,
        # so it cannot be COMPUTED before the entry slider renders, but a
        # placeholder lets it still be DISPLAYED here.
        stop_threshold_slot = st.empty()
        entry_min, entry_max = entry_threshold_bounds(
            entry_min, entry_max, stop_max, step, stop_enabled=enable_stop
        )
        entry_default = min(max(entry_default, entry_min), entry_max)
        entry_threshold = st.slider(
            "Entry threshold",
            entry_min,
            entry_max,
            entry_default,
            step,
            key=f"mr_entry_threshold_{indicator}",
            help=(
                "Open a position once the indicator moves beyond "
                "± this threshold, betting on reversion back toward normal. "
                "On this indicator's own scale — see Indicator's help above."
            ),
        )
        params["entry_threshold"] = entry_threshold
        exit_bounds = exit_threshold_bounds(entry_threshold, step)
        if exit_bounds is None:
            st.caption(
                "Exit threshold: 0.0 (the only value possible this close to "
                "zero — entry threshold minus one step is not positive)."
            )
            params["exit_threshold"] = 0.0
        else:
            exit_min, exit_max = exit_bounds
            params["exit_threshold"] = st.slider(
                "Exit threshold",
                exit_min,
                exit_max,
                min(exit_default, exit_max),
                step,
                key=f"mr_exit_threshold_{indicator}",
                help=(
                    "Close the position once the indicator reverts back "
                    "inside ± this threshold."
                ),
            )
        params["long_only"] = st.checkbox(
            "Long only",
            value=True,
            help=(
                "Off: also open short positions when the indicator rises "
                "above the entry threshold, not only long positions on a "
                "drop below it — separate from the portfolio-level 'Long "
                "only' below."
            ),
        )
        if enable_stop:
            with stop_threshold_slot:
                params["stop_threshold"] = st.slider(
                    "Stop threshold",
                    entry_threshold + step,
                    stop_max,
                    max(stop_default, entry_threshold + step),
                    step,
                    key=f"mr_stop_threshold_{indicator}",
                    help=(
                        "Indicator magnitude beyond which the position is "
                        "forced flat instead of waiting for reversion — "
                        "protects against a move that keeps extending instead "
                        "of reverting."
                    ),
                )
        else:
            params["stop_threshold"] = None
        with st.expander("Advanced parameters"):
            st.caption(
                "Independent of the indicator-based stop_threshold above — "
                "either or both can be active."
            )
            params["stop_loss_pct"], params["take_profit_pct"] = (
                _risk_control_pct_inputs("mr")
            )
    elif strategy_name == "trend_following":
        fast_window = st.slider(
            "Fast window",
            5,
            60,
            20,
            1,
            help=(
                "Short moving-average window; crossing above the slow "
                "average signals an uptrend."
            ),
        )
        params["fast_window"] = fast_window
        minimum_slow = max(30, fast_window + 1)
        params["slow_window"] = st.slider(
            "Slow window",
            minimum_slow,
            250,
            max(100, minimum_slow),
            1,
            help=(
                "Long moving-average window that defines the trend "
                "baseline; always kept larger than the fast window above."
            ),
        )
        params["long_only"] = st.checkbox(
            "Long only",
            value=True,
            help=(
                "Off: also go short when the fast average crosses below the "
                "slow average, not only long on an upward crossover — "
                "separate from the portfolio-level 'Long only' below."
            ),
        )
        with st.expander("Advanced parameters"):
            params["stop_loss_pct"], params["take_profit_pct"] = (
                _risk_control_pct_inputs("tf")
            )
    elif strategy_name == "pairs_trading":
        st.caption(
            "Pairs trading needs at least two symbols loaded above; exactly "
            "two of them (symbol_a, symbol_b) are then chosen below as the "
            "pair's legs."
        )
        if len(symbols) < 2:
            st.error("Enter at least two symbols above to configure a pairs trade.")
        else:
            params["symbol_a"] = st.selectbox(
                "Symbol A",
                symbols,
                index=0,
                help=(
                    "One leg of the pair. The strategy trades the price "
                    "spread between Symbol A and Symbol B, long one and "
                    "short the other in a hedge-ratio-weighted amount."
                ),
            )
            remaining = [s for s in symbols if s != params["symbol_a"]] or symbols
            params["symbol_b"] = st.selectbox(
                "Symbol B", remaining, index=0, help="The other leg of the pair."
            )
            params["formation_window"] = st.slider(
                "Formation window (periods)",
                60,
                500,
                252,
                10,
                help=(
                    "Trailing window used to estimate the hedge ratio (fixed "
                    "once here unless Dynamic hedge ratio is on below) and "
                    "to run the ADF cointegration test that gates entries."
                ),
            )
            pt_indicator = st.selectbox(
                "Indicator",
                list(MEAN_REVERSION_UI_INDICATORS),
                key="pt_indicator",
                help=(
                    "Which zero-centered indicator of the spread drives "
                    "entries/exits/stop -- same three choices as Mean "
                    "Reversion's own Indicator, applied to the spread "
                    "residual instead of a raw price."
                ),
            )
            params["indicator"] = pt_indicator
            (
                pt_entry_min,
                pt_entry_max,
                pt_entry_default,
                pt_exit_default,
                pt_stop_max,
                pt_stop_default,
                pt_step,
            ) = mean_reversion_slider_bounds(pt_indicator)
            # Asked BEFORE the entry slider (not after, and moved out of
            # the "Advanced pairs parameters" expander below to make that
            # possible) so entry's own bounds can already know whether the
            # stop slider will even be rendered -- see entry_threshold_
            # bounds's own docstring and the mean_reversion block above.
            pt_enable_stop = st.checkbox(
                "Enable stop threshold",
                value=True,
                key="pt_enable_stop",
                help=(
                    "Force the pair flat when the indicator moves past "
                    "this threshold, e.g. because the hedge relationship "
                    "itself has broken down."
                ),
            )
            # Reserved right here (directly below the checkbox it belongs
            # to) and filled in further down, once pt_entry's own value is
            # known -- the stop slider's lower bound is pt_entry + pt_step,
            # so it cannot be COMPUTED before the entry slider renders, but
            # a placeholder lets it still be DISPLAYED here, outside the
            # "Advanced pairs parameters" expander below.
            pt_stop_threshold_slot = st.empty()
            pt_entry_min, pt_entry_max = entry_threshold_bounds(
                pt_entry_min,
                pt_entry_max,
                pt_stop_max,
                pt_step,
                stop_enabled=pt_enable_stop,
            )
            pt_entry_default = min(max(pt_entry_default, pt_entry_min), pt_entry_max)
            params["indicator_window"] = st.slider(
                "Indicator window (periods)",
                10,
                150,
                63,
                1,
                help=(
                    "Trailing window used to compute the spread's centered "
                    "indicator, and the cadence of the ADF gate re-check."
                ),
            )
            pt_entry = st.slider(
                "Entry threshold",
                pt_entry_min,
                pt_entry_max,
                pt_entry_default,
                pt_step,
                key=f"pt_entry_threshold_{pt_indicator}",
                help=(
                    "Open the pair once the spread's indicator moves beyond "
                    "± this threshold, betting the spread reverts toward "
                    "normal."
                ),
            )
            params["entry_threshold"] = pt_entry
            if pt_enable_stop:
                with pt_stop_threshold_slot:
                    params["stop_threshold"] = st.slider(
                        "Stop threshold",
                        pt_entry + pt_step,
                        pt_stop_max,
                        max(pt_stop_default, pt_entry + pt_step),
                        pt_step,
                        key=f"pt_stop_threshold_{pt_indicator}",
                        help=(
                            "Indicator magnitude beyond which the pair is "
                            "forced flat instead of waiting for reversion."
                        ),
                    )
            else:
                params["stop_threshold"] = None
            pt_exit_bounds = exit_threshold_bounds(pt_entry, pt_step)
            if pt_exit_bounds is None:
                st.caption(
                    "Exit threshold: 0.0 (the only value possible this close "
                    "to zero — entry threshold minus one step is not "
                    "positive)."
                )
                params["exit_threshold"] = 0.0
            else:
                pt_exit_min, pt_exit_max = pt_exit_bounds
                params["exit_threshold"] = st.slider(
                    "Exit threshold",
                    pt_exit_min,
                    pt_exit_max,
                    min(pt_exit_default, pt_exit_max),
                    pt_step,
                    key=f"pt_exit_threshold_{pt_indicator}",
                    help=(
                        "Close the pair once the indicator reverts back "
                        "inside ± this threshold."
                    ),
                )
            with st.expander("Advanced pairs parameters"):
                params["dynamic_hedge_ratio"] = st.checkbox(
                    "Dynamic hedge ratio",
                    value=True,
                    help=(
                        "Re-estimate the hedge ratio on every trailing window "
                        "instead of fixing it once at the formation window."
                    ),
                )
                if st.checkbox(
                    "Enable ADF stationarity gate",
                    value=True,
                    key="pt_enable_adf_gate",
                    help=(
                        "New entries require an Augmented Dickey-Fuller test "
                        "on the trailing spread to produce a p-value at or "
                        "below the selected threshold -- lower is a "
                        "stricter mean-reversion filter and rejects more "
                        "entries. Open positions are exempt: they still "
                        "exit only on the entry/exit/stop threshold rules."
                    ),
                ):
                    params["adf_pvalue_threshold"] = st.slider(
                        "ADF p-value threshold (entry gate)",
                        0.01,
                        0.50,
                        0.10,
                        0.01,
                    )
                else:
                    params["adf_pvalue_threshold"] = None
                params["stop_loss_pct"], params["take_profit_pct"] = (
                    _risk_control_pct_inputs("pt")
                )
    return params


# --------------------------------------------------------------------------- #
# Home / header
# --------------------------------------------------------------------------- #
st.title("📈 QuantLab")
st.caption("Reproducible quantitative research and backtesting")
st.warning(
    "Research and educational tool only — **not investment advice**. "
    "Historical performance does not guarantee future results.",
    icon="⚠️",
)
with st.expander("About this platform"):
    st.markdown(
        """
        QuantLab turns a financial hypothesis into a reproducible, bias-aware
        experiment: **data → cleaning & validation → feature functions → signals →
        allocation → execution costs → delayed-execution accounting → risk metrics
        → validation → reporting**. Signals are strictly shifted before returns
        to prevent common look-ahead leakage; a custom strategy remains
        responsible for its own causal construction.

        Source: [github.com/sefaav/QuantLab](https://github.com/sefaav/QuantLab)
        · [Report an issue](https://github.com/sefaav/QuantLab/issues)
        · [Author portfolio](https://sefaav.github.io/website/index.html)
        """
    )

mode = st.segmented_control(
    "Mode",
    ["Backtest", "Walk-forward", "Strategies"],
    default="Backtest",
    key="dashboard_mode",
    help=(
        "Backtest: a single run over the full sample, optionally with a "
        "chronological holdout. Walk-forward: repeatedly select parameters "
        "on a validation block and evaluate them out-of-sample on the "
        "following test block, stitched across the whole history — this is "
        "QuantLab's grid-search mechanism. Strategies: research each "
        "strategy's own economics, mathematics and parameters, with an "
        "interactive lab -- no experiment configuration needed."
    ),
)
if mode is None:
    mode = "Backtest"

# --------------------------------------------------------------------------- #
# Sidebar configuration
# --------------------------------------------------------------------------- #
run = False
if mode != "Strategies":
    with st.sidebar:
        st.header("Experiment configuration")
        st.subheader("Instruments")
        st.caption(
            "Pick symbols from any of the three sources below — they all "
            "combine into one multi-market universe."
        )
        with st.expander("Yahoo Finance", expanded=False):
            yahoo_symbols = _yahoo_symbols_picker()
        with st.expander("Binance", expanded=False):
            binance_symbols = _binance_symbols_picker()
        with st.expander("CSV (local files)", expanded=True):
            csv_symbols = _csv_symbols_picker()

        symbols, provenance, conflicts = _combine_instrument_picks(
            yahoo_symbols, binance_symbols, csv_symbols
        )
        if conflicts:
            st.error(
                "Picked from more than one source, so source/calendar would be "
                "ambiguous — remove the duplicate from one picker: "
                + ", ".join(sorted(conflicts))
            )
        if not symbols:
            st.warning("Pick at least one symbol above to configure an instrument.")

        instrument_rows = _instrument_table(symbols, provenance)
        use_bundled_demo_data = False
        if any(row["Source"] == "csv" for row in instrument_rows):
            use_bundled_demo_data = st.toggle(
                "Allow bundled synthetic demo data",
                value=False,
                help=(
                    "If every requested CSV is absent from data/raw, use the "
                    "bundled synthetic SPY/QQQ/TLT/GLD files instead. QuantLab "
                    "never mixes local and bundled files, and unsupported "
                    "symbols still fail explicitly."
                ),
            )

        instrument_calendars = {row["Calendar"] for row in instrument_rows}
        periods_per_year: int | None = None
        if len(instrument_calendars) > 1:
            st.warning(
                "Instruments span more than one calendar "
                f"({', '.join(sorted(instrument_calendars))}), so QuantLab "
                "cannot infer a single annualisation factor automatically — "
                "set one explicitly."
            )
            # Default to the 24/7 convention as soon as any instrument actually
            # trades continuously -- a mixed portfolio that includes one is
            # closer to a "always-open" annualisation than a pure business-day
            # one, and 252 silently understates volatility/Sharpe for it.
            default_periods_per_year = 365 if "24/7" in instrument_calendars else 252
            periods_per_year = int(
                st.number_input(
                    "Periods per year (annualisation factor)",
                    min_value=1,
                    value=default_periods_per_year,
                    step=1,
                    help=(
                        "Used to annualise Sharpe, volatility and vol-targeting "
                        "across the whole portfolio. 252 for a business-day "
                        "equity convention, 365 for a continuous 24/7 one."
                    ),
                )
            )

        # Sidebar date fields need the full width to keep labels and values readable.
        start_date = st.date_input(
            "Start date",
            value=date(2019, 1, 1),
            help=(
                "Requested start of the sample. The actually observed range "
                "after data-quality filtering is reported once the backtest runs."
            ),
        )
        end_date = st.date_input(
            "End date",
            value=default_end_date(),
            help="Requested end of the sample, same caveat as Start date above.",
        )

        # Offer only frequencies compatible with every selected instrument's
        # source — the same intersection ExperimentConfig itself validates, so
        # the picker can never offer something the config would then reject.
        instrument_sources = {DataSourceName(row["Source"]) for row in instrument_rows}
        frequency_options = sorted(
            compatible_frequencies_for_sources(instrument_sources)
        )
        # '1h' has no verified-closure handling for a mixed-calendar universe
        # (that machinery only operates at daily frequency) -- offering it would
        # let the config validator reject the run only after "Run backtest" is
        # clicked, so it's excluded here too, matching what ExperimentConfig
        # itself would refuse.
        intraday_blocked_by_mixed_calendar = (
            len(instrument_calendars) > 1 and "1h" in frequency_options
        )
        if intraday_blocked_by_mixed_calendar:
            frequency_options = [f for f in frequency_options if f != "1h"]
        frequency = st.selectbox(
            "Frequency",
            frequency_options,
            index=0 if frequency_options else None,
            help=(
                "Bar size requested from the data source; also sets the "
                "annualisation factor used by every risk metric."
            ),
        )
        if not frequency_options:
            st.error(
                "No frequency is compatible with every selected source — remove "
                "one of the conflicting sources above."
            )
        elif intraday_blocked_by_mixed_calendar:
            st.caption(
                "'1h' is unavailable for a mixed-calendar universe — verified "
                "closures only work at daily frequency."
            )
        if any(row["Source"] == "csv" for row in instrument_rows):
            st.caption(
                "For CSV data, frequency controls annualisation and data-quality "
                "checks but does not resample the file. A mismatch with the "
                "observed timestamps is reported after the run."
            )
        with st.expander("Advanced data settings"):
            missing_value_policy = st.selectbox(
                "Missing value policy",
                ["drop", "forward_fill", "raise", "none"],
                index=0,
                help=(
                    "How the cleaner treats missing canonical market data bars: "
                    "drop removes affected rows; forward_fill fills up to the "
                    "limit below; raise fails the run on any gap; none leaves "
                    "gaps as-is."
                ),
            )
            if missing_value_policy == "forward_fill":
                forward_fill_limit = st.number_input(
                    "Forward-fill limit (consecutive bars)",
                    min_value=1,
                    value=1,
                    step=1,
                    help=(
                        "Maximum consecutive missing bars filled per symbol "
                        "before the gap is left as-is."
                    ),
                )
            else:
                forward_fill_limit = 1

        strategy_options = available_strategies()
        if mode == "Walk-forward":
            # buy_and_hold has no parameters (BuyAndHoldStrategy._freeze_parameters()),
            # so there is nothing for fold-by-fold validation-block selection to
            # select — walk-forward would just repeat the same signal on every
            # fold for no benefit over a single backtest.
            strategy_options = [
                name for name in strategy_options if name != "buy_and_hold"
            ]
        strategy_name = st.selectbox(
            "Strategy",
            strategy_options,
            index=0,
            help=(
                "Signal-generation method: time_series_momentum / "
                "cross_sectional_momentum (trend continuation); trend_following "
                "(moving-average trend); mean_reversion / pairs_trading "
                "(reversion to a trailing mean or spread)."
                + (
                    ""
                    if mode == "Walk-forward"
                    else " buy_and_hold (no signal, baseline exposure) is also "
                    "available here, but has no parameters to select, so it is "
                    "hidden in Walk-forward mode."
                )
                + " Its own parameters appear below."
            ),
        )

        if strategy_name != "buy_and_hold":
            st.subheader("Strategy parameters")
        strategy_parameters = _strategy_param_inputs(strategy_name, symbols)

        st.subheader("Portfolio")
        signal_scaling = strategy_parameters.get("signal_scaling")
        # Mirror ExperimentConfig's validators so an invalid combination can never
        # be selected in the first place, instead of failing only after "Run
        # backtest": equal_weight discards signal magnitude (breaks non-binary
        # time-series scaling), and volatility_adjusted already divides by
        # volatility itself (an inverse-volatility allocator would apply that
        # sizing a second time).
        allocator_note: str | None
        if strategy_name == "pairs_trading":
            allocator_options = ["signal_proportional"]
            allocator_note = (
                "Only signal_proportional is offered: pairs trading needs its "
                "signed hedge magnitude preserved exactly, not re-sized or "
                "reduced to a sign, to keep the pair's two legs offsetting."
            )
        elif (
            strategy_name == "time_series_momentum"
            and signal_scaling == "volatility_adjusted"
        ):
            allocator_options = ["signal_proportional"]
            allocator_note = (
                "Only signal_proportional is offered: volatility_adjusted "
                "scaling above already divides the signal by trailing "
                "volatility, so inverse_volatility or volatility_targeting "
                "would apply that sizing a second time."
            )
        elif (
            strategy_name in ("time_series_momentum", "cross_sectional_momentum")
            and signal_scaling == "continuous"
        ):
            allocator_options = [
                "signal_proportional",
                "inverse_volatility",
                "volatility_targeting",
            ]
            allocator_note = (
                "equal_weight is not offered: it discards signal magnitude and "
                "keeps only its sign, which would throw away the continuous "
                "scaling selected above."
            )
        else:
            allocator_options = [
                "equal_weight",
                "signal_proportional",
                "inverse_volatility",
                "volatility_targeting",
            ]
            allocator_note = None
        preferred_default = (
            "inverse_volatility"
            if "inverse_volatility" in allocator_options
            else allocator_options[0]
        )
        allocator = st.selectbox(
            "Allocator",
            allocator_options,
            index=allocator_options.index(preferred_default),
            help=(
                "equal_weight: same absolute weight on every active signal. "
                "signal_proportional: weight scales with signed signal magnitude. "
                "inverse_volatility: weight scales inversely with each asset's "
                "trailing volatility. volatility_targeting: inverse-volatility "
                "weights further scaled so total exposure tracks the target "
                "volatility below. Only allocators compatible with the selected "
                "strategy (and its signal scaling, for time_series_momentum) are "
                "listed."
            ),
        )
        if allocator_note is not None:
            st.caption(allocator_note)
        if strategy_name == "pairs_trading":
            maximum_weight = None
            long_only = False
            st.caption(
                "Per-asset weight caps and long-only are disabled to preserve the "
                "pair hedge (a pairs trade always holds one long and one short leg)."
            )
        else:
            maximum_weight = st.slider(
                "Max weight per asset",
                0.05,
                1.0,
                0.30,
                0.05,
                help=(
                    "Hard cap on any single asset's absolute target weight, "
                    "enforced as a portfolio constraint regardless of the "
                    "allocator chosen above."
                ),
            )
            long_only = st.checkbox(
                "Long only (portfolio)",
                value=False,
                help=(
                    "Reject any negative target weight at the portfolio level, on "
                    "top of whatever the strategy's own signals already allow."
                ),
            )
        rebalance_frequency = st.selectbox(
            "Rebalance frequency",
            ["daily", "weekly", "monthly", "quarterly"],
            index=2,
            help=(
                "How often target weights are recomputed and traded toward. "
                "Between rebalances, each asset's own price move drifts its "
                "executed weight (model_weight_drift, on by default); a "
                "scheduled rebalance still always trades toward its "
                "freshly-decided target, even if that target happens to "
                "numerically match the previous one."
            ),
        )
        if allocator == "volatility_targeting":
            enable_volatility_targeting = True
            st.caption("Volatility targeting is inherent to this allocator.")
        else:
            enable_volatility_targeting = st.toggle(
                "Enable volatility targeting",
                value=True,
                help=(
                    "Scale the portfolio's overall exposure toward the annual "
                    "volatility target. Disable this to keep the allocator's "
                    "unscaled weights."
                ),
            )
        if allocator in {"inverse_volatility", "volatility_targeting"} or (
            enable_volatility_targeting
        ):
            volatility_window = st.slider(
                "Volatility window (periods)",
                10,
                252,
                63,
                1,
                help=(
                    "Trailing window used to estimate realised volatility, for "
                    "both inverse-volatility sizing and volatility targeting."
                ),
            )
        else:
            volatility_window = 63
        if enable_volatility_targeting:
            target_volatility: float | None = st.slider(
                "Target volatility (annual)",
                0.05,
                0.40,
                0.12,
                0.01,
                help=(
                    "Desired annualised portfolio volatility. Exposure is "
                    "scaled, up to 'Max leverage' below, toward this target "
                    "using the volatility window above."
                ),
            )
            maximum_leverage = st.slider(
                "Max leverage",
                1.0,
                3.0,
                1.5,
                0.1,
                help=(
                    "Ceiling on the volatility-targeting scale-up, e.g. 1.5 "
                    "allows up to 150% gross exposure even if hitting the "
                    "target volatility would ask for more."
                ),
            )
        else:
            target_volatility = None
            maximum_leverage = 1.0

        if strategy_name == "pairs_trading":
            target_minimum_weight = None
            maximum_gross_exposure = None
            maximum_net_exposure = None
            target_maximum_positions = None
            maximum_turnover = None
            st.caption(
                "Advanced portfolio constraints are disabled for pairs_trading: "
                "a minimum position size, position count cap, or exposure cap "
                "could drop one leg and break the pair hedge."
            )
        else:
            # Set by the non-pairs_trading branch above whenever this branch runs.
            assert maximum_weight is not None
            with st.expander("Advanced portfolio constraints"):
                if st.checkbox(
                    "Enable minimum position size",
                    value=False,
                    help=(
                        "Reject any target weight smaller than this instead of "
                        "holding a near-zero position."
                    ),
                ):
                    target_minimum_weight = st.slider(
                        "Minimum position size",
                        0.0,
                        maximum_weight,
                        min(0.02, maximum_weight),
                        0.01,
                        help="Smallest allowed non-zero target weight per asset.",
                    )
                else:
                    target_minimum_weight = None
                if st.checkbox(
                    "Cap gross exposure",
                    value=False,
                    help=(
                        "Limit total absolute exposure (sum of |weight|) "
                        "across all assets."
                    ),
                ):
                    maximum_gross_exposure = st.slider(
                        "Max gross exposure",
                        0.1,
                        3.0,
                        1.0,
                        0.1,
                        help=(
                            "Ceiling on gross exposure, enforced on top of "
                            "Max leverage."
                        ),
                    )
                else:
                    maximum_gross_exposure = None
                if st.checkbox(
                    "Cap net exposure",
                    value=False,
                    help="Limit net directional exposure (sum of signed weights).",
                ):
                    maximum_net_exposure = st.slider(
                        "Max net exposure",
                        0.0,
                        3.0,
                        1.0,
                        0.1,
                        help=(
                            "Ceiling on |long weight - short weight| across all assets."
                        ),
                    )
                else:
                    maximum_net_exposure = None
                if st.checkbox(
                    "Cap number of positions",
                    value=False,
                    help=(
                        "Limit how many assets can be held with a non-zero "
                        "target weight at once."
                    ),
                ):
                    target_maximum_positions = st.number_input(
                        "Max number of positions",
                        min_value=1,
                        value=min(10, max(1, len(symbols))),
                        step=1,
                        help=(
                            "Largest number of simultaneously non-zero target weights."
                        ),
                    )
                else:
                    target_maximum_positions = None
                if st.checkbox(
                    "Cap turnover per period",
                    value=False,
                    help=(
                        "Limit how much total weight can change on any single "
                        "row, spreading a large rebalance over several periods."
                    ),
                ):
                    maximum_turnover = st.slider(
                        "Max turnover per period",
                        0.05,
                        2.0,
                        0.5,
                        0.05,
                        help="Maximum L1 weight change allowed on any single row.",
                    )
                else:
                    maximum_turnover = None

        st.subheader("Validation")
        validation_ratio: float | None = None
        test_ratio: float | None = None
        train_window = 500
        validation_window = 126
        test_window = 126
        expanding = True
        optimization_metric = "sharpe"
        parameter_grid: dict[str, list] = {}
        if mode == "Walk-forward":
            st.caption(
                "Select parameters on each fold's validation block and evaluate "
                "them out-of-sample on the following test block, repeated and "
                "stitched across the whole history. This mode's own Results "
                "tab shows that stitched out-of-sample evidence, not a "
                "full-sample fit."
            )
            train_window = st.number_input(
                "Train window (periods)",
                min_value=10,
                value=500,
                step=10,
                help="Training periods per fold, used to fit the strategy state.",
            )
            validation_window = st.number_input(
                "Validation window (periods)",
                min_value=5,
                value=126,
                step=5,
                help="Validation periods per fold, used to select parameters.",
            )
            test_window = st.number_input(
                "Test window (periods)",
                min_value=5,
                value=126,
                step=5,
                help=(
                    "Out-of-sample test periods per fold, stitched into the OOS series."
                ),
            )
            expanding = st.toggle(
                "Expanding training window",
                value=True,
                help=(
                    "On: each fold's training block grows to include everything "
                    "before it. Off: training slides forward, always Train "
                    "window periods long."
                ),
            )
            optimization_metric = st.selectbox(
                "Optimization metric",
                ["sharpe", "sortino", "calmar", "total_return"],
                index=0,
                help="Metric used to pick the best parameter combination on each fold.",
            )
            st.caption(
                "Parameters to search below — leave empty to use a compact, "
                "strategy-specific default grid at run time."
            )
            grid_param_names = sorted(strategy_parameter_names(strategy_name))
            selected_grid_params = st.multiselect(
                "Grid parameters",
                grid_param_names,
                help="Strategy parameters to vary across candidate values.",
            )
            for parameter_name in selected_grid_params:
                raw_values = st.text_input(
                    f"Candidate values for {parameter_name} (comma-separated)",
                    key=f"wf_grid_{parameter_name}",
                )
                parameter_grid[parameter_name] = parse_parameter_grid_values(raw_values)
            # A rough estimate only (used to warn about a slow configuration
            # before data is loaded) -- treat the universe as 24/7 only when
            # every instrument genuinely is, otherwise fall back to the
            # business-day convention.
            is_247_market = bool(instrument_rows) and all(
                row["Calendar"] == "24/7" for row in instrument_rows
            )
            estimated_backtests = estimate_walk_forward_backtest_count(
                start_date=start_date,
                end_date=end_date or default_end_date(),
                is_247_market=is_247_market,
                train_window=train_window,
                validation_window=validation_window,
                test_window=test_window,
                expanding=expanding,
                parameter_grid=parameter_grid,
            )
            if estimated_backtests <= 0:
                st.warning(
                    "No walk-forward fold fits the requested date range and "
                    "windows — widen the date range or shorten the windows.",
                    icon="⚠️",
                )
        else:
            enable_holdout = st.checkbox(
                "Chronological holdout (train / validation / test)",
                value=False,
                help=(
                    "Split one continuous backtest chronologically and report "
                    "each block separately. No fitting or parameter tuning "
                    "happens here. Treat the trailing test block as "
                    "out-of-sample only if you fixed the strategy and "
                    "parameters before inspecting it; the headline metric "
                    "cards still describe the full sample."
                ),
            )
            if enable_holdout:
                validation_ratio = st.slider(
                    "Validation fraction",
                    0.05,
                    0.4,
                    0.2,
                    0.05,
                    help=(
                        "Middle chronological slice reported separately for "
                        "manual assessment. This dashboard backtest does not "
                        "tune or select parameters automatically."
                    ),
                )
                test_ratio = st.slider(
                    "Test fraction",
                    0.05,
                    0.4,
                    0.2,
                    0.05,
                    help=(
                        "Final chronological slice, reported separately as the "
                        "'Test' block. Genuinely out-of-sample only if the "
                        "strategy and parameters were fixed before it was ever "
                        "inspected — this dashboard has no way to verify that."
                    ),
                )

        st.subheader("Costs & capital")
        initial_capital = st.number_input(
            "Initial capital",
            1_000.0,
            value=100_000.0,
            step=1_000.0,
            help=(
                "Starting portfolio value in currency units. Scales every "
                "currency-denominated figure (costs, traded notional) but not "
                "percentage returns."
            ),
        )
        risk_free_rate_percent = float(
            st.number_input(
                "Risk-free rate (annual %)",
                value=2.0,
                step=0.1,
                format="%.2f",
                help=(
                    "Annual rate used for excess-return metrics and the cash "
                    "benchmark. Enter 2 for 2%."
                ),
            )
        )
        benchmark_kind = st.selectbox(
            "Benchmark",
            options=["symbol", "equal_weight", "first_asset", "cash"],
            index=0,
            format_func={
                "symbol": "Symbol",
                "equal_weight": "Equal weight",
                "first_asset": "First asset",
                "cash": "Cash",
            }.__getitem__,
            help=(
                "Compare the strategy with an external symbol, an equal-weight "
                "portfolio, the first universe asset, or cash earning the "
                "configured risk-free rate."
            ),
        )
        # The benchmark symbol is itself an instrument (source + calendar), with
        # the same provenance rule as the table above: if it duplicates a
        # tradable instrument, its source/calendar are reused verbatim rather
        # than letting the user configure an inconsistency the config would
        # reject anyway (source/calendar must match exactly when they overlap).
        if benchmark_kind == "symbol":
            benchmark_symbol = (
                st.text_input(
                    "Benchmark symbol",
                    "SPY",
                    help="External symbol to compare the strategy against.",
                )
                .strip()
                .upper()
            )
            matching_instrument = next(
                (
                    row
                    for row in instrument_rows
                    if row["Instrument"] == benchmark_symbol
                ),
                None,
            )
            if matching_instrument is not None:
                benchmark_source = matching_instrument["Source"]
                benchmark_calendar = matching_instrument["Calendar"]
                st.caption(
                    f"{benchmark_symbol} is already a tradable instrument — its "
                    f"source ({benchmark_source}) and calendar "
                    f"({benchmark_calendar}) are reused as-is."
                )
            elif benchmark_symbol:
                detected_source = detect_source(benchmark_symbol)
                source_options = ["yahoo", "binance", "csv"]
                benchmark_source = st.selectbox(
                    "Benchmark source",
                    source_options,
                    index=source_options.index(
                        detected_source.value if detected_source else "csv"
                    ),
                    key="benchmark_source_select",
                    help="Data source for the external benchmark symbol.",
                )
                benchmark_calendar = st.text_input(
                    "Benchmark calendar",
                    detect_calendar(benchmark_symbol, DataSourceName(benchmark_source))
                    or "XNYS",
                    key="benchmark_calendar_input",
                    help="'24/7' for a continuous market, or a "
                    "pandas_market_calendars name such as XNYS, XHKG, XLON.",
                )
            else:
                benchmark_source = "csv"
                benchmark_calendar = "XNYS"
        else:
            benchmark_symbol = ""
            benchmark_source = "csv"
            benchmark_calendar = "XNYS"
            if benchmark_kind == "first_asset":
                first_symbol = symbols[0] if symbols else "the first universe symbol"
                st.caption(f"Benchmark asset: {first_symbol}")
        commission_bps = st.slider(
            "Commission (bps)",
            0.0,
            20.0,
            2.0,
            0.5,
            help=(
                "Broker commission charged per unit of traded notional, "
                "in basis points."
            ),
        )
        spread_bps = st.slider(
            "Spread (bps)",
            0.0,
            20.0,
            3.0,
            0.5,
            help=(
                "Full quoted bid-ask spread in basis points; half is charged "
                "whenever a trade crosses it."
            ),
        )
        slippage_bps = st.slider(
            "Slippage (bps)",
            0.0,
            20.0,
            2.0,
            0.5,
            help=(
                "Additional execution cost beyond commission and spread, "
                "modelling market impact under the constant slippage model "
                "below."
            ),
        )
        with st.expander("Advanced execution settings"):
            slippage_model = st.selectbox(
                "Slippage model",
                ["constant", "volume"],
                index=0,
                help=(
                    "constant applies the slippage bps above uniformly to every "
                    "trade. volume instead scales slippage with each trade's "
                    "size relative to average daily volume, using the impact "
                    "coefficient below."
                ),
            )
            if slippage_model == "volume":
                impact_coefficient = st.number_input(
                    "Volume impact coefficient",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    help=(
                        "Multiplies sqrt(order size / average daily volume) — "
                        "added on top of the slippage bps above. For liquid "
                        "instruments (e.g. SPY, QQQ) and a modest position size "
                        "relative to their average daily volume, that square "
                        "root is tiny, so even a large coefficient can leave "
                        "results looking identical to the constant model — this "
                        "term is built to matter for large orders in thin "
                        "markets, not small ones in deep markets."
                    ),
                )
            else:
                impact_coefficient = 0.1

        submission_blocked = bool(conflicts) or not symbols or not frequency_options
        # Same stable key for both -- exactly one of the two ever renders in
        # a given run (mutually exclusive on `mode`), and a stable key lets
        # callers (dashboard tests) find this specific button regardless of
        # how many OTHER sidebar buttons render before it.
        if mode == "Walk-forward":
            run = st.button(
                "Run walk-forward",
                type="primary",
                width="stretch",
                disabled=submission_blocked,
                key="run_button",
            )
        else:
            run = st.button(
                "Run backtest",
                type="primary",
                width="stretch",
                disabled=submission_blocked,
                key="run_button",
            )


def _run_and_store(
    session_key: str,
    label: str,
    compute: Callable[[Callable[[int, int], None]], Any],
) -> None:
    """Run one robustness technique, storing its result or showing the error.

    Shared by every individual "Run X" button and the "Run all" button below
    it, so there is exactly one place each technique's on-demand execution
    happens — no separate logic path for the bulk button. ``compute``
    receives a progress callback; functions that don't report progress
    (bootstrap, permutation — genuinely fast, resampling only — and the
    Backtest-mode plain sensitivity variant) simply ignore it, so the bar
    sits at an indeterminate 0% for their duration instead of stepping
    through counts.
    """
    st.session_state.pop(session_key, None)
    st.session_state.pop("report_html", None)
    st.session_state.pop("wf_report_html", None)
    title = f"{label[0].upper()}{label[1:]}"
    progress_bar = st.progress(0.0, text=f"{title}: starting…")
    try:
        st.session_state[session_key] = compute(
            _make_progress_callback(progress_bar, title)
        )
    except Exception as exc:
        logger.exception("Dashboard %s failed", label)
        st.error(f"{title} failed: {exc}")
    finally:
        progress_bar.empty()


def _make_progress_callback(
    progress_bar: Any, title: str
) -> Callable[[int, int], None]:
    """Build an ``on_progress(done, total)`` callback driving a live progress bar.

    Text/ETA come from a shared `ProgressReporter` (also used by the CLI's
    terminal progress line, for the same estimate on both interfaces) — this
    function only renders it into the Streamlit widget.
    """
    reporter = ProgressReporter(title)

    def _on_progress(done: int, total: int) -> None:
        progress_bar.progress(
            reporter.fraction(done, total), text=reporter.text(done, total)
        )

    return _on_progress


def _render_bootstrap_interpretation(confidence_level: float = 0.90) -> None:
    """Explain how to read the bootstrap summary table's columns."""
    pct = f"{confidence_level:.0%}"
    st.caption(
        f"How to read this: p_lower/p_upper form a {pct} interval across "
        "resamples of these same, already-realised returns. If a "
        "statistic's p_lower sits on the wrong side of zero (e.g. a "
        "negative CAGR or Sharpe), ordinary resampling variation in this "
        "exact history could plausibly have produced a loss — the result "
        "isn't robust to resampling yet, regardless of how good the point "
        "estimate looks."
    )


def _render_permutation_interpretation(n_iterations: int) -> None:
    """Explain how to read the permutation test's p-value."""
    floor = 1.0 / (n_iterations + 1)
    st.caption(
        "How to read this: the p-value is the fraction of random-sign "
        "permutations whose Sharpe matched or beat the real one. Below "
        "~0.05 is the conventional threshold for treating the result as "
        f"unlikely under the no-edge null. With {n_iterations} iterations, "
        f"the p-value can't go below {floor:.4f} — landing there just means "
        "none of the permutations beat the real Sharpe, not that the edge "
        "is certain; more iterations would refine the number further."
    )


def _expected_data_hash(result: BacktestResult) -> str:
    """Return the displayed result's data hash, or fail loudly if missing."""
    data_hash = result.metadata.get("data_hash")
    if not isinstance(data_hash, str):
        raise RuntimeError(
            "The displayed result has no valid data hash. Run it again "
            "before running this check."
        )
    return data_hash


def _render_robustness_tab(result: BacktestResult) -> None:
    """Render chronological holdout evidence and on-demand robustness checks."""
    holdout_report = result.metadata.get("holdout_report")
    st.markdown("#### Chronological holdout (train / validation / test)")
    if holdout_report:
        # Two-way train/test holdouts omit the validation row. Labeled
        # plainly "Test", not "out-of-sample": whether it's genuinely OOS
        # depends on parameters having been fixed before looking at it, a
        # property of the user's own workflow this table can't verify.
        blocks = [("Train", "train"), ("Test", "test")]
        if "validation_metrics" in holdout_report:
            blocks.insert(1, ("Validation", "validation"))
        rows = [
            {
                "Block": block,
                "Period": f"{period[0][:10]} to {period[1][:10]}",
                "Sharpe": metrics.get("sharpe_ratio", float("nan")),
                "CAGR": metrics.get("cagr", float("nan")),
                "Max Drawdown": metrics.get("max_drawdown", float("nan")),
            }
            for block, key in blocks
            for metrics, period in [
                (
                    holdout_report[f"{key}_metrics"],
                    holdout_report[f"{key}_period"],
                )
            ]
        ]
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "CAGR": st.column_config.NumberColumn(format="percent"),
                "Max Drawdown": st.column_config.NumberColumn(format="percent"),
            },
        )
    else:
        st.info(
            "No out-of-sample holdout is attached to this run — tick "
            "**Chronological holdout** in the sidebar's Validation section "
            "and re-run the backtest to see train/validation/test metrics "
            "here instead of full-sample-only numbers."
        )

    st.markdown("#### Stress tests")
    st.caption(
        "Re-runs the backtest under several perturbations to check how "
        "sensitive the result is to modelling assumptions: commission x2/x5, "
        "slippage x2, one extra period of execution delay, the 10 best days "
        "removed, and — only when the universe has more than 2 symbols — a "
        "**reduced universe** that drops the last configured symbol and "
        "re-runs on what remains, to see how much the result leans on that "
        "one asset. Not run automatically since it re-executes the backtest "
        "several times. Re-running **Run backtest** for any reason "
        "(including just enabling holdout) invalidates these numbers — "
        "rerun stress tests afterwards if you want them in the downloaded "
        "report."
    )
    if st.button("Run stress tests"):
        _run_and_store(
            "stress_tests",
            "stress tests",
            lambda progress: run_dashboard_stress_tests(
                result.config, _expected_data_hash(result), on_progress=progress
            ),
        )
    stress = st.session_state.get("stress_tests")
    if stress is not None:
        st.dataframe(
            stress,
            width="stretch",
            hide_index=True,
            column_config={
                "scenario": st.column_config.TextColumn("Scenario"),
                "total_return": st.column_config.NumberColumn(
                    "Total return", format="percent"
                ),
                "cagr": st.column_config.NumberColumn("CAGR", format="percent"),
                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn(
                    "Max drawdown", format="percent"
                ),
            },
        )

    st.markdown("#### Bootstrap")
    st.caption(
        "Resamples the realised returns (block bootstrap) into a "
        "distribution of plausible CAGR/Sharpe/drawdown/final-value "
        "outcomes. Resamples already-realised returns only — no strategy "
        "re-run, no parameter optimization."
    )
    bootstrap_n_iterations = st.number_input(
        "Bootstrap iterations",
        min_value=100,
        value=1000,
        step=100,
        key="bt_bootstrap_n",
    )
    bootstrap_block_size = st.number_input(
        "Block size",
        min_value=1,
        value=1,
        step=1,
        key="bt_bootstrap_block",
        help="Consecutive-period block length; 1 resamples individual periods.",
    )
    if st.button("Run bootstrap"):
        _run_and_store(
            "bootstrap_summary",
            "bootstrap",
            lambda _progress: run_dashboard_bootstrap(
                result.config,
                result.returns,
                n_iterations=bootstrap_n_iterations,
                block_size=bootstrap_block_size,
            ),
        )
    bootstrap_summary = st.session_state.get("bootstrap_summary")
    if bootstrap_summary is not None:
        st.dataframe(
            format_bootstrap_summary(bootstrap_summary),
            width="stretch",
            hide_index=True,
        )
        _render_bootstrap_interpretation(
            result.config.robustness.bootstrap.confidence_level
        )

    st.markdown("#### Permutation Monte Carlo")
    st.caption(
        "Randomly flips the sign of excess returns to test the realised "
        "Sharpe against a no-edge random-sign null. A low p-value is "
        "evidence against that specific null, not a probability of future "
        "profitability."
    )
    permutation_n_iterations = st.number_input(
        "Permutation iterations",
        min_value=100,
        value=1000,
        step=100,
        key="bt_permutation_n",
    )
    if st.button("Run permutation test"):
        _run_and_store(
            "permutation_test",
            "permutation test",
            lambda _progress: run_dashboard_permutation_test(
                result.config, result.returns, n_iterations=permutation_n_iterations
            ),
        )
    permutation = st.session_state.get("permutation_test")
    if permutation is not None:
        perm_cols = st.columns(2)
        perm_cols[0].metric("Real Sharpe", f"{permutation['real_sharpe']:.2f}")
        perm_cols[1].metric("p-value", f"{permutation['p_value']:.4f}")
        _render_permutation_interpretation(int(permutation["n_iterations"]))

    st.markdown("#### Parameter sensitivity")
    st.caption(
        "Re-runs the backtest across a 2-parameter grid to show how "
        "sensitive the result is to the exact parameter choice. Boolean/"
        "structural parameters (e.g. long_only) aren't offered — they "
        "change which other parameters are even meaningful, so sweeping "
        "them isn't well-defined for a 2D heatmap."
    )
    sensitivity_param_names = sorted(
        strategy_sweepable_parameter_names(result.config.strategy_name)
    )
    sens_col1, sens_col2 = st.columns(2)
    with sens_col1:
        sensitivity_x = st.selectbox(
            "Parameter (x-axis)", sensitivity_param_names, key="bt_sens_x"
        )
        sensitivity_x_values = st.text_input(
            "Candidate values (x, comma-separated)", key="bt_sens_x_values"
        )
    with sens_col2:
        remaining_params = [p for p in sensitivity_param_names if p != sensitivity_x]
        sensitivity_y = st.selectbox(
            "Parameter (y-axis)", remaining_params, key="bt_sens_y"
        )
        sensitivity_y_values = st.text_input(
            "Candidate values (y, comma-separated)", key="bt_sens_y_values"
        )
    sensitivity_ready = bool(
        sensitivity_x
        and sensitivity_y
        and sensitivity_x_values
        and sensitivity_y_values
    )

    def _run_sensitivity() -> None:
        _run_and_store(
            "sensitivity",
            "parameter sensitivity",
            lambda _progress: run_dashboard_sensitivity(
                result.config,
                _expected_data_hash(result),
                sensitivity_x,
                parse_parameter_grid_values(sensitivity_x_values),
                sensitivity_y,
                parse_parameter_grid_values(sensitivity_y_values),
            ),
        )

    if st.button("Run parameter sensitivity", disabled=not sensitivity_ready):
        _run_sensitivity()
    sensitivity = st.session_state.get("sensitivity")
    if sensitivity is not None:
        # Read the axes back off the result itself, not the (possibly since
        # changed) sidebar selection above -- otherwise changing the axis
        # pickers after a run without re-running would render a stale
        # result under mismatched labels, or crash pivoting on a column the
        # stored result never had.
        used_x, used_y = infer_sensitivity_parameter_columns(sensitivity)
        if (used_x, used_y) != (sensitivity_x, sensitivity_y):
            st.caption(
                f"Showing the last run's axes ({used_x} / {used_y}) — "
                "the pickers above have changed since. Run again to update."
            )
        render_sensitivity_heatmap(st, sensitivity, used_x, used_y)

    st.divider()
    if st.button("Run all robustness tests", type="secondary"):
        _run_and_store(
            "stress_tests",
            "stress tests",
            lambda progress: run_dashboard_stress_tests(
                result.config, _expected_data_hash(result), on_progress=progress
            ),
        )
        _run_and_store(
            "bootstrap_summary",
            "bootstrap",
            lambda _progress: run_dashboard_bootstrap(
                result.config,
                result.returns,
                n_iterations=bootstrap_n_iterations,
                block_size=bootstrap_block_size,
            ),
        )
        _run_and_store(
            "permutation_test",
            "permutation test",
            lambda _progress: run_dashboard_permutation_test(
                result.config, result.returns, n_iterations=permutation_n_iterations
            ),
        )
        if sensitivity_ready:
            _run_sensitivity()
        else:
            st.caption(
                "Skipped parameter sensitivity: pick both parameters and "
                "candidate values above first."
            )
        st.rerun()


def _render_walk_forward_robustness_tab(wf: WalkForwardResult) -> None:
    """Render per-fold selection evidence and on-demand robustness checks.

    Stress tests and parameter sensitivity here always re-run the whole
    walk-forward selection process per scenario/cell
    (``run_dashboard_walk_forward_stress_tests`` /
    ``run_dashboard_walk_forward_sensitivity``) rather than reuse Backtest
    mode's plain-backtest variants — this tab must never silently show
    numbers from a different validation method than the one in effect.
    """
    oos_result = wf.oos_result
    assert oos_result is not None  # guarded by _execute_walk_forward

    st.markdown("#### Fold summary")
    st.caption(
        "Selected parameters and realised metrics for each walk-forward "
        "fold's out-of-sample test block."
    )
    st.dataframe(
        wf.summary_table(),
        width="stretch",
        hide_index=True,
        column_config={
            "test_return": st.column_config.NumberColumn(format="percent"),
            "test_sharpe": st.column_config.NumberColumn(format="%.2f"),
            "validation_score": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.markdown("#### Parameter stability across folds")
    stability = wf.parameter_stability()
    if stability:
        st.caption(
            "Coefficient of variation of each selected numeric parameter "
            "across folds — lower means walk-forward selection was more "
            "consistent, though this alone does not establish robustness."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "parameter": list(stability),
                    "coefficient_of_variation": list(stability.values()),
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "coefficient_of_variation": st.column_config.NumberColumn(
                    format="%.3f"
                ),
            },
        )
    else:
        st.info(
            "Not enough numeric parameter selections across folds to "
            "compute stability (e.g. a single fold, or no grid parameters)."
        )

    st.markdown("#### Stress tests")
    st.caption(
        "**Commission x2/x5 and slippage x2** re-select parameters under "
        "the new costs from a per-candidate weight cache built once for "
        "all three — cheap to re-score since signals and portfolio "
        "allocation never depend on execution costs, but this still "
        "genuinely re-selects, it does not just rescale the baseline's "
        "fixed weights. **Execution delay +1** and — with more than 2 "
        "symbols — **reduced universe** instead re-run the whole "
        "walk-forward process (every fold's validation-block selection, "
        "then OOS reconstruction) from scratch, since delay changes the "
        "weights themselves and a reduced universe changes signal "
        "generation; substantially slower than the three cost-only "
        "scenarios. **Best 10 days removed** does not re-run anything — "
        "it directly zeroes the 10 best days already in the baseline OOS "
        "returns above, since removing realised returns changes no "
        "configuration to re-select against."
    )
    if st.button("Run stress tests", key="wf_run_stress"):
        _run_and_store(
            "wf_stress_tests",
            "stress tests",
            lambda progress: run_dashboard_walk_forward_stress_tests(
                oos_result.config,
                wf,
                _expected_data_hash(oos_result),
                on_progress=progress,
            ),
        )
    wf_stress = st.session_state.get("wf_stress_tests")
    if wf_stress is not None:
        st.dataframe(
            wf_stress,
            width="stretch",
            hide_index=True,
            column_config={
                "scenario": st.column_config.TextColumn("Scenario"),
                "total_return": st.column_config.NumberColumn(
                    "Total return", format="percent"
                ),
                "cagr": st.column_config.NumberColumn("CAGR", format="percent"),
                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn(
                    "Max drawdown", format="percent"
                ),
            },
        )

    st.markdown("#### Bootstrap")
    st.caption(
        "Resamples the stitched out-of-sample returns (block bootstrap) — "
        "resamples already-realised returns only, so this is exactly as "
        "fast as in Backtest mode regardless of the walk-forward cost above."
    )
    wf_bootstrap_n_iterations = st.number_input(
        "Bootstrap iterations",
        min_value=100,
        value=1000,
        step=100,
        key="wf_bootstrap_n",
    )
    wf_bootstrap_block_size = st.number_input(
        "Block size", min_value=1, value=1, step=1, key="wf_bootstrap_block"
    )
    if st.button("Run bootstrap", key="wf_run_bootstrap"):
        _run_and_store(
            "wf_bootstrap_summary",
            "bootstrap",
            lambda _progress: run_dashboard_bootstrap(
                oos_result.config,
                oos_result.returns,
                n_iterations=wf_bootstrap_n_iterations,
                block_size=wf_bootstrap_block_size,
            ),
        )
    wf_bootstrap_summary = st.session_state.get("wf_bootstrap_summary")
    if wf_bootstrap_summary is not None:
        st.dataframe(
            format_bootstrap_summary(wf_bootstrap_summary),
            width="stretch",
            hide_index=True,
        )
        _render_bootstrap_interpretation(
            oos_result.config.robustness.bootstrap.confidence_level
        )

    st.markdown("#### Permutation Monte Carlo")
    st.caption(
        "Randomly flips the sign of the stitched OOS excess returns to "
        "test the realised Sharpe against a no-edge random-sign null."
    )
    wf_permutation_n_iterations = st.number_input(
        "Permutation iterations",
        min_value=100,
        value=1000,
        step=100,
        key="wf_permutation_n",
    )
    if st.button("Run permutation test", key="wf_run_permutation"):
        _run_and_store(
            "wf_permutation_test",
            "permutation test",
            lambda _progress: run_dashboard_permutation_test(
                oos_result.config,
                oos_result.returns,
                n_iterations=wf_permutation_n_iterations,
            ),
        )
    wf_permutation = st.session_state.get("wf_permutation_test")
    if wf_permutation is not None:
        wf_perm_cols = st.columns(2)
        wf_perm_cols[0].metric("Real Sharpe", f"{wf_permutation['real_sharpe']:.2f}")
        wf_perm_cols[1].metric("p-value", f"{wf_permutation['p_value']:.4f}")
        _render_permutation_interpretation(int(wf_permutation["n_iterations"]))

    st.markdown("#### Parameter sensitivity")
    st.caption(
        "Re-runs the **whole walk-forward process** for each grid cell "
        "(the two swept parameters pinned, no further inner optimization) "
        "— can be slow: every cell costs roughly a full walk-forward run. "
        "Boolean/structural parameters (e.g. long_only) aren't offered — "
        "they change which other parameters are even meaningful, so "
        "sweeping them isn't well-defined for a 2D heatmap."
    )
    wf_sensitivity_param_names = sorted(
        strategy_sweepable_parameter_names(oos_result.config.strategy_name)
    )
    wf_sens_col1, wf_sens_col2 = st.columns(2)
    with wf_sens_col1:
        wf_sensitivity_x = st.selectbox(
            "Parameter (x-axis)", wf_sensitivity_param_names, key="wf_sens_x"
        )
        wf_sensitivity_x_values = st.text_input(
            "Candidate values (x, comma-separated)", key="wf_sens_x_values"
        )
    with wf_sens_col2:
        wf_remaining_params = [
            p for p in wf_sensitivity_param_names if p != wf_sensitivity_x
        ]
        wf_sensitivity_y = st.selectbox(
            "Parameter (y-axis)", wf_remaining_params, key="wf_sens_y"
        )
        wf_sensitivity_y_values = st.text_input(
            "Candidate values (y, comma-separated)", key="wf_sens_y_values"
        )
    wf_sensitivity_ready = bool(
        wf_sensitivity_x
        and wf_sensitivity_y
        and wf_sensitivity_x_values
        and wf_sensitivity_y_values
    )

    def _run_wf_sensitivity() -> None:
        _run_and_store(
            "wf_sensitivity",
            "parameter sensitivity",
            lambda progress: run_dashboard_walk_forward_sensitivity(
                oos_result.config,
                _expected_data_hash(oos_result),
                wf_sensitivity_x,
                parse_parameter_grid_values(wf_sensitivity_x_values),
                wf_sensitivity_y,
                parse_parameter_grid_values(wf_sensitivity_y_values),
                on_progress=progress,
            ),
        )

    if st.button(
        "Run parameter sensitivity",
        key="wf_run_sensitivity",
        disabled=not wf_sensitivity_ready,
    ):
        _run_wf_sensitivity()
    wf_sensitivity = st.session_state.get("wf_sensitivity")
    if wf_sensitivity is not None:
        # See the Backtest-mode sensitivity section above for why the axes
        # are read back off the result itself, not the sidebar's current
        # (possibly since changed) selection.
        used_x, used_y = infer_sensitivity_parameter_columns(wf_sensitivity)
        if (used_x, used_y) != (wf_sensitivity_x, wf_sensitivity_y):
            st.caption(
                f"Showing the last run's axes ({used_x} / {used_y}) — "
                "the pickers above have changed since. Run again to update."
            )
        render_sensitivity_heatmap(st, wf_sensitivity, used_x, used_y)

    st.divider()
    if st.button("Run all robustness tests", key="wf_run_all", type="secondary"):
        _run_and_store(
            "wf_stress_tests",
            "stress tests",
            lambda progress: run_dashboard_walk_forward_stress_tests(
                oos_result.config,
                wf,
                _expected_data_hash(oos_result),
                on_progress=progress,
            ),
        )
        _run_and_store(
            "wf_bootstrap_summary",
            "bootstrap",
            lambda _progress: run_dashboard_bootstrap(
                oos_result.config,
                oos_result.returns,
                n_iterations=wf_bootstrap_n_iterations,
                block_size=wf_bootstrap_block_size,
            ),
        )
        _run_and_store(
            "wf_permutation_test",
            "permutation test",
            lambda _progress: run_dashboard_permutation_test(
                oos_result.config,
                oos_result.returns,
                n_iterations=wf_permutation_n_iterations,
            ),
        )
        if wf_sensitivity_ready:
            _run_wf_sensitivity()
        else:
            st.caption(
                "Skipped parameter sensitivity: pick both parameters and "
                "candidate values above first."
            )
        st.rerun()


def _collect_backtest_robustness_evidence() -> tuple[
    dict[str, object], tuple[object, ...]
]:
    """Gather every on-demand Backtest-mode robustness result for the report.

    Also folds in the current strategy's own results diagnostics (e.g.
    pairs trading's correlation/spread/ADF section), when its profile
    declares any -- otherwise a report downloaded from the dashboard would
    show that section live in the Results tab but silently drop it from
    the exported HTML. A failed diagnostics computation still contributes
    a status note rather than vanishing without a trace (mirrors the
    Results tab's own warning; see `_render_strategy_diagnostics`).
    """
    evidence: dict[str, object] = {}
    cache_parts: list[object] = []
    for session_key, label in (
        ("stress_tests", "stress_tests"),
        ("bootstrap_summary", "bootstrap"),
        ("permutation_test", "permutation_test"),
        ("sensitivity", "sensitivity"),
    ):
        value = st.session_state.get(session_key)
        cache_parts.append(id(value) if value is not None else None)
        if value is not None:
            evidence[label] = value

    result = st.session_state.get("result")
    diagnostics = st.session_state.get("strategy_diagnostics")
    diagnostics_error = st.session_state.get("strategy_diagnostics_error")
    diagnostics_cache_part: object = diagnostics_error
    if result is not None:
        profile = get_profile(result.config.strategy_name)
        if profile is not None and profile.results_diagnostics is not None:
            if diagnostics is not None:
                section = profile.results_diagnostics.report_section(diagnostics)
                evidence[profile.results_diagnostics.key] = section
                # A live Results-tab widget belonging to this diagnostic
                # (e.g. its own forward-return horizon or window slider)
                # can change this section's content without changing
                # `diagnostics` itself -- `report_section` reads such a
                # widget's current value straight from session_state (see
                # `live_widget_value`). Cache on the section's own
                # rendered content, not `id(diagnostics)`, so the exported
                # report is correctly regenerated when the user moves one
                # of those widgets instead of silently reusing a stale
                # cached HTML report.
                diagnostics_cache_part = (
                    tuple(section.table.columns),
                    tuple(section.table.itertuples(index=False, name=None)),
                    section.chart_data_uri,
                    section.note,
                )
            elif diagnostics_error is not None:
                evidence[profile.results_diagnostics.key] = DiagnosticsSection(
                    table=pd.DataFrame(
                        {"Status": [f"Could not compute: {diagnostics_error}"]}
                    ),
                )
    cache_parts.append(diagnostics_cache_part)
    return evidence, tuple(cache_parts)


def _collect_walk_forward_robustness_evidence(
    wf: WalkForwardResult,
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Gather fold evidence plus every on-demand Walk-forward robustness result."""
    evidence: dict[str, object] = {"walk_forward": wf.summary_table()}
    cache_parts: list[object] = [id(wf)]
    for session_key, label in (
        ("wf_stress_tests", "stress_tests"),
        ("wf_bootstrap_summary", "bootstrap"),
        ("wf_permutation_test", "permutation_test"),
        ("wf_sensitivity", "sensitivity"),
    ):
        value = st.session_state.get(session_key)
        cache_parts.append(id(value) if value is not None else None)
        if value is not None:
            evidence[label] = value
    return evidence, tuple(cache_parts)


def _render_report_tab(
    result: BacktestResult,
    robustness: dict[str, object] | None,
    *,
    cache_key_extra: object,
    session_key: str,
) -> None:
    """Render the HTML report tab, shared by Backtest and Walk-forward modes."""
    st.markdown("### Research report")
    cache_key = (id(result), cache_key_extra)
    cached_report = cast(
        tuple[tuple[int, object], tuple[str, list[str]]] | None,
        st.session_state.get(session_key),
    )
    if cached_report is not None and cached_report[0] == cache_key:
        html, chart_warnings = cached_report[1]
    else:
        chart_warnings = []
        html = result.to_html(robustness=robustness, warnings=chart_warnings)
        st.session_state[session_key] = (cache_key, (html, chart_warnings))
    if chart_warnings:
        # Surface individual chart failures without discarding the report.
        st.warning(
            "Some charts could not be rendered into this report:\n"
            + "\n".join(f"- {w}" for w in chart_warnings)
        )
    st.download_button(
        "Download HTML report",
        html.encode("utf-8"),
        file_name=f"{result.config.experiment_name}_report.html",
        mime="text/html",
    )
    st.iframe(html, height=800)


# --------------------------------------------------------------------------- #
# Run and results
# --------------------------------------------------------------------------- #
def _collect_inputs() -> dict:
    """Return the current sidebar values used to build the experiment config."""
    instruments = [
        {
            "symbol": row["Instrument"],
            "source": row["Source"],
            "calendar": row["Calendar"],
        }
        for row in instrument_rows
    ]
    benchmark = (
        {
            "symbol": benchmark_symbol,
            "source": benchmark_source,
            "calendar": benchmark_calendar,
        }
        if benchmark_kind == "symbol" and benchmark_symbol
        else None
    )
    inputs: dict = {
        "experiment_name": f"dashboard_{strategy_name}",
        "instruments": instruments,
        "use_bundled_demo_data": use_bundled_demo_data,
        "start_date": start_date,
        "end_date": end_date or default_end_date(),
        "frequency": frequency,
        "missing_value_policy": missing_value_policy,
        "forward_fill_limit": forward_fill_limit,
        "strategy_name": strategy_name,
        "strategy_parameters": strategy_parameters,
        "allocator": allocator,
        "maximum_weight": maximum_weight,
        "long_only": long_only,
        "target_minimum_weight": target_minimum_weight,
        "maximum_gross_exposure": maximum_gross_exposure,
        "maximum_net_exposure": maximum_net_exposure,
        "target_maximum_positions": target_maximum_positions,
        "maximum_turnover": maximum_turnover,
        "target_volatility": target_volatility,
        "volatility_window": volatility_window,
        "maximum_leverage": maximum_leverage,
        "rebalance_frequency": rebalance_frequency,
        "initial_capital": initial_capital,
        "benchmark_kind": benchmark_kind,
        "benchmark": benchmark,
        "periods_per_year": periods_per_year,
        "risk_free_rate": risk_free_rate_percent / 100.0,
        "commission_bps": commission_bps,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "slippage_model": slippage_model,
        "impact_coefficient": impact_coefficient,
    }
    if mode == "Walk-forward":
        inputs["validation_method"] = "walk_forward"
        inputs["train_window"] = train_window
        inputs["validation_window"] = validation_window
        inputs["test_window"] = test_window
        inputs["expanding"] = expanding
        inputs["optimization_metric"] = optimization_metric
        inputs["parameter_grid"] = parameter_grid
    else:
        inputs["validation_ratio"] = validation_ratio
        inputs["test_ratio"] = test_ratio
    return inputs


def _clear_backtest_result_state() -> None:
    """Remove artefacts that no longer describe a successful backtest."""
    for key in (
        "result",
        "result_inputs",
        "warnings",
        "stress_tests",
        "bootstrap_summary",
        "permutation_test",
        "sensitivity",
        "report_html",
        "strategy_diagnostics",
        "strategy_diagnostics_error",
    ):
        st.session_state.pop(key, None)


def _clear_walk_forward_result_state() -> None:
    """Remove artefacts that no longer describe a successful walk-forward run."""
    for key in (
        "wf_result",
        "wf_result_inputs",
        "wf_warnings",
        "wf_stress_tests",
        "wf_bootstrap_summary",
        "wf_permutation_test",
        "wf_sensitivity",
        "wf_report_html",
    ):
        st.session_state.pop(key, None)


def _compute_strategy_diagnostics(data: pd.DataFrame, config: ExperimentConfig) -> Any:
    """Compute the current strategy's results diagnostics, if it declares any.

    Generic by design -- reads ``profile.results_diagnostics`` off whichever
    profile is registered for ``config.strategy_name`` instead of branching
    on a strategy name, so a strategy without diagnostics (the common case)
    costs nothing beyond one dict lookup. ``data`` must be the EXACT frame
    the displayed backtest itself ran on (see
    ``run_dashboard_backtest_with_data``) -- reloading it separately could
    silently observe different data for a remote source between the two
    calls.
    """
    profile = get_profile(config.strategy_name)
    if profile is None or profile.results_diagnostics is None:
        return None
    return profile.results_diagnostics.compute(data, config)


def _render_strategy_diagnostics(result: BacktestResult) -> None:
    """Render the Results-tab diagnostics declared by the current strategy's profile.

    Renders nothing for a strategy without a declared ``results_diagnostics``
    (the default) -- never an empty section, per the Strategy Explorer
    architecture's generic, name-free dispatch. When the strategy DOES
    declare one but computing it raised, surfaces that failure as a visible
    warning instead of the diagnostics silently vanishing from the page.
    """
    profile = get_profile(result.config.strategy_name)
    if profile is None or profile.results_diagnostics is None:
        return
    error = st.session_state.get("strategy_diagnostics_error")
    if error is not None:
        st.warning(
            f"Backtest completed, but strategy diagnostics could not be "
            f"computed: {error}"
        )
        return
    diagnostics = st.session_state.get("strategy_diagnostics")
    if diagnostics is None:
        return
    profile.results_diagnostics.render(st, diagnostics)


def _render_walk_forward_diagnostics_note(oos_result: BacktestResult) -> None:
    """Explain why Strategy Explorer results diagnostics aren't shown here.

    Each walk-forward fold can select different strategy parameters than
    the base config and covers only that fold's slice of history -- a
    diagnostic computed once, on the full history with the base config's
    parameters, would not actually describe what any individual fold
    traded. Shown only for a strategy that DOES declare
    ``results_diagnostics`` (so a strategy without one, the common case,
    never gets a stray note here).
    """
    profile = get_profile(oos_result.config.strategy_name)
    if profile is None or profile.results_diagnostics is None:
        return
    st.caption(
        "This strategy has Strategy Explorer results diagnostics (see "
        "Backtest mode) -- not shown here because each walk-forward fold "
        "can select different parameters than this run's base config, so "
        "a single full-history diagnostic would not accurately describe "
        "every fold's own selection."
    )


def _execute_backtest() -> None:
    inputs = _collect_inputs()
    _clear_backtest_result_state()
    with st.spinner("Running backtest…"):
        try:
            config = build_config_from_inputs(inputs)
            result, warnings, data = run_dashboard_backtest_with_data(config)
        except Exception as exc:
            logger.exception("Dashboard backtest failed")
            st.error(f"Backtest failed: {exc}")
            return
        diagnostics: Any = None
        diagnostics_error: str | None = None
        try:
            diagnostics = _compute_strategy_diagnostics(data, config)
        except Exception as exc:
            logger.exception("Strategy Explorer results diagnostics failed")
            diagnostics_error = str(exc)

    st.session_state["result"] = result
    st.session_state["result_inputs"] = inputs
    st.session_state["warnings"] = warnings
    st.session_state["strategy_diagnostics"] = diagnostics
    st.session_state["strategy_diagnostics_error"] = diagnostics_error


def _execute_walk_forward() -> None:
    inputs = _collect_inputs()
    _clear_walk_forward_result_state()
    progress_bar = st.progress(0.0, text="Walk-forward: starting…")
    try:
        config = build_config_from_inputs(inputs)
        wf, warnings = run_dashboard_walk_forward(
            config,
            on_progress=_make_progress_callback(progress_bar, "Walk-forward"),
        )
    except Exception as exc:
        logger.exception("Dashboard walk-forward failed")
        st.error(f"Walk-forward failed: {exc}")
        return
    finally:
        progress_bar.empty()
    if wf.oos_result is None:
        st.error(
            "No walk-forward fold fit the requested date range and "
            "windows — widen the date range or shorten the windows."
        )
        return

    st.session_state["wf_result"] = wf
    st.session_state["wf_result_inputs"] = inputs
    st.session_state["wf_warnings"] = warnings


if run:
    if mode == "Walk-forward":
        _execute_walk_forward()
    else:
        _execute_backtest()

if mode == "Walk-forward":
    wf = st.session_state.get("wf_result")
    if wf is None:
        st.info(
            "Configure an experiment in the sidebar and click **Run walk-forward**."
        )
    else:
        oos_result = wf.oos_result
        assert oos_result is not None  # guarded by _execute_walk_forward
        if _collect_inputs() != st.session_state.get("wf_result_inputs"):
            st.warning(
                "Sidebar configuration has changed since this result was "
                "computed — click **Run walk-forward** to refresh it.",
                icon="⚠️",
            )
        data_warnings = st.session_state.get("wf_warnings", [])
        frequency_warnings = [
            w for w in data_warnings if "does not match the declared frequency" in w
        ]
        other_warnings = [w for w in data_warnings if w not in frequency_warnings]
        if frequency_warnings:
            st.error(
                "**Frequency mismatch detected** — the metrics below use "
                "the wrong annualisation factor and should not be "
                "trusted:\n" + "\n".join(f"- {w}" for w in frequency_warnings),
                icon="🚫",
            )
        for warning in other_warnings:
            st.caption(f"⚠️ {warning}")

        tab_results, tab_trades, tab_robustness, tab_report = st.tabs(
            ["Results", "Trades", "Robustness", "Report"],
            on_change="rerun",
            key="dashboard_active_tab",
        )
        if tab_results.open:
            with tab_results:
                render_metric_cards(st, oos_result)
                render_charts(st, oos_result)
                render_gross_net_comparison(st, oos_result)
                render_exposure_and_cost_charts(st, oos_result)
                _render_walk_forward_diagnostics_note(oos_result)
        if tab_trades.open:
            with tab_trades:
                render_trade_table(st, oos_result)
        if tab_robustness.open:
            with tab_robustness:
                _render_walk_forward_robustness_tab(wf)
        if tab_report.open:
            with tab_report:
                wf_robustness, wf_cache_parts = (
                    _collect_walk_forward_robustness_evidence(wf)
                )
                _render_report_tab(
                    oos_result,
                    wf_robustness,
                    cache_key_extra=wf_cache_parts,
                    session_key="wf_report_html",
                )
elif mode == "Strategies":
    selected_strategy = st.session_state.get("explorer_strategy")
    if selected_strategy is None:
        explorer_gallery.render(st)
    else:
        explorer_detail.render(st, selected_strategy)
else:
    result = st.session_state.get("result")
    if result is None:
        st.info("Configure an experiment in the sidebar and click **Run backtest**.")
    else:
        # Warn when the current controls no longer describe the saved result.
        if _collect_inputs() != st.session_state.get("result_inputs"):
            st.warning(
                "Sidebar configuration has changed since this result was "
                "computed — click **Run backtest** to refresh it.",
                icon="⚠️",
            )
        data_warnings = st.session_state.get("warnings", [])
        frequency_warnings = [
            w for w in data_warnings if "does not match the declared frequency" in w
        ]
        other_warnings = [w for w in data_warnings if w not in frequency_warnings]
        if frequency_warnings:
            # Frequency mismatches invalidate all annualised metrics.
            st.error(
                "**Frequency mismatch detected** — the metrics below use "
                "the wrong annualisation factor and should not be "
                "trusted:\n" + "\n".join(f"- {w}" for w in frequency_warnings),
                icon="🚫",
            )
        for warning in other_warnings:
            st.caption(f"⚠️ {warning}")

        # Dynamic tabs render only the selected tab; the stable key also
        # supports tests.
        tab_results, tab_trades, tab_robustness, tab_report = st.tabs(
            ["Results", "Trades", "Robustness", "Report"],
            on_change="rerun",
            key="dashboard_active_tab",
        )
        if tab_results.open:
            with tab_results:
                render_metric_cards(st, result)
                render_charts(st, result)
                render_gross_net_comparison(st, result)
                render_exposure_and_cost_charts(st, result)
                _render_strategy_diagnostics(result)
        if tab_trades.open:
            with tab_trades:
                render_trade_table(st, result)
        if tab_robustness.open:
            with tab_robustness:
                _render_robustness_tab(result)
        if tab_report.open:
            with tab_report:
                bt_robustness, bt_cache_parts = _collect_backtest_robustness_evidence()
                _render_report_tab(
                    result,
                    bt_robustness or None,
                    cache_key_extra=bt_cache_parts,
                    session_key="report_html",
                )
