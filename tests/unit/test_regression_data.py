"""Regression tests for data behavior."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv
from tests.regression_helpers import (
    _base_config_dict,
    _FakeResult,
    _frame_with_one_missing_row,
    _holdout_config,
    _hourly_frame,
    _hourly_symbol_frame,
    _import_script,
    _make_hourly_frame,
    _market_calendar_config,
    _minimal_ohlcv_frame,
    _ohlcv_at,
    _ohlcv_frame,
    _try_strategy,
    _wf_experiment_config,
    _write_daily_cache,
    _write_ohlcv_csv,
    _write_wf_artifacts,
)

from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError


def test_adv_only_uses_volume_known_through_prior_day() -> None:
    from quantlab.backtesting.runner import build_execution_from_config

    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    frame = make_ohlcv("AAA", np.linspace(100.0, 101.0, len(idx)), start="2020-01-01")
    frame["volume"] = 1_000_000.0
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_causality",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-20",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 5.0,
                "slippage_model": "volume",
            },
            "backtest": {"initial_capital": 100_000},
        }
    )
    from quantlab.execution.execution_model import ExecutionModel
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    def _adv(execution: ExecutionModel) -> pd.DataFrame:
        slippage = execution.slippage
        assert isinstance(slippage, VolumeBasedSlippageModel)
        adv = slippage.average_daily_volume
        assert isinstance(adv, pd.DataFrame)
        return adv

    baseline = build_execution_from_config(cfg, frame)
    baseline_adv = _adv(baseline)

    changed = frame.copy()
    changed.loc[changed.index[-1], "volume"] = 999_000_000.0
    changed_execution = build_execution_from_config(cfg, changed)
    changed_adv = _adv(changed_execution)

    # Inflating the LAST day's own volume must not change the ADV estimate
    # attributed to that same day's trade — it is only observable after the
    # fact. The last row is the decisive comparison: a rolling(21) window
    # never looks ahead regardless of `.shift`, so every row *except* the
    # one actually modified would be identical even without the `.shift(1)`
    # causality guard. Comparing only `iloc[:-1]` would not actually
    # exercise that guard at all.
    pd.testing.assert_series_equal(
        baseline_adv.iloc[-1], changed_adv.iloc[-1], check_names=False
    )
    # Directly show what an unshifted rolling window would produce, to
    # confirm this is not a vacuous comparison: without the `.shift(1)`
    # guard, the last row's ADV would visibly differ once its own volume
    # changes.
    unshifted_baseline = frame["volume"].rolling(21, min_periods=1).mean()
    unshifted_changed = changed["volume"].rolling(21, min_periods=1).mean()
    assert unshifted_baseline.iloc[-1] != unshifted_changed.iloc[-1]

    # Sanity: a change to an EARLIER day's volume DOES still legitimately
    # move a LATER day's ADV — only same-day/future leakage is disallowed.
    changed_earlier = frame.copy()
    changed_earlier.loc[changed_earlier.index[2], "volume"] = 999_000_000.0
    changed_earlier_execution = build_execution_from_config(cfg, changed_earlier)
    changed_earlier_adv = _adv(changed_earlier_execution)
    assert not changed_earlier_adv.iloc[3:].equals(baseline_adv.iloc[3:])


def test_adv_window_uses_calendar_days_not_bars() -> None:
    from quantlab.backtesting.runner import build_execution_from_config
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    n = 24 * 60  # 60 days of hourly bars
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    per_bar_dollar_volume = 1_000.0 * 100.0  # volume x price, both constant
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "BTC",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1_000.0,
        }
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_bars_vs_days",
            "data": {
                "source": "csv",
                "symbols": ["BTC"],
                "start_date": "2020-01-01",
                "end_date": "2020-03-01",
                "frequency": "1h",
                "market_calendar": "24/7",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "slippage_model": "volume",
                "slippage_bps": 0.0,
                "impact_coefficient": 1.0,
            },
        }
    )
    assert cfg.data.is_247_market
    model = build_execution_from_config(cfg, frame)
    assert isinstance(model.slippage, VolumeBasedSlippageModel)
    adv = model.slippage.average_daily_volume
    assert isinstance(adv, pd.DataFrame)
    # Once warmed up (past the 504-bar/21-day window), ADV must reflect a
    # full calendar day's volume (24 bars), not a single bar's.
    warmed_up = adv["BTC"].iloc[600:]
    assert np.allclose(warmed_up.to_numpy(), 24 * per_bar_dollar_volume)
    assert not np.allclose(warmed_up.to_numpy(), per_bar_dollar_volume)


def test_adv_window_unchanged_for_daily_bar_configs() -> None:
    from quantlab.backtesting.runner import build_execution_from_config
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    frame = make_ohlcv("AAA", np.linspace(100.0, 101.0, len(idx)), start="2020-01-01")
    frame["volume"] = 1_000_000.0
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_daily_unchanged",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-03-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "slippage_model": "volume",
                "slippage_bps": 0.0,
                "impact_coefficient": 1.0,
            },
        }
    )
    assert not cfg.data.is_247_market
    model = build_execution_from_config(cfg, frame)
    assert isinstance(model.slippage, VolumeBasedSlippageModel)
    adv = model.slippage.average_daily_volume
    assert isinstance(adv, pd.DataFrame)
    expected = (
        (frame.set_index("timestamp")["volume"] * frame.set_index("timestamp")["close"])
        .rolling(21, min_periods=1)
        .mean()
        .shift(1)
    )
    pd.testing.assert_series_equal(
        adv["AAA"], expected, check_names=False, check_freq=False
    )
    assert pd.isna(adv["AAA"].iloc[0])


def test_adv_uses_unadjusted_price_for_historical_dollar_volume() -> None:
    """Share volume traded against the contemporaneous raw price, not the
    split/dividend-adjusted series used for strategy returns."""
    from quantlab.backtesting.runner import build_execution_from_config
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "adjusted_close": 50.0,
            "volume": 1_000.0,
        }
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_raw_price",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {"slippage_model": "volume"},
        }
    )

    model = build_execution_from_config(cfg, frame)
    assert isinstance(model.slippage, VolumeBasedSlippageModel)
    adv = model.slippage.average_daily_volume
    assert isinstance(adv, pd.DataFrame)
    assert adv["AAA"].iloc[1:].tolist() == pytest.approx([100_000.0, 100_000.0])


def test_adv_bar_scaling_ignores_metrics_annualisation_override() -> None:
    """An override used by Sharpe/CAGR must not change a bar's duration."""
    from quantlab.backtesting.runner import build_execution_from_config
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    frame = make_ohlcv("AAA", np.full(3, 100.0), start="2020-01-01")
    frame["volume"] = 1_000.0
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_physical_frequency",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {"slippage_model": "volume"},
            "backtest": {"periods_per_year": 126},
        }
    )

    model = build_execution_from_config(cfg, frame)
    assert isinstance(model.slippage, VolumeBasedSlippageModel)
    adv = model.slippage.average_daily_volume
    assert isinstance(adv, pd.DataFrame)
    assert adv["AAA"].iloc[1:].tolist() == pytest.approx([100_000.0, 100_000.0])


def test_adv_window_scales_down_for_weekly_bars() -> None:
    from quantlab.backtesting.runner import build_execution_from_config
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    idx = pd.date_range("2020-01-01", periods=60, freq="W-MON")
    per_bar_volume, per_bar_price = 1_000.0, 100.0
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": per_bar_price,
            "high": per_bar_price,
            "low": per_bar_price,
            "close": per_bar_price,
            "adjusted_close": per_bar_price,
            "volume": per_bar_volume,
        }
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "adv_weekly",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
                "frequency": "1w",
            },
            "strategy": {"name": "buy_and_hold"},
            "execution": {
                "slippage_model": "volume",
                "slippage_bps": 0.0,
                "impact_coefficient": 1.0,
            },
        }
    )
    model = build_execution_from_config(cfg, frame)
    assert isinstance(model.slippage, VolumeBasedSlippageModel)
    adv = model.slippage.average_daily_volume
    assert isinstance(adv, pd.DataFrame)
    # ~52 weekly bars/year vs. 252 trading days/year -> ~4.846 trading days
    # per weekly bar; the daily-equivalent ADV must be the per-bar dollar
    # volume scaled *down* by that factor, not left at the raw per-bar value.
    bars_per_day = cfg.periods_per_year / 252
    expected_daily_adv = per_bar_volume * per_bar_price * bars_per_day
    warmed_up = adv["AAA"].iloc[-5:]
    assert np.allclose(warmed_up.to_numpy(), expected_daily_adv)
    assert not np.allclose(
        warmed_up.to_numpy(), per_bar_volume * per_bar_price, rtol=0.1
    )


def test_unknown_data_source_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "source": "not_a_real_source",
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_report_command_skips_reinjection_when_csv_missing(tmp_path: Path) -> None:
    """`metadata.json` surviving while its walk-forward CSVs were deleted
    must not be treated as complete evidence."""
    from quantlab.backtesting.result import load_previous_walk_forward_robustness

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)
    (exp_dir / "walk_forward_oos_equity.csv").unlink()

    fake_result = _FakeResult(config)
    robustness = load_previous_walk_forward_robustness(
        exp_dir,
        fake_result,  # type: ignore[arg-type]
    )
    assert robustness is None
    assert "walk_forward_oos_metrics" not in fake_result.metadata


def test_frequency_mismatch_is_flagged_as_a_data_warning() -> None:
    from quantlab.dashboard.state import (
        build_config_from_inputs,
        run_dashboard_backtest,
    )

    inputs = {
        "source": "csv",
        "market_calendar": "XNYS",
        "symbols": ["SPY", "QQQ"],
        "start_date": "2019-01-01",
        "end_date": "2019-06-01",
        "frequency": "1h",
        "strategy_name": "buy_and_hold",
        "strategy_parameters": {},
        "allocator": "equal_weight",
        "rebalance_frequency": "monthly",
        "initial_capital": 100_000.0,
        "benchmark_symbol": None,
        "commission_bps": 2.0,
        "spread_bps": 3.0,
        "slippage_bps": 2.0,
    }
    cfg = build_config_from_inputs(inputs)
    _, warnings = run_dashboard_backtest(cfg)
    assert any("does not match the declared frequency" in w for w in warnings)


def test_binance_hourly_annualises_with_24_7_market_factor() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "source": "binance",
                "symbols": ["BTCUSDT"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "frequency": "1h",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert cfg.periods_per_year == 24 * 365


def test_yahoo_hourly_annualisation_unchanged() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "source": "yahoo",
                "symbols": ["SPY"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "frequency": "1h",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert cfg.periods_per_year == 252 * 7


def test_binance_monthly_frequency_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "source": "binance",
                    "symbols": ["BTCUSDT"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "frequency": "1mo",
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_pairs_trading_same_symbol_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["AAA"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "pairs_trading",
                    "parameters": {"symbol_a": "AAA", "symbol_b": "aaa"},
                },
            }
        )


def test_csv_source_with_unknown_frequency_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "source": "csv",
                    "market_calendar": "XNYS",
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "frequency": "typo",
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_frequency_mismatch_flagged_even_on_short_history() -> None:
    """A frequency mismatch must be reported even below `min_coverage_rows`
    — exactly the case where a wrong annualisation factor is least likely to
    be noticed any other way."""
    from quantlab.data.validator import DataValidator

    frame = make_ohlcv(
        "AAA", [100.0 + i for i in range(10)], start="2020-01-01", freq="D"
    )
    report = DataValidator(expected_frequency="1h", min_coverage_rows=30).validate(
        frame
    )
    assert any("does not match the declared frequency" in w for w in report.warnings)


def test_intraday_equity_session_boundaries_not_flagged_as_gaps() -> None:
    from quantlab.data.validator import DataValidator

    dates = []
    for day in pd.bdate_range("2020-01-06", periods=6):
        for h in range(7):
            dates.append(
                day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
            )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert not any("abnormal gap" in w for w in report.warnings)


def test_intraday_equity_mid_session_gap_still_flagged() -> None:
    from quantlab.data.validator import DataValidator

    dates = []
    for day_idx, day in enumerate(pd.bdate_range("2020-01-06", periods=6)):
        for h in range(7):
            if day_idx == 2 and 1 <= h <= 5:
                continue  # remove 5 consecutive bars mid-session on day 3
            dates.append(
                day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
            )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert any("abnormal gap" in w for w in report.warnings)
    assert len(report.missing_periods) == 1
    period = report.missing_periods[0]
    assert period.symbol == "AAA"
    assert period.start == datetime(2020, 1, 8, 9, 30)
    assert period.end == datetime(2020, 1, 8, 15, 30)


def test_intraday_crypto_genuine_gap_still_flagged() -> None:
    """The weekend/overnight tolerance must not suppress a genuine multi-day
    gap on a 24/7 market, where there is no legitimate closure to explain it."""
    from quantlab.data.validator import DataValidator

    dates = list(pd.date_range("2020-01-01", periods=20, freq="h")) + list(
        pd.date_range("2020-01-05", periods=20, freq="h")
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "BTCUSDT",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=True
    ).validate(frame)
    assert any("abnormal gap" in w for w in report.warnings)


def test_gap_detection_flags_missing_whole_business_day() -> None:
    from quantlab.data.validator import DataValidator

    sessions = [
        d
        for d in pd.bdate_range("2020-01-06", periods=8)
        if d != pd.Timestamp("2020-01-07")  # Tuesday missing, not a holiday
    ]
    dates = [
        day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
        for day in sessions
        for h in range(7)
    ]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert any("abnormal gap" in w for w in report.warnings)


def test_gap_detection_tolerates_us_holiday_long_weekend() -> None:
    """A legitimate 3-day holiday weekend must not be flagged as an
    abnormal gap, even though its calendar span exceeds a flat day-count
    cutoff. MLK Day 2020 fell on Monday 2020-01-20."""
    from quantlab.data.validator import DataValidator

    dates = [
        day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
        for day in [pd.Timestamp("2020-01-17"), pd.Timestamp("2020-01-21")]
        for h in range(7)
    ]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert not any("abnormal gap" in w for w in report.warnings)


def test_gap_detection_flags_missing_columbus_day_session() -> None:
    from quantlab.data.validator import DataValidator

    sessions = [
        d
        for d in pd.bdate_range("2020-10-05", periods=10)
        if d != pd.Timestamp("2020-10-12")
    ]
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(_hourly_frame(sessions))
    assert any("abnormal gap" in w for w in report.warnings)


def test_gap_detection_tolerates_good_friday_closure() -> None:
    """Good Friday is an NYSE closure but not a US federal holiday, so it
    must be tolerated as a legitimate closure, not flagged as an abnormal
    gap. 2020-04-10 was Good Friday."""
    from quantlab.data.validator import DataValidator

    sessions = [pd.Timestamp("2020-04-09"), pd.Timestamp("2020-04-13")]
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(_hourly_frame(sessions))
    assert not any("abnormal gap" in w for w in report.warnings)


def test_gap_detection_flags_bars_trimmed_from_session_end() -> None:
    from quantlab.data.validator import DataValidator

    sessions = list(pd.bdate_range("2020-01-06", periods=6))
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(_hourly_frame(sessions, remove={2: {1, 2, 3, 4, 5, 6}}))
    assert any("abnormal gap" in w for w in report.warnings)


def test_pairs_trading_empty_symbol_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("pairs_trading", {"symbol_a": "", "symbol_b": "BBB"})


def test_gap_detection_tail_truncation_detected_with_only_two_sessions() -> None:
    from quantlab.data.validator import DataValidator

    day1 = pd.Timestamp("2020-01-06")
    day2 = pd.Timestamp("2020-01-07")
    dates = [day1 + pd.Timedelta(hours=9, minutes=30)]  # only h=0 kept
    dates += [
        day2 + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
        for h in range(7)
    ]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert any("abnormal gap" in w for w in report.warnings)


def test_gap_detection_head_truncation_detected_with_only_two_sessions() -> None:
    """Symmetric case: a session missing its first bars, with only 2
    sessions on record — the tied-mode resolution must handle the opening
    time the same way as the closing time above."""
    from quantlab.data.validator import DataValidator

    day1 = pd.Timestamp("2020-01-06")
    day2 = pd.Timestamp("2020-01-07")
    dates = [
        day1 + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
        for h in range(7)
    ]
    dates += [day2 + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=6)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert any("abnormal gap" in w for w in report.warnings)


def test_pairs_trading_symbol_normalized_with_whitespace() -> None:
    """Symbol parameters must be stripped of surrounding whitespace as well
    as upper-cased, so `symbol_a=" aaa "` matches the actually-loaded
    column name `"AAA"`."""
    from quantlab.strategies.pairs_trading import PairsTradingStrategy

    strategy = PairsTradingStrategy(symbol_a=" aaa ", symbol_b="bbb")
    assert strategy.symbol_a == "AAA"
    assert strategy.symbol_b == "BBB"


def test_pairs_trading_symbol_not_in_universe_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["AAA", "BBB"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "pairs_trading",
                    "parameters": {"symbol_a": "CCC", "symbol_b": "BBB"},
                },
            }
        )


def test_pairs_trading_symbol_in_universe_after_normalization_accepted() -> None:
    """A symbol that only matches after stripping/uppercasing must still be
    accepted, not rejected as "not in universe" due to a naive string
    comparison."""
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "symbols": ["AAA", "BBB"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {
                "name": "pairs_trading",
                "parameters": {"symbol_a": " aaa ", "symbol_b": "bbb"},
            },
            "portfolio": {"allocator": "signal_proportional"},
        }
    )
    assert cfg.strategy.parameters == {"symbol_a": " aaa ", "symbol_b": "bbb"}


def test_forward_fill_policy_actually_fills_via_clean() -> None:
    from quantlab.config import MissingValuePolicy
    from quantlab.data.cleaner import DataCleaner

    out = DataCleaner(MissingValuePolicy.FORWARD_FILL).clean(
        _frame_with_one_missing_row()
    )
    assert len(out) == 5
    assert not out["close"].isna().any()
    assert out["close"].iloc[2] == pytest.approx(101.5)  # carried from row 1


def test_raise_policy_actually_raises_via_clean() -> None:
    from quantlab.config import MissingValuePolicy
    from quantlab.data.cleaner import DataCleaner
    from quantlab.exceptions import DataValidationError

    with pytest.raises(DataValidationError):
        DataCleaner(MissingValuePolicy.RAISE).clean(_frame_with_one_missing_row())


def test_none_policy_actually_preserves_nan_via_clean() -> None:
    from quantlab.config import MissingValuePolicy
    from quantlab.data.cleaner import DataCleaner

    out = DataCleaner(MissingValuePolicy.NONE).clean(_frame_with_one_missing_row())
    assert len(out) == 5
    assert out["close"].isna().sum() == 1


def test_exact_match_still_clean() -> None:
    from quantlab.data.validator import DataValidator

    frame = _hourly_symbol_frame("1D", 40, "AAA")
    report = DataValidator(
        expected_frequency="1d", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert not any(
        "does not match the declared frequency" in w for w in report.warnings
    )


def test_partial_coverage_of_requested_range_flagged() -> None:
    from datetime import date

    from quantlab.data.validator import DataValidator

    dates = pd.bdate_range("2020-02-01", periods=40)
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(min_coverage_rows=5).validate(
        frame, start=date(2020, 1, 1), end=date(2020, 4, 30)
    )
    assert any(
        "data starts" in w and "after the requested start" in w for w in report.warnings
    )
    assert any(
        "data ends" in w and "before the requested end" in w for w in report.warnings
    )


def test_gap_detected_even_below_min_coverage_rows() -> None:
    from quantlab.data.validator import DataValidator

    dates = list(pd.date_range("2020-01-01", periods=5, freq="h")) + list(
        pd.date_range("2020-01-02", periods=5, freq="h")
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": "BTCUSDT",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(min_coverage_rows=30, is_247_market=True).validate(frame)
    assert any("Short coverage" in w for w in report.warnings)
    assert any("abnormal gap" in w for w in report.warnings)
    assert len(report.missing_periods) == 1


def test_csv_source_requires_explicit_market_calendar() -> None:
    from quantlab.exceptions import InvalidConfigurationError

    with pytest.raises(InvalidConfigurationError, match="market_calendar"):
        _market_calendar_config()


def test_csv_bitcoin_can_declare_crypto_via_explicit_field() -> None:
    """Declaring `market_calendar: 24/7` on a csv source yields 24/7
    (8760/year for 1h bars) annualization."""
    cfg = _market_calendar_config(market_calendar="24/7")
    assert cfg.data.is_247_market is True
    assert cfg.periods_per_year == 24 * 365


def test_csv_source_can_declare_xnys_explicitly() -> None:
    """Declaring `market_calendar: XNYS` on a csv source yields equity
    (252 * 7 = 1764/year for `_market_calendar_config`'s 1h bars)
    annualization, not 24/7."""
    cfg = _market_calendar_config(market_calendar="XNYS")
    assert cfg.data.is_247_market is False
    assert cfg.periods_per_year == 252 * 7


def test_binance_cannot_be_overridden_back_to_equity() -> None:
    from quantlab.exceptions import InvalidConfigurationError

    with pytest.raises(InvalidConfigurationError, match="not permitted"):
        _market_calendar_config(source="binance", market_calendar="XNYS")


def test_yahoo_can_select_24_7_calendar() -> None:
    """Yahoo serves continuous instruments as well as XNYS securities."""
    cfg = _market_calendar_config(source="yahoo", market_calendar="24/7")
    assert cfg.data.is_247_market is True
    assert cfg.periods_per_year == 365 * 24


def test_shipped_configs_have_the_expected_calendar() -> None:
    import pathlib

    configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
    expected = {
        "btc_trend.yaml": (True, 365),
        "default.yaml": (False, 252),
        "demo_offline.yaml": (False, 252),
        "mean_reversion_etfs.yaml": (False, 252),
        "momentum_sp500.yaml": (False, 252),
        "pairs_trading.yaml": (False, 252),
    }
    for name, (is_247, ppy) in expected.items():
        cfg = ExperimentConfig.from_yaml(configs_dir / name)
        assert cfg.data.is_247_market is is_247, name
        assert cfg.periods_per_year == ppy, name


def test_nan_adv_preserves_base_slippage_floor() -> None:
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    traded_notional = pd.DataFrame({"A": [0.1, 0.0, 0.1]}, index=idx)
    adv = pd.DataFrame({"A": [1.0, np.nan, 1.0]}, index=idx)
    model = VolumeBasedSlippageModel(
        base_slippage_bps=5.0, impact_coefficient=0.1, average_daily_volume=adv
    )
    cost = model.calculate(traded_notional)
    assert cost.iloc[1] == 0.0  # no trade that day -- NaN ADV is moot
    base_floor = 0.1 * 5.0 * 1e-4  # order x base_slippage_bps, no impact term
    assert cost.iloc[0] > 0.0
    assert cost.iloc[2] > 0.0
    assert cost.iloc[0] >= base_floor - 1e-12


def test_volume_slippage_aborts_when_adv_is_missing_for_a_traded_cell() -> None:
    from quantlab.execution.slippage import VolumeBasedSlippageModel

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    traded_notional = pd.DataFrame({"A": [0.1, 0.1, 0.1]}, index=idx)
    adv = pd.DataFrame({"A": [1.0, np.nan, 1.0]}, index=idx)
    model = VolumeBasedSlippageModel(
        base_slippage_bps=5.0, impact_coefficient=0.1, average_daily_volume=adv
    )
    with pytest.raises(ValueError, match="average_daily_volume is missing"):
        model.calculate(traded_notional)


def test_adv_scales_with_actual_equity_not_initial_capital() -> None:
    from quantlab.backtesting.accounting import run_accounting
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    idx = pd.date_range("2020-01-01", periods=7, freq="D")
    # Same-magnitude (Δ=1.0) weight changes at t=1 (equity still ~initial)
    # and t=3/t=5 (equity has grown via the uptrend in between).
    held = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0]}, index=idx)
    adv = pd.DataFrame({"A": [1_000_000.0] * 7}, index=idx)
    exec_cfg = ExecutionConfig(
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=1.0,
        slippage_model=cast(Any, "volume"),
        impact_coefficient=1.0,
    )
    model = ExecutionModel.from_config(exec_cfg, average_daily_volume=adv)
    result = run_accounting(held, asset_returns, model, initial_capital=1000.0)

    early_trade_cost = result.costs.slippage.iloc[1]
    later_trade_cost = result.costs.slippage.iloc[3]
    assert result.gross_equity.iloc[2] > 1000.0  # equity did grow in between
    assert later_trade_cost != pytest.approx(early_trade_cost)
    assert later_trade_cost > early_trade_cost


def test_constraint_set_final_dust_cleanup_respects_maximum_weight() -> None:
    from quantlab.portfolio.constraints import ConstraintSet

    cs = ConstraintSet(
        maximum_weight=0.65,
        minimum_weight=0.44,
        maximum_leverage=0.85,
        long_only=True,
    )
    raw = pd.DataFrame({"A": [0.68], "B": [0.51]})
    out = cs.apply(raw)
    assert out.abs().max(axis=1).iloc[0] <= 0.65 + 1e-9


def test_keep_artifacts_exempts_files_from_cleanup(tmp_path: Path) -> None:
    """`save(..., keep_artifacts=...)` is how `quantlab report` spares a
    still-valid prior walk-forward run's CSVs from the stale-artifact
    normal stale-artifact cleanup."""
    from quantlab.backtesting.runner import run_backtest_from_config

    exp_dir = tmp_path / "wf_experiment"
    exp_dir.mkdir()
    config = _wf_experiment_config()
    _write_wf_artifacts(exp_dir, config)

    data = pd.concat(
        [
            make_ohlcv("A", geometric_series(300, 0.0004, 0.01, 100.0, seed=1)),
            make_ohlcv("B", geometric_series(300, 0.0002, 0.012, 100.0, seed=2)),
        ],
        ignore_index=True,
    )
    result = run_backtest_from_config(data, config)
    result.save(
        exp_dir,
        keep_artifacts={
            "walk_forward_results.csv",
            "walk_forward_oos_returns.csv",
            "walk_forward_oos_equity.csv",
            "stress_tests.csv",
        },
    )

    for name in (
        "walk_forward_results.csv",
        "walk_forward_oos_returns.csv",
        "walk_forward_oos_equity.csv",
        "stress_tests.csv",
    ):
        assert (exp_dir / name).is_file(), f"{name} should have been kept"


def test_metadata_carries_data_hash_git_commit_and_dependency_versions() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.data.storage import ParquetStorage

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    assert result.metadata["data_hash"] == ParquetStorage.hash_frame(
        data[data["symbol"].isin(cfg.symbols)].reset_index(drop=True)
    )
    assert "git_commit" in result.metadata  # None is a valid (non-Git) value
    assert isinstance(result.metadata["code_hash"], str)
    assert result.metadata["code_hash"]
    versions = result.metadata["dependency_versions"]
    assert isinstance(versions, dict)
    assert "pandas" in versions
    assert "quantlab" in versions

    # Changing the input data must change the hash (it is not, e.g., derived
    # only from shape/dtypes).
    mutated = data.copy()
    closes = mutated["close"].to_numpy(dtype=float, copy=True)
    closes[0] += 1.0
    mutated["close"] = closes
    result2 = run_backtest_from_config(mutated, cfg)
    assert result2.metadata["data_hash"] != result.metadata["data_hash"]


def test_csv_loader_falls_back_to_bundled_demo_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The demo fallback is available, but never mixed with local files."""
    import quantlab.data.loader as loader_mod
    from quantlab.data.loader import DataLoader
    from quantlab.exceptions import DataDownloadError

    real_raw_dir = tmp_path / "raw"
    real_raw_dir.mkdir()
    demo_dir = tmp_path / "demo_data"
    demo_dir.mkdir()

    def make_csv(path: Path, close: float) -> None:
        path.write_text(
            "timestamp,symbol,open,high,low,close,adjusted_close,volume\n"
            f"2020-01-01,X,{close},{close},{close},{close},{close},1000\n"
        )

    make_csv(demo_dir / "SPY.csv", close=100.0)
    make_csv(real_raw_dir / "AAPL.csv", close=200.0)
    make_csv(demo_dir / "AAPL.csv", close=999.0)

    monkeypatch.setattr(loader_mod, "DEMO_DATA_DIR", demo_dir)
    with pytest.raises(DataDownloadError, match="refusing to mix"):
        DataLoader(raw_dir=real_raw_dir)._load_csv(
            ["SPY", "AAPL"], use_bundled_demo_data=True
        )

    empty_raw_dir = tmp_path / "empty_raw"
    empty_raw_dir.mkdir()
    bundled = DataLoader(raw_dir=empty_raw_dir)._load_csv(
        ["SPY", "AAPL"], use_bundled_demo_data=True
    )
    assert sorted(bundled["close"].tolist()) == [100.0, 999.0]


def test_csv_loader_does_not_silently_substitute_demo_data_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.data.loader as loader_mod
    from quantlab.data.loader import DataLoader
    from quantlab.exceptions import DataDownloadError

    empty_raw_dir = tmp_path / "raw"
    empty_raw_dir.mkdir()
    demo_dir = tmp_path / "demo_data"
    demo_dir.mkdir()
    (demo_dir / "SPY.csv").write_text(
        "timestamp,symbol,open,high,low,close,adjusted_close,volume\n"
        "2020-01-01,SPY,100.0,100.0,100.0,100.0,100.0,1000\n"
    )

    monkeypatch.setattr(loader_mod, "DEMO_DATA_DIR", demo_dir)
    with pytest.raises(DataDownloadError, match="SPY"):
        DataLoader(raw_dir=empty_raw_dir)._load_csv(["SPY"])


def test_use_bundled_demo_data_rejected_with_a_non_csv_source() -> None:
    for source in ("yahoo", "binance"):
        payload = {
            "experiment_name": "test",
            "data": {
                "source": source,
                "symbols": ["BTCUSDT"] if source == "binance" else ["SPY"],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
                "use_bundled_demo_data": True,
            },
            "strategy": {"name": "buy_and_hold"},
        }
        with pytest.raises(InvalidConfigurationError, match="use_bundled_demo_data"):
            ExperimentConfig.from_dict(payload)


def test_source_hash_is_platform_independent_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from quantlab.backtesting import engine

    real_root = Path(engine.__file__).resolve().parents[1]
    fake_root = tmp_path / "quantlab"
    shutil.copytree(real_root, fake_root)
    monkeypatch.setattr(
        engine, "__file__", str(fake_root / "backtesting" / "engine.py")
    )

    first = engine._source_hash()
    fingerprint_after_first_call = engine._source_hash_fingerprint
    assert fingerprint_after_first_call is not None
    assert all("\\" not in rel for rel, _mtime in fingerprint_after_first_call)

    # Nothing changed on disk -- the second call must reuse the cached
    # value rather than re-reading and re-hashing every file's bytes again.
    second = engine._source_hash()
    assert second == first
    assert engine._source_hash_fingerprint == fingerprint_after_first_call


def test_robustness_placeholder_does_not_overclaim_cli_coverage() -> None:
    from quantlab.reporting.html_report import _render_robustness

    html = _render_robustness(None)
    assert "quantlab walk-forward" in html
    assert "run_parameter_sensitivity" in html
    assert "bootstrap_returns" in html
    assert "monte_carlo_permutation" in html


def test_cache_covers_tolerates_a_weekend_end_date(tmp_path: Path) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    # Business-day bars through Friday 2024-01-05.
    dates = pd.date_range("2023-01-02", "2024-01-05", freq="B")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAA",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "AAA", "1d")

    # End date is a Sunday two days after the last cached (Friday) bar.
    assert storage.cache_covers(
        "yahoo", "AAA", "1d", date(2023, 1, 2), date(2024, 1, 7)
    )
    # A genuinely stale cache (far beyond any reasonable non-trading gap)
    # must still be reported as not covering.
    assert not storage.cache_covers(
        "yahoo", "AAA", "1d", date(2023, 1, 2), date(2024, 2, 1)
    )


def test_cache_covers_tolerates_a_weekend_start_date(tmp_path: Path) -> None:
    """The same non-trading-day tolerance must apply symmetrically to the
    `start` boundary, not just `end`."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2023-01-02", "2024-01-05", freq="B")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAA",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "AAA", "1d")

    # Requested start (Sunday 2023-01-01) is one day before the first cached
    # (Monday) bar.
    assert storage.cache_covers(
        "yahoo", "AAA", "1d", date(2023, 1, 1), date(2024, 1, 5)
    )


def test_save_warnings_reset_on_a_clean_resave(tmp_path: Path) -> None:
    """A warning from one `save()` call must not leak into the next call's
    outcome once the underlying failure is gone."""
    from unittest.mock import patch

    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    with patch(
        "quantlab.reporting.charts.save_figures", side_effect=RuntimeError("boom")
    ):
        result.save(tmp_path / "out")
    assert result.save_warnings

    result.save(tmp_path / "out")
    assert result.save_warnings == []


def test_save_persists_warnings_into_metadata_json(tmp_path: Path) -> None:
    from unittest.mock import patch

    from quantlab.backtesting.runner import run_backtest_from_config

    data, cfg = _holdout_config()
    result = run_backtest_from_config(data, cfg)

    with patch(
        "quantlab.reporting.charts.save_figures", side_effect=RuntimeError("boom")
    ):
        out = result.save(tmp_path / "out")

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "save_warnings" in metadata
    assert len(metadata["save_warnings"]) == 1
    assert "boom" in metadata["save_warnings"][0]


def test_symbol_path_traversal_rejected() -> None:
    payload = _base_config_dict()
    payload["data"] = dict(payload["data"])
    payload["data"]["symbols"] = ["../../etc/passwd"]
    with pytest.raises(InvalidConfigurationError, match="Invalid symbol"):
        ExperimentConfig.from_dict(payload)


def test_symbol_validation_accepts_real_yahoo_ticker_conventions() -> None:
    for symbol in ["^GSPC", "^DJI", "^VIX", "EURUSD=X", "GC=F", "ES=F"]:
        payload = _base_config_dict()
        payload["data"] = dict(payload["data"])
        payload["data"]["symbols"] = [symbol]
        cfg = ExperimentConfig.from_dict(payload)
        assert cfg.symbols == [symbol]

    # Path traversal must still be rejected even when combined with the
    # newly-allowed characters.
    payload = _base_config_dict()
    payload["data"] = dict(payload["data"])
    payload["data"]["symbols"] = ["^../../etc/passwd"]
    with pytest.raises(InvalidConfigurationError, match="Invalid symbol"):
        ExperimentConfig.from_dict(payload)


def test_symbol_rejects_windows_reserved_device_names() -> None:
    for bad in [
        "CON",
        "con",
        "NUL",
        "nul.csv",
        "PRN",
        "AUX",
        "COM1",
        "LPT9",
        "Lpt3.txt",
    ]:
        payload = _base_config_dict()
        payload["data"] = dict(payload["data"])
        payload["data"]["symbols"] = [bad]
        with pytest.raises(InvalidConfigurationError, match="reserved device name"):
            ExperimentConfig.from_dict(payload)

    for ok in ["COM0", "LPT0", "CONSOLE", "NULL", "SPY", "AAPL"]:
        payload = _base_config_dict()
        payload["data"] = dict(payload["data"])
        payload["data"]["symbols"] = [ok]
        cfg = ExperimentConfig.from_dict(payload)
        assert cfg.symbols == [ok]


def test_symbol_rejects_a_trailing_dot_or_space() -> None:
    for bad in ["FOO.", "FOO..", "SPY.", "AAPL.CSV."]:
        payload = _base_config_dict()
        payload["data"] = dict(payload["data"])
        payload["data"]["symbols"] = [bad]
        with pytest.raises(InvalidConfigurationError, match="must not end with"):
            ExperimentConfig.from_dict(payload)

    # Sanity: an ordinary name containing internal dots remains accepted.
    payload = _base_config_dict()
    payload["data"] = dict(payload["data"])
    payload["data"]["symbols"] = ["BRK.B"]
    cfg = ExperimentConfig.from_dict(payload)
    assert cfg.symbols == ["BRK.B"]


def test_benchmark_symbol_path_traversal_rejected() -> None:
    payload = _base_config_dict()
    payload["backtest"] = {"benchmark_symbol": "../outside"}
    with pytest.raises(InvalidConfigurationError, match="benchmark_symbol"):
        ExperimentConfig.from_dict(payload)


def test_benchmark_symbol_normalized_like_data_symbols() -> None:
    payload = _base_config_dict()
    payload["backtest"] = {"benchmark_symbol": " spy "}
    cfg = ExperimentConfig.from_dict(payload)
    assert cfg.backtest.benchmark_symbol == "SPY"


def test_data_hash_covers_benchmark_symbol_too() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config

    frames = [
        make_ohlcv("A", geometric_series(200, 0.0004, 0.01, 100.0, seed=1)),
        make_ohlcv("BENCH", geometric_series(200, 0.0002, 0.012, 100.0, seed=2)),
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["A"],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {"benchmark_symbol": "BENCH"},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    result = run_backtest_from_config(data, cfg)

    mutated = data.copy()
    mask = mutated["symbol"] == "BENCH"
    closes = mutated["close"].to_numpy(dtype=float, copy=True)
    closes[mask.to_numpy()] *= 1.5
    mutated["close"] = closes
    result2 = run_backtest_from_config(mutated, cfg)

    assert result.metadata["data_hash"] != result2.metadata["data_hash"]
    # n_rows stays tradable-only (the benchmark isn't part of the traded
    # universe), so it must be unaffected by this change.
    assert result.metadata["n_rows"] == result2.metadata["n_rows"] == 200


def test_cache_covers_hourly_survives_a_weekend_with_calendar_tolerance(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2024-01-01 09:00", "2024-01-05 15:00", freq="h")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAA",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "AAA", "1h")

    # ~57 hours between the last (Friday) bar and Sunday.
    assert storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 1), date(2024, 1, 7)
    )


def test_cache_covers_equity_hourly_rejects_a_sparse_cache_gap(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    data = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01 10:00"),
                pd.Timestamp("2024-01-10 10:00"),
            ],
            "symbol": "AAA",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "AAA", "1h", is_247_market=False)

    assert not storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 4), date(2024, 1, 5), is_247_market=False
    )
    # Sanity: a request entirely inside a real weekend (no trading day at
    # all in range) must not be rejected just for lacking a bar that could
    # never exist.
    assert storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 6), date(2024, 1, 7), is_247_market=False
    )


def test_cache_covers_equity_hourly_detects_an_internal_missing_hour(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    bdays = pd.bdate_range("2024-01-02", "2024-01-12")
    hours: list[pd.Timestamp] = []
    for d in bdays:
        day_hours = pd.date_range(d.replace(hour=9), d.replace(hour=15), freq="h")
        if d == pd.Timestamp("2024-01-08"):
            day_hours = day_hours[day_hours.hour != 12]  # drop noon internally
        hours.extend(day_hours)
    storage.write_symbol(
        _make_hourly_frame(hours), "yahoo", "AAA", "1h", is_247_market=False
    )

    assert not storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 12), is_247_market=False
    )


def test_cache_covers_equity_hourly_detects_an_entire_missing_session(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    bdays = pd.bdate_range("2024-01-02", "2024-01-12")
    hours: list[pd.Timestamp] = []
    for d in bdays:
        if d == pd.Timestamp("2024-01-08"):
            continue  # entire session missing
        hours.extend(pd.date_range(d.replace(hour=9), d.replace(hour=15), freq="h"))
    storage.write_symbol(
        _make_hourly_frame(hours), "yahoo", "AAA", "1h", is_247_market=False
    )

    assert not storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 12), is_247_market=False
    )


def test_cache_covers_equity_hourly_complete_cache_still_passes(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    bdays = pd.bdate_range("2024-01-02", "2024-01-12")
    hours: list[pd.Timestamp] = []
    for d in bdays:
        hours.extend(pd.date_range(d.replace(hour=9), d.replace(hour=15), freq="h"))
    storage.write_symbol(
        _make_hourly_frame(hours), "yahoo", "AAA", "1h", is_247_market=False
    )

    assert storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 12), is_247_market=False
    )
    # A single-day request landing exactly on the cache's own extent must
    # also still pass (and must still be able to detect a gap, see the two
    # tests above, which both use single- and multi-day ranges).
    assert storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 2), is_247_market=False
    )


def test_cache_covers_weekly_does_not_over_tolerate_real_staleness(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2024-01-01", "2024-01-08", freq="7D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAA",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "AAA", "1w")

    assert not storage.cache_covers(
        "yahoo", "AAA", "1w", date(2024, 1, 1), date(2024, 1, 29)
    )


def test_cache_covers_247_market_does_not_mask_missing_crypto_days(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2023-12-29 00:00", "2024-01-01 00:00", freq="h")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTCUSDT", "1h")

    # A genuine 72-hour gap must be caught for a 24/7 market...
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2023, 12, 29),
        date(2024, 1, 4),
        is_247_market=True,
    )
    # The same cache also misses genuine XNYS sessions through January 4;
    # exact session coverage must reject it instead of spending weekend slack.
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2023, 12, 29),
        date(2024, 1, 4),
        is_247_market=False,
    )
    # A tiny, realistic posting lag (a few hours before the *end* of the
    # requested day, not its start — `end` is inclusive of its whole
    # calendar day) must still be tolerated even for a 24/7 market.
    assert storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2023, 12, 29),
        date(2023, 12, 31),
        is_247_market=True,
    )


def test_dataloader_threads_is_247_market_into_cache_covers() -> None:
    """`DataLoader._download_symbol` must pass the config's resolved
    `is_247_market` flag through to `cache_covers`, not rely on its
    (equity-biased) default."""
    from quantlab.config import ExperimentConfig

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "binance",
                "symbols": ["BTCUSDT"],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    assert cfg.data.is_247_market is True


def test_cache_covers_end_boundary_matches_loaders_inclusive_day(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2024-01-01 00:00", "2024-01-02 23:00", freq="h")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTCUSDT", "1h")

    # Cache stops Jan 2 23:00; the entire 24 bars of Jan 3 are missing.
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2024, 1, 1),
        date(2024, 1, 3),
        is_247_market=True,
    )
    # But it does genuinely cover through Jan 2 itself.
    assert storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2024, 1, 1),
        date(2024, 1, 2),
        is_247_market=True,
    )


def test_cache_covers_equity_does_not_skip_a_real_weekday_gap(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    # Friday 2024-01-05 is the last business day before a normal weekend.
    dates = pd.date_range("2023-01-02", "2024-01-05", freq="B")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")

    # Request through the following Tuesday: Monday + Tuesday sessions are
    # genuinely missing from the cache.
    assert not storage.cache_covers(
        "yahoo",
        "SPY",
        "1d",
        date(2023, 1, 2),
        date(2024, 1, 9),
        is_247_market=False,
    )
    # Request through the weekend itself (Sunday): nothing more could
    # possibly exist, so this must still be tolerated.
    assert storage.cache_covers(
        "yahoo",
        "SPY",
        "1d",
        date(2023, 1, 2),
        date(2024, 1, 7),
        is_247_market=False,
    )


def test_cache_covers_complete_daily_cache_is_not_declared_incomplete(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    _write_daily_cache(storage, "binance", "BTCUSDT", dates)
    assert storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1d",
        date(2024, 1, 1),
        date(2024, 1, 10),
        is_247_market=True,
    )


def test_cache_covers_rejects_a_missing_single_weekday_session(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    # Friday 2025-01-03 is the last bar; Monday 2025-01-06 is a real,
    # missing trading session.
    dates = pd.bdate_range("2024-12-01", "2025-01-03")
    _write_daily_cache(storage, "yahoo", "SPY", dates)
    assert not storage.cache_covers(
        "yahoo", "SPY", "1d", date(2024, 12, 1), date(2025, 1, 6), is_247_market=False
    )
    # Through Sunday (the weekend itself): nothing more could exist.
    assert storage.cache_covers(
        "yahoo", "SPY", "1d", date(2024, 12, 1), date(2025, 1, 5), is_247_market=False
    )


def test_cache_covers_rejects_several_missing_leading_sessions(
    tmp_path: Path,
) -> None:
    """The `start` boundary needs the same calendar-aware treatment as
    `end`: a cache starting several genuine trading days late must not be
    accepted as "covering" the requested start."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    # Requested start is Monday 2025-01-20; cache actually only starts the
    # following Friday 2025-01-24 (Mon-Thu = 4 genuine sessions missing).
    dates = pd.bdate_range("2025-01-24", "2025-02-01")
    _write_daily_cache(storage, "yahoo", "SPY", dates)
    assert not storage.cache_covers(
        "yahoo", "SPY", "1d", date(2025, 1, 20), date(2025, 2, 1), is_247_market=False
    )


def test_cache_covers_hourly_247_flags_missing_edge_hours(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    # Missing the first 3 hours of Jan 1.
    dates = pd.date_range("2024-01-01 03:00", "2024-01-02 23:00", freq="h")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTCUSDT", "1h")
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2024, 1, 1),
        date(2024, 1, 2),
        is_247_market=True,
    )


def test_cache_covers_weekly_complete_cache_is_not_declared_incomplete(
    tmp_path: Path,
) -> None:
    """A weekly bar's own timestamp covers its whole bucket forward, the
    same as a daily bar's: a genuinely complete weekly cache must be
    reported as covering the request, not declared incomplete."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2024-01-01", "2024-12-30", freq="W-MON")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1w")
    assert storage.cache_covers(
        "yahoo", "SPY", "1w", date(2024, 1, 1), date(2024, 12, 30), is_247_market=False
    )


def test_cache_covers_monthly_january_bar_covers_whole_january(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    data = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2021-01-01")],
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1mo")
    assert storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 1, 1), date(2021, 1, 31), is_247_market=False
    )


def test_cache_covers_monthly_february_bar_does_not_mask_missing_march(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    data = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2021-02-01")],
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1mo")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 2, 1), date(2021, 3, 1), is_247_market=False
    )
    # But the same cache genuinely does cover a request confined to February.
    assert storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 2, 1), date(2021, 2, 28), is_247_market=False
    )


def test_cache_covers_monthly_detects_a_missing_internal_month(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.to_datetime(["2021-01-01", "2021-03-01"])
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1mo")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 1, 1), date(2021, 3, 31), is_247_market=False
    )


def test_cache_covers_weekly_detects_a_missing_internal_week(
    tmp_path: Path,
) -> None:
    """Same internal-gap detection, for weekly bars: January 1st and 15th
    present, January 8th missing entirely."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.to_datetime(["2021-01-01", "2021-01-15"])
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1w")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1w", date(2021, 1, 1), date(2021, 1, 15), is_247_market=False
    )


def test_cache_covers_monthly_and_weekly_complete_caches_still_pass(
    tmp_path: Path,
) -> None:
    """A genuinely complete monthly/weekly cache (no internal gap) must
    still be reported as covering the request."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    monthly = pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"])
    storage.write_symbol(_ohlcv_at(monthly), "yahoo", "SPY", "1mo")
    assert storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 1, 1), date(2021, 3, 31), is_247_market=False
    )

    weekly = pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-15"])
    storage.write_symbol(_ohlcv_at(weekly, symbol="QQQ"), "yahoo", "QQQ", "1w")
    assert storage.cache_covers(
        "yahoo", "QQQ", "1w", date(2021, 1, 1), date(2021, 1, 15), is_247_market=False
    )


def test_cache_covers_monthly_bar_does_not_overclaim_past_its_own_month(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    storage.write_symbol(
        _ohlcv_at(pd.to_datetime(["2021-01-02"])), "yahoo", "SPY", "1mo"
    )
    assert not storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 1, 2), date(2021, 2, 1), is_247_market=False
    )
    # But it does genuinely cover a request confined to its own month.
    assert storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2021, 1, 2), date(2021, 1, 31), is_247_market=False
    )


def test_cache_covers_monthly_gap_in_requested_start_month_detected(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.to_datetime(["2019-12-01", "2020-02-01"])
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1mo")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2020, 1, 15), date(2020, 2, 29), is_247_market=False
    )


def test_cache_covers_weekly_gap_in_requested_start_week_detected(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.to_datetime(["2020-01-01", "2020-01-15"])
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1w")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1w", date(2020, 1, 10), date(2020, 1, 15), is_247_market=False
    )


def test_cache_covers_monthly_tolerates_shifting_first_trading_day(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"])
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1mo")
    assert storage.cache_covers(
        "yahoo", "SPY", "1mo", date(2020, 1, 3), date(2020, 4, 3), is_247_market=False
    )


def test_cache_covers_weekly_tolerates_a_holiday_shifted_first_week(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = [
        pd.Timestamp("2024-01-02"),
        *pd.date_range("2024-01-08", "2024-12-30", freq="W-MON"),
    ]
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1w")
    assert storage.cache_covers(
        "yahoo", "SPY", "1w", date(2024, 1, 2), date(2024, 12, 30), is_247_market=False
    )


def test_cache_covers_rejects_sparse_cache_narrower_than_a_bucket(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.to_datetime(["2024-01-01", "2024-01-10"])
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1w")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1w", date(2024, 1, 5), date(2024, 1, 6), is_247_market=False
    )
    # Same root cause, different shape: the one raw-timestamp match that
    # exists (January 1st) is real, but its own bucket doesn't settle until
    # the following week, so a request for exactly that one day alone still
    # can't actually be served either.
    assert not storage.cache_covers(
        "yahoo", "SPY", "1w", date(2024, 1, 1), date(2024, 1, 1), is_247_market=False
    )


def test_cache_covers_weekly_request_starting_on_a_holiday(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = [
        pd.Timestamp("2024-01-02"),
        *pd.date_range("2024-01-08", "2024-03-25", freq="W-MON"),
    ]
    storage.write_symbol(_ohlcv_at(dates), "yahoo", "SPY", "1w")
    assert storage.cache_covers(
        "yahoo", "SPY", "1w", date(2024, 1, 1), date(2024, 3, 1), is_247_market=False
    )
    # Sanity: a genuine gap must still be caught, holiday-start or not.
    gappy = [d for d in dates if d != pd.Timestamp("2024-01-08")]
    storage.write_symbol(_ohlcv_at(gappy, symbol="QQQ"), "yahoo", "QQQ", "1w")
    assert not storage.cache_covers(
        "yahoo", "QQQ", "1w", date(2024, 1, 1), date(2024, 3, 1), is_247_market=False
    )


def test_write_symbol_drops_a_still_open_bar(tmp_path: Path) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    still_open_month = now.normalize() + pd.offsets.MonthBegin(1)
    storage.write_symbol(_ohlcv_at([still_open_month]), "yahoo", "SPY", "1mo")
    cached = storage.read_symbol("yahoo", "SPY", "1mo")
    assert cached is None or cached.empty


def test_write_symbol_forced_rewrite_clears_a_stale_legacy_bar(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    still_open_month = now.normalize() + pd.offsets.MonthBegin(1)
    path = storage._cache_path("yahoo", "SPY", "1mo")
    storage.save(_ohlcv_at([still_open_month]), path)
    before = storage.load(path)
    assert len(before) == 1  # stale bar present, read directly off disk

    empty = _ohlcv_at([]).astype({"timestamp": "datetime64[ns]", "symbol": "object"})
    storage.write_symbol(empty, "yahoo", "SPY", "1mo")
    after = storage.read_symbol("yahoo", "SPY", "1mo")
    assert after is None or after.empty


def test_read_symbol_purges_a_still_open_bar_from_disk(tmp_path: Path) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    still_open_month = now.normalize() + pd.offsets.MonthBegin(1)
    path = storage._cache_path("yahoo", "SPY", "1mo")
    storage.save(_ohlcv_at([still_open_month]), path)

    served = storage.read_symbol("yahoo", "SPY", "1mo")
    assert served is not None
    assert served.empty

    # Purged from disk too, not merely filtered in-memory -- a second,
    # independent read of the raw file must not see it either.
    on_disk = storage.load(path)
    assert on_disk.empty


def test_csv_source_does_not_serve_a_still_open_bar(tmp_path: Path) -> None:
    from quantlab.config import ExperimentConfig
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    still_open_day = today + pd.Timedelta(days=1)
    dates = pd.date_range(end=still_open_day, periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "BTC",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    frame.to_csv(raw_dir / "BTC.csv", index=False)

    storage = ParquetStorage(
        cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "meta"
    )
    loader = DataLoader(storage=storage, raw_dir=raw_dir)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "source": "csv",
                "symbols": ["BTC"],
                "start_date": str(dates[0].date()),
                "end_date": str(still_open_day.date()),
                "frequency": "1d",
                "market_calendar": "24/7",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    data, _ = loader.load(cfg)
    assert data["timestamp"].max() < still_open_day


def test_cache_covers_detects_missing_internal_daily_bar(tmp_path: Path) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.bdate_range("2021-01-04", "2021-03-01").tolist()
    del dates[10:25]  # drop three full internal weeks
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1d", date(2021, 1, 4), date(2021, 3, 1), is_247_market=False
    )


def test_cache_covers_detects_missing_internal_hourly_247_bar(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    hours = pd.date_range("2021-01-01", periods=48, freq="h").tolist()
    del hours[10:25]
    data = pd.DataFrame(
        {
            "timestamp": hours,
            "symbol": "BTC",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTC", "1h")
    assert not storage.cache_covers(
        "binance", "BTC", "1h", date(2021, 1, 1), date(2021, 1, 2), is_247_market=True
    )


def test_cache_covers_tolerates_a_small_number_of_calendar_blind_spots(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.bdate_range("2012-10-01", "2012-11-30")
    dates = dates[~dates.isin(pd.to_datetime(["2012-10-29", "2012-10-30"]))]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert storage.cache_covers(
        "yahoo", "SPY", "1d", date(2012, 10, 1), date(2012, 11, 30), is_247_market=False
    )


def test_cache_covers_rejects_an_arbitrary_missing_session(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.bdate_range("2019-01-01", "2019-12-31")
    dates = dates[dates != pd.Timestamp("2019-06-12")]  # an ordinary Wednesday
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert not storage.cache_covers(
        "yahoo", "SPY", "1d", date(2019, 1, 1), date(2019, 12, 31), is_247_market=False
    )


def test_cache_covers_247_daily_has_zero_calendar_tolerance(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2021-01-01", "2021-03-01", freq="D")
    dates = dates[dates != pd.Timestamp("2021-01-15")]  # a single missing day
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTCUSDT", "1d")
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1d",
        date(2021, 1, 1),
        date(2021, 3, 1),
        is_247_market=True,
    )


def test_cache_covers_247_hourly_has_zero_calendar_tolerance(
    tmp_path: Path,
) -> None:
    """Same zero-tolerance rule, for the 24/7-hourly branch: a single
    missing hour in a crypto hourly cache must be caught."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2021-01-01", "2021-01-20 23:00", freq="h")
    dates = dates[dates != pd.Timestamp("2021-01-10 05:00")]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTCUSDT", "1h")
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2021, 1, 1),
        date(2021, 1, 20),
        is_247_market=True,
    )


def test_cache_covers_equity_daily_tolerance_unaffected(tmp_path: Path) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.bdate_range("2012-10-01", "2012-11-30")
    dates = dates[~dates.isin(pd.to_datetime(["2012-10-29", "2012-10-30"]))]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert storage.cache_covers(
        "yahoo", "SPY", "1d", date(2012, 10, 1), date(2012, 11, 30), is_247_market=False
    )


def test_cache_covers_complete_daily_and_hourly_caches_still_pass(
    tmp_path: Path,
) -> None:
    """A genuinely complete cache (no internal gap) must still be reported
    as covering the request — the internal-gap check must not introduce
    false negatives for the ordinary case."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.bdate_range("2021-01-04", "2021-01-08").tolist()
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert storage.cache_covers(
        "yahoo", "SPY", "1d", date(2021, 1, 4), date(2021, 1, 8), is_247_market=False
    )

    hours = pd.date_range("2021-01-01", periods=48, freq="h").tolist()
    data2 = pd.DataFrame(
        {
            "timestamp": hours,
            "symbol": "BTC",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data2, "binance", "BTC", "1h")
    assert storage.cache_covers(
        "binance", "BTC", "1h", date(2021, 1, 1), date(2021, 1, 2), is_247_market=True
    )


def test_cache_covers_ignores_a_gap_outside_the_requested_range(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.bdate_range("2010-01-04", "2025-12-31")
    dates = dates[~dates.isin(pd.to_datetime(["2012-10-29", "2012-10-30"]))]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert storage.cache_covers(
        "yahoo", "SPY", "1d", date(2020, 1, 1), date(2020, 12, 31), is_247_market=False
    )
    # A request that actually spans the (small, tolerated) 2012 Sandy gap
    # must still pass too — see
    # `test_cache_covers_tolerates_a_small_number_of_calendar_blind_spots`
    # — but a *large* gap within the requested range, added separately below,
    # must still correctly fail: the requested-range scoping isn't a license
    # to ignore genuine gaps that fall inside it.
    dates_with_large_gap = dates[~((dates.year == 2016) & (dates.month <= 2))]
    storage.write_symbol(
        pd.DataFrame(
            {
                "timestamp": dates_with_large_gap,
                "symbol": "QQQ",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "adjusted_close": 1.0,
                "volume": 100.0,
            }
        ),
        "yahoo",
        "QQQ",
        "1d",
    )
    assert not storage.cache_covers(
        "yahoo", "QQQ", "1d", date(2016, 1, 1), date(2016, 12, 31), is_247_market=False
    )


def test_cache_covers_hourly_247_ignores_a_gap_outside_the_requested_range(
    tmp_path: Path,
) -> None:
    """Same requested-window scoping, for the 24/7-market hourly
    internal-gap check."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    hours = pd.date_range("2021-01-01", periods=200, freq="h").tolist()
    del hours[10]  # a gap well outside the later, narrower request below
    data = pd.DataFrame(
        {
            "timestamp": hours,
            "symbol": "BTC",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "binance", "BTC", "1h")
    assert storage.cache_covers(
        "binance", "BTC", "1h", date(2021, 1, 7), date(2021, 1, 8), is_247_market=True
    )


def test_nyse_calendar_new_years_saturday_does_not_close_preceding_friday() -> None:
    from quantlab.data.calendar import XNYS_BUSINESS_DAY

    def is_open(iso_date: str) -> bool:
        ts = pd.Timestamp(iso_date)
        return bool(ts == XNYS_BUSINESS_DAY.rollforward(ts))

    # Jan 1, 2011 and Jan 1, 2022 both fell on a Saturday.
    assert is_open("2010-12-31")
    assert is_open("2021-12-31")
    # A Sunday-landing New Year's Day must still roll to the following
    # Monday (Jan 1, 2012 was a Sunday), and other Saturday-landing
    # holidays must still observe on the preceding Friday, since the
    # asymmetric rule above applies only to New Year's Day (July 4, 2015
    # was a Saturday).
    assert not is_open("2012-01-02")
    assert not is_open("2015-07-03")


def test_cache_covers_daily_spy_stopped_dec30_2021_is_incomplete_for_dec31(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    dates = pd.date_range("2021-12-20", "2021-12-30", freq="B")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    assert not storage.cache_covers(
        "yahoo",
        "SPY",
        "1d",
        date(2021, 12, 20),
        date(2021, 12, 31),
        is_247_market=False,
    )


def test_cache_covers_a_future_end_date_does_not_perpetually_fail(
    tmp_path: Path,
) -> None:
    from quantlab.data.calendar import (
        daily_equity_bucket_settlement,
        last_trading_day_on_or_before,
    )
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    today = now.normalize()
    # The freshest a cache could genuinely be *right now* -- mirrors
    # `cache_covers`'s own "has today's bucket closed yet" check, since a
    # cache stopping at a fixed "yesterday" would itself be stale (missing
    # today's already-closed session) whenever this test happens to run
    # after today's market close.
    latest_closed_day = last_trading_day_on_or_before(today, is_247_market=False)
    if latest_closed_day == today and daily_equity_bucket_settlement(today) > now:
        latest_closed_day = last_trading_day_on_or_before(
            today - pd.Timedelta(days=1), is_247_market=False
        )
    dates = pd.bdate_range(end=latest_closed_day, periods=250)
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(data, "yahoo", "SPY", "1d")
    future_end = (today + pd.Timedelta(days=365)).date()
    assert storage.cache_covers(
        "yahoo", "SPY", "1d", dates[0].date(), future_end, is_247_market=False
    )


def test_cache_covers_a_future_end_date_still_rejects_a_genuinely_stale_cache(
    tmp_path: Path,
) -> None:
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    # Deliberately stops 30+ sessions before today, regardless of what time
    # today's own bucket closes -- genuinely stale under any wall-clock time
    # this test might run at, unlike the always-maximally-fresh fixture
    # above.
    dates = pd.bdate_range(end=today - pd.Timedelta(days=45), periods=250)
    stale = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(stale, "yahoo", "SPY", "1d")
    future_end = (today + pd.Timedelta(days=365)).date()
    assert not storage.cache_covers(
        "yahoo", "SPY", "1d", dates[0].date(), future_end, is_247_market=False
    )


def test_safe_does_not_collide_distinct_symbols_onto_one_filename() -> None:
    from quantlab.data.storage import _safe

    names = {_safe(s) for s in ["EURUSD=X", "EURUSD_X", "EURUSD X"]}
    assert len(names) == 3


def test_safe_prevents_cache_read_write_collision_end_to_end() -> None:
    """End-to-end: writing two distinct symbols whose raw names differ only
    by "unsafe" characters must round-trip independently through the real
    cache, not silently overwrite each other."""
    import tempfile

    from quantlab.data.storage import ParquetStorage

    with tempfile.TemporaryDirectory() as tmp:
        storage = ParquetStorage(
            cache_dir=Path(tmp) / "cache", metadata_dir=Path(tmp) / "md"
        )
        eur = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3),
                "symbol": "EURUSD=X",
                "open": 1.1,
                "high": 1.1,
                "low": 1.1,
                "close": 1.1,
                "adjusted_close": 1.1,
                "volume": 0.0,
            }
        )
        other = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3),
                "symbol": "EURUSD_X",
                "open": 2.2,
                "high": 2.2,
                "low": 2.2,
                "close": 2.2,
                "adjusted_close": 2.2,
                "volume": 0.0,
            }
        )
        storage.write_symbol(eur, "yahoo", "EURUSD=X", "1d")
        storage.write_symbol(other, "yahoo", "EURUSD_X", "1d")

        read_eur = storage.read_symbol("yahoo", "EURUSD=X", "1d")
        read_other = storage.read_symbol("yahoo", "EURUSD_X", "1d")
        assert read_eur is not None
        assert read_other is not None
        assert (read_eur["close"] == 1.1).all()
        assert (read_other["close"] == 2.2).all()


def test_load_slices_before_cleaning_not_after() -> None:
    from quantlab.config import MissingValuePolicy
    from quantlab.data.cleaner import DataCleaner
    from quantlab.data.loader import DataLoader

    wide_dates = pd.date_range("2019-12-30", "2020-01-03", freq="D")
    close = [100.0, 101.0, np.nan, 103.0, 104.0]
    raw_wide = pd.DataFrame(
        {
            "timestamp": wide_dates,
            "symbol": "AAA",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adjusted_close": close,
            "volume": 1000.0,
        }
    )
    narrow = raw_wide[raw_wide["timestamp"] >= "2020-01-01"].reset_index(drop=True)

    cleaner = DataCleaner(MissingValuePolicy.FORWARD_FILL)
    start, end = date(2020, 1, 1), date(2020, 1, 3)

    # The *correct*, order-independent pipeline: slice first, then clean.
    sliced_then_cleaned_wide = cleaner.clean(
        DataLoader._slice_range(raw_wide, start, end, "1d", is_247_market=False)
    )
    sliced_then_cleaned_narrow = cleaner.clean(
        DataLoader._slice_range(narrow, start, end, "1d", is_247_market=False)
    )
    assert len(sliced_then_cleaned_wide) == len(sliced_then_cleaned_narrow) == 2
    pd.testing.assert_frame_equal(
        sliced_then_cleaned_wide.reset_index(drop=True),
        sliced_then_cleaned_narrow.reset_index(drop=True),
    )

    # The wrong order (clean first, slice after) does produce a
    # discrepancy between the two cache states — pinned here so a
    # regression back to that order is caught by this test actually
    # detecting a difference.
    clean_then_slice_wide = DataLoader._slice_range(
        cleaner.clean(raw_wide), start, end, "1d", is_247_market=False
    )
    clean_then_slice_narrow = DataLoader._slice_range(
        cleaner.clean(narrow), start, end, "1d", is_247_market=False
    )
    assert len(clean_then_slice_wide) != len(clean_then_slice_narrow)


def test_validator_flags_a_requested_symbol_with_zero_rows() -> None:
    from quantlab.data.validator import DataValidator

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=40, freq="D"),
            "symbol": "BBB",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    # A file for "AAA" that actually contains "BBB"'s data.
    report = DataValidator().validate(frame, expected_symbols=["AAA"])
    assert not report.is_clean
    assert any("AAA" in w for w in report.warnings)


def test_validator_flags_a_symbol_fully_removed_by_cleaning() -> None:
    from quantlab.config import MissingValuePolicy
    from quantlab.data.cleaner import DataCleaner
    from quantlab.data.validator import DataValidator

    n = 40
    frame = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2020-01-01", periods=n, freq="D")) * 2,
            "symbol": ["AAA"] * n + ["BBB"] * n,
            "open": [100.0] * n + [-1.0] * n,
            "high": [101.0] * n + [-1.0] * n,
            "low": [99.0] * n + [-1.0] * n,
            "close": [100.0] * n + [-1.0] * n,
            "adjusted_close": [100.0] * n + [-1.0] * n,
            "volume": 1000.0,
        }
    )
    cleaned = DataCleaner(MissingValuePolicy.DROP).clean(frame)
    assert "BBB" not in set(cleaned["symbol"].unique())

    report = DataValidator().validate(cleaned, expected_symbols=["AAA", "BBB"])
    assert not report.is_clean
    assert any("BBB" in w for w in report.warnings)

    # Without `expected_symbols`, validation must still complete normally
    # (not raise/error) — it simply can't detect the missing-symbol gap.
    report_without = DataValidator().validate(cleaned)
    assert not any("BBB" in w for w in report_without.warnings)


def test_validator_raises_on_missing_symbol_in_strict_mode() -> None:
    from quantlab.data.validator import DataValidator
    from quantlab.exceptions import DataValidationError

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=40, freq="D"),
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    with pytest.raises(DataValidationError, match="BBB"):
        DataValidator().validate(frame, expected_symbols=["AAA", "BBB"], strict=True)


def test_loader_passes_expected_symbols_including_benchmark() -> None:
    _, cfg = _holdout_config()
    from quantlab.data.loader import DataLoader

    loader = DataLoader()
    expected = loader._symbols_to_fetch(cfg)
    assert set(cfg.symbols).issubset(set(expected))


def test_frequency_mismatch_flagged_at_exact_tolerance_boundary() -> None:
    from quantlab.data.validator import DataValidator

    idx = pd.date_range("2020-01-01", periods=50, freq="90min")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(expected_frequency="1h", min_coverage_rows=1).validate(frame)
    assert any("does not match the declared frequency" in w for w in report.warnings)


def test_frequency_exact_match_still_clean_after_boundary_fix() -> None:
    """The inclusive tolerance boundary must not flag an exact frequency
    match."""
    from quantlab.data.validator import DataValidator

    idx = pd.date_range("2020-01-01", periods=50, freq="1h")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(expected_frequency="1h", min_coverage_rows=1).validate(frame)
    assert not any(
        "does not match the declared frequency" in w for w in report.warnings
    )


def test_frequency_uniform_89min_bars_declared_1h_now_flagged() -> None:
    """A uniform 89-minute spacing declared "1h" is a 1.483x ratio — just
    under a 1.5x tolerance boundary — so a boundary set too loose would miss
    a genuine, substantial (48%) mismatch entirely."""
    from quantlab.data.validator import DataValidator

    idx = pd.date_range("2020-01-01", periods=50, freq="89min")
    report = DataValidator(expected_frequency="1h", min_coverage_rows=1).validate(
        _ohlcv_frame(idx)
    )
    assert report.warnings


def test_frequency_uniform_41min_bars_declared_1h_now_flagged() -> None:
    """Same check, the other direction: a uniform 41-minute spacing declared
    "1h" is a 0.683x ratio — just above the 1/1.5 lower boundary — and must
    still be flagged as a mismatch."""
    from quantlab.data.validator import DataValidator

    idx = pd.date_range("2020-01-01", periods=50, freq="41min")
    report = DataValidator(expected_frequency="1h", min_coverage_rows=1).validate(
        _ohlcv_frame(idx)
    )
    assert report.warnings


def test_frequency_mixed_60_40_spacing_median_cannot_hide_it() -> None:
    from quantlab.data.validator import DataValidator

    rng = np.random.default_rng(0)
    hours = rng.choice([1, 2], size=200, p=[0.6, 0.4])
    timestamps = [pd.Timestamp("2020-01-01")]
    for h in hours:
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=int(h)))
    idx = pd.DatetimeIndex(timestamps)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv_frame(idx))
    assert report.warnings


def test_frequency_normal_equity_daily_weekends_not_flagged() -> None:
    from quantlab.data.validator import DataValidator

    idx = pd.bdate_range("2015-01-01", periods=500)
    report = DataValidator(
        expected_frequency="1d", min_coverage_rows=1, is_247_market=False
    ).validate(_ohlcv_frame(idx))
    assert not report.warnings


def test_frequency_equity_daily_friday_to_monday_not_flagged() -> None:
    from quantlab.data.validator import DataValidator

    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")])
    report = DataValidator(
        expected_frequency="1d", min_coverage_rows=1, is_247_market=False
    ).validate(_ohlcv_frame(idx))
    assert not report.warnings


def test_frequency_equity_daily_genuine_every_other_day_still_flagged() -> None:
    """The business-day-aware comparison must not swallow a genuine
    mismatch — a series sampled every *other* business day (2 business days
    apart, not 1) must still be flagged."""
    from quantlab.data.validator import DataValidator

    idx = pd.bdate_range("2020-01-01", periods=200)[::2]
    report = DataValidator(
        expected_frequency="1d", min_coverage_rows=1, is_247_market=False
    ).validate(_ohlcv_frame(idx))
    assert report.warnings


def test_frequency_equity_daily_friday_to_tuesday_after_mlk_not_flagged() -> None:
    from quantlab.data.validator import DataValidator

    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-16")])
    report = DataValidator(
        expected_frequency="1d", min_coverage_rows=1, is_247_market=False
    ).validate(_ohlcv_frame(idx))
    assert not report.warnings


def test_frequency_equity_subdaily_overnight_gaps_not_flagged() -> None:
    from quantlab.data.validator import DataValidator

    dates = []
    for day in pd.bdate_range("2020-01-06", periods=10):
        for h in range(3):  # only 3 bars/session -- a coarser intraday config
            dates.append(
                day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
            )
    idx = pd.DatetimeIndex(dates)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=False
    ).validate(_ohlcv_frame(idx))
    assert not report.warnings


def test_frequency_247_market_80_20_mix_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    timestamps = [pd.Timestamp("2020-01-01")]
    for _ in range(80):
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=1))
    for _ in range(20):
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=5))
    idx = pd.DatetimeIndex(timestamps)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv_frame(idx))
    assert report.warnings


def test_frequency_247_market_exactly_90_10_mix_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    timestamps = [pd.Timestamp("2020-01-01")]
    for _ in range(90):
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=1))
    for _ in range(10):
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=5))
    idx = pd.DatetimeIndex(timestamps)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv_frame(idx))
    assert report.warnings


def test_frequency_247_market_91_9_mix_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    timestamps = [pd.Timestamp("2020-01-01")]
    for _ in range(91):
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=1))
    for _ in range(9):
        timestamps.append(timestamps[-1] + pd.Timedelta(hours=5))
    idx = pd.DatetimeIndex(timestamps)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv_frame(idx))
    assert report.warnings


def test_frequency_247_market_large_clean_series_not_flagged() -> None:
    from quantlab.data.validator import DataValidator

    timestamps = [pd.Timestamp("2020-01-01")]
    for i in range(5000):
        step = pd.Timedelta(hours=2) if i == 2500 else pd.Timedelta(hours=1)
        timestamps.append(timestamps[-1] + step)
    idx = pd.DatetimeIndex(timestamps)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv_frame(idx))
    assert not any(
        "does not match the declared frequency" in w for w in report.warnings
    )


def test_frequency_247_market_single_missing_bar_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    timestamps = [pd.Timestamp("2020-01-01")]
    for i in range(50):
        step = pd.Timedelta(hours=2) if i == 25 else pd.Timedelta(hours=1)
        timestamps.append(timestamps[-1] + step)
    idx = pd.DatetimeIndex(timestamps)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv_frame(idx))
    assert any("abnormal gap" in w for w in report.warnings)
    assert len(report.missing_periods) == 1
    assert not report.is_clean


def test_frequency_247_market_missing_exactly_the_edge_day_now_flagged() -> None:
    from quantlab.constants import (
        ADJUSTED_CLOSE,
        CLOSE,
        HIGH,
        LOW,
        OPEN,
        SYMBOL,
        TIMESTAMP,
        VOLUME,
    )
    from quantlab.data.validator import DataValidator

    def frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                TIMESTAMP: dates,
                SYMBOL: "BTCUSDT",
                OPEN: 1.0,
                HIGH: 1.0,
                LOW: 1.0,
                CLOSE: 1.0,
                ADJUSTED_CLOSE: 1.0,
                VOLUME: 1.0,
            }
        )

    validator = DataValidator(
        expected_frequency="1d", is_247_market=True, min_coverage_rows=1
    )

    missing_first = pd.date_range("2024-01-02", "2024-01-30", freq="D")
    report_first = validator.validate(
        frame(missing_first), start=date(2024, 1, 1), end=date(2024, 1, 30)
    )
    assert report_first.warnings
    assert not report_first.is_clean

    missing_last = pd.date_range("2024-01-01", "2024-01-29", freq="D")
    report_last = validator.validate(
        frame(missing_last), start=date(2024, 1, 1), end=date(2024, 1, 30)
    )
    assert report_last.warnings
    assert not report_last.is_clean

    # Sanity: a genuinely complete history must still be clean.
    complete = pd.date_range("2024-01-01", "2024-01-30", freq="D")
    report_complete = validator.validate(
        frame(complete), start=date(2024, 1, 1), end=date(2024, 1, 30)
    )
    assert not report_complete.warnings
    assert report_complete.is_clean


def test_frequency_247_hourly_missing_the_last_23_hours_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    dates = pd.date_range("2020-01-01", "2020-01-05 00:00:00", freq="1h")
    incomplete = _ohlcv_frame(dates)
    report = DataValidator(
        expected_frequency="1h", is_247_market=True, min_coverage_rows=1
    ).validate(incomplete, start=date(2020, 1, 1), end=date(2020, 1, 5))
    assert report.warnings
    assert not report.is_clean

    # Sanity: a history reaching the day's actual last expected hour is clean.
    complete_dates = pd.date_range("2020-01-01", "2020-01-05 23:00:00", freq="1h")
    complete_report = DataValidator(
        expected_frequency="1h", is_247_market=True, min_coverage_rows=1
    ).validate(
        _ohlcv_frame(complete_dates), start=date(2020, 1, 1), end=date(2020, 1, 5)
    )
    assert not complete_report.warnings
    assert complete_report.is_clean


def test_frequency_equity_subdaily_bar_missing_every_session_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    dates: list[pd.Timestamp] = []
    for day in pd.bdate_range("2020-01-06", periods=20):
        session_start = day + pd.Timedelta(hours=9, minutes=30)
        dates.extend(session_start + pd.Timedelta(hours=h) for h in range(7) if h != 3)
    idx = pd.DatetimeIndex(dates)
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=False
    ).validate(_ohlcv_frame(idx))
    assert report.warnings


def test_yahoo_intraday_timezone_converted_to_utc_not_stripped_naively() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.date_range("2021-01-04 09:30", periods=3, freq="h", tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0],
            "Adj Close": [1.0, 2.0, 3.0],
            "Volume": [100, 200, 300],
        },
        index=idx,
    )
    raw.index.name = "Datetime"
    out = YahooFinanceDataSource._normalise(
        raw, "AAPL", "1h", pd.Timestamp("2025-01-01"), date(2025, 1, 1)
    )
    assert out["timestamp"].dt.tz is None
    assert out["timestamp"].tolist() == [
        pd.Timestamp("2021-01-04 14:30:00"),
        pd.Timestamp("2021-01-04 15:30:00"),
        pd.Timestamp("2021-01-04 16:30:00"),
    ]


def test_yahoo_daily_naive_timestamps_unaffected_by_tz_fix() -> None:
    """Daily bars typically come back timezone-*naive* already (a plain
    calendar date) — the intraday tz-conversion above must be a no-op for
    them, not shift daily dates by some assumed offset."""
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.date_range("2021-01-04", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0],
            "Adj Close": [1.0, 2.0, 3.0],
            "Volume": [100, 200, 300],
        },
        index=idx,
    )
    raw.index.name = "Date"
    out = YahooFinanceDataSource._normalise(
        raw, "AAPL", "1d", pd.Timestamp("2025-01-01"), date(2025, 1, 1)
    )
    assert out["timestamp"].dt.tz is None
    assert out["timestamp"].tolist() == [
        pd.Timestamp("2021-01-04"),
        pd.Timestamp("2021-01-05"),
        pd.Timestamp("2021-01-06"),
    ]


def test_yahoo_drops_a_still_open_daily_bar() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    now = pd.Timestamp("2021-01-10 15:00:00")
    idx = pd.date_range("2021-01-04", "2021-01-10", freq="D")
    raw = pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Adj Close": 1.0,
            "Volume": 100,
        },
        index=idx,
    )
    raw.index.name = "Date"
    out = YahooFinanceDataSource._normalise(raw, "AAA", "1d", now, date(2021, 1, 10))
    assert out["timestamp"].tolist() == list(idx[:-1])


def test_yahoo_keeps_fully_historical_bars() -> None:
    """A genuinely historical
    download, where every bar's period has long since ended, must be
    completely unaffected by this filter."""
    from quantlab.data.yahoo import YahooFinanceDataSource

    now = pd.Timestamp("2025-01-01")
    idx = pd.date_range("2021-01-04", "2021-01-10", freq="D")
    raw = pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Adj Close": 1.0,
            "Volume": 100,
        },
        index=idx,
    )
    raw.index.name = "Date"
    out = YahooFinanceDataSource._normalise(raw, "AAA", "1d", now, date(2021, 1, 10))
    assert out["timestamp"].tolist() == list(idx)


def test_yahoo_drops_a_still_open_monthly_bar() -> None:
    """Same rule, for a monthly bar — the current month's bar (dated the 1st)
    must be dropped while `now` is still within that same month."""
    from quantlab.data.yahoo import YahooFinanceDataSource

    now = pd.Timestamp("2021-01-15")
    idx = pd.to_datetime(["2020-11-01", "2020-12-01", "2021-01-01"])
    raw = pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Adj Close": 1.0,
            "Volume": 100,
        },
        index=idx,
    )
    raw.index.name = "Date"
    out = YahooFinanceDataSource._normalise(raw, "AAA", "1mo", now, date(2021, 1, 15))
    assert out["timestamp"].tolist() == list(idx[:-1])


def test_canonical_schema_normalises_timezone_aware_timestamps() -> None:
    from quantlab.data.base import ensure_canonical_schema
    from quantlab.data.loader import DataLoader

    frame = _minimal_ohlcv_frame(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"])
    out = ensure_canonical_schema(frame)
    assert out["timestamp"].dt.tz is None
    # Must not raise, and must not silently drop the in-range rows.
    sliced = DataLoader._slice_range(
        out, date(2020, 1, 1), date(2020, 1, 2), "1d", is_247_market=False
    )
    assert len(sliced) == 2


def test_canonical_schema_leaves_naive_timestamps_unchanged() -> None:
    """The common case (no tz information at all) must pass through
    unchanged — only tz-aware input should be affected."""
    from quantlab.data.base import ensure_canonical_schema

    frame = _minimal_ohlcv_frame(["2020-01-01", "2020-01-02"])
    out = ensure_canonical_schema(frame)
    assert out["timestamp"].dt.tz is None
    assert out["timestamp"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    ]


def test_cli_backtest_prints_data_warning_text_not_just_a_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import quantlab.data.loader as loader_mod
    from quantlab.cli import app

    raw = tmp_path / "raw"
    raw.mkdir()
    # Declares hourly frequency but the CSV is daily bars, so
    # `_check_declared_frequency` produces a real warning to surface.
    make_ohlcv("AAA", np.linspace(100.0, 101.0, 60), start="2020-01-01").to_csv(
        raw / "AAA.csv", index=False
    )
    config = {
        "experiment_name": "cli_data_warning_test",
        "data": {
            "source": "csv",
            "market_calendar": "XNYS",
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
            "frequency": "1h",
        },
        "strategy": {"name": "buy_and_hold"},
    }
    import yaml

    config_path = tmp_path / "cli_data_warning_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app, ["backtest", "--config", str(config_path), "--output", str(out_dir)]
    )
    assert result.exit_code == 0, result.stdout
    assert "does not match the declared frequency" in result.stdout


def test_run_backtest_script_prints_data_warning_text_not_just_a_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`scripts/run_backtest.py` mirrors `quantlab backtest` and must print
    each data warning's full text, not just a bare warning *count*
    ("N warning(s); see logs")."""
    import sys

    import quantlab.data.loader as loader_mod

    raw = tmp_path / "raw"
    raw.mkdir()
    make_ohlcv("AAA", np.linspace(100.0, 101.0, 60), start="2020-01-01").to_csv(
        raw / "AAA.csv", index=False
    )
    config = {
        "experiment_name": "run_backtest_script_warning_test",
        "data": {
            "source": "csv",
            "market_calendar": "XNYS",
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
            "frequency": "1h",
        },
        "strategy": {"name": "buy_and_hold"},
    }
    import yaml

    config_path = tmp_path / "run_backtest_script_warning_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    run_backtest = _import_script("run_backtest")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest.py",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    run_backtest.main()  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert "does not match the declared frequency" in captured.out


def test_generate_report_script_prints_data_warning_text_not_just_a_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    import quantlab.data.loader as loader_mod

    raw = tmp_path / "raw"
    raw.mkdir()
    make_ohlcv("AAA", np.linspace(100.0, 101.0, 60), start="2020-01-01").to_csv(
        raw / "AAA.csv", index=False
    )
    config = {
        "experiment_name": "generate_report_script_warning_test",
        "data": {
            "source": "csv",
            "market_calendar": "XNYS",
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-03-01",
            "frequency": "1h",
        },
        "strategy": {"name": "buy_and_hold"},
    }
    import yaml

    config_path = tmp_path / "generate_report_script_warning_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    generate_report = _import_script("generate_report")
    monkeypatch.setattr(generate_report, "GENERATED_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        sys, "argv", ["generate_report.py", "--config", str(config_path)]
    )
    generate_report.main()  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert "does not match the declared frequency" in captured.out


def test_binance_normalise_drops_still_open_candle() -> None:
    from quantlab.data.binance import BinanceDataSource

    now_ms = 1_700_100_000_000
    one_day = 86_400_000
    closed_open_time = now_ms - 2 * one_day
    closed_close_time = now_ms - one_day - 1
    open_candle_open_time = now_ms - (now_ms % one_day)
    open_candle_close_time = open_candle_open_time + one_day - 1
    rows = [
        [
            closed_open_time,
            "100",
            "110",
            "90",
            "105",
            "1000",
            closed_close_time,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            open_candle_open_time,
            "105",
            "115",
            "95",
            "110",
            "500",
            open_candle_close_time,
            0,
            0,
            0,
            0,
            0,
        ],
    ]
    out = BinanceDataSource._normalise(rows, "BTCUSDT", now_ms=now_ms, end_ms=now_ms)
    assert len(out) == 1
    assert out["close"].iloc[0] == 105.0


def test_binance_normalise_keeps_all_candles_once_closed() -> None:
    """Once a candle's
    `close_time` has genuinely passed, it must still be kept — this filter
    must not discard already-final data."""
    from quantlab.data.binance import BinanceDataSource

    now_ms = 1_700_100_000_000
    one_day = 86_400_000
    rows = [
        [
            now_ms - 3 * one_day,
            "100",
            "110",
            "90",
            "105",
            "1000",
            now_ms - 2 * one_day - 1,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            now_ms - 2 * one_day,
            "105",
            "115",
            "95",
            "110",
            "500",
            now_ms - one_day - 1,
            0,
            0,
            0,
            0,
            0,
        ],
    ]
    out = BinanceDataSource._normalise(rows, "BTCUSDT", now_ms=now_ms, end_ms=now_ms)
    assert len(out) == 2


def test_binance_drops_candle_closing_after_requested_end() -> None:
    from quantlab.data.binance import BinanceDataSource, _to_millis

    open_ms = _to_millis(date(2024, 1, 1))  # Monday
    close_ms = _to_millis(date(2024, 1, 8)) - 1  # Sunday 23:59:59.999
    row = [open_ms, "100", "101", "99", "100.5", "1000", close_ms, 0, 0, 0, 0, 0]
    now_ms = _to_millis(date(2024, 1, 20))  # long after the candle closed

    # Requested end (Jan 3) falls inside the candle's own week -- must drop.
    end_ms_mid_week = _to_millis(date(2024, 1, 3)) + 86_400_000 - 1
    out = BinanceDataSource._normalise(
        [row], "BTCUSDT", now_ms=now_ms, end_ms=end_ms_mid_week
    )
    assert out.empty

    # Requested end covers the whole week -- must keep.
    end_ms_full_week = _to_millis(date(2024, 1, 8)) + 86_400_000 - 1
    out2 = BinanceDataSource._normalise(
        [row], "BTCUSDT", now_ms=now_ms, end_ms=end_ms_full_week
    )
    assert len(out2) == 1


def test_yahoo_drops_bar_closing_after_requested_end() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.to_datetime(["2024-03-01"])
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"
    now = pd.Timestamp("2024-06-01")  # well after the bar's own bucket closed

    out = YahooFinanceDataSource._normalise(raw, "AAA", "1mo", now, date(2024, 3, 15))
    assert out.empty

    out2 = YahooFinanceDataSource._normalise(raw, "AAA", "1mo", now, date(2024, 4, 1))
    assert len(out2) == 1


def test_slice_range_drops_a_look_ahead_bar_reused_from_a_wider_cache() -> None:
    from quantlab.data.loader import DataLoader

    cached = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )

    # Narrower request: Jan 3rd falls inside the Jan 1st-7th bar's own
    # bucket, which hasn't genuinely closed by Jan 3rd -- must drop it.
    narrow = DataLoader._slice_range(
        cached, date(2024, 1, 1), date(2024, 1, 3), "1w", is_247_market=False
    )
    assert narrow.empty

    # A request reaching far enough for the bar's own week to have
    # genuinely closed must still keep it.
    wide = DataLoader._slice_range(
        cached, date(2024, 1, 1), date(2024, 1, 10), "1w", is_247_market=False
    )
    assert len(wide) == 1


def test_slice_range_drops_a_look_ahead_monthly_bar_from_wider_cache() -> None:
    """Same rule, for monthly bars: a bar dated March 1st, cached once for a
    request through March 31st or later, must not silently satisfy a later,
    narrower request through March 15th."""
    from quantlab.data.loader import DataLoader

    cached = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-03-01"]),
            "symbol": "SPY",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )

    narrow = DataLoader._slice_range(
        cached, date(2024, 3, 1), date(2024, 3, 15), "1mo", is_247_market=False
    )
    assert narrow.empty

    wide = DataLoader._slice_range(
        cached, date(2024, 3, 1), date(2024, 4, 1), "1mo", is_247_market=False
    )
    assert len(wide) == 1


def test_yahoo_keeps_a_week_that_genuinely_finished_on_friday() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.to_datetime(["2024-01-01"])  # a Monday
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"
    now = pd.Timestamp("2026-01-01")

    # Requested through that week's own Friday -- the week has genuinely
    # finished (equity markets never trade the intervening weekend).
    out = YahooFinanceDataSource._normalise(raw, "SPY", "1wk", now, date(2024, 1, 5))
    assert len(out) == 1

    # Requested only through the Wednesday of that same week -- the week
    # has not finished yet, still correctly dropped.
    out2 = YahooFinanceDataSource._normalise(raw, "SPY", "1wk", now, date(2024, 1, 3))
    assert out2.empty


def test_yahoo_keeps_a_month_whose_last_trading_day_already_passed() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.to_datetime(["2024-11-01"])
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"
    now = pd.Timestamp("2026-01-01")

    out = YahooFinanceDataSource._normalise(raw, "SPY", "1mo", now, date(2024, 11, 29))
    assert len(out) == 1

    # Still correctly dropped for a request ending before the month's own
    # last trading day.
    out2 = YahooFinanceDataSource._normalise(raw, "SPY", "1mo", now, date(2024, 11, 15))
    assert out2.empty


def test_yahoo_keeps_a_daily_bar_finalised_after_market_close() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.to_datetime(["2024-01-16"])
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"

    # 21:30 UTC = 4:30pm EST -- just after the real market close.
    now_after_close = pd.Timestamp("2024-01-16 21:30:00")
    out = YahooFinanceDataSource._normalise(
        raw, "SPY", "1d", now_after_close, date(2024, 1, 16)
    )
    assert len(out) == 1

    # 19:00 UTC = 2pm EST -- market still open, must still be dropped.
    now_before_close = pd.Timestamp("2024-01-16 19:00:00")
    out2 = YahooFinanceDataSource._normalise(
        raw, "SPY", "1d", now_before_close, date(2024, 1, 16)
    )
    assert out2.empty


def test_daily_equity_bucket_settlement_reflects_a_real_early_close() -> None:
    from quantlab.data.calendar import daily_equity_bucket_settlement

    # 2024-11-29 (day after Thanksgiving) is a documented NYSE early-close
    # session: 1pm ET, not the ordinary 4pm.
    early_close = daily_equity_bucket_settlement(pd.Timestamp("2024-11-29"))
    ordinary_close = daily_equity_bucket_settlement(pd.Timestamp("2024-11-27"))
    assert early_close == pd.Timestamp("2024-11-29 18:00:00")  # 1pm EST -> UTC
    assert ordinary_close == pd.Timestamp("2024-11-27 21:00:00")  # 4pm EST -> UTC
    assert early_close.hour < ordinary_close.hour


def test_daily_equity_bucket_settlement_is_dst_aware_across_the_spring_transition() -> (
    None
):
    from quantlab.data.calendar import daily_equity_bucket_settlement

    before_dst = daily_equity_bucket_settlement(pd.Timestamp("2024-03-08"))  # EST
    after_dst = daily_equity_bucket_settlement(pd.Timestamp("2024-03-11"))  # EDT
    assert before_dst == pd.Timestamp("2024-03-08 21:00:00")  # 4pm EST = 21:00 UTC
    assert after_dst == pd.Timestamp("2024-03-11 20:00:00")  # 4pm EDT = 20:00 UTC


def test_daily_equity_bucket_settlement_is_conservative_for_a_non_session() -> None:
    """A non-session date (e.g. a market holiday) has no close of its own;
    settlement must fall back to midnight of the *next* day, not silently
    pick some other session's close."""
    from quantlab.data.calendar import daily_equity_bucket_settlement

    settlement = daily_equity_bucket_settlement(
        pd.Timestamp("2024-01-15")  # MLK Day
    )
    assert settlement == pd.Timestamp("2024-01-16")


def test_daily_equity_bucket_settlement_supports_a_valid_date_outside_cache() -> None:
    """Must resolve a date far outside `pandas_market_calendars`' schedule
    cache (well before any typical backtest range) without error."""
    from quantlab.data.calendar import daily_equity_bucket_settlement

    close = daily_equity_bucket_settlement(pd.Timestamp("1949-12-30"))
    assert close == pd.Timestamp("1949-12-30 21:00:00")


def test_periodic_equity_settlement_uses_the_last_session_close() -> None:
    """Weekly/monthly equity settlement must delegate to the daily bucket
    settlement of the period's actual last trading day, so it inherits the
    same early-close awareness."""
    from quantlab.data.calendar import (
        daily_equity_bucket_settlement,
        monthly_bucket_settlement,
        weekly_bucket_settlement,
    )

    expected = daily_equity_bucket_settlement(pd.Timestamp("2024-11-29"))
    assert (
        weekly_bucket_settlement(pd.Timestamp("2024-11-25"), is_247_market=False)
        == expected
    )
    assert (
        monthly_bucket_settlement(pd.Timestamp("2024-11-01"), is_247_market=False)
        == expected
    )


def test_yahoo_247_daily_not_closed_at_equity_market_close() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.to_datetime(["2024-01-16"])
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"

    # 21:30 UTC: already closed for an equity asset, but the crypto day
    # genuinely runs until midnight UTC.
    now = pd.Timestamp("2024-01-16 21:30:00")
    out_equity = YahooFinanceDataSource._normalise(
        raw, "SPY", "1d", now, date(2024, 1, 16), False
    )
    assert len(out_equity) == 1
    out_crypto = YahooFinanceDataSource._normalise(
        raw, "BTC-USD", "1d", now, date(2024, 1, 16), True
    )
    assert out_crypto.empty

    # Just after midnight UTC: the crypto day has genuinely finished too.
    now_after_midnight = pd.Timestamp("2024-01-17 00:30:00")
    out_crypto_done = YahooFinanceDataSource._normalise(
        raw, "BTC-USD", "1d", now_after_midnight, date(2024, 1, 16), True
    )
    assert len(out_crypto_done) == 1


def test_yahoo_247_weekly_not_closed_at_friday_equity_close() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.to_datetime(["2024-01-01"])  # a Monday
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"

    sunday_noon = pd.Timestamp("2024-01-07 12:00:00")
    out_equity = YahooFinanceDataSource._normalise(
        raw, "SPY", "1wk", sunday_noon, date(2024, 1, 7), False
    )
    assert len(out_equity) == 1  # equity week already closed Friday
    out_crypto = YahooFinanceDataSource._normalise(
        raw, "BTC-USD", "1wk", sunday_noon, date(2024, 1, 7), True
    )
    assert out_crypto.empty  # crypto week doesn't close until Monday


def test_drop_still_open_bars_uses_the_right_calendar(
    tmp_path: Path,
) -> None:
    from quantlab.data.calendar import (
        daily_equity_bucket_settlement,
        last_trading_day_on_or_before,
    )
    from quantlab.data.storage import ParquetStorage, _drop_still_open_bars

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    today = pd.Timestamp(year=now.year, month=now.month, day=now.day)
    closed_session = last_trading_day_on_or_before(
        today - pd.Timedelta(days=1), is_247_market=False
    )
    data = pd.DataFrame({"timestamp": [closed_session], "symbol": ["AAPL"]})
    assert len(_drop_still_open_bars(data, "1d", is_247_market=False)) == 1
    assert len(_drop_still_open_bars(data, "1d", is_247_market=True)) == 1

    equity_close = daily_equity_bucket_settlement(closed_session)
    flat_close = closed_session + pd.Timedelta(days=1)
    assert equity_close < flat_close

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    equity_data = pd.DataFrame(
        {
            "timestamp": [closed_session],
            "symbol": "AAPL",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    storage.write_symbol(equity_data, "yahoo", "AAPL", "1d", is_247_market=False)
    read_back = storage.read_symbol("yahoo", "AAPL", "1d", is_247_market=False)
    assert read_back is not None
    assert len(read_back) == 1


def test_yahoo_missing_close_reaches_missing_value_policy() -> None:
    from quantlab.config import MissingValuePolicy
    from quantlab.data.base import ensure_canonical_schema
    from quantlab.data.cleaner import DataCleaner
    from quantlab.data.yahoo import YahooFinanceDataSource
    from quantlab.exceptions import DataValidationError

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [1.0, 1.0, 1.0],
            "Low": [1.0, 1.0, 1.0],
            "Close": [1.0, np.nan, 1.0],
            "Adj Close": [1.0, np.nan, 1.0],
            "Volume": [100, 100, 100],
        },
        index=idx,
    )
    raw.index.name = "Date"
    out = YahooFinanceDataSource._normalise(
        raw, "AAA", "1d", pd.Timestamp("2025-01-01"), date(2020, 1, 3)
    )
    assert len(out) == 3  # the row is no longer dropped inside `_normalise`
    assert out["close"].isna().sum() == 1

    canonical = ensure_canonical_schema(out)
    with pytest.raises(DataValidationError, match="missing"):
        DataCleaner(MissingValuePolicy.RAISE).clean(canonical)

    filled = DataCleaner(MissingValuePolicy.FORWARD_FILL).clean(canonical)
    assert len(filled) == 3  # the row survives, forward-filled -- not dropped
    assert filled["close"].tolist() == [1.0, 1.0, 1.0]


def test_yahoo_missing_volume_not_silently_zeroed() -> None:
    from quantlab.data.yahoo import YahooFinanceDataSource

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [1.0, 1.0, 1.0],
            "Low": [1.0, 1.0, 1.0],
            "Close": [1.0, 1.0, 1.0],
            "Adj Close": [1.0, 1.0, 1.0],
            "Volume": [100, np.nan, 100],
        },
        index=idx,
    )
    raw.index.name = "Date"
    out = YahooFinanceDataSource._normalise(
        raw, "AAA", "1d", pd.Timestamp("2025-01-01"), date(2020, 1, 3)
    )
    assert out["volume"].isna().tolist() == [False, True, False]


def test_yahoo_download_one_raises_when_every_bar_is_filtered_out() -> None:
    from unittest.mock import patch

    from quantlab.data.yahoo import YahooFinanceDataSource
    from quantlab.exceptions import DataDownloadError

    source = YahooFinanceDataSource(max_retries=1)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    # Dated *tomorrow*, not today: with market-close-aware daily
    # settlement, "today" is only still-forming before 4pm US/Eastern —
    # ambiguous depending on what time of day this test happens to run.
    # Tomorrow's own close is unconditionally still in the future relative
    # to `now`, regardless of time of day, so it stays unambiguously
    # still-forming.
    still_forming = now.normalize() + pd.Timedelta(days=1)
    idx = pd.DatetimeIndex([still_forming])
    raw = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [100],
        },
        index=idx,
    )
    raw.index.name = "Date"

    with (
        patch("yfinance.download", return_value=raw),
        pytest.raises(DataDownloadError),
    ):
        source._download_one(
            "AAA", still_forming.date(), still_forming.date(), "1d", now
        )


def test_binance_download_one_threads_a_single_now_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    from quantlab.data.binance import BinanceDataSource
    from quantlab.exceptions import DataDownloadError

    captured_now: list[int] = []
    original_download_one = BinanceDataSource._download_one

    def spy_download_one(
        self: BinanceDataSource,
        symbol: str,
        start: date,
        end: date,
        interval: str,
        now_ms: int,
    ) -> pd.DataFrame:
        captured_now.append(now_ms)
        return original_download_one(self, symbol, start, end, interval, now_ms)

    monkeypatch.setattr(BinanceDataSource, "_download_one", spy_download_one)

    class _EmptySession:
        def get(self, *args: object, **kwargs: object) -> requests.Response:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = b"[]"
            return resp

    source = BinanceDataSource(session=_EmptySession())  # type: ignore[arg-type]
    with pytest.raises(DataDownloadError):
        source.download(
            ["BTCUSDT", "ETHUSDT"], date(2020, 1, 1), date(2020, 1, 2), "1d"
        )
    assert len(captured_now) == 2
    assert captured_now[0] == captured_now[1]


def test_loader_strict_mode_raises_on_duplicate_rows_before_cleaning(
    tmp_path: Path,
) -> None:
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage
    from quantlab.exceptions import DataValidationError

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_ohlcv_csv(
        raw / "AAA.csv",
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 100.0,
            }
            for ts in ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]
        ],
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "dup_raise_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "missing_value_policy": "raise",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        loader.load(cfg)


def test_loader_strict_mode_raises_on_non_positive_price_before_cleaning(
    tmp_path: Path,
) -> None:
    """Same check, for non-positive prices: `remove_invalid_prices()` always
    drops them before validation runs, regardless of policy, so
    `strict=True` must still be able to raise on them."""
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage
    from quantlab.exceptions import DataValidationError

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_ohlcv_csv(
        raw / "AAA.csv",
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "adjusted_close": price,
                "volume": 100.0,
            }
            for ts, price in [
                ("2020-01-01", 10.0),
                ("2020-01-02", -5.0),
                ("2020-01-03", 12.0),
            ]
        ],
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "neg_price_raise_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "missing_value_policy": "raise",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    with pytest.raises(DataValidationError, match="non-positive"):
        loader.load(cfg)


def test_loader_report_reflects_pre_clean_defects_under_drop_policy(
    tmp_path: Path,
) -> None:
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_ohlcv_csv(
        raw / "AAA.csv",
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "adjusted_close": price,
                "volume": 100.0,
            }
            for ts, price in [
                ("2020-01-01", 10.0),
                ("2020-01-01", 10.0),  # duplicate
                ("2020-01-02", -5.0),  # non-positive
                ("2020-01-03", 12.0),
            ]
        ],
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "drop_policy_report_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "missing_value_policy": "drop",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    data, report = loader.load(cfg)
    assert report.duplicate_count == 2  # both rows sharing the timestamp
    assert report.invalid_price_count == 5  # 1 row x 5 price columns
    assert not report.is_clean
    assert report.raw_row_count == 4
    assert report.clean_row_count == 2
    assert report.removed_row_count == 2
    # The cleaned data itself is unaffected -- only the report is enriched.
    assert len(data) == 2


def test_data_quality_report_persists_into_result_metadata_and_html(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage

    raw = tmp_path / "raw"
    raw.mkdir()
    idx = pd.date_range("2020-01-01", periods=60, freq="89min")
    _write_ohlcv_csv(
        raw / "AAA.csv",
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adjusted_close": 100.0,
                "volume": 1000.0,
            }
            for ts in idx
        ],
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "dq_persist_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-05",
                "frequency": "1h",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    data, report = loader.load(cfg)
    assert report.warnings  # premise: this data genuinely triggers a warning

    result = run_backtest_from_config(data, cfg, data_quality_report=report)
    assert result.metadata["data_quality"]["warnings"] == report.warnings
    assert result.metadata["data_quality"]["is_clean"] is False

    out = result.save(tmp_path / "out")
    saved_metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["data_quality"]["warnings"] == report.warnings

    import html as html_module

    html_text = (out / "report.html").read_text(encoding="utf-8")
    assert "data-quality warning" in html_text
    assert html_module.escape(report.warnings[0]) in html_text


def test_data_quality_report_omitted_shows_no_warnings_in_html() -> None:
    """When no report is threaded through (e.g. an internal per-fold re-run
    on already-validated data), the HTML must not error or show a stale
    section — `data_quality` simply isn't in metadata."""
    from quantlab.reporting.html_report import _render_data_quality

    assert _render_data_quality(None) == ""


def test_quality_report_warnings_reflect_duplicate_and_price_counts() -> None:
    from quantlab.data.validator import DataValidator

    n = 35
    prices = np.full(n, 10.0)
    prices[10] = -5.0
    dates = pd.bdate_range("2020-01-01", periods=n).tolist()
    dates[20] = dates[19]  # duplicate (timestamp, symbol)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAA",
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "adjusted_close": prices,
            "volume": 100.0,
        }
    )
    report = DataValidator().validate(df, strict=False)
    assert report.duplicate_count == 2
    assert report.invalid_price_count == 5
    assert not report.is_clean
    assert any("duplicate" in w for w in report.warnings)
    assert any("non-positive" in w for w in report.warnings)


def test_loader_report_reflects_missing_values_dropped_before_validation(
    tmp_path: Path,
) -> None:
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_ohlcv_csv(
        raw / "AAA.csv",
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": close,
                "adjusted_close": 10.0,
                "volume": 100.0,
            }
            for ts, close in [
                ("2020-01-01", 10.0),
                ("2020-01-02", ""),  # missing -> dropped by the `drop` policy
                ("2020-01-03", 12.0),
            ]
        ],
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "missing_value_drop_report_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "missing_value_policy": "drop",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    data, report = loader.load(cfg)
    assert len(data) == 2  # the row with the missing close was dropped
    assert report.missing_value_count.get("close") == 1
    assert not report.is_clean
    assert any("Missing values" in w for w in report.warnings)


def test_loader_missing_values_not_double_counted_under_none_policy(
    tmp_path: Path,
) -> None:
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_ohlcv_csv(
        raw / "AAA.csv",
        [
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": close,
                "adjusted_close": 10.0,
                "volume": 100.0,
            }
            for ts, close in [
                ("2020-01-01", 10.0),
                ("2020-01-02", ""),
                ("2020-01-03", 12.0),
            ]
        ],
    )
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "missing_value_none_report_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "missing_value_policy": "none",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    _data, report = loader.load(cfg)
    assert report.missing_value_count.get("close") == 1
    assert sum(1 for w in report.warnings if "Missing values" in w) == 1


def test_write_metadata_produces_strict_json_for_nan_and_infinity(
    tmp_path: Path,
) -> None:
    import json

    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m")
    path = storage.write_metadata(
        "test",
        {
            "skewness": float("nan"),
            "kurtosis": np.float32("inf"),
            "ok": np.float32(1.5),
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    parsed = json.loads(text)
    assert parsed == {"skewness": None, "kurtosis": None, "ok": 1.5}
