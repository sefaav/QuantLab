"""Regression tests for public market-data APIs and quality reporting."""

from __future__ import annotations

import logging
import sys
from dataclasses import FrozenInstanceError
from datetime import date
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantlab.config import ExperimentConfig
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
from quantlab.data.universe import Universe
from quantlab.data.validator import DataValidator
from quantlab.data.yahoo import YahooFinanceDataSource
from quantlab.exceptions import DataDownloadError, DataValidationError


def _ohlcv(timestamps: pd.DatetimeIndex, *, symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIMESTAMP: timestamps,
            SYMBOL: symbol,
            OPEN: 100.0,
            HIGH: 101.0,
            LOW: 99.0,
            CLOSE: 100.0,
            ADJUSTED_CLOSE: 100.0,
            VOLUME: 1_000.0,
        }
    )


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, 123, " "])
def test_universe_rejects_invalid_symbols(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="Symbol"):
        Universe.from_symbols(["SPY", value])  # type: ignore[list-item]


def test_universe_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        Universe.from_symbols([])


def test_universe_is_immutable() -> None:
    universe = Universe.from_symbols(["SPY", "QQQ"])
    assert universe.symbols == ("SPY", "QQQ")
    with pytest.raises(FrozenInstanceError):
        universe.name = "changed"  # type: ignore[misc]


def test_universe_csv_requires_requested_column(tmp_path: Any) -> None:
    path = tmp_path / "symbols.csv"
    pd.DataFrame({"ticker": ["SPY"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no 'symbol' column"):
        Universe.from_csv(path)


def test_universe_csv_rejects_empty_file(tmp_path: Any) -> None:
    path = tmp_path / "symbols.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        Universe.from_csv(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_gap_periods": 0}, "max_gap_periods"),
        ({"max_gap_periods": True}, "max_gap_periods"),
        ({"min_coverage_rows": -1}, "min_coverage_rows"),
        ({"expected_frequency": "typo"}, "expected_frequency"),
        ({"is_247_market": 1}, "is_247_market"),
    ],
)
def test_validator_rejects_invalid_constructor_arguments(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DataValidator(**kwargs)  # type: ignore[arg-type]


def test_validator_uses_xnys_session_for_weekend_end() -> None:
    timestamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2024-01-01", "2024-01-05"):
        timestamps.extend(
            pd.date_range(
                day + pd.Timedelta(hours=14, minutes=30), periods=7, freq="1h"
            )
        )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=False
    ).validate(
        _ohlcv(pd.DatetimeIndex(timestamps), symbol="SPY"),
        start=date(2024, 1, 1),
        end=date(2024, 1, 7),
    )
    assert not any("data ends" in warning for warning in report.warnings)


def test_validator_counts_each_ohlc_inconsistent_row_once() -> None:
    frame = _ohlcv(pd.DatetimeIndex(["2024-01-02"]))
    frame.loc[0, [OPEN, HIGH, LOW, CLOSE]] = [10.0, 5.0, 15.0, 10.0]
    report = DataValidator(min_coverage_rows=1).validate(frame)
    assert report.invalid_price_count == 1
    assert "1 OHLC-inconsistent rows" in report.warnings[0]


def test_validator_rejects_non_numeric_canonical_value_cleanly() -> None:
    frame = _ohlcv(pd.DatetimeIndex(["2024-01-02"]))
    frame[OPEN] = frame[OPEN].astype(object)
    frame.loc[0, OPEN] = "not-a-price"
    with pytest.raises(DataValidationError, match="non-numeric"):
        DataValidator(min_coverage_rows=1).validate(frame)


def test_validator_rejects_missing_canonical_columns() -> None:
    with pytest.raises(DataValidationError, match="canonical columns"):
        DataValidator().validate(pd.DataFrame({TIMESTAMP: ["2024-01-01"]}))


def test_missing_period_keeps_symbol_and_serialises_dates() -> None:
    timestamps = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]
    )
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv(timestamps, symbol="BTCUSDT"))
    assert len(report.missing_periods) == 1
    period = report.missing_periods[0]
    assert period.symbol == "BTCUSDT"
    assert report.to_dict()["missing_periods"] == [period.to_dict()]


def test_declared_frequency_drives_gap_detection() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="2h")
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=1, is_247_market=True
    ).validate(_ohlcv(timestamps, symbol="BTCUSDT"))
    assert len(report.missing_periods) == 3


def test_daily_equity_gap_counts_xnys_sessions() -> None:
    tolerated = DataValidator(
        expected_frequency="1d", max_gap_periods=5, min_coverage_rows=1
    ).validate(_ohlcv(pd.DatetimeIndex(["2024-01-05", "2024-01-12"])))
    assert tolerated.missing_periods == []

    flagged = DataValidator(
        expected_frequency="1d", max_gap_periods=5, min_coverage_rows=1
    ).validate(_ohlcv(pd.DatetimeIndex(["2024-01-05", "2024-01-16"])))
    assert len(flagged.missing_periods) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_retries": 0}, "max_retries"),
        ({"max_retries": True}, "max_retries"),
        ({"retry_backoff_seconds": -1}, "retry_backoff_seconds"),
        ({"retry_backoff_seconds": float("nan")}, "retry_backoff_seconds"),
        ({"retry_backoff_seconds": True}, "retry_backoff_seconds"),
    ],
)
def test_yahoo_rejects_invalid_retry_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        YahooFinanceDataSource(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("symbols", "start", "end", "frequency", "is_247_market", "message"),
    [
        ([], date(2024, 1, 1), date(2024, 1, 2), "1d", False, "at least one symbol"),
        ([""], date(2024, 1, 1), date(2024, 1, 2), "1d", False, "non-empty string"),
        (
            ["SPY"],
            date(2024, 1, 2),
            date(2024, 1, 1),
            "1d",
            False,
            "on or before end",
        ),
        (["SPY"], date(2024, 1, 1), date(2024, 1, 2), "1d", 1, "boolean"),
    ],
)
def test_yahoo_validates_direct_download_arguments(
    symbols: list[str],
    start: date,
    end: date,
    frequency: str,
    is_247_market: object,
    message: str,
) -> None:
    with pytest.raises(DataDownloadError, match=message):
        YahooFinanceDataSource().download(
            symbols,
            start,
            end,
            frequency,
            is_247_market=is_247_market,  # type: ignore[arg-type]
        )


def test_yahoo_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(DataDownloadError, match="optional 'yfinance' dependency"):
        YahooFinanceDataSource(max_retries=1).download(
            ["SPY"], date(2024, 1, 1), date(2024, 1, 2)
        )


def test_yahoo_does_not_retry_invalid_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    fake = ModuleType("yfinance")

    def download(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {"Date": [pd.Timestamp("2024-01-01")], "Close": [100.0]}
        ).set_index("Date")

    fake.download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    with pytest.raises(DataDownloadError, match="invalid schema"):
        YahooFinanceDataSource(max_retries=3, retry_backoff_seconds=0).download(
            ["SPY"], date(2024, 1, 1), date(2024, 1, 2)
        )
    assert calls == 1


def test_yahoo_logs_adjusted_close_fallback(caplog: pytest.LogCaptureFixture) -> None:
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.0],
            "Volume": [1_000.0],
        }
    ).set_index("Date")
    with caplog.at_level(logging.WARNING):
        out = YahooFinanceDataSource._normalise(
            raw,
            "SPY",
            "1d",
            pd.Timestamp("2024-01-03 22:00"),
            date(2024, 1, 2),
        )
    assert out[ADJUSTED_CLOSE].iloc[0] == out[CLOSE].iloc[0]
    assert "has no adjusted close" in caplog.text


def test_yahoo_config_can_select_continuous_calendar() -> None:
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "yahoo_crypto",
            "data": {
                "source": "yahoo",
                "symbols": ["BTC-USD"],
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "market_calendar": "24/7",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert config.data.is_247_market is True
