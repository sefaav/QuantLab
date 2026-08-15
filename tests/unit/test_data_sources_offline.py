"""Offline tests for Yahoo/Binance normalisation logic (no network calls).

These exercise normalisation and error paths without making HTTP calls, keeping
the tests deterministic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from quantlab.data.binance import BinanceDataSource, _to_millis
from quantlab.data.yahoo import YahooFinanceDataSource
from quantlab.exceptions import DataDownloadError


def test_yahoo_normalise_flat_columns() -> None:
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [100.5, 101.5],
            "Volume": [1000, 2000],
        },
        index=pd.date_range("2020-01-01", periods=2, name="Date"),
    )
    out = YahooFinanceDataSource._normalise(
        raw, "spy", "1d", pd.Timestamp("2025-01-01"), date(2025, 1, 1)
    )
    assert list(out["symbol"].unique()) == ["SPY"]
    assert out["close"].tolist() == [100.5, 101.5]
    assert out["timestamp"].is_monotonic_increasing


def test_yahoo_normalise_multiindex_columns() -> None:
    idx = pd.date_range("2020-01-01", periods=2, name="Date")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY"]]
    )
    raw = pd.DataFrame(
        np.array([[100, 101, 99, 100.5, 1000], [101, 102, 100, 101.5, 2000]]),
        index=idx,
        columns=cols,
    )
    out = YahooFinanceDataSource._normalise(
        raw, "SPY", "1d", pd.Timestamp("2025-01-01"), date(2025, 1, 1)
    )
    assert len(out) == 2
    assert (out["close"] > 0).all()


def test_yahoo_unsupported_frequency() -> None:
    source = YahooFinanceDataSource()
    with pytest.raises(DataDownloadError, match="Unsupported frequency"):
        source.download(
            ["SPY"],
            __import__("datetime").date(2020, 1, 1),
            __import__("datetime").date(2020, 2, 1),
            "5m",
        )


def test_binance_normalise_klines() -> None:
    rows = [
        [1577836800000, "100.0", "101.0", "99.0", "100.5", "1000", 0, 0, 0, 0, 0, 0],
        [1577923200000, "100.5", "102.0", "100.0", "101.5", "1200", 0, 0, 0, 0, 0, 0],
    ]
    # `close_time` is a placeholder (0) here since this test only exercises
    # column mapping, not the open-candle filter — `now_ms`/`end_ms` just
    # need to be any instant at or after that placeholder for both rows to
    # count as closed and within the requested range.
    out = BinanceDataSource._normalise(rows, "btcusdt", now_ms=1, end_ms=0)
    assert list(out["symbol"].unique()) == ["BTCUSDT"]
    assert out["adjusted_close"].tolist() == out["close"].tolist()
    assert out["timestamp"].is_monotonic_increasing


def test_binance_unsupported_frequency() -> None:
    source = BinanceDataSource()
    with pytest.raises(DataDownloadError, match="Unsupported frequency"):
        source.download(
            ["BTCUSDT"],
            __import__("datetime").date(2020, 1, 1),
            __import__("datetime").date(2020, 2, 1),
            "5m",
        )


@pytest.mark.parametrize("max_retries", [0, -1, True, 1.5])
def test_binance_rejects_invalid_retry_count(max_retries: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BinanceDataSource(max_retries=max_retries)  # type: ignore[arg-type]


def test_binance_preserves_an_injected_falsy_session() -> None:
    class FalsySession:
        def __bool__(self) -> bool:
            return False

    session = FalsySession()
    source = BinanceDataSource(session=session)  # type: ignore[arg-type]
    assert source.session is session


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_binance_does_not_retry_a_permanent_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession([_FakeResponse(400, {"code": -1})])
    sleeps: list[float] = []
    monkeypatch.setattr("quantlab.data.binance.time.sleep", sleeps.append)
    source = BinanceDataSource(max_retries=3, session=session)  # type: ignore[arg-type]

    with pytest.raises(DataDownloadError, match="HTTP 400"):
        source._request("BTCUSDT", "1d", 0, 1)

    assert session.calls == 1
    assert sleeps == []


def test_binance_rate_limit_error_is_retried_without_final_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession([_FakeResponse(429, [], {"Retry-After": "0.25"})])
    sleeps: list[float] = []
    monkeypatch.setattr("quantlab.data.binance.time.sleep", sleeps.append)
    source = BinanceDataSource(max_retries=2, session=session)  # type: ignore[arg-type]

    with pytest.raises(DataDownloadError, match="HTTP 429"):
        source._request("BTCUSDT", "1d", 0, 1)

    assert session.calls == 2
    assert sleeps == [0.25]


def test_binance_rejects_malformed_json_payload() -> None:
    session = _FakeSession([_FakeResponse(200, {"code": 0})])
    source = BinanceDataSource(max_retries=1, session=session)  # type: ignore[arg-type]
    with pytest.raises(DataDownloadError, match="non-list"):
        source._request("BTCUSDT", "1d", 0, 1)


def test_binance_rejects_malformed_kline_shape() -> None:
    session = _FakeSession([_FakeResponse(200, [[1, 2, 3]])])
    source = BinanceDataSource(max_retries=1, session=session)  # type: ignore[arg-type]
    with pytest.raises(DataDownloadError, match="12 fields"):
        source._request("BTCUSDT", "1d", 0, 1)


def test_to_millis_utc() -> None:
    import datetime as dt

    ms = _to_millis(dt.date(2020, 1, 1))
    assert ms == 1577836800000
