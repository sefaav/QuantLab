"""Direct tests for benchmark construction and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.benchmark import build_benchmark, cash_returns
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
from quantlab.data.loader import DataLoader
from quantlab.exceptions import BacktestError


def _market_data(
    symbols: dict[str, list[float]], index: pd.DatetimeIndex
) -> pd.DataFrame:
    frames = []
    for symbol, prices in symbols.items():
        frames.append(
            pd.DataFrame(
                {
                    TIMESTAMP: index,
                    SYMBOL: symbol,
                    OPEN: prices,
                    HIGH: prices,
                    LOW: prices,
                    CLOSE: prices,
                    ADJUSTED_CLOSE: prices,
                    VOLUME: 1_000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_unknown_benchmark_kind_raises() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = _market_data({"AAA": [100.0, 101.0, 102.0]}, index)

    with pytest.raises(ValueError, match="Unknown benchmark kind"):
        build_benchmark(data, index, kind="typo", benchmark_symbol="AAA")


def test_configured_benchmark_symbol_must_be_loaded() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = _market_data({"AAA": [100.0, 101.0, 102.0]}, index)

    with pytest.raises(BacktestError, match="is absent"):
        build_benchmark(data, index, benchmark_symbol="SPY")


def test_missing_benchmark_return_after_initial_period_raises() -> None:
    benchmark_index = pd.DatetimeIndex(["2024-01-01", "2024-01-03"])
    portfolio_index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = _market_data({"AAA": [100.0, 102.0]}, benchmark_index)

    with pytest.raises(BacktestError, match="missing on portfolio dates"):
        build_benchmark(data, portfolio_index, benchmark_symbol="AAA")


def test_first_benchmark_period_is_zero() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = _market_data({"AAA": [100.0, 110.0, 121.0]}, index)

    result = build_benchmark(data, index, benchmark_symbol="AAA")

    assert result is not None
    assert result.tolist() == pytest.approx([0.0, 0.1, 0.1])


def test_equal_weight_benchmark_is_rebalanced_each_period() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = _market_data(
        {"AAA": [100.0, 110.0, 110.0], "BBB": [100.0, 100.0, 120.0]}, index
    )

    result = build_benchmark(data, index, kind="equal_weight")

    assert result is not None
    assert result.tolist() == pytest.approx([0.0, 0.05, 0.10])


def test_equal_weight_benchmark_rejects_missing_asset_return() -> None:
    full_index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.concat(
        [
            _market_data({"AAA": [100.0, 101.0, 102.0]}, full_index),
            _market_data(
                {"BBB": [100.0, 102.0]},
                pd.DatetimeIndex([full_index[0], full_index[2]]),
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(BacktestError, match="missing on portfolio dates"):
        build_benchmark(data, full_index, kind="equal_weight")


def test_first_asset_benchmark_uses_explicit_universe_order() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    # Deliberately put BBB first in the frame: the benchmark must follow the
    # configured universe order, not an incidental pivot-column order.
    data = _market_data(
        {"BBB": [100.0, 100.0, 100.0], "AAA": [100.0, 110.0, 121.0]}, index
    )

    result = build_benchmark(data, index, kind="first_asset", first_asset_symbol="AAA")

    assert result is not None
    assert result.tolist() == pytest.approx([0.0, 0.1, 0.1])


def test_first_asset_benchmark_requires_explicit_symbol() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    data = _market_data({"AAA": [100.0, 101.0]}, index)

    with pytest.raises(ValueError, match="first_asset_symbol"):
        build_benchmark(data, index, kind="first_asset")


def test_cash_benchmark_uses_configured_rate() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = _market_data({"AAA": [100.0, 101.0, 102.0]}, index)

    result = build_benchmark(
        data,
        index,
        kind="cash",
        risk_free_rate=0.12,
        periods_per_year=12,
    )

    assert result is not None
    assert result.tolist() == pytest.approx([0.0, 0.01, 0.01])


def test_loader_fetches_extra_data_only_for_symbol_benchmark() -> None:
    base = {
        "experiment_name": "benchmark_fetch",
        "data": {
            "instruments": [
                {"symbol": s, "source": "csv", "calendar": "XNYS"}
                for s in ["AAA", "BBB"]
            ],
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
        "strategy": {"name": "buy_and_hold"},
    }
    symbol_cfg = ExperimentConfig.from_dict(
        {
            **base,
            "backtest": {
                "benchmark_kind": "symbol",
                "benchmark": {"symbol": "SPY", "source": "csv", "calendar": "XNYS"},
            },
        }
    )
    equal_weight_cfg = ExperimentConfig.from_dict(
        {**base, "backtest": {"benchmark_kind": "equal_weight"}}
    )

    assert DataLoader._symbols_to_fetch(symbol_cfg) == ["AAA", "BBB", "SPY"]
    assert DataLoader._symbols_to_fetch(equal_weight_cfg) == ["AAA", "BBB"]


@pytest.mark.parametrize("periods_per_year", [0, -1, 1.5, True])
def test_cash_returns_rejects_invalid_periods_per_year(
    periods_per_year: object,
) -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")

    with pytest.raises(ValueError, match="periods_per_year"):
        cash_returns(index, 0.02, periods_per_year)  # type: ignore[arg-type]


@pytest.mark.parametrize("risk_free_rate", [np.nan, np.inf, True])
def test_cash_returns_rejects_invalid_rate(risk_free_rate: object) -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")

    with pytest.raises(ValueError, match="risk_free_rate"):
        cash_returns(index, risk_free_rate, 252)  # type: ignore[arg-type]
