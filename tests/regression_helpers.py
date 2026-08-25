"""Shared fixtures and helpers for regression tests."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import ExperimentConfig
from tests.conftest import geometric_series, make_ohlcv


class _UniformCalendar(Mapping[str, str]):
    """A ``symbol_calendars`` mapping returning the same calendar for every
    symbol -- for tests that only care about one calendar and don't want to
    enumerate every symbol a frame happens to contain."""

    def __init__(self, calendar: str) -> None:
        self._calendar = calendar

    def __getitem__(self, key: str) -> str:
        return self._calendar

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _holdout_config() -> tuple[pd.DataFrame, ExperimentConfig]:
    frames = [
        make_ohlcv(sym, geometric_series(500, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [
            ("AAA", 1, 0.0007),
            ("BBB", 2, 0.0003),
            ("CCC", 3, 0.0005),
        ]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "holdout_report",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 60,
                    "skip_period": 5,
                    "top_fraction": 0.5,
                },
            },
            "portfolio": {"allocator": "inverse_volatility"},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {"initial_capital": 100_000},
            "validation": {
                "method": "holdout",
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
            },
        }
    )
    return data, cfg


def _wf_experiment_config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "wf_experiment",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "yahoo", "calendar": "XNYS"},
                    {"symbol": "B", "source": "yahoo", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )


_FAKE_DATA_HASH = "deadbeef"


_FAKE_CODE_HASH = "c0defeed"


_FAKE_GENERATOR_HASH = "9ea3a708"


_FAKE_GIT_COMMIT = "abc1234"


_FAKE_DEPENDENCY_VERSIONS = {"quantlab": "0.1.0", "pandas": "2.0.0"}


_FAKE_WF_RUN_TIMESTAMP = "2025-01-09T12:00:00+00:00"


def _write_wf_artifacts(exp_dir: Path, config: ExperimentConfig) -> None:
    import hashlib

    config.to_yaml(exp_dir / "config.yaml")
    pd.DataFrame({"fold": [0, 1], "test_sharpe": [0.3, 0.5]}).to_csv(
        exp_dir / "walk_forward_results.csv", index=False
    )
    pd.Series([0.01, 0.02], name="return").to_csv(
        exp_dir / "walk_forward_oos_returns.csv"
    )
    pd.Series([100.0, 101.0], name="equity").to_csv(
        exp_dir / "walk_forward_oos_equity.csv"
    )
    pd.DataFrame({"scenario": ["baseline"], "sharpe": [0.4]}).to_csv(
        exp_dir / "stress_tests.csv", index=False
    )
    checksums = {
        name: hashlib.sha256((exp_dir / name).read_bytes()).hexdigest()
        for name in (
            "walk_forward_results.csv",
            "walk_forward_oos_returns.csv",
            "walk_forward_oos_equity.csv",
            "stress_tests.csv",
        )
    }
    old_metadata = {
        "walk_forward_oos_metrics": {"sharpe_ratio": 0.42, "cagr": 0.05},
        "walk_forward_config_snapshot": config.model_dump(mode="json"),
        "walk_forward_run_timestamp": _FAKE_WF_RUN_TIMESTAMP,
        "walk_forward_csv_checksums": checksums,
        "data_hash": _FAKE_DATA_HASH,
        "code_hash": _FAKE_CODE_HASH,
        "generator_hash": _FAKE_GENERATOR_HASH,
        "git_commit": _FAKE_GIT_COMMIT,
        "git_dirty": False,
        "dependency_versions": _FAKE_DEPENDENCY_VERSIONS,
        "some_other_key": "unrelated",
    }
    (exp_dir / "metadata.json").write_text(json.dumps(old_metadata), encoding="utf-8")


class _FakeResult:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.metadata: dict = {
            "data_hash": _FAKE_DATA_HASH,
            "code_hash": _FAKE_CODE_HASH,
            "generator_hash": _FAKE_GENERATOR_HASH,
            "git_commit": _FAKE_GIT_COMMIT,
            "git_dirty": False,
            "dependency_versions": _FAKE_DEPENDENCY_VERSIONS,
        }


def _try_strategy(
    name: str,
    parameters: dict[str, object],
    *,
    portfolio: dict[str, object] | None = None,
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "B", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {"name": name, "parameters": parameters},
            "portfolio": portfolio or {},
        }
    )


def _hourly_frame(
    sessions: list[pd.Timestamp], remove: dict[int, set[int]] | None = None
) -> pd.DataFrame:
    remove = remove or {}
    dates = [
        day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h)
        for i, day in enumerate(sessions)
        for h in range(7)
        if h not in remove.get(i, set())
    ]
    return pd.DataFrame(
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


def _rf_test_setup(
    risk_free_rate: float = 0.02,
) -> tuple[pd.DataFrame, ExperimentConfig]:
    frames = [
        make_ohlcv(sym, geometric_series(400, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [("AAA", 1, 0.0007), ("BBB", 2, 0.0003)]
    ]
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "rf_consistency",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-06-01",
            },
            "strategy": {"name": "buy_and_hold"},
            "backtest": {"risk_free_rate": risk_free_rate},
        }
    )
    return data, cfg


def _frame_with_one_missing_row() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": [100.0, 101.0, np.nan, 103.0, 104.0],
            "high": [101.0, 102.0, np.nan, 104.0, 105.0],
            "low": [99.0, 100.0, np.nan, 102.0, 103.0],
            "close": [100.5, 101.5, np.nan, 103.5, 104.5],
            "adjusted_close": [100.5, 101.5, np.nan, 103.5, 104.5],
            "volume": [1000.0] * 5,
        }
    )


def _hourly_symbol_frame(freq: str, periods: int, symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=periods, freq=freq)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )


def _market_calendar_config(
    *, source: str = "csv", calendar: str | None = None, **data_overrides: Any
) -> ExperimentConfig:
    instrument: dict[str, Any] = {"symbol": "BTC", "source": source}
    if calendar is not None:
        instrument["calendar"] = calendar
    payload: dict[str, Any] = {
        "experiment_name": "test",
        "data": {
            "instruments": [instrument],
            "start_date": "2020-01-01",
            "end_date": "2020-06-01",
            "frequency": "1h",
            **data_overrides,
        },
        "strategy": {"name": "buy_and_hold", "parameters": {}},
        "portfolio": {"allocator": "equal_weight"},
        "execution": {},
        "backtest": {},
        "validation": {"method": "holdout"},
        "reproducibility": {"random_seed": 42},
    }
    return ExperimentConfig.from_dict(payload)


def _benchmark_contamination_frame() -> pd.DataFrame:
    """A short tradable symbol plus a much longer-history benchmark.

    ``BENCH`` starts 10 days earlier than ``AAA``, so pivoting over both
    symbols together yields a calendar the tradable universe never actually
    traded on.
    """
    tradable_dates = pd.date_range("2020-01-11", periods=10, freq="D")
    benchmark_dates = pd.date_range("2020-01-01", periods=30, freq="D")
    frames = [
        make_ohlcv(
            "AAA", np.linspace(100.0, 110.0, len(tradable_dates)), start="2020-01-11"
        ),
        make_ohlcv(
            "BENCH", np.linspace(50.0, 80.0, len(benchmark_dates)), start="2020-01-01"
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def _flat_execution_model() -> Any:
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    cfg = ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    return ExecutionModel.from_config(cfg, average_daily_volume=1.0)


def _base_config_dict() -> dict[str, Any]:
    return {
        "experiment_name": "test",
        "data": {
            "instruments": [
                {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
            ],
            "start_date": "2020-01-01",
            "end_date": "2020-06-01",
        },
        "strategy": {"name": "buy_and_hold", "parameters": {}},
        "portfolio": {"allocator": "equal_weight"},
        "execution": {},
        "backtest": {},
        "validation": {"method": "holdout"},
        "reproducibility": {"random_seed": 42},
    }


def _make_hourly_frame(
    dates_hours: list[pd.Timestamp], symbol: str = "AAA"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": dates_hours,
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )


def _write_daily_cache(
    storage: object, source: str, symbol: str, dates: pd.DatetimeIndex
) -> None:
    from quantlab.data.storage import ParquetStorage

    assert isinstance(storage, ParquetStorage)
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )
    calendar = "24/7" if source == "binance" else "XNYS"
    storage.write_symbol(data, source, symbol, "1d", calendar=calendar)


def _ohlcv_at(dates: pd.DatetimeIndex | list, symbol: str = "SPY") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )


def _ohlcv_frame(idx: pd.DatetimeIndex, symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )


def _minimal_ohlcv_frame(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["AAA"] * len(timestamps),
            "open": [1.0] * len(timestamps),
            "high": [1.0] * len(timestamps),
            "low": [1.0] * len(timestamps),
            "close": [1.0] * len(timestamps),
            "adjusted_close": [1.0] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        }
    )


def _import_script(name: str) -> object:
    import importlib
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(scripts_dir))


def _write_ohlcv_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
