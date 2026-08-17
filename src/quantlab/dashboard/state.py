"""Streamlit-independent dashboard configuration and execution helpers."""

from __future__ import annotations

import csv
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.data.base import SymbolSuggestion
from quantlab.data.binance import BinanceDataSource
from quantlab.data.loader import DataLoader
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import BacktestError

#: A curated, bundled reference list (S&P 500 constituents plus major ETFs)
#: shipped with the package. Yahoo has no downloadable "every symbol"
#: endpoint the way Binance does, so this stands in as an offline, instant
#: universe for the dashboard's autocomplete — not an exhaustive directory of
#: every symbol Yahoo Finance can actually serve.
_YAHOO_COMMON_SYMBOLS_CSV = Path(__file__).parent / "data" / "yahoo_common_symbols.csv"


def build_config_from_inputs(inputs: dict[str, Any]) -> ExperimentConfig:
    """Assemble a validated :class:`ExperimentConfig` from dashboard inputs."""
    return ExperimentConfig.from_dict(
        {
            "experiment_name": inputs.get("experiment_name", "dashboard_run"),
            "data": {
                "source": inputs["source"],
                "symbols": inputs["symbols"],
                "start_date": inputs["start_date"],
                "end_date": inputs["end_date"],
                "frequency": inputs.get("frequency", "1d"),
                "missing_value_policy": inputs.get("missing_value_policy", "drop"),
                "market_calendar": inputs.get("market_calendar"),
                "use_bundled_demo_data": inputs.get("use_bundled_demo_data", False),
            },
            "strategy": {
                "name": inputs["strategy_name"],
                "parameters": inputs.get("strategy_parameters", {}),
            },
            "portfolio": {
                "allocator": inputs["allocator"],
                "maximum_weight": inputs.get("maximum_weight"),
                "long_only": inputs.get("long_only", False),
                "target_volatility": inputs.get("target_volatility"),
                "volatility_window": inputs.get("volatility_window", 63),
                "maximum_leverage": inputs.get("maximum_leverage", 1.5),
                "rebalance_frequency": inputs.get("rebalance_frequency", "monthly"),
            },
            "execution": {
                "commission_bps": inputs.get("commission_bps", 2.0),
                "spread_bps": inputs.get("spread_bps", 3.0),
                "slippage_bps": inputs.get("slippage_bps", 2.0),
            },
            "backtest": {
                "initial_capital": inputs.get("initial_capital", 100_000.0),
                "benchmark_kind": inputs.get("benchmark_kind", "symbol"),
                "benchmark_symbol": inputs.get("benchmark_symbol") or None,
                "risk_free_rate": inputs.get("risk_free_rate", 0.02),
            },
            "validation": {
                "method": "holdout",
                "validation_ratio": inputs.get("validation_ratio"),
                "test_ratio": inputs.get("test_ratio"),
            },
            "reproducibility": {"random_seed": inputs.get("random_seed", 42)},
        }
    )


def run_dashboard_backtest(
    config: ExperimentConfig,
) -> tuple[BacktestResult, list[str]]:
    """Load data and run the backtest, returning the result and data warnings."""
    data, report = DataLoader().load(config)
    result = run_backtest_from_config(data, config, data_quality_report=report)
    return result, report.warnings


def run_dashboard_stress_tests(
    config: ExperimentConfig, expected_data_hash: str
) -> pd.DataFrame:
    """Run stress scenarios only if reloaded data matches the displayed run."""
    from quantlab.validation.robustness import run_stress_tests

    data, _ = DataLoader().load(config)
    actual_data_hash = ParquetStorage.hash_frame(data)
    if actual_data_hash != expected_data_hash:
        raise BacktestError(
            "Market data changed since the displayed backtest. Run the "
            "backtest again before running stress tests."
        )
    return run_stress_tests(data, config)


def default_end_date() -> date:
    """A safe default end date that does not depend on wall-clock time."""
    return date(2024, 12, 31)


@lru_cache(maxsize=1)
def yahoo_common_symbols() -> list[SymbolSuggestion]:
    """The bundled S&P 500 + major-ETF reference list, loaded once.

    Static and shipped with the package (no network call), so the dashboard
    can offer it as one instant, client-side-filtered dropdown — the same
    shape as :func:`binance_trading_symbols`, just from a file instead of a
    live download.
    """
    with _YAHOO_COMMON_SYMBOLS_CSV.open(newline="", encoding="utf-8") as f:
        return [
            SymbolSuggestion(symbol=row["symbol"], description=row["description"])
            for row in csv.DictReader(f)
        ]


def binance_trading_symbols() -> list[SymbolSuggestion]:
    """Fetch Binance's full active spot-symbol universe.

    Meant to be cached by the caller (fetching the whole universe once):
    small enough to load entirely upfront, so the dashboard can offer it as
    one instant, client-side-filtered dropdown instead of querying per
    keystroke.
    """
    return BinanceDataSource().list_trading_symbols()
