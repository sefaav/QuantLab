"""Backtesting engine, accounting and results."""

from __future__ import annotations

from quantlab.backtesting.accounting import AccountingResult, run_accounting
from quantlab.backtesting.benchmark import build_benchmark
from quantlab.backtesting.engine import BacktestEngine
from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.backtesting.trade_log import build_trade_log

__all__ = [
    "AccountingResult",
    "BacktestEngine",
    "BacktestResult",
    "build_benchmark",
    "build_trade_log",
    "run_accounting",
    "run_backtest_from_config",
]
