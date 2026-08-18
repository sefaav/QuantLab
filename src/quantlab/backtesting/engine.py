"""Backtest engine.

Orchestrates the exact ordered pipeline:

1-2. load & validate data (done by the caller / DataLoader)
3.   compute features (inside the strategy)
4.   generate signals at t
5.   transform signals into target weights
6.   apply constraints
7.   shift weights before computing returns  ← look-ahead barrier
8.   compute turnover
9.   compute gross returns
10.  compute costs
11.  subtract costs
12.  update portfolio value
13.  record trades
14.  compute metrics
15.  produce the result

Signal → allocation → execution → accounting are distinct, independently
tested steps, kept separate rather than mixed together so each stage's
correctness can be verified on its own.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata
from numbers import Integral
from pathlib import Path

import pandas as pd

from quantlab.backtesting.accounting import (
    compute_asset_returns,
    portfolio_metrics_from_accounting,
    run_accounting,
)
from quantlab.backtesting.benchmark import build_benchmark
from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.trade_log import build_trade_log
from quantlab.config import BenchmarkKind, ExperimentConfig
from quantlab.constants import SYMBOL, TIMESTAMP
from quantlab.data.base import price_matrix
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionModel
from quantlab.execution.orders import validate_execution_frame
from quantlab.logging_config import get_logger
from quantlab.portfolio.allocator import PortfolioAllocator
from quantlab.portfolio.constraints import constraints_from_config
from quantlab.portfolio.rebalancing import rebalance_and_cap_turnover
from quantlab.portfolio.volatility_targeting import apply_volatility_target
from quantlab.risk.metrics import compute_metrics
from quantlab.strategies.base import BaseStrategy

logger = get_logger(__name__)

#: Numerical dependencies recorded in result metadata for reproducibility.
_TRACKED_PACKAGES = ("quantlab", "pandas", "numpy", "pydantic", "scipy", "statsmodels")


def _git_commit_hash() -> str | None:
    """Return the current Git commit hash, or ``None`` when unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _git_is_dirty() -> bool | None:
    """Return whether tracked files differ from HEAD, or ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        return bool(completed.stdout.strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _dependency_versions() -> dict[str, str]:
    """Installed versions of quantlab and its key numerical dependencies."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


#: Cached source hash and the fingerprint used to validate it.
_source_hash_fingerprint: tuple[tuple[str, int], ...] | None = None
_source_hash_value: str | None = None
_source_hash_computed_at: float | None = None

#: Maximum cache lifetime when file mtimes appear unchanged.
_SOURCE_HASH_TTL_SECONDS = 60.0


#: Excluded from `_source_hash()`: neither the dashboard UI nor the CLI
#: entry point is imported by any notebook cell or by the computational
#: backtest path, so editing them would otherwise force spurious notebook
#: rebuilds and spuriously invalidate walk-forward artifact reuse.
_SOURCE_HASH_EXCLUDED_TOP_LEVEL_PARTS = frozenset({"dashboard"})
_SOURCE_HASH_EXCLUDED_FILES = frozenset({"cli.py"})


def _source_hash() -> str:
    r"""Return a SHA-256 hash of QuantLab's installed Python sources.

    Scoped to the modules that actually affect computed results (excludes
    `dashboard/` and `cli.py`; see `_SOURCE_HASH_EXCLUDED_TOP_LEVEL_PARTS`
    and `_SOURCE_HASH_EXCLUDED_FILES`). POSIX relative paths keep the hash
    platform-independent. A path/mtime fingerprint avoids rereading
    unchanged files, while a short TTL bounds staleness when a
    synchronisation tool preserves mtimes.
    """
    global _source_hash_fingerprint, _source_hash_value, _source_hash_computed_at
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for path in root.rglob("*.py")
        if path.relative_to(root).parts[0] not in _SOURCE_HASH_EXCLUDED_TOP_LEVEL_PARTS
        and path.relative_to(root).name not in _SOURCE_HASH_EXCLUDED_FILES
    )
    fingerprint = tuple(
        (path.relative_to(root).as_posix(), path.stat().st_mtime_ns) for path in paths
    )
    now = time.monotonic()
    ttl_expired = (
        _source_hash_computed_at is None
        or now - _source_hash_computed_at >= _SOURCE_HASH_TTL_SECONDS
    )
    if (
        not ttl_expired
        and fingerprint == _source_hash_fingerprint
        and _source_hash_value is not None
    ):
        return _source_hash_value
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    _source_hash_fingerprint = fingerprint
    _source_hash_computed_at = now
    _source_hash_value = digest.hexdigest()
    return _source_hash_value


class BacktestEngine:
    """Vectorised, look-ahead-safe backtest engine."""

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        allocator: PortfolioAllocator,
        execution_model: ExecutionModel,
        config: ExperimentConfig,
        *,
        execution_delay: int = 0,
    ) -> BacktestResult:
        """Run a backtest and return a :class:`BacktestResult`.

        Args:
            data: Canonical long OHLCV frame (may also carry a benchmark
                symbol outside ``config.symbols``, see below).
            strategy: Signal generator.
            allocator: Turns signals into target portfolio weights.
            execution_model: Cost model (commission/spread/slippage).
            config: Experiment configuration.
            execution_delay: Additional bars between target formation and
                execution. Turnover, costs, trades and returns are recomputed
                from the delayed weights.
        """
        started = time.perf_counter()
        if not isinstance(data, pd.DataFrame):
            raise BacktestError("data must be a pandas DataFrame.")
        missing_columns = [
            column for column in (TIMESTAMP, SYMBOL) if column not in data.columns
        ]
        if missing_columns:
            raise BacktestError(
                "Market data is missing required column(s): "
                f"{missing_columns}. Expected canonical long OHLCV data."
            )
        if data.empty:
            raise BacktestError("Cannot backtest on empty data.")
        if isinstance(execution_delay, bool) or not isinstance(
            execution_delay, Integral
        ):
            raise BacktestError("execution_delay must be a non-negative integer.")
        if execution_delay < 0:
            raise BacktestError("execution_delay must be a non-negative integer.")
        delay = int(execution_delay)

        # Retain benchmark-only rows for comparison but exclude them from trading.
        tradable_symbols = set(config.symbols)
        tradable_data = data[data[SYMBOL].isin(tradable_symbols)].reset_index(drop=True)
        present_symbols = set(tradable_data[SYMBOL].unique())
        missing_symbols = [
            symbol for symbol in config.symbols if symbol not in present_symbols
        ]
        if missing_symbols:
            raise BacktestError(
                "Market data is missing configured tradable symbol(s): "
                f"{missing_symbols}. Refusing to backtest a silently reduced universe."
            )

        prices = price_matrix(tradable_data, adjusted=True)
        if prices.shape[0] < 2:
            raise BacktestError(
                f"Need at least 2 dates to backtest; got {prices.shape[0]}."
            )
        asset_returns = compute_asset_returns(prices)

        # Require the strategy to cover the exact tradable panel.
        signals = strategy.generate_signals(tradable_data)
        signals = strategy._validate_signals(signals, prices)

        # Convert signals to target weights.
        allocated = validate_execution_frame(
            allocator.allocate(signals, tradable_data), name="allocator output"
        )
        if not allocated.index.equals(prices.index) or not allocated.columns.equals(
            prices.columns
        ):
            raise BacktestError(
                "allocator output must have exactly the price-matrix index and "
                "columns; missing or extra dates/symbols must not be filled "
                "silently."
            )
        target_weights = allocated

        # Apply portfolio-level volatility targeting only when the allocator
        # does not already perform that scaling.
        if (
            config.portfolio.target_volatility is not None
            and config.portfolio.allocator != "volatility_targeting"
        ):
            target_weights = apply_volatility_target(
                target_weights,
                asset_returns.reindex(
                    index=target_weights.index, columns=target_weights.columns
                ),
                config.portfolio.target_volatility,
                window=config.portfolio.volatility_window,
                maximum_leverage=config.portfolio.maximum_leverage,
                periods_per_year=config.periods_per_year,
            )

        # Enforce portfolio constraints.
        constraints = constraints_from_config(config.portfolio)
        constrained = constraints.apply(target_weights)

        # Apply the shared stateful rebalancing/turnover pipeline once over
        # the full index so its state remains continuous.
        held_weights = rebalance_and_cap_turnover(constrained, config.portfolio)
        if delay > 0:
            held_weights = held_weights.shift(delay).fillna(0.0)

        # Accounting contains the one-period look-ahead barrier.
        accounting = run_accounting(
            held_weights, asset_returns, execution_model, config.initial_capital
        )

        # Align the benchmark to the simulated portfolio dates.
        benchmark_data = (
            data if config.benchmark_kind is BenchmarkKind.SYMBOL else tradable_data
        )
        benchmark_returns = build_benchmark(
            benchmark_data,
            pd.DatetimeIndex(prices.index),
            benchmark_symbol=config.benchmark_symbol,
            first_asset_symbol=config.symbols[0],
            risk_free_rate=config.backtest.risk_free_rate,
            periods_per_year=config.periods_per_year,
            kind=str(config.benchmark_kind),
        )

        # Reuse accounting's slippage model and cost-sizing equity so per-fill
        # costs match the aggregate costs already charged.
        trades = build_trade_log(
            accounting.executed_weights,
            accounting.weight_changes,
            accounting.equity,
            price_matrix(tradable_data, adjusted=False),
            commission_bps=config.commission_bps,
            spread_bps=config.spread_bps,
            slippage_model=execution_model.slippage,
            slippage_equity=accounting.equity_for_costs,
        )

        # Compute performance, risk and portfolio metrics.
        metrics = compute_metrics(
            accounting.net_returns,
            accounting.equity,
            benchmark_returns=benchmark_returns,
            risk_free_rate=config.backtest.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
        metrics.update(
            portfolio_metrics_from_accounting(accounting, config.periods_per_year)
        )
        metrics["number_of_trades"] = float(len(trades))
        metrics["average_trade_size"] = (
            float(trades["traded_notional"].mean()) if len(trades) else 0.0
        )
        # Fractional cost summed over time; currency costs live in the trade log.
        metrics["total_cost_fraction"] = float(accounting.costs.total.sum())

        calculation_elapsed = time.perf_counter() - started
        logger.info(
            "Backtest '%s' completed in %.3fs (Sharpe=%.2f, CAGR=%.2f%%).",
            config.experiment_name,
            calculation_elapsed,
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("cagr", 0.0) * 100.0,
        )

        # Assemble the immutable run result.
        return BacktestResult(
            config=config,
            equity_curve=accounting.equity,
            returns=accounting.net_returns,
            benchmark_returns=benchmark_returns,
            positions=accounting.executed_weights,
            weights=held_weights,
            target_weights=constrained,
            signals=signals,
            trades=trades,
            costs=accounting.costs.to_frame(),
            metrics=metrics,
            metadata=self._build_metadata(
                config, tradable_data, calculation_elapsed, data
            ),
            gross_returns=accounting.gross_returns,
            gross_equity=accounting.gross_equity,
            turnover=accounting.turnover,
        )

    @staticmethod
    def _build_metadata(
        config: ExperimentConfig,
        tradable_data: pd.DataFrame,
        calculation_elapsed: float,
        full_data: pd.DataFrame,
    ) -> dict[str, object]:
        """Build metadata used to compare and audit runs.

        ``data_hash`` covers the full input, including a separate benchmark,
        while ``n_rows`` counts only tradable rows. Hashes and dependency
        versions detect differences but do not recreate a full environment.
        ``elapsed_seconds`` measures calculation through metrics; metadata
        hashing and result serialisation are excluded.
        """
        return {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "experiment_name": config.experiment_name,
            "strategy": config.strategy_name,
            "symbols": config.symbols,
            "start_date": str(config.start_date),
            "end_date": str(config.end_date),
            "random_seed": config.random_seed,
            "n_rows": len(tradable_data),
            "elapsed_seconds": round(calculation_elapsed, 4),
            "periods_per_year": config.periods_per_year,
            "data_hash": ParquetStorage.hash_frame(full_data),
            "git_commit": _git_commit_hash(),
            "git_dirty": _git_is_dirty(),
            "dependency_versions": _dependency_versions(),
            "code_hash": _source_hash(),
        }
