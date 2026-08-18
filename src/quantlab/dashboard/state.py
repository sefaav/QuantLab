"""Streamlit-independent dashboard configuration and execution helpers."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.data.base import SymbolSuggestion
from quantlab.data.binance import BinanceDataSource
from quantlab.data.loader import DataLoader
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import BacktestError

if TYPE_CHECKING:
    from quantlab.validation.walk_forward import WalkForwardResult

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
                "forward_fill_limit": inputs.get("forward_fill_limit", 1),
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
                "target_minimum_weight": inputs.get("target_minimum_weight"),
                "maximum_gross_exposure": inputs.get("maximum_gross_exposure"),
                "maximum_net_exposure": inputs.get("maximum_net_exposure"),
                "target_maximum_positions": inputs.get("target_maximum_positions"),
                "maximum_turnover": inputs.get("maximum_turnover"),
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
                "slippage_model": inputs.get("slippage_model", "constant"),
                "impact_coefficient": inputs.get("impact_coefficient", 0.1),
            },
            "backtest": {
                "initial_capital": inputs.get("initial_capital", 100_000.0),
                "benchmark_kind": inputs.get("benchmark_kind", "symbol"),
                "benchmark_symbol": inputs.get("benchmark_symbol") or None,
                "risk_free_rate": inputs.get("risk_free_rate", 0.02),
            },
            "validation": _validation_block_from_inputs(inputs),
            "reproducibility": {"random_seed": inputs.get("random_seed", 42)},
        }
    )


def _validation_block_from_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build the ``validation:`` config block for the active dashboard mode."""
    if inputs.get("validation_method") == "walk_forward":
        return {
            "method": "walk_forward",
            "train_window": inputs.get("train_window"),
            "validation_window": inputs.get("validation_window"),
            "test_window": inputs.get("test_window"),
            "expanding": inputs.get("expanding", True),
            "optimization_metric": inputs.get("optimization_metric", "sharpe"),
            "parameter_grid": inputs.get("parameter_grid") or None,
        }
    return {
        "method": "holdout",
        "validation_ratio": inputs.get("validation_ratio"),
        "test_ratio": inputs.get("test_ratio"),
    }


def run_dashboard_backtest(
    config: ExperimentConfig,
) -> tuple[BacktestResult, list[str]]:
    """Load data and run the backtest, returning the result and data warnings."""
    data, report = DataLoader().load(config)
    result = run_backtest_from_config(data, config, data_quality_report=report)
    return result, report.warnings


def run_dashboard_walk_forward(
    config: ExperimentConfig,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[WalkForwardResult, list[str]]:
    """Load data and run walk-forward validation, returning the result and warnings.

    Windows, expanding mode, optimization metric and the parameter grid are
    all read from ``config.validation`` (as assembled by
    :func:`build_config_from_inputs`), matching how ``quantlab walk-forward``
    resolves the same settings from a YAML config.

    Args:
        config: Validated experiment config, with ``validation.method`` set
            to ``"walk_forward"``.
        on_progress: Optional callback invoked as ``on_progress(done, total)``
            once before the first fold and once after each fold completes,
            for the caller to drive a live progress bar.
    """
    from quantlab.validation.parameter_grid import parameter_grid_for_config
    from quantlab.validation.walk_forward import (
        WalkForwardValidator,
        resolve_walk_forward_windows,
    )

    data, report = DataLoader().load(config)
    validator = WalkForwardValidator(config)
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    result = validator.run(
        data,
        parameter_grid=parameter_grid_for_config(config),
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
        on_progress=on_progress,
    )
    return result, report.warnings


def estimate_walk_forward_backtest_count(
    *,
    start_date: date,
    end_date: date,
    is_247_market: bool,
    train_window: int,
    validation_window: int,
    test_window: int,
    expanding: bool,
    parameter_grid: dict[str, list[Any]],
) -> int:
    """Estimate how many single backtests a walk-forward run will execute.

    Approximates the number of bars from the requested date range (business
    days for an XNYS-like calendar, calendar days for a 24/7 market) since
    the actual data is not loaded yet at this point in the sidebar — this is
    an estimate to warn about a slow configuration, not an exact count.
    """
    from quantlab.validation.splits import walk_forward_windows

    if is_247_market:
        approximate_bars = max(0, (end_date - start_date).days) + 1
        index = pd.date_range(start_date, periods=approximate_bars, freq="D")
    else:
        index = pd.bdate_range(start_date, end_date)
    if len(index) == 0:
        return 0
    windows = walk_forward_windows(
        pd.DatetimeIndex(index),
        train_window,
        validation_window,
        test_window,
        expanding=expanding,
    )
    n_combinations = 1
    for values in parameter_grid.values():
        n_combinations *= max(1, len(values))
    return len(windows) * (n_combinations + 1)


@dataclass
class ProgressPacer:
    """Tracks a seconds-per-unit pace across progress ticks, asymmetrically.

    An exponential moving average (nudged, not replaced, by each
    observation — the approach `tqdm` uses), smoothed asymmetrically:
    ``rising_smoothing`` (0.4) partially adopts a tick implying a slower
    pace than currently tracked, ``falling_smoothing`` (0.2) partially
    adopts one implying a faster pace. Weighting a slowdown more than a
    speedup catches up to a genuine sustained slowdown (e.g. an expanding
    walk-forward's later, bigger-training-window folds) faster than a
    symmetric average would; keeping both partial rather than full (no
    single tick fully overrides the tracked rate) avoids one noisy tick
    (parameter-grid candidates genuinely cost different amounts) swinging
    the estimate on its own.
    """

    rising_smoothing: float = 0.4
    falling_smoothing: float = 0.2
    _rate_seconds_per_unit: float | None = field(default=None, init=False)
    _last_done: int = field(default=0, init=False)
    _last_elapsed: float = field(default=0.0, init=False)

    def update(self, done: int, elapsed: float) -> None:
        """Record a new ``(done, elapsed)`` observation."""
        delta_done = done - self._last_done
        delta_elapsed = elapsed - self._last_elapsed
        if delta_done > 0 and delta_elapsed > 0:
            instantaneous = delta_elapsed / delta_done
            if self._rate_seconds_per_unit is None:
                self._rate_seconds_per_unit = instantaneous
            else:
                smoothing = (
                    self.rising_smoothing
                    if instantaneous >= self._rate_seconds_per_unit
                    else self.falling_smoothing
                )
                self._rate_seconds_per_unit = (
                    smoothing * instantaneous
                    + (1 - smoothing) * self._rate_seconds_per_unit
                )
        self._last_done = done
        self._last_elapsed = elapsed

    def remaining(self, done: int, total: int) -> float | None:
        """Return estimated seconds left, or ``None`` before any pace is known."""
        if self._rate_seconds_per_unit is None:
            return None
        return self._rate_seconds_per_unit * max(0, total - done)


def run_dashboard_stress_tests(
    config: ExperimentConfig,
    expected_data_hash: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
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
    return run_stress_tests(data, config, on_progress=on_progress)


def run_dashboard_walk_forward_stress_tests(
    config: ExperimentConfig,
    wf_baseline: WalkForwardResult,
    expected_data_hash: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Walk-forward-aware stress scenarios, staleness-checked like the above.

    Re-runs the whole walk-forward process per scenario
    (:func:`~quantlab.validation.robustness.run_walk_forward_stress_tests`)
    instead of :func:`run_dashboard_stress_tests`'s plain-backtest variant —
    Walk-forward mode's Robustness tab must never silently show numbers from
    a different validation method than the one currently in effect.
    """
    from quantlab.validation.robustness import run_walk_forward_stress_tests

    data, _ = DataLoader().load(config)
    actual_data_hash = ParquetStorage.hash_frame(data)
    if actual_data_hash != expected_data_hash:
        raise BacktestError(
            "Market data changed since the displayed walk-forward run. Run "
            "walk-forward again before running stress tests."
        )
    return run_walk_forward_stress_tests(
        data, config, wf_baseline, on_progress=on_progress
    )


def run_dashboard_bootstrap(
    config: ExperimentConfig,
    returns: pd.Series,
    *,
    n_iterations: int,
    block_size: int,
) -> pd.DataFrame:
    """Block-bootstrap the given returns.

    Mode-agnostic: bootstrap resamples already-realised returns and
    optimizes nothing, so the same function applies whether ``returns`` is
    ``result.returns`` (Backtest mode) or ``wf.oos_result.returns``
    (Walk-forward mode) — no data reload or staleness check is needed since
    it never touches market data, only the already-displayed series.
    """
    from quantlab.validation.bootstrap import bootstrap_returns

    return bootstrap_returns(
        returns,
        n_iterations=n_iterations,
        block_size=block_size,
        seed=config.random_seed,
        periods_per_year=config.periods_per_year,
        initial_capital=config.initial_capital,
        risk_free_rate=config.risk_free_rate,
    ).summary()


def run_dashboard_permutation_test(
    config: ExperimentConfig,
    returns: pd.Series,
    *,
    n_iterations: int,
) -> dict[str, float]:
    """Random-sign Monte Carlo permutation test (mode-agnostic, see bootstrap)."""
    from quantlab.validation.robustness import monte_carlo_permutation

    return monte_carlo_permutation(
        returns,
        n_iterations=n_iterations,
        seed=config.random_seed,
        periods_per_year=config.periods_per_year,
        risk_free_rate=config.risk_free_rate,
    )


def run_dashboard_sensitivity(
    config: ExperimentConfig,
    expected_data_hash: str,
    parameter_x: str,
    values_x: list[Any],
    parameter_y: str,
    values_y: list[Any],
) -> pd.DataFrame:
    """Run a plain-backtest parameter-sensitivity sweep.

    Staleness-checked like :func:`run_dashboard_stress_tests`.
    """
    from quantlab.validation.parameter_sensitivity import run_parameter_sensitivity

    data, _ = DataLoader().load(config)
    actual_data_hash = ParquetStorage.hash_frame(data)
    if actual_data_hash != expected_data_hash:
        raise BacktestError(
            "Market data changed since the displayed backtest. Run the "
            "backtest again before running parameter sensitivity."
        )
    return run_parameter_sensitivity(
        data, config, parameter_x, values_x, parameter_y, values_y
    )


def run_dashboard_walk_forward_sensitivity(
    config: ExperimentConfig,
    expected_data_hash: str,
    parameter_x: str,
    values_x: list[Any],
    parameter_y: str,
    values_y: list[Any],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Walk-forward-aware parameter-sensitivity sweep, staleness-checked.

    Each grid cell re-runs the whole walk-forward process
    (:func:`~quantlab.validation.parameter_sensitivity.
    run_walk_forward_parameter_sensitivity`) instead of the plain
    single-backtest variant, for the same reason as stress tests above.
    """
    from quantlab.validation.parameter_sensitivity import (
        run_walk_forward_parameter_sensitivity,
    )

    data, _ = DataLoader().load(config)
    actual_data_hash = ParquetStorage.hash_frame(data)
    if actual_data_hash != expected_data_hash:
        raise BacktestError(
            "Market data changed since the displayed walk-forward run. Run "
            "walk-forward again before running parameter sensitivity."
        )
    return run_walk_forward_parameter_sensitivity(
        data,
        config,
        parameter_x,
        values_x,
        parameter_y,
        values_y,
        on_progress=on_progress,
    )


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
