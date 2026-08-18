"""Stress scenarios and a random-sign test for realised returns."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.constants import SYMBOL
from quantlab.exceptions import QuantLabError
from quantlab.logging_config import get_logger
from quantlab.risk import metrics as M
from quantlab.risk._validation import (
    finite_real,
    nonnegative_int,
    numeric_series,
    positive_int,
)
from quantlab.risk.drawdown import max_drawdown
from quantlab.risk.stress import remove_best_days, scale_costs

if TYPE_CHECKING:
    from quantlab.validation.walk_forward import WalkForwardResult

logger = get_logger(__name__)

_STRESS_COLUMNS = [
    "scenario",
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "status",
    "error",
]


def _metrics_row(
    name: str, returns: pd.Series, ppy: int, risk_free_rate: float = 0.0
) -> dict[str, object]:
    """Build one successful stress-scenario row."""
    equity = M.equity_from_returns(returns)
    return {
        "scenario": name,
        "total_return": M.total_return(equity),
        "cagr": M.cagr(equity, ppy),
        "sharpe": M.sharpe_ratio(returns, risk_free_rate, ppy),
        "max_drawdown": max_drawdown(equity),
        "status": "ok",
        "error": None,
    }


def _failed_row(name: str, error: QuantLabError) -> dict[str, object]:
    """Keep an expected scenario failure visible in the result table."""
    return {
        "scenario": name,
        "total_return": float("nan"),
        "cagr": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "status": "failed",
        "error": str(error),
    }


def run_stress_tests(
    data: pd.DataFrame,
    config: ExperimentConfig,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Re-run the experiment under cost, delay and universe perturbations.

    Args:
        data: Canonical long OHLCV frame.
        config: Experiment configuration.
        on_progress: Optional callback invoked as ``on_progress(done, total)``
            once before the first scenario and once after each of the
            (baseline plus) up to 6 scenarios below completes — each is a
            single backtest, so this is coarser than a fold-level signal but
            still enough to show a stalled run is actually progressing.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    if SYMBOL not in data.columns:
        raise ValueError(f"data must contain a {SYMBOL!r} column.")
    periods_per_year = positive_int(config.periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(config.risk_free_rate, name="risk_free_rate")

    reduced_universe = len(config.symbols) > 2
    total_scenarios = 1 + 3 + 2 + (1 if reduced_universe else 0)
    completed_scenarios = 0
    if on_progress is not None:
        on_progress(0, total_scenarios)

    baseline = run_backtest_from_config(data, config)
    rows = [
        _metrics_row("baseline", baseline.returns, periods_per_year, risk_free_rate)
    ]
    completed_scenarios += 1
    if on_progress is not None:
        on_progress(completed_scenarios, total_scenarios)

    scenarios = {
        "commission x2": scale_costs(config, commission_mult=2.0),
        "commission x5": scale_costs(config, commission_mult=5.0),
        "slippage x2": scale_costs(config, slippage_mult=2.0),
    }
    for name, scenario_config in scenarios.items():
        result = run_backtest_from_config(data, scenario_config)
        rows.append(
            _metrics_row(name, result.returns, periods_per_year, risk_free_rate)
        )
        completed_scenarios += 1
        if on_progress is not None:
            on_progress(completed_scenarios, total_scenarios)

    delayed = run_backtest_from_config(data, config, execution_delay=1)
    rows.append(
        _metrics_row(
            "execution delay +1",
            delayed.returns,
            periods_per_year,
            risk_free_rate,
        )
    )
    completed_scenarios += 1
    if on_progress is not None:
        on_progress(completed_scenarios, total_scenarios)

    rows.append(
        _metrics_row(
            "best 10 days removed",
            remove_best_days(baseline.returns, 10),
            periods_per_year,
            risk_free_rate,
        )
    )
    completed_scenarios += 1
    if on_progress is not None:
        on_progress(completed_scenarios, total_scenarios)

    if reduced_universe:
        reduced_symbols = config.symbols[:-1]
        data_config = config.data.revalidated_copy(update={"symbols": reduced_symbols})
        reduced_config = config.revalidated_copy(update={"data": data_config})
        required_symbols = set(reduced_symbols)
        if config.benchmark_symbol is not None:
            required_symbols.add(config.benchmark_symbol)
        subset = data[data[SYMBOL].isin(required_symbols)].reset_index(drop=True)
        try:
            result = run_backtest_from_config(subset, reduced_config)
        except QuantLabError as exc:
            logger.warning("Reduced-universe scenario failed: %s", exc)
            rows.append(_failed_row("reduced universe", exc))
        else:
            rows.append(
                _metrics_row(
                    "reduced universe",
                    result.returns,
                    periods_per_year,
                    risk_free_rate,
                )
            )
        completed_scenarios += 1
        if on_progress is not None:
            on_progress(completed_scenarios, total_scenarios)

    return pd.DataFrame(rows, columns=_STRESS_COLUMNS)


def run_walk_forward_stress_tests(
    data: pd.DataFrame,
    config: ExperimentConfig,
    wf_baseline: WalkForwardResult,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Re-run the whole walk-forward process under cost/delay/universe stress.

    Unlike :func:`run_stress_tests`, every scenario except "best 10 days
    removed" re-evaluates parameter selection under the scenario's
    perturbation instead of a single plain backtest — so a Walk-forward
    mode's Robustness numbers never silently come from a different
    validation method than the one currently in effect.

    The three cost-only scenarios ("commission x2", "commission x5",
    "slippage x2") never change signals or portfolio allocation — only the
    accounting step depends on execution costs — so they share a single
    :class:`~quantlab.validation.walk_forward.WalkForwardWeightCache` built
    once from the baseline config, cheaply re-scoring each fold's cached
    candidates under the new costs via
    :meth:`~quantlab.validation.walk_forward.WalkForwardValidator.
    rescore_with_costs` instead of re-running signal generation and
    allocation three more times. "Execution delay +1" and "reduced
    universe" genuinely change the weights themselves (the delay shift and
    the tradable universe respectively), so they still re-execute
    :class:`~quantlab.validation.walk_forward.WalkForwardValidator` end to
    end. "Best 10 days removed" stays a post-hoc transform of the
    already-realised baseline OOS returns: it changes no configuration, so
    re-running walk-forward would only waste time reproducing the exact
    same selection.

    Args:
        data: Canonical long OHLCV frame.
        config: The baseline experiment config (``validation.method`` must
            be ``"walk_forward"``).
        wf_baseline: The already-computed baseline ``WalkForwardResult``,
            reused for "baseline" and "best 10 days removed".
        on_progress: Optional callback invoked as ``on_progress(done, total)``
            once before any work starts and once after each unit of work
            completes: one tick per candidate (folds x grid size) while the
            weight cache used by the three cost-only scenarios is
            (re)built — the bulk of the total cost — then one tick per
            remaining scenario ("execution delay +1" and "reduced universe"
            are each a full walk-forward run; the three cost-only rescores
            and "best 10 days removed" are
            cheap).
    """
    from quantlab.validation.parameter_grid import parameter_grid_for_config
    from quantlab.validation.walk_forward import (
        WalkForwardValidator,
        resolve_walk_forward_windows,
    )

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    if SYMBOL not in data.columns:
        raise ValueError(f"data must contain a {SYMBOL!r} column.")
    if wf_baseline.oos_result is None:
        raise ValueError(
            "wf_baseline has no OOS result — no fold fit its windows, so "
            "there is nothing to stress-test against."
        )
    periods_per_year = positive_int(config.periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(config.risk_free_rate, name="risk_free_rate")
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    expanding = config.validation.expanding
    grid = parameter_grid_for_config(config)

    def _run_walk_forward(
        scenario_config: ExperimentConfig,
        scenario_data: pd.DataFrame,
        scenario_grid: dict[str, list[object]],
        *,
        execution_delay: int = 0,
    ) -> WalkForwardResult:
        return WalkForwardValidator(scenario_config).run(
            scenario_data,
            parameter_grid=scenario_grid,
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            expanding=expanding,
            execution_delay=execution_delay,
        )

    rows = [
        _metrics_row(
            "baseline", wf_baseline.oos_result.returns, periods_per_year, risk_free_rate
        )
    ]

    cost_scenarios = {
        "commission x2": scale_costs(config, commission_mult=2.0),
        "commission x5": scale_costs(config, commission_mult=5.0),
        "slippage x2": scale_costs(config, slippage_mult=2.0),
    }
    reduced_universe = len(config.symbols) > 2
    n_baseline_folds = len(wf_baseline.folds)
    n_combinations = math.prod(len(values) for values in grid.values()) if grid else 1
    n_cache_units = n_baseline_folds * n_combinations
    total_units = (
        n_cache_units + len(cost_scenarios) + 2 + (1 if reduced_universe else 0)
    )

    def _cache_progress(done: int, _total: int) -> None:
        if on_progress is not None:
            on_progress(done, total_units)

    # Signals and portfolio allocation never depend on execution costs, so
    # build the per-candidate weight cache once (this is where the
    # candidate-level progress reported above comes from — finer-grained
    # than one tick per fold, since this build does roughly twice the work
    # of a plain walk-forward fold) and reuse it for every cost-only
    # scenario below instead of paying for a full re-run each.
    validator = WalkForwardValidator(config)
    _, weight_cache = validator.run_with_weight_cache(
        data,
        parameter_grid=grid,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=expanding,
        execution_delay=0,
        on_progress=_cache_progress,
    )
    completed_units = n_cache_units

    for name, scenario_config in cost_scenarios.items():
        wf = validator.rescore_with_costs(weight_cache, scenario_config)
        assert wf.oos_result is not None  # same data/windows as the baseline
        rows.append(
            _metrics_row(name, wf.oos_result.returns, periods_per_year, risk_free_rate)
        )
        completed_units += 1
        if on_progress is not None:
            on_progress(completed_units, total_units)

    delayed = _run_walk_forward(config, data, grid, execution_delay=1)
    assert delayed.oos_result is not None
    rows.append(
        _metrics_row(
            "execution delay +1",
            delayed.oos_result.returns,
            periods_per_year,
            risk_free_rate,
        )
    )
    completed_units += 1
    if on_progress is not None:
        on_progress(completed_units, total_units)

    rows.append(
        _metrics_row(
            "best 10 days removed",
            remove_best_days(wf_baseline.oos_result.returns, 10),
            periods_per_year,
            risk_free_rate,
        )
    )
    completed_units += 1
    if on_progress is not None:
        on_progress(completed_units, total_units)

    if reduced_universe:
        reduced_symbols = config.symbols[:-1]
        data_config = config.data.revalidated_copy(update={"symbols": reduced_symbols})
        reduced_config = config.revalidated_copy(update={"data": data_config})
        required_symbols = set(reduced_symbols)
        if config.benchmark_symbol is not None:
            required_symbols.add(config.benchmark_symbol)
        subset = data[data[SYMBOL].isin(required_symbols)].reset_index(drop=True)
        try:
            reduced_grid = parameter_grid_for_config(reduced_config)
            wf_reduced = _run_walk_forward(reduced_config, subset, reduced_grid)
            if wf_reduced.oos_result is None:
                raise QuantLabError(
                    "No walk-forward fold fit the reduced universe's history."
                )
        except QuantLabError as exc:
            logger.warning("Reduced-universe walk-forward scenario failed: %s", exc)
            rows.append(_failed_row("reduced universe", exc))
        else:
            rows.append(
                _metrics_row(
                    "reduced universe",
                    wf_reduced.oos_result.returns,
                    periods_per_year,
                    risk_free_rate,
                )
            )
        completed_units += 1
        if on_progress is not None:
            on_progress(completed_units, total_units)

    return pd.DataFrame(rows, columns=_STRESS_COLUMNS)


def monte_carlo_permutation(
    returns: pd.Series,
    *,
    n_iterations: int = 1000,
    seed: int = 42,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compare realised Sharpe with random sign flips of excess returns.

    The test preserves return magnitudes and randomises direction around the
    per-period risk-free return. Its empirical p-value is evidence against
    this specific random-sign null, not a probability of future profitability.
    """
    n_iterations = positive_int(n_iterations, name="n_iterations")
    seed = nonnegative_int(seed, name="seed")
    periods_per_year = positive_int(periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(risk_free_rate, name="risk_free_rate")
    validated = numeric_series(
        returns,
        name="returns",
        allow_nan=True,
        require_unique_index=True,
        require_sorted_index=True,
    )
    clean = validated.dropna().to_numpy(dtype=float)
    if (clean < -1.0).any():
        raise ValueError("returns must not contain values below -1.0.")
    if len(clean) < 2:
        return {
            "real_sharpe": 0.0,
            "p_value": 1.0,
            "n_iterations": float(n_iterations),
        }

    rng = np.random.default_rng(seed)
    risk_free_per_period = risk_free_rate / periods_per_year
    excess = clean - risk_free_per_period
    real = M.sharpe_ratio(pd.Series(clean), risk_free_rate, periods_per_year)
    count = 0
    for _ in range(n_iterations):
        signs = rng.choice([-1.0, 1.0], size=len(clean))
        random_returns = risk_free_per_period + excess * signs
        random_sharpe = M.sharpe_ratio(
            pd.Series(random_returns), risk_free_rate, periods_per_year
        )
        if random_sharpe >= real:
            count += 1
    return {
        "real_sharpe": float(real),
        "p_value": float((count + 1) / (n_iterations + 1)),
        "n_iterations": float(n_iterations),
    }
