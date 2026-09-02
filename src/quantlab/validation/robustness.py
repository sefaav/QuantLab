"""Stress scenarios and a random-sign test for realised returns."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.constants import SYMBOL, TIMESTAMP
from quantlab.exceptions import InvalidConfigurationError, QuantLabError
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
from quantlab.validation.checkpoint import (
    clear_checkpoint,
    compute_provenance,
    load_checkpoint,
    save_checkpoint,
)

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


def _ensure_unique_scenario_names(names: list[str]) -> None:
    """Reject scenario names that collide once formatted for display.

    Multipliers close enough together (e.g. 1.0000001 and 1.0000002) both
    format to "x1" under ``:g`` -- distinct configured values must not
    silently collapse onto the same row identity.
    """
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise InvalidConfigurationError(
                f"Two configured stress-test scenarios both format to the "
                f"name {name!r} -- pick magnitudes that remain visibly "
                "distinct once rounded for display."
            )
        seen.add(name)


_STRESS_METRIC_COLUMNS = ("total_return", "cagr", "sharpe", "max_drawdown")


def _stress_row_is_consistent(row: dict[str, object]) -> bool:
    """Return whether a checkpointed row's status, metrics and error agree.

    Mirrors the two shapes ``_metrics_row``/``_failed_row`` actually
    produce: ``status == "ok"`` means every metric is a finite number and
    ``error`` is ``None``; ``status == "failed"`` means every metric is NaN
    and ``error`` is a real (non-empty) message. A checkpoint claiming
    success while carrying NaN metrics, or failure while carrying finite
    ones and no error, is corrupted regardless of whether its schema and
    scenario name already checked out.
    """
    status = row.get("status")
    error = row.get("error")
    metric_values = [row.get(name) for name in _STRESS_METRIC_COLUMNS]
    if status == "ok":
        if error is not None:
            return False
        return all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in metric_values
        )
    if status == "failed":
        if not (isinstance(error, str) and error):
            return False
        return all(
            isinstance(value, float) and math.isnan(value) for value in metric_values
        )
    return False


def _baseline_returns_is_valid(
    baseline: object,
    data: pd.DataFrame,
    baseline_row: dict[str, object] | None,
    periods_per_year: int,
    risk_free_rate: float,
) -> bool:
    """Return whether a checkpointed ``baseline_returns`` is trustworthy.

    A bare ``isinstance(baseline, pd.Series)`` check would accept an empty
    series, one full of NaN/inf, one indexed in 1900, or one that
    contradicts the "baseline" row already saved alongside it -- and
    "best 10 days removed" (see ``remove_best_days`` below) operates on
    this Series directly, not on the metrics row, so a corrupted series
    reaches a real computation even when the row itself looks fine. Checked
    here: non-empty, genuinely numeric, entirely finite; a strictly
    increasing, duplicate-free datetime index; that index falling within
    ``data``'s own date coverage (a stray 1900 date can't belong to this
    run); and, once a "baseline" row exists to compare against, that
    recomputing its metrics from this exact series reproduces it exactly --
    the row and the series must describe the same backtest, not two
    independently-forged values that happen to both look plausible alone.
    Wrapped in a blanket try/except since every input here is untrusted
    checkpoint content, by definition capable of raising in ways no
    isinstance check anticipates (e.g. an index type that can't convert to
    datetimes at all).
    """
    try:
        if not isinstance(baseline, pd.Series) or baseline.empty:
            return False
        if not pd.api.types.is_numeric_dtype(baseline.dtype):
            return False
        values = baseline.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return False
        index = pd.DatetimeIndex(baseline.index)
        if not index.is_monotonic_increasing or index.has_duplicates:
            return False
        data_timestamps = pd.to_datetime(data[TIMESTAMP])
        if index.min() < data_timestamps.min() or index.max() > data_timestamps.max():
            return False
        if baseline_row is not None:
            recomputed = _metrics_row(
                "baseline", baseline, periods_per_year, risk_free_rate
            )
            if any(
                recomputed[key] != baseline_row[key] for key in _STRESS_METRIC_COLUMNS
            ):
                return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class _CostScenario:
    """One commission- or slippage-multiplier stress scenario."""

    name: str
    kind: str  # "commission" | "slippage"
    multiplier: float


def _cost_scenarios(config: ExperimentConfig) -> list[_CostScenario]:
    """Return every configured commission/slippage scenario, in fixed order.

    Commission scenarios first (in the configured order), then slippage.
    """
    settings = config.robustness.stress_test
    return [
        _CostScenario(f"commission x{multiplier:g}", "commission", multiplier)
        for multiplier in settings.commission_multipliers
    ] + [
        _CostScenario(f"slippage x{multiplier:g}", "slippage", multiplier)
        for multiplier in settings.slippage_multipliers
    ]


def _cost_scenario_config(
    config: ExperimentConfig, scenario: _CostScenario
) -> ExperimentConfig:
    """Return ``config`` with exactly one cost component scaled."""
    if scenario.kind == "commission":
        return scale_costs(config, commission_mult=scenario.multiplier)
    return scale_costs(config, slippage_mult=scenario.multiplier)


def _execution_delay_scenarios(config: ExperimentConfig) -> list[tuple[str, int]]:
    """Return ``(name, delay)`` for every configured execution-delay scenario."""
    return [
        (f"execution delay +{delay}", delay)
        for delay in config.robustness.stress_test.execution_delays
    ]


def _best_days_removed_scenarios(config: ExperimentConfig) -> list[tuple[str, int]]:
    """Return ``(name, n)`` for every configured best-days-removed scenario."""
    return [
        (f"best {n} days removed", n)
        for n in config.robustness.stress_test.best_days_removed
    ]


def _reduced_universe_scenarios(config: ExperimentConfig) -> list[tuple[str, int]]:
    """Return ``(name, count)`` for every configured reduced-universe scenario.

    Every configured ``count`` gets a row: one whose universe is too small
    to leave at least 2 tradable symbols is recorded with status="failed"
    at run time (see ``_universe_reduction_is_feasible``), never silently
    omitted from the table.
    """
    return [
        (f"reduced universe (-{count})", count)
        for count in config.robustness.stress_test.reduce_universe_by
    ]


def _universe_reduction_is_feasible(config: ExperimentConfig, count: int) -> bool:
    """Return whether dropping ``count`` symbols leaves >=2 tradable ones."""
    return len(config.symbols) > count + 1


def run_stress_tests(
    data: pd.DataFrame,
    config: ExperimentConfig,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Re-run the experiment under cost, delay and universe perturbations.

    Every scenario's magnitude comes from ``config.robustness.stress_test``
    (commission/slippage multipliers, execution delays, days-removed
    counts, universe-reduction counts) -- each a list, so more than one
    magnitude can be evaluated per scenario type; an empty list disables
    that scenario type entirely. See :class:`~quantlab.config.
    StressTestSettings`.

    Args:
        data: Canonical long OHLCV frame.
        config: Experiment configuration.
        on_progress: Optional callback invoked as ``on_progress(done, total)``
            once before the first scenario and once after each scenario
            (baseline plus every configured cost/delay/best-days/universe
            scenario) completes — each is a single backtest, so this is
            coarser than a fold-level signal but still enough to show a
            stalled run is actually progressing.
        checkpoint_path: Optional path to persist per-scenario progress to,
            so an interrupted run resumes from its last completed scenario
            instead of starting over. See ``quantlab.validation.checkpoint``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    if SYMBOL not in data.columns:
        raise ValueError(f"data must contain a {SYMBOL!r} column.")
    periods_per_year = positive_int(config.periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(config.risk_free_rate, name="risk_free_rate")

    cost_scenarios = _cost_scenarios(config)
    delay_scenarios = _execution_delay_scenarios(config)
    best_days_scenarios = _best_days_removed_scenarios(config)
    universe_scenarios = _reduced_universe_scenarios(config)
    total_scenarios = (
        1
        + len(cost_scenarios)
        + len(delay_scenarios)
        + len(best_days_scenarios)
        + len(universe_scenarios)
    )

    rows: list[dict[str, object]] = []
    # "best N days removed" needs the actual baseline returns Series later,
    # not just its already-computed metrics row, so it has to be part of the
    # checkpointed state too, or resuming past "baseline" would lose it.
    baseline_returns: pd.Series | None = None
    provenance: dict[str, Any] | None = None
    completed_scenarios = 0

    # The exact scenario name each row must have, in order -- fixed by this
    # function's own scenario sequence below (built from the same
    # configured lists, so it can never drift from what actually runs),
    # independent of `progress` (which only ever names a prefix of it).
    _scenario_names_in_order = (
        ["baseline"]
        + [scenario.name for scenario in cost_scenarios]
        + [name for name, _ in delay_scenarios]
        + [name for name, _ in best_days_scenarios]
        + [name for name, _ in universe_scenarios]
    )
    _ensure_unique_scenario_names(_scenario_names_in_order)

    def _validate_stress_state(state: Any, progress: int) -> bool:
        # One row per scenario, in lockstep with `progress` -- so an exact
        # length match (not just an upper bound) is meaningful here, and so
        # is each row's schema, scenario name/order, and internal
        # status/metrics/error consistency: a structurally-plausible but
        # incoherent checkpoint (wrong row count, malformed rows, rows all
        # named "baseline", or a row claiming success while carrying NaN
        # metrics) must never be resumed from -- it would silently skip or
        # duplicate real scenarios, corrupt the table, or crash "best 10
        # days removed", which needs baseline_returns specifically, not just
        # its metrics row.
        if not (0 <= progress <= total_scenarios and isinstance(state, tuple)):
            return False
        if len(state) != 2:
            return False
        rows, baseline = state
        if not isinstance(rows, list) or len(rows) != progress:
            return False
        if not all(
            isinstance(row, dict) and row.keys() == set(_STRESS_COLUMNS) for row in rows
        ):
            return False
        expected_names = _scenario_names_in_order[:progress]
        if [row["scenario"] for row in rows] != expected_names:
            return False
        if not all(_stress_row_is_consistent(row) for row in rows):
            return False
        if progress >= 1:
            return _baseline_returns_is_valid(
                baseline, data, rows[0], periods_per_year, risk_free_rate
            )
        return baseline is None

    if checkpoint_path is not None:
        provenance = compute_provenance(config, data)
        checkpoint_result = load_checkpoint(
            checkpoint_path, provenance, validate=_validate_stress_state
        )
        if checkpoint_result is not None:
            (rows, baseline_returns), completed_scenarios = checkpoint_result
            logger.info(
                "Resuming stress tests from checkpoint: %d/%d scenarios already done.",
                completed_scenarios,
                total_scenarios,
            )

    def _checkpoint() -> None:
        if checkpoint_path is not None and provenance is not None:
            save_checkpoint(
                checkpoint_path,
                provenance,
                (rows, baseline_returns),
                completed_scenarios,
            )

    if on_progress is not None:
        on_progress(completed_scenarios, total_scenarios)

    if completed_scenarios < 1:
        baseline = run_backtest_from_config(data, config)
        baseline_returns = baseline.returns
        rows.append(
            _metrics_row("baseline", baseline_returns, periods_per_year, risk_free_rate)
        )
        completed_scenarios += 1
        _checkpoint()
        if on_progress is not None:
            on_progress(completed_scenarios, total_scenarios)

    position = 2
    for scenario in cost_scenarios:
        if completed_scenarios < position:
            try:
                scenario_config = _cost_scenario_config(config, scenario)
                result = run_backtest_from_config(data, scenario_config)
            except QuantLabError as exc:
                logger.warning("%s scenario failed: %s", scenario.name, exc)
                rows.append(_failed_row(scenario.name, exc))
            else:
                rows.append(
                    _metrics_row(
                        scenario.name, result.returns, periods_per_year, risk_free_rate
                    )
                )
            completed_scenarios += 1
            _checkpoint()
            if on_progress is not None:
                on_progress(completed_scenarios, total_scenarios)
        position += 1

    for name, delay in delay_scenarios:
        if completed_scenarios < position:
            try:
                delayed = run_backtest_from_config(data, config, execution_delay=delay)
            except QuantLabError as exc:
                logger.warning("%s scenario failed: %s", name, exc)
                rows.append(_failed_row(name, exc))
            else:
                rows.append(
                    _metrics_row(
                        name, delayed.returns, periods_per_year, risk_free_rate
                    )
                )
            completed_scenarios += 1
            _checkpoint()
            if on_progress is not None:
                on_progress(completed_scenarios, total_scenarios)
        position += 1

    for name, n in best_days_scenarios:
        if completed_scenarios < position:
            # completed_scenarios >= 1 guarantees baseline_returns is set.
            assert baseline_returns is not None
            try:
                scenario_returns = remove_best_days(baseline_returns, n)
            except QuantLabError as exc:
                logger.warning("%s scenario failed: %s", name, exc)
                rows.append(_failed_row(name, exc))
            else:
                rows.append(
                    _metrics_row(
                        name, scenario_returns, periods_per_year, risk_free_rate
                    )
                )
            completed_scenarios += 1
            _checkpoint()
            if on_progress is not None:
                on_progress(completed_scenarios, total_scenarios)
        position += 1

    for name, count in universe_scenarios:
        if completed_scenarios < position:
            if not _universe_reduction_is_feasible(config, count):
                rows.append(
                    _failed_row(
                        name,
                        QuantLabError(
                            f"Universe has only {len(config.symbols)} symbols; "
                            f"removing {count} would leave fewer than 2 tradable."
                        ),
                    )
                )
            else:
                try:
                    reduced_symbols = config.symbols[:-count]
                    reduced_instruments = [
                        instrument
                        for instrument in config.data.instruments
                        if instrument.symbol in reduced_symbols
                    ]
                    data_config = config.data.revalidated_copy(
                        update={"instruments": reduced_instruments}
                    )
                    reduced_config = config.revalidated_copy(
                        update={"data": data_config}
                    )
                    required_symbols = set(reduced_symbols)
                    if config.benchmark_symbol is not None:
                        required_symbols.add(config.benchmark_symbol)
                    subset = data[data[SYMBOL].isin(required_symbols)].reset_index(
                        drop=True
                    )
                    result = run_backtest_from_config(subset, reduced_config)
                except QuantLabError as exc:
                    logger.warning("%s scenario failed: %s", name, exc)
                    rows.append(_failed_row(name, exc))
                else:
                    rows.append(
                        _metrics_row(
                            name, result.returns, periods_per_year, risk_free_rate
                        )
                    )
            completed_scenarios += 1
            _checkpoint()
            if on_progress is not None:
                on_progress(completed_scenarios, total_scenarios)
        position += 1

    if checkpoint_path is not None:
        clear_checkpoint(checkpoint_path)
    return pd.DataFrame(rows, columns=_STRESS_COLUMNS)


def stress_test_checkpoint_paths(checkpoint_path: Path) -> tuple[Path, ...]:
    """Return every on-disk checkpoint file a stress-test run can create.

    ``run_walk_forward_stress_tests`` writes a second, nested file for its
    weight-cache build (see its own docstring) alongside the main
    scenario-block checkpoint at ``checkpoint_path`` -- a caller discarding
    "all saved progress" (the CLI's ``--fresh``) must clear every one of
    these, not just the main file, or the nested cache can silently survive
    and be reused on the next run. The single source of truth for this
    derivation, reused by :func:`run_walk_forward_stress_tests` itself so
    the two can never drift apart.
    """
    return (
        checkpoint_path,
        checkpoint_path.with_name(checkpoint_path.stem + "_cache.pkl"),
    )


def run_walk_forward_stress_tests(
    data: pd.DataFrame,
    config: ExperimentConfig,
    wf_baseline: WalkForwardResult,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Re-run the whole walk-forward process under cost/delay/universe stress.

    Every scenario's magnitude comes from ``config.robustness.stress_test``
    (see :func:`run_stress_tests` and :class:`~quantlab.config.
    StressTestSettings`) -- each a list, so more than one magnitude can be
    evaluated per scenario type; an empty list disables that scenario type
    entirely.

    Unlike :func:`run_stress_tests`, every scenario except "best N days
    removed" re-evaluates parameter selection under the scenario's
    perturbation instead of a single plain backtest — so a Walk-forward
    mode's Robustness numbers never silently come from a different
    validation method than the one currently in effect.

    Every commission/slippage scenario never changes signals or portfolio
    allocation — only the accounting step depends on execution costs — so
    they all share a single :class:`~quantlab.validation.walk_forward.
    WalkForwardWeightCache` built once from the baseline config, cheaply
    re-scoring each fold's cached candidates under the new costs via
    :meth:`~quantlab.validation.walk_forward.WalkForwardValidator.
    rescore_with_costs` instead of re-running signal generation and
    allocation once per scenario. Every execution-delay and reduced-universe
    scenario genuinely changes the weights themselves (the delay shift and
    the tradable universe respectively), so each still re-executes
    :class:`~quantlab.validation.walk_forward.WalkForwardValidator` end to
    end. Every "best N days removed" scenario stays a post-hoc transform of
    the already-realised baseline OOS returns: it changes no configuration,
    so re-running walk-forward would only waste time reproducing the exact
    same selection.

    Args:
        data: Canonical long OHLCV frame.
        config: The baseline experiment config (``validation.method`` must
            be ``"walk_forward"``).
        wf_baseline: The already-computed baseline ``WalkForwardResult``,
            reused for "baseline" and every "best N days removed" scenario.
        on_progress: Optional callback invoked as ``on_progress(done, total)``
            once before any work starts and once after each unit of work
            completes: one tick per candidate (folds x grid size) while the
            weight cache used by the commission/slippage scenarios is
            (re)built — the bulk of the total cost — then one tick per
            remaining scenario (each execution-delay/reduced-universe
            scenario is a full walk-forward run; the commission/slippage
            rescores and best-N-days-removed scenarios are cheap).
        checkpoint_path: Optional path to persist progress to, so an
            interrupted run resumes instead of starting over — at the
            scenario-block level (baseline / [cache-build + every
            commission/slippage rescore] / every execution-delay scenario /
            every best-N-days scenario / every reduced-universe scenario)
            for this function's own progress, plus a separate, nested
            checkpoint for the cache-build block specifically (passed
            through to :meth:`~quantlab.validation.walk_forward.
            WalkForwardValidator.run_with_weight_cache`), since that block is
            the expensive one and the whole point of the weight-cache
            optimisation is not to have to redo it. See
            ``quantlab.validation.checkpoint``.
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
    # wf_baseline must actually describe `data`/`config` -- otherwise every
    # scenario below (built from `config`, some sharing wf_baseline's own
    # cached weights via rescore_with_costs) would silently perturb a
    # baseline that doesn't correspond to this run at all.
    baseline_config = wf_baseline.oos_result.config
    if baseline_config.model_dump(mode="json") != config.model_dump(mode="json"):
        raise ValueError(
            "wf_baseline was not built from `config`: its own attached "
            "config differs from the config passed to this call."
        )
    # Required, not merely checked-when-present: a baseline with no recorded
    # data_hash at all is not "unverifiable, proceed anyway" -- it is
    # exactly the case this check exists to catch, so it must refuse the
    # same as a genuine mismatch would.
    baseline_data_hash = wf_baseline.oos_result.metadata.get("data_hash")
    if baseline_data_hash is None:
        raise ValueError(
            "wf_baseline was not built from `data`: its own metadata has no "
            "recorded data_hash, so it cannot be verified against this "
            "call's data."
        )
    from quantlab.data.storage import ParquetStorage

    if baseline_data_hash != ParquetStorage.hash_frame(data):
        raise ValueError(
            "wf_baseline was not built from `data`: its own recorded "
            "data_hash differs from this data's hash."
        )
    periods_per_year = positive_int(config.periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(config.risk_free_rate, name="risk_free_rate")
    train_window, validation_window, test_window, step = resolve_walk_forward_windows(
        config
    )
    expanding = config.validation.expanding
    grid = parameter_grid_for_config(config)

    # `config`'s own equality with `baseline_config` (checked above) does
    # NOT by itself guarantee these were the grid/windows/expanding
    # actually used to build `wf_baseline`: a caller can build it with an
    # explicit grid/windows that diverge from what `config` alone would
    # derive (WalkForwardValidator.run() accepts them as independent
    # arguments, not solely inferred from config). Every scenario below
    # reuses this recomputed grid/windows, including via wf_baseline's own
    # cached candidate weights (rescore_with_costs) -- comparing a baseline
    # built under one methodology against scenarios re-derived under a
    # silently different one would be methodologically incoherent, so this
    # must be verified explicitly rather than assumed from config alone.
    baseline_windows = wf_baseline.oos_result.metadata.get("walk_forward_windows")
    expected_windows = {
        "train_window": train_window,
        "validation_window": validation_window,
        "test_window": test_window,
        "step": step,
        "expanding": expanding,
    }
    if baseline_windows != expected_windows:
        raise ValueError(
            "wf_baseline was not built with this call's own train/"
            "validation/test windows or expanding setting: its own recorded "
            f"walk_forward_windows {baseline_windows!r} does not match "
            f"{expected_windows!r} derived from `config`."
        )
    baseline_grid = wf_baseline.oos_result.metadata.get("walk_forward_parameter_grid")
    if baseline_grid != grid:
        raise ValueError(
            "wf_baseline was not built with this call's own parameter grid: "
            f"its own recorded walk_forward_parameter_grid {baseline_grid!r} "
            f"does not match {grid!r} derived from `config`."
        )
    # Every configured execution-delay scenario below is a delay relative to
    # the baseline's own delay; a baseline already run with a non-zero delay
    # would make "delay +N" mean "N more than the baseline's own", not a
    # delay of exactly N. Only checked when a delay scenario is actually
    # configured -- an unrelated non-zero baseline delay is not this
    # function's problem when no delay scenario will ever use it.
    delay_scenarios = _execution_delay_scenarios(config)
    if delay_scenarios:
        baseline_delay = wf_baseline.oos_result.metadata.get(
            "walk_forward_execution_delay"
        )
        if baseline_delay != 0:
            raise ValueError(
                "wf_baseline was not built with execution_delay=0: its own "
                f"recorded walk_forward_execution_delay is {baseline_delay!r}, "
                "so this call's configured execution-delay scenarios cannot "
                "be compared against it."
            )

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
            step=step,
            execution_delay=execution_delay,
        )

    cost_scenarios = _cost_scenarios(config)
    best_days_scenarios = _best_days_removed_scenarios(config)
    universe_scenarios = _reduced_universe_scenarios(config)
    n_baseline_folds = len(wf_baseline.folds)
    n_combinations = math.prod(len(values) for values in grid.values()) if grid else 1
    # No cost scenario means the weight cache below is never consulted --
    # skip counting (and later, building) it entirely rather than paying for
    # the second-most-expensive step of this whole function for nothing.
    n_cache_units = (n_baseline_folds * n_combinations) if cost_scenarios else 0
    total_units = (
        n_cache_units
        + len(cost_scenarios)
        + len(delay_scenarios)
        + len(best_days_scenarios)
        + len(universe_scenarios)
    )

    def _cache_progress(done: int, _total: int) -> None:
        if on_progress is not None:
            on_progress(done, total_units)

    # Scenario-block-level checkpoint (own file): baseline / [cache-build +
    # every commission/slippage rescore, as one block] / every
    # execution-delay scenario / every best-N-days scenario / every
    # reduced-universe scenario -- always exactly 5 blocks, even when a
    # scenario type's own list is empty (that block then simply appends no
    # rows). `rows` is the only state that needs to survive between blocks
    # — every block computes from `config`/`data`/`wf_baseline`, already
    # covered by `provenance`, not from a prior block's output.
    rows: list[dict[str, object]] = []
    provenance: dict[str, Any] | None = None
    cache_checkpoint_path: Path | None = None
    completed_blocks = 0
    total_blocks = 5
    _block_row_counts = [
        1,
        len(cost_scenarios),
        len(delay_scenarios),
        len(best_days_scenarios),
        len(universe_scenarios),
    ]

    def _expected_block_row_count(progress: int) -> int:
        """Return exactly how many rows the first ``progress`` blocks append.

        Not a simple linear formula in ``progress`` alone (each block can
        append a different, configured number of rows), but still fully
        determined by it -- there is no ambiguity to fall back to a mere
        upper bound for.
        """
        return sum(_block_row_counts[:progress])

    # The exact scenario name each row must have, in order -- not just how
    # many rows there should be. Fixed by the block structure above:
    # baseline, then every commission/slippage rescore (in `cost_scenarios`'
    # own order), then every execution-delay, best-N-days and
    # reduced-universe scenario.
    _scenario_names_in_order = (
        ["baseline"]
        + [scenario.name for scenario in cost_scenarios]
        + [name for name, _ in delay_scenarios]
        + [name for name, _ in best_days_scenarios]
        + [name for name, _ in universe_scenarios]
    )
    _ensure_unique_scenario_names(_scenario_names_in_order)

    def _validate_block_state(state: Any, progress: int) -> bool:
        # A structurally-plausible-but-incoherent checkpoint (e.g.
        # progress=5 with zero rows, four rows all named "baseline", or a
        # row claiming success while carrying NaN metrics) must never be
        # resumed from -- it would silently skip every remaining block,
        # corrupt the table with duplicated/misnamed scenarios, or return a
        # result as if the whole run had succeeded when a scenario actually
        # failed. Row count is exactly determined by progress (see
        # _expected_block_row_count), so an exact match -- not just an
        # upper bound -- is meaningful here, and so is each row's own
        # scenario name and order, schema, and status/metrics/error
        # consistency (a malformed row would otherwise only surface later,
        # as a confusing KeyError deep in table assembly).
        if not (0 <= progress <= total_blocks and isinstance(state, list)):
            return False
        expected_row_count = _expected_block_row_count(progress)
        if len(state) != expected_row_count:
            return False
        if not all(
            isinstance(row, dict) and row.keys() == set(_STRESS_COLUMNS)
            for row in state
        ):
            return False
        expected_names = _scenario_names_in_order[:expected_row_count]
        if [row["scenario"] for row in state] != expected_names:
            return False
        return all(_stress_row_is_consistent(row) for row in state)

    if checkpoint_path is not None:
        provenance = compute_provenance(config, data)
        cache_checkpoint_path = stress_test_checkpoint_paths(checkpoint_path)[1]
        checkpoint_result = load_checkpoint(
            checkpoint_path, provenance, validate=_validate_block_state
        )
        if checkpoint_result is not None:
            # `progress` (block count) is loaded as-saved, never re-derived
            # from `len(rows)`: block 2 alone appends one row per cost
            # scenario, so row count and block count diverge -- deriving
            # "how many blocks are done" from len(rows) would misjudge that
            # and silently skip the blocks after it on resume.
            rows, completed_blocks = checkpoint_result
            logger.info(
                "Resuming walk-forward stress tests from checkpoint: %d/%d "
                "scenario blocks already done.",
                completed_blocks,
                total_blocks,
            )
    # Units already behind us for on_progress purposes, seeded from however
    # many blocks a resume already found done — block 1 (baseline) isn't
    # itself counted in total_units (see its definition above), only 2-5 are.
    completed_units = 0
    if completed_blocks >= 2:
        completed_units += n_cache_units + len(cost_scenarios)
    if completed_blocks >= 3:
        completed_units += len(delay_scenarios)
    if completed_blocks >= 4:
        completed_units += len(best_days_scenarios)
    if completed_blocks >= 5:
        completed_units += len(universe_scenarios)
    if on_progress is not None and completed_blocks >= 2:
        # Otherwise block 2 is about to run (fresh or resumed) and
        # run_with_weight_cache() below already emits its own accurate
        # first tick via _cache_progress — a seed call here would either
        # duplicate its (0, total_units) or, when resuming mid-cache-build,
        # jump straight from a stale 0 to whatever candidate it actually
        # resumes at.
        on_progress(completed_units, total_units)

    def _checkpoint_block() -> None:
        if checkpoint_path is not None and provenance is not None:
            save_checkpoint(checkpoint_path, provenance, rows, completed_blocks)

    if completed_blocks < 1:
        rows.append(
            _metrics_row(
                "baseline",
                wf_baseline.oos_result.returns,
                periods_per_year,
                risk_free_rate,
            )
        )
        completed_blocks += 1
        _checkpoint_block()

    if completed_blocks < 2:
        if cost_scenarios:
            # Signals and portfolio allocation never depend on execution
            # costs, so build the per-candidate weight cache once (this is
            # where the candidate-level progress reported above comes from —
            # finer-grained than one tick per fold, since this build does
            # roughly twice the work of a plain walk-forward fold) and reuse
            # it for every cost-only scenario below instead of paying for a
            # full re-run each. This block gets its own nested checkpoint
            # (see the docstring): it's the expensive one, and the whole
            # point of the weight-cache optimisation is not having to redo
            # it. Skipped entirely when there is no cost scenario to use it
            # for -- building it would waste this function's second-most
            # expensive step on nothing.
            validator = WalkForwardValidator(config)
            _, weight_cache = validator.run_with_weight_cache(
                data,
                parameter_grid=grid,
                train_window=train_window,
                validation_window=validation_window,
                test_window=test_window,
                expanding=expanding,
                step=step,
                execution_delay=0,
                on_progress=_cache_progress,
                checkpoint_path=cache_checkpoint_path,
            )
            # No separate on_progress call needed here: run_with_weight_
            # cache's own ticks (via _cache_progress) already reached
            # completed_units == n_cache_units by the time it returns.
            completed_units = n_cache_units

            for scenario in cost_scenarios:
                scenario_config = _cost_scenario_config(config, scenario)
                wf = validator.rescore_with_costs(weight_cache, scenario_config)
                if wf.oos_result is None:
                    # Rescoring reuses the baseline's own already-fitted
                    # folds/windows, so this should be unreachable -- a hard
                    # invariant violation, not an expected external failure
                    # to report as a "failed" row.
                    raise QuantLabError(
                        f"{scenario.name}: rescore_with_costs produced no OOS "
                        "result despite reusing the baseline's own windows."
                    )
                rows.append(
                    _metrics_row(
                        scenario.name,
                        wf.oos_result.returns,
                        periods_per_year,
                        risk_free_rate,
                    )
                )
                completed_units += 1
                if on_progress is not None:
                    on_progress(completed_units, total_units)
        completed_blocks += 1
        if cache_checkpoint_path is not None:
            clear_checkpoint(cache_checkpoint_path)
        _checkpoint_block()
    # else: already done on a previous attempt — completed_units was already
    # seeded correctly above, nothing left to recompute for this block.

    if completed_blocks < 3:
        for name, delay in delay_scenarios:
            try:
                delayed = _run_walk_forward(config, data, grid, execution_delay=delay)
                if delayed.oos_result is None:
                    raise QuantLabError(
                        f"{name}: no walk-forward fold fit under this delay."
                    )
            except QuantLabError as exc:
                logger.warning("%s scenario failed: %s", name, exc)
                rows.append(_failed_row(name, exc))
            else:
                rows.append(
                    _metrics_row(
                        name,
                        delayed.oos_result.returns,
                        periods_per_year,
                        risk_free_rate,
                    )
                )
            completed_units += 1
            if on_progress is not None:
                on_progress(completed_units, total_units)
        completed_blocks += 1
        _checkpoint_block()

    if completed_blocks < 4:
        for name, n in best_days_scenarios:
            try:
                scenario_returns = remove_best_days(wf_baseline.oos_result.returns, n)
            except QuantLabError as exc:
                logger.warning("%s scenario failed: %s", name, exc)
                rows.append(_failed_row(name, exc))
            else:
                rows.append(
                    _metrics_row(
                        name, scenario_returns, periods_per_year, risk_free_rate
                    )
                )
            completed_units += 1
            if on_progress is not None:
                on_progress(completed_units, total_units)
        completed_blocks += 1
        _checkpoint_block()

    if completed_blocks < 5:
        for name, count in universe_scenarios:
            if not _universe_reduction_is_feasible(config, count):
                rows.append(
                    _failed_row(
                        name,
                        QuantLabError(
                            f"Universe has only {len(config.symbols)} symbols; "
                            f"removing {count} would leave fewer than 2 tradable."
                        ),
                    )
                )
                completed_units += 1
                if on_progress is not None:
                    on_progress(completed_units, total_units)
                continue
            reduced_symbols = config.symbols[:-count]
            reduced_instruments = [
                instrument
                for instrument in config.data.instruments
                if instrument.symbol in reduced_symbols
            ]
            try:
                data_config = config.data.revalidated_copy(
                    update={"instruments": reduced_instruments}
                )
                reduced_config = config.revalidated_copy(update={"data": data_config})
                required_symbols = set(reduced_symbols)
                if config.benchmark_symbol is not None:
                    required_symbols.add(config.benchmark_symbol)
                subset = data[data[SYMBOL].isin(required_symbols)].reset_index(
                    drop=True
                )
                reduced_grid = parameter_grid_for_config(reduced_config)
                wf_reduced = _run_walk_forward(reduced_config, subset, reduced_grid)
                if wf_reduced.oos_result is None:
                    raise QuantLabError(
                        "No walk-forward fold fit the reduced universe's history."
                    )
            except QuantLabError as exc:
                logger.warning("Reduced-universe walk-forward scenario failed: %s", exc)
                rows.append(_failed_row(name, exc))
            else:
                rows.append(
                    _metrics_row(
                        name,
                        wf_reduced.oos_result.returns,
                        periods_per_year,
                        risk_free_rate,
                    )
                )
            completed_units += 1
            if on_progress is not None:
                on_progress(completed_units, total_units)
        completed_blocks += 1
        _checkpoint_block()

    if checkpoint_path is not None:
        clear_checkpoint(checkpoint_path)
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
