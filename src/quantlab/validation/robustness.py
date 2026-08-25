"""Stress scenarios and a random-sign test for realised returns."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.constants import SYMBOL, TIMESTAMP
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


def run_stress_tests(
    data: pd.DataFrame,
    config: ExperimentConfig,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
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

    reduced_universe = len(config.symbols) > 2
    total_scenarios = 1 + 3 + 2 + (1 if reduced_universe else 0)

    rows: list[dict[str, object]] = []
    # "best 10 days removed" needs the actual baseline returns Series later,
    # not just its already-computed metrics row, so it has to be part of the
    # checkpointed state too, or resuming past "baseline" would lose it.
    baseline_returns: pd.Series | None = None
    provenance: dict[str, Any] | None = None
    completed_scenarios = 0

    # The exact scenario name each row must have, in order -- fixed by this
    # function's own scenario sequence below, independent of `progress`
    # (which only ever names a prefix of it: a config with two symbols or
    # fewer never reaches "reduced universe", and neither does an in-progress
    # resume).
    _scenario_names_in_order = [
        "baseline",
        "commission x2",
        "commission x5",
        "slippage x2",
        "execution delay +1",
        "best 10 days removed",
        "reduced universe",
    ]

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

    scenarios = {
        "commission x2": scale_costs(config, commission_mult=2.0),
        "commission x5": scale_costs(config, commission_mult=5.0),
        "slippage x2": scale_costs(config, slippage_mult=2.0),
    }
    for position, (name, scenario_config) in enumerate(scenarios.items(), start=2):
        if completed_scenarios < position:
            result = run_backtest_from_config(data, scenario_config)
            rows.append(
                _metrics_row(name, result.returns, periods_per_year, risk_free_rate)
            )
            completed_scenarios += 1
            _checkpoint()
            if on_progress is not None:
                on_progress(completed_scenarios, total_scenarios)

    if completed_scenarios < 5:
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
        _checkpoint()
        if on_progress is not None:
            on_progress(completed_scenarios, total_scenarios)

    if completed_scenarios < 6:
        assert baseline_returns is not None
        rows.append(
            _metrics_row(
                "best 10 days removed",
                remove_best_days(baseline_returns, 10),
                periods_per_year,
                risk_free_rate,
            )
        )
        completed_scenarios += 1
        _checkpoint()
        if on_progress is not None:
            on_progress(completed_scenarios, total_scenarios)

    if reduced_universe and completed_scenarios < 7:
        reduced_symbols = config.symbols[:-1]
        reduced_instruments = [
            instrument
            for instrument in config.data.instruments
            if instrument.symbol in reduced_symbols
        ]
        data_config = config.data.revalidated_copy(
            update={"instruments": reduced_instruments}
        )
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
        _checkpoint()
        if on_progress is not None:
            on_progress(completed_scenarios, total_scenarios)

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
        checkpoint_path: Optional path to persist progress to, so an
            interrupted run resumes instead of starting over — at the
            scenario-block level (baseline / [cache-build + the 3 cost-only
            rescores] / execution-delay+1 / best-10-days / reduced-universe)
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
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
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
    # The "execution delay +1" scenario below is only meaningful relative to
    # a zero-delay baseline; a baseline already run with a non-zero delay
    # would need "+1" to mean "one more than the baseline's own", not a
    # hardcoded absolute 1.
    baseline_delay = wf_baseline.oos_result.metadata.get("walk_forward_execution_delay")
    if baseline_delay != 0:
        raise ValueError(
            "wf_baseline was not built with execution_delay=0: its own "
            f"recorded walk_forward_execution_delay is {baseline_delay!r}, "
            "so this call's 'execution delay +1' scenario cannot be "
            "compared against it."
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
            execution_delay=execution_delay,
        )

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

    # Scenario-block-level checkpoint (own file): baseline / [cache-build +
    # the 3 cost-only rescores, as one block] / execution-delay+1 /
    # best-10-days / reduced-universe. `rows` is the only state that needs
    # to survive between blocks — every block computes from `config`/`data`/
    # `wf_baseline`, already covered by `provenance`, not from a prior
    # block's output.
    rows: list[dict[str, object]] = []
    provenance: dict[str, Any] | None = None
    cache_checkpoint_path: Path | None = None
    completed_blocks = 0
    total_blocks = 4 + (1 if reduced_universe else 0)

    def _expected_block_row_count(progress: int) -> int:
        """Return exactly how many rows each block count has appended.

        Block 1 ("baseline") appends 1 row; block 2 appends one row per
        cost-only scenario (``len(cost_scenarios)`` = 3: commission x2,
        commission x5, slippage x2); blocks 3-5 ("execution delay +1",
        "best 10 days removed", "reduced universe") each append exactly 1.
        Not a simple linear formula in ``progress`` alone, but still fully
        determined by it -- there is no ambiguity to fall back to a mere
        upper bound for.
        """
        if progress <= 0:
            return 0
        if progress == 1:
            return 1
        if progress == 2:
            return 1 + len(cost_scenarios)
        return 1 + len(cost_scenarios) + (progress - 2)

    # The exact scenario name each row must have, in order -- not just how
    # many rows there should be. Fixed by the block structure above:
    # baseline, then the 3 cost-only rescores (in `cost_scenarios`' own
    # order), then execution delay, best-10-days, and (if applicable)
    # reduced universe.
    _scenario_names_in_order = [
        "baseline",
        *cost_scenarios.keys(),
        "execution delay +1",
        "best 10 days removed",
        "reduced universe",
    ]

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
    completed_units += max(0, completed_blocks - 2)
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
        # Signals and portfolio allocation never depend on execution costs,
        # so build the per-candidate weight cache once (this is where the
        # candidate-level progress reported above comes from — finer-grained
        # than one tick per fold, since this build does roughly twice the
        # work of a plain walk-forward fold) and reuse it for every
        # cost-only scenario below instead of paying for a full re-run each.
        # This block gets its own nested checkpoint (see the docstring):
        # it's the expensive one, and the whole point of the weight-cache
        # optimisation is not having to redo it.
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
            checkpoint_path=cache_checkpoint_path,
        )
        # No separate on_progress call needed here: run_with_weight_cache's
        # own ticks (via _cache_progress) already reached completed_units ==
        # n_cache_units by the time it returns.
        completed_units = n_cache_units

        for name, scenario_config in cost_scenarios.items():
            wf = validator.rescore_with_costs(weight_cache, scenario_config)
            assert wf.oos_result is not None  # same data/windows as the baseline
            rows.append(
                _metrics_row(
                    name, wf.oos_result.returns, periods_per_year, risk_free_rate
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
        completed_blocks += 1
        _checkpoint_block()
        if on_progress is not None:
            on_progress(completed_units, total_units)

    if completed_blocks < 4:
        rows.append(
            _metrics_row(
                "best 10 days removed",
                remove_best_days(wf_baseline.oos_result.returns, 10),
                periods_per_year,
                risk_free_rate,
            )
        )
        completed_units += 1
        completed_blocks += 1
        _checkpoint_block()
        if on_progress is not None:
            on_progress(completed_units, total_units)

    if reduced_universe and completed_blocks < 5:
        reduced_symbols = config.symbols[:-1]
        reduced_instruments = [
            instrument
            for instrument in config.data.instruments
            if instrument.symbol in reduced_symbols
        ]
        data_config = config.data.revalidated_copy(
            update={"instruments": reduced_instruments}
        )
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
        completed_blocks += 1
        _checkpoint_block()
        if on_progress is not None:
            on_progress(completed_units, total_units)

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
