"""Walk-forward parameter selection and out-of-sample evaluation."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from quantlab.backtesting.accounting import (
    AccountingResult,
    compute_asset_returns,
    portfolio_metrics_from_accounting,
    run_accounting,
)
from quantlab.backtesting.benchmark import build_benchmark
from quantlab.backtesting.engine import (
    _dependency_versions,
    _git_commit_hash,
    _git_is_dirty,
    _source_hash,
)
from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.runner import (
    build_execution_from_config,
    build_strategy_from_config,
    run_backtest_from_config,
)
from quantlab.backtesting.trade_log import build_trade_log
from quantlab.config import BenchmarkKind, ExperimentConfig
from quantlab.constants import SYMBOL
from quantlab.data.base import price_matrix
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import InvalidConfigurationError
from quantlab.execution.execution_model import ExecutionModel
from quantlab.logging_config import get_logger
from quantlab.portfolio.rebalancing import (
    rebalance_and_cap_turnover,
)
from quantlab.portfolio.rebalancing import (
    rebalance_dates as compute_rebalance_dates,
)
from quantlab.risk import metrics as M
from quantlab.risk._validation import boolean, finite_real, positive_int
from quantlab.strategies import strategy_parameter_names
from quantlab.validation.checkpoint import (
    clear_checkpoint,
    compute_provenance,
    load_checkpoint,
    save_checkpoint,
)
from quantlab.validation.splits import WalkForwardWindow, walk_forward_windows

logger = get_logger(__name__)

# A common signature ensures risk-adjusted scorers receive the configured
# annual risk-free rate.
_SCORERS: dict[str, Callable[[pd.Series, pd.Series, int, float], float]] = {
    "sharpe": lambda returns, equity, ppy, rf: M.sharpe_ratio(returns, rf, ppy),
    "sortino": lambda returns, equity, ppy, rf: M.sortino_ratio(returns, rf, ppy),
    "calmar": lambda returns, equity, ppy, rf: M.calmar_ratio(equity, ppy),
    "total_return": lambda returns, equity, ppy, rf: M.total_return(equity),
}


@dataclass
class FoldResult:
    """Selected parameters and realised metrics for one OOS interval."""

    fold: int
    best_params: dict[str, Any]
    validation_score: float
    test_return: float
    test_sharpe: float
    test_returns: pd.Series


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output."""

    folds: list[FoldResult] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    oos_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # A genuine BacktestResult built from the stitched out-of-sample series,
    # reusing the same trade-log/benchmark/metrics pipeline as a single
    # backtest so existing result-rendering code (dashboard, HTML report)
    # works unchanged. `None` only when no fold produced any OOS weights.
    oos_result: BacktestResult | None = None

    def summary_table(self) -> pd.DataFrame:
        """Return selected parameters and OOS metrics for each fold."""
        rows = [
            {
                "fold": fold.fold,
                **{f"param_{key}": value for key, value in fold.best_params.items()},
                "validation_score": fold.validation_score,
                "test_return": fold.test_return,
                "test_sharpe": fold.test_sharpe,
            }
            for fold in self.folds
        ]
        return pd.DataFrame(rows)

    def parameter_stability(self) -> dict[str, float]:
        """Return coefficients of variation for finite numeric choices.

        Lower values mean that selection varied less across folds, but do not
        by themselves establish strategy robustness.
        """
        if not self.folds:
            return {}
        stability: dict[str, float] = {}
        for key in self.folds[0].best_params:
            values: list[float] = []
            for fold in self.folds:
                value = fold.best_params.get(key)
                if (
                    isinstance(value, Real)
                    and not isinstance(value, (bool, np.bool_))
                    and np.isfinite(float(value))
                ):
                    values.append(float(value))
            mean = float(np.mean(values)) if values else 0.0
            if len(values) >= 2 and not np.isclose(mean, 0.0):
                stability[key] = float(np.std(values) / abs(mean))
        return stability

    def oos_metrics(
        self, periods_per_year: int = 252, risk_free_rate: float = 0.0
    ) -> dict[str, float]:
        """Compute metrics for the stitched out-of-sample curve."""
        if len(self.oos_returns) < 2:
            return {}
        return M.compute_metrics(
            self.oos_returns,
            self.oos_equity,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )


@dataclass
class _FoldCandidateWeights:
    """One candidate's cost-independent weights, cached for one fold.

    ``None`` fields mean this candidate was skipped for insufficient
    train/validation warm-up on this fold (mirrors ``_select_on_validation``'s
    own skip condition).
    """

    validation_weights: pd.DataFrame | None
    test_targets: pd.DataFrame | None


@dataclass
class WalkForwardWeightCache:
    """Per-fold, per-candidate weights captured while selecting a baseline.

    Signal generation and portfolio allocation never depend on execution
    costs (commission/spread/slippage) — only the accounting step does (see
    :meth:`WalkForwardValidator.rescore_with_costs`). Built by
    :meth:`WalkForwardValidator.run_with_weight_cache`.
    """

    data: pd.DataFrame
    combinations: list[dict[str, Any]]
    fold_windows: list[WalkForwardWindow]
    candidates: list[list[_FoldCandidateWeights]]
    grid: Mapping[str, Sequence[Any]]
    train_window: int
    validation_window: int
    test_window: int
    expanding: bool
    execution_delay: int


def resolve_walk_forward_windows(config: ExperimentConfig) -> tuple[int, int, int]:
    """Resolve train/validation/test windows, applying the documented default.

    Shared by the CLI, dashboard and walk-forward-aware robustness/sensitivity
    functions so the 500/126/126 fallback lives in exactly one place.
    """
    return (
        config.validation.train_window or 500,
        config.validation.validation_window or 126,
        config.validation.test_window or 126,
    )


def _with_params(config: ExperimentConfig, params: dict[str, Any]) -> ExperimentConfig:
    """Return a revalidated config with strategy parameters overridden."""
    merged = {**config.strategy.parameters, **params}
    try:
        strategy = config.strategy.revalidated_copy(update={"parameters": merged})
        return config.revalidated_copy(update={"strategy": strategy})
    except ValidationError as exc:
        raise InvalidConfigurationError(
            f"Invalid walk-forward parameter combination {params}: {exc}"
        ) from exc


@dataclass
class _PreparedWalkForward:
    """Validated inputs shared by run() and run_with_weight_cache()."""

    tradable: pd.DataFrame
    windows: list[WalkForwardWindow]
    combinations: list[dict[str, Any]]
    periods_per_year: int
    risk_free_rate: float
    scorer: Callable[[pd.Series, pd.Series, int, float], float]
    grid: dict[str, Sequence[Any]]
    delay: int


class WalkForwardValidator:
    """Select parameters on validation blocks and evaluate them OOS."""

    def __init__(self, base_config: ExperimentConfig) -> None:
        if not isinstance(base_config, ExperimentConfig):
            raise TypeError("base_config must be an ExperimentConfig.")
        self.base_config = base_config

    def run(
        self,
        data: pd.DataFrame,
        parameter_grid: Mapping[str, Sequence[Any]],
        train_window: int,
        validation_window: int,
        test_window: int,
        *,
        expanding: bool = True,
        execution_delay: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
        checkpoint_path: Path | None = None,
    ) -> WalkForwardResult:
        """Execute chronological parameter selection and OOS accounting.

        Args:
            data: Canonical long OHLCV frame.
            parameter_grid: Candidate values per strategy parameter to select
                on each fold's validation block.
            train_window: Training periods per fold.
            validation_window: Validation periods per fold, used for
                parameter selection.
            test_window: Out-of-sample test periods per fold.
            expanding: Grow the training window across folds instead of
                sliding it.
            on_progress: Optional callback invoked as ``on_progress(done,
                total)`` once before the first candidate (``done=0``, or the
                resumed count if ``checkpoint_path`` supplied a partial run)
                and once after each candidate is considered plus once more
                after each fold's out-of-sample weights are computed —
                ``total`` being folds x (grid size + 1), matching
                ``estimate_walk_forward_backtest_count`` — for a caller (e.g.
                the dashboard) to drive a live progress bar instead of
                estimating a duration upfront. One tick per whole fold was
                too coarse in practice: a fold's own grid search can take
                long enough that a caller's pace estimate had too few, too
                lumpy data points to track a real, sustained slowdown across
                an expanding walk-forward's later, bigger folds.
            execution_delay: Extra periods of execution delay applied to both
                validation-block parameter selection and the final
                out-of-sample weights (the "acting on stale signals" stress
                scenario, re-run through the whole selection process rather
                than only rescaling the final numbers).
            checkpoint_path: Optional path to persist per-fold progress to,
                so an interrupted run resumes from its last completed fold
                instead of starting over. Silently ignored (fresh start) if
                nothing valid is there yet; automatically cleared once this
                call completes. See ``quantlab.validation.checkpoint``.
        """
        started = time.perf_counter()
        expanding = boolean(expanding, name="expanding")
        prepared = self._prepare(
            data,
            parameter_grid,
            train_window,
            validation_window,
            test_window,
            expanding=expanding,
            execution_delay=execution_delay,
        )

        fold_windows: list[WalkForwardWindow] = []
        fold_parameters: list[dict[str, Any]] = []
        fold_scores: list[float] = []
        target_pieces: list[pd.DataFrame] = []

        provenance: dict[str, Any] | None = None
        if checkpoint_path is not None:
            provenance = compute_provenance(
                self.base_config,
                data,
                train_window=train_window,
                validation_window=validation_window,
                test_window=test_window,
                expanding=expanding,
                execution_delay=execution_delay,
                parameter_grid={k: list(v) for k, v in parameter_grid.items()},
            )
            checkpoint_state = load_checkpoint(checkpoint_path, provenance)
            if checkpoint_state is not None:
                fold_windows, fold_parameters, fold_scores, target_pieces = (
                    checkpoint_state
                )
                logger.info(
                    "Resuming walk-forward from checkpoint: %d/%d folds already done.",
                    len(fold_windows),
                    len(prepared.windows),
                )
        resumed_folds = len(fold_windows)

        # One tick per candidate considered plus one for the fold's final
        # out-of-sample weights, not one tick per whole fold: see the
        # on_progress docstring above for why the coarser version made a
        # caller's live pace estimate too imprecise.
        total_units = len(prepared.windows) * (len(prepared.combinations) + 1)
        units_done = resumed_folds * (len(prepared.combinations) + 1)
        if on_progress is not None:
            on_progress(units_done, total_units)

        def _tick() -> None:
            nonlocal units_done
            units_done += 1
            if on_progress is not None:
                on_progress(units_done, total_units)

        for window in prepared.windows[resumed_folds:]:
            best = self._select_on_validation(
                data,
                window,
                prepared.combinations,
                prepared.scorer,
                prepared.periods_per_year,
                prepared.risk_free_rate,
                execution_delay=prepared.delay,
                on_candidate=_tick,
            )
            targets = self._weights_on_test(data, window, best["params"])
            fold_windows.append(window)
            fold_parameters.append(best["params"])
            fold_scores.append(best["score"])
            _tick()
            target_pieces.append(targets)
            if checkpoint_path is not None and provenance is not None:
                save_checkpoint(
                    checkpoint_path,
                    provenance,
                    (fold_windows, fold_parameters, fold_scores, target_pieces),
                    len(fold_windows),
                )

        result = self._finalize(
            fold_windows,
            fold_parameters,
            fold_scores,
            target_pieces,
            prepared.tradable,
            data,
            self.base_config,
            prepared.periods_per_year,
            prepared.risk_free_rate,
            prepared.delay,
            prepared.grid,
            train_window,
            validation_window,
            test_window,
            expanding,
            started,
        )
        if checkpoint_path is not None:
            clear_checkpoint(checkpoint_path)
        return result

    def run_with_weight_cache(
        self,
        data: pd.DataFrame,
        parameter_grid: Mapping[str, Sequence[Any]],
        train_window: int,
        validation_window: int,
        test_window: int,
        *,
        expanding: bool = True,
        execution_delay: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
        checkpoint_path: Path | None = None,
    ) -> tuple[WalkForwardResult, WalkForwardWeightCache]:
        """Like :meth:`run`, but also return a cache of per-candidate weights.

        Signal generation and portfolio allocation never depend on execution
        costs, so a scenario that only rescales commission/spread/slippage
        can reuse the returned cache via :meth:`rescore_with_costs` instead
        of paying for a full walk-forward re-run. Building the cache
        computes out-of-sample target weights for every candidate on every
        fold (not only each fold's winner, since a different candidate can
        win once costs change), so this method itself costs somewhat more
        than :meth:`run` — the saving comes from amortising that extra cost
        across every cost-only scenario that reuses the cache instead of
        paying for a full re-run each.

        Args: as :meth:`run`, except ``on_progress`` is reported per
            candidate evaluated (``total`` = folds x candidates) rather than
            per fold, since this method does roughly twice the work of
            :meth:`run` per fold. ``checkpoint_path``, if given, checkpoints
            per fold (a fold's cached candidates are only ever used or
            dropped as a whole) — an interruption while this cache is being
            built resumes at the right fold instead of recomputing it from
            scratch, which matters here specifically because building it is
            the expensive part the cost-only stress scenarios are meant to
            amortise.
        """
        started = time.perf_counter()
        expanding = boolean(expanding, name="expanding")
        prepared = self._prepare(
            data,
            parameter_grid,
            train_window,
            validation_window,
            test_window,
            expanding=expanding,
            execution_delay=execution_delay,
        )

        fold_windows: list[WalkForwardWindow] = []
        fold_parameters: list[dict[str, Any]] = []
        fold_scores: list[float] = []
        target_pieces: list[pd.DataFrame] = []
        cached_candidates: list[list[_FoldCandidateWeights]] = []

        provenance: dict[str, Any] | None = None
        if checkpoint_path is not None:
            provenance = compute_provenance(
                self.base_config,
                data,
                train_window=train_window,
                validation_window=validation_window,
                test_window=test_window,
                expanding=expanding,
                execution_delay=execution_delay,
                parameter_grid={k: list(v) for k, v in parameter_grid.items()},
            )
            checkpoint_state = load_checkpoint(checkpoint_path, provenance)
            if checkpoint_state is not None:
                (
                    fold_windows,
                    fold_parameters,
                    fold_scores,
                    target_pieces,
                    cached_candidates,
                ) = checkpoint_state
                logger.info(
                    "Resuming weight-cache build from checkpoint: %d/%d folds "
                    "already done.",
                    len(fold_windows),
                    len(prepared.windows),
                )
        resumed_folds = len(fold_windows)

        # Reported per candidate evaluated, not per fold: this method does
        # roughly twice the work of run() per fold (it also captures every
        # candidate's test-block targets, not only the winner's), so
        # fold-level ticks alone could go a long time between updates —
        # especially with few folds — and look stalled to a caller like the
        # dashboard's progress bar.
        total_candidates = len(prepared.windows) * len(prepared.combinations)
        candidates_done = resumed_folds * len(prepared.combinations)
        if on_progress is not None:
            on_progress(candidates_done, total_candidates)
        for window in prepared.windows[resumed_folds:]:

            def _tick() -> None:
                nonlocal candidates_done
                candidates_done += 1
                if on_progress is not None:
                    on_progress(candidates_done, total_candidates)

            best_index, score, captured = self._select_and_capture(
                data,
                window,
                prepared.combinations,
                prepared.scorer,
                prepared.periods_per_year,
                prepared.risk_free_rate,
                execution_delay=prepared.delay,
                on_candidate=_tick,
            )
            fold_windows.append(window)
            fold_parameters.append(prepared.combinations[best_index])
            fold_scores.append(score)
            cached_candidates.append(captured)
            test_targets = captured[best_index].test_targets
            assert test_targets is not None
            target_pieces.append(test_targets)
            if checkpoint_path is not None and provenance is not None:
                save_checkpoint(
                    checkpoint_path,
                    provenance,
                    (
                        fold_windows,
                        fold_parameters,
                        fold_scores,
                        target_pieces,
                        cached_candidates,
                    ),
                    len(fold_windows),
                )

        result = self._finalize(
            fold_windows,
            fold_parameters,
            fold_scores,
            target_pieces,
            prepared.tradable,
            data,
            self.base_config,
            prepared.periods_per_year,
            prepared.risk_free_rate,
            prepared.delay,
            prepared.grid,
            train_window,
            validation_window,
            test_window,
            expanding,
            started,
        )
        cache = WalkForwardWeightCache(
            data=data,
            combinations=prepared.combinations,
            fold_windows=fold_windows,
            candidates=cached_candidates,
            grid=prepared.grid,
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            expanding=expanding,
            execution_delay=prepared.delay,
        )
        if checkpoint_path is not None:
            clear_checkpoint(checkpoint_path)
        return result, cache

    def rescore_with_costs(
        self,
        cache: WalkForwardWeightCache,
        scenario_config: ExperimentConfig,
    ) -> WalkForwardResult:
        """Cheaply re-score a cost-only scenario against a cached baseline.

        For every cached candidate on every fold, this re-runs only the
        accounting step under ``scenario_config``'s execution model — never
        the signal/allocation computation that produced the cached weights
        — to re-select each fold's winner, which can differ from the
        original selection since a stressed cost can change which candidate
        scores best. It then stitches the out-of-sample series exactly as
        :meth:`run` would from a fresh full re-run.

        Only valid when ``scenario_config`` differs from the config that
        built ``cache`` in execution-cost fields (commission/spread/
        slippage). Anything that would change signals, weights, the
        tradable universe or execution delay needs a fresh :meth:`run` or
        :meth:`run_with_weight_cache` instead.
        """
        started = time.perf_counter()
        periods_per_year = positive_int(
            scenario_config.periods_per_year, name="periods_per_year"
        )
        risk_free_rate = finite_real(
            scenario_config.backtest.risk_free_rate, name="risk_free_rate"
        )
        metric = scenario_config.validation.optimization_metric
        try:
            scorer = _SCORERS[metric]
        except KeyError as exc:
            raise InvalidConfigurationError(
                f"Unsupported walk-forward optimization metric: {metric!r}."
            ) from exc

        tradable = cache.data[cache.data[SYMBOL].isin(set(scenario_config.symbols))]
        fold_parameters: list[dict[str, Any]] = []
        fold_scores: list[float] = []
        target_pieces: list[pd.DataFrame] = []
        for window, row in zip(cache.fold_windows, cache.candidates, strict=True):
            sliced = _slice_between(cache.data, window.train[0], window.validation[-1])
            fold_tradable = sliced[sliced[SYMBOL].isin(set(scenario_config.symbols))]
            prices = price_matrix(fold_tradable, adjusted=True)
            asset_returns = compute_asset_returns(prices).loc[
                window.validation[0] : window.validation[-1]
            ]
            execution_model = build_execution_from_config(scenario_config, sliced)

            best_score = -np.inf
            best_index: int | None = None
            for index, candidate in enumerate(row):
                if candidate.validation_weights is None:
                    continue
                accounting = run_accounting(
                    candidate.validation_weights,
                    asset_returns,
                    execution_model,
                    scenario_config.initial_capital,
                )
                equity = M.equity_from_returns(accounting.net_returns)
                score = scorer(
                    accounting.net_returns, equity, periods_per_year, risk_free_rate
                )
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                raise InvalidConfigurationError(
                    f"Fold {window.fold}: every cached parameter combination "
                    "produced a non-finite validation score under this "
                    "scenario's execution costs."
                )
            fold_parameters.append(cache.combinations[best_index])
            fold_scores.append(best_score)
            test_targets = row[best_index].test_targets
            assert test_targets is not None
            target_pieces.append(test_targets)

        return self._finalize(
            cache.fold_windows,
            fold_parameters,
            fold_scores,
            target_pieces,
            tradable,
            cache.data,
            scenario_config,
            periods_per_year,
            risk_free_rate,
            cache.execution_delay,
            cache.grid,
            cache.train_window,
            cache.validation_window,
            cache.test_window,
            cache.expanding,
            started,
        )

    def _prepare(
        self,
        data: pd.DataFrame,
        parameter_grid: Mapping[str, Sequence[Any]],
        train_window: int,
        validation_window: int,
        test_window: int,
        *,
        expanding: bool,
        execution_delay: int,
    ) -> _PreparedWalkForward:
        """Validate inputs and compute what run() and run_with_weight_cache() share."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        required_columns = {"timestamp", SYMBOL}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise InvalidConfigurationError(
                f"data is missing required columns: {sorted(missing_columns)}."
            )
        if isinstance(execution_delay, bool) or not isinstance(
            execution_delay, Integral
        ):
            raise InvalidConfigurationError(
                "execution_delay must be a non-negative integer."
            )
        if execution_delay < 0:
            raise InvalidConfigurationError(
                "execution_delay must be a non-negative integer."
            )
        delay = int(execution_delay)
        grid = _validate_parameter_grid(parameter_grid, self.base_config.strategy_name)

        # A benchmark can have a different calendar, so folds follow only the
        # tradable universe.
        tradable = data[data[SYMBOL].isin(set(self.base_config.symbols))]
        present_symbols = set(tradable[SYMBOL].unique())
        missing_symbols = [
            symbol
            for symbol in self.base_config.symbols
            if symbol not in present_symbols
        ]
        if missing_symbols:
            raise InvalidConfigurationError(
                "Walk-forward data is missing configured tradable symbol(s): "
                f"{missing_symbols}. Refusing to validate a reduced universe."
            )
        index = pd.DatetimeIndex(price_matrix(tradable, adjusted=True).index)
        windows = walk_forward_windows(
            index, train_window, validation_window, test_window, expanding=expanding
        )
        if not windows:
            logger.warning(
                "No walk-forward windows fit in %d observations with "
                "train=%d validation=%d test=%d.",
                len(index),
                train_window,
                validation_window,
                test_window,
            )

        combinations = _grid_combinations(grid)
        periods_per_year = positive_int(
            self.base_config.periods_per_year, name="periods_per_year"
        )
        risk_free_rate = finite_real(
            self.base_config.backtest.risk_free_rate, name="risk_free_rate"
        )
        metric = self.base_config.validation.optimization_metric
        try:
            scorer = _SCORERS[metric]
        except KeyError as exc:
            raise InvalidConfigurationError(
                f"Unsupported walk-forward optimization metric: {metric!r}."
            ) from exc
        return _PreparedWalkForward(
            tradable=tradable,
            windows=windows,
            combinations=combinations,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            scorer=scorer,
            grid=grid,
            delay=delay,
        )

    def _finalize(
        self,
        fold_windows: list[WalkForwardWindow],
        fold_parameters: list[dict[str, Any]],
        fold_scores: list[float],
        target_pieces: list[pd.DataFrame],
        tradable: pd.DataFrame,
        data: pd.DataFrame,
        active_config: ExperimentConfig,
        periods_per_year: int,
        risk_free_rate: float,
        delay: int,
        grid: Mapping[str, Sequence[Any]],
        train_window: int,
        validation_window: int,
        test_window: int,
        expanding: bool,
        started: float,
    ) -> WalkForwardResult:
        """Stitch fold selections into OOS accounting and a WalkForwardResult.

        Shared by every selection path (:meth:`run`,
        :meth:`run_with_weight_cache`, :meth:`rescore_with_costs`) so
        rebalancing/turnover-capping, delay-shifting and final accounting
        run identically regardless of which one produced the winners, using
        whichever config (``active_config``) is actually in effect for the
        scenario being finalised.
        """
        # Applying rebalancing, turnover and accounting once preserves state
        # and transaction costs across fold boundaries.
        if target_pieces:
            all_targets = pd.concat(target_pieces).sort_index()
            if not all_targets.index.is_unique:
                duplicates = all_targets.index[all_targets.index.duplicated()].unique()
                raise InvalidConfigurationError(
                    "Walk-forward test blocks overlap; duplicate target dates: "
                    f"{list(duplicates[:5])}."
                )
            all_weights = rebalance_and_cap_turnover(
                all_targets, active_config.portfolio
            )

            # A target chosen on a rebalance bar becomes the executed position
            # on the following bar.
            schedule = compute_rebalance_dates(
                pd.DatetimeIndex(all_weights.index),
                active_config.portfolio.rebalance_frequency,
            )
            aligned_starts: list[pd.Timestamp] = []
            for window in fold_windows:
                candidates = schedule[
                    (schedule >= window.test[0]) & (schedule <= window.test[-1])
                ]
                if len(candidates) == 0:
                    raise InvalidConfigurationError(
                        f"Fold {window.fold}: its test block contains no rebalance "
                        "date; increase test_window or rebalance more often."
                    )
                location = all_weights.index.get_loc(candidates[0])
                if not isinstance(location, int):
                    raise InvalidConfigurationError(
                        f"Fold {window.fold}: rebalance date is not unique."
                    )
                execution_location = location + 1
                if (
                    execution_location >= len(all_weights.index)
                    or all_weights.index[execution_location] > window.test[-1]
                ):
                    raise InvalidConfigurationError(
                        f"Fold {window.fold}: no test return occurs after its first "
                        "rebalance; increase test_window or rebalance more often."
                    )
                aligned_starts.append(all_weights.index[execution_location])

            prices = price_matrix(tradable, adjusted=True)
            asset_returns = compute_asset_returns(prices).reindex(all_weights.index)
            execution_model = build_execution_from_config(active_config, data)
            executed_weights = all_weights
            if delay > 0:
                executed_weights = executed_weights.shift(delay).fillna(0.0)
            accounting = run_accounting(
                executed_weights,
                asset_returns,
                execution_model,
                active_config.initial_capital,
            )
            oos_returns = accounting.net_returns
            oos_equity = accounting.equity
            oos_result = self._build_oos_result(
                data,
                tradable,
                active_config,
                accounting,
                execution_model,
                all_weights,
                all_targets,
                periods_per_year,
                risk_free_rate,
                delay,
                started,
                grid,
                train_window,
                validation_window,
                test_window,
                expanding,
            )
        else:
            oos_returns = pd.Series(dtype=float)
            oos_equity = pd.Series(dtype=float)
            oos_result = None
            aligned_starts = []

        fold_results: list[FoldResult] = []
        for position, (window, parameters, score) in enumerate(
            zip(fold_windows, fold_parameters, fold_scores, strict=True)
        ):
            test_start = aligned_starts[position]
            if position + 1 < len(aligned_starts):
                next_location = oos_returns.index.get_loc(aligned_starts[position + 1])
                if not isinstance(next_location, int):
                    raise InvalidConfigurationError(
                        "Walk-forward reporting boundaries must be unique."
                    )
                test_end = oos_returns.index[next_location - 1]
            else:
                test_end = oos_returns.index[-1]
            fold_returns = oos_returns.loc[test_start:test_end]
            fold_equity = M.equity_from_returns(fold_returns)
            fold_sharpe = M.sharpe_ratio(fold_returns, risk_free_rate, periods_per_year)
            fold_results.append(
                FoldResult(
                    fold=window.fold,
                    best_params=parameters,
                    validation_score=score,
                    test_return=M.total_return(fold_equity),
                    test_sharpe=fold_sharpe,
                    test_returns=fold_returns,
                )
            )
            logger.info(
                "Fold %d: best=%s validation_score=%.3f test_sharpe=%.3f",
                window.fold,
                parameters,
                score,
                fold_sharpe,
            )

        return WalkForwardResult(fold_results, oos_returns, oos_equity, oos_result)

    def _build_oos_result(
        self,
        data: pd.DataFrame,
        tradable: pd.DataFrame,
        active_config: ExperimentConfig,
        accounting: AccountingResult,
        execution_model: ExecutionModel,
        all_weights: pd.DataFrame,
        all_targets: pd.DataFrame,
        periods_per_year: int,
        risk_free_rate: float,
        delay: int,
        started: float,
        grid: Mapping[str, Sequence[Any]],
        train_window: int,
        validation_window: int,
        test_window: int,
        expanding: bool,
    ) -> BacktestResult:
        """Build a genuine BacktestResult from the stitched OOS series.

        Reuses the exact trade-log/benchmark/metrics pipeline
        :class:`~quantlab.backtesting.engine.BacktestEngine` uses for a
        single backtest, so existing result-rendering code (dashboard, HTML
        report) works unchanged on a walk-forward's out-of-sample result.
        ``active_config`` is the config actually in effect for this result
        (``self.base_config`` from :meth:`run`/:meth:`run_with_weight_cache`,
        or a cost-scenario config from :meth:`rescore_with_costs`), so its
        cost fields and metadata describe what was actually run.
        """
        trades = build_trade_log(
            accounting.executed_weights,
            accounting.weight_changes,
            accounting.equity,
            price_matrix(tradable, adjusted=False),
            commission_bps=active_config.commission_bps,
            spread_bps=active_config.spread_bps,
            slippage_model=execution_model.slippage,
            slippage_equity=accounting.equity_for_costs,
        )
        benchmark_data = (
            data if active_config.benchmark_kind is BenchmarkKind.SYMBOL else tradable
        )
        benchmark_returns = build_benchmark(
            benchmark_data,
            pd.DatetimeIndex(accounting.net_returns.index),
            benchmark_symbol=active_config.benchmark_symbol,
            first_asset_symbol=active_config.symbols[0],
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            kind=str(active_config.benchmark_kind),
        )
        metrics = M.compute_metrics(
            accounting.net_returns,
            accounting.equity,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        metrics.update(portfolio_metrics_from_accounting(accounting, periods_per_year))
        metrics["number_of_trades"] = float(len(trades))
        metrics["average_trade_size"] = (
            float(trades["traded_notional"].mean()) if len(trades) else 0.0
        )
        metrics["total_cost_fraction"] = float(accounting.costs.total.sum())

        metadata: dict[str, Any] = {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "experiment_name": active_config.experiment_name,
            "strategy": active_config.strategy_name,
            "symbols": active_config.symbols,
            "start_date": str(active_config.start_date),
            "end_date": str(active_config.end_date),
            "random_seed": active_config.random_seed,
            "n_rows": len(accounting.net_returns),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "periods_per_year": periods_per_year,
            "data_hash": ParquetStorage.hash_frame(data),
            "git_commit": _git_commit_hash(),
            "git_dirty": _git_is_dirty(),
            "dependency_versions": _dependency_versions(),
            "code_hash": _source_hash(),
            "walk_forward_execution_delay": delay,
            # `metrics` (below) *are* the out-of-sample metrics — this result
            # is the stitched OOS series, not a full-sample fit. Mirrored
            # into metadata under this exact key because
            # `research_summary._oos_metrics()` and the HTML report's
            # methodology/hypothesis/conclusion text look for it there to
            # decide whether to describe evidence as out-of-sample; without
            # it, a genuinely-OOS walk-forward result was mislabelled
            # "full-sample only, no out-of-sample evidence attached".
            "walk_forward_oos_metrics": metrics,
            "walk_forward_parameter_grid": dict(grid),
            "walk_forward_windows": {
                "train_window": train_window,
                "validation_window": validation_window,
                "test_window": test_window,
                "expanding": expanding,
            },
        }

        return BacktestResult(
            config=active_config,
            equity_curve=accounting.equity,
            returns=accounting.net_returns,
            benchmark_returns=benchmark_returns,
            positions=accounting.executed_weights,
            weights=all_weights,
            target_weights=all_targets,
            signals=pd.DataFrame(),
            trades=trades,
            costs=accounting.costs.to_frame(),
            metrics=metrics,
            metadata=metadata,
            gross_returns=accounting.gross_returns,
            gross_equity=accounting.gross_equity,
            turnover=accounting.turnover,
        )

    def _select_on_validation(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        combinations: list[dict[str, Any]],
        scorer: Callable[[pd.Series, pd.Series, int, float], float],
        periods_per_year: int,
        risk_free_rate: float,
        *,
        execution_delay: int = 0,
        on_candidate: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Select the finite, highest-scoring parameter combination.

        ``on_candidate``, if given, is called once after each candidate is
        considered (including ones skipped for insufficient warm-up), for
        finer-grained progress reporting than one tick per fold — see
        :meth:`_select_and_capture`, which does the same.
        """
        best_score = -np.inf
        best_parameters: dict[str, Any] | None = None
        insufficient_warmup = 0
        for combination in combinations:
            config = _with_params(self.base_config, combination)
            required = _minimum_observations_for_executable_weight(config)
            available = len(window.train) + len(window.validation)
            if len(window.validation) < 2 or available < required:
                insufficient_warmup += 1
                logger.debug(
                    "Fold %d: skipping %s because validation ends after %d "
                    "observations but this configuration needs at least %d "
                    "before an executable weight can be scored.",
                    window.fold,
                    combination,
                    available,
                    required,
                )
                if on_candidate is not None:
                    on_candidate()
                continue
            returns = _evaluate_fresh_from_window_start(
                data,
                config,
                window.train[0],
                window.validation[0],
                window.validation[-1],
                execution_delay=execution_delay,
            )
            equity = M.equity_from_returns(returns)
            score = scorer(returns, equity, periods_per_year, risk_free_rate)
            if on_candidate is not None:
                on_candidate()
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_parameters = combination
        if best_parameters is None:
            if combinations and insufficient_warmup == len(combinations):
                raise InvalidConfigurationError(
                    f"Fold {window.fold}: no parameter combination has enough "
                    "train/validation history to produce an executable weight "
                    "inside the validation block. Increase the windows or use "
                    "shorter strategy warm-up parameters."
                )
            raise InvalidConfigurationError(
                f"Fold {window.fold}: every parameter combination produced a "
                "non-finite validation score."
            )
        return {"params": best_parameters, "score": float(best_score)}

    def _select_and_capture(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        combinations: list[dict[str, Any]],
        scorer: Callable[[pd.Series, pd.Series, int, float], float],
        periods_per_year: int,
        risk_free_rate: float,
        *,
        execution_delay: int = 0,
        on_candidate: Callable[[], None] | None = None,
    ) -> tuple[int, float, list[_FoldCandidateWeights]]:
        """Capture every candidate's weights alongside selection.

        Like :meth:`_select_on_validation`, but also captures every
        candidate's weights into a :class:`WalkForwardWeightCache` row.
        Test-block target weights are computed for every candidate, not
        only the fold's winner, since a different candidate can win once a
        cost-only scenario re-scores this fold with
        :meth:`rescore_with_costs`. ``on_candidate``, if given, is called
        once after each candidate is considered (including ones skipped for
        insufficient warm-up), for finer-grained progress reporting than
        one tick per fold.
        """
        best_score = -np.inf
        best_index: int | None = None
        insufficient_warmup = 0
        captured: list[_FoldCandidateWeights] = []
        for index, combination in enumerate(combinations):
            config = _with_params(self.base_config, combination)
            required = _minimum_observations_for_executable_weight(config)
            available = len(window.train) + len(window.validation)
            if len(window.validation) < 2 or available < required:
                insufficient_warmup += 1
                logger.debug(
                    "Fold %d: skipping %s because validation ends after %d "
                    "observations but this configuration needs at least %d "
                    "before an executable weight can be scored.",
                    window.fold,
                    combination,
                    available,
                    required,
                )
                captured.append(_FoldCandidateWeights(None, None))
                if on_candidate is not None:
                    on_candidate()
                continue
            validation_weights, returns = _weights_and_returns_for_validation(
                data,
                config,
                window.train[0],
                window.validation[0],
                window.validation[-1],
                execution_delay=execution_delay,
            )
            equity = M.equity_from_returns(returns)
            score = scorer(returns, equity, periods_per_year, risk_free_rate)
            test_targets = _target_weights_for_window(
                data, config, window.train[0], window.test[0], window.test[-1]
            )
            captured.append(_FoldCandidateWeights(validation_weights, test_targets))
            if on_candidate is not None:
                on_candidate()
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            if combinations and insufficient_warmup == len(combinations):
                raise InvalidConfigurationError(
                    f"Fold {window.fold}: no parameter combination has enough "
                    "train/validation history to produce an executable weight "
                    "inside the validation block. Increase the windows or use "
                    "shorter strategy warm-up parameters."
                )
            raise InvalidConfigurationError(
                f"Fold {window.fold}: every parameter combination produced a "
                "non-finite validation score."
            )
        return best_index, float(best_score), captured

    def _weights_on_test(
        self, data: pd.DataFrame, window: WalkForwardWindow, params: dict[str, Any]
    ) -> pd.DataFrame:
        """Return constrained test targets before scheduling and turnover."""
        config = _with_params(self.base_config, params)
        return _target_weights_for_window(
            data, config, window.train[0], window.test[0], window.test[-1]
        )


def _validate_parameter_grid(
    grid: object, strategy_name: str
) -> dict[str, Sequence[Any]]:
    """Validate grid shape and parameter names before running backtests."""
    if not isinstance(grid, Mapping):
        raise InvalidConfigurationError("parameter_grid must be a mapping.")
    accepted = strategy_parameter_names(strategy_name)
    validated: dict[str, Sequence[Any]] = {}
    for name, values in grid.items():
        if not isinstance(name, str) or not name:
            raise InvalidConfigurationError(
                "Parameter-grid names must be non-empty strings."
            )
        if name not in accepted:
            raise InvalidConfigurationError(
                f"Unknown parameter {name!r} for strategy {strategy_name!r}."
            )
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise InvalidConfigurationError(
                f"Grid parameter {name!r} must contain a sequence of values."
            )
        if _contains_duplicate_candidates(values):
            raise InvalidConfigurationError(
                f"Grid parameter {name!r} must not contain duplicate values."
            )
        validated[name] = values
    return validated


def _contains_duplicate_candidates(values: Sequence[Any]) -> bool:
    """Detect repeated candidates without requiring hashable values."""
    for position, value in enumerate(values):
        for previous in values[:position]:
            equal = value == previous
            if isinstance(equal, (np.ndarray, pd.Series)):
                if bool(np.asarray(equal).all()):
                    return True
            elif bool(equal):
                return True
    return False


def _grid_combinations(
    grid: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """Expand a validated parameter grid into concrete combinations."""
    if not grid:
        return [{}]
    empty = [name for name, values in grid.items() if not values]
    if empty:
        raise InvalidConfigurationError(
            f"Parameter grid entries {empty} have no candidate values."
        )
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*grid.values())
    ]


def _slice_between(
    data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Return rows whose timestamps lie in the inclusive interval."""
    timestamps = pd.to_datetime(data["timestamp"])
    mask = (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))
    return data.loc[mask].reset_index(drop=True)


def _weights_for_window(
    data: pd.DataFrame,
    config: ExperimentConfig,
    lookback_start: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    execution_delay: int = 0,
) -> pd.DataFrame:
    """Return held weights after supplying the required lookback history."""
    sliced = _slice_between(data, lookback_start, window_end)
    result = run_backtest_from_config(sliced, config, execution_delay=execution_delay)
    return result.weights.loc[window_start:window_end]


def _target_weights_for_window(
    data: pd.DataFrame,
    config: ExperimentConfig,
    lookback_start: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return constrained targets before scheduling and turnover capping."""
    sliced = _slice_between(data, lookback_start, window_end)
    result = run_backtest_from_config(sliced, config)
    if result.target_weights is None:
        raise InvalidConfigurationError("Backtest did not expose target weights.")
    return result.target_weights.loc[window_start:window_end]


def _weights_and_returns_for_validation(
    data: pd.DataFrame,
    config: ExperimentConfig,
    lookback_start: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    execution_delay: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return one candidate's (held weights, net returns) on a block.

    Split out of the former ``_evaluate_fresh_from_window_start`` so
    ``WalkForwardValidator._select_and_capture`` can retain the weights
    instead of discarding them, without duplicating this accounting logic.
    """
    sliced = _slice_between(data, lookback_start, window_end)
    window_weights = _weights_for_window(
        data,
        config,
        lookback_start,
        window_start,
        window_end,
        execution_delay=execution_delay,
    )
    tradable = sliced[sliced[SYMBOL].isin(set(config.symbols))]
    prices = price_matrix(tradable, adjusted=True)
    asset_returns = compute_asset_returns(prices).loc[window_start:window_end]
    execution_model = build_execution_from_config(config, sliced)
    accounting = run_accounting(
        window_weights, asset_returns, execution_model, config.initial_capital
    )
    return window_weights, accounting.net_returns


def _evaluate_fresh_from_window_start(
    data: pd.DataFrame,
    config: ExperimentConfig,
    lookback_start: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    execution_delay: int = 0,
) -> pd.Series:
    """Return independently accounted net returns for candidate scoring."""
    _, net_returns = _weights_and_returns_for_validation(
        data,
        config,
        lookback_start,
        window_start,
        window_end,
        execution_delay=execution_delay,
    )
    return net_returns


def _minimum_observations_for_executable_weight(config: ExperimentConfig) -> int:
    """Return the structural warm-up needed to score one executed weight.

    This only detects combinations that cannot possibly leave warm-up. It does
    not inspect whether a sufficiently warmed-up strategy actually chooses to
    trade, so a legitimate flat validation result remains a valid score.
    Unknown registered strategies and allocators default to no inferred warm-up.
    """
    parameters = build_strategy_from_config(config).parameters()
    strategy_name = config.strategy_name

    signal_observations = 1
    if strategy_name in {"time_series_momentum", "cross_sectional_momentum"}:
        lookback = int(parameters["lookback_period"])
        signal_observations = lookback + 1
        if (
            strategy_name == "time_series_momentum"
            and parameters["signal_scaling"] == "continuous"
        ):
            signal_observations = lookback + min(20, lookback)
        elif (
            strategy_name == "time_series_momentum"
            and parameters["signal_scaling"] == "volatility_adjusted"
        ):
            signal_observations = max(
                signal_observations, int(parameters["volatility_window"]) + 1
            )
    elif strategy_name == "mean_reversion":
        signal_observations = int(parameters["lookback_period"])
    elif strategy_name == "trend_following":
        signal_observations = int(parameters["slow_window"])
    elif strategy_name == "pairs_trading":
        signal_observations = int(parameters["formation_window"]) + int(
            parameters["zscore_window"]
        )

    allocator_observations = 1
    if config.portfolio.allocator in {"inverse_volatility", "volatility_targeting"}:
        allocator_observations = config.portfolio.volatility_window + 1

    # Accounting shifts target weights by one bar before applying returns.
    return max(signal_observations, allocator_observations) + 1
