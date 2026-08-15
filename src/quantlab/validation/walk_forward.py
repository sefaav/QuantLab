"""Walk-forward parameter selection and out-of-sample evaluation."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from quantlab.backtesting.accounting import compute_asset_returns, run_accounting
from quantlab.backtesting.runner import (
    build_execution_from_config,
    build_strategy_from_config,
    run_backtest_from_config,
)
from quantlab.config import ExperimentConfig
from quantlab.constants import SYMBOL
from quantlab.data.base import price_matrix
from quantlab.exceptions import InvalidConfigurationError
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
    ) -> WalkForwardResult:
        """Execute chronological parameter selection and OOS accounting."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        required_columns = {"timestamp", SYMBOL}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise InvalidConfigurationError(
                f"data is missing required columns: {sorted(missing_columns)}."
            )
        expanding = boolean(expanding, name="expanding")
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

        fold_windows: list[WalkForwardWindow] = []
        fold_parameters: list[dict[str, Any]] = []
        fold_scores: list[float] = []
        target_pieces: list[pd.DataFrame] = []
        for window in windows:
            best = self._select_on_validation(
                data,
                window,
                combinations,
                scorer,
                periods_per_year,
                risk_free_rate,
            )
            targets = self._weights_on_test(data, window, best["params"])
            fold_windows.append(window)
            fold_parameters.append(best["params"])
            fold_scores.append(best["score"])
            target_pieces.append(targets)

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
                all_targets, self.base_config.portfolio
            )

            # A target chosen on a rebalance bar becomes the executed position
            # on the following bar.
            schedule = compute_rebalance_dates(
                pd.DatetimeIndex(all_weights.index),
                self.base_config.portfolio.rebalance_frequency,
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
            execution_model = build_execution_from_config(self.base_config, data)
            accounting = run_accounting(
                all_weights,
                asset_returns,
                execution_model,
                self.base_config.initial_capital,
            )
            oos_returns = accounting.net_returns
            oos_equity = accounting.equity
        else:
            oos_returns = pd.Series(dtype=float)
            oos_equity = pd.Series(dtype=float)
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

        return WalkForwardResult(fold_results, oos_returns, oos_equity)

    def _select_on_validation(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        combinations: list[dict[str, Any]],
        scorer: Callable[[pd.Series, pd.Series, int, float], float],
        periods_per_year: int,
        risk_free_rate: float,
    ) -> dict[str, Any]:
        """Select the finite, highest-scoring parameter combination."""
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
                continue
            returns = _evaluate_fresh_from_window_start(
                data,
                config,
                window.train[0],
                window.validation[0],
                window.validation[-1],
            )
            equity = M.equity_from_returns(returns)
            score = scorer(returns, equity, periods_per_year, risk_free_rate)
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
) -> pd.DataFrame:
    """Return held weights after supplying the required lookback history."""
    sliced = _slice_between(data, lookback_start, window_end)
    result = run_backtest_from_config(sliced, config)
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


def _evaluate_fresh_from_window_start(
    data: pd.DataFrame,
    config: ExperimentConfig,
    lookback_start: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.Series:
    """Return independently accounted net returns for candidate scoring."""
    sliced = _slice_between(data, lookback_start, window_end)
    window_weights = _weights_for_window(
        data, config, lookback_start, window_start, window_end
    )
    tradable = sliced[sliced[SYMBOL].isin(set(config.symbols))]
    prices = price_matrix(tradable, adjusted=True)
    asset_returns = compute_asset_returns(prices).loc[window_start:window_end]
    execution_model = build_execution_from_config(config, sliced)
    accounting = run_accounting(
        window_weights, asset_returns, execution_model, config.initial_capital
    )
    return accounting.net_returns


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
