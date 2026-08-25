"""Chronological holdout metrics for a completed backtest.

The report slices one continuously simulated return series. This preserves
historical warm-up and portfolio state across split boundaries; it is not the
same experiment as restarting from cash inside every block. The test block is
out of sample only when its parameters were fixed without using that block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlab.backtesting.result import BacktestResult
from quantlab.config import ExperimentConfig
from quantlab.constants import SYMBOL
from quantlab.data.base import price_matrix
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import InvalidConfigurationError
from quantlab.logging_config import get_logger
from quantlab.risk import metrics as M
from quantlab.validation.splits import ChronologicalSplit, chronological_split

logger = get_logger(__name__)


@dataclass(frozen=True)
class HoldoutReport:
    """Metrics, dates and test series for a chronological holdout."""

    train_metrics: dict[str, float]
    validation_metrics: dict[str, float] | None
    test_metrics: dict[str, float]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_returns: pd.Series = field(repr=False)
    test_equity: pd.Series = field(repr=False)

    def __post_init__(self) -> None:
        """Detach mutable inputs from their caller-owned containers."""
        object.__setattr__(self, "train_metrics", dict(self.train_metrics))
        if self.validation_metrics is not None:
            object.__setattr__(
                self, "validation_metrics", dict(self.validation_metrics)
            )
        object.__setattr__(self, "test_metrics", dict(self.test_metrics))
        object.__setattr__(self, "test_returns", self.test_returns.copy())
        object.__setattr__(self, "test_equity", self.test_equity.copy())

    @property
    def has_validation_block(self) -> bool:
        """Return whether a usable validation block was computed."""
        return self.validation_metrics is not None

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-serialisable summary."""
        metadata: dict[str, object] = {
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "train_period": [self.train_start.isoformat(), self.train_end.isoformat()],
            "test_period": [self.test_start.isoformat(), self.test_end.isoformat()],
        }
        if self.has_validation_block:
            assert self.validation_start is not None
            assert self.validation_end is not None
            metadata["validation_metrics"] = self.validation_metrics
            metadata["validation_period"] = [
                self.validation_start.isoformat(),
                self.validation_end.isoformat(),
            ]
        return metadata

    def summary_table(self) -> pd.DataFrame:
        """Return train, optional validation and OOS test metrics."""
        blocks: list[tuple[str, dict[str, float], pd.Timestamp, pd.Timestamp]] = [
            ("Train", self.train_metrics, self.train_start, self.train_end)
        ]
        if self.has_validation_block:
            assert self.validation_metrics is not None
            assert self.validation_start is not None
            assert self.validation_end is not None
            blocks.append(
                (
                    "Validation",
                    self.validation_metrics,
                    self.validation_start,
                    self.validation_end,
                )
            )
        # Labeled plainly "Test", not "out-of-sample": whether this block is
        # genuinely OOS depends on parameters having been fixed *before*
        # looking at it, which is a property of the user's own workflow,
        # not something this table can verify (see the module docstring).
        blocks.append(("Test", self.test_metrics, self.test_start, self.test_end))
        return pd.DataFrame(
            [
                {
                    "Block": label,
                    "Start": start.date().isoformat(),
                    "End": end.date().isoformat(),
                    "CAGR": metrics.get("cagr", float("nan")),
                    "Sharpe": metrics.get("sharpe_ratio", float("nan")),
                    "Max Drawdown": metrics.get("max_drawdown", float("nan")),
                }
                for label, metrics, start, end in blocks
            ]
        )


def compute_holdout_split(
    data: pd.DataFrame, config: ExperimentConfig
) -> ChronologicalSplit | None:
    """Return the configured tradable-universe split, if usable."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig.")
    if SYMBOL not in data.columns:
        raise InvalidConfigurationError(f"data must contain a {SYMBOL!r} column.")

    test_ratio = config.validation.test_ratio or 0.0
    if test_ratio <= 0.0:
        return None
    validation_ratio = config.validation.validation_ratio or 0.0
    train_ratio = 1.0 - validation_ratio - test_ratio
    if train_ratio <= 0.0:
        logger.warning("Holdout ratios leave no training block; skipping holdout.")
        return None

    # The benchmark may follow a different calendar and must not move the
    # tradable universe's split boundaries.
    tradable = data[data[SYMBOL].isin(set(config.symbols))]
    if tradable.empty:
        raise InvalidConfigurationError(
            "data contains none of the configured tradable symbols."
        )
    index = pd.DatetimeIndex(price_matrix(tradable, adjusted=True).index)
    try:
        return chronological_split(index, train_ratio, validation_ratio, test_ratio)
    except InvalidConfigurationError as exc:
        logger.warning("Holdout split is unusable: %s", exc)
        return None


def _block_metrics(
    result: BacktestResult,
    config: ExperimentConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    """Return metrics, returns and indexed equity for one block."""
    block_returns = result.returns.loc[start:end]
    if len(block_returns) < 2:
        return {}, block_returns, pd.Series(dtype=float)
    benchmark = (
        result.benchmark_returns.loc[start:end]
        if result.benchmark_returns is not None
        else None
    )
    equity = M.equity_from_returns(block_returns, preserve_index=True)
    metrics = M.compute_metrics(
        block_returns,
        equity,
        benchmark_returns=benchmark,
        risk_free_rate=config.backtest.risk_free_rate,
        periods_per_year=config.periods_per_year,
    )
    metrics["n_periods"] = float(len(block_returns))
    return metrics, block_returns, equity


def run_holdout_report(
    data: pd.DataFrame, config: ExperimentConfig, result: BacktestResult
) -> HoldoutReport | None:
    """Compute train, validation and test metrics from one continuous run."""
    _validate_result_inputs(data, config, result)
    split = compute_holdout_split(data, config)
    if split is None:
        return None
    if len(split.train) < 2 or len(split.test) < 2:
        logger.warning("Holdout train or test block has fewer than two rows.")
        return None
    if 0 < len(split.validation) < 2:
        logger.warning("Holdout validation block has fewer than two rows.")
        return None

    train_metrics, _, _ = _block_metrics(
        result, config, split.train[0], split.train[-1]
    )
    if not train_metrics:
        return None

    validation_metrics: dict[str, float] | None
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None
    if len(split.validation) >= 2:
        validation_metrics, _, _ = _block_metrics(
            result, config, split.validation[0], split.validation[-1]
        )
        validation_start = split.validation[0]
        validation_end = split.validation[-1]
    else:
        validation_metrics = None
        validation_start = None
        validation_end = None

    test_metrics, test_returns, test_equity = _block_metrics(
        result, config, split.test[0], split.test[-1]
    )
    if not test_metrics:
        return None

    logger.info(
        "Holdout split: train %s..%s, validation %s, test %s..%s.",
        split.train[0].date(),
        split.train[-1].date(),
        f"{validation_start.date()}..{validation_end.date()}"
        if validation_start is not None and validation_end is not None
        else "none",
        split.test[0].date(),
        split.test[-1].date(),
    )
    return HoldoutReport(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        train_start=split.train[0],
        train_end=split.train[-1],
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=split.test[0],
        test_end=split.test[-1],
        test_returns=test_returns,
        test_equity=test_equity,
    )


def _validate_result_inputs(
    data: pd.DataFrame, config: ExperimentConfig, result: BacktestResult
) -> None:
    """Ensure the report combines artefacts from the same experiment."""
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult.")
    if result.config.model_dump(mode="json") != config.model_dump(mode="json"):
        raise InvalidConfigurationError(
            "result.config does not match the holdout configuration."
        )
    expected_hash = result.metadata.get("data_hash")
    if not isinstance(expected_hash, str):
        raise InvalidConfigurationError("result metadata has no valid data_hash.")
    if ParquetStorage.hash_frame(data) != expected_hash:
        raise InvalidConfigurationError(
            "data does not match the frame used to produce result."
        )


def run_holdout_validation(
    data: pd.DataFrame, config: ExperimentConfig, result: BacktestResult
) -> dict[str, float]:
    """Return test-block metrics for callers that do not need the full report."""
    report = run_holdout_report(data, config, result)
    return report.test_metrics if report is not None else {}
