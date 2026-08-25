"""Two-dimensional strategy-parameter sensitivity analysis."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError, QuantLabError
from quantlab.logging_config import get_logger
from quantlab.strategies import strategy_sweepable_parameter_names
from quantlab.validation.checkpoint import (
    clear_checkpoint,
    compute_provenance,
    load_checkpoint,
    save_checkpoint,
)

logger = get_logger(__name__)


def _with_two_params(
    config: ExperimentConfig, name_x: str, x: Any, name_y: str, y: Any
) -> ExperimentConfig:
    """Return a revalidated config with two strategy parameters overridden."""
    merged = {**config.strategy.parameters, name_x: x, name_y: y}
    strategy = config.strategy.revalidated_copy(update={"parameters": merged})
    return config.revalidated_copy(update={"strategy": strategy})


def run_parameter_sensitivity(
    data: pd.DataFrame,
    base_config: ExperimentConfig,
    parameter_x: str,
    values_x: list[Any],
    parameter_y: str,
    values_y: list[Any],
) -> pd.DataFrame:
    """Run a two-parameter sweep and return one row per combination.

    Expected configuration failures remain visible as rows with
    ``status='failed'`` and an error message. Unexpected programming errors
    still propagate.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(base_config, ExperimentConfig):
        raise TypeError("base_config must be an ExperimentConfig.")
    _validate_parameter_axes(base_config, parameter_x, values_x, parameter_y, values_y)

    rows: list[dict[str, Any]] = []
    for x in values_x:
        for y in values_y:
            row: dict[str, Any] = {parameter_x: x, parameter_y: y}
            try:
                config = _with_two_params(base_config, parameter_x, x, parameter_y, y)
                result = run_backtest_from_config(data, config)
            except (QuantLabError, ValidationError) as exc:
                logger.warning(
                    "Sensitivity combination %s=%s, %s=%s failed: %s",
                    parameter_x,
                    x,
                    parameter_y,
                    y,
                    exc,
                )
                row.update(
                    {
                        "sharpe": float("nan"),
                        "cagr": float("nan"),
                        "max_drawdown": float("nan"),
                        "turnover": float("nan"),
                        "num_trades": float("nan"),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            else:
                row.update(
                    {
                        "sharpe": result.metrics.get("sharpe_ratio", float("nan")),
                        "cagr": result.metrics.get("cagr", float("nan")),
                        "max_drawdown": result.metrics.get(
                            "max_drawdown", float("nan")
                        ),
                        "turnover": result.metrics.get("annual_turnover", float("nan")),
                        "num_trades": result.metrics.get("number_of_trades", 0.0),
                        "status": "ok",
                        "error": None,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


_SENSITIVITY_CELL_METRIC_COLUMNS = (
    "sharpe",
    "cagr",
    "max_drawdown",
    "turnover",
    "num_trades",
)


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two candidate values without requiring them to be scalar.

    ``a`` comes from an untrusted checkpoint, so ``a == b`` itself is not
    safe to trust blindly: a scalar sentinel like ``pd.NA`` compares equal
    to nothing (``bool(pd.NA)`` raises ``TypeError: boolean value of NA is
    ambiguous`` rather than returning ``False``), and an arbitrary corrupted
    value could make ``==`` raise outright (e.g. comparing against a type
    that doesn't support it). Either way, that only means "not a match" --
    it must never propagate out and abort a checkpoint validation that
    otherwise exists precisely to catch corrupted input like this.
    """
    try:
        equal = a == b
        if isinstance(equal, (np.ndarray, pd.Series)):
            return bool(np.asarray(equal).all())
        return bool(equal)
    except Exception:
        return False


def _sensitivity_cell_row_is_consistent(row: dict[str, Any]) -> bool:
    """Return whether a checkpointed cell's status, metrics and error agree.

    Mirrors the two shapes the loop below actually produces: ``status ==
    "ok"`` means every metric is a finite number and ``error`` is ``None``;
    ``status == "failed"`` means every metric is NaN and ``error`` is a real
    (non-empty) message. A cell claiming success while carrying NaN metrics,
    or failure while carrying finite ones and no error, is corrupted
    regardless of whether its schema and parameter values already checked
    out.
    """
    status = row.get("status")
    error = row.get("error")
    metric_values = [row.get(name) for name in _SENSITIVITY_CELL_METRIC_COLUMNS]
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


def run_walk_forward_parameter_sensitivity(
    data: pd.DataFrame,
    base_config: ExperimentConfig,
    parameter_x: str,
    values_x: list[Any],
    parameter_y: str,
    values_y: list[Any],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Two-parameter sweep where each cell re-runs the full walk-forward process.

    Unlike :func:`run_parameter_sensitivity`, each ``(x, y)`` combination is
    not evaluated as a single plain backtest: it is pinned as the *only*
    candidate in a fresh :class:`~quantlab.validation.walk_forward.
    WalkForwardValidator` run (all folds, OOS reconstruction), scored on that
    run's out-of-sample metrics. This keeps Walk-forward mode's sensitivity
    heatmap genuinely walk-forward-derived rather than silently reusing plain
    single-backtest numbers.  Train/validation/test windows and expanding
    mode come from ``base_config.validation``
    (:func:`~quantlab.validation.walk_forward.resolve_walk_forward_windows`).

    Args:
        data: Canonical long OHLCV frame.
        base_config: Validated experiment config; its own strategy
            parameters supply every field not swept by parameter_x/y.
        parameter_x: First swept strategy-parameter name (x-axis).
        values_x: Candidate values for parameter_x.
        parameter_y: Second swept strategy-parameter name (y-axis).
        values_y: Candidate values for parameter_y.
        on_progress: Optional callback invoked as ``on_progress(done, total)``
            once before the first cell (or the resumed count, see
            ``checkpoint_path``) and once after each cell completes — each
            cell here is itself a full walk-forward run, so this reports
            coarser, cell-level progress, not individual fold progress.
        checkpoint_path: Optional path to persist per-cell progress to, so
            an interrupted sweep resumes from its last completed cell
            instead of starting over. See ``quantlab.validation.checkpoint``.
    """
    from quantlab.validation.walk_forward import (
        WalkForwardValidator,
        resolve_walk_forward_windows,
    )

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(base_config, ExperimentConfig):
        raise TypeError("base_config must be an ExperimentConfig.")
    _validate_parameter_axes(base_config, parameter_x, values_x, parameter_y, values_y)
    train_window, validation_window, test_window = resolve_walk_forward_windows(
        base_config
    )
    expanding = base_config.validation.expanding

    # A flat list, not the nested loop below directly, so a resume can slice
    # off however many cells a checkpoint already has rows for.
    combinations = [(x, y) for x in values_x for y in values_y]
    total_cells = len(combinations)

    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] | None = None
    completed_cells = 0

    expected_cell_keys = {
        parameter_x,
        parameter_y,
        *_SENSITIVITY_CELL_METRIC_COLUMNS,
        "status",
        "error",
    }

    def _validate_cell_state(state: Any, progress: int) -> bool:
        # One row per cell, in lockstep with `progress` (unlike the
        # scenario-block checkpoints elsewhere, where one unit can append
        # several rows at once) -- so an exact length match is meaningful
        # here, not just an upper bound. A structurally-plausible but
        # incoherent checkpoint (right length, wrong content -- e.g. a
        # single-cell state of `["garbage"]`, or a row carrying some other
        # cell's parameter values) must never be resumed from: it would
        # silently corrupt the sweep with a mismatched or malformed row.
        if not (0 <= progress <= total_cells and isinstance(state, list)):
            return False
        if len(state) != progress:
            return False
        if not all(
            isinstance(row, dict) and row.keys() == expected_cell_keys for row in state
        ):
            return False
        # Each row's own (x, y) must match the combination actually assigned
        # to its position -- the same order the loop below resumes from
        # (`combinations[completed_cells:]`), so a resumed row can never be
        # silently attributed to the wrong cell.
        for row, (x, y) in zip(state, combinations[:progress], strict=True):
            x_matches = _values_equal(row[parameter_x], x)
            y_matches = _values_equal(row[parameter_y], y)
            if not (x_matches and y_matches):
                return False
        return all(_sensitivity_cell_row_is_consistent(row) for row in state)

    if checkpoint_path is not None:
        provenance = compute_provenance(
            base_config,
            data,
            parameter_x=parameter_x,
            values_x=list(values_x),
            parameter_y=parameter_y,
            values_y=list(values_y),
        )
        checkpoint_result = load_checkpoint(
            checkpoint_path, provenance, validate=_validate_cell_state
        )
        if checkpoint_result is not None:
            rows, completed_cells = checkpoint_result
            logger.info(
                "Resuming walk-forward sensitivity from checkpoint: %d/%d "
                "cells already done.",
                completed_cells,
                total_cells,
            )
    if on_progress is not None:
        on_progress(completed_cells, total_cells)

    for x, y in combinations[completed_cells:]:
        row: dict[str, Any] = {parameter_x: x, parameter_y: y}
        try:
            config = _with_two_params(base_config, parameter_x, x, parameter_y, y)
            wf = WalkForwardValidator(config).run(
                data,
                # No inner grid: x/y are pinned, so this cell measures
                # exactly that combination's walk-forward OOS behaviour,
                # not a further optimization on top of it.
                parameter_grid={},
                train_window=train_window,
                validation_window=validation_window,
                test_window=test_window,
                expanding=expanding,
            )
            if wf.oos_result is None:
                raise InvalidConfigurationError(
                    "No walk-forward fold fit this parameter combination."
                )
        except (QuantLabError, ValidationError) as exc:
            logger.warning(
                "Walk-forward sensitivity combination %s=%s, %s=%s failed: %s",
                parameter_x,
                x,
                parameter_y,
                y,
                exc,
            )
            row.update(
                {
                    "sharpe": float("nan"),
                    "cagr": float("nan"),
                    "max_drawdown": float("nan"),
                    "turnover": float("nan"),
                    "num_trades": float("nan"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
        else:
            metrics = wf.oos_result.metrics
            row.update(
                {
                    "sharpe": metrics.get("sharpe_ratio", float("nan")),
                    "cagr": metrics.get("cagr", float("nan")),
                    "max_drawdown": metrics.get("max_drawdown", float("nan")),
                    "turnover": metrics.get("annual_turnover", float("nan")),
                    "num_trades": metrics.get("number_of_trades", 0.0),
                    "status": "ok",
                    "error": None,
                }
            )
        rows.append(row)
        completed_cells += 1
        if checkpoint_path is not None and provenance is not None:
            save_checkpoint(checkpoint_path, provenance, rows, completed_cells)
        if on_progress is not None:
            on_progress(completed_cells, total_cells)

    if checkpoint_path is not None:
        clear_checkpoint(checkpoint_path)
    return pd.DataFrame(rows)


def sensitivity_heatmap_data(
    sensitivity: pd.DataFrame,
    parameter_x: str,
    parameter_y: str,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Pivot successful sensitivity rows into a ``y by x`` metric matrix."""
    if not isinstance(sensitivity, pd.DataFrame):
        raise TypeError("sensitivity must be a pandas DataFrame.")
    required = {parameter_x, parameter_y, metric}
    missing = required.difference(sensitivity.columns)
    if missing:
        raise ValueError(f"sensitivity is missing columns: {sorted(missing)}.")
    selected = sensitivity
    if "status" in selected.columns:
        selected = selected[selected["status"] == "ok"]
    if selected.empty:
        raise ValueError("sensitivity contains no successful combinations.")
    if selected.duplicated([parameter_x, parameter_y]).any():
        raise ValueError("sensitivity contains duplicate parameter combinations.")
    metric_values = pd.to_numeric(selected[metric], errors="coerce")
    if np.isinf(metric_values.to_numpy(dtype=float)).any():
        raise ValueError(f"{metric} must not contain infinite values.")
    selected = selected.assign(**{metric: metric_values})
    return selected.pivot(index=parameter_y, columns=parameter_x, values=metric)


#: Columns a sensitivity result always carries besides its two swept
#: parameters -- whatever's left after excluding these is the axis pair.
_SENSITIVITY_METRIC_COLUMNS = frozenset(
    {"sharpe", "cagr", "max_drawdown", "turnover", "num_trades", "status", "error"}
)


def infer_sensitivity_parameter_columns(sensitivity: pd.DataFrame) -> tuple[str, str]:
    """Return the two swept-parameter columns a sensitivity result carries.

    A sensitivity DataFrame self-describes which two parameters it was
    computed for (whatever columns aren't one of the fixed metric columns);
    reading them back off the DataFrame itself, rather than trusting a
    caller's separately-tracked "current" axis selection, is what keeps a
    live UI (e.g. the dashboard) from rendering a stale result under axis
    labels that no longer match what was actually computed -- the picker
    widgets can drift after the run without the displayed result becoming
    wrong or crashing on a missing column.
    """
    parameter_columns = [
        column
        for column in sensitivity.columns
        if column not in _SENSITIVITY_METRIC_COLUMNS
    ]
    if len(parameter_columns) != 2:
        raise ValueError(
            "sensitivity must have exactly two swept-parameter columns; found "
            f"{parameter_columns}."
        )
    parameter_x, parameter_y = parameter_columns
    return parameter_x, parameter_y


def _validate_parameter_axes(
    config: ExperimentConfig,
    parameter_x: object,
    values_x: object,
    parameter_y: object,
    values_y: object,
) -> None:
    """Validate sweep names and candidate collections."""
    names = (parameter_x, parameter_y)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter names must be non-empty strings.")
    assert isinstance(parameter_x, str)
    assert isinstance(parameter_y, str)
    names = (parameter_x, parameter_y)
    if parameter_x == parameter_y:
        raise ValueError("parameter_x and parameter_y must be different.")
    accepted = strategy_sweepable_parameter_names(config.strategy_name)
    unknown = set(names).difference(accepted)
    if unknown:
        raise ValueError(
            f"Unknown or unsweepable parameter(s) for strategy "
            f"{config.strategy_name!r}: {sorted(unknown)}. Boolean/structural "
            "parameters (e.g. long_only, long_short) cannot be swept — they "
            "change which other parameters are even meaningful, so "
            "sensitivity treats them as fixed, matching the default "
            f"walk-forward grid. Accepted parameters: {sorted(accepted)}."
        )
    for name, values in ((parameter_x, values_x), (parameter_y, values_y)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"Values for {name!r} must be a sequence.")
        if not values:
            raise ValueError(f"Values for {name!r} must not be empty.")
        if _contains_duplicates(values):
            raise ValueError(f"Values for {name!r} must not contain duplicates.")
        if any(isinstance(value, bool) for value in values):
            raise ValueError(
                f"Values for {name!r} must not be boolean — sensitivity "
                "sweeps numeric or categorical candidates only."
            )


def _contains_duplicates(values: Sequence[Any]) -> bool:
    """Compare candidate values without requiring them to be hashable."""
    for position, value in enumerate(values):
        for previous in values[:position]:
            equal = value == previous
            if isinstance(equal, (np.ndarray, pd.Series)):
                if bool(np.asarray(equal).all()):
                    return True
            elif bool(equal):
                return True
    return False
