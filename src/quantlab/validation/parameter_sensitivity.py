"""Two-dimensional strategy-parameter sensitivity analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.exceptions import QuantLabError
from quantlab.logging_config import get_logger
from quantlab.strategies import strategy_parameter_names

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
    accepted = strategy_parameter_names(config.strategy_name)
    unknown = set(names).difference(accepted)
    if unknown:
        raise ValueError(
            f"Unknown parameters for strategy {config.strategy_name!r}: "
            f"{sorted(unknown)}."
        )
    for name, values in ((parameter_x, values_x), (parameter_y, values_y)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"Values for {name!r} must be a sequence.")
        if not values:
            raise ValueError(f"Values for {name!r} must not be empty.")
        if _contains_duplicates(values):
            raise ValueError(f"Values for {name!r} must not contain duplicates.")


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
