"""Combined execution-cost model.

:class:`ExecutionModel` bundles commission, spread and slippage. Given the
per-symbol change in executed weights it returns each cost component *and* the
total, as a fraction of equity (so the vectorised engine can subtract them
directly from gross returns). The components are always reported separately so
gross vs net can be compared honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import ExecutionConfig
from quantlab.exceptions import BacktestError
from quantlab.execution.costs import CommissionModel, SpreadModel
from quantlab.execution.orders import (
    validate_equity_series,
    validate_execution_frame,
)
from quantlab.execution.slippage import (
    SlippageModel,
    build_slippage_model,
    validate_slippage_cost_frame,
)


@dataclass
class ExecutionCosts:
    """Per-date cost components (fractions of equity) and their total."""

    commission: pd.Series
    spread: pd.Series
    slippage: pd.Series

    def __post_init__(self) -> None:
        """Require aligned finite non-negative component series."""
        expected_index: pd.Index | None = None
        for name, component in (
            ("commission", self.commission),
            ("spread", self.spread),
            ("slippage", self.slippage),
        ):
            if not isinstance(component, pd.Series):
                raise BacktestError(f"{name} cost must be a pandas Series.")
            if not component.index.is_unique:
                raise BacktestError(f"{name} cost index must not contain duplicates.")
            if expected_index is None:
                expected_index = component.index
            elif not component.index.equals(expected_index):
                raise BacktestError("Execution-cost components must share one index.")
            try:
                values = component.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise BacktestError(f"{name} cost must be numeric.") from exc
            if not np.isfinite(values).all() or (values < 0.0).any():
                raise BacktestError(
                    f"{name} cost is negative or non-finite; every component "
                    "must be finite and non-negative."
                )

    @property
    def total(self) -> pd.Series:
        """Total cost per date = commission + spread + slippage."""
        return self.commission + self.spread + self.slippage

    def to_frame(self) -> pd.DataFrame:
        """Return the components and total as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "commission": self.commission,
                "spread": self.spread,
                "slippage": self.slippage,
                "total": self.total,
            }
        )


class ExecutionModel:
    """Aggregate commission + spread + slippage.

    Args:
        commission: Commission model.
        spread: Spread model.
        slippage: Slippage model.
    """

    def __init__(
        self,
        commission: CommissionModel,
        spread: SpreadModel,
        slippage: SlippageModel,
    ) -> None:
        if not isinstance(commission, CommissionModel):
            raise TypeError("commission must be a CommissionModel.")
        if not isinstance(spread, SpreadModel):
            raise TypeError("spread must be a SpreadModel.")
        if not isinstance(slippage, SlippageModel):
            raise TypeError("slippage must implement SlippageModel.")
        self.commission = commission
        self.spread = spread
        self.slippage = slippage

    @classmethod
    def from_config(
        cls,
        execution_config: ExecutionConfig,
        *,
        average_daily_volume: pd.DataFrame | float | None = None,
    ) -> ExecutionModel:
        """Build an execution model from an :class:`ExecutionConfig`."""
        if not isinstance(execution_config, ExecutionConfig):
            raise TypeError("execution_config must be an ExecutionConfig.")
        slippage = build_slippage_model(
            str(execution_config.slippage_model),
            execution_config.slippage_bps,
            impact_coefficient=execution_config.impact_coefficient,
            average_daily_volume=average_daily_volume,
        )
        return cls(
            commission=CommissionModel(execution_config.commission_bps),
            spread=SpreadModel(execution_config.spread_bps),
            slippage=slippage,
        )

    def compute(
        self,
        executed_weight_changes: pd.DataFrame,
        equity: pd.Series | None = None,
    ) -> ExecutionCosts:
        """Return the three cost components for a weight-change matrix.

        Passing weight changes (notional per unit equity) yields costs as a
        fraction of equity, which is what the engine subtracts from gross
        returns.

        Args:
            executed_weight_changes: Per-symbol weight change (fraction of
                equity), ``dates × symbols``.
            equity: Per-date equity (dollars), forwarded to the slippage
                model only — it needs the actual dollar size of each trade
                to compare against a dollar-denominated average daily
                volume (see :class:`~quantlab.execution.slippage.
                VolumeBasedSlippageModel`). Commission/spread are already
                unit-consistent without it.

        Raises:
            BacktestError: If an input or computed cost is misaligned,
                negative, missing or non-finite.
        """
        changes = validate_execution_frame(
            executed_weight_changes, name="executed_weight_changes"
        )
        validated_equity = (
            validate_equity_series(equity, changes.index)
            if equity is not None
            else None
        )
        try:
            commission = self.commission.calculate(changes)
        except ValueError as exc:
            raise BacktestError(
                f"commission cost is negative or non-finite: {exc}"
            ) from exc
        try:
            spread = self.spread.calculate(changes)
        except ValueError as exc:
            raise BacktestError(
                f"spread cost is negative or non-finite: {exc}"
            ) from exc
        try:
            slippage_by_symbol = self.slippage.per_symbol_cost(
                changes, validated_equity
            )
        except ValueError as exc:
            raise BacktestError(f"slippage cost could not be computed: {exc}") from exc
        slippage = validate_slippage_cost_frame(slippage_by_symbol, changes).sum(axis=1)
        for name, component in (
            ("commission", commission),
            ("spread", spread),
            ("slippage", slippage),
        ):
            values = component.to_numpy(dtype=float)
            invalid = ~np.isfinite(values) | (values < 0)
            if invalid.any():
                bad_dates = component.index[invalid]
                raise BacktestError(
                    f"{name} cost is negative or non-finite on "
                    f"{list(bad_dates)[:5]}{'…' if len(bad_dates) > 5 else ''} "
                    f"— every cost component must be non-negative and "
                    "finite before being combined into the total."
                )
        return ExecutionCosts(commission=commission, spread=spread, slippage=slippage)
