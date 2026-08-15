"""Tests for execution cost models."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from quantlab.config import ExecutionConfig
from quantlab.exceptions import BacktestError
from quantlab.execution.costs import CommissionModel, SpreadModel
from quantlab.execution.execution_model import ExecutionModel
from quantlab.execution.slippage import (
    ConstantSlippageModel,
    SlippageModel,
    VolumeBasedSlippageModel,
)


def test_commission_manual_example() -> None:
    """Capital 100_000, turnover 0.5, 10 bps → cost 50."""
    capital = 100_000.0
    # One symbol traded so that summed notional = turnover × capital.
    traded_notional = pd.DataFrame({"AAA": [0.5 * capital]})
    cost = CommissionModel(commission_bps=10.0).calculate(traded_notional)
    assert cost.iloc[0] == pytest.approx(50.0)


def test_commission_as_fraction_of_equity() -> None:
    # Passing weight changes (fraction units) gives cost as a fraction.
    weight_changes = pd.DataFrame({"AAA": [0.5]})  # turnover 0.5
    frac = CommissionModel(commission_bps=10.0).calculate(weight_changes)
    assert frac.iloc[0] == pytest.approx(0.5 * 10.0 / 10_000.0)


def test_spread_uses_half_spread() -> None:
    """spread_cost = traded_notional × spread_bps / 20_000."""
    traded_notional = pd.DataFrame({"AAA": [100_000.0]})
    cost = SpreadModel(spread_bps=4.0).calculate(traded_notional)
    assert cost.iloc[0] == pytest.approx(100_000.0 * 4.0 / 20_000.0)


def test_constant_slippage() -> None:
    traded_notional = pd.DataFrame({"AAA": [100_000.0]})
    cost = ConstantSlippageModel(slippage_bps=2.0).calculate(traded_notional)
    assert cost.iloc[0] == pytest.approx(100_000.0 * 2.0 / 10_000.0)


def test_volume_slippage_grows_with_size() -> None:
    adv = 1_000_000.0
    small = pd.DataFrame({"AAA": [10_000.0]})
    large = pd.DataFrame({"AAA": [500_000.0]})
    model = VolumeBasedSlippageModel(
        base_slippage_bps=1.0, impact_coefficient=0.5, average_daily_volume=adv
    )
    # Effective bps rises with order/ADV, so per-dollar cost is higher for large.
    small_rate = model.calculate(small).iloc[0] / 10_000.0
    large_rate = model.calculate(large).iloc[0] / 500_000.0
    assert large_rate > small_rate


def test_volume_slippage_rejects_zero_adv_for_a_trade() -> None:
    model = VolumeBasedSlippageModel(average_daily_volume=0.0, impact_coefficient=0.1)
    with pytest.raises(ValueError, match="positive ADV"):
        model.calculate(pd.DataFrame({"AAA": [1000.0]}))


def test_zero_adv_is_allowed_when_no_trade_occurs() -> None:
    model = VolumeBasedSlippageModel(average_daily_volume=0.0)
    cost = model.calculate(pd.DataFrame({"AAA": [0.0]}))
    assert cost.iloc[0] == 0.0


def test_execution_model_totals() -> None:
    cfg = ExecutionConfig(commission_bps=2.0, spread_bps=4.0, slippage_bps=2.0)
    model = ExecutionModel.from_config(cfg)
    changes = pd.DataFrame({"AAA": [0.3], "BBB": [0.2]})  # turnover 0.5
    costs = model.compute(changes)
    # commission 0.5*2/1e4 ; spread 0.5*4/2e4 ; slippage 0.5*2/1e4
    assert costs.commission.iloc[0] == pytest.approx(0.5 * 2 / 1e4)
    assert costs.spread.iloc[0] == pytest.approx(0.5 * 4 / 2e4)
    assert costs.slippage.iloc[0] == pytest.approx(0.5 * 2 / 1e4)
    assert costs.total.iloc[0] == pytest.approx(
        costs.commission.iloc[0] + costs.spread.iloc[0] + costs.slippage.iloc[0]
    )


def test_negative_bps_rejected() -> None:
    with pytest.raises(ValueError, match="commission_bps"):
        CommissionModel(commission_bps=-1.0)
    with pytest.raises(ValueError, match="spread_bps"):
        SpreadModel(spread_bps=-1.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CommissionModel(True),
        lambda: SpreadModel(True),
        lambda: ConstantSlippageModel(True),
        lambda: VolumeBasedSlippageModel(
            impact_coefficient=True, average_daily_volume=1.0
        ),
        lambda: VolumeBasedSlippageModel(average_daily_volume=True),
    ],
)
def test_direct_cost_models_reject_boolean_rates(factory: object) -> None:
    with pytest.raises(ValueError, match="finite number"):
        factory()  # type: ignore[operator]


def test_execution_model_rejects_nan_weight_changes() -> None:
    model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0)
    )
    changes = pd.DataFrame({"AAA": [np.nan]})

    with pytest.raises(BacktestError, match=r"executed_weight_changes.*finite"):
        model.compute(changes)


def test_volume_slippage_requires_equity_on_every_execution_date() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    model = VolumeBasedSlippageModel(average_daily_volume=1_000_000.0)
    changes = pd.DataFrame({"AAA": [0.1, 0.1]}, index=index)
    incomplete_equity = pd.Series([100_000.0], index=index[:1])

    with pytest.raises(BacktestError, match="exactly the execution-date index"):
        model.calculate(changes, incomplete_equity)


class _NaNSlippage(SlippageModel):
    def per_symbol_cost(
        self, traded_notional: pd.DataFrame, equity: pd.Series | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            np.nan, index=traded_notional.index, columns=traded_notional.columns
        )


def test_execution_model_rejects_nan_from_custom_slippage() -> None:
    model = ExecutionModel(CommissionModel(0.0), SpreadModel(0.0), _NaNSlippage())
    changes = pd.DataFrame({"AAA": [0.1]})

    with pytest.raises(BacktestError, match=r"slippage costs.*finite"):
        model.compute(changes)


def test_volume_slippage_copies_adv_defensively() -> None:
    index = pd.date_range("2024-01-01", periods=1, freq="D")
    adv = pd.DataFrame({"AAA": [1_000_000.0]}, index=index)
    model = VolumeBasedSlippageModel(impact_coefficient=1.0, average_daily_volume=adv)
    adv.iloc[0, 0] = -1.0

    exposed = model.average_daily_volume
    assert isinstance(exposed, pd.DataFrame)
    assert exposed.iloc[0, 0] == 1_000_000.0
    exposed.iloc[0, 0] = -1.0
    assert model.calculate(pd.DataFrame({"AAA": [100_000.0]}, index=index)).iloc[0] > 0


def test_volume_slippage_requires_explicit_adv() -> None:
    with pytest.raises(ValueError, match="average_daily_volume is required"):
        VolumeBasedSlippageModel()
    with pytest.raises(ValueError, match="average_daily_volume is required"):
        ExecutionModel.from_config(ExecutionConfig(slippage_model=cast(Any, "volume")))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((object(), SpreadModel(0.0), ConstantSlippageModel(0.0)), "commission"),
        ((CommissionModel(0.0), object(), ConstantSlippageModel(0.0)), "spread"),
        ((CommissionModel(0.0), SpreadModel(0.0), object()), "slippage"),
    ],
)
def test_execution_model_validates_component_types(
    args: tuple[object, object, object], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        ExecutionModel(*args)  # type: ignore[arg-type]
