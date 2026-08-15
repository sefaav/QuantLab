"""Validation and schema tests for synthetic trade-log fills."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.trade_log import build_trade_log
from quantlab.exceptions import BacktestError
from quantlab.execution.slippage import ConstantSlippageModel, SlippageModel


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=index)
    changes = pd.DataFrame({"AAA": [0.0, 1.0, 0.0]}, index=index)
    equity = pd.Series([100.0, 100.0, 105.0], index=index)
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=index)
    return executed, changes, equity, prices


def _build(
    executed: pd.DataFrame,
    changes: pd.DataFrame,
    equity: pd.Series,
    prices: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    return build_trade_log(
        executed,
        changes,
        equity,
        prices,
        commission_bps=kwargs.pop("commission_bps", 2.0),
        spread_bps=kwargs.pop("spread_bps", 3.0),
        slippage_model=kwargs.pop("slippage_model", ConstantSlippageModel(1.0)),
        **kwargs,
    )


def test_trade_log_uses_new_weight_schema() -> None:
    trades = _build(*_inputs())

    assert trades["new_weight"].tolist() == [1.0]
    assert "target_weight" not in trades.columns
    assert trades["reference_price"].tolist() == [10.0]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("commission_bps", -1.0),
        ("commission_bps", np.nan),
        ("spread_bps", -1.0),
        ("spread_bps", np.inf),
        ("spread_bps", True),
    ],
)
def test_trade_log_rejects_invalid_direct_cost_rates(name: str, value: object) -> None:
    def invoke() -> None:
        if name == "commission_bps":
            _build(*_inputs(), commission_bps=value)
        else:
            _build(*_inputs(), spread_bps=value)

    with pytest.raises(BacktestError, match=name):
        invoke()


def test_trade_log_rejects_non_finite_weights() -> None:
    executed, changes, equity, prices = _inputs()
    changes.iloc[1, 0] = np.nan

    with pytest.raises(BacktestError, match=r"weight_changes.*finite"):
        _build(executed, changes, equity, prices)


def test_trade_log_rejects_misaligned_axes() -> None:
    executed, changes, equity, prices = _inputs()
    changes = changes.rename(columns={"AAA": "BBB"})

    with pytest.raises(BacktestError, match="identical axes"):
        _build(executed, changes, equity, prices)


def test_trade_log_rejects_missing_reference_price_for_a_fill() -> None:
    executed, changes, equity, prices = _inputs()
    prices.iloc[0, 0] = np.nan

    with pytest.raises(BacktestError, match="reference price"):
        _build(executed, changes, equity, prices)


def test_trade_log_rejects_invalid_equity() -> None:
    executed, changes, equity, prices = _inputs()
    equity.iloc[0] = -1.0

    with pytest.raises(BacktestError, match=r"equity.*non-negative"):
        _build(executed, changes, equity, prices)


class _NegativeSlippage(SlippageModel):
    def per_symbol_cost(
        self, traded_notional: pd.DataFrame, equity: pd.Series | None = None
    ) -> pd.DataFrame:
        return -traded_notional.abs()


def test_trade_log_rejects_invalid_per_symbol_slippage() -> None:
    with pytest.raises(BacktestError, match=r"slippage costs.*non-negative"):
        _build(*_inputs(), slippage_model=_NegativeSlippage())
