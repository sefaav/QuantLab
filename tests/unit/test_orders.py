"""Tests for shared execution-order helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import BacktestError
from quantlab.execution.orders import (
    executed_weights,
    traded_notional,
    weight_changes,
)


def test_order_helpers_share_the_accounting_timing_convention() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 1.0, 0.0]}, index=index)
    executed = executed_weights(held)

    assert executed["AAA"].tolist() == [0.0, 0.5, 1.0]
    assert weight_changes(executed)["AAA"].tolist() == [0.0, 0.5, 0.5]


def test_executed_weights_rejects_missing_decisions() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, np.nan, 0.0]}, index=index)

    with pytest.raises(BacktestError, match=r"held_weights.*finite"):
        executed_weights(held)


def test_traded_notional_uses_prior_equity() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    changes = pd.DataFrame({"AAA": [0.1, -0.2]}, index=index)
    equity = pd.Series([100.0, 110.0], index=index)

    assert traded_notional(changes, equity)["AAA"].tolist() == [10.0, 20.0]


def test_traded_notional_rejects_missing_equity_dates() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    changes = pd.DataFrame({"AAA": [0.1, 0.2]}, index=index)
    equity = pd.Series([100.0], index=index[:1])

    with pytest.raises(BacktestError, match="exactly the execution-date index"):
        traded_notional(changes, equity)
