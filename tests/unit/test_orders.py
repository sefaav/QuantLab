"""Tests for shared execution-order helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import BacktestError
from quantlab.execution.orders import (
    executed_weights,
    shift_respecting_tradability,
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


def test_shift_respecting_tradability_no_leading_nan_through_closure() -> None:
    """A symbol whose very first tradable row is immediately followed by a
    multi-row closure (e.g. a mixed-calendar history starting on a Friday,
    where the weekend rows only exist because another, always-open
    instrument shares the combined index) must never leave NaN on those
    closed rows: there is no prior decision, so -- same "start flat"
    convention as everywhere else -- they hold 0.0, not NaN propagated by a
    leading gap that ffill can't reach."""
    index = pd.date_range("2024-01-05", periods=4, freq="D")  # Fri, Sat, Sun, Mon
    held = pd.DataFrame(
        {"AAPL": [0.5, 0.5, 0.5, 0.6], "BTC": [0.5, 0.5, 0.5, 0.4]}, index=index
    )
    tradable = pd.DataFrame(
        {"AAPL": [True, False, False, True], "BTC": [True, True, True, True]},
        index=index,
    )
    shifted = shift_respecting_tradability(held, 1, tradable)
    assert shifted["AAPL"].tolist() == [0.0, 0.0, 0.0, 0.5]
    assert np.isfinite(shifted.to_numpy()).all()

    executed = executed_weights(held, tradable=tradable)
    assert np.isfinite(executed.to_numpy()).all()
    assert executed["AAPL"].tolist() == [0.0, 0.0, 0.0, 0.5]


def test_shift_respecting_tradability_rejects_a_negative_periods() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]}, index=index)
    tradable = pd.DataFrame({"AAA": [True, True, True]}, index=index)

    with pytest.raises(BacktestError, match="non-negative integer"):
        shift_respecting_tradability(held, -1, tradable)


def test_shift_respecting_tradability_rejects_misaligned_tradable_axes() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]}, index=index)
    # Same values, different (reversed) row order -- a silent misalignment
    # this must catch rather than trading the wrong row against the wrong
    # tradability flag.
    tradable = pd.DataFrame({"AAA": [True, True, True]}, index=index[::-1])

    with pytest.raises(BacktestError, match="same index and columns"):
        shift_respecting_tradability(held, 1, tradable)


def test_shift_respecting_tradability_rejects_non_boolean_tradable_values() -> None:
    """A string 'False' would otherwise silently coerce to True under a bare
    ``.to_numpy(dtype=bool)`` -- this must be rejected outright instead."""
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]}, index=index)
    tradable = pd.DataFrame({"AAA": ["True", "False", "True"]}, index=index)

    with pytest.raises(BacktestError, match="boolean dtype"):
        shift_respecting_tradability(held, 1, tradable)
