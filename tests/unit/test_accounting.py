"""Accounting and anti-look-ahead tests.

These are the most important correctness tests in the project: they prove the
engine cannot earn a return on a position before it was actually put on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.accounting import compute_asset_returns, run_accounting
from quantlab.config import ExecutionConfig
from quantlab.exceptions import BacktestError
from quantlab.execution.execution_model import ExecutionModel


def _zero_cost_model() -> ExecutionModel:
    return ExecutionModel.from_config(
        ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    )


def test_no_lookahead_gain_starts_one_period_late() -> None:
    """Signal goes long at date 2 → gain must start at date 3.

    The position decided at *t* only earns from *t+1* because executed weights
    are held-weights shifted by one.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    # Asset jumps +10% on date index 3 (the 4th day).
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.0, 0.10, 0.0]}, index=idx)
    # Held weights: flat, then long from date index 2 onward.
    held = pd.DataFrame({"AAA": [0.0, 0.0, 1.0, 1.0, 1.0]}, index=idx)

    acc = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)

    # executed = held.shift(1) → long only from date index 3.
    assert acc.executed_weights["AAA"].iloc[2] == 0.0
    assert acc.executed_weights["AAA"].iloc[3] == 1.0
    # The +10% on date 3 IS captured (position was on from date 3).
    assert acc.gross_returns.iloc[3] == pytest.approx(0.10)
    # Critically, date 2 earns nothing despite the signal turning long at date 2.
    assert acc.gross_returns.iloc[2] == pytest.approx(0.0)


def test_short_position_profits_on_decline() -> None:
    """A negative weight on a negative return yields a gain."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.0, -0.05]}, index=idx)
    held = pd.DataFrame({"AAA": [-1.0, -1.0, -1.0]}, index=idx)
    acc = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    # executed[-1] = -1, return[-1] = -5% → contribution +5%.
    assert acc.gross_returns.iloc[2] == pytest.approx(0.05)


def test_equity_compounding_formula() -> None:
    """equity_t = equity_{t-1} × (1 + net_return_t), equity_0 = capital."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.1, -0.1, 0.2]}, index=idx)
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    acc = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    manual = 100_000.0
    for r in acc.net_returns:
        manual *= 1.0 + r
    assert acc.equity.iloc[-1] == pytest.approx(manual)


def test_costs_reduce_net_below_gross() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.01, 0.01, 0.01]}, index=idx)
    # Alternate weights so there is turnover (and therefore cost).
    held = pd.DataFrame({"AAA": [1.0, 0.0, 1.0, 0.0]}, index=idx)
    costly = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=50.0, spread_bps=50.0, slippage_bps=50.0)
    )
    acc = run_accounting(held, asset_returns, costly, 100_000.0)
    assert acc.equity.iloc[-1] < acc.gross_equity.iloc[-1]
    assert (acc.costs.total >= 0).all()


def test_compute_asset_returns_matches_pct_change() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    rets = compute_asset_returns(prices)
    assert np.isnan(rets["AAA"].iloc[0])
    assert rets["AAA"].iloc[1] == pytest.approx(0.10)
    assert rets["AAA"].iloc[2] == pytest.approx(-0.10)


def test_missing_return_for_held_position_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="Asset return is missing"):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


@pytest.mark.parametrize("bad_capital", [0.0, -1.0, np.nan, np.inf, True])
def test_invalid_initial_capital_raises(bad_capital: object) -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="initial_capital"):
        run_accounting(held, asset_returns, _zero_cost_model(), bad_capital)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_input", ["weights", "returns"])
def test_non_finite_accounting_input_raises(invalid_input: str) -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, 0.01]}, index=idx)
    if invalid_input == "weights":
        held.iloc[1, 0] = np.inf
        message = "held_weights"
    else:
        asset_returns.iloc[1, 0] = np.inf
        message = "asset_returns"

    with pytest.raises(BacktestError, match=message):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


def test_asset_simple_return_below_total_loss_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, -1.01]}, index=idx)

    with pytest.raises(BacktestError, match=r"below -1\.0"):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


def test_missing_return_remains_valid_when_asset_is_not_held() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 0.0]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [np.nan, np.nan]}, index=idx)

    result = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)

    assert result.net_returns.eq(0.0).all()


def test_asset_returns_must_cover_held_weight_axes() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"BBB": [np.nan, 0.01]}, index=idx)

    with pytest.raises(BacktestError, match="must cover every"):
        run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)


def test_tradable_tolerates_a_differently_ordered_frame() -> None:
    """`tradable`'s column order need not match `held_weights`' own order --
    only the *set* of dates and symbols must agree."""
    idx = pd.date_range("2024-01-05", periods=3, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5, 0.5], "BBB": [0.5, 0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame(
        {"AAA": [0.0, 0.01, -0.01], "BBB": [0.0, 0.02, 0.01]}, index=idx
    )
    # Same labels as `held`, deliberately reversed column order.
    tradable = pd.DataFrame(
        {"BBB": [True, True, True], "AAA": [True, True, True]}, index=idx
    )

    result = run_accounting(held, asset_returns, _zero_cost_model(), 100_000.0)
    result_with_mask = run_accounting(
        held, asset_returns, _zero_cost_model(), 100_000.0, tradable=tradable
    )
    pd.testing.assert_series_equal(result.net_returns, result_with_mask.net_returns)


def test_tradable_must_cover_the_same_set_of_symbols_as_held_weights() -> None:
    """A genuine set mismatch (not just reordering) must still raise --
    silently defaulting an unrecognized symbol to "tradable" could let it
    trade on a date it should have stayed closed."""
    idx = pd.date_range("2024-01-05", periods=2, freq="D")
    held = pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [0.5, 0.5]}, index=idx)
    asset_returns = pd.DataFrame({"AAA": [0.0, 0.01], "BBB": [0.0, 0.02]}, index=idx)
    tradable = pd.DataFrame({"AAA": [True, True]}, index=idx)  # missing BBB

    with pytest.raises(BacktestError, match="dates and symbols"):
        run_accounting(
            held, asset_returns, _zero_cost_model(), 100_000.0, tradable=tradable
        )
