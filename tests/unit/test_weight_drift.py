"""Direct tests for `apply_weight_drift` (the weight-drift feedback loop).

Each test asserts against either a GENUINELY independent reference (a
different computational path -- price levels and share counts, never a
copy of `apply_weight_drift`'s own return-compounding recursion, which
would silently reproduce the same bug it's meant to catch) or a precise
hand-derived expectation.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.accounting import apply_weight_drift, run_accounting
from quantlab.constants import EPSILON
from quantlab.exceptions import BacktestError


def _reference_drift_from_prices(
    executed: pd.DataFrame, asset_returns: pd.DataFrame
) -> pd.DataFrame:
    """Independent reference: reconstructs PRICE LEVELS from returns and
    tracks SHARE-COUNT-implied dollar exposure plus an EXPLICIT residual
    cash balance -- a genuinely different computational path from
    `apply_weight_drift`'s own dollar/E return-compounding recursion, so
    it cannot silently share the same bug.

    ``price_entering[t]`` is each asset's price level just BEFORE row t's
    own return is applied (i.e. entering row t) -- consistent with the
    expectation that ``executed[t]`` (this function's own output, and
    `apply_weight_drift`'s) is likewise a PRE-period value. Any weight not
    allocated at the anchor (``1 - sum(target)``) is tracked as explicit,
    zero-return cash -- normalizing by ``sum(dollar)`` alone (ignoring
    that residual) would be wrong for a partially-invested portfolio.
    """
    columns = executed.columns
    growth = asset_returns.fillna(0.0) + 1.0
    cum_growth = growth.cumprod()
    price_entering = cum_growth.shift(1).fillna(1.0)

    out = pd.DataFrame(0.0, index=executed.index, columns=columns)
    shares = pd.Series(0.0, index=columns)
    cash = 0.0
    previous: np.ndarray | None = None
    for raw_date in executed.index:
        date = pd.Timestamp(raw_date)
        row = cast("pd.Series", executed.loc[date])
        row_np = row.to_numpy(dtype=float)
        is_anchor = previous is None or bool(
            np.any(np.abs(row_np - previous) > EPSILON)
        )
        previous = row_np
        price_now = cast("pd.Series", price_entering.loc[date])
        if is_anchor:
            shares = (row / price_now.replace(0.0, np.nan)).fillna(0.0)
            cash = 1.0 - float(row.sum())
            out.loc[date] = row
        else:
            dollar = shares * price_now
            total = cash + float(dollar.sum())
            out.loc[date] = (dollar / total) if total != 0.0 else 0.0
    return out


def test_price_round_trip_produces_the_mathematically_correct_nav() -> None:
    """50/50 A/B, A gains 10% then loses exactly 1/11 (9.0909...%), B
    never moves, no rebalancing, no fees. The true final NAV is 1.0
    exactly (A round-trips back to its starting price: 1.10 * 10/11 = 1.0)
    and NOT ONE unit of turnover beyond the single initial entry should
    ever be attributed to the pure-price drift in between."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates)
    asset_returns = pd.DataFrame(
        {"A": [np.nan, 0.10, -1.0 / 11.0], "B": [np.nan, 0.0, 0.0]}, index=dates
    )
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    # First: zero costs, to check the exact NAV claim cleanly.
    free_execution_model = ExecutionModel.from_config(ExecutionConfig())
    free_result = run_accounting(
        held, asset_returns, free_execution_model, 100_000.0, model_weight_drift=True
    )
    assert free_result.equity.iloc[-1] == pytest.approx(100_000.0, rel=1e-9)
    # Turnover: 1.0 on the entry (0 -> 50/50), exactly 0.0 on the pure-
    # drift row -- never "interpreted as a daily transaction".
    assert free_result.turnover.tolist() == pytest.approx([0.0, 1.0, 0.0])

    # Second: nonzero costs -- if drift were ever mistaken for a trade,
    # this would show up as a nonzero cost drag on the pure-drift row.
    execution_model = ExecutionModel.from_config(
        ExecutionConfig(commission_bps=10.0, spread_bps=10.0, slippage_bps=10.0)
    )
    result = run_accounting(
        held, asset_returns, execution_model, 100_000.0, model_weight_drift=True
    )
    assert result.costs.total.iloc[2] == pytest.approx(0.0, abs=1e-9)
    assert result.costs.total.iloc[1] > 0.0


def test_hand_computed_drift_matches_a_genuinely_independent_price_reference() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    executed = pd.DataFrame(
        {
            "A": [0.0, 0.5, 0.5, 0.5, 0.4],
            "B": [0.0, 0.3, 0.3, 0.3, 0.2],
        },
        index=dates,
    )
    asset_returns = pd.DataFrame(
        {
            "A": [np.nan, 0.0, 0.10, 0.02, 0.0],
            "B": [np.nan, 0.0, -0.05, 0.01, 0.0],
        },
        index=dates,
    )
    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    expected = _reference_drift_from_prices(executed, asset_returns)
    pd.testing.assert_frame_equal(drifted, expected, check_exact=False)
    assert not provenance.drift_compliance_forced.to_numpy().any()
    assert not provenance.drift_compliance_pending.to_numpy().any()

    # Row 1 is the anchor (0 -> 0.5/0.3): a real trade, output == target.
    assert drifted.loc[dates[1], "A"] == pytest.approx(0.5)
    assert trade_changes.loc[dates[1], "A"] == pytest.approx(0.5)
    # Row 2 is a pure-drift row, but row 1's OWN return was flat (0%), so
    # the weight ENTERING row 2 is still unchanged at 0.5 -- zero trade.
    assert cast(float, trade_changes.loc[dates[2]].abs().sum()) == pytest.approx(0.0)
    assert drifted.loc[dates[2], "A"] == pytest.approx(0.5)
    # Row 3 is where drift actually becomes visible: row 2's OWN +10%/-5%
    # returns are what move the weight ENTERING row 3 away from 0.5/0.3.
    assert cast(float, trade_changes.loc[dates[3]].abs().sum()) == pytest.approx(0.0)
    assert drifted.loc[dates[3], "A"] != pytest.approx(0.5)
    # Row 4 is a fresh anchor (a real rebalance) -- output exactly as
    # given, and the trade delta is the REAL size (target minus whatever
    # drift alone would have produced), not simply the row-to-row diff of
    # `drifted` (which would conflate the rebalance with prior drift).
    assert drifted.loc[dates[4], "A"] == pytest.approx(0.4)
    assert drifted.loc[dates[4], "B"] == pytest.approx(0.2)
    # The trade delta is the REAL size (target minus whatever drift alone
    # would have produced entering row 4, i.e. row 3's own return applied
    # on top of `drifted.loc[dates[3]]`) -- genuinely nonzero, and NOT
    # simply `target - drifted.loc[dates[3]]` (that would ignore row 3's
    # own return, which further moved the pre-anchor drifted value).
    assert trade_changes.loc[dates[4], "A"] != pytest.approx(0.0)
    row3 = pd.Timestamp(dates[3])
    naive_diff = 0.4 - cast(float, drifted.loc[row3, "A"])
    assert trade_changes.loc[dates[4], "A"] != pytest.approx(naive_diff)


def test_pre_period_weight_is_not_double_counted_against_its_own_return() -> None:
    """The core bug this rewrite fixes: `apply_weight_drift`'s output for
    row t must be the weight HELD ENTERING row t (pre-return), never the
    weight AFTER row t's own return has already been baked in -- reusing
    the post-return value would double-count that return when accounting.py
    multiplies it by `asset_returns[t]` a second time."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=dates)
    asset_returns = pd.DataFrame({"A": [np.nan, 0.20, 0.0]}, index=dates)
    drifted, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    # Row 0 is the anchor: weight entering row 0 is 0.5 (the target itself).
    assert drifted.loc[dates[0], "A"] == pytest.approx(0.5)
    # Row 1: weight ENTERING row 1 -- i.e. BEFORE row 1's own (flat, 0%)
    # return -- is still just 0.5 (only row 0's anchor has happened so
    # far; row 1 hasn't earned its own return yet from this frame's point
    # of view). A buggy "post-return" convention would instead already
    # show row 0's +20% baked in here, which never happened for row 1.
    assert drifted.loc[dates[1], "A"] == pytest.approx(0.5)
    assert cast(float, trade_changes.loc[dates[1]].abs().sum()) == pytest.approx(0.0)


def test_constant_target_on_schedule_still_rebalances_back_to_target() -> None:
    """A scheduled rebalance whose freshly-decided target happens to
    numerically equal the immediately preceding one (e.g. a constant
    50/50 target under a daily schedule) must still be treated as a real
    trade back to target -- value-diffing `executed` against its own
    previous row alone cannot detect this, since the two numbers are
    identical. Without `rebalance_date`, the position would silently keep
    drifting away from target forever with zero recorded turnover."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.20, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=dates
    )
    tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])

    # Without rebalance_date: the position drifts to ~54.55/45.45 after
    # A's +20% and never snaps back, even though every day is nominally a
    # scheduled (daily) rebalance.
    without_schedule, _, _ = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert without_schedule.loc[dates[2], "A"] == pytest.approx(0.6 / 1.1)
    assert without_schedule.loc[dates[3], "A"] == pytest.approx(0.6 / 1.1)

    # With rebalance_date=True every day (a daily schedule): the position
    # correctly snaps back to 50/50 the next row after it drifted -- the fix.
    rebalance_date = pd.DataFrame(True, index=dates, columns=["A", "B"])
    with_schedule, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        rebalance_date=rebalance_date,
    )
    assert with_schedule.loc[dates[2], "A"] == pytest.approx(0.5)
    assert with_schedule.loc[dates[3], "A"] == pytest.approx(0.5)
    # A real, nonzero trade is recorded snapping back to target.
    assert cast(float, trade_changes.loc[dates[2]].abs().sum()) > 0.0


def test_rebalance_date_between_schedule_dates_still_drifts_normally() -> None:
    """A non-daily schedule (`rebalance_date` True only on a few rows)
    must still let drift accumulate normally BETWEEN those rows -- the
    fix must not force a snap-back on every row, only on genuine
    scheduled dates."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    executed = pd.DataFrame(
        {"A": [0.5] * 5, "B": [0.5] * 5},
        index=dates,
    )
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.20, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=dates,
    )
    tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
    # Rebalance only on the first and last row (like a weekly/monthly
    # schedule where the middle rows aren't rebalance dates).
    rebalance_date = pd.DataFrame(
        {
            "A": [True, False, False, False, True],
            "B": [True, False, False, False, True],
        },
        index=dates,
    )

    drifted, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        rebalance_date=rebalance_date,
    )
    # Drift continues normally through the non-rebalance rows.
    assert drifted.loc[dates[2], "A"] == pytest.approx(0.6 / 1.1)
    assert drifted.loc[dates[3], "A"] == pytest.approx(0.6 / 1.1)
    # The scheduled rebalance date snaps back to target with a real trade.
    assert drifted.loc[dates[4], "A"] == pytest.approx(0.5)
    assert cast(float, trade_changes.loc[dates[4]].abs().sum()) > 0.0


def test_closed_asset_dollar_frozen_but_weight_still_drifts() -> None:
    """A closed asset's own dollar exposure does not move (its
    `asset_return` is 0 on a synthetic closure bar), but its WEIGHT still
    drifts purely through E's own movement from the other, tradable
    asset's real return."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.3, 0.3, 0.3, 0.3]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [np.nan, 0.0, 0.20, 0.0], "B": [np.nan, 0.0, 0.0, 0.0]}, index=dates
    )
    drifted, _, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    # Weight ENTERING row 3 reflects row 2's +20% A move (E grew to 1.10);
    # B's dollar exposure never moved, but its weight (0.3/1.10) shrank.
    assert drifted.loc[dates[3], "B"] == pytest.approx(0.3 / 1.10)
    assert drifted.loc[dates[3], "B"] != pytest.approx(0.3)


def test_bankruptcy_guard_flattens_and_never_produces_inf_or_nan() -> None:
    """A leveraged/short scenario engineered so relative E crosses <= 0.
    The row whose OWN return causes the ruin still reports its real
    (catastrophic) pre-ruin weight -- only the FOLLOWING row is flat."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame(
        {"A": [2.0, 2.0, 2.0, 2.0]}, index=dates
    )  # 2x leveraged long
    asset_returns = pd.DataFrame(
        {"A": [np.nan, -0.60, 0.0, 0.05]}, index=dates
    )  # -60% move: gross_return = 2.0 * -0.60 = -1.20 -> E <= 0 by row 2
    drifted, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    values = drifted.to_numpy()
    assert np.isfinite(values).all()
    assert not np.isnan(values).any()
    # Row 0 is the anchor (2.0 decided/executed there); row 1 is the
    # first drift row, and the weight ENTERING it is still 2.0 (nothing
    # has drifted yet -- row 0 itself had a flat/undefined return).
    assert drifted.loc[dates[1], "A"] == pytest.approx(2.0)
    # Row 1's OWN return (-60%) is what wipes the position out (gross_
    # return = 2.0 * -0.60 = -1.20 -> E <= 0 while advancing past row 1) --
    # so row 2 is the first row whose ENTERING weight is force-flattened.
    assert drifted.loc[dates[2], "A"] == pytest.approx(0.0)
    # No phantom closing trade is charged for the forced flatten itself.
    assert trade_changes.loc[dates[2], "A"] == pytest.approx(0.0)


def test_maximum_weight_breach_correction_lands_next_row_never_same_row() -> None:
    """Look-ahead / temporal-convention test: a breach detected at row t
    must leave row t's own (entering-t) output untouched (bit-for-bit
    identical to a no-cap control run), and the correction must land
    starting row t+1, never at t itself."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    executed = pd.DataFrame({"A": [0.0, 0.5, 0.5, 0.5, 0.5]}, index=dates)
    asset_returns = pd.DataFrame({"A": [np.nan, 0.0, 1.0, 0.0, 0.0]}, index=dates)

    uncapped, _, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    capped, capped_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=0.6,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )

    # Weight ENTERING row 3 already reflects row 2's +100% A move (E grew
    # to 1.5, weight = 1.0/1.5 = 0.6667) -- genuinely breaching 0.6.
    breach_row = pd.Timestamp(dates[3])
    assert uncapped.loc[breach_row, "A"] == pytest.approx(1.0 / 1.5)
    assert capped.loc[breach_row, "A"] == pytest.approx(uncapped.loc[breach_row, "A"])
    assert cast(float, capped.loc[breach_row, "A"]) > 0.6
    assert bool(provenance.drift_compliance_pending.loc[breach_row, "A"])

    landed_row = pd.Timestamp(dates[4])
    assert capped.loc[landed_row, "A"] == pytest.approx(0.6)
    assert bool(provenance.drift_compliance_forced.loc[landed_row, "A"])
    assert capped_changes.loc[landed_row, "A"] == pytest.approx(
        0.6 - cast(float, capped.loc[breach_row, "A"])
    )


def test_position_group_correction_moves_both_legs_coherently() -> None:
    """A declared position group's breach correction must scale both legs
    via one shared `k_g`, preserving the drifted ratio exactly -- never a
    single leg moving alone."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    executed = pd.DataFrame(
        {
            "X": [0.0, 0.45, 0.45, 0.45, 0.45, 0.45],
            "Y": [0.0, -0.20, -0.20, -0.20, -0.20, -0.20],
        },
        index=dates,
    )
    asset_returns = pd.DataFrame(
        {
            "X": [np.nan, 0.0, 0.0, 0.30, 0.0, 0.0],
            "Y": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )
    drifted, _, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        [("X", "Y")],
        maximum_weight=0.5,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    # Row 1 is the anchor; row 3's OWN +30% X return (applied while
    # advancing past row 3) is what makes the weight ENTERING row 4 breach.
    breach_row = pd.Timestamp(dates[4])
    assert cast(float, drifted.loc[breach_row, "X"]) > 0.5
    breach_ratio = cast(float, drifted.loc[breach_row, "X"]) / cast(
        float, drifted.loc[breach_row, "Y"]
    )

    landed_row = pd.Timestamp(dates[5])
    assert drifted.loc[landed_row, "X"] == pytest.approx(0.5)
    assert bool(provenance.drift_compliance_forced.loc[landed_row, "X"])
    assert bool(provenance.drift_compliance_forced.loc[landed_row, "Y"])
    landed_ratio = cast(float, drifted.loc[landed_row, "X"]) / cast(
        float, drifted.loc[landed_row, "Y"]
    )
    assert landed_ratio == pytest.approx(breach_ratio)


def test_closed_asset_responsible_for_breach_resolves_once_it_reopens() -> None:
    """A breach caused by a currently-untradable asset cannot be fixed
    immediately -- the residual is carried as `pending` and resolves once
    the asset reopens, landing one bar after the LP can finally solve it
    without slack."""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    executed = pd.DataFrame({"A": [0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]}, index=dates)
    asset_returns = pd.DataFrame(
        {"A": [np.nan, 0.0, 0.0, 0.30, 0.0, 0.0, 0.0]}, index=dates
    )
    tradable = pd.DataFrame(
        {"A": [True, True, True, False, False, True, True]}, index=dates
    )
    drifted, _, provenance = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=0.45,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    # Row 1 is the anchor; row 3's OWN +30% return (applied while
    # advancing past row 3) is what makes the weight ENTERING row 4
    # breach -- and row 4 is closed, so it is unresolvable that row.
    breach_row = pd.Timestamp(dates[4])
    assert drifted.loc[breach_row, "A"] == pytest.approx(0.52 / 1.12)
    assert cast(float, drifted.loc[breach_row, "A"]) > 0.45
    assert bool(provenance.drift_compliance_pending.loc[breach_row, "A"])
    # Row 5 (`dates[5]`) is where A actually reopens -- the correction
    # lands there, at the EARLIEST row it is achievable using that row's
    # own information (not one row later): landing is not deferred an
    # extra row beyond what tradability itself requires.
    reopens_row = pd.Timestamp(dates[5])
    assert drifted.loc[reopens_row, "A"] == pytest.approx(0.45)
    assert bool(provenance.drift_compliance_forced.loc[reopens_row, "A"])
    # Stays resolved on the following row -- no further correction needed.
    assert drifted.loc[dates[6], "A"] == pytest.approx(0.45)
    assert not bool(provenance.drift_compliance_forced.loc[dates[6], "A"])


def test_model_weight_drift_defaults_to_no_drift_via_run_accounting() -> None:
    """Regression gate: `run_accounting`'s `model_weight_drift=False`
    reproduces the plain constant-weight step function exactly."""
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    held = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5]}, index=dates)
    returns = pd.DataFrame({"A": [np.nan, 0.10, 0.02, -0.01]}, index=dates)
    execution_model = ExecutionModel.from_config(ExecutionConfig())
    result = run_accounting(
        held, returns, execution_model, 100_000.0, model_weight_drift=False
    )
    # Constant-weight step function: every row's executed weight equals
    # the held weight shifted by exactly one period (no drift).
    assert result.executed_weights.loc[dates[2], "A"] == pytest.approx(0.5)
    assert not result.drift_compliance_forced.to_numpy().any()
    assert not result.drift_compliance_pending.to_numpy().any()


def test_run_accounting_rebalance_date_forces_on_schedule_rebalance() -> None:
    """`run_accounting`'s own `rebalance_date` parameter must reach
    `apply_weight_drift` and produce real turnover on a scheduled date
    whose freshly-decided target coincidentally matches the previous one
    -- not silently absorbed into drift. Without `rebalance_date`,
    turnover on that date is exactly zero; with it, nonzero."""
    from quantlab.config import ExecutionConfig
    from quantlab.execution.execution_model import ExecutionModel

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    held = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]}, index=dates
    )
    returns = pd.DataFrame(
        {"A": [np.nan, 0.20, 0.0, 0.0], "B": [np.nan, 0.0, 0.0, 0.0]}, index=dates
    )
    execution_model = ExecutionModel.from_config(ExecutionConfig())

    without_schedule = run_accounting(
        held, returns, execution_model, 100_000.0, model_weight_drift=True
    )
    assert without_schedule.turnover.loc[dates[2]] == pytest.approx(0.0)
    assert without_schedule.executed_weights.loc[dates[2], "A"] == pytest.approx(
        0.6 / 1.1
    )

    with_schedule = run_accounting(
        held,
        returns,
        execution_model,
        100_000.0,
        model_weight_drift=True,
        rebalance_date=pd.DataFrame(True, index=dates, columns=["A", "B"]),
    )
    assert with_schedule.turnover.loc[dates[2]] > 0.0


def test_closed_column_never_anchors_just_because_another_column_does() -> None:
    """A closed instrument must never be force-reset to its stale decided
    value just because a DIFFERENT, open instrument's own schedule/value
    change anchors the same row -- it must keep drifting undisturbed,
    with zero trade/turnover attributed to it, until it is itself
    genuinely decided on a date it is actually tradable."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=dates)
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.20, 0.0, 0.0], "B": [0.0, 0.0, 0.20, 0.0]}, index=dates
    )
    # A is closed on date[2], the day B's own schedule fires.
    tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
    tradable.loc[dates[2], "A"] = False
    rebalance_date = pd.DataFrame(False, index=dates, columns=["A", "B"])
    rebalance_date.loc[dates[2], "B"] = True

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        rebalance_date=rebalance_date,
    )
    # A keeps drifting from its own +20% move entering date[2] -- never
    # reset to the stale 0.5 target just because B anchors this row.
    assert drifted.loc[dates[2], "A"] == pytest.approx(0.6 / 1.1)
    assert trade_changes.loc[dates[2], "A"] == pytest.approx(0.0)
    # B genuinely anchors (its own schedule date): real trade recorded.
    assert drifted.loc[dates[2], "B"] == pytest.approx(0.5)
    assert trade_changes.loc[dates[2], "B"] != pytest.approx(0.0)
    assert not provenance.drift_compliance_forced.to_numpy().any()

    # A reopens on date[3] with no fresh decision of its own: it should
    # continue drifting from where it actually was (still not snapped to
    # any stale target), not suddenly reset either.
    assert trade_changes.loc[dates[3], "A"] == pytest.approx(0.0)


def test_full_portfolio_anchor_unaffected_by_partial_anchor_logic() -> None:
    """When EVERY column anchors together (the ordinary single-calendar
    case), behavior must stay byte-identical to a plain whole-row reset:
    both columns land exactly on target with a real trade, and `E`/`
    dollar` fully renormalize."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=dates)
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.20, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates
    )
    rebalance_date = pd.DataFrame(True, index=dates, columns=["A", "B"])
    drifted, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        rebalance_date=rebalance_date,
    )
    assert drifted.loc[dates[2], "A"] == pytest.approx(0.5)
    assert drifted.loc[dates[2], "B"] == pytest.approx(0.5)
    assert cast(float, trade_changes.loc[dates[2]].abs().sum()) > 0.0


def test_maximum_turnover_caps_an_anchor_catch_up_and_carries_the_remainder() -> None:
    """A scheduled anchor's catch-up trade must respect `maximum_turnover`
    exactly like an ordinary decision-level rebalance does -- landing
    partially, carrying the unresolved remainder forward, and never
    exceeding the cap on any single row. The initial entry (magnitude 1.0
    from the conventional `w_{-1} = 0` treatment `cap_turnover` already
    uses) is itself subject to the same cap, so it needs several rows to
    fully resolve before the interesting scheduled-catch-up scenario
    (drift away from a numerically unchanged target, then a schedule date)
    even begins."""
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    executed = pd.DataFrame({"A": [0.5] * n, "B": [0.5] * n}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * n, "B": [0.0] * n}, index=dates)
    asset_returns.loc[dates[20], "A"] = 0.20
    rebalance_date = pd.DataFrame(False, index=dates, columns=["A", "B"])
    rebalance_date.loc[dates[21]] = True
    cap = 0.05

    drifted, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        rebalance_date=rebalance_date,
        maximum_turnover=cap,
    )
    # Never exceeds the cap on any row, including the initial entry.
    assert trade_changes.abs().sum(axis=1).max() <= cap + 1e-9
    # The initial entry (from an implicit all-cash start) fully resolves
    # well before the drift/schedule scenario begins at row 20.
    assert drifted.loc[dates[19], "A"] == pytest.approx(0.5)
    assert drifted.loc[dates[19], "B"] == pytest.approx(0.5)
    # The schedule fires at row 21, but the catch-up doesn't land in one
    # shot -- it is genuinely throttled by the cap.
    assert drifted.loc[dates[21], "A"] == pytest.approx(0.5204545454545454)
    assert cast(float, trade_changes.loc[dates[21]].abs().sum()) == pytest.approx(cap)
    # The remainder is fully caught up by the very next row, with no
    # further schedule/value trigger needed.
    assert drifted.loc[dates[22], "A"] == pytest.approx(0.5, abs=1e-9)
    assert drifted.loc[dates[22], "B"] == pytest.approx(0.5, abs=1e-9)
    assert drifted.loc[dates[-1], "A"] == pytest.approx(0.5, abs=1e-9)


def test_partial_anchor_respects_maximum_turnover_and_debt_survives_a_closure() -> None:
    """Two invariants checked together: (1) `maximum_turnover` must bound
    a MIXED-tradability partial anchor's catch-up exactly like a
    whole-portfolio one -- never applying a partial anchor's fresh
    decision directly and uncapped; (2) a turnover-capped catch-up debt
    must never trade a column while it is closed -- never chasing
    `catchup_target` on a column regardless of that column's own
    tradability. A is tradable only for the very first row (its own
    capped entry), then closed for the rest of the window, while B
    anchors on every row and is always tradable."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame({"A": [0.3] * 4, "B": [0.3, 0.3, 0.9, 0.9]}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * 4, "B": [0.0] * 4}, index=dates)
    tradable = pd.DataFrame(
        {"A": [True, False, False, False], "B": [True] * 4}, index=dates
    )
    cap = 0.2

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        maximum_turnover=cap,
    )
    # Never exceeds the cap on any row, including the mixed-tradability
    # partial-anchor rows.
    assert trade_changes.abs().sum(axis=1).max() <= cap + 1e-9
    # A's own entry (target 0.3) is capped below its full size on row 0
    # (shared budget with B's own simultaneous entry) and then genuinely
    # frozen -- zero further trade -- for every row it stays closed,
    # never silently caught up while untradable.
    assert drifted.loc[dates[0], "A"] == pytest.approx(0.1)
    assert drifted.loc[dates[0], "A"] != pytest.approx(0.3)
    for date in dates[1:]:
        assert drifted.loc[date, "A"] == pytest.approx(0.1)
        assert trade_changes.loc[date, "A"] == pytest.approx(0.0)
    # B's own debt (entry, then the row-2 anchor to 0.9) keeps resolving,
    # entirely unaffected by A sitting closed with its own debt untouched.
    b_last = cast(float, drifted.loc[dates[-1], "B"])
    b_first = cast(float, drifted.loc[dates[0], "B"])
    assert b_last < 0.9
    assert b_last > b_first
    assert not provenance.drift_compliance_forced.to_numpy().any()
    assert not provenance.drift_compliance_pending.to_numpy().any()


def test_a_new_partial_decision_never_wipes_an_unrelated_columns_debt() -> None:
    """A fresh decision on ONE column must only supersede that column's
    own outstanding ordinary debt, never wipe an unrelated column's own
    still-resolving debt or the WHOLE portfolio's turnover-catch-up
    state. Scenario: target A=B=1, a
    turnover-capped first fill lands both at 0.25, then a fresh decision
    arrives for B ALONE (B=0.80) -- A must
    keep converging toward 1.0, and B must converge toward its OWN new
    target without exceeding the shared cap."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 0.80]}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3}, index=dates)
    cap = 0.5

    drifted, trade_changes, _ = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=None,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        maximum_turnover=cap,
    )
    assert trade_changes.abs().sum(axis=1).max() <= cap + 1e-9
    # First (shared) fill: both capped at 0.25.
    assert drifted.loc[dates[0], "A"] == pytest.approx(0.25)
    assert drifted.loc[dates[0], "B"] == pytest.approx(0.25)
    # A's OWN debt (toward 1.0) keeps converging even after B gets its own
    # fresh, unrelated decision -- never stuck at 0.25 forever.
    a1 = cast(float, drifted.loc[dates[1], "A"])
    a2 = cast(float, drifted.loc[dates[2], "A"])
    assert a2 > a1 > 0.25
    # B converges toward its NEW target (0.80), never overshooting past it.
    b2 = cast(float, drifted.loc[dates[2], "B"])
    assert b2 <= 0.80 + 1e-9
    assert b2 > 0.25


def test_mixing_closed_drift_with_a_fresh_partial_target_stays_compliant() -> None:
    """Regression test: combining a currently-closed, already-drifted
    column with another column's freshly-decided partial target can
    create a NEW hard-limit violation neither had alone -- this must
    never be visible in the output, not even for one row, since every
    input needed to detect and fix it is already known before this row's
    own output is finalized (unlike organic drift, which genuinely needs
    a one-row lag). `maximum_gross_exposure=1.1` is chosen so drift is
    load-bearing: A's own UNDRAFTED target (0.5) plus B's fresh target
    (0.6) sum to exactly 1.1 -- compliant on its own, proven by the
    no-drift control below -- and it is only A's organic drift (closed,
    to 0.5833 after its own 0.4 return) combined with B's fresh decision
    that tips gross exposure over the cap."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.3, 0.3, 0.3, 0.6]}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [0.0, 0.4, 0.0, 0.0], "B": [0.0] * 4}, index=dates
    )
    tradable = pd.DataFrame(
        {"A": [True, True, False, False], "B": [True] * 4}, index=dates
    )
    cap = 1.1

    # Control: with no drift at all (A's own return held at zero), the
    # same executed/tradable/cap combination never breaches -- proving
    # the violation below genuinely requires drift, not just mixing.
    no_drift_returns = pd.DataFrame({"A": [0.0] * 4, "B": [0.0] * 4}, index=dates)
    control, _, control_provenance = apply_weight_drift(
        executed,
        no_drift_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=cap,
        maximum_net_exposure=None,
        long_only=False,
    )
    assert control.abs().sum(axis=1).max() <= cap + 1e-9
    assert not control_provenance.drift_compliance_forced.to_numpy().any()

    drifted, _, provenance = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=cap,
        maximum_net_exposure=None,
        long_only=False,
    )
    gross = drifted.abs().sum(axis=1)
    assert gross.max() <= cap + 1e-9
    # The mixing row (B's fresh target combined with A's closed, drifted
    # value) is corrected in the SAME row -- no look-ahead lag needed.
    assert bool(provenance.drift_compliance_forced.loc[dates[3], "B"])
    assert not bool(provenance.drift_compliance_pending.loc[dates[3]].any())


def test_drift_compliance_forced_correction_is_exempt_from_the_turnover_cap() -> None:
    """A hard-risk-limit-forced drift-compliance correction must NEVER be
    subject to `maximum_turnover` -- capping it could leave a genuine
    `maximum_weight`/exposure breach uncorrected indefinitely, which is
    strictly worse than a large one-off corrective trade. Enough rows are
    given for the (also turnover-capped) initial entry to fully resolve
    well before the drift/violation scenario begins."""
    n = 130
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    executed = pd.DataFrame({"A": [0.5] * n, "B": [0.5] * n}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * n, "B": [0.0] * n}, index=dates)
    asset_returns.loc[dates[110], "A"] = 1.0
    cap = 0.01

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=0.55,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        maximum_turnover=cap,
    )
    assert drifted.loc[dates[109], "A"] == pytest.approx(0.5)
    # Row 111 (drift-only, no correction landed yet) genuinely breaches
    # maximum_weight -- the breach is real and reported, not pre-empted.
    assert drifted.loc[dates[111], "A"] == pytest.approx(2.0 / 3.0)
    forced_row = provenance.drift_compliance_forced.any(axis=1)
    assert forced_row.sum() == 1
    landed_date = drifted.index[forced_row][0]
    assert landed_date == dates[112]
    # The correction's own trade size on its landing row exceeds the tiny
    # turnover cap -- proving it was never throttled by it.
    assert trade_changes.loc[landed_date].abs().sum() > cap
    assert cast(float, drifted.loc[landed_date, "A"]) <= 0.55 + 1e-9


def test_compliance_correction_is_never_delayed_behind_ordinary_debt() -> None:
    """Regression test: a queued drift-compliance correction must be
    checked and landed BEFORE ordinary rebalance debt is processed, every
    row -- an unrelated column's own outstanding, still-resolving
    turnover-capped debt must never delay or throttle a compliance
    correction on a DIFFERENT column. C carries a large, slowly-resolving
    ordinary debt the entire time A's own maximum_weight breach is
    detected and lands -- A's correction still lands in full, on the same
    exempt-from-`maximum_turnover` schedule as if C's debt did not exist."""
    n = 130
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    executed = pd.DataFrame({"A": [0.5] * n, "B": [0.5] * n}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * n, "B": [0.0] * n}, index=dates)
    asset_returns.loc[dates[110], "A"] = 1.0
    cap = 0.01
    executed["C"] = 0.0
    executed.loc[dates[50] :, "C"] = 0.2
    asset_returns["C"] = 0.0

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=0.55,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        maximum_turnover=cap,
    )
    # C's own ordinary debt is genuinely still outstanding (mid-catch-up,
    # not yet at its 0.2 target) when A's compliance breach lands.
    forced_row = provenance.drift_compliance_forced["A"]
    assert forced_row.any()
    landed_date = drifted.index[forced_row][0]
    assert 0.0 < cast(float, drifted.loc[landed_date, "C"]) < 0.2
    # A's correction still lands in full, exempt from the tiny cap.
    assert cast(float, drifted.loc[landed_date, "A"]) == pytest.approx(0.55)
    assert cast(float, trade_changes.loc[landed_date, "A"]) != pytest.approx(0.0)
    assert abs(cast(float, trade_changes.loc[landed_date, "A"])) > cap


def test_landed_compliance_correction_never_uses_a_stale_closed_asset_value() -> None:
    """Regression test: when a queued drift-compliance correction finally
    lands, a column the LP never moved (here, A -- closed after its own
    initial entry, fixed by the LP's own equality constraint) must
    reflect its CURRENT, naturally-continued weight at landing time,
    never the STALE value it had back on the row the breach was first
    detected. A closed asset's weight keeps drifting via `E` even while a
    compliance correction is pending elsewhere; landing the stale
    detection-time snapshot would silently force-reset it -- the exact
    closed-asset-gets-traded bug this function exists to prevent, just
    reached through the compliance path instead of the schedule-anchor
    path."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    executed = pd.DataFrame({"A": [0.5] * 6, "B": [0.3] * 6}, index=dates)
    asset_returns = pd.DataFrame(
        {"A": [0.0] * 6, "B": [0.0, 1.0, 0.3, 0.2, 0.0, 0.0]}, index=dates
    )
    # A is tradable only long enough to enter at the anchor row, then
    # closed for the rest of the window.
    tradable = pd.DataFrame(
        {"A": [True, False, False, False, False, False], "B": [True] * 6}, index=dates
    )

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=0.85,
        maximum_net_exposure=None,
        long_only=False,
    )
    # Breach detected at row 3 (2024-01-04, gross ~0.865 > 0.85), landed
    # at row 4 (2024-01-05) -- the standard one-row-lag temporal
    # convention, unaffected by this fix.
    assert bool(provenance.drift_compliance_pending.loc[dates[3], "B"])
    assert bool(provenance.drift_compliance_forced.loc[dates[4], "B"])
    # A is never marked as moved/forced -- the LP fixed it, never traded it.
    assert not provenance.drift_compliance_forced.loc[dates[4], "A"]
    assert trade_changes.loc[dates[4], "A"] == pytest.approx(0.0)
    # A's own weight at landing time reflects its CURRENT continued drift
    # (it moved between detection and landing purely because B's own
    # further return shifted E), NOT the stale value frozen at detection.
    assert drifted.loc[dates[3], "A"] == pytest.approx(0.337838, abs=1e-5)
    assert drifted.loc[dates[4], "A"] == pytest.approx(0.305623, abs=1e-5)
    assert drifted.loc[dates[4], "A"] != pytest.approx(drifted.loc[dates[3], "A"])
    # The landed row is fully compliant.
    assert cast(float, drifted.loc[dates[4]].abs().sum()) <= 0.85 + 1e-6


def test_best_effort_correction_actually_lands_on_open_columns_while_blocked() -> None:
    """Regression test: when the breach is caused by a currently-closed
    column the LP cannot move (A, fixed by its own equality constraint,
    genuinely too large to fix even by fully zeroing every other column),
    `restore_drift_compliance` still proposes the best ACHIEVABLE
    improvement using whatever IS tradable (B) -- this must actually be
    APPLIED to the row's own output, not merely recomputed and discarded
    every day while the correction sits pending forever. A surges (its
    own price return, still open) right before closing, leaving a
    genuinely irresolvable breach; B must still be walked toward zero on
    the very next row, even though the overall breach remains `pending`
    (A alone already exceeds the cap)."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    executed = pd.DataFrame({"A": [0.3] * 6, "B": [0.05] * 6}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * 6, "B": [0.0] * 6}, index=dates)
    asset_returns.loc[dates[2], "A"] = 3.0  # A quadruples right before closing.
    tradable = pd.DataFrame(
        {"A": [True, True, True, False, False, False], "B": [True] * 6}, index=dates
    )

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        tradable,
        None,
        maximum_weight=None,
        maximum_gross_exposure=0.5,
        maximum_net_exposure=None,
        long_only=False,
    )
    # A's own surge (applied while advancing past row 2) breaches the cap
    # entering row 3, the row A closes -- detected there, not yet acted on.
    breach_row = pd.Timestamp(dates[3])
    assert cast(float, drifted.loc[breach_row].abs().sum()) > 0.5
    assert bool(provenance.drift_compliance_pending.loc[breach_row, "A"])
    assert trade_changes.loc[breach_row, "B"] == pytest.approx(0.0)
    # The following row: B is walked to its best achievable value (zero,
    # the only feasible reduction) -- a REAL trade, not merely recomputed
    # and left unapplied. The overall breach stays pending (A alone still
    # exceeds the cap; nothing further is achievable), but B's own
    # component of the fix is genuinely done.
    applied_row = pd.Timestamp(dates[4])
    assert drifted.loc[applied_row, "B"] == pytest.approx(0.0, abs=1e-9)
    assert trade_changes.loc[applied_row, "B"] != pytest.approx(0.0)
    assert bool(provenance.drift_compliance_pending.loc[applied_row, "A"])
    # B never moves again once it has nothing left to give.
    assert drifted.loc[dates[5], "B"] == pytest.approx(0.0, abs=1e-9)
    assert trade_changes.loc[dates[5], "B"] == pytest.approx(0.0)


def test_apply_weight_drift_rejects_a_non_boolean_rebalance_date_directly() -> None:
    """`apply_weight_drift` itself -- not just `run_accounting` -- must
    reject a non-boolean `rebalance_date` (e.g. the string `'False'`,
    which would otherwise silently coerce to truthy) rather than letting
    it corrupt anchor detection."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3}, index=dates)
    bad_rebalance_date = pd.DataFrame(
        {"A": ["False", "True", "False"], "B": ["False", "True", "False"]},
        index=dates,
    )
    with pytest.raises(BacktestError, match="boolean"):
        apply_weight_drift(
            executed,
            asset_returns,
            None,
            None,
            maximum_weight=None,
            maximum_gross_exposure=None,
            maximum_net_exposure=None,
            long_only=False,
            rebalance_date=bad_rebalance_date,
        )


def _valid_apply_weight_drift_args() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    executed = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3}, index=dates)
    return executed, asset_returns


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("asset_returns_not_a_frame", "asset_returns must be a pandas DataFrame"),
        ("asset_returns_dup_index", "asset_returns index must not contain duplicate"),
        (
            "asset_returns_dup_columns",
            "asset_returns columns must not contain duplicate",
        ),
        ("asset_returns_missing_symbol", "asset_returns must cover every executed"),
        ("asset_returns_non_numeric", "asset_returns must contain only numeric"),
        ("asset_returns_infinity", "asset_returns must not contain Infinity"),
        ("asset_returns_below_total_loss", r"asset_returns must not contain simple"),
        (
            "asset_returns_missing_held_return",
            "asset_returns is missing a return for a held position",
        ),
        ("tradable_not_a_frame", "tradable must be a pandas DataFrame"),
        ("tradable_dup_index", "tradable index must not contain duplicate"),
        ("tradable_missing_values", "tradable must not contain missing values"),
        ("tradable_non_boolean", "tradable must contain only boolean"),
        ("rebalance_date_not_a_frame", "rebalance_date must be a pandas DataFrame"),
        (
            "rebalance_date_dup_index",
            "rebalance_date index must not contain duplicate",
        ),
        (
            "rebalance_date_axis_mismatch",
            "rebalance_date must have the same dates and symbols",
        ),
        (
            "rebalance_date_missing_values",
            "rebalance_date must not contain missing values",
        ),
    ],
)
def test_apply_weight_drift_rejects_malformed_direct_call_arguments(
    override: str, match: str
) -> None:
    """`apply_weight_drift` is a directly-callable public function (see its
    own docstring) that must not silently accept malformed
    `asset_returns`/`tradable`/`rebalance_date` -- unlike `run_accounting`,
    a caller can invoke it directly with entirely unvalidated data. Each
    case here reaches this function's own validation block, not
    `run_accounting`'s (which never delegates to `apply_weight_drift` for
    a malformed-input test, since it validates and raises first)."""
    executed, valid_asset_returns = _valid_apply_weight_drift_args()
    dates = executed.index
    # Typed `Any`, not `pd.DataFrame | None`: several branches below
    # deliberately assign a wrong-typed value (a bare string) to prove
    # apply_weight_drift's own runtime validation rejects it -- that is
    # the point of this test, not a type error to suppress per line.
    asset_returns: Any = valid_asset_returns
    tradable: Any = None
    rebalance_date: Any = None

    if override == "asset_returns_not_a_frame":
        asset_returns = "not a frame"
    elif override == "asset_returns_dup_index":
        asset_returns = asset_returns.copy()
        asset_returns.index = pd.DatetimeIndex([dates[0], dates[0], dates[2]])
    elif override == "asset_returns_dup_columns":
        asset_returns = asset_returns.copy()
        asset_returns.columns = ["A", "A"]
    elif override == "asset_returns_missing_symbol":
        asset_returns = asset_returns.drop(columns=["B"])
    elif override == "asset_returns_non_numeric":
        asset_returns = asset_returns.astype(object)
        asset_returns.iloc[1, 0] = "not a number"
    elif override == "asset_returns_infinity":
        asset_returns = asset_returns.copy()
        asset_returns.iloc[1, 0] = np.inf
    elif override == "asset_returns_below_total_loss":
        asset_returns = asset_returns.copy()
        asset_returns.iloc[1, 0] = -1.5
    elif override == "asset_returns_missing_held_return":
        asset_returns = asset_returns.copy()
        asset_returns.iloc[1, 0] = np.nan
    elif override == "tradable_not_a_frame":
        tradable = "not a frame"
    elif override == "tradable_dup_index":
        tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
        tradable.index = pd.DatetimeIndex([dates[0], dates[0], dates[2]])
    elif override == "tradable_missing_values":
        tradable = pd.DataFrame({"A": [True, None, True], "B": [True] * 3}, index=dates)
    elif override == "tradable_non_boolean":
        tradable = pd.DataFrame(
            {"A": ["False", "True", "False"], "B": ["True"] * 3}, index=dates
        )
    elif override == "rebalance_date_not_a_frame":
        rebalance_date = "not a frame"
    elif override == "rebalance_date_dup_index":
        rebalance_date = pd.DataFrame(False, index=dates, columns=["A", "B"])
        rebalance_date.index = pd.DatetimeIndex([dates[0], dates[0], dates[2]])
    elif override == "rebalance_date_axis_mismatch":
        rebalance_date = pd.DataFrame(False, index=dates, columns=["A"])
    elif override == "rebalance_date_missing_values":
        rebalance_date = pd.DataFrame(
            {"A": [False, None, False], "B": [False] * 3}, index=dates
        )

    with pytest.raises(BacktestError, match=match):
        apply_weight_drift(
            executed,
            asset_returns,
            tradable,
            None,
            maximum_weight=None,
            maximum_gross_exposure=None,
            maximum_net_exposure=None,
            long_only=False,
            rebalance_date=rebalance_date,
        )


def test_forced_and_pending_are_never_both_true_for_the_same_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: the same-row re-check (below the ordinary-debt
    step) is a SECOND, independent `_try_restore` call that can re-
    implicate a column Step 1 (the compliance-debt branch above) already
    landed and marked `forced=True` THIS row -- e.g. Step 1 fully
    resolves column A's own `maximum_weight` breach, but the resulting
    gross exposure (combined with column B) is a NEW violation the
    same-row re-check discovers, only partially fixable, re-marking A
    `pending=True` too. `forced`/`pending` must never both be True for
    the same cell in the same row -- whichever call's write to `landed`
    is temporally LAST (the same-row re-check, since it runs after Step
    1) must be the one whose verdict survives, since it is what the
    final output value actually reflects. `restore_drift_compliance` is
    mocked to force this exact sequence deterministically -- the natural
    LP's own behavior makes this specific overlap rare enough that a
    hand-constructed numeric scenario proved too fragile to rely on."""
    import quantlab.backtesting.accounting as acct_mod
    from quantlab.portfolio.drift_compliance import DriftComplianceResult

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    executed = pd.DataFrame({"A": [0.5, 0.5], "B": [0.3, 0.3]}, index=dates)
    asset_returns = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=dates)

    calls = {"n": 0}

    def fake_restore(row: np.ndarray, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            # Row 0: initial maximum_weight breach on A -- carried pending.
            return DriftComplianceResult(corrected=row.copy(), pending=True)
        corrected = row.copy()
        corrected[0] = 0.3  # A resolved to exactly the maximum_weight cap.
        if calls["n"] == 2:
            # Row 1, Step 1: A's own maximum_weight breach fully resolved.
            # Gross exposure (A+B = 0.3+0.3 = 0.6) still exceeds the 0.5
            # cap this leaves behind, so the same-row re-check below will
            # find a genuine violation and actually invoke this mock again
            # (its own entry guard checks real, unmocked constraints).
            return DriftComplianceResult(corrected=corrected, pending=False)
        # Row 1, same-row re-check: resolves the NEW gross-exposure
        # violation, re-implicating A -- only partially achievable.
        corrected2 = corrected.copy()
        corrected2[0] = 0.25
        return DriftComplianceResult(corrected=corrected2, pending=True)

    monkeypatch.setattr(acct_mod, "restore_drift_compliance", fake_restore)

    _, _, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=0.3,
        maximum_gross_exposure=0.5,
        maximum_net_exposure=None,
        long_only=False,
    )

    overlap = provenance.drift_compliance_forced & provenance.drift_compliance_pending
    assert not overlap.to_numpy().any()
    # The same-row re-check's own verdict (still pending) is what survives
    # for A, matching the actual final landed value (0.25, not 0.3).
    assert bool(provenance.drift_compliance_pending.loc[dates[1], "A"])
    assert not bool(provenance.drift_compliance_forced.loc[dates[1], "A"])


def test_a_row_believed_fully_restored_but_still_violating_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `restore_drift_compliance` reporting `pending=False`
    (fully compliant) is trusted at face value -- a formulation bug in the
    LP that still reports success despite leaving a real violation behind
    must not silently slip through and get reported as a clean "compliance
    restored" trade-log event. `restore_drift_compliance` is mocked to
    return exactly this broken response (unchanged, still-breaching
    `corrected`, `pending=False`) regardless of how many times the loop
    calls it this row -- the row-walk must raise once it reaches the end
    of the row still believing no compliance debt remains outstanding."""
    import quantlab.backtesting.accounting as acct_mod
    from quantlab.portfolio.drift_compliance import DriftComplianceResult

    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    executed = pd.DataFrame({"A": [0.5]}, index=dates)
    asset_returns = pd.DataFrame({"A": [np.nan]}, index=dates)

    def broken_restore(row: np.ndarray, *args: object, **kwargs: object) -> object:
        # Always claims success without actually fixing anything --
        # simulates a bug in the LP's own formulation/status handling.
        return DriftComplianceResult(corrected=row.copy(), pending=False)

    monkeypatch.setattr(acct_mod, "restore_drift_compliance", broken_restore)

    with pytest.raises(BacktestError, match="believed fully compliant"):
        apply_weight_drift(
            executed,
            asset_returns,
            None,
            None,
            maximum_weight=0.3,
            maximum_gross_exposure=None,
            maximum_net_exposure=None,
            long_only=False,
        )


def _random_drift_scenario(rng: np.random.Generator) -> dict[str, object]:
    """Build one randomized, but internally consistent, drift scenario."""
    n_cols = int(rng.integers(2, 5))
    n_rows = int(rng.integers(25, 60))
    columns = [f"S{i}" for i in range(n_cols)]
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")

    long_only = bool(rng.random() < 0.3)
    # Anchor weights compliant by construction: split a random total gross
    # budget across columns, with a random sign per column (all-positive
    # under long_only).
    gross_budget = float(rng.uniform(0.3, 0.9))
    raw = rng.random(n_cols)
    raw = raw / raw.sum() * gross_budget
    signs = np.ones(n_cols) if long_only else rng.choice([-1.0, 1.0], size=n_cols)
    anchor = raw * signs
    executed = pd.DataFrame(np.tile(anchor, (n_rows, 1)), index=dates, columns=columns)

    asset_returns = pd.DataFrame(
        rng.normal(0.0, 0.03, size=(n_rows, n_cols)), index=dates, columns=columns
    )
    asset_returns.iloc[0] = np.nan
    # Occasional larger shocks, to actually exercise compliance breaches
    # and the LP, not just small in-bounds drift.
    shock_mask = rng.random(size=(n_rows, n_cols)) < 0.08
    asset_returns = asset_returns.mask(
        shock_mask, asset_returns + rng.normal(0.0, 0.25, size=(n_rows, n_cols))
    )
    asset_returns.iloc[0] = np.nan

    tradable = pd.DataFrame(
        rng.random(size=(n_rows, n_cols)) > 0.15, index=dates, columns=columns
    )
    tradable.iloc[0] = True  # the anchor row must be tradable

    rebalance_date = pd.DataFrame(
        rng.random(size=(n_rows, n_cols)) < 0.05, index=dates, columns=columns
    )

    maximum_weight = float(rng.uniform(0.15, 0.6)) if rng.random() < 0.7 else None
    maximum_gross_exposure = (
        float(rng.uniform(0.4, 1.5)) if rng.random() < 0.7 else None
    )
    maximum_net_exposure = float(rng.uniform(0.2, 1.2)) if rng.random() < 0.5 else None
    maximum_turnover = float(rng.uniform(0.02, 0.3)) if rng.random() < 0.5 else None

    return {
        "executed": executed,
        "asset_returns": asset_returns,
        "tradable": tradable,
        "rebalance_date": rebalance_date,
        "maximum_weight": maximum_weight,
        "maximum_gross_exposure": maximum_gross_exposure,
        "maximum_net_exposure": maximum_net_exposure,
        "long_only": long_only,
        "maximum_turnover": maximum_turnover,
    }


def test_drift_invariants_hold_across_randomized_deterministic_scenarios() -> None:
    """Deterministic fuzz test: across many randomized drift scenarios
    (variable columns, returns including large shocks, tradability gaps,
    schedules, and hard-risk-limit combinations), `apply_weight_drift`
    must never produce a NaN/Inf weight, never raise (the internal
    "believed fully compliant but still violates" guard alone already
    re-verifies every non-pending row's actual compliance across every
    draw below), and never let ordinary (non-compliance-forced) turnover
    exceed `maximum_turnover`. A fixed seed keeps this reproducible."""
    rng = np.random.default_rng(20260209)
    n_scenarios = 200
    exercised_compliance_forced = False
    exercised_maximum_turnover_binding = False

    for _ in range(n_scenarios):
        scenario = _random_drift_scenario(rng)
        drifted, trade_changes, provenance = apply_weight_drift(
            cast(pd.DataFrame, scenario["executed"]),
            cast(pd.DataFrame, scenario["asset_returns"]),
            cast(pd.DataFrame, scenario["tradable"]),
            None,
            maximum_weight=cast("float | None", scenario["maximum_weight"]),
            maximum_gross_exposure=cast(
                "float | None", scenario["maximum_gross_exposure"]
            ),
            maximum_net_exposure=cast("float | None", scenario["maximum_net_exposure"]),
            long_only=cast(bool, scenario["long_only"]),
            rebalance_date=cast(pd.DataFrame, scenario["rebalance_date"]),
            maximum_turnover=cast("float | None", scenario["maximum_turnover"]),
        )

        assert np.isfinite(drifted.to_numpy()).all()
        assert np.isfinite(trade_changes.to_numpy()).all()

        if provenance.drift_compliance_forced.to_numpy().any():
            exercised_compliance_forced = True

        maximum_turnover = scenario["maximum_turnover"]
        if maximum_turnover is not None:
            # Compliance-forced/pending rows are explicitly EXEMPT from
            # maximum_turnover (see apply_weight_drift's own docstring) --
            # only rows with no compliance activity at all are checked.
            compliance_active = (
                provenance.drift_compliance_forced.to_numpy()
                | provenance.drift_compliance_pending.to_numpy()
            ).any(axis=1)
            ordinary_turnover = trade_changes.abs().sum(axis=1).to_numpy()
            ordinary_only = ordinary_turnover[~compliance_active]
            assert (ordinary_only <= cast(float, maximum_turnover) + 1e-6).all()
            if len(ordinary_only) and (ordinary_only > 1e-9).any():
                exercised_maximum_turnover_binding = True

    # Not vacuous: the randomized shocks/caps must have actually exercised
    # both the compliance-restoration path and a real turnover cap at
    # least once across 200 draws.
    assert exercised_compliance_forced
    assert exercised_maximum_turnover_binding
