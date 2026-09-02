"""Validation and schema tests for synthetic trade-log fills."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.trade_log import (
    ADJUSTMENT_ORDER,
    TRADE_LOG_COLUMNS,
    TRADE_LOG_SCHEMA_VERSION,
    TradeReason,
    _classify_action,
    _classify_reason,
    build_trade_log,
    parse_adjustment_codes,
    serialize_adjustment_codes,
)
from quantlab.exceptions import BacktestError
from quantlab.execution.slippage import ConstantSlippageModel, SlippageModel
from quantlab.portfolio.constraints import ConstraintTouch


def _touch(
    touched: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    direct: pd.DataFrame | None = None,
) -> ConstraintTouch:
    """Build a ConstraintTouch, defaulting `direct` to `touched` (no
    redistribution concept -- matches _mark_touched's own default)."""
    return ConstraintTouch(
        touched=touched,
        before=before,
        after=after,
        direct=direct if direct is not None else touched,
    )


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


def test_trade_log_schema_has_21_columns_in_order() -> None:
    """`action` is always computed; the trigger/adjustment/position_
    strategy_origin columns stay `None`/`NaT` when the optional reason
    frames are omitted -- the walk-forward call site (which rebuilds
    trades from a stitched out-of-sample series with no per-fold
    diagnostic frames surviving the stitch) must keep working unchanged."""
    trades = _build(*_inputs())

    assert list(trades.columns) == TRADE_LOG_COLUMNS
    assert len(TRADE_LOG_COLUMNS) == 21
    assert TRADE_LOG_SCHEMA_VERSION == 2
    assert trades["action"].tolist() == ["entry_long"]
    for column in (
        "trigger_reason_code",
        "trigger_reason_detail_code",
        "trigger_reason_details",
        "adjustment_reason_codes",
        "adjustment_reason_details",
        "position_strategy_origin_code",
        "position_strategy_origin_details",
    ):
        assert trades[column].tolist() == [None]
    assert pd.isna(trades["position_strategy_origin_timestamp"].iloc[0])


def test_previous_weight_reflects_organic_drift_not_the_prior_rows_own_value() -> None:
    """Regression test: `previous_weight` must be the value organic drift
    actually left the position at going into this trade -- NOT the
    previous ROW's own reported `executed_weight`, which under
    `model_weight_drift=True` can differ from it whenever drift moved the
    position between rows with no trade of its own. Reproduces the exact
    scenario reported: a position drifts from 0.50 up to 0.60 with no
    trade recorded (row 1, `weight_changes=0`), then a sell brings it to
    0.55 (`weight_change=-0.05`) -- `previous_weight` must read 0.60, and
    the action must be `reduce_long`, not `increase_long` (what the old
    `executed_weights.shift(1)` formula -- which would have read 0.50 --
    would have produced)."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    # A leading flat (no-trade) row, matching every real backtest's own
    # warm-up convention -- gives the entry on row 1 a valid prior-period
    # reference price (row 0's), avoiding an unrelated "no prior price"
    # error for what would otherwise be the very first row.
    executed = pd.DataFrame({"AAA": [0.0, 0.5, 0.6, 0.55]}, index=index)
    changes = pd.DataFrame({"AAA": [0.0, 0.5, 0.0, -0.05]}, index=index)
    equity = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    prices = pd.DataFrame({"AAA": [9.0, 10.0, 11.0, 12.0]}, index=index)

    trades = _build(executed, changes, equity, prices)

    assert trades["previous_weight"].tolist() == [0.0, pytest.approx(0.6)]
    assert trades["new_weight"].tolist() == [0.5, pytest.approx(0.55)]
    assert trades["weight_change"].tolist() == [0.5, pytest.approx(-0.05)]
    assert trades["side"].tolist() == ["buy", "sell"]
    assert trades["action"].tolist() == ["entry_long", "reduce_long"]
    assert np.allclose(
        (trades["new_weight"] - trades["previous_weight"]).to_numpy(),
        trades["weight_change"].to_numpy(),
    )


def test_previous_weight_reflects_downward_drift_before_a_buy() -> None:
    """Symmetric case: a long position drifts DOWN from 0.5 to 0.4 with no
    trade of its own, then a buy tops it back up to 0.45
    (`weight_change=+0.05`) -- `previous_weight` must read 0.40, and the
    action must be `increase_long`."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame({"AAA": [0.0, 0.5, 0.4, 0.45]}, index=index)
    changes = pd.DataFrame({"AAA": [0.0, 0.5, 0.0, 0.05]}, index=index)
    equity = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    prices = pd.DataFrame({"AAA": [11.0, 10.0, 9.0, 9.5]}, index=index)

    trades = _build(executed, changes, equity, prices)

    assert trades["previous_weight"].tolist() == [0.0, pytest.approx(0.4)]
    assert trades["new_weight"].tolist() == [0.5, pytest.approx(0.45)]
    assert trades["side"].tolist() == ["buy", "buy"]
    assert trades["action"].tolist() == ["entry_long", "increase_long"]
    assert np.allclose(
        (trades["new_weight"] - trades["previous_weight"]).to_numpy(),
        trades["weight_change"].to_numpy(),
    )


def test_previous_weight_correct_across_a_long_short_reversal() -> None:
    """A position drifts from a long anchor down to a small residual long
    (0.5 -> 0.1, no trade), then a trade flips it to short (-0.2,
    `weight_change=-0.3`) -- `previous_weight` must read the drifted 0.10,
    not the anchor 0.5, and the action must be `reverse_long_to_short`."""
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    executed = pd.DataFrame({"AAA": [0.0, 0.5, 0.1, -0.2]}, index=index)
    changes = pd.DataFrame({"AAA": [0.0, 0.5, 0.0, -0.3]}, index=index)
    equity = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    prices = pd.DataFrame({"AAA": [11.0, 10.0, 6.0, 6.0]}, index=index)

    trades = _build(executed, changes, equity, prices)

    assert trades["previous_weight"].tolist() == [0.0, pytest.approx(0.1)]
    assert trades["new_weight"].tolist() == [0.5, pytest.approx(-0.2)]
    assert trades["side"].tolist() == ["buy", "sell"]
    assert trades["action"].tolist() == ["entry_long", "reverse_long_to_short"]
    assert np.allclose(
        (trades["new_weight"] - trades["previous_weight"]).to_numpy(),
        trades["weight_change"].to_numpy(),
    )


def test_previous_weight_correct_for_a_drift_compliance_forced_trade() -> None:
    """`previous_weight` for a row where the drift-compliance LP forces a
    correction must be the value organic drift actually pushed the
    position to (the breach itself), not the previous row's own value --
    reuses the exact numeric scenario `test_maximum_weight_breach_
    correction_lands_next_row_never_same_row` (test_weight_drift.py)
    already proves the underlying drift/correction mechanics for, this
    time feeding `apply_weight_drift`'s own output straight into
    `build_trade_log`, exactly like the real pipeline (`engine.py` passes
    `accounting.executed_weights`/`accounting.weight_changes`, which ARE
    `apply_weight_drift`'s two frames when drift is active)."""
    from quantlab.backtesting.accounting import apply_weight_drift

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    executed = pd.DataFrame({"A": [0.0, 0.5, 0.5, 0.5, 0.5]}, index=dates)
    asset_returns = pd.DataFrame({"A": [np.nan, 0.0, 1.0, 0.0, 0.0]}, index=dates)

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=0.6,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
    )
    breach_row = dates[3]
    landed_row = dates[4]
    assert bool(provenance.drift_compliance_pending.loc[breach_row, "A"])
    assert bool(provenance.drift_compliance_forced.loc[landed_row, "A"])

    equity = pd.Series(100.0, index=dates)
    prices = pd.DataFrame({"A": [10.0] * 5}, index=dates)
    trades = _build(
        drifted,
        trade_changes,
        equity,
        prices,
        # `drifted` stands in for every decision-level diagnostic frame:
        # nothing else is being attributed here, so this isolates
        # drift_compliance as the sole adjustment reason.
        executed_desired=drifted,
        executed_constrained=drifted,
        executed_signal_diag=drifted,
        executed_allocated_diag=drifted,
        executed_desired_diag=drifted,
        executed_drift_compliance_forced=provenance.drift_compliance_forced,
        executed_drift_compliance_pending=provenance.drift_compliance_pending,
    )

    landed_trade = trades[trades["timestamp"] == landed_row].iloc[0]
    # The breach itself (organic drift, no trade landed yet) is never a
    # trade-log row -- the first row for this symbol after the initial
    # entry is the correction landing.
    assert landed_trade["previous_weight"] == pytest.approx(
        drifted.loc[breach_row, "A"]
    )
    assert landed_trade["new_weight"] == pytest.approx(0.6)
    assert landed_trade["adjustment_reason_codes"] == "drift_compliance"
    assert np.allclose(
        (trades["new_weight"] - trades["previous_weight"]).to_numpy(),
        trades["weight_change"].to_numpy(),
    )


def test_previous_weight_correct_for_a_maximum_turnover_deferred_catchup() -> None:
    """`previous_weight` across a `maximum_turnover`-throttled, multi-row
    catch-up must reflect each row's own true entering value -- reuses
    `test_maximum_turnover_caps_an_anchor_catch_up_and_carries_the_
    remainder`'s exact scenario (test_weight_drift.py), this time
    verifying the trade log's own invariant across both the partial-
    landing row and the remainder-landing row."""
    from quantlab.backtesting.accounting import apply_weight_drift

    # One extra leading flat (no-trade) row versus test_weight_drift.py's
    # own version of this scenario -- gives the initial entry (itself
    # subject to the turnover cap, per that test's own docstring) a valid
    # prior-period reference price; every index below is shifted by +1
    # accordingly (drift shock at 21, schedule at 22, etc.).
    n = 41
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    executed = pd.DataFrame(
        {"A": [0.0] + [0.5] * (n - 1), "B": [0.0] + [0.5] * (n - 1)}, index=dates
    )
    asset_returns = pd.DataFrame(
        {"A": [np.nan] + [0.0] * (n - 1), "B": [np.nan] + [0.0] * (n - 1)},
        index=dates,
    )
    asset_returns.loc[dates[21], "A"] = 0.20
    rebalance_date = pd.DataFrame(False, index=dates, columns=["A", "B"])
    rebalance_date.loc[dates[22]] = True
    cap = 0.05

    drifted, trade_changes, _provenance = apply_weight_drift(
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
    equity = pd.Series(100.0, index=dates)
    prices = pd.DataFrame({"A": [10.0] * n, "B": [10.0] * n}, index=dates)
    trades = _build(drifted, trade_changes, equity, prices)

    # B also has a real row on both these dates (the schedule flag marks
    # both columns "fresh" even though B's own target doesn't numerically
    # change; B's actual weight still drifted slightly, since A's outsized
    # gain grows total equity and so shrinks B's share of it) -- select A
    # specifically rather than assuming a single row per date.
    partial_row = trades[(trades["timestamp"] == dates[22]) & (trades["symbol"] == "A")]
    remainder_row = trades[
        (trades["timestamp"] == dates[23]) & (trades["symbol"] == "A")
    ]
    assert len(partial_row) == 1
    assert len(remainder_row) == 1
    # The value organic drift ACTUALLY pushed A to entering this row (0.5 *
    # 1.2 / 1.1), not 0.5 -- this is precisely the discrepancy this whole
    # fix is about: the previous ROW's own reported value (0.5, before the
    # shock landed) differs from what genuinely entered this row.
    assert partial_row.iloc[0]["previous_weight"] == pytest.approx(0.5 * 1.2 / 1.1)
    assert partial_row.iloc[0]["new_weight"] == pytest.approx(0.5204545454545454)
    assert remainder_row.iloc[0]["previous_weight"] == pytest.approx(0.5204545454545454)
    assert remainder_row.iloc[0]["new_weight"] == pytest.approx(0.5, abs=1e-9)
    assert np.allclose(
        (trades["new_weight"] - trades["previous_weight"]).to_numpy(),
        trades["weight_change"].to_numpy(),
    )


def test_new_minus_previous_always_equals_change_across_a_full_drift_run() -> None:
    """Property-style regression guard: across every row a broader,
    multi-asset drift scenario produces (mixed drift, scheduled
    rebalances, and a maximum_weight breach/correction all in one run),
    `new_weight - previous_weight == weight_change` must hold for EVERY
    trade-log row, not just the hand-picked ones the scenario-specific
    tests above check."""
    from quantlab.backtesting.accounting import apply_weight_drift

    # A leading flat (no-trade) row (see the other tests above) plus one
    # guaranteed large shock mixed into the random walk -- the fuzz alone
    # can't be relied on to reliably breach maximum_weight for every seed,
    # but the invariant below must be checked across a real correction,
    # not just ordinary small-drift rows.
    n = 61
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(3)
    a_returns = np.concatenate([[np.nan], rng.normal(0.0, 0.02, n - 1)])
    b_returns = np.concatenate([[np.nan], rng.normal(0.0, 0.015, n - 1)])
    a_returns[15] = 0.3
    executed = pd.DataFrame(
        {"A": [0.0] + [0.5] * (n - 1), "B": [0.0] + [0.5] * (n - 1)},
        index=dates,
    )
    asset_returns = pd.DataFrame({"A": a_returns, "B": b_returns}, index=dates)
    rebalance_date = pd.DataFrame(False, index=dates, columns=["A", "B"])
    rebalance_date.iloc[::10] = True

    drifted, trade_changes, provenance = apply_weight_drift(
        executed,
        asset_returns,
        None,
        None,
        maximum_weight=0.55,
        maximum_gross_exposure=None,
        maximum_net_exposure=None,
        long_only=False,
        rebalance_date=rebalance_date,
    )
    assert provenance.drift_compliance_forced.to_numpy().any(), (
        "expected at least one drift-compliance correction in this "
        "scenario -- otherwise the invariant check below doesn't "
        "actually exercise that path"
    )

    equity = pd.Series(100.0, index=dates)
    prices = pd.DataFrame({"A": [10.0] * n, "B": [10.0] * n}, index=dates)
    trades = _build(
        drifted,
        trade_changes,
        equity,
        prices,
        executed_desired=drifted,
        executed_constrained=drifted,
        executed_signal_diag=drifted,
        executed_allocated_diag=drifted,
        executed_desired_diag=drifted,
        executed_drift_compliance_forced=provenance.drift_compliance_forced,
        executed_drift_compliance_pending=provenance.drift_compliance_pending,
    )

    assert len(trades) > 0
    assert np.allclose(
        (trades["new_weight"] - trades["previous_weight"]).to_numpy(),
        trades["weight_change"].to_numpy(),
    )
    # side must always agree with the sign of weight_change.
    assert ((trades["side"] == "buy") == (trades["weight_change"] > 0)).all()


def test_trade_log_reason_frames_must_be_all_or_nothing() -> None:
    executed, changes, equity, prices = _inputs()

    with pytest.raises(BacktestError, match="all together or not at all"):
        _build(executed, changes, equity, prices, executed_desired=executed)


def test_trade_log_reason_frame_axes_must_match_executed_weights() -> None:
    executed, changes, equity, prices = _inputs()
    mismatched = executed.rename(columns={"AAA": "BBB"})
    reason_kwargs = {
        "executed_desired": executed,
        "executed_constrained": executed,
        "executed_signal_diag": executed,
        "executed_allocated_diag": executed,
        "executed_desired_diag": mismatched,
    }

    with pytest.raises(BacktestError, match="executed_desired_diag"):
        _build(executed, changes, equity, prices, **reason_kwargs)


def test_trade_log_populates_reason_when_all_frames_are_supplied() -> None:
    """A minimal end-to-end sanity check that supplying the reason frames
    actually reaches _classify_reason -- full pipeline scenarios (turnover
    cap, vol-targeting, tradability, ruin) live in test_trade_reasons.py."""
    executed, changes, equity, prices = _inputs()
    # The fill at index 1 goes 0.0 -> 1.0; make the desired target agree
    # (no constraint) and the signal change since the (implicit, flat)
    # prior rebalance, so this resolves to a clean strategy_signal fill.
    desired = executed.copy()
    signal = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=executed.index)
    reason_kwargs = {
        "executed_desired": desired,
        "executed_constrained": desired,
        "executed_signal_diag": signal,
        "executed_allocated_diag": desired,
        "executed_desired_diag": desired,
    }

    trades = _build(executed, changes, equity, prices, **reason_kwargs)

    assert trades["trigger_reason_code"].tolist() == ["strategy_signal"]
    assert trades["trigger_reason_detail_code"].tolist() == [None]
    assert trades["trigger_reason_details"].iloc[0] is not None
    assert trades["adjustment_reason_codes"].tolist() == [None]


def _base_reason_kwargs(
    executed: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    desired = executed.copy()
    signal = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=executed.index)
    return {
        "executed_desired": desired,
        "executed_constrained": desired,
        "executed_signal_diag": signal,
        "executed_allocated_diag": desired,
        "executed_desired_diag": desired,
    }


def test_trade_log_strategy_reason_frames_must_be_supplied_together() -> None:
    executed, changes, equity, prices = _inputs()

    with pytest.raises(BacktestError, match="must be supplied all together"):
        _build(
            executed,
            changes,
            equity,
            prices,
            **_base_reason_kwargs(executed),
            executed_strategy_reason_code=executed,
        )


def test_trade_log_strategy_reason_frames_require_the_base_reason_frames() -> None:
    executed, changes, equity, prices = _inputs()
    strategy_code = pd.DataFrame(
        {"AAA": [None, "oversold_entry", None]}, index=executed.index
    )

    with pytest.raises(BacktestError, match="requires the reason-attribution frames"):
        _build(
            executed,
            changes,
            equity,
            prices,
            executed_strategy_reason_code=strategy_code,
            executed_strategy_reason_details=strategy_code,
        )


def test_trade_log_populates_strategy_specific_reason_when_supplied() -> None:
    executed, changes, equity, prices = _inputs()
    strategy_code = pd.DataFrame(
        {"AAA": [None, "oversold_entry", None]}, index=executed.index, dtype=object
    )
    strategy_details = pd.DataFrame(
        {"AAA": [None, "z-score -2.5000 crossed entry threshold -2.0000", None]},
        index=executed.index,
        dtype=object,
    )

    trades = _build(
        executed,
        changes,
        equity,
        prices,
        **_base_reason_kwargs(executed),
        executed_strategy_reason_code=strategy_code,
        executed_strategy_reason_details=strategy_details,
    )

    assert trades["trigger_reason_code"].tolist() == ["strategy_signal"]
    assert trades["trigger_reason_detail_code"].tolist() == ["oversold_entry"]
    assert trades["trigger_reason_details"].iloc[0] == (
        "signal 0.0000 -> 1.0000 since last rebalance; "
        "z-score -2.5000 crossed entry threshold -2.0000"
    )


def test_trade_log_constraint_provenance_requires_the_base_reason_frames() -> None:
    executed, changes, equity, prices = _inputs()
    touch = _touch(
        touched=pd.DataFrame({"AAA": [False, True, False]}, index=executed.index),
        before=executed,
        after=executed,
    )

    with pytest.raises(BacktestError, match="requires the reason-attribution frames"):
        _build(
            executed,
            changes,
            equity,
            prices,
            constraint_provenance={"maximum_weight": touch},
        )


def test_trade_log_constraint_provenance_rejects_mismatched_axes() -> None:
    executed, changes, equity, prices = _inputs()
    mismatched = executed.rename(columns={"AAA": "BBB"})
    touch = _touch(touched=mismatched, before=executed, after=executed)

    pattern = r"constraint_provenance\['maximum_weight'\]"
    with pytest.raises(BacktestError, match=pattern):
        _build(
            executed,
            changes,
            equity,
            prices,
            **_base_reason_kwargs(executed),
            constraint_provenance={"maximum_weight": touch},
        )


def test_trade_log_populates_precise_constraint_code_when_provenance_is_supplied() -> (
    None
):
    executed, changes, equity, prices = _inputs()
    # Row 1 (the only fill) landed exactly on a post-constraint target
    # (1.0) that ConstraintSet itself already trimmed from the desired
    # 1.2 -- supplying provenance must yield the precise constraint name.
    desired = pd.DataFrame({"AAA": [0.0, 1.2, 1.0]}, index=executed.index)
    constrained = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=executed.index)
    reason_kwargs = _base_reason_kwargs(executed)
    reason_kwargs["executed_desired"] = desired
    reason_kwargs["executed_constrained"] = constrained
    touched = pd.DataFrame({"AAA": [False, True, False]}, index=executed.index)
    touch = _touch(touched=touched, before=desired, after=constrained)

    trades = _build(
        executed,
        changes,
        equity,
        prices,
        **reason_kwargs,
        constraint_provenance={"maximum_weight": touch},
    )

    assert trades["adjustment_reason_codes"].tolist() == ["maximum_weight"]
    assert trades["adjustment_reason_details"].iloc[0] == (
        "maximum_weight: 1.2000 -> 1.0000"
    )


@pytest.mark.parametrize(
    ("previous", "new", "expected"),
    [
        (0.0, 0.5, "entry_long"),
        (0.0, -0.5, "entry_short"),
        (0.5, 0.0, "exit_long"),
        (-0.5, 0.0, "exit_short"),
        (0.5, -0.5, "reverse_long_to_short"),
        (-0.5, 0.5, "reverse_short_to_long"),
        (0.3, 0.6, "increase_long"),
        (0.6, 0.3, "reduce_long"),
        (-0.3, -0.6, "increase_short"),
        (-0.6, -0.3, "reduce_short"),
        (0.6e-12, -0.6e-12, "flat_to_flat"),
    ],
)
def test_classify_action(previous: float, new: float, expected: str) -> None:
    assert _classify_action(previous, new) == expected


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


class _ReasonKwargs(TypedDict, total=False):
    new: float
    previous: float
    executed_desired: float
    executed_desired_prev: float
    executed_constrained: float
    signal_now: float
    signal_prev: float
    allocated_now: float
    allocated_prev: float
    desired_diag_now: float
    desired_diag_prev: float


def _reason_kwargs(**overrides: Any) -> _ReasonKwargs:
    """A baseline where nothing looks changed anywhere in the pipeline --
    each test overrides only the specific comparison it wants to exercise,
    so a passing test proves *that* branch fired, not an accidental
    combination of several at once."""
    base: dict[str, Any] = {
        "new": 0.5,
        "previous": 0.5,
        "executed_desired": 0.5,
        "executed_desired_prev": 0.5,
        "executed_constrained": 0.5,
        "signal_now": 1.0,
        "signal_prev": 1.0,
        "allocated_now": 0.5,
        "allocated_prev": 0.5,
        "desired_diag_now": 0.5,
        "desired_diag_prev": 0.5,
    }
    base.update(overrides)
    return cast(_ReasonKwargs, base)


def test_classify_reason_contributing_constraint() -> None:
    reason = _classify_reason(
        **_reason_kwargs(),
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.7892},
        constraint_after={"maximum_weight": 0.7000},
    )
    assert reason.adjustment_codes == "maximum_weight"
    assert reason.adjustment_details == "maximum_weight: 0.7892 -> 0.7000"
    assert reason.trigger_code is None


def test_classify_reason_redistribution_detail_text_is_stage_specific() -> None:
    """Each redistribution-capable constraint gets ITS OWN honest text --
    never a generic "another position was capped" sentence borrowed from
    maximum_weight."""
    for base_name, expected_fragment in (
        ("maximum_weight", "another position was capped"),
        ("minimum_weight", "dust/small positions were removed"),
        ("maximum_positions", "dropped to satisfy maximum_positions"),
    ):
        name = f"{base_name}_redistribution"
        reason = _classify_reason(
            **_reason_kwargs(),
            contributing_constraints=[name],
            constraint_before={name: 0.3761},
            constraint_after={name: 0.3770},
        )
        assert reason.adjustment_codes == name
        assert expected_fragment in (reason.adjustment_details or "")
        # Never implies the asset itself exceeded a threshold.
        assert "0.3761 -> 0.3770" in (reason.adjustment_details or "")


def test_classify_reason_tradability_touched() -> None:
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, executed_constrained=0.5),
        tradability_touched=True,
    )
    assert reason.adjustment_codes == "tradability"
    assert "closed" in (reason.adjustment_details or "")


def test_classify_reason_tradability_compliance_limited_has_distinct_text() -> None:
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, executed_constrained=0.5),
        tradability_touched=True,
        tradability_compliance_limited=True,
    )
    assert reason.adjustment_codes == "tradability"
    assert "feasibility limit" in (reason.adjustment_details or "")


def test_classify_reason_turnover_touched() -> None:
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, executed_constrained=0.5),
        turnover_touched=True,
        turnover_actively_limited=True,
    )
    assert reason.adjustment_codes == "turnover_cap"
    assert "turnover-capped" in (reason.adjustment_details or "")


def test_classify_reason_turnover_touched_catchup_has_distinct_text() -> None:
    """A row still catching up an earlier episode's debt, but not itself
    actively capped, must say so -- not claim it's being capped today."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, executed_constrained=0.5),
        turnover_touched=True,
        turnover_actively_limited=False,
    )
    assert reason.adjustment_codes == "turnover_cap"
    assert "previously deferred" in (reason.adjustment_details or "")


def test_classify_reason_multi_cause_adjustment_constraint_and_turnover() -> None:
    """The core bug this whole redesign fixes: a constraint AND turnover_
    cap acting on the SAME trade must both be visible, in ADJUSTMENT_ORDER
    (constraints before turnover_cap), never one masking the other."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, executed_constrained=0.5),
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.9},
        constraint_after={"maximum_weight": 0.5},
        turnover_touched=True,
        turnover_actively_limited=True,
    )
    assert reason.adjustment_codes == "maximum_weight+turnover_cap"
    assert "maximum_weight: 0.9000 -> 0.5000" in (reason.adjustment_details or "")
    assert "turnover_cap" in (reason.adjustment_details or "")


def test_classify_reason_multi_cause_adjustment_constraint_and_tradability() -> None:
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, executed_constrained=0.5),
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.9},
        constraint_after={"maximum_weight": 0.5},
        tradability_touched=True,
    )
    assert reason.adjustment_codes == "maximum_weight+tradability"


def test_classify_reason_trigger_and_adjustment_coexist() -> None:
    """A strategy-driven entry that is ALSO capped by a constraint must
    show BOTH -- the original masking bug this redesign fixes."""
    reason = _classify_reason(
        **_reason_kwargs(
            new=0.3,
            executed_constrained=0.5,
            signal_now=1.0,
            signal_prev=0.0,
        ),
        strategy_detail_code="oversold_entry",
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.9},
        constraint_after={"maximum_weight": 0.5},
    )
    assert reason.trigger_code == "strategy_signal"
    assert reason.trigger_detail_code == "oversold_entry"
    assert reason.adjustment_codes == "maximum_weight"


def test_classify_reason_strategy_signal_wins_over_downstream_changes() -> None:
    """Even when the allocator/desired-target ALSO changed (a signal change
    always cascades downstream), strategy_signal must be reported -- the
    most upstream, most specific cause -- not portfolio_rebalance or
    volatility_target_adjustment."""
    reason = _classify_reason(
        **_reason_kwargs(
            signal_now=1.0,
            signal_prev=0.0,
            allocated_now=0.6,
            allocated_prev=0.4,
            desired_diag_now=0.6,
            desired_diag_prev=0.4,
        )
    )
    assert (reason.trigger_code, reason.trigger_detail_code) == (
        "strategy_signal",
        None,
    )


def test_classify_reason_portfolio_rebalance() -> None:
    reason = _classify_reason(**_reason_kwargs(allocated_now=0.6, allocated_prev=0.4))
    assert (reason.trigger_code, reason.trigger_detail_code) == (
        "portfolio_rebalance",
        None,
    )


def test_classify_reason_volatility_target_adjustment() -> None:
    reason = _classify_reason(
        **_reason_kwargs(desired_diag_now=0.6, desired_diag_prev=0.4)
    )
    assert (reason.trigger_code, reason.trigger_detail_code) == (
        "volatility_target_adjustment",
        None,
    )


def test_classify_reason_position_rescaling_when_target_still_drifting() -> None:
    """No trigger, no known adjustment layer, but the pre-turnover target
    itself is still drifting row-over-row -- the pairs_trading price/beta
    residual case."""
    reason = _classify_reason(
        **_reason_kwargs(
            new=0.5, previous=0.3, executed_desired=0.55, executed_desired_prev=0.5
        )
    )
    assert reason.adjustment_codes == "position_rescaling"
    assert reason.trigger_code is None


def test_classify_reason_deferred_catchup_when_target_is_static() -> None:
    """Nothing upstream changed since the last rebalance, the pre-turnover
    target has been STATIC, yet the position still moved -- a turnover-
    cap/tradability shortfall completing with no real cause identifiable
    (genuinely unknown, not one of the real provenance signals)."""
    reason = _classify_reason(**_reason_kwargs(new=0.5, previous=0.3))
    assert reason.adjustment_codes == "deferred_catchup"
    assert reason.trigger_code is None


def test_classify_reason_position_rescaling_never_fires_alongside_a_trigger() -> None:
    """Strict fallback guard (point 1): a value combination that would
    satisfy position_rescaling's own condition must still be preempted by
    a real trigger -- position_rescaling is reached ONLY via the `elif`
    after trigger is confirmed None."""
    reason = _classify_reason(
        **_reason_kwargs(
            new=0.5,
            previous=0.3,
            executed_desired=0.55,
            executed_desired_prev=0.5,
            signal_now=1.0,
            signal_prev=0.0,
        )
    )
    assert reason.trigger_code == "strategy_signal"
    assert reason.adjustment_codes is None


def test_classify_reason_position_rescaling_never_fires_alongside_real_adjustment() -> (
    None
):
    """Strict fallback guard: a real adjustment layer (here, a contributing
    constraint) must preempt position_rescaling even though the drifting-
    target condition also holds."""
    reason = _classify_reason(
        **_reason_kwargs(
            new=0.3,
            previous=0.3,
            executed_constrained=0.5,
            executed_desired=0.55,
            executed_desired_prev=0.5,
        ),
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.9},
        constraint_after={"maximum_weight": 0.5},
    )
    assert reason.adjustment_codes == "maximum_weight"
    assert "position_rescaling" not in reason.adjustment_codes


def test_classify_reason_unknown_when_nothing_explains_the_row() -> None:
    reason = _classify_reason(**_reason_kwargs())
    assert reason == TradeReason(
        trigger_code="unknown",
        trigger_detail_code=None,
        trigger_details="no upstream driver identified",
        adjustment_codes=None,
        adjustment_details=None,
    )


def test_classify_reason_strategy_detail_code_overrides_the_generic_text() -> None:
    """reason_detail_code becomes the precise code, but reason_details
    keeps the generic "signal X -> Y" text WITH the strategy-specific
    text appended, never the specific text alone."""
    reason = _classify_reason(
        **_reason_kwargs(
            signal_now=1.0,
            signal_prev=0.0,
            strategy_detail_code="oversold_entry",
            strategy_details="z-score -2.5000 crossed entry threshold -2.0000",
        )
    )
    assert (reason.trigger_code, reason.trigger_detail_code) == (
        "strategy_signal",
        "oversold_entry",
    )
    assert reason.trigger_details == (
        "signal 0.0000 -> 1.0000 since last rebalance; "
        "z-score -2.5000 crossed entry threshold -2.0000"
    )


def test_classify_reason_strategy_signal_without_detail_code_is_unchanged() -> None:
    reason = _classify_reason(**_reason_kwargs(signal_now=1.0, signal_prev=0.0))
    assert (reason.trigger_code, reason.trigger_detail_code) == (
        "strategy_signal",
        None,
    )
    assert reason.trigger_details is not None
    assert "since last rebalance" in reason.trigger_details


def test_classify_reason_forced_liquidation_overrides_every_other_adjustment() -> None:
    """Once ruined, no other layer's specific clip value still explains
    the executed weight -- forced_liquidation replaces the whole
    adjustment list rather than composing with it."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.0, previous=0.5, executed_constrained=0.5),
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.9},
        constraint_after={"maximum_weight": 0.5},
        turnover_touched=True,
        turnover_actively_limited=True,
        forced_liquidation=True,
    )
    assert reason.adjustment_codes == "forced_liquidation"


def test_classify_reason_forced_liquidation_never_overrides_trigger() -> None:
    """The strategy's own wish (trigger) survives even when the executed
    weight was forced to zero -- the two are independent concepts."""
    reason = _classify_reason(
        **_reason_kwargs(
            new=0.0,
            previous=0.5,
            executed_constrained=0.5,
            signal_now=1.0,
            signal_prev=0.5,
        ),
        forced_liquidation=True,
    )
    assert reason.trigger_code == "strategy_signal"
    assert reason.adjustment_codes == "forced_liquidation"


def test_classify_reason_drift_compliance_overrides_ordinary_constraints() -> None:
    """A row whose magnitude comes from the drift-compliance LP is NOT
    decision-pipeline-driven at all, so it overrides an ordinary
    constraint adjustment that would otherwise also apply to the same
    row -- the constraint's own before/after clip value becomes moot."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, previous=0.5, executed_constrained=0.5),
        contributing_constraints=["maximum_weight"],
        constraint_before={"maximum_weight": 0.9},
        constraint_after={"maximum_weight": 0.5},
        drift_compliance_forced=True,
    )
    assert reason.adjustment_codes == "drift_compliance"


def test_classify_reason_drift_compliance_pending_is_its_own_code() -> None:
    """A still-unresolved drift breach (responsible symbol/group still
    untradable) gets its own distinct code, not conflated with a landed
    correction."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.3, previous=0.3), drift_compliance_pending=True
    )
    assert reason.adjustment_codes == "drift_compliance_pending"


def test_classify_reason_stop_loss_overrides_drift_compliance() -> None:
    """A stop-loss/take-profit breach detected on the drift-corrected
    weight is a still more specific, more severe cause and wins over a
    drift-compliance adjustment on the same row."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.0, previous=0.4),
        drift_compliance_forced=True,
        stop_loss_triggered=True,
    )
    assert reason.adjustment_codes == "stop_loss"


def test_classify_reason_forced_liquidation_overrides_drift_compliance() -> None:
    """Portfolio ruin is more severe than a drift-compliance correction."""
    reason = _classify_reason(
        **_reason_kwargs(new=0.0, previous=0.4),
        drift_compliance_forced=True,
        forced_liquidation=True,
    )
    assert reason.adjustment_codes == "forced_liquidation"


def test_classify_reason_every_branch_emits_only_adjustment_order_codes() -> None:
    """Exhaustive sweep of every branch of _classify_reason (point 3):
    every code it can ever emit must be a member of ADJUSTMENT_ORDER."""
    scenarios: list[dict[str, Any]] = [
        {
            "contributing_constraints": ["maximum_weight"],
            "constraint_before": {"maximum_weight": 0.9},
            "constraint_after": {"maximum_weight": 0.5},
        },
        {
            "contributing_constraints": ["maximum_weight_redistribution"],
            "constraint_before": {"maximum_weight_redistribution": 0.3},
            "constraint_after": {"maximum_weight_redistribution": 0.31},
        },
        {"tradability_touched": True},
        {"tradability_touched": True, "tradability_compliance_limited": True},
        {"turnover_touched": True, "turnover_actively_limited": True},
        {"turnover_touched": True, "turnover_actively_limited": False},
        {"stop_loss_triggered": True},
        {"take_profit_triggered": True},
        {"drift_compliance_forced": True},
        {"drift_compliance_pending": True},
        {"forced_liquidation": True},
    ]
    for extra in scenarios:
        reason = _classify_reason(
            **_reason_kwargs(new=0.3, previous=0.3, executed_constrained=0.5), **extra
        )
        if reason.adjustment_codes is not None:
            for code in reason.adjustment_codes.split("+"):
                assert code in ADJUSTMENT_ORDER

    fallback_reason = _classify_reason(
        **_reason_kwargs(
            new=0.5, previous=0.3, executed_desired=0.55, executed_desired_prev=0.5
        )
    )
    assert fallback_reason.adjustment_codes in ADJUSTMENT_ORDER
    catchup_reason = _classify_reason(**_reason_kwargs(new=0.5, previous=0.3))
    assert catchup_reason.adjustment_codes in ADJUSTMENT_ORDER


# --------------------------------------------------------------------------- #
# serialize_adjustment_codes / parse_adjustment_codes
# --------------------------------------------------------------------------- #
def test_serialize_adjustment_codes_orders_by_pipeline_order_not_input_order() -> None:
    assert (
        serialize_adjustment_codes(["turnover_cap", "maximum_weight", "tradability"])
        == "maximum_weight+tradability+turnover_cap"
    )


def test_serialize_adjustment_codes_single_name_has_no_separator() -> None:
    assert serialize_adjustment_codes(["maximum_weight"]) == "maximum_weight"


def test_serialize_adjustment_codes_deduplicates() -> None:
    assert (
        serialize_adjustment_codes(["maximum_weight", "maximum_weight", "tradability"])
        == "maximum_weight+tradability"
    )


def test_serialize_adjustment_codes_covers_every_adjustment_order_entry() -> None:
    """Round-trips the full canonical order in one call as a sanity check
    that ADJUSTMENT_ORDER and serialize_adjustment_codes stay in sync."""
    assert serialize_adjustment_codes(ADJUSTMENT_ORDER) == "+".join(ADJUSTMENT_ORDER)


def test_serialize_adjustment_codes_rejects_unknown_code() -> None:
    with pytest.raises(BacktestError, match="Unknown adjustment code"):
        serialize_adjustment_codes(["not_a_real_code"])


def test_parse_adjustment_codes_round_trips_serialize() -> None:
    codes = ["maximum_gross_exposure", "long_only"]
    assert parse_adjustment_codes(serialize_adjustment_codes(codes)) == [
        "long_only",
        "maximum_gross_exposure",
    ]


def test_parse_adjustment_codes_single_token_has_nothing_to_split() -> None:
    assert parse_adjustment_codes("tradability") == ["tradability"]


def test_parse_adjustment_codes_strict_rejects_unknown_code() -> None:
    with pytest.raises(BacktestError, match="Unknown adjustment code"):
        parse_adjustment_codes("not_a_real_code")


def test_parse_adjustment_codes_permissive_preserves_unknown_code() -> None:
    assert parse_adjustment_codes("not_a_real_code", strict=False) == [
        "not_a_real_code"
    ]
    assert parse_adjustment_codes("maximum_weight+not_a_real_code", strict=False) == [
        "maximum_weight",
        "not_a_real_code",
    ]
