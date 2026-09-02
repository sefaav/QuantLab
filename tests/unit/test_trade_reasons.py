"""End-to-end reason-attribution scenarios via the real BacktestEngine.

Unlike test_trade_log.py's direct _classify_reason unit tests (which pin
down the classifier's priority logic in isolation), these tests exercise
the actual engine-side plumbing added to feed it: capturing signals/
allocated/desired-target, resampling them to rebalance dates, and aligning
them to accounting.executed_weights via the same executed_weights() shift
run_accounting uses internally. A scripted, deterministic BaseStrategy
subclass is used throughout instead of a registered strategy name, for
exact control over when the signal changes -- passed directly to
BacktestEngine.run() (which takes strategy/allocator as instances, not
resolved from config), matching the pattern already established by
test_reporting_hardening.py.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from quantlab.backtesting.engine import BacktestEngine
from quantlab.config import ExperimentConfig
from quantlab.execution.execution_model import ExecutionModel
from quantlab.portfolio.allocator import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    build_allocator,
)
from quantlab.portfolio.rebalancing import rebalance_dates
from quantlab.strategies.base import BaseStrategy, SignalReasons
from quantlab.strategies.mean_reversion import MeanReversionStrategy


class _ScriptedStrategy(BaseStrategy):
    """Returns a hand-specified signal path, ignoring the market data.

    ``schedule`` maps a symbol to its full signal path (one value per row
    of whatever ``data`` the engine hands it, aligned positionally) --
    lets a test dictate exactly which date a signal changes, rather than
    reverse-engineering a real strategy's parameters to do it indirectly.
    """

    name = "scripted"

    def __init__(self, schedule: dict[str, list[float]]) -> None:
        self.schedule = schedule

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        prices = self._prices(data)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for symbol, path in self.schedule.items():
            signals[symbol] = path
        return self._validate_signals(signals, prices)


class _ScriptedStrategyWithReasons(BaseStrategy):
    """Like ``_ScriptedStrategy``, but also implements ``explain_signals()``
    with a hand-specified per-row reason schedule.

    Lets a test dictate exactly which RAW row carries the "true"
    transition reason, independent of which row a later rebalance/
    execution step ends up consuming it on -- the crux of the alignment
    fix under test.
    """

    name = "scripted_with_reasons"

    def __init__(
        self,
        schedule: dict[str, list[float]],
        reason_schedule: dict[str, list[str | None]],
    ) -> None:
        self.schedule = schedule
        self.reason_schedule = reason_schedule

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        prices = self._prices(data)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for symbol, path in self.schedule.items():
            signals[symbol] = path
        return self._validate_signals(signals, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons:
        prices = self._prices(data)
        detail_code = np.full(prices.shape, None, dtype=object)
        details = np.full(prices.shape, None, dtype=object)
        for symbol, path in self.reason_schedule.items():
            column_index = prices.columns.get_loc(symbol)
            for row_index, code in enumerate(path):
                if code is not None:
                    detail_code[row_index, column_index] = code
                    details[row_index, column_index] = f"scripted: {code}"
        return self._validate_signal_reasons(
            pd.DataFrame(
                detail_code, index=prices.index, columns=prices.columns, dtype=object
            ),
            pd.DataFrame(
                details, index=prices.index, columns=prices.columns, dtype=object
            ),
            prices,
        )


def _config(**portfolio_overrides: object) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "trade_reasons",
            "data": {
                "instruments": [{"symbol": "A", "source": "csv", "calendar": "XNYS"}],
                "start_date": "2020-01-01",
                "end_date": "2020-03-01",
            },
            "strategy": {"name": "buy_and_hold"},  # unused: an instance is passed
            "portfolio": {"allocator": "equal_weight", **portfolio_overrides},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000},
        }
    )


def _run(schedule: dict[str, list[float]], config: ExperimentConfig) -> pd.DataFrame:
    n = len(next(iter(schedule.values())))
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        _ScriptedStrategy(schedule),
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    return result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)


def _run_strategy(
    strategy: BaseStrategy, prices: list[float], config: ExperimentConfig
) -> pd.DataFrame:
    data = make_ohlcv("A", prices, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    return result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)


def test_turnover_cap_then_deferred_catchup_on_the_same_entry() -> None:
    """Signal jumps 0 -> 1 and stays there; a tight maximum_turnover (0.5)
    forces the entry to land in two exact half-steps (0.5 then 1.0) instead
    of one -- round numbers (0.5 x 2 = 1.0 exactly) sidestep float-precision
    flakiness. Under the trigger/adjustment model, BOTH fills carry a real
    `turnover_cap` adjustment (the second is an episode-scoped catch-up of
    the same still-unresolved decision, not `deferred_catchup` -- that
    fallback is reserved for a genuinely unknown cause) -- and the FIRST
    fill ALSO correctly shows the real `strategy_signal` trigger that
    caused it, no longer masked by the constraint (the core bug this
    redesign fixes)."""
    n = 20
    schedule = {"A": [0.0] * 5 + [1.0] * (n - 5)}
    config = _config(rebalance_frequency="daily", maximum_turnover=0.5)

    trades = _run(schedule, config)

    assert len(trades) == 2
    assert trades.loc[0, "action"] == "entry_long"
    assert trades.loc[0, "trigger_reason_code"] == "strategy_signal"
    assert trades.loc[0, "adjustment_reason_codes"] == "turnover_cap"
    assert "turnover-capped" in str(trades.loc[0, "adjustment_reason_details"])
    assert trades.loc[1, "action"] == "increase_long"
    assert trades.loc[1, "trigger_reason_code"] is None
    assert trades.loc[1, "adjustment_reason_codes"] == "turnover_cap"
    assert "previously deferred" in str(trades.loc[1, "adjustment_reason_details"])


def test_clean_signal_driven_entry_with_no_binding_constraint() -> None:
    """No turnover cap, no portfolio constraint active -- the entry reaches
    its full desired size in one fill, so this is a clean strategy_signal,
    and (being the symbol's very first trade) needs no special-casing."""
    n = 10
    schedule = {"A": [0.0] * 3 + [1.0] * (n - 3)}
    config = _config(rebalance_frequency="daily")

    trades = _run(schedule, config)

    assert len(trades) == 1
    assert trades.loc[0, "previous_weight"] == pytest.approx(0.0)
    assert trades.loc[0, "new_weight"] == pytest.approx(1.0)
    assert trades.loc[0, "action"] == "entry_long"
    assert trades.loc[0, "trigger_reason_code"] == "strategy_signal"
    assert trades.loc[0, "trigger_reason_detail_code"] is None
    assert trades.loc[0, "trigger_reason_details"] is not None
    assert trades.loc[0, "adjustment_reason_codes"] is None


def test_reverse_long_to_short_reports_the_correct_action_and_reason() -> None:
    """A single-step long-to-short flip, unconstrained -- the action must
    say "reverse", not the generic side="sell" a naive buy/sell label would
    give a covering-and-shorting fill indistinguishable from a partial
    reduction."""
    n = 12
    schedule = {"A": [0.0] * 3 + [1.0] * 3 + [-1.0] * (n - 6)}
    config = _config(rebalance_frequency="daily")

    trades = _run(schedule, config)

    assert len(trades) == 2
    assert trades.loc[0, "action"] == "entry_long"
    flip = trades.loc[1]
    assert cast(float, flip["previous_weight"]) == pytest.approx(1.0)
    assert cast(float, flip["new_weight"]) == pytest.approx(-1.0)
    assert cast(str, flip["action"]) == "reverse_long_to_short"
    assert cast(str, flip["trigger_reason_code"]) == "strategy_signal"


def test_portfolio_constraint_sub_code_when_max_weight_trims_the_target() -> None:
    """maximum_weight=0.4 trims the allocator's desired 1.0 down to 0.4
    inside ConstraintSet itself, before turnover-cap ever runs (no turnover
    cap configured here) -- must read as the precise constraint name
    (maximum_weight) in `adjustment_reason_codes`. Under the trigger/
    adjustment model, this trade ALSO correctly keeps its `strategy_signal`
    trigger -- the entry is no longer masked by the constraint that capped
    its size (the core bug this redesign fixes)."""
    n = 10
    schedule = {"A": [0.0] * 3 + [1.0] * (n - 3)}
    config = _config(rebalance_frequency="daily", maximum_weight=0.4)

    trades = _run(schedule, config)

    assert len(trades) == 1
    assert trades.loc[0, "new_weight"] == pytest.approx(0.4)
    assert trades.loc[0, "trigger_reason_code"] == "strategy_signal"
    assert trades.loc[0, "adjustment_reason_codes"] == "maximum_weight"


def test_multiple_constraints_combine_into_one_adjustment_reason_codes() -> None:
    """maximum_weight caps the single desired weight (1.0 -> 0.5), then
    maximum_gross_exposure (tighter than the post-cap gross) rescales it
    further (0.5 -> 0.3) -- both constraints genuinely fired on the same
    fill and must both show up, joined via the canonical "+" convention
    in pipeline-execution order, not just the last one to run -- alongside
    the real `strategy_signal` trigger that caused the entry."""
    n = 10
    schedule = {"A": [0.0] * 3 + [1.0] * (n - 3)}
    config = _config(
        rebalance_frequency="daily", maximum_weight=0.5, maximum_gross_exposure=0.3
    )

    trades = _run(schedule, config)

    assert len(trades) == 1
    assert trades.loc[0, "new_weight"] == pytest.approx(0.3)
    assert trades.loc[0, "trigger_reason_code"] == "strategy_signal"
    expected_codes = "maximum_weight+maximum_gross_exposure"
    assert trades.loc[0, "adjustment_reason_codes"] == expected_codes
    details = str(trades.loc[0, "adjustment_reason_details"])
    assert "maximum_weight" in details
    assert "maximum_gross_exposure" in details


def test_mean_reversion_strategy_signal_gets_precise_reason_via_full_engine() -> None:
    """The strategy-specific reason from MeanReversionStrategy.
    explain_signals() must reach the trade log through the FULL engine
    pipeline -- rebalance-date sampling, the extra-delay shift and the
    executed_weights alignment, via the positional gather in engine.py --
    not just work when the classifier or the strategy is tested alone."""
    n = 60
    prices = list(np.full(40, 100.0)) + list(np.linspace(100, 70, n - 40))
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "mean_reversion_full_engine",
            "data": {
                "instruments": [{"symbol": "A", "source": "csv", "calendar": "XNYS"}],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
            },
            "strategy": {"name": "mean_reversion"},  # unused: an instance is passed
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000},
        }
    )
    strategy = MeanReversionStrategy(
        lookback_period=20, entry_threshold=1.5, exit_threshold=0.5, long_only=True
    )

    trades = _run_strategy(strategy, prices, config)

    entries = trades[trades["action"] == "entry_long"]
    assert len(entries) == 1
    assert entries.iloc[0]["trigger_reason_code"] == "strategy_signal"
    assert entries.iloc[0]["trigger_reason_detail_code"] == "oversold_entry"
    assert "entry threshold" in entries.iloc[0]["trigger_reason_details"]


# --------------------------------------------------------------------------- #
# position_strategy_origin: driven purely by decision_proxy's own regime
# (flat/long/short via sign) -- cleared only when decision_proxy itself
# returns to flat, never by a downstream layer, and insensitive to a
# continuous signal's own magnitude drift.
# --------------------------------------------------------------------------- #
def test_position_strategy_origin_entry_exit_flat_then_new_entry() -> None:
    n = 20
    schedule = {"A": [0.0] * 3 + [1.0] * 4 + [0.0] * 4 + [1.0] * (n - 11)}
    reason_schedule: dict[str, list[str | None]] = {"A": [None] * n}
    reason_schedule["A"][3] = "first_entry"
    reason_schedule["A"][7] = "first_exit"
    reason_schedule["A"][11] = "second_entry"
    config = _config(rebalance_frequency="daily")
    strategy = _ScriptedStrategyWithReasons({"A": schedule["A"]}, reason_schedule)
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)
    entries = trades[trades["action"] == "entry_long"].reset_index(drop=True)
    exits = trades[trades["action"] == "exit_long"].reset_index(drop=True)

    assert len(entries) == 2
    assert len(exits) == 1
    assert entries.loc[0, "position_strategy_origin_code"] == "first_entry"
    # Origin is cleared the moment decision_proxy itself returns to flat.
    assert pd.isna(exits.loc[0, "position_strategy_origin_timestamp"])
    assert exits.loc[0, "position_strategy_origin_code"] is None
    assert entries.loc[1, "position_strategy_origin_code"] == "second_entry"
    assert cast(
        pd.Timestamp, entries.loc[1, "position_strategy_origin_timestamp"]
    ) > cast(pd.Timestamp, entries.loc[0, "position_strategy_origin_timestamp"])


def test_position_strategy_origin_reversal_replaces_not_merges() -> None:
    n = 15
    schedule = {"A": [0.0] * 3 + [1.0] * 4 + [-1.0] * (n - 7)}
    reason_schedule: dict[str, list[str | None]] = {"A": [None] * n}
    reason_schedule["A"][3] = "long_entry"
    reason_schedule["A"][7] = "short_entry"
    config = _config(rebalance_frequency="daily")
    strategy = _ScriptedStrategyWithReasons({"A": schedule["A"]}, reason_schedule)
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    reversal = trades[trades["action"] == "reverse_long_to_short"].iloc[0]
    assert reversal["position_strategy_origin_code"] == "short_entry"


def test_position_strategy_origin_insensitive_to_continuous_magnitude_drift() -> None:
    """A continuous signal's own magnitude drift (0.4 -> 0.5 -> 0.3 -> 0.6,
    same regime throughout) must never recreate the origin -- only a
    flat<->non-flat regime change does. A second, constant-signal symbol B
    is included so signal_proportional's relative split actually moves A's
    executed weight as A's own magnitude drifts (a single-asset universe
    would always normalize to a constant sign-only weight, masking drift)."""
    n = 20
    schedule_a = [0.0] * 3 + [0.4, 0.5, 0.3, 0.6] + [0.0] * (n - 7)
    schedule_b = [0.5] * n
    reason_schedule: dict[str, list[str | None]] = {
        "A": [None] * n,
        "B": [None] * n,
    }
    reason_schedule["A"][3] = "continuous_entry"
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "trade_reasons_continuous_drift",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "B", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-03-01",
            },
            "strategy": {"name": "buy_and_hold"},  # unused: an instance is passed
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000},
        }
    )
    strategy = _ScriptedStrategyWithReasons(
        {"A": schedule_a, "B": schedule_b}, reason_schedule
    )
    data_a = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    data_b = make_ohlcv("B", [100.0] * n, start="2020-01-01")
    data = pd.concat([data_a, data_b], ignore_index=True)
    result = BacktestEngine().run(
        data,
        strategy,
        build_allocator("signal_proportional"),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)
    held_trades = trades[
        trades["action"].isin(["entry_long", "increase_long", "reduce_long"])
    ]

    assert len(held_trades) >= 2  # entry + at least one magnitude-drift fill
    origins = held_trades["position_strategy_origin_code"].unique().tolist()
    assert origins == ["continuous_entry"]
    timestamps = held_trades["position_strategy_origin_timestamp"].unique()
    assert len(timestamps) == 1


def test_position_strategy_origin_survives_allocator_warmup_before_first_trade() -> (
    None
):
    """buy_and_hold-style scenario (points 2/7): the strategy's decision
    becomes active immediately, but InverseVolatilityAllocator only
    produces its first non-zero weight once its volatility window fills
    -- the resulting first trade's TRIGGER is portfolio_rebalance (no NEW
    signal transition that day), but position_strategy_origin must still
    correctly point back to the original strategic decision -- confirming
    it tracks the strategic regime, not the executed-weight episode."""
    n = 40
    schedule = {"A": [1.0] * n}  # active from day 0
    reason_schedule: dict[str, list[str | None]] = {"A": [None] * n}
    reason_schedule["A"][0] = "price_became_available"
    config = _config(rebalance_frequency="daily")
    strategy = _ScriptedStrategyWithReasons(schedule, reason_schedule)
    rng = np.random.default_rng(0)
    prices = list(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n)))
    data = make_ohlcv("A", prices, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        # A short volatility_window guarantees the warmup mismatch
        # (signal active from day 0, allocator only produces its first
        # non-zero weight once its own window fills) resolves well within
        # this test's 40-row window.
        InverseVolatilityAllocator(volatility_window=10),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    assert len(trades) >= 1
    first_trade = trades.iloc[0]
    assert first_trade["action"] == "entry_long"
    assert first_trade["trigger_reason_code"] == "portfolio_rebalance"
    assert first_trade["position_strategy_origin_code"] == "price_became_available"
    assert pd.notna(first_trade["position_strategy_origin_timestamp"])
    assert first_trade["position_strategy_origin_timestamp"] <= first_trade["timestamp"]


def test_forced_liquidation_after_ruin_is_correctly_unrepresented_not_fabricated() -> (
    None
):
    """A full short position hit by a price spike large enough to floor
    the period return at exactly -100% (see accounting.py's own
    `_floor_at_total_loss`) ruins the portfolio. `AccountingResult.
    force_flat` (see `_run_accounting_steps`'s own docstring: "preventing
    a closing trade with no remaining capital") deliberately zeroes
    `weight_changes` on every ruined date -- there is genuinely no
    capital left to execute a real closing trade, so NO trade-log row is
    -- correctly -- ever produced for the liquidation moment itself, even
    though `positions`/`equity_curve` show the position and equity
    dropping to zero there. The `forced_liquidation` adjustment wiring
    (`executed_forced_liquidation`, sourced from the real
    `AccountingResult.ruined`) exists for whichever cell/row combination
    WOULD have a recorded change on a ruined date; today's accounting
    semantics make that combination unreachable, and this test pins that
    down explicitly so a future accounting change that DOES produce such
    a row is caught by the adjacent `_classify_reason` unit test
    (`test_classify_reason_forced_liquidation_*` in test_trade_log.py)
    rather than silently reverting to `unknown`/`deferred_catchup`.

    `model_weight_drift` is explicitly disabled: a position held constant
    for several days before one catastrophic spike is exactly the shape
    `apply_weight_drift`'s OWN, separate per-episode bankruptcy guard
    (`E <= EPSILON`) is designed to catch -- with drift enabled (the
    default), that guard fires first and flattens the position before
    `AccountingResult`'s own absolute-equity `ruined` mechanism, the one
    this test specifically targets, ever sees the floored return. The two
    guards are independent and BOTH legitimate (see `apply_weight_drift`'s
    own docstring); this test isolates the one it is actually about."""
    n = 15
    schedule = {"A": [0.0] * 3 + [-1.0] * (n - 3)}  # full short from day 3
    # Price flat through day 4, then a +300% spike on day 5 -- applied to
    # the already-short position decided the prior day, this floors that
    # period's return at exactly -100% (total loss).
    prices = [100.0] * 5 + [400.0] * (n - 5)
    config = _config(rebalance_frequency="daily", model_weight_drift=False)
    data = make_ohlcv("A", prices, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        _ScriptedStrategy(schedule),
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    # Ruin genuinely happened (equity hits exactly zero, and the position
    # is force-flattened starting the NEXT date -- `ruined` itself is a
    # shift(1) of the equity condition, see run_accounting) ...
    assert (result.equity_curve <= 0.0).any()
    first_ruin_date = result.equity_curve.index[result.equity_curve <= 0.0][0]
    after_ruin = result.positions["A"].loc[result.positions.index > first_ruin_date]
    assert len(after_ruin) > 0
    assert (after_ruin == 0.0).all()
    # ... but no trade row exists for it (no capital to execute a real
    # closing trade) -- exactly one trade total, the original entry.
    assert len(trades) == 1
    assert trades.iloc[0]["action"] == "entry_short"
    # No row anywhere is left unattributed or mis-attributed as a result.
    assert not (trades["trigger_reason_code"] == "unknown").any()
    assert not (trades["adjustment_reason_codes"] == "deferred_catchup").any()


# --------------------------------------------------------------------------- #
# Alignment: the strategy-specific reason must follow the transition that
# actually produced the executed signal value, not the row a later
# rebalance/execution step happens to sample.
# --------------------------------------------------------------------------- #
def _weekly_rebalance_positions(n: int) -> list[int]:
    """Row positions of each weekly rebalance date for an n-row XNYS index
    starting 2020-01-01 -- computed via the real rebalance_dates(), not
    guessed, so the test never silently drifts from actual calendar
    behaviour."""
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    index = pd.DatetimeIndex(data["timestamp"].drop_duplicates().sort_values())
    dates = rebalance_dates(index, "weekly", calendar="XNYS")
    return [int(position) for position in index.get_indexer(dates)]


def test_off_cycle_strategy_reason_survives_until_the_next_rebalance() -> None:
    """A transition on a day that is NOT a rebalance date must still be
    the reason attached to the trade once the persisting signal is
    finally rebalanced/executed -- not None (the sampled rebalance-date
    row's own, empty, reason cell) and not some unrelated later reason."""
    n = 30
    positions = _weekly_rebalance_positions(n)
    # positions e.g. [0, 3, 8, 13, ...] -- pick a row strictly between the
    # 2nd and 3rd rebalance date (never rely on a fixed magic number).
    previous_rebalance, next_rebalance = positions[1], positions[2]
    off_cycle = previous_rebalance + 1
    assert off_cycle < next_rebalance, "need at least one off-cycle row"

    schedule_a = [0.0] * off_cycle + [1.0] * (n - off_cycle)
    reason_schedule_a: list[str | None] = [None] * n
    reason_schedule_a[off_cycle] = "off_cycle_entry"

    config = _config(rebalance_frequency="weekly")
    strategy = _ScriptedStrategyWithReasons({"A": schedule_a}, {"A": reason_schedule_a})
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    assert len(trades) == 1
    assert trades.loc[0, "trigger_reason_code"] == "strategy_signal"
    assert trades.loc[0, "trigger_reason_detail_code"] == "off_cycle_entry"
    assert "scripted: off_cycle_entry" in str(trades.loc[0, "trigger_reason_details"])


def test_the_last_of_two_off_cycle_transitions_wins() -> None:
    """Two transitions happen before the next rebalance consumes them --
    the trade must carry the LATER transition's reason, not the earlier
    one (a newer transition always overwrites an older one, see
    engine.py's last_transition_seed)."""
    n = 30
    positions = _weekly_rebalance_positions(n)
    previous_rebalance, next_rebalance = positions[1], positions[2]
    first_step = previous_rebalance + 1
    second_step = first_step + 1
    assert second_step < next_rebalance, (
        "need two off-cycle rows before the next rebalance"
    )

    schedule_a = (
        [0.0] * first_step
        + [0.5] * (second_step - first_step)
        + [1.0] * (n - second_step)
    )
    reason_schedule_a: list[str | None] = [None] * n
    reason_schedule_a[first_step] = "first_step"
    reason_schedule_a[second_step] = "second_step"

    config = _config(rebalance_frequency="weekly")
    strategy = _ScriptedStrategyWithReasons({"A": schedule_a}, {"A": reason_schedule_a})
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    assert len(trades) == 1
    assert trades.loc[0, "trigger_reason_detail_code"] == "second_step"
    reason_details = str(trades.loc[0, "trigger_reason_details"])
    assert "scripted: second_step" in reason_details
    assert "first_step" not in reason_details


def test_off_cycle_strategy_reason_survives_an_execution_delay() -> None:
    """The alignment fix must still work once an extra execution_delay
    shift is layered on top of the rebalance-date sampling."""
    n = 30
    positions = _weekly_rebalance_positions(n)
    previous_rebalance, next_rebalance = positions[1], positions[2]
    off_cycle = previous_rebalance + 1
    assert off_cycle < next_rebalance, "need at least one off-cycle row"

    schedule_a = [0.0] * off_cycle + [1.0] * (n - off_cycle)
    reason_schedule_a: list[str | None] = [None] * n
    reason_schedule_a[off_cycle] = "off_cycle_entry"

    config = _config(rebalance_frequency="weekly")
    strategy = _ScriptedStrategyWithReasons({"A": schedule_a}, {"A": reason_schedule_a})
    data = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
        execution_delay=2,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    assert len(trades) == 1
    assert trades.loc[0, "trigger_reason_detail_code"] == "off_cycle_entry"
    assert "scripted: off_cycle_entry" in str(trades.loc[0, "trigger_reason_details"])


def test_off_cycle_strategy_reason_survives_a_non_tradable_gap() -> None:
    """The alignment fix must still work when the symbol has its own
    calendar closures (a mixed-calendar universe, so engine.py actually
    computes a per-symbol ``tradable`` mask instead of taking the
    ``tradable=None`` fast path)."""
    n = 30
    positions = _weekly_rebalance_positions(n)
    previous_rebalance, next_rebalance = positions[1], positions[2]
    off_cycle = previous_rebalance + 1
    assert off_cycle < next_rebalance, "need at least one off-cycle row"

    schedule_a = [0.0] * off_cycle + [1.0] * (n - off_cycle)
    schedule_b = [0.0] * n
    reason_schedule_a: list[str | None] = [None] * n
    reason_schedule_a[off_cycle] = "off_cycle_entry"
    reason_schedule_b: list[str | None] = [None] * n

    # Two instruments on DIFFERENT calendars -- forces engine.py to build
    # a real per-symbol tradable mask (uniform_calendar returns None
    # otherwise, and the whole tradable-mask machinery is skipped).
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "trade_reasons_tradability",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "B", "source": "csv", "calendar": "24/7"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-03-01",
            },
            "strategy": {"name": "buy_and_hold"},  # unused: an instance is passed
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "weekly"},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000, "periods_per_year": 252},
        }
    )
    strategy = _ScriptedStrategyWithReasons(
        {"A": schedule_a, "B": schedule_b},
        {"A": reason_schedule_a, "B": reason_schedule_b},
    )
    data_a = make_ohlcv("A", [100.0] * n, start="2020-01-01")
    data_b = make_ohlcv("B", [100.0] * n, start="2020-01-01")
    data = pd.concat([data_a, data_b], ignore_index=True)
    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )
    trades = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)

    assert len(trades) == 1
    assert trades.loc[0, "trigger_reason_detail_code"] == "off_cycle_entry"
    assert "scripted: off_cycle_entry" in str(trades.loc[0, "trigger_reason_details"])


# --------------------------------------------------------------------------- #
# stop_loss_pct/take_profit_pct: end-to-end through the real BacktestEngine,
# operating on the REAL executed position (never the raw signal) -- see
# quantlab.backtesting.accounting._detect_stop_loss_take_profit.
# --------------------------------------------------------------------------- #
class _ScriptedPairStrategy(_ScriptedStrategy):
    """Like `_ScriptedStrategy`, but declares its two symbols as one
    `position_groups()` -- the pairs_trading pattern, for an end-to-end
    test that a stop-loss triggers on the GROUP's combined P&L."""

    name = "scripted_pair"

    def __init__(
        self, schedule: dict[str, list[float]], symbol_a: str, symbol_b: str
    ) -> None:
        super().__init__(schedule)
        self._symbol_a = symbol_a
        self._symbol_b = symbol_b

    def position_groups(self) -> tuple[tuple[str, ...], ...] | None:
        return ((self._symbol_a, self._symbol_b),)


def test_stop_loss_pct_closes_a_real_position_and_the_trade_log_shows_it() -> None:
    """Long throughout, -6% then another -6% (cumulative -11.64%, past a
    10% stop): the loss-realizing bar keeps its return, and the position
    is closed the FOLLOWING bar with a real, non-zero exit trade whose
    adjustment_reason_codes is exactly 'stop_loss' -- end-to-end proof
    that the mechanism operates on the real executed position and is
    correctly surfaced in the trade log."""
    n = 8
    schedule = {"A": [1.0] * n}
    prices = [100.0, 100.0, 94.0, 88.36, 88.36, 88.36, 88.36, 88.36]
    config = _config(rebalance_frequency="daily")
    strategy = _ScriptedStrategy(schedule)
    strategy.stop_loss_pct = 0.10

    trades = _run_strategy(strategy, prices, config)

    exits = trades[trades["action"] == "exit_long"]
    assert len(exits) == 1
    assert exits.iloc[0]["adjustment_reason_codes"] == "stop_loss"
    assert "stop_loss_pct" in str(exits.iloc[0]["adjustment_reason_details"])


def test_stop_loss_pct_none_by_default_changes_nothing_end_to_end() -> None:
    """The critical non-regression guarantee at the engine level: a
    strategy that never sets stop_loss_pct/take_profit_pct (the default
    on every built-in strategy) must produce a trade log identical to
    today's, with no 'stop_loss'/'take_profit' adjustment ever appearing."""
    n = 8
    schedule = {"A": [1.0] * n}
    prices = [100.0, 100.0, 94.0, 88.36, 88.36, 88.36, 88.36, 88.36]
    config = _config(rebalance_frequency="daily")
    strategy = _ScriptedStrategy(schedule)

    trades = _run_strategy(strategy, prices, config)

    assert not (trades["adjustment_reason_codes"] == "stop_loss").any()
    assert not (trades["adjustment_reason_codes"] == "take_profit").any()
    assert len(trades[trades["action"] == "exit_long"]) == 0


def test_position_groups_stop_loss_closes_both_legs_together() -> None:
    """A scripted pair strategy (mirroring pairs_trading's own
    `position_groups()`) with legs A=+1.0/B=-1.0 scaled down to +0.5/-0.5
    mid-hold (a rebalance) -- A drops 20%/20%, B is flat, so the GROUP's
    combined return per unit of ITS OWN exposure breaches a 15% stop
    (cumulative 0.90*0.90-1=-19%) regardless of the leg-size change.
    BOTH legs must show the exit AND the 'stop_loss' code on the SAME
    date -- proof position_groups() is honored end-to-end, not just at
    the accounting-layer unit test level."""
    schedule_a = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    schedule_b = [-1.0, -1.0, -0.5, -0.5, -0.5, -0.5]
    prices_a = [100.0, 100.0, 80.0, 64.0, 64.0, 64.0]
    prices_b = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "trade_reasons_stop_loss_pair",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "B", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-03-01",
            },
            "strategy": {"name": "buy_and_hold"},  # unused: an instance is passed
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {
                "commission_bps": 0.0,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000},
        }
    )
    strategy = _ScriptedPairStrategy({"A": schedule_a, "B": schedule_b}, "A", "B")
    strategy.stop_loss_pct = 0.15
    data_a = make_ohlcv("A", prices_a, start="2020-01-01")
    data_b = make_ohlcv("B", prices_b, start="2020-01-01")
    data = pd.concat([data_a, data_b], ignore_index=True)

    result = BacktestEngine().run(
        data,
        strategy,
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
    )

    trades_a = result.trades[result.trades["symbol"] == "A"].reset_index(drop=True)
    trades_b = result.trades[result.trades["symbol"] == "B"].reset_index(drop=True)
    stop_a = trades_a[trades_a["adjustment_reason_codes"] == "stop_loss"]
    stop_b = trades_b[trades_b["adjustment_reason_codes"] == "stop_loss"]
    assert len(stop_a) == 1
    assert len(stop_b) == 1
    assert stop_a.iloc[0]["timestamp"] == stop_b.iloc[0]["timestamp"]
    assert stop_a.iloc[0]["action"] == "exit_long"
    assert stop_b.iloc[0]["action"] == "exit_short"
