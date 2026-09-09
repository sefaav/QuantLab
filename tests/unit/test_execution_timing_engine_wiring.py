"""End-to-end wiring test for ExecutionConfig.execution_timing.

The math itself is exhaustively hand-verified at the accounting level in
tests/unit/test_overnight_intraday_returns.py. This test only proves the
plumbing: BacktestEngine.run() actually derives overnight_returns via
RatioApproximatedExecutionPrices and threads it into run_accounting when
execution_timing="next_bar_open", and leaves same_bar_close's own numbers
byte-identical (the default, and the only value used before this field
existed).
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.backtesting.engine import BacktestEngine
from quantlab.backtesting.result import BacktestResult
from quantlab.config import ExperimentConfig
from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.data.base import pivot_field
from quantlab.execution.execution_model import ExecutionModel
from quantlab.portfolio.allocator import EqualWeightAllocator
from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy


def _panel_with_overnight_gap() -> pd.DataFrame:
    """A single symbol with a deliberate open != previous-close gap.

    Unlike tests/conftest.py's make_ohlcv (which sets open[T] = close[T-1],
    making the overnight leg trivially zero), this panel has a genuine gap
    on the entry date so next_bar_open and same_bar_close produce visibly
    different numbers -- a wiring bug that silently dropped overnight_returns
    would not be caught by a fixture where the overnight return is always 0.
    """
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    close = [100.0, 100.0, 110.0, 111.0, 112.0]
    # buy_and_hold + equal_weight enters at index 1 (executed[1] = held[0]).
    # A real overnight gap up right there (close[0]=100 -> open[1]=105)
    # means next_bar_open's entry-day contribution is the intraday leg
    # only, materially different from same_bar_close's full-day return.
    open_ = [100.0, 105.0, 108.0, 110.0, 111.0]
    return pd.DataFrame(
        {
            TIMESTAMP: idx,
            SYMBOL: "AAA",
            OPEN: open_,
            HIGH: [max(o, c) * 1.01 for o, c in zip(open_, close, strict=True)],
            LOW: [min(o, c) * 0.99 for o, c in zip(open_, close, strict=True)],
            CLOSE: close,
            ADJUSTED_CLOSE: close,
            VOLUME: [1_000_000.0] * 5,
        }
    )


def _config(execution_timing: str, *, calendar: str = "XNYS") -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "execution_timing_wiring",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": calendar}
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {"execution_timing": execution_timing},
            "backtest": {"initial_capital": 100_000},
        }
    )


def _run(
    execution_timing: str, *, calendar: str = "XNYS", execution_delay: int = 0
) -> BacktestResult:
    config = _config(execution_timing, calendar=calendar)
    return BacktestEngine().run(
        _panel_with_overnight_gap(),
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
        execution_delay=execution_delay,
    )


def test_next_bar_open_differs_from_same_bar_close_given_a_real_gap() -> None:
    same_close = _run("same_bar_close")
    next_open = _run("next_bar_open")
    # Both must complete and produce a full equity curve.
    assert len(same_close.equity_curve) == len(next_open.equity_curve)
    # A genuine overnight gap on the entry date means the two conventions
    # must disagree on at least one date's return -- if they matched
    # everywhere, overnight_returns was silently dropped somewhere in the
    # wiring.
    assert same_close.gross_returns is not None
    assert next_open.gross_returns is not None
    assert not same_close.gross_returns.equals(next_open.gross_returns)
    # Exact value, not just inequality: entry at index 1 is a fresh
    # position (pre=0), so next_bar_open's contribution there is purely
    # the intraday leg -- (1 + asset_return) / (1 + overnight) - 1, with
    # asset_return = close[1]/close[0] - 1 = 100/100 - 1 = 0.0 and
    # overnight = open[1]/close[0] - 1 = 105/100 - 1 = 0.05.
    expected_entry_day_return = (1.0 + 0.0) / (1.0 + 0.05) - 1.0
    assert next_open.gross_returns.iloc[1] == pytest.approx(expected_entry_day_return)
    # same_bar_close's own entry-day contribution, for contrast: the full
    # close-to-close return (0.0 here), not the decomposed intraday-only one.
    assert same_close.gross_returns.iloc[1] == pytest.approx(0.0)


def test_adjusted_open_is_derived_only_once_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: BacktestEngine.run() needs the adjusted-open
    matrix (RatioApproximatedExecutionPrices.adjusted_open -- 3
    full-panel pivots plus validation) for both overnight_returns_for_
    config and reference_prices_for_config under next_bar_open. It must
    derive it exactly ONCE and thread the result into both, not pay for
    the derivation twice."""
    from quantlab.backtesting.execution_prices import RatioApproximatedExecutionPrices

    calls = {"n": 0}
    original = RatioApproximatedExecutionPrices.adjusted_open

    def counting_adjusted_open(self: object, data: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        return original(self, data)  # type: ignore[arg-type]

    monkeypatch.setattr(
        RatioApproximatedExecutionPrices, "adjusted_open", counting_adjusted_open
    )
    _run("next_bar_open")
    assert calls["n"] == 1


def test_a_custom_execution_price_source_is_actually_used() -> None:
    """A non-default `execution_price_source` passed to `BacktestEngine.run`
    must actually change the result -- a dropped/ignored override would
    otherwise silently fall back to `RatioApproximatedExecutionPrices`."""

    class _FixedAdjustedOpen:
        """Deliberately different from the real ratio approximation."""

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            close_matrix = pivot_field(data, CLOSE)
            return close_matrix * 0.0 + 90.0  # a fixed, recognizable value

    default_source = _run("next_bar_open")
    custom_source = BacktestEngine().run(
        _panel_with_overnight_gap(),
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(_config("next_bar_open").execution),
        _config("next_bar_open"),
        execution_price_source=_FixedAdjustedOpen(),
    )
    assert default_source.gross_returns is not None
    assert custom_source.gross_returns is not None
    assert not default_source.gross_returns.equals(custom_source.gross_returns)
    # Exact value at the entry date (index 1): pre=0 (fresh entry), so the
    # contribution is purely the intraday leg from the FIXED 90.0 open
    # forward to close[1]=100 -- (1 + asset_return)/(1 + overnight) - 1
    # with asset_return = 100/100 - 1 = 0.0 and overnight = 90/100 - 1 = -0.10.
    expected = (1.0 + 0.0) / (1.0 - 0.10) - 1.0
    assert custom_source.gross_returns.iloc[1] == pytest.approx(expected)


def test_custom_execution_price_source_marks_the_run_not_reproducible_from_yaml() -> (
    None
):
    """A custom `execution_price_source` can change the result without
    appearing anywhere in `config.yaml`, so a run given one must not be
    reported as reproducible from config.yaml alone."""

    class _FixedAdjustedOpen:
        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            close_matrix = pivot_field(data, CLOSE)
            return close_matrix * 0.0 + 90.0

    config = _config("next_bar_open")
    default_result = _run("next_bar_open")
    custom_result = BacktestEngine().run(
        _panel_with_overnight_gap(),
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
        execution_price_source=_FixedAdjustedOpen(),
    )
    assert default_result.metadata["config_yaml_reflects_execution"] is True
    assert default_result.metadata["execution_price_source_fingerprint"] == (
        "default (RatioApproximatedExecutionPrices)"
    )
    assert custom_result.metadata["config_yaml_reflects_execution"] is False
    assert "_FixedAdjustedOpen" in str(
        custom_result.metadata["execution_price_source_fingerprint"]
    )


def test_custom_source_does_not_taint_reproducibility_outside_next_bar_open() -> None:
    """A custom `execution_price_source` is never even consulted outside
    `next_bar_open` (`overnight_returns_for_config`/
    `reference_prices_for_config` both early-return before touching it),
    so pairing one with `same_bar_close` has provably zero effect on the
    run's numbers -- `config_yaml_reflects_execution` must stay True
    (given every other component matches), not be forced False just
    because a (dormant) source was supplied."""

    class _FixedAdjustedOpen:
        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            close_matrix = pivot_field(data, CLOSE)
            return close_matrix * 0.0 + 90.0

    config = _config("same_bar_close")
    result = BacktestEngine().run(
        _panel_with_overnight_gap(),
        BuyAndHoldStrategy(),
        EqualWeightAllocator(),
        ExecutionModel.from_config(config.execution),
        config,
        execution_price_source=_FixedAdjustedOpen(),
    )
    assert result.metadata["config_yaml_reflects_execution"] is True
    # The source's identity is still recorded in metadata even though it
    # was dormant -- only the reproducibility verdict changes.
    assert "_FixedAdjustedOpen" in str(
        result.metadata["execution_price_source_fingerprint"]
    )


def test_immediate_after_close_matches_same_bar_close_numerically() -> None:
    """Different deployability, identical math (continuous-market rationale).

    ``immediate_after_close`` requires a 24/7 market (see
    ``ExperimentConfig._check_immediate_after_close_needs_a_247_market``),
    unlike every other test in this file -- explicit here for that reason.
    """
    same_close = _run("same_bar_close", calendar="24/7")
    immediate = _run("immediate_after_close", calendar="24/7")
    assert same_close.gross_returns is not None
    assert immediate.gross_returns is not None
    pd.testing.assert_series_equal(
        same_close.gross_returns, immediate.gross_returns, check_names=False
    )


def test_same_bar_close_is_the_config_default() -> None:
    cfg = _config("same_bar_close")
    assert cfg.execution.execution_timing == "same_bar_close"
    default_cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "default_timing",
            "data": {
                "instruments": [{"symbol": "AAA", "source": "csv", "calendar": "XNYS"}],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert default_cfg.execution.execution_timing == "same_bar_close"


def test_next_bar_open_combines_with_a_nonzero_execution_delay() -> None:
    """`execution_delay` re-simulates the whole trade schedule shifted by
    extra bars (see
    `test_execution_delay_is_a_true_resimulation_not_a_returns_shift` in
    tests/unit/test_regression_execution.py), independent of which
    execution_timing mode is active -- both must combine without error
    and the delay must still materially change the result, exactly as it
    does under the default same_bar_close."""
    on_time = _run("next_bar_open", execution_delay=0)
    delayed = _run("next_bar_open", execution_delay=1)
    assert on_time.gross_returns is not None
    assert delayed.gross_returns is not None
    assert not on_time.gross_returns.equals(delayed.gross_returns)


@pytest.mark.parametrize(
    "execution_timing", ["same_bar_close", "next_bar_open", "immediate_after_close"]
)
def test_every_execution_timing_mode_runs_without_error(execution_timing: str) -> None:
    # 24/7 so the same parametrization covers immediate_after_close too
    # (session-based calendars are rejected for it, see
    # ExperimentConfig._check_immediate_after_close_needs_a_247_market) --
    # calendar choice is otherwise irrelevant to this smoke test.
    result = _run(execution_timing, calendar="24/7")
    assert result.equity_curve.notna().all()
