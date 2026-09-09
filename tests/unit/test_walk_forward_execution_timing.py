"""Parity: walk-forward accounting must honor execution_timing like the engine.

BacktestEngine.run() decomposes returns into overnight/intraday legs
whenever execution_timing="next_bar_open" (proven in
tests/unit/test_execution_timing_engine_wiring.py and hand-verified at the
accounting level in tests/unit/test_overnight_intraday_returns.py).
walk_forward.py has three of its own run_accounting call sites (the main
run()/_finalize path, rescore_with_costs, and the validation-block scorer
_weights_and_returns_for_validation), each of which must thread
overnight_returns_for_config() through (shared with BacktestEngine.run()
itself, see quantlab.backtesting.execution_prices) to keep the documented
parity contract with a fresh backtest. These tests prove each call site
does, using a panel with a genuine, persistent open != previous-close gap
-- with no gap (as in tests/conftest.py's make_ohlcv), the decomposition
is indistinguishable from close-to-close and a dropped wire-up would go
undetected.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.execution_prices import (
    RatioApproximatedExecutionPrices,
    adjusted_open_for_config,
    overnight_returns_for_config,
    reference_prices_for_config,
)
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
from quantlab.exceptions import BacktestError
from quantlab.validation.walk_forward import (
    WalkForwardValidator,
    _overnight_returns_for_fold,
    _weights_and_returns_for_validation,
)


def _panel_with_persistent_overnight_gap(
    n: int = 90, seed: int = 7, gap: float = 0.02
) -> pd.DataFrame:
    """A single symbol whose open is always 2% above the previous close.

    A persistent, nonzero gap (rather than a single one-off jump) means
    every day -- not just an entry day -- has a nonzero overnight leg, so
    next_bar_open's stitched returns differ from same_bar_close's on
    (almost) every date, even for a strategy that never changes its
    weight (e.g. buy_and_hold): the decomposed return is
    ``pre*overnight + post*intraday``, which differs from the plain
    close-to-close return by ``w*overnight*intraday`` whenever
    ``overnight != 0`` (see accounting.py's own docstring for this exact
    cross term).
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=n)))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1.0 + gap)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {
            TIMESTAMP: idx,
            SYMBOL: "AAA",
            OPEN: open_,
            HIGH: high,
            LOW: low,
            CLOSE: close,
            ADJUSTED_CLOSE: close,
            VOLUME: [1_000_000.0] * n,
        }
    )


def _config(execution_timing: str) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "wf_execution_timing",
            "data": {
                "instruments": [{"symbol": "AAA", "source": "csv", "calendar": "XNYS"}],
                "start_date": "2018-01-01",
                "end_date": "2018-12-31",
            },
            "strategy": {"name": "buy_and_hold"},
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {"execution_timing": execution_timing},
            "backtest": {"initial_capital": 100_000},
        }
    )


class _Windows(TypedDict):
    train_window: int
    validation_window: int
    test_window: int
    expanding: bool


_WINDOWS: _Windows = {
    "train_window": 20,
    "validation_window": 10,
    "test_window": 10,
    "expanding": True,
}


def test_overnight_returns_for_config_matches_the_engines_own_price_source() -> None:
    """The helper's own math, isolated from any walk-forward plumbing."""
    data = _panel_with_persistent_overnight_gap()
    prices = data.set_index(TIMESTAMP)[[CLOSE]].rename(columns={CLOSE: "AAA"})
    assert overnight_returns_for_config(_config("same_bar_close"), data, prices) is None

    result = overnight_returns_for_config(_config("next_bar_open"), data, prices)
    assert result is not None
    expected = (
        RatioApproximatedExecutionPrices()
        .adjusted_open(data)
        .reindex(index=prices.index, columns=prices.columns)
        / prices.shift(1)
        - 1.0
    )
    pd.testing.assert_frame_equal(result, expected)


def test_reference_prices_for_config_uses_the_current_open_under_next_bar_open() -> (
    None
):
    """A trade-log fill's reference price must be the SAME bar's own
    (adjusted) open under next_bar_open -- the fill genuinely happens
    there, never the previous bar's close a same_bar_close/
    immediate_after_close fill uses instead."""
    data = _panel_with_persistent_overnight_gap()
    prices = data.set_index(TIMESTAMP)[[CLOSE]].rename(columns={CLOSE: "AAA"})

    same_close_prices, same_close_needs_shift = reference_prices_for_config(
        _config("same_bar_close"), data, prices
    )
    assert same_close_needs_shift is True
    pd.testing.assert_frame_equal(same_close_prices, prices)

    next_open_prices, next_open_needs_shift = reference_prices_for_config(
        _config("next_bar_open"), data, prices
    )
    assert next_open_needs_shift is False
    expected_open = (
        RatioApproximatedExecutionPrices()
        .adjusted_open(data)
        .reindex(index=prices.index, columns=prices.columns)
    )
    pd.testing.assert_frame_equal(next_open_prices, expected_open)
    # The two must genuinely differ here (a persistent, nonzero gap), or
    # this test would not actually distinguish the two conventions.
    assert not next_open_prices.equals(same_close_prices)


@pytest.mark.parametrize(
    "call",
    [
        "adjusted_open_for_config",
        "overnight_returns_for_config",
        "reference_prices_for_config",
    ],
)
def test_a_malformed_custom_price_source_raises_a_clear_backtest_error(
    call: str,
) -> None:
    """Regression test: a custom ``ExecutionPriceSource`` returning the
    wrong type from ``adjusted_open()`` (e.g. a bare ``ndarray``) must
    raise a clear ``BacktestError`` naming the misbehaving source at
    this module's own public boundary -- not silently propagate an
    opaque, far-downstream ``AttributeError`` (``'numpy.ndarray' object
    has no attribute 'reindex'``) from deep inside a caller that assumes
    a DataFrame."""

    class _NdarrayExecutionPrices:
        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            return np.zeros((len(data), 1))  # type: ignore[return-value]

    data = _panel_with_persistent_overnight_gap()
    prices = data.set_index(TIMESTAMP)[[CLOSE]].rename(columns={CLOSE: "AAA"})
    config = _config("next_bar_open")
    source = _NdarrayExecutionPrices()
    calls = {
        "adjusted_open_for_config": lambda: adjusted_open_for_config(
            config, data, execution_price_source=source
        ),
        "overnight_returns_for_config": lambda: overnight_returns_for_config(
            config, data, prices, execution_price_source=source
        ),
        "reference_prices_for_config": lambda: reference_prices_for_config(
            config, data, prices, execution_price_source=source
        ),
    }

    with pytest.raises(BacktestError, match="must return a pandas DataFrame"):
        calls[call]()


def test_a_custom_price_source_returning_non_positive_values_raises() -> None:
    """A source returning a real DataFrame, but with a zero/negative
    adjusted-open value it did NOT mask to NaN, must also be rejected --
    the shipped ``RatioApproximatedExecutionPrices`` masks exactly this
    case itself, so a custom source silently skipping that guard should
    not be allowed to feed a nonsensical price into the overnight leg."""

    class _NonPositiveExecutionPrices:
        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            close_matrix = data.pivot(index=TIMESTAMP, columns=SYMBOL, values=CLOSE)
            return close_matrix * 0.0  # every cell exactly zero, not NaN

    data = _panel_with_persistent_overnight_gap()
    config = _config("next_bar_open")

    with pytest.raises(BacktestError, match="invalid adjusted-open matrix"):
        adjusted_open_for_config(
            config, data, execution_price_source=_NonPositiveExecutionPrices()
        )


@pytest.mark.parametrize(
    "call",
    ["overnight_returns_for_config", "reference_prices_for_config"],
)
def test_a_directly_passed_malformed_adjusted_open_still_raises(call: str) -> None:
    """Regression test: a caller-supplied ``adjusted_open=`` must be
    validated exactly like a freshly-derived one -- both
    ``overnight_returns_for_config`` and ``reference_prices_for_config``
    used to skip validation entirely whenever ``adjusted_open`` was
    already given (the ``if adjusted_open is None:`` branch that called
    ``_validate_adjusted_open`` never ran), letting a bare ``ndarray`` (or
    any other malformed value) reach ``.reindex(...)`` unchecked and raise
    an opaque ``AttributeError`` instead of this module's usual clear
    ``BacktestError`` naming the actual problem."""
    data = _panel_with_persistent_overnight_gap()
    prices = data.set_index(TIMESTAMP)[[CLOSE]].rename(columns={CLOSE: "AAA"})
    config = _config("next_bar_open")
    bad_adjusted_open: object = np.ones((len(data), 1))
    calls = {
        "overnight_returns_for_config": lambda: overnight_returns_for_config(
            config,
            data,
            prices,
            adjusted_open=bad_adjusted_open,  # type: ignore[arg-type]
        ),
        "reference_prices_for_config": lambda: reference_prices_for_config(
            config,
            data,
            prices,
            adjusted_open=bad_adjusted_open,  # type: ignore[arg-type]
        ),
    }
    with pytest.raises(BacktestError, match="must return a pandas DataFrame"):
        calls[call]()


def test_walk_forward_run_reflects_execution_timing_given_an_overnight_gap() -> None:
    """Covers the main run()/_finalize call site."""
    data = _panel_with_persistent_overnight_gap()
    same_close = WalkForwardValidator(_config("same_bar_close")).run(
        data, parameter_grid={}, **_WINDOWS
    )
    next_open = WalkForwardValidator(_config("next_bar_open")).run(
        data, parameter_grid={}, **_WINDOWS
    )
    assert len(same_close.oos_returns) > 0
    assert len(next_open.oos_returns) > 0
    assert not same_close.oos_returns.equals(next_open.oos_returns)


def test_walk_forward_finalize_derives_adjusted_open_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `_finalize`'s own `overnight_returns_for_config`
    call and `_build_oos_result`'s `reference_prices_for_config` call
    both need the adjusted-open matrix -- they must share ONE derivation
    (mirroring `BacktestEngine.run()`'s own pattern), not pay for
    `RatioApproximatedExecutionPrices.adjusted_open`'s full-panel pivots
    twice on every `run()`/`run_with_weight_cache()`/
    `rescore_with_costs()` call. Isolates the count to `_finalize`'s own
    call specifically (before/after delta), since `run()` as a whole also
    derives it elsewhere (once per fold, lazily, in the selection loop;
    once more per fold's own `BacktestEngine.run()` for its target
    weights) -- unrelated call sites this fix does not touch."""
    calls = {"n": 0}
    original_adjusted_open = RatioApproximatedExecutionPrices.adjusted_open

    def counting_adjusted_open(self: object, data: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        return original_adjusted_open(self, data)  # type: ignore[arg-type]

    monkeypatch.setattr(
        RatioApproximatedExecutionPrices, "adjusted_open", counting_adjusted_open
    )
    original_finalize = WalkForwardValidator._finalize
    finalize_deltas: list[int] = []

    def spying_finalize(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        before = calls["n"]
        result = original_finalize(self, *args, **kwargs)  # type: ignore[arg-type]
        finalize_deltas.append(calls["n"] - before)
        return result

    monkeypatch.setattr(WalkForwardValidator, "_finalize", spying_finalize)

    data = _panel_with_persistent_overnight_gap()
    WalkForwardValidator(_config("next_bar_open")).run(
        data, parameter_grid={}, **_WINDOWS
    )
    assert finalize_deltas == [1]


def test_rescore_with_costs_reflects_execution_timing_given_an_overnight_gap() -> None:
    """Covers the rescore_with_costs() call site."""
    data = _panel_with_persistent_overnight_gap()
    base_config = _config("same_bar_close")
    _baseline, cache = WalkForwardValidator(base_config).run_with_weight_cache(
        data, parameter_grid={}, **_WINDOWS
    )
    same_close = WalkForwardValidator(base_config).rescore_with_costs(
        cache, base_config
    )
    next_open_config = _config("next_bar_open")
    next_open = WalkForwardValidator(next_open_config).rescore_with_costs(
        cache, next_open_config
    )
    assert not same_close.oos_returns.equals(next_open.oos_returns)


def test_weights_and_returns_for_validation_reflects_execution_timing() -> None:
    """Covers the validation-block scorer call site directly."""
    data = _panel_with_persistent_overnight_gap()
    lookback_start = data[TIMESTAMP].iloc[0]
    window_start = data[TIMESTAMP].iloc[20]
    window_end = data[TIMESTAMP].iloc[29]

    same_close_config = _config("same_bar_close")
    next_open_config = _config("next_bar_open")
    _weights, same_close_returns = _weights_and_returns_for_validation(
        data,
        same_close_config,
        lookback_start,
        window_start,
        window_end,
        overnight_returns=_overnight_returns_for_fold(
            data,
            same_close_config,
            lookback_start,
            window_end,
            execution_price_source=None,
        ),
    )
    _weights, next_open_returns = _weights_and_returns_for_validation(
        data,
        next_open_config,
        lookback_start,
        window_start,
        window_end,
        overnight_returns=_overnight_returns_for_fold(
            data,
            next_open_config,
            lookback_start,
            window_end,
            execution_price_source=None,
        ),
    )
    assert not same_close_returns.equals(next_open_returns)


def test_walk_forward_validator_honors_a_custom_execution_price_source() -> None:
    """WalkForwardValidator's own execution_price_source parameter must
    actually reach every internal accounting call site, not just be
    accepted and silently dropped -- proven the same way as
    test_a_custom_execution_price_source_is_actually_used in
    tests/unit/test_execution_timing_engine_wiring.py, but for the
    walk-forward path (run(), which exercises both fold selection and
    OOS stitching)."""

    class _FixedAdjustedOpen:
        """Deliberately different from the real ratio approximation."""

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            close_matrix = data.pivot(index=TIMESTAMP, columns=SYMBOL, values=CLOSE)
            return close_matrix * 0.0 + 90.0  # a fixed, recognizable value

    data = _panel_with_persistent_overnight_gap()
    config = _config("next_bar_open")

    default_result = WalkForwardValidator(config).run(
        data, parameter_grid={}, **_WINDOWS
    )
    custom_result = WalkForwardValidator(
        config, execution_price_source=_FixedAdjustedOpen()
    ).run(data, parameter_grid={}, **_WINDOWS)

    assert not default_result.oos_returns.equals(custom_result.oos_returns)


def test_walk_forward_custom_source_does_not_taint_reproducibility() -> None:
    """Regression test: a custom execution_price_source is never even
    consulted outside next_bar_open (overnight_returns_for_config/
    reference_prices_for_config both early-return before touching it),
    so pairing one with same_bar_close has provably zero effect on the
    run's numbers -- WalkForwardResult.oos_result.metadata['config_yaml_
    reflects_execution'] must stay True, matching BacktestEngine.run()'s
    own identical exemption (test_custom_source_does_not_taint_
    reproducibility_outside_next_bar_open in
    test_execution_timing_engine_wiring.py) -- not be forced False just
    because a (dormant) source was supplied."""

    class _FixedAdjustedOpen:
        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            close_matrix = data.pivot(index=TIMESTAMP, columns=SYMBOL, values=CLOSE)
            return close_matrix * 0.0 + 90.0

    data = _panel_with_persistent_overnight_gap()

    same_close_result = WalkForwardValidator(
        _config("same_bar_close"), execution_price_source=_FixedAdjustedOpen()
    ).run(data, parameter_grid={}, **_WINDOWS)
    assert same_close_result.oos_result is not None
    same_close_meta = same_close_result.oos_result.metadata
    assert same_close_meta["config_yaml_reflects_execution"] is True

    next_open_result = WalkForwardValidator(
        _config("next_bar_open"), execution_price_source=_FixedAdjustedOpen()
    ).run(data, parameter_grid={}, **_WINDOWS)
    assert next_open_result.oos_result is not None
    assert (
        next_open_result.oos_result.metadata["config_yaml_reflects_execution"] is False
    )


def test_walk_forward_validator_custom_source_reaches_weight_generation_too() -> None:
    """Regression test: a custom execution_price_source must also be
    honored on the WEIGHT-generation path (_weights_for_window/
    _target_weights_for_window, reached via run_backtest_from_config),
    not only the accounting call sites walk_forward.py owns directly --
    the previous test above cannot catch a dropped wire-up there, since
    it only compares two DIFFERENT sources that both stay finite.

    Proven with a zero raw open, which the DEFAULT source
    (RatioApproximatedExecutionPrices) masks to NaN, and which
    run_accounting's missing_overnight check then rejects since the
    position is genuinely held through that bar -- run_backtest_from_
    config (called by both weight-generation helpers) crashes on this
    data using the default source, but must NOT crash once a custom
    source that avoids the NaN is actually threaded all the way
    through."""
    data = _panel_with_persistent_overnight_gap()
    data.loc[data.index[45], OPEN] = 0.0
    config = _config("next_bar_open")

    with pytest.raises(BacktestError, match="missing a return"):
        WalkForwardValidator(config).run(data, parameter_grid={}, **_WINDOWS)

    class _FallBackToClose:
        """Sidesteps the zero-open NaN by never looking at open at all."""

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.pivot(index=TIMESTAMP, columns=SYMBOL, values=CLOSE)

    result = WalkForwardValidator(
        config, execution_price_source=_FallBackToClose()
    ).run(data, parameter_grid={}, **_WINDOWS)
    assert len(result.oos_returns) > 0
