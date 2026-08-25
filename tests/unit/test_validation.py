"""Tests for walk-forward, sensitivity, bootstrap and stress tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv

from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError
from quantlab.risk import metrics as M
from quantlab.risk.stress import scale_costs
from quantlab.validation.bootstrap import bootstrap_returns
from quantlab.validation.parameter_sensitivity import (
    run_parameter_sensitivity,
    run_walk_forward_parameter_sensitivity,
    sensitivity_heatmap_data,
)
from quantlab.validation.robustness import (
    monte_carlo_permutation,
    run_stress_tests,
    run_walk_forward_stress_tests,
)
from quantlab.validation.walk_forward import (
    WalkForwardValidator,
    resolve_walk_forward_windows,
)


class _WalkForwardWindows(TypedDict):
    """Precise keyword types for ``**windows`` call-site unpacking below.

    A plain ``dict[str, object]`` erases each key's own type, so a type
    checker cannot verify ``**windows`` against ``run()``'s per-parameter
    types (``train_window: int``, ``expanding: bool``, ...) — a TypedDict
    keeps each field's real type across the unpack.
    """

    train_window: int
    validation_window: int
    test_window: int
    expanding: bool


class _WalkForwardRunKwargs(TypedDict):
    """Like :class:`_WalkForwardWindows`, plus ``parameter_grid``."""

    parameter_grid: dict[str, list[int]]
    train_window: int
    validation_window: int
    test_window: int
    expanding: bool


class _SensitivitySweepKwargs(TypedDict):
    """Precise keyword types for ``**sweep_kwargs`` call-site unpacking."""

    parameter_x: str
    values_x: list[Any]
    parameter_y: str
    values_y: list[Any]


def _panel(n: int = 900) -> pd.DataFrame:
    frames = []
    for sym, seed, mu in [("AAA", 1, 0.0008), ("BBB", 2, 0.0002), ("CCC", 3, 0.0005)]:
        prices = geometric_series(n, mu=mu, sigma=0.012, s0=100.0, seed=seed)
        frames.append(make_ohlcv(sym, prices, start="2018-01-01"))
    return pd.concat(frames, ignore_index=True)


def _mean_reverting_prices(
    n: int, seed: int, phi: float = 0.85, scale: float = 0.03
) -> np.ndarray:
    """AR(1) log-price deviations around a stationary mean.

    ``phi < 1`` mean-reverts (same construction as
    ``test_features.py::test_half_life_detects_mean_reversion``), giving
    ``mean_reversion`` a genuine, tradeable edge — unlike the plain GBM from
    ``geometric_series`` used elsewhere in this file.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, scale)
    return 100.0 * np.exp(x)


def _cost_sensitive_panel(n: int = 500) -> pd.DataFrame:
    frames = [
        make_ohlcv("AAA", _mean_reverting_prices(n, seed=11), start="2018-01-01"),
        make_ohlcv("BBB", _mean_reverting_prices(n, seed=12), start="2018-01-01"),
    ]
    return pd.concat(frames, ignore_index=True)


def _cost_sensitive_config(commission_bps: float = 50.0) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "cost_sensitivity",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
            },
            "strategy": {
                "name": "mean_reversion",
                "parameters": {
                    "lookback_period": 10,
                    "entry_zscore": 1.0,
                    "exit_zscore": 0.1,
                },
            },
            "portfolio": {"allocator": "equal_weight", "rebalance_frequency": "daily"},
            "execution": {
                "commission_bps": commission_bps,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            "backtest": {"initial_capital": 100_000},
            "validation": {
                "method": "walk_forward",
                "optimization_metric": "sharpe",
                "train_window": 150,
                "validation_window": 60,
                "test_window": 60,
                # Pinned explicitly: run_walk_forward_stress_tests() resolves
                # its grid from this config (parameter_grid_for_config), not
                # from an argument, so it must match the grid used below.
                "parameter_grid": {"entry_zscore": [0.5, 3.0]},
            },
        }
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "val",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 120,
                    "skip_period": 5,
                    "top_fraction": 0.4,
                },
            },
            "portfolio": {"allocator": "inverse_volatility", "maximum_weight": 0.6},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {
                "initial_capital": 100_000,
                "benchmark": {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
            },
            "validation": {"optimization_metric": "sharpe"},
        }
    )


def _config_with_grid(grid: dict[str, list[Any]]) -> ExperimentConfig:
    """`_config()` with an explicit `validation.parameter_grid`.

    `run_walk_forward_stress_tests()` verifies a passed-in `wf_baseline`
    was actually built with `parameter_grid_for_config(config)`'s own
    grid -- a baseline built with some other explicit grid (as several
    tests below do, for a cheaper/faster walk-forward run) must have its
    config configure that same grid, not rely on the strategy's default.
    """
    config = _config()
    return config.revalidated_copy(
        update={
            "validation": config.validation.revalidated_copy(
                update={"method": "walk_forward", "parameter_grid": grid}
            )
        }
    )


@pytest.mark.slow
def test_walk_forward_produces_oos_curve() -> None:
    data = _panel()
    validator = WalkForwardValidator(_config())
    result = validator.run(
        data,
        parameter_grid={"lookback_period": [60, 120]},
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
    )
    assert len(result.folds) >= 1
    # OOS curve exists and every chosen param is from the grid.
    assert len(result.oos_returns) > 0
    for fold in result.folds:
        assert fold.best_params["lookback_period"] in (60, 120)
    table = result.summary_table()
    assert "test_sharpe" in table.columns


@pytest.mark.slow
def test_walk_forward_run_reports_fold_progress() -> None:
    """on_progress must fire once before the first unit of work (done=0) and
    once per completed unit thereafter — one per candidate considered on a
    fold's validation block plus one more for that fold's out-of-sample
    weights, ending at (total_units, total_units). Not one tick per whole
    fold: a fold's own grid search can take long enough that fold-level
    ticks alone left the dashboard's live pace estimate with too few, too
    lumpy data points to track a real, sustained slowdown across an
    expanding walk-forward's later, bigger folds."""
    data = _panel()
    validator = WalkForwardValidator(_config())
    progress_calls: list[tuple[int, int]] = []
    grid = {"lookback_period": [60, 120]}
    result = validator.run(
        data,
        parameter_grid=grid,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )
    n_folds = len(result.folds)
    assert n_folds >= 1
    total_units = n_folds * (len(grid["lookback_period"]) + 1)
    assert progress_calls[0] == (0, total_units)
    assert progress_calls[-1] == (total_units, total_units)
    assert [done for done, _ in progress_calls] == list(range(total_units + 1))
    assert all(total == total_units for _, total in progress_calls)


@pytest.mark.slow
def test_walk_forward_run_resumes_from_a_checkpoint_and_matches_a_fresh_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted run's checkpoint, resumed, must produce exactly the
    same WalkForwardResult as an uninterrupted run — resuming is not an
    approximation, it picks up the identical remaining computation."""
    data = _panel()
    grid = {"lookback_period": [60, 120]}
    checkpoint_path = tmp_path / "checkpoint.pkl"

    fresh = WalkForwardValidator(_config()).run(
        data,
        parameter_grid=grid,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(_config()).run(
            data,
            parameter_grid=grid,
            train_window=300,
            validation_window=120,
            test_window=120,
            expanding=True,
            checkpoint_path=checkpoint_path,
        )
    assert checkpoint_path.is_file()
    monkeypatch.undo()

    resumed = WalkForwardValidator(_config()).run(
        data,
        parameter_grid=grid,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        checkpoint_path=checkpoint_path,
    )
    assert not checkpoint_path.is_file()  # cleared after completing successfully
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)
    assert [f.best_params for f in resumed.folds] == [
        f.best_params for f in fresh.folds
    ]


@pytest.mark.slow
def test_walk_forward_run_refuses_a_checkpoint_whose_window_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose fold_windows entry has drifted from what this
    run's own prepared.windows actually are (right list length, wrong
    content) must never be resumed from -- a length-only check would accept
    it and silently stitch the wrong fold's parameters/targets into the OOS
    curve. Content, not just count, must match."""
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )
    from quantlab.validation.splits import WalkForwardWindow

    data = _panel()
    grid = {"lookback_period": [60, 120]}
    checkpoint_path = tmp_path / "checkpoint.pkl"
    windows_kwargs: _WalkForwardWindows = {
        "train_window": 300,
        "validation_window": 120,
        "test_window": 120,
        "expanding": True,
    }

    fresh = WalkForwardValidator(_config()).run(
        data, parameter_grid=grid, **windows_kwargs
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(_config()).run(
            data,
            parameter_grid=grid,
            checkpoint_path=checkpoint_path,
            **windows_kwargs,
        )
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(
        _config(),
        data,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        execution_delay=0,
        parameter_grid={"lookback_period": [60, 120]},
    )
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (fold_windows, fold_parameters, fold_scores, target_pieces), progress = loaded
    real_window = fold_windows[0]
    corrupted_window = WalkForwardWindow(
        fold=real_window.fold,
        train=real_window.train,
        validation=real_window.validation,
        test=real_window.test[:-1],  # a genuinely different test block
    )
    save_checkpoint(
        checkpoint_path,
        provenance,
        ([corrupted_window], fold_parameters, fold_scores, target_pieces),
        progress,
    )

    resumed = WalkForwardValidator(_config()).run(
        data,
        parameter_grid=grid,
        checkpoint_path=checkpoint_path,
        **windows_kwargs,
    )
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)


@pytest.mark.slow
def test_walk_forward_run_refuses_a_checkpoint_with_a_garbage_target_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpointed target-weights frame with the wrong columns and a
    date from a completely different era (e.g. 1900) passes a bare
    isinstance(pd.DataFrame) check but is obviously not this fold's real
    target weights -- content, not just Python type, must be verified."""
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )

    data = _panel()
    grid = {"lookback_period": [60, 120]}
    checkpoint_path = tmp_path / "checkpoint.pkl"
    windows_kwargs: _WalkForwardWindows = {
        "train_window": 300,
        "validation_window": 120,
        "test_window": 120,
        "expanding": True,
    }

    fresh = WalkForwardValidator(_config()).run(
        data, parameter_grid=grid, **windows_kwargs
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(_config()).run(
            data,
            parameter_grid=grid,
            checkpoint_path=checkpoint_path,
            **windows_kwargs,
        )
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(
        _config(),
        data,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        execution_delay=0,
        parameter_grid={"lookback_period": [60, 120]},
    )
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (fold_windows, fold_parameters, fold_scores, target_pieces), progress = loaded
    garbage = pd.DataFrame({"WRONG": [1.0]}, index=pd.to_datetime(["1900-01-01"]))
    corrupted_targets = [garbage, *target_pieces[1:]]
    save_checkpoint(
        checkpoint_path,
        provenance,
        (fold_windows, fold_parameters, fold_scores, corrupted_targets),
        progress,
    )

    resumed = WalkForwardValidator(_config()).run(
        data,
        parameter_grid=grid,
        checkpoint_path=checkpoint_path,
        **windows_kwargs,
    )
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)


@pytest.mark.slow
def test_walk_forward_run_refuses_a_checkpoint_with_an_incomplete_target_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpointed target-weights frame missing one date from its own
    fold's test window is a genuine *subset* -- right columns, only
    in-window dates -- so a subset-only index check (``index <=
    expected``) would wrongly accept it. The stitched OOS curve needs every
    date in the window, so a strictly incomplete frame must be rejected
    too, not only one carrying foreign or extra dates."""
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )

    data = _panel()
    grid = {"lookback_period": [60, 120]}
    checkpoint_path = tmp_path / "checkpoint.pkl"
    windows_kwargs: _WalkForwardWindows = {
        "train_window": 300,
        "validation_window": 120,
        "test_window": 120,
        "expanding": True,
    }

    fresh = WalkForwardValidator(_config()).run(
        data, parameter_grid=grid, **windows_kwargs
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(_config()).run(
            data,
            parameter_grid=grid,
            checkpoint_path=checkpoint_path,
            **windows_kwargs,
        )
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(
        _config(),
        data,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        execution_delay=0,
        parameter_grid={"lookback_period": [60, 120]},
    )
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (fold_windows, fold_parameters, fold_scores, target_pieces), progress = loaded
    incomplete = target_pieces[0].iloc[1:]
    corrupted_targets = [incomplete, *target_pieces[1:]]
    save_checkpoint(
        checkpoint_path,
        provenance,
        (fold_windows, fold_parameters, fold_scores, corrupted_targets),
        progress,
    )

    resumed = WalkForwardValidator(_config()).run(
        data,
        parameter_grid=grid,
        checkpoint_path=checkpoint_path,
        **windows_kwargs,
    )
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)


@pytest.mark.slow
def test_walk_forward_run_refuses_a_checkpoint_with_a_non_grid_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpointed parameter dict that isn't actually one of this run's
    own grid candidates (e.g. left over from a since-changed grid) passes a
    bare isinstance(dict) check but could only have come from a stale or
    unrelated checkpoint."""
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )

    data = _panel()
    grid = {"lookback_period": [60, 120]}
    checkpoint_path = tmp_path / "checkpoint.pkl"
    windows_kwargs: _WalkForwardWindows = {
        "train_window": 300,
        "validation_window": 120,
        "test_window": 120,
        "expanding": True,
    }

    fresh = WalkForwardValidator(_config()).run(
        data, parameter_grid=grid, **windows_kwargs
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(_config()).run(
            data,
            parameter_grid=grid,
            checkpoint_path=checkpoint_path,
            **windows_kwargs,
        )
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(
        _config(),
        data,
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        execution_delay=0,
        parameter_grid={"lookback_period": [60, 120]},
    )
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (fold_windows, fold_parameters, fold_scores, target_pieces), progress = loaded
    # 999 was never in the grid [60, 120].
    corrupted_parameters = [{"lookback_period": 999}, *fold_parameters[1:]]
    save_checkpoint(
        checkpoint_path,
        provenance,
        (fold_windows, corrupted_parameters, fold_scores, target_pieces),
        progress,
    )

    resumed = WalkForwardValidator(_config()).run(
        data,
        parameter_grid=grid,
        checkpoint_path=checkpoint_path,
        **windows_kwargs,
    )
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)


@pytest.mark.slow
def test_walk_forward_run_does_not_reuse_a_checkpoint_from_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint written by one config must not be silently reused by a
    later run against a different one sharing the same checkpoint path —
    every candidate must actually be (re-)computed, not skipped."""
    data = _panel()
    checkpoint_path = tmp_path / "checkpoint.pkl"

    # Interrupt after the 1st fold completes, so there is something on disk
    # to (not) reuse below.
    real_select = WalkForwardValidator._select_on_validation
    starts = {"n": 0}

    def _flaky_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_select(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _flaky_select)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(_config()).run(
            data,
            parameter_grid={"lookback_period": [60, 120]},
            train_window=300,
            validation_window=120,
            test_window=120,
            expanding=True,
            checkpoint_path=checkpoint_path,
        )
    assert checkpoint_path.is_file()
    monkeypatch.undo()

    calls = {"n": 0}
    real_select2 = WalkForwardValidator._select_on_validation

    def _counting_select(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_select2(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_on_validation", _counting_select)
    # A different grid: same checkpoint path, different provenance.
    different_config = _config()
    result = WalkForwardValidator(different_config).run(
        data,
        parameter_grid={"lookback_period": [60, 90, 120]},
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
        checkpoint_path=checkpoint_path,
    )
    # Every fold was actually selected on, not skipped as "already done".
    assert calls["n"] == len(result.folds)


@pytest.mark.slow
def test_walk_forward_oos_result_is_a_coherent_backtest_result() -> None:
    """`oos_result` must be a genuine BacktestResult built from the stitched
    OOS series, reusing the same trade-log/benchmark/metrics pipeline as a
    single backtest, so dashboard/report rendering works on it unchanged."""
    data = _panel()
    validator = WalkForwardValidator(_config())
    result = validator.run(
        data,
        parameter_grid={"lookback_period": [60, 120]},
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
    )

    oos = result.oos_result
    assert oos is not None
    assert oos.config is validator.base_config
    # The BacktestResult's own series must be exactly the stitched OOS series.
    assert oos.returns.equals(result.oos_returns)
    assert oos.equity_curve.equals(result.oos_equity)
    assert list(oos.returns.index) == list(oos.equity_curve.index)

    for key in ("sharpe_ratio", "total_return", "cagr", "max_drawdown"):
        assert np.isfinite(oos.metrics[key])

    # cross_sectional_momentum with monthly rebalancing over multiple folds
    # must produce real trades, not an empty placeholder trade log.
    assert len(oos.trades) > 0
    assert set(oos.weights.columns) == set(validator.base_config.symbols)
    assert oos.target_weights is not None
    assert set(oos.target_weights.columns) == set(validator.base_config.symbols)
    assert oos.gross_returns is not None
    assert oos.gross_equity is not None

    assert oos.metadata["code_hash"]
    assert oos.metadata["data_hash"]
    assert oos.metadata["walk_forward_execution_delay"] == 0


def test_walk_forward_oos_result_is_none_without_any_fold() -> None:
    """No fold fits the requested windows in the available history, so
    there is no OOS series to build a BacktestResult from."""
    data = _panel()
    validator = WalkForwardValidator(_config())
    result = validator.run(
        data,
        parameter_grid={},
        train_window=10_000,
        validation_window=120,
        test_window=120,
        expanding=True,
    )

    assert result.folds == []
    assert result.oos_result is None


@pytest.mark.slow
def test_walk_forward_execution_delay_changes_the_oos_series() -> None:
    """execution_delay must be re-run through the whole selection process
    (not just rescale the final numbers): it feeds every per-fold candidate
    evaluation via `_weights_for_window`, so a delayed run's OOS series
    should differ from an undelayed one for a strategy that actually trades."""
    data = _panel()
    base_kwargs: _WalkForwardRunKwargs = {
        "parameter_grid": {"lookback_period": [60, 120]},
        "train_window": 300,
        "validation_window": 120,
        "test_window": 120,
        "expanding": True,
    }
    undelayed = WalkForwardValidator(_config()).run(data, **base_kwargs)
    delayed = WalkForwardValidator(_config()).run(
        data, **base_kwargs, execution_delay=1
    )

    assert undelayed.oos_result is not None
    assert delayed.oos_result is not None
    assert delayed.oos_result.metadata["walk_forward_execution_delay"] == 1
    assert not delayed.oos_result.returns.equals(undelayed.oos_result.returns)


@pytest.mark.parametrize("bad_delay", [-1, True, 1.5])
def test_walk_forward_run_rejects_invalid_execution_delay(bad_delay: object) -> None:
    data = _panel()
    validator = WalkForwardValidator(_config())
    with pytest.raises(InvalidConfigurationError, match="execution_delay"):
        validator.run(
            data,
            parameter_grid={},
            train_window=300,
            validation_window=120,
            test_window=120,
            execution_delay=bad_delay,  # type: ignore[arg-type]
        )


@pytest.mark.slow
def test_parameter_sensitivity_grid() -> None:
    data = _panel()
    sens = run_parameter_sensitivity(
        data,
        _config(),
        parameter_x="lookback_period",
        values_x=[60, 120],
        parameter_y="top_fraction",
        values_y=[0.3, 0.5],
    )
    assert len(sens) == 4
    assert {"sharpe", "cagr", "max_drawdown"} <= set(sens.columns)
    heat = sensitivity_heatmap_data(sens, "lookback_period", "top_fraction", "sharpe")
    assert heat.shape == (2, 2)


def test_parameter_sensitivity_rejects_a_boolean_parameter_name() -> None:
    """long_short is a structural switch (default False) — sweeping it
    changes whether bottom_fraction even matters, so it must be rejected
    the same way an unknown parameter name is, for both the plain and
    walk-forward-aware sensitivity sweeps."""
    data = _panel()
    for sweep in (run_parameter_sensitivity, run_walk_forward_parameter_sensitivity):
        with pytest.raises(ValueError, match="Unknown or unsweepable"):
            sweep(
                data,
                _config(),
                parameter_x="lookback_period",
                values_x=[60, 120],
                parameter_y="long_short",
                values_y=[True, False],
            )


def test_parameter_sensitivity_rejects_boolean_candidate_values() -> None:
    """Even a numeric-looking parameter must reject boolean candidate
    values — Python's bool is an int subclass, so True/False could
    otherwise slip through undetected as 1/0."""
    data = _panel()
    for sweep in (run_parameter_sensitivity, run_walk_forward_parameter_sensitivity):
        with pytest.raises(ValueError, match="boolean"):
            sweep(
                data,
                _config(),
                parameter_x="lookback_period",
                values_x=[True, False],
                parameter_y="top_fraction",
                values_y=[0.3, 0.5],
            )


@pytest.mark.slow
def test_run_walk_forward_stress_tests_reselects_parameters_under_higher_costs() -> (
    None
):
    """The methodological point of the whole process-level/returns-level
    split: a Walk-forward mode stress scenario must genuinely re-run
    selection, not rescale a fixed baseline's weights. entry_zscore=0.5
    trades far more often than 3.0 on this mean-reverting panel, so it must
    lose ground once commission is stressed 5x — and the stress-test
    function's own numbers must come from that re-selected run."""
    data = _cost_sensitive_panel()
    config = _cost_sensitive_config(commission_bps=50.0)
    grid = {"entry_zscore": [0.5, 3.0]}
    windows: _WalkForwardWindows = {
        "train_window": 150,
        "validation_window": 60,
        "test_window": 60,
        "expanding": True,
    }

    wf_baseline = WalkForwardValidator(config).run(data, parameter_grid=grid, **windows)
    assert wf_baseline.oos_result is not None
    baseline_choices = [fold.best_params["entry_zscore"] for fold in wf_baseline.folds]
    # The high-turnover parameter must win at least one fold at baseline cost
    # — otherwise there is nothing for higher costs to knock it away from.
    assert 0.5 in baseline_choices

    x5_config = scale_costs(config, commission_mult=5.0)
    wf_x5 = WalkForwardValidator(x5_config).run(data, parameter_grid=grid, **windows)
    assert wf_x5.oos_result is not None
    x5_choices = [fold.best_params["entry_zscore"] for fold in wf_x5.folds]

    # The core assertion: re-running walk-forward selection under 5x
    # commission actually changes which parameter wins on at least one fold.
    assert x5_choices != baseline_choices
    assert x5_choices.count(0.5) < baseline_choices.count(0.5)

    # run_walk_forward_stress_tests's own "commission x5" row must be derived
    # from exactly this re-selected run, not from the baseline's fixed weights.
    stress = run_walk_forward_stress_tests(data, config, wf_baseline)
    row = stress.loc[stress["scenario"] == "commission x5"].iloc[0]
    expected_sharpe = M.sharpe_ratio(
        wf_x5.oos_result.returns, config.risk_free_rate, config.periods_per_year
    )
    assert row["sharpe"] == pytest.approx(expected_sharpe)
    assert row["status"] == "ok"


@pytest.mark.slow
def test_run_with_weight_cache_matches_plain_run() -> None:
    """run_with_weight_cache() must select the same winners and produce the
    same OOS series as run() for identical inputs — the cache is a
    by-product of the same computation, not a different one."""
    data = _cost_sensitive_panel()
    config = _cost_sensitive_config(commission_bps=50.0)
    grid = {"entry_zscore": [0.5, 3.0]}
    windows: _WalkForwardWindows = {
        "train_window": 150,
        "validation_window": 60,
        "test_window": 60,
        "expanding": True,
    }

    plain = WalkForwardValidator(config).run(data, parameter_grid=grid, **windows)
    cached, _weight_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, **windows
    )

    assert plain.oos_result is not None
    assert cached.oos_result is not None
    assert [f.best_params for f in cached.folds] == [f.best_params for f in plain.folds]
    pd.testing.assert_series_equal(cached.oos_result.returns, plain.oos_result.returns)


@pytest.mark.slow
def test_run_with_weight_cache_resumes_from_a_checkpoint_and_matches_a_fresh_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee as :func:`test_walk_forward_run_resumes_from_a_checkpoint_
    and_matches_a_fresh_run`, for run_with_weight_cache() specifically — this is
    the method run_walk_forward_stress_tests() relies on to avoid rebuilding
    its weight cache from scratch on resume, so it must be independently
    resumable and bit-for-bit reproducible on its own."""
    data = _cost_sensitive_panel()
    config = _cost_sensitive_config(commission_bps=50.0)
    grid = {"entry_zscore": [0.5, 3.0]}
    windows: _WalkForwardWindows = {
        "train_window": 150,
        "validation_window": 60,
        "test_window": 60,
        "expanding": True,
    }
    checkpoint_path = tmp_path / "checkpoint.pkl"

    fresh, _fresh_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, **windows
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_capture = WalkForwardValidator._select_and_capture
    starts = {"n": 0}

    def _flaky_capture(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_capture(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_and_capture", _flaky_capture)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(config).run_with_weight_cache(
            data, parameter_grid=grid, checkpoint_path=checkpoint_path, **windows
        )
    assert checkpoint_path.is_file()
    monkeypatch.undo()

    resumed, _resumed_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, checkpoint_path=checkpoint_path, **windows
    )
    assert not checkpoint_path.is_file()
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)
    assert [f.best_params for f in resumed.folds] == [
        f.best_params for f in fresh.folds
    ]


@pytest.mark.slow
def test_run_with_weight_cache_refuses_a_checkpoint_with_a_mismatched_candidate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee as :func:`test_walk_forward_run_refuses_a_checkpoint_
    whose_window_does_not_match`, for run_with_weight_cache()'s own 5-list
    state: a fold's cached-candidates entry with the wrong number of
    candidates (stale from a different parameter_grid) must never be
    resumed from -- only a length-only list-of-lists check would miss this,
    since the outer list count alone stays correct."""
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )

    data = _cost_sensitive_panel()
    config = _cost_sensitive_config(commission_bps=50.0)
    grid = {"entry_zscore": [0.5, 3.0]}
    windows: _WalkForwardWindows = {
        "train_window": 150,
        "validation_window": 60,
        "test_window": 60,
        "expanding": True,
    }
    checkpoint_path = tmp_path / "checkpoint.pkl"

    fresh, _fresh_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, **windows
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_capture = WalkForwardValidator._select_and_capture
    starts = {"n": 0}

    def _flaky_capture(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_capture(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_and_capture", _flaky_capture)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(config).run_with_weight_cache(
            data, parameter_grid=grid, checkpoint_path=checkpoint_path, **windows
        )
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(
        config,
        data,
        train_window=150,
        validation_window=60,
        test_window=60,
        expanding=True,
        execution_delay=0,
        parameter_grid={"entry_zscore": [0.5, 3.0]},
    )
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (windows_l, parameters, scores, targets, cached_candidates), progress = loaded
    # Drop one cached candidate from the first fold -- stale as if this
    # cache had been built against a smaller grid.
    corrupted_cache = [cached_candidates[0][:-1], *cached_candidates[1:]]
    save_checkpoint(
        checkpoint_path,
        provenance,
        (windows_l, parameters, scores, targets, corrupted_cache),
        progress,
    )

    resumed, _resumed_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, checkpoint_path=checkpoint_path, **windows
    )
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)


@pytest.mark.slow
def test_run_with_weight_cache_refuses_a_checkpoint_with_a_corrupted_candidate_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee as :func:`test_run_with_weight_cache_refuses_a_
    checkpoint_with_a_mismatched_candidate_cache`, but for a candidate's own
    weights content instead of the cache's length: a candidate whose
    ``validation_weights`` carries a NaN is structurally a DataFrame of the
    right shape (columns/index untouched) -- only a content check, not a
    bare isinstance/None check, catches it."""
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )
    from quantlab.validation.walk_forward import _FoldCandidateWeights

    data = _cost_sensitive_panel()
    config = _cost_sensitive_config(commission_bps=50.0)
    grid = {"entry_zscore": [0.5, 3.0]}
    windows: _WalkForwardWindows = {
        "train_window": 150,
        "validation_window": 60,
        "test_window": 60,
        "expanding": True,
    }
    checkpoint_path = tmp_path / "checkpoint.pkl"

    fresh, _fresh_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, **windows
    )
    assert len(fresh.folds) >= 2, "need at least 2 folds to interrupt after the 1st"

    real_capture = WalkForwardValidator._select_and_capture
    starts = {"n": 0}

    def _flaky_capture(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_capture(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_and_capture", _flaky_capture)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        WalkForwardValidator(config).run_with_weight_cache(
            data, parameter_grid=grid, checkpoint_path=checkpoint_path, **windows
        )
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(
        config,
        data,
        train_window=150,
        validation_window=60,
        test_window=60,
        expanding=True,
        execution_delay=0,
        parameter_grid={"entry_zscore": [0.5, 3.0]},
    )
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (windows_l, parameters, scores, targets, cached_candidates), progress = loaded

    first_fold = cached_candidates[0]
    first_candidate = next(
        c for c in first_fold if isinstance(c.validation_weights, pd.DataFrame)
    )
    corrupted_weights = first_candidate.validation_weights.copy()
    corrupted_weights.iloc[0, 0] = np.nan
    corrupted_candidate = _FoldCandidateWeights(
        corrupted_weights, first_candidate.test_targets
    )
    corrupted_first_fold = [
        corrupted_candidate if c is first_candidate else c for c in first_fold
    ]
    corrupted_cache = [corrupted_first_fold, *cached_candidates[1:]]
    save_checkpoint(
        checkpoint_path,
        provenance,
        (windows_l, parameters, scores, targets, corrupted_cache),
        progress,
    )

    resumed, _resumed_cache = WalkForwardValidator(config).run_with_weight_cache(
        data, parameter_grid=grid, checkpoint_path=checkpoint_path, **windows
    )
    assert resumed.oos_result is not None
    assert fresh.oos_result is not None
    pd.testing.assert_series_equal(resumed.oos_result.returns, fresh.oos_result.returns)


@pytest.mark.slow
def test_rescore_with_costs_matches_a_fresh_scenario_run() -> None:
    """rescore_with_costs() must reproduce a fresh full re-run's fold
    selection and OOS returns under a scaled-cost scenario — it must be a
    faster way to compute the same answer, not an approximation."""
    data = _cost_sensitive_panel()
    config = _cost_sensitive_config(commission_bps=50.0)
    grid = {"entry_zscore": [0.5, 3.0]}
    windows: _WalkForwardWindows = {
        "train_window": 150,
        "validation_window": 60,
        "test_window": 60,
        "expanding": True,
    }

    validator = WalkForwardValidator(config)
    _, weight_cache = validator.run_with_weight_cache(
        data, parameter_grid=grid, **windows
    )

    x5_config = scale_costs(config, commission_mult=5.0)
    rescored = validator.rescore_with_costs(weight_cache, x5_config)
    fresh = WalkForwardValidator(x5_config).run(data, parameter_grid=grid, **windows)

    assert rescored.oos_result is not None
    assert fresh.oos_result is not None
    assert [f.best_params for f in rescored.folds] == [
        f.best_params for f in fresh.folds
    ]
    pd.testing.assert_series_equal(
        rescored.oos_result.returns, fresh.oos_result.returns
    )
    assert rescored.oos_result.metrics["total_cost_fraction"] == pytest.approx(
        fresh.oos_result.metrics["total_cost_fraction"]
    )


@pytest.mark.slow
def test_run_walk_forward_stress_tests_best_days_removed_reuses_baseline() -> None:
    """The one scenario that changes no configuration must not re-run the
    walk-forward process — it is a direct post-hoc transform of the
    baseline's already-realised OOS returns."""
    data = _panel()
    grid = {"lookback_period": [60, 120]}
    # run_walk_forward_stress_tests() now verifies wf_baseline was actually
    # built with this call's own grid (parameter_grid_for_config(config))
    # and windows (resolve_walk_forward_windows(config)), so both must match
    # what the baseline below is built with.
    config = _config_with_grid(grid)
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid=grid,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
    )
    assert wf_baseline.oos_result is not None

    stress = run_walk_forward_stress_tests(data, config, wf_baseline)
    assert set(stress["scenario"]) >= {
        "baseline",
        "commission x2",
        "commission x5",
        "slippage x2",
        "execution delay +1",
        "best 10 days removed",
    }
    from quantlab.risk.stress import remove_best_days

    expected_returns = remove_best_days(wf_baseline.oos_result.returns, 10)
    expected_sharpe = M.sharpe_ratio(
        expected_returns, config.risk_free_rate, config.periods_per_year
    )
    row = stress.loc[stress["scenario"] == "best 10 days removed"].iloc[0]
    assert row["sharpe"] == pytest.approx(expected_sharpe)


def test_run_walk_forward_stress_tests_rejects_a_baseline_without_oos_result() -> None:
    from quantlab.validation.walk_forward import WalkForwardResult

    data = _panel()
    config = _config()
    with pytest.raises(ValueError, match="no OOS result"):
        run_walk_forward_stress_tests(data, config, WalkForwardResult())


def _baseline_windows() -> _WalkForwardWindows:
    return {
        "train_window": 300,
        "validation_window": 120,
        "test_window": 120,
        "expanding": True,
    }


@pytest.mark.slow
def test_wf_stress_tests_rejects_a_baseline_built_from_a_different_config() -> None:
    """Every stress scenario below is derived from `config` (windows, grid)
    and assumed to correspond to `wf_baseline`'s own methodology -- a
    baseline actually built from a *different* config would silently mix
    two methodologies (e.g. different train/test windows) without this
    check, corrupting every scenario that reuses its cached weights."""
    data = _panel()
    grid = {"lookback_period": [60, 120]}
    baseline_config = _config()
    wf_baseline = WalkForwardValidator(baseline_config).run(
        data, parameter_grid=grid, **_baseline_windows()
    )
    assert wf_baseline.oos_result is not None

    different_config = baseline_config.revalidated_copy(
        update={
            "portfolio": baseline_config.portfolio.revalidated_copy(
                update={"maximum_weight": 0.9}
            )
        }
    )
    with pytest.raises(ValueError, match="wf_baseline was not built from `config`"):
        run_walk_forward_stress_tests(data, different_config, wf_baseline)


@pytest.mark.slow
def test_wf_stress_tests_rejects_a_baseline_built_from_different_data() -> None:
    """A baseline computed against one data panel, passed alongside a
    *different* panel, must be rejected rather than silently stress-testing
    against data that never actually produced it."""
    data = _panel()
    grid = {"lookback_period": [60, 120]}
    config = _config()
    wf_baseline = WalkForwardValidator(config).run(
        data, parameter_grid=grid, **_baseline_windows()
    )
    assert wf_baseline.oos_result is not None

    different_data = _panel(n=901)  # different length -> different data_hash
    with pytest.raises(ValueError, match="wf_baseline was not built from `data`"):
        run_walk_forward_stress_tests(different_data, config, wf_baseline)


@pytest.mark.slow
def test_run_walk_forward_stress_tests_reports_scenario_progress() -> None:
    """on_progress must fire once before any work starts (done=0) and once
    per completed unit of work — first one tick per fold while the weight
    cache shared by the cost-only scenarios is (re)built, then one tick per
    remaining scenario — ending at (total, total)."""
    data = _panel()
    grid = {"lookback_period": [60, 120]}
    config = _config_with_grid(grid)
    # run_walk_forward_stress_tests() resolves its own windows from `config`
    # (resolve_walk_forward_windows), so wf_baseline must be built the same
    # way real callers (dashboard/CLI) do — otherwise its fold count
    # wouldn't match the weight cache's, which this progress accounting
    # relies on.
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid=grid,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
    )
    assert wf_baseline.oos_result is not None

    progress_calls: list[tuple[int, int]] = []
    stress = run_walk_forward_stress_tests(
        data,
        config,
        wf_baseline,
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )
    # 3 symbols configured -> the reduced-universe scenario is also included.
    n_scenarios = len(stress) - 1  # every row except "baseline" is a scenario
    # run_walk_forward_stress_tests() resolves its own grid from `config`
    # too (parameter_grid_for_config), independently of the grid used above
    # to build wf_baseline — cache-building progress is reported per
    # candidate (folds x this grid's size), not per fold.
    from quantlab.validation.parameter_grid import parameter_grid_for_config

    grid = parameter_grid_for_config(config)
    n_combinations = math.prod(len(values) for values in grid.values())
    total_units = len(wf_baseline.folds) * n_combinations + n_scenarios
    assert progress_calls[0] == (0, total_units)
    assert progress_calls[-1] == (total_units, total_units)
    assert [done for done, _ in progress_calls] == list(range(total_units + 1))


@pytest.mark.slow
def test_run_walk_forward_stress_tests_resumes_the_weight_cache_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weight-cache build inside run_walk_forward_stress_tests() — the
    expensive part the cache exists to amortise across the 3 cost-only
    scenarios — must itself be resumable via its own nested checkpoint: an
    interruption partway through it must not force rebuilding the cache
    from scratch, or the whole point of caching is defeated on any restart.
    """
    data = _panel()
    grid = {"lookback_period": [60, 120]}
    config = _config_with_grid(grid)
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid=grid,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
    )
    assert wf_baseline.oos_result is not None
    n_folds = len(wf_baseline.folds)
    assert n_folds >= 2, "need >= 2 folds to interrupt mid-cache-build"

    checkpoint_path = tmp_path / "checkpoint.pkl"
    cache_checkpoint_path = tmp_path / "checkpoint_cache.pkl"

    real_capture = WalkForwardValidator._select_and_capture
    starts = {"n": 0}

    def _flaky_capture(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_capture(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_and_capture", _flaky_capture)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_walk_forward_stress_tests(
            data, config, wf_baseline, checkpoint_path=checkpoint_path
        )
    # "baseline" (block 1) completed and checkpointed before the cache-build
    # (block 2) even started, so the outer, scenario-level checkpoint exists
    # too — it just still only has 1 block recorded.
    assert checkpoint_path.is_file()
    assert cache_checkpoint_path.is_file()
    monkeypatch.undo()

    calls = {"n": 0}
    real_capture2 = WalkForwardValidator._select_and_capture

    def _counting_capture(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_capture2(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "_select_and_capture", _counting_capture)
    resumed = run_walk_forward_stress_tests(
        data, config, wf_baseline, checkpoint_path=checkpoint_path
    )
    # Only the folds *not* already cached at interruption time were
    # recomputed — proof the cache build actually resumed, not restarted.
    assert calls["n"] == n_folds - 1
    assert not checkpoint_path.is_file()
    assert not cache_checkpoint_path.is_file()
    monkeypatch.undo()

    fresh = run_walk_forward_stress_tests(data, config, wf_baseline)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_run_walk_forward_stress_tests_resume_after_cost_block_runs_every_later_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: the scenario-block checkpoint's progress must be
    tracked explicitly, never re-derived from len(rows). Block 2 (cache
    build + 3 cost-only rescores: commission x2, commission x5, slippage x2)
    alone appends 1 (baseline) + 3 = 4 rows after only 2 blocks are done --
    re-deriving "how many blocks are done" from len(rows) would think 4
    blocks are done and skip block 3 (execution delay) and block 4 (best 10
    days removed) entirely on resume."""
    from quantlab.validation.checkpoint import compute_provenance, load_checkpoint

    data = _panel()
    config = _config_with_grid({})
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid={},
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
    )
    assert wf_baseline.oos_result is not None

    checkpoint_path = tmp_path / "checkpoint.pkl"

    # Block 2 (cache-build + cost rescores) uses run_with_weight_cache(), not
    # .run() -- patching .run() to always fail only intercepts block 3's
    # (execution delay) _run_walk_forward() call, which is a plain .run().
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated interruption right after block 2")

    monkeypatch.setattr(WalkForwardValidator, "run", _boom)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_walk_forward_stress_tests(
            data, config, wf_baseline, checkpoint_path=checkpoint_path
        )
    monkeypatch.undo()

    provenance = compute_provenance(config, data)
    checkpoint_result = load_checkpoint(checkpoint_path, provenance)
    assert checkpoint_result is not None
    state, progress = checkpoint_result
    assert len(state) == 4  # baseline + 3 cost scenarios
    assert progress == 2  # but only 2 *blocks* are actually done

    resumed = run_walk_forward_stress_tests(
        data, config, wf_baseline, checkpoint_path=checkpoint_path
    )
    # The blocks the len(rows)-based bug would have skipped on resume.
    assert {"execution delay +1", "best 10 days removed"} <= set(resumed["scenario"])
    assert not checkpoint_path.is_file()


@pytest.mark.slow
def test_run_walk_forward_stress_tests_refuses_a_checkpoint_with_wrong_scenario_names(
    tmp_path: Path,
) -> None:
    """A checkpoint whose row count and schema look right for progress=2 (4
    rows: baseline + 3 cost scenarios) but whose scenario names are wrong
    (e.g. all four rows literally named "baseline") is structurally
    plausible but incoherent -- it must never be resumed from, or the
    result table would silently carry duplicated/misnamed scenario rows."""
    import quantlab.validation.robustness as robustness_module
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    config = _config_with_grid({})
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid={},
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
    )
    assert wf_baseline.oos_result is not None

    checkpoint_path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(config, data)

    def make_row(name: str) -> dict[str, object]:
        # dict.fromkeys(...) would read more naturally here, but its return
        # type is inferred independently of `row`'s own declared type --
        # unlike a comprehension, which a type checker matches bidirectionally
        # against it -- so fromkeys(...) alone doesn't satisfy dict[str,
        # object] (dict is invariant in its value type).
        row: dict[str, object] = {  # noqa: C420
            column: None for column in robustness_module._STRESS_COLUMNS
        }
        row.update({"scenario": name, "status": "ok"})
        return row

    # Four rows, correct schema, correct count for progress=2 -- but every
    # row is wrongly named "baseline" instead of the real scenario names.
    corrupted_rows = [make_row("baseline") for _ in range(4)]
    save_checkpoint(checkpoint_path, provenance, corrupted_rows, 2)

    resumed = run_walk_forward_stress_tests(
        data, config, wf_baseline, checkpoint_path=checkpoint_path
    )
    fresh = run_walk_forward_stress_tests(data, config, wf_baseline)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_run_walk_forward_stress_tests_refuses_a_checkpoint_with_an_inconsistent_row(
    tmp_path: Path,
) -> None:
    """Same guarantee as :func:`test_run_walk_forward_stress_tests_refuses_
    a_checkpoint_with_wrong_scenario_names`, but for status/metrics/error
    consistency instead of scenario naming: a row claiming ``status=
    "failed"`` while still carrying finite metrics and no error message is
    just as incoherent as a misnamed one."""
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    config = _config_with_grid({})
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid={},
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        expanding=config.validation.expanding,
    )
    assert wf_baseline.oos_result is not None

    checkpoint_path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(config, data)

    scenario_names = ["baseline", "commission x2", "commission x5", "slippage x2"]

    def contradictory_row(name: str) -> dict[str, object]:
        return {
            "scenario": name,
            "total_return": 0.1,
            "cagr": 0.05,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "status": "failed",  # claims failure while metrics are finite
            "error": None,  # and no error message to go with it
        }

    corrupted_rows = [contradictory_row(name) for name in scenario_names]
    save_checkpoint(checkpoint_path, provenance, corrupted_rows, 2)

    resumed = run_walk_forward_stress_tests(
        data, config, wf_baseline, checkpoint_path=checkpoint_path
    )
    fresh = run_walk_forward_stress_tests(data, config, wf_baseline)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_walk_forward_parameter_sensitivity_grid() -> None:
    data = _panel()
    sens = run_walk_forward_parameter_sensitivity(
        data,
        _config(),
        parameter_x="lookback_period",
        values_x=[60, 120],
        parameter_y="top_fraction",
        values_y=[0.3, 0.5],
    )
    assert len(sens) == 4
    assert {"sharpe", "cagr", "max_drawdown", "status"} <= set(sens.columns)
    assert (sens["status"] == "ok").all()


@pytest.mark.slow
def test_walk_forward_parameter_sensitivity_reports_cell_progress() -> None:
    """on_progress must fire once before the first cell (done=0) and once
    per completed cell, ending at (n_cells, n_cells) — a coarser,
    cell-level signal since each cell is itself a full walk-forward run."""
    data = _panel()
    progress_calls: list[tuple[int, int]] = []
    sens = run_walk_forward_parameter_sensitivity(
        data,
        _config(),
        parameter_x="lookback_period",
        values_x=[60, 120],
        parameter_y="top_fraction",
        values_y=[0.3, 0.5],
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )
    n_cells = len(sens)
    assert n_cells == 4
    assert progress_calls[0] == (0, n_cells)
    assert progress_calls[-1] == (n_cells, n_cells)
    assert [done for done, _ in progress_calls] == list(range(n_cells + 1))


@pytest.mark.slow
def test_walk_forward_parameter_sensitivity_resumes_from_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted sweep, resumed, must produce exactly the same rows
    (same order, same values) as an uninterrupted one — cells are
    independent, so resuming is a matter of not recomputing already-done
    ones, not approximating anything."""
    data = _panel()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    sweep_kwargs: _SensitivitySweepKwargs = {
        "parameter_x": "lookback_period",
        "values_x": [60, 120],
        "parameter_y": "top_fraction",
        "values_y": [0.3, 0.5],
    }

    real_run = WalkForwardValidator.run
    starts = {"n": 0}

    def _flaky_run(self: WalkForwardValidator, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        starts["n"] += 1
        if starts["n"] == 3:
            raise RuntimeError("simulated interruption")
        return real_run(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WalkForwardValidator, "run", _flaky_run)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_walk_forward_parameter_sensitivity(
            data, _config(), checkpoint_path=checkpoint_path, **sweep_kwargs
        )
    assert checkpoint_path.is_file()
    monkeypatch.undo()

    resumed = run_walk_forward_parameter_sensitivity(
        data, _config(), checkpoint_path=checkpoint_path, **sweep_kwargs
    )
    assert not checkpoint_path.is_file()

    fresh = run_walk_forward_parameter_sensitivity(data, _config(), **sweep_kwargs)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_walk_forward_parameter_sensitivity_refuses_a_checkpoint_with_a_garbage_cell(
    tmp_path: Path,
) -> None:
    """A checkpoint state of ``["garbage"]`` for a one-cell sweep has the
    right Python type (list) and the right length (matches progress=1), but
    its single element is not a row dict at all -- a length-only check
    would wrongly accept it and let "garbage" flow straight into the result
    table."""
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    sweep_kwargs: _SensitivitySweepKwargs = {
        "parameter_x": "lookback_period",
        "values_x": [60],
        "parameter_y": "top_fraction",
        "values_y": [0.3],
    }
    provenance = compute_provenance(
        _config(),
        data,
        parameter_x="lookback_period",
        values_x=[60],
        parameter_y="top_fraction",
        values_y=[0.3],
    )
    save_checkpoint(checkpoint_path, provenance, ["garbage"], 1)

    resumed = run_walk_forward_parameter_sensitivity(
        data, _config(), checkpoint_path=checkpoint_path, **sweep_kwargs
    )
    fresh = run_walk_forward_parameter_sensitivity(data, _config(), **sweep_kwargs)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )
    assert "garbage" not in resumed.to_numpy()


@pytest.mark.slow
def test_walk_forward_parameter_sensitivity_recovers_from_a_pd_na_checkpoint(
    tmp_path: Path,
) -> None:
    """A checkpointed cell whose swept-parameter value is ``pd.NA`` used to
    crash the resume path with ``TypeError: boolean value of NA is
    ambiguous`` deep inside the comparison that checks a cell's value
    against its expected combination -- an exception that then propagated
    out of ``load_checkpoint`` despite its own contract that a corrupted
    checkpoint is only ever skipped, never raised. Must now be treated as
    "not a match" and trigger a fresh recompute instead of crashing the
    whole sweep."""
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    sweep_kwargs: _SensitivitySweepKwargs = {
        "parameter_x": "lookback_period",
        "values_x": [60],
        "parameter_y": "top_fraction",
        "values_y": [0.3],
    }
    provenance = compute_provenance(
        _config(),
        data,
        parameter_x="lookback_period",
        values_x=[60],
        parameter_y="top_fraction",
        values_y=[0.3],
    )
    corrupted_row = {
        "lookback_period": pd.NA,
        "top_fraction": 0.3,
        "sharpe": 1.0,
        "cagr": 0.1,
        "max_drawdown": -0.1,
        "turnover": 0.2,
        "num_trades": 5.0,
        "status": "ok",
        "error": None,
    }
    save_checkpoint(checkpoint_path, provenance, [corrupted_row], 1)

    resumed = run_walk_forward_parameter_sensitivity(
        data, _config(), checkpoint_path=checkpoint_path, **sweep_kwargs
    )
    fresh = run_walk_forward_parameter_sensitivity(data, _config(), **sweep_kwargs)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_walk_forward_sensitivity_differs_from_plain_backtest_sensitivity() -> None:
    """A walk-forward sensitivity cell re-runs the whole selection process on
    each fold, so it must not simply reproduce the plain single-backtest
    sensitivity's numbers for the same parameter combination."""
    data = _panel()
    config = _config()

    plain = run_parameter_sensitivity(
        data,
        config,
        parameter_x="lookback_period",
        values_x=[60],
        parameter_y="top_fraction",
        values_y=[0.3],
    )
    wf = run_walk_forward_parameter_sensitivity(
        data,
        config,
        parameter_x="lookback_period",
        values_x=[60],
        parameter_y="top_fraction",
        values_y=[0.3],
    )
    assert wf["status"].iloc[0] == "ok"
    assert plain["sharpe"].iloc[0] != pytest.approx(wf["sharpe"].iloc[0])


def test_bootstrap_summary_percentiles() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    boot = bootstrap_returns(returns, n_iterations=200, block_size=5, seed=42)
    summary = boot.summary()
    assert {"cagr", "sharpe", "max_drawdown", "final_value"} == set(
        summary["statistic"]
    )
    # Percentile ordering holds.
    for _, row in summary.iterrows():
        assert row["p05"] <= row["median"] <= row["p95"]


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300))
    a = bootstrap_returns(returns, n_iterations=100, seed=7).samples
    b = bootstrap_returns(returns, n_iterations=100, seed=7).samples
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.slow
def test_stress_tests_include_expected_scenarios() -> None:
    data = _panel()
    table = run_stress_tests(data, _config())
    scenarios = set(table["scenario"])
    assert "baseline" in scenarios
    assert "commission x5" in scenarios
    assert "best 10 days removed" in scenarios
    # Higher commissions cannot improve the net total return vs baseline.
    base = table.loc[table["scenario"] == "baseline", "total_return"].iloc[0]
    c5 = table.loc[table["scenario"] == "commission x5", "total_return"].iloc[0]
    assert c5 <= base + 1e-9


@pytest.mark.slow
def test_stress_tests_reports_scenario_progress() -> None:
    """on_progress must fire once before the first scenario (done=0) and
    once per completed scenario (baseline included), ending at
    (n_scenarios, n_scenarios)."""
    data = _panel()
    progress_calls: list[tuple[int, int]] = []
    table = run_stress_tests(
        data,
        _config(),
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )
    n_scenarios = len(table)  # every row, including "baseline", is a scenario
    assert progress_calls[0] == (0, n_scenarios)
    assert progress_calls[-1] == (n_scenarios, n_scenarios)
    assert [done for done, _ in progress_calls] == list(range(n_scenarios + 1))


@pytest.mark.slow
def test_stress_tests_resumes_from_a_checkpoint_including_baseline_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupting right after "baseline" and resuming must still get
    "best 10 days removed" right — it needs the actual baseline returns
    Series, not just its already-computed metrics row, so the checkpoint
    has to carry that Series across the interruption, not only `rows`."""
    import quantlab.validation.robustness as robustness_module

    data = _panel()
    config = _config()
    checkpoint_path = tmp_path / "checkpoint.pkl"

    real_backtest = robustness_module.run_backtest_from_config
    calls = {"n": 0}

    def _flaky_backtest(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:  # 1st call is "baseline"; interrupt right after it
            raise RuntimeError("simulated interruption")
        return real_backtest(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(robustness_module, "run_backtest_from_config", _flaky_backtest)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    assert checkpoint_path.is_file()
    monkeypatch.undo()

    resumed = run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    assert not checkpoint_path.is_file()
    fresh = run_stress_tests(data, config)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_stress_tests_refuses_a_structurally_plausible_but_incoherent_checkpoint(
    tmp_path: Path,
) -> None:
    """A checkpoint whose `progress` claims 1 scenario is done, but whose
    `rows` list is empty (or whose baseline_returns Series is missing), is
    structurally valid (a 2-tuple of a list and an optional Series) but
    incoherent -- it must never be resumed from, since it would otherwise
    silently under-report scenarios or crash "best 10 days removed" (which
    needs the actual baseline_returns Series, not just its metrics row)."""
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    config = _config()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(config, data)

    # progress=1 (baseline claimed done) but no rows and no baseline_returns.
    save_checkpoint(checkpoint_path, provenance, ([], None), 1)

    resumed = run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    fresh = run_stress_tests(data, config)
    # A trusted-but-corrupted checkpoint would have skipped recomputing
    # "baseline" entirely, producing a table missing rows (or crashing) --
    # matching a truly fresh run proves it started over instead.
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_stress_tests_refuses_a_checkpoint_with_wrong_scenario_names(
    tmp_path: Path,
) -> None:
    """A checkpoint whose row count and schema look right for progress=7
    (every scenario `_config()`'s 3-symbol universe produces, including
    "reduced universe") but whose scenario names are all "baseline" (with
    every metric missing to boot) is structurally plausible but incoherent
    -- it must never be resumed from, or the result table would silently
    carry seven duplicated "baseline" rows instead of the real scenarios."""
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    config = _config()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(config, data)

    def make_row(name: str) -> dict[str, object]:
        return {
            "scenario": name,
            "total_return": float("nan"),
            "cagr": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "status": "ok",
            "error": None,
        }

    corrupted_rows = [make_row("baseline") for _ in range(7)]
    save_checkpoint(checkpoint_path, provenance, (corrupted_rows, pd.Series([0.0])), 7)

    resumed = run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    fresh = run_stress_tests(data, config)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )
    assert list(resumed["scenario"]) != ["baseline"] * 7


@pytest.mark.slow
def test_stress_tests_refuses_a_checkpoint_with_an_inconsistent_row(
    tmp_path: Path,
) -> None:
    """A checkpoint row with the right scenario name and schema, but whose
    status/metrics/error disagree (here: ``status="ok"`` while every metric
    is NaN) is structurally plausible but incoherent -- it must never be
    resumed from."""
    from quantlab.validation.checkpoint import compute_provenance, save_checkpoint

    data = _panel()
    config = _config()
    checkpoint_path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(config, data)

    contradictory_row = {
        "scenario": "baseline",
        "total_return": float("nan"),
        "cagr": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "status": "ok",  # claims success while every metric is NaN
        "error": None,
    }
    save_checkpoint(
        checkpoint_path, provenance, ([contradictory_row], pd.Series([0.0])), 1
    )

    resumed = run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    fresh = run_stress_tests(data, config)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "kind", ["empty", "non_datetime_index", "object_dtype", "nan", "infinite"]
)
def test_stress_tests_refuses_a_checkpoint_with_a_corrupted_baseline_series(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare ``isinstance(baseline, pd.Series)`` check would accept an
    empty series, one indexed by something other than real dates, one
    holding non-numeric data, or one containing NaN/infinite values -- and
    "best 10 days removed" operates on this Series directly, not on the
    already-validated "baseline" metrics row, so a corrupted series reaches
    a real computation even when the row next to it looks fine. Interrupts
    a real run right after "baseline" (so the paired row is genuine), then
    swaps in each kind of corrupted series before resuming -- every case
    must be rejected and trigger a fresh recompute, never a crash or a
    silently-accepted bad series."""
    import quantlab.validation.robustness as robustness_module
    from quantlab.validation.checkpoint import (
        compute_provenance,
        load_checkpoint,
        save_checkpoint,
    )

    data = _panel()
    config = _config()
    checkpoint_path = tmp_path / "checkpoint.pkl"

    real_backtest = robustness_module.run_backtest_from_config
    calls = {"n": 0}

    def _flaky_backtest(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:  # 1st call is "baseline"; interrupt right after it
            raise RuntimeError("simulated interruption")
        return real_backtest(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(robustness_module, "run_backtest_from_config", _flaky_backtest)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    monkeypatch.undo()
    assert checkpoint_path.is_file()

    provenance = compute_provenance(config, data)
    loaded = load_checkpoint(checkpoint_path, provenance)
    assert loaded is not None
    (rows, _baseline), progress = loaded

    stray_index = pd.date_range("1900-01-01", periods=3)
    corrupted: object
    if kind == "empty":
        corrupted = pd.Series(dtype=float)
    elif kind == "non_datetime_index":
        corrupted = pd.Series([0.01, 0.02, 0.03])  # plain RangeIndex, not dates
    elif kind == "object_dtype":
        corrupted = pd.Series(["a", "b", "c"], index=stray_index)
    elif kind == "nan":
        corrupted = pd.Series([0.01, float("nan"), 0.02], index=stray_index)
    else:
        corrupted = pd.Series([0.01, float("inf"), 0.02], index=stray_index)

    save_checkpoint(checkpoint_path, provenance, (rows, corrupted), progress)

    resumed = run_stress_tests(data, config, checkpoint_path=checkpoint_path)
    fresh = run_stress_tests(data, config)
    pd.testing.assert_frame_equal(
        resumed.reset_index(drop=True), fresh.reset_index(drop=True)
    )


@pytest.mark.slow
def test_run_stress_tests_does_not_reuse_a_checkpoint_from_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantlab.validation.robustness as robustness_module

    data = _panel()
    checkpoint_path = tmp_path / "checkpoint.pkl"

    real_backtest = robustness_module.run_backtest_from_config
    calls = {"n": 0}

    def _flaky_backtest(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_backtest(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(robustness_module, "run_backtest_from_config", _flaky_backtest)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_stress_tests(data, _config(), checkpoint_path=checkpoint_path)
    assert checkpoint_path.is_file()
    monkeypatch.undo()

    # A config with a different commission -> different provenance -> the
    # checkpoint above must be ignored, not partially reused.
    different_config = scale_costs(_config(), commission_mult=3.0)
    result = run_stress_tests(data, different_config, checkpoint_path=checkpoint_path)
    expected = run_stress_tests(data, different_config)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_monte_carlo_permutation_reports_pvalue() -> None:
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    out = monte_carlo_permutation(returns, n_iterations=200, seed=42)
    assert 0.0 <= out["p_value"] <= 1.0
    assert "real_sharpe" in out
