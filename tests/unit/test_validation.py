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
                "source": "csv",
                "symbols": ["AAA", "BBB"],
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
                "market_calendar": "XNYS",
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
                "source": "csv",
                "symbols": ["AAA", "BBB", "CCC"],
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
                "market_calendar": "XNYS",
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
            "backtest": {"initial_capital": 100_000, "benchmark_symbol": "AAA"},
            "validation": {"optimization_metric": "sharpe"},
        }
    )


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


def test_run_walk_forward_stress_tests_best_days_removed_reuses_baseline() -> None:
    """The one scenario that changes no configuration must not re-run the
    walk-forward process — it is a direct post-hoc transform of the
    baseline's already-realised OOS returns."""
    data = _panel()
    config = _config()
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid={"lookback_period": [60, 120]},
        train_window=300,
        validation_window=120,
        test_window=120,
        expanding=True,
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


def test_run_walk_forward_stress_tests_reports_scenario_progress() -> None:
    """on_progress must fire once before any work starts (done=0) and once
    per completed unit of work — first one tick per fold while the weight
    cache shared by the cost-only scenarios is (re)built, then one tick per
    remaining scenario — ending at (total, total)."""
    data = _panel()
    config = _config()
    # run_walk_forward_stress_tests() resolves its own windows from `config`
    # (resolve_walk_forward_windows), so wf_baseline must be built the same
    # way real callers (dashboard/CLI) do — otherwise its fold count
    # wouldn't match the weight cache's, which this progress accounting
    # relies on.
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid={"lookback_period": [60, 120]},
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
    config = _config()
    train_window, validation_window, test_window = resolve_walk_forward_windows(config)
    wf_baseline = WalkForwardValidator(config).run(
        data,
        parameter_grid={"lookback_period": [60, 120]},
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
