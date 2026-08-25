"""Regression tests for validation-layer input and time-alignment contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import geometric_series, make_ohlcv

from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.cli import _default_grid
from quantlab.config import ExperimentConfig, ValidationConfig
from quantlab.exceptions import BacktestError, InvalidConfigurationError
from quantlab.validation.bootstrap import BootstrapResult, bootstrap_returns
from quantlab.validation.holdout import (
    _block_metrics,
    _validate_result_inputs,
    compute_holdout_split,
    run_holdout_report,
    run_holdout_validation,
)
from quantlab.validation.parameter_grid import (
    default_parameter_grid,
    parameter_grid_for_config,
)
from quantlab.validation.parameter_sensitivity import (
    run_parameter_sensitivity,
    sensitivity_heatmap_data,
)
from quantlab.validation.robustness import monte_carlo_permutation, run_stress_tests
from quantlab.validation.splits import (
    WalkForwardWindow,
    chronological_split,
    walk_forward_windows,
)
from quantlab.validation.walk_forward import (
    _SCORERS,
    FoldResult,
    WalkForwardResult,
    WalkForwardValidator,
    _contains_duplicate_candidates,
    _grid_combinations,
    _target_weights_for_window,
    _validate_parameter_grid,
    _with_params,
)


def _config(*, benchmark: str | None = None) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "validation_hardening",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 20,
                    "skip_period": 2,
                    "top_fraction": 0.3,
                },
            },
            "backtest": {
                "benchmark": (
                    {"symbol": benchmark, "source": "csv", "calendar": "XNYS"}
                    if benchmark is not None
                    else None
                )
            },
        }
    )


@pytest.mark.parametrize(
    ("strategy_name", "parameters", "expected_grid_parameters"),
    [
        ("buy_and_hold", {}, set()),
        (
            "time_series_momentum",
            {"lookback_period": 189, "skip_period": 21},
            {"lookback_period", "skip_period"},
        ),
        (
            "cross_sectional_momentum",
            {
                "lookback_period": 189,
                "skip_period": 21,
                "top_fraction": 0.25,
                "long_short": False,
            },
            {"lookback_period", "skip_period", "top_fraction"},
        ),
        (
            "mean_reversion",
            {
                "lookback_period": 20,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "stop_zscore": 4.0,
            },
            {"lookback_period", "entry_zscore"},
        ),
        (
            "trend_following",
            {"fast_window": 20, "slow_window": 100},
            {"fast_window", "slow_window"},
        ),
        (
            "pairs_trading",
            {
                "symbol_a": "AAA",
                "symbol_b": "BBB",
                "formation_window": 252,
                "zscore_window": 63,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "stop_zscore": 4.0,
            },
            {"formation_window", "zscore_window", "entry_zscore"},
        ),
    ],
)
def test_default_walk_forward_grid_covers_each_builtin_strategy_with_valid_combinations(
    strategy_name: str,
    parameters: dict[str, Any],
    expected_grid_parameters: set[str],
) -> None:
    portfolio = (
        {"allocator": "signal_proportional"} if strategy_name == "pairs_trading" else {}
    )
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": f"grid_{strategy_name}",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "DDD", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "EEE", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2010-01-01",
                "end_date": "2020-12-31",
            },
            "strategy": {"name": strategy_name, "parameters": parameters},
            "portfolio": portfolio,
            "backtest": {"benchmark_kind": "cash"},
        }
    )

    grid = default_parameter_grid(config)

    assert set(grid) == expected_grid_parameters
    for name, values in grid.items():
        assert values
        if name in parameters:
            assert parameters[name] in values
    for combination in _grid_combinations(grid):
        _with_params(config, combination)


def test_default_cross_sectional_long_short_grid_remains_disjoint() -> None:
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "grid_cross_sectional_long_short",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "CCC", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "DDD", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "EEE", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2010-01-01",
                "end_date": "2020-12-31",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 20,
                    "skip_period": 5,
                    "top_fraction": 0.6,
                    "bottom_fraction": 0.2,
                    "long_short": True,
                },
            },
            "backtest": {"benchmark_kind": "cash"},
        }
    )

    grid = default_parameter_grid(config)

    assert set(grid) == {
        "lookback_period",
        "skip_period",
        "top_fraction",
        "bottom_fraction",
    }
    for combination in _grid_combinations(grid):
        assert combination["top_fraction"] + combination["bottom_fraction"] <= 1.0
        _with_params(config, combination)


def test_cli_delegates_to_the_shared_default_parameter_grid() -> None:
    config = _config()

    assert _default_grid(config) == default_parameter_grid(config)


def test_yaml_parameter_grid_is_shared_by_cli_and_not_mutable_through_copy() -> None:
    validation = ValidationConfig.model_validate(
        {
            "method": "walk_forward",
            "parameter_grid": {"lookback_period": [10, 20]},
        }
    )
    config = _config().revalidated_copy(update={"validation": validation})

    grid = parameter_grid_for_config(config)

    assert _default_grid(config) == {"lookback_period": [10, 20]}
    grid["lookback_period"].append(30)
    assert config.validation.parameter_grid == {"lookback_period": [10, 20]}


def _pairs_walk_forward_config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "pairs_warmup",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
            },
            "strategy": {
                "name": "pairs_trading",
                "parameters": {
                    "symbol_a": "AAA",
                    "symbol_b": "BBB",
                    "formation_window": 252,
                    "zscore_window": 63,
                },
            },
            "portfolio": {
                "allocator": "signal_proportional",
                "rebalance_frequency": "daily",
            },
            "backtest": {"benchmark_kind": "cash"},
        }
    )


def test_walk_forward_skips_only_structurally_unwarmed_pair_combinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.validation.walk_forward as walk_forward_module

    config = _pairs_walk_forward_config()
    index = pd.bdate_range("2020-01-01", periods=752)
    window = WalkForwardWindow(
        fold=0,
        train=index[:500],
        validation=index[500:626],
        test=index[626:752],
    )
    insufficient = {"formation_window": 504, "zscore_window": 126}
    usable = {"formation_window": 252, "zscore_window": 63}
    evaluated: list[dict[str, Any]] = []

    def fake_evaluate(
        _data: pd.DataFrame,
        candidate: ExperimentConfig,
        *_bounds: pd.Timestamp,
        execution_delay: int = 0,
    ) -> pd.Series:
        evaluated.append(
            {
                "formation_window": candidate.strategy_parameters["formation_window"],
                "zscore_window": candidate.strategy_parameters["zscore_window"],
            }
        )
        # A sufficiently warmed-up strategy may legitimately choose to stay flat.
        return pd.Series([0.0, 0.0])

    monkeypatch.setattr(
        walk_forward_module, "_evaluate_fresh_from_window_start", fake_evaluate
    )

    selected = WalkForwardValidator(config)._select_on_validation(
        pd.DataFrame(),
        window,
        [insufficient, usable],
        _SCORERS["sharpe"],
        periods_per_year=252,
        risk_free_rate=0.0,
    )

    assert evaluated == [usable]
    assert selected["params"] == usable
    assert selected["score"] == 0.0


def test_walk_forward_explains_when_every_combination_is_still_in_warmup() -> None:
    config = _pairs_walk_forward_config()
    index = pd.bdate_range("2020-01-01", periods=752)
    window = WalkForwardWindow(
        fold=0,
        train=index[:500],
        validation=index[500:626],
        test=index[626:752],
    )

    with pytest.raises(InvalidConfigurationError, match="enough train/validation"):
        WalkForwardValidator(config)._select_on_validation(
            pd.DataFrame(),
            window,
            [{"formation_window": 504, "zscore_window": 126}],
            _SCORERS["sharpe"],
            periods_per_year=252,
            risk_free_rate=0.0,
        )


@pytest.mark.parametrize(
    "ratios",
    [(-0.1, 0.5, 0.6), (0.6, 0.2, 0.19), (True, 0.0, 0.0)],
)
def test_chronological_split_rejects_unsafe_ratios(
    ratios: tuple[object, object, object],
) -> None:
    index = pd.date_range("2020-01-01", periods=100)
    with pytest.raises(InvalidConfigurationError):
        chronological_split(index, *ratios)  # type: ignore[arg-type]


def test_walk_forward_windows_require_canonical_index_and_boolean() -> None:
    duplicate = pd.DatetimeIndex([pd.Timestamp("2020-01-01")] * 20)
    with pytest.raises(InvalidConfigurationError, match="duplicate"):
        walk_forward_windows(duplicate, 10, 5, 5)
    index = pd.date_range("2020-01-01", periods=20)
    with pytest.raises(InvalidConfigurationError, match="boolean"):
        walk_forward_windows(index, 10, 5, 5, expanding="false")  # type: ignore[arg-type]


def test_bootstrap_validates_arguments_before_short_data() -> None:
    returns = pd.Series([0.01])
    with pytest.raises(TypeError, match="n_iterations"):
        bootstrap_returns(returns, n_iterations=True)
    with pytest.raises(ValueError, match="initial_capital"):
        bootstrap_returns(returns, initial_capital=0.0)


def test_block_bootstrap_does_not_bridge_missing_periods() -> None:
    returns = pd.Series([0.01, np.nan, -0.01])
    with pytest.raises(ValueError, match="block bootstrapping"):
        bootstrap_returns(returns, block_size=2)


def test_random_sign_test_centres_on_risk_free_return() -> None:
    ppy = 252
    risk_free_rate = 0.05
    excess = pd.Series([0.01, -0.004, 0.006, -0.002] * 25)
    shifted = excess.add(risk_free_rate / ppy)
    zero_rate = monte_carlo_permutation(
        excess, n_iterations=200, seed=7, periods_per_year=ppy
    )
    nonzero_rate = monte_carlo_permutation(
        shifted,
        n_iterations=200,
        seed=7,
        periods_per_year=ppy,
        risk_free_rate=risk_free_rate,
    )
    assert nonzero_rate["real_sharpe"] == pytest.approx(zero_rate["real_sharpe"])
    assert nonzero_rate["p_value"] == zero_rate["p_value"]


def test_walk_forward_rejects_unknown_grid_parameter() -> None:
    with pytest.raises(InvalidConfigurationError, match="Unknown parameter"):
        WalkForwardValidator(_config()).run(
            pd.DataFrame(columns=["timestamp", "symbol"]),
            {"typo": [1]},
            10,
            5,
            5,
        )
    with pytest.raises(InvalidConfigurationError, match="candidate"):
        _grid_combinations({"lookback_period": []})


def test_walk_forward_rejects_all_nonfinite_validation_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.validation.walk_forward as walk_forward

    dates = pd.date_range("2020-01-01", periods=24)
    window = WalkForwardWindow(
        fold=0,
        train=dates[:20],
        validation=dates[20:22],
        test=dates[22:],
    )
    monkeypatch.setattr(
        walk_forward,
        "_evaluate_fresh_from_window_start",
        lambda *args, **kwargs: pd.Series([0.0, 0.0]),
    )
    validator = WalkForwardValidator(_config())
    with pytest.raises(InvalidConfigurationError, match="non-finite"):
        validator._select_on_validation(
            pd.DataFrame(),
            window,
            [{}],
            lambda returns, equity, ppy, rf: float("nan"),
            252,
            0.0,
        )


def test_parameter_sensitivity_rejects_ambiguous_axes() -> None:
    with pytest.raises(ValueError, match="different"):
        run_parameter_sensitivity(
            pd.DataFrame(),
            _config(),
            "lookback_period",
            [20],
            "lookback_period",
            [40],
        )


def test_parameter_sensitivity_keeps_invalid_combinations_visible() -> None:
    table = run_parameter_sensitivity(
        pd.DataFrame(),
        _config(),
        "lookback_period",
        [0],
        "top_fraction",
        [0.3],
    )
    assert len(table) == 1
    assert table.loc[0, "status"] == "failed"
    error = table.loc[0, "error"]
    assert isinstance(error, str)
    assert "lookback_period" in error


def test_reduced_universe_keeps_external_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.validation.robustness as robustness

    seen_reduced_symbols: set[str] = set()

    def fake_run(data: pd.DataFrame, config: ExperimentConfig, **_: object) -> object:
        if len(config.symbols) == 2:
            seen_reduced_symbols.update(data["symbol"].unique())
        return SimpleNamespace(returns=pd.Series([0.0, 0.01, -0.005]))

    monkeypatch.setattr(robustness, "run_backtest_from_config", fake_run)
    data = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "BENCH"],
            "timestamp": pd.date_range("2020-01-01", periods=4),
        }
    )
    table = run_stress_tests(data, _config(benchmark="BENCH"))
    assert "BENCH" in seen_reduced_symbols
    reduced = table.loc[table["scenario"] == "reduced universe"].iloc[0]
    assert reduced["status"] == "ok"


def test_reduced_universe_scenario_records_failure_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.validation.robustness as robustness

    def fake_run(data: pd.DataFrame, config: ExperimentConfig, **_: object) -> object:
        if len(config.symbols) == 2:
            raise BacktestError("synthetic reduced-universe failure")
        return SimpleNamespace(returns=pd.Series([0.0, 0.01, -0.005]))

    monkeypatch.setattr(robustness, "run_backtest_from_config", fake_run)
    data = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "timestamp": pd.date_range("2020-01-01", periods=3),
        }
    )
    table = run_stress_tests(data, _config())
    reduced = table.loc[table["scenario"] == "reduced universe"].iloc[0]
    assert reduced["status"] == "failed"
    assert "synthetic reduced-universe failure" in reduced["error"]


def test_run_stress_tests_rejects_bad_input_types_and_missing_symbol_column() -> None:
    config = _config()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        run_stress_tests("not a frame", config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ExperimentConfig"):
        run_stress_tests(pd.DataFrame({"symbol": ["AAA"]}), "not a config")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'symbol'"):
        run_stress_tests(pd.DataFrame({"other": [1]}), config)


def test_monte_carlo_permutation_rejects_returns_below_total_loss() -> None:
    returns = pd.Series([0.01, -1.2, 0.02] * 10)
    with pytest.raises(ValueError, match=r"below -1\.0"):
        monte_carlo_permutation(returns)


def test_monte_carlo_permutation_handles_a_single_observation() -> None:
    out = monte_carlo_permutation(pd.Series([0.01]), n_iterations=50)
    assert out == {"real_sharpe": 0.0, "p_value": 1.0, "n_iterations": 50.0}


def test_bootstrap_result_validates_shape_and_required_columns() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        BootstrapResult(samples="not a frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing columns"):
        BootstrapResult(samples=pd.DataFrame({"cagr": [0.1]}))


def test_bootstrap_rejects_returns_below_total_loss() -> None:
    returns = pd.Series([0.01, -1.5, 0.02])
    with pytest.raises(ValueError, match=r"below -1\.0"):
        bootstrap_returns(returns)


def test_bootstrap_returns_empty_samples_for_a_single_observation() -> None:
    result = bootstrap_returns(pd.Series([0.01]), n_iterations=50)
    assert result.samples.empty
    assert list(result.samples.columns) == [
        "cagr",
        "sharpe",
        "max_drawdown",
        "final_value",
    ]


def test_run_parameter_sensitivity_rejects_bad_input_types() -> None:
    config = _config()
    bad_data: Any = "not a frame"
    bad_config: Any = "not a config"
    with pytest.raises(TypeError, match="pandas DataFrame"):
        run_parameter_sensitivity(
            bad_data, config, "lookback_period", [20], "top_fraction", [0.3]
        )
    with pytest.raises(TypeError, match="ExperimentConfig"):
        run_parameter_sensitivity(
            pd.DataFrame(),
            bad_config,
            "lookback_period",
            [20],
            "top_fraction",
            [0.3],
        )


def test_parameter_axes_reject_bad_names_and_value_collections() -> None:
    config = _config()
    with pytest.raises(ValueError, match="non-empty strings"):
        run_parameter_sensitivity(
            pd.DataFrame(), config, "", [1], "top_fraction", [0.3]
        )
    with pytest.raises(ValueError, match="Unknown or unsweepable"):
        run_parameter_sensitivity(
            pd.DataFrame(), config, "not_a_param", [1], "top_fraction", [0.3]
        )
    not_a_sequence: Any = "not_a_sequence"
    with pytest.raises(TypeError, match="must be a sequence"):
        run_parameter_sensitivity(
            pd.DataFrame(),
            config,
            "lookback_period",
            not_a_sequence,
            "top_fraction",
            [0.3],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        run_parameter_sensitivity(
            pd.DataFrame(), config, "lookback_period", [], "top_fraction", [0.3]
        )
    with pytest.raises(ValueError, match="duplicates"):
        run_parameter_sensitivity(
            pd.DataFrame(), config, "lookback_period", [20, 20], "top_fraction", [0.3]
        )


def test_sensitivity_heatmap_data_validates_shape_and_rejects_bad_rows() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        sensitivity_heatmap_data("not a frame", "x", "y", "sharpe")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing columns"):
        sensitivity_heatmap_data(pd.DataFrame({"x": [1]}), "x", "y", "sharpe")

    only_failed = pd.DataFrame(
        {"x": [1], "y": [2], "sharpe": [0.1], "status": ["failed"]}
    )
    with pytest.raises(ValueError, match="no successful combinations"):
        sensitivity_heatmap_data(only_failed, "x", "y", "sharpe")

    duplicated = pd.DataFrame({"x": [1, 1], "y": [2, 2], "sharpe": [0.1, 0.2]})
    with pytest.raises(ValueError, match="duplicate parameter combinations"):
        sensitivity_heatmap_data(duplicated, "x", "y", "sharpe")

    infinite = pd.DataFrame({"x": [1, 2], "y": [2, 3], "sharpe": [0.1, float("inf")]})
    with pytest.raises(ValueError, match="infinite values"):
        sensitivity_heatmap_data(infinite, "x", "y", "sharpe")


def _holdout_panel(n: int = 300) -> pd.DataFrame:
    frames = [
        make_ohlcv(sym, geometric_series(n, mu=mu, sigma=0.012, s0=100.0, seed=seed))
        for sym, seed, mu in [("AAA", 11, 0.0006), ("BBB", 12, 0.0003)]
    ]
    return pd.concat(frames, ignore_index=True)


def _holdout_config(**validation_overrides: object) -> ExperimentConfig:
    validation: dict[str, object] = {
        "method": "holdout",
        "validation_ratio": 0.2,
        "test_ratio": 0.2,
    }
    validation.update(validation_overrides)
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "holdout_hardening",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {"allocator": "equal_weight"},
            "validation": validation,
        }
    )


def test_holdout_report_summary_table_lists_train_validation_and_test() -> None:
    data = _holdout_panel()
    config = _holdout_config()
    result = run_backtest_from_config(data, config)
    report = run_holdout_report(data, config, result)
    assert report is not None
    table = report.summary_table()
    assert list(table["Block"]) == ["Train", "Validation", "Test"]
    assert {"Start", "End", "CAGR", "Sharpe", "Max Drawdown"} <= set(table.columns)


def test_holdout_report_summary_table_omits_an_absent_validation_block() -> None:
    data = _holdout_panel()
    config = _holdout_config(validation_ratio=None)
    result = run_backtest_from_config(data, config)
    report = run_holdout_report(data, config, result)
    assert report is not None
    assert not report.has_validation_block
    table = report.summary_table()
    assert list(table["Block"]) == ["Train", "Test"]
    assert "validation_metrics" not in report.to_metadata()


def test_run_holdout_validation_returns_only_the_test_block_metrics() -> None:
    data = _holdout_panel()
    config = _holdout_config()
    result = run_backtest_from_config(data, config)
    metrics = run_holdout_validation(data, config, result)
    assert "sharpe_ratio" in metrics
    assert "n_periods" in metrics


def test_run_holdout_validation_returns_empty_without_a_configured_test_block() -> None:
    data = _holdout_panel()
    config = _holdout_config(validation_ratio=None, test_ratio=None)
    result = run_backtest_from_config(data, config)
    assert run_holdout_validation(data, config, result) == {}


def test_compute_holdout_split_rejects_bad_input_types_and_missing_symbol_column() -> (
    None
):
    config = _holdout_config()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        compute_holdout_split("not a frame", config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ExperimentConfig"):
        compute_holdout_split(pd.DataFrame({"symbol": ["AAA"]}), "not a config")  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigurationError, match="'symbol'"):
        compute_holdout_split(pd.DataFrame({"other_column": [1]}), config)


def test_compute_holdout_split_rejects_data_without_tradable_symbols() -> None:
    config = _holdout_config()
    data = make_ohlcv(
        "ZZZ", geometric_series(50, mu=0.0, sigma=0.01, s0=100.0, seed=99)
    )
    with pytest.raises(
        InvalidConfigurationError, match="none of the configured tradable"
    ):
        compute_holdout_split(data, config)


def test_block_metrics_returns_empty_for_a_single_row_range() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    fake_result: Any = SimpleNamespace(
        returns=pd.Series([0.0, 0.01, -0.01, 0.02, 0.0], index=index),
        benchmark_returns=None,
    )
    metrics, block_returns, block_equity = _block_metrics(
        fake_result, _holdout_config(), index[0], index[0]
    )
    assert metrics == {}
    assert len(block_returns) == 1
    assert block_equity.empty


def test_validate_result_inputs_catches_mismatched_artifacts() -> None:
    data = _holdout_panel()
    config = _holdout_config()
    result = run_backtest_from_config(data, config)

    with pytest.raises(TypeError, match="BacktestResult"):
        _validate_result_inputs(data, config, "not a result")  # type: ignore[arg-type]

    other_config = _holdout_config(test_ratio=0.3)
    with pytest.raises(InvalidConfigurationError, match="does not match"):
        _validate_result_inputs(data, other_config, result)

    tampered_hash = replace(
        result, metadata={**result.metadata, "data_hash": "not-a-real-hash"}
    )
    with pytest.raises(InvalidConfigurationError, match="does not match the frame"):
        _validate_result_inputs(data, config, tampered_hash)

    no_hash = replace(
        result, metadata={k: v for k, v in result.metadata.items() if k != "data_hash"}
    )
    with pytest.raises(InvalidConfigurationError, match="no valid data_hash"):
        _validate_result_inputs(data, config, no_hash)

    other_data = _holdout_panel(n=250)
    with pytest.raises(InvalidConfigurationError, match="does not match the frame"):
        _validate_result_inputs(other_data, config, result)


def test_walk_forward_validator_requires_an_experiment_config() -> None:
    with pytest.raises(TypeError, match="ExperimentConfig"):
        WalkForwardValidator("not a config")  # type: ignore[arg-type]


def test_walk_forward_run_rejects_bad_data_type_and_missing_columns() -> None:
    validator = WalkForwardValidator(_config())
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validator.run("not a frame", {}, 10, 5, 5)  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigurationError, match="missing required columns"):
        validator.run(pd.DataFrame({"foo": [1]}), {}, 10, 5, 5)


def test_walk_forward_run_reports_no_windows_without_crashing() -> None:
    # Windows require train + validation + test observations; a range that
    # is shorter than that produces zero folds instead of raising.
    validator = WalkForwardValidator(_config())
    data = pd.concat(
        [make_ohlcv(symbol, [100.0, 101.0, 102.0]) for symbol in ("AAA", "BBB", "CCC")],
        ignore_index=True,
    )
    result = validator.run(
        data, {}, train_window=10, validation_window=5, test_window=5
    )
    assert result.folds == []
    assert result.oos_returns.empty
    assert result.oos_equity.empty
    assert result.oos_metrics() == {}


@pytest.mark.parametrize("observations", [3, 25])
def test_walk_forward_rejects_a_reduced_universe_before_building_folds(
    observations: int,
) -> None:
    validator = WalkForwardValidator(_config())
    data = pd.concat(
        [
            make_ohlcv("AAA", np.linspace(100.0, 110.0, observations)),
            make_ohlcv("BBB", np.linspace(90.0, 105.0, observations)),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        InvalidConfigurationError,
        match=r"missing configured tradable symbol.*CCC",
    ):
        validator.run(data, {}, train_window=10, validation_window=5, test_window=5)


def test_validation_package_exports_fold_result() -> None:
    import quantlab.validation as validation

    assert "FoldResult" in validation.__all__
    assert validation.FoldResult is FoldResult


def test_validation_package_exports_walk_forward_robustness_functions() -> None:
    """run_walk_forward_stress_tests/run_walk_forward_parameter_sensitivity
    are public functionality used by the CLI/dashboard, so they must be
    reachable from the package root like every other public entry point
    here, not only via their own submodule."""
    import quantlab.validation as validation
    from quantlab.validation.parameter_sensitivity import (
        run_walk_forward_parameter_sensitivity,
    )
    from quantlab.validation.robustness import run_walk_forward_stress_tests

    assert "run_walk_forward_stress_tests" in validation.__all__
    assert validation.run_walk_forward_stress_tests is run_walk_forward_stress_tests
    assert "run_walk_forward_parameter_sensitivity" in validation.__all__
    assert (
        validation.run_walk_forward_parameter_sensitivity
        is run_walk_forward_parameter_sensitivity
    )


def test_walk_forward_rejects_an_unsupported_optimization_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.validation.walk_forward as walk_forward

    monkeypatch.setattr(walk_forward, "_SCORERS", {})
    validator = WalkForwardValidator(_config())
    # The scorer is resolved before any fold is evaluated, so even a range
    # too short to produce a single fold still reaches this check.
    data = pd.concat(
        [make_ohlcv(symbol, [100.0, 101.0, 102.0]) for symbol in ("AAA", "BBB", "CCC")],
        ignore_index=True,
    )
    with pytest.raises(InvalidConfigurationError, match="Unsupported"):
        validator.run(data, {}, train_window=10, validation_window=5, test_window=5)


def test_parameter_stability_summarises_finite_numeric_choices() -> None:
    folds = [
        FoldResult(
            fold=0,
            best_params={"lookback_period": 60, "signal_scaling": "binary"},
            validation_score=1.0,
            test_return=0.05,
            test_sharpe=0.5,
            test_returns=pd.Series(dtype=float),
        ),
        FoldResult(
            fold=1,
            best_params={"lookback_period": 120, "signal_scaling": "binary"},
            validation_score=1.2,
            test_return=0.03,
            test_sharpe=0.4,
            test_returns=pd.Series(dtype=float),
        ),
    ]
    stability = WalkForwardResult(folds=folds).parameter_stability()
    # Numeric parameters get a coefficient of variation; non-numeric ones do not.
    assert "lookback_period" in stability
    assert "signal_scaling" not in stability
    assert stability["lookback_period"] > 0.0
    assert WalkForwardResult(folds=[]).parameter_stability() == {}


def test_oos_metrics_returns_empty_for_a_short_stitched_curve() -> None:
    result = WalkForwardResult(oos_returns=pd.Series([0.01]))
    assert result.oos_metrics() == {}


def test_validate_parameter_grid_rejects_malformed_shapes() -> None:
    with pytest.raises(InvalidConfigurationError, match="must be a mapping"):
        _validate_parameter_grid(["not", "a", "mapping"], "trend_following")
    with pytest.raises(InvalidConfigurationError, match="non-empty strings"):
        _validate_parameter_grid({123: [1]}, "trend_following")
    with pytest.raises(InvalidConfigurationError, match="must contain a sequence"):
        _validate_parameter_grid({"fast_window": 20}, "trend_following")
    with pytest.raises(InvalidConfigurationError, match="duplicate values"):
        _validate_parameter_grid({"fast_window": [20, 20]}, "trend_following")


def test_contains_duplicate_candidates_compares_array_like_values() -> None:
    assert _contains_duplicate_candidates([np.array([1, 2]), np.array([1, 2])])
    assert not _contains_duplicate_candidates([np.array([1, 2]), np.array([1, 3])])


def test_target_weights_for_window_requires_exposed_target_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.validation.walk_forward as walk_forward

    monkeypatch.setattr(
        walk_forward,
        "run_backtest_from_config",
        lambda data, config: SimpleNamespace(target_weights=None),
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3),
            "symbol": ["AAA"] * 3,
        }
    )
    with pytest.raises(
        InvalidConfigurationError, match="did not expose target weights"
    ):
        _target_weights_for_window(
            data,
            _config(),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-03"),
        )
