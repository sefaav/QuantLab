"""Tests for experiment configuration loading and validation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from quantlab.config import (
    BacktestConfig,
    BenchmarkKind,
    ExecutionConfig,
    ExperimentConfig,
    MissingValuePolicy,
    PortfolioConfig,
    ReproducibilityConfig,
    ValidationConfig,
)
from quantlab.constants import CONFIGS_DIR
from quantlab.exceptions import InvalidConfigurationError

ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


def test_shipped_configs_exist() -> None:
    assert ALL_CONFIGS, "expected at least one shipped YAML configuration"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    """Every YAML shipped in configs/ must validate."""
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.experiment_name
    assert cfg.symbols
    assert cfg.periods_per_year > 0


def test_walk_forward_parameter_grid_is_validated_at_config_load() -> None:
    base: dict[str, Any] = {
        "experiment_name": "yaml_grid",
        "data": {
            "source": "csv",
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
            "market_calendar": "XNYS",
        },
        "strategy": {
            "name": "mean_reversion",
            "parameters": {"lookback_period": 20},
        },
        "validation": {
            "method": "walk_forward",
            "parameter_grid": {"lookback_period": [10, 40]},
        },
    }

    config = ExperimentConfig.from_dict(base)
    assert config.validation.parameter_grid == {"lookback_period": [10, 40]}

    invalid = {
        **base,
        "validation": {
            **base["validation"],
            "parameter_grid": {"lookback_period": [0]},
        },
    }
    with pytest.raises(InvalidConfigurationError, match="lookback_period"):
        ExperimentConfig.from_dict(invalid)


def test_parameter_grid_is_rejected_for_non_walk_forward_validation() -> None:
    with pytest.raises(ValueError, match="applies only"):
        ValidationConfig.model_validate(
            {"method": "holdout", "parameter_grid": {"lookback_period": [20]}}
        )


def _robustness_base_dict(**robustness_overrides: Any) -> dict[str, Any]:
    return {
        "experiment_name": "robustness_cfg",
        "data": {
            "source": "csv",
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
            "market_calendar": "XNYS",
        },
        "strategy": {
            "name": "mean_reversion",
            "parameters": {"lookback_period": 20},
        },
        "robustness": robustness_overrides,
    }


def test_robustness_config_defaults_to_everything_disabled() -> None:
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "no_robustness_block",
            "data": {
                "source": "csv",
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2022-01-01",
                "market_calendar": "XNYS",
            },
            "strategy": {
                "name": "mean_reversion",
                "parameters": {"lookback_period": 20},
            },
        }
    )
    assert config.robustness.stress_test.enabled is False
    assert config.robustness.bootstrap.enabled is False
    assert config.robustness.bootstrap.n_iterations == 1000
    assert config.robustness.bootstrap.block_size == 1
    assert config.robustness.permutation_test.enabled is False
    assert config.robustness.permutation_test.n_iterations == 1000
    assert config.robustness.sensitivity.enabled is False
    assert config.robustness.sensitivity.parameters is None


def test_robustness_bootstrap_settings_accept_overrides() -> None:
    config = ExperimentConfig.from_dict(
        _robustness_base_dict(
            bootstrap={"enabled": True, "n_iterations": 500, "block_size": 5}
        )
    )
    assert config.robustness.bootstrap.enabled is True
    assert config.robustness.bootstrap.n_iterations == 500
    assert config.robustness.bootstrap.block_size == 5


@pytest.mark.parametrize("bad_n_iterations", [0, -1])
def test_robustness_bootstrap_n_iterations_must_be_positive(
    bad_n_iterations: int,
) -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            _robustness_base_dict(bootstrap={"n_iterations": bad_n_iterations})
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"lookback_period": [10, 20]},
        {"lookback_period": [10, 20], "entry_zscore": [1.0, 2.0], "exit_zscore": [0.5]},
    ],
)
def test_robustness_sensitivity_parameters_must_have_exactly_two_keys(
    parameters: dict[str, list[Any]],
) -> None:
    with pytest.raises(InvalidConfigurationError, match="exactly 2"):
        ExperimentConfig.from_dict(
            _robustness_base_dict(sensitivity={"parameters": parameters})
        )


def test_robustness_sensitivity_parameters_rejects_empty_candidate_list() -> None:
    with pytest.raises(InvalidConfigurationError, match="at least one candidate"):
        ExperimentConfig.from_dict(
            _robustness_base_dict(
                sensitivity={
                    "parameters": {"lookback_period": [10, 20], "entry_zscore": []}
                }
            )
        )


def test_robustness_sensitivity_parameters_names_validated_against_strategy() -> None:
    with pytest.raises(InvalidConfigurationError, match="Unknown"):
        ExperimentConfig.from_dict(
            _robustness_base_dict(
                sensitivity={
                    "parameters": {
                        "lookback_period": [10, 20],
                        "not_a_real_parameter": [1, 2],
                    }
                }
            )
        )


def test_robustness_sensitivity_parameters_rejects_boolean_parameter() -> None:
    """long_only is a structural switch (default value True/False) — sweeping
    it changes which other parameters are even meaningful, so sensitivity
    must reject it the same way it rejects an unknown parameter name."""
    with pytest.raises(InvalidConfigurationError, match="Unknown or unsweepable"):
        ExperimentConfig.from_dict(
            _robustness_base_dict(
                sensitivity={
                    "parameters": {
                        "lookback_period": [10, 20],
                        "long_only": [True, False],
                    }
                }
            )
        )


def test_robustness_sensitivity_parameters_accepts_two_valid_keys() -> None:
    config = ExperimentConfig.from_dict(
        _robustness_base_dict(
            sensitivity={
                "enabled": True,
                "parameters": {
                    "lookback_period": [10, 20],
                    "entry_zscore": [1.0, 2.0],
                },
            }
        )
    )
    assert config.robustness.sensitivity.enabled is True
    assert config.robustness.sensitivity.parameters == {
        "lookback_period": [10, 20],
        "entry_zscore": [1.0, 2.0],
    }


def test_flat_accessors(sample_config: ExperimentConfig) -> None:
    """The flat view must mirror the nested config."""
    assert sample_config.data_source == "csv"
    assert sample_config.symbols == ["AAA", "BBB", "CCC"]
    assert sample_config.commission_bps == 2.0
    assert sample_config.spread_bps == 3.0
    assert sample_config.benchmark_symbol == "AAA"
    assert sample_config.benchmark_kind is BenchmarkKind.SYMBOL
    assert sample_config.benchmark_label == "AAA"
    assert sample_config.random_seed == 42
    assert sample_config.start_date == date(2020, 1, 1)


@pytest.mark.parametrize(
    ("kind", "expected_label"),
    [
        ("equal_weight", "Equal weight"),
        ("first_asset", "AAA"),
        ("cash", "Cash"),
    ],
)
def test_alternative_benchmark_kinds(kind: str, expected_label: str) -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "benchmark_kind",
            "data": {
                "source": "csv",
                "symbols": ["AAA", "BBB"],
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "market_calendar": "XNYS",
            },
            "strategy": {"name": "buy_and_hold"},
            "backtest": {"benchmark_kind": kind},
        }
    )

    assert str(cfg.benchmark_kind) == kind
    assert cfg.benchmark_label == expected_label


def test_non_symbol_benchmark_rejects_benchmark_symbol() -> None:
    with pytest.raises(InvalidConfigurationError, match="benchmark_symbol"):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "benchmark_kind",
                "data": {
                    "source": "csv",
                    "symbols": ["AAA"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-02-01",
                    "market_calendar": "XNYS",
                },
                "strategy": {"name": "buy_and_hold"},
                "backtest": {
                    "benchmark_kind": "cash",
                    "benchmark_symbol": "SPY",
                },
            }
        )


def test_symbols_are_normalised() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "symbols": [" spy ", "spy", "QQQ", "qqq"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    # Deduped, uppercased, order preserved.
    assert cfg.symbols == ["SPY", "QQQ"]


def test_missing_value_policy_enum() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "symbols": ["SPY"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "missing_value_policy": "forward_fill",
                "forward_fill_limit": 2,
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert cfg.data.missing_value_policy is MissingValuePolicy.FORWARD_FILL
    assert cfg.data.forward_fill_limit == 2


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_forward_fill_limit_is_rejected(value: object) -> None:
    with pytest.raises(InvalidConfigurationError, match="forward_fill_limit"):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["SPY"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "missing_value_policy": "forward_fill",
                    "forward_fill_limit": value,
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_non_default_forward_fill_limit_requires_forward_fill_policy() -> None:
    with pytest.raises(InvalidConfigurationError, match="forward_fill_limit"):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["SPY"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "missing_value_policy": "drop",
                    "forward_fill_limit": 2,
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["SPY"],
                    "start_date": "2021-01-01",
                    "end_date": "2020-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_same_day_intraday_experiment_is_allowed() -> None:
    """Inclusive date bounds may describe several intraday bars on one day."""
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "same_day",
            "data": {
                "source": "csv",
                "symbols": ["SPY"],
                "start_date": "2024-01-02",
                "end_date": "2024-01-02",
                "frequency": "1h",
                "market_calendar": "XNYS",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert cfg.start_date == cfg.end_date


def test_same_day_non_intraday_experiment_is_rejected() -> None:
    with pytest.raises(InvalidConfigurationError, match="only for intraday"):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "same_day_daily",
                "data": {
                    "symbols": ["SPY"],
                    "start_date": "2024-01-02",
                    "end_date": "2024-01-02",
                    "frequency": "1d",
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_unknown_key_is_rejected() -> None:
    """extra='forbid' catches typos in config keys."""
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["SPY"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "typo_field": 123,
                },
                "strategy": {"name": "buy_and_hold"},
            }
        )


def test_missing_file_raises() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_yaml("configs/does_not_exist.yaml")


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "experiment_name: first\nexperiment_name: second\n", encoding="utf-8"
    )
    with pytest.raises(InvalidConfigurationError, match="duplicate key"):
        ExperimentConfig.from_yaml(path)


def test_unreadable_yaml_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.yaml"
    path.write_text("experiment_name: x\n", encoding="utf-8")

    def deny_read(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("denied for test")

    monkeypatch.setattr(Path, "read_text", deny_read)
    with pytest.raises(InvalidConfigurationError, match="Could not read YAML config"):
        ExperimentConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: BacktestConfig(initial_capital=True), "initial_capital"),
        (lambda: ExecutionConfig(commission_bps=True), "commission_bps"),
        (lambda: PortfolioConfig(maximum_weight=True), "maximum_weight"),
        (lambda: ValidationConfig(train_window=True), "train_window"),
        (lambda: ReproducibilityConfig(random_seed=True), "random_seed"),
        (
            lambda: ReproducibilityConfig(random_seed=cast(Any, np.bool_(True))),
            "random_seed",
        ),
    ],
)
def test_boolean_is_rejected_for_numeric_fields(
    factory: Callable[[], object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        factory()


def test_negative_random_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="random_seed"):
        ReproducibilityConfig(random_seed=-1)


def test_custom_rebalancing_is_rejected_at_config_load() -> None:
    with pytest.raises(ValueError, match=r"custom.*not implemented"):
        PortfolioConfig(rebalance_frequency=cast(Any, "custom"))


def test_holdout_validation_ratio_requires_test_ratio() -> None:
    with pytest.raises(ValueError, match="validation_ratio has no effect"):
        ValidationConfig(method=cast(Any, "holdout"), validation_ratio=0.2)


def test_walk_forward_rejects_holdout_ratios() -> None:
    with pytest.raises(ValueError, match="apply only to method 'holdout'"):
        ValidationConfig(method=cast(Any, "walk_forward"), test_ratio=0.2)


def test_periods_per_year_from_frequency() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "symbols": ["BTCUSDT"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "frequency": "1d",
            },
            "strategy": {"name": "buy_and_hold"},
            "backtest": {"periods_per_year": 365},
        }
    )
    assert cfg.periods_per_year == 365


def test_yaml_roundtrip(sample_config: ExperimentConfig, tmp_path: Path) -> None:
    out = sample_config.to_yaml(tmp_path / "cfg.yaml")
    reloaded = ExperimentConfig.from_yaml(out)
    assert reloaded == sample_config
