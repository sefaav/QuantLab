"""Regression tests for strategies behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from tests.regression_helpers import (
    _rf_test_setup,
    _try_strategy,
)

from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError


def test_unknown_strategy_name_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "not_a_real_strategy"},
            }
        )


def test_unknown_strategy_parameter_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "mean_reversion",
                    "parameters": {"lookback_perido": 20},
                },
            }
        )


def test_known_strategy_parameters_still_accepted() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {
                "name": "mean_reversion",
                "parameters": {"lookback_period": 20, "entry_zscore": 2.0},
            },
        }
    )
    assert cfg.strategy.parameters == {"lookback_period": 20, "entry_zscore": 2.0}


def test_var_keyword_catch_all_does_not_admit_bogus_parameters() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold", "parameters": {"kwargs": 123}},
            }
        )


def test_wrong_type_strategy_parameter_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "mean_reversion",
                    "parameters": {"lookback_period": "twenty"},
                },
            }
        )


def test_pairs_trading_missing_required_parameter_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                        {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "pairs_trading",
                    "parameters": {"symbol_a": "AAA"},
                },
            }
        )


def test_mean_reversion_zero_lookback_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "mean_reversion",
                    "parameters": {"lookback_period": 0},
                },
            }
        )


def test_cross_sectional_momentum_out_of_range_top_fraction_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                        {"symbol": "B", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "cross_sectional_momentum",
                    "parameters": {"top_fraction": 1.5},
                },
            }
        )


def test_time_series_momentum_unknown_signal_scaling_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "instruments": [
                        {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                    ],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {
                    "name": "time_series_momentum",
                    "parameters": {"signal_scaling": "typo"},
                },
            }
        )


def test_int_accepted_for_float_strategy_parameter() -> None:
    """`entry_zscore: 2` (an int) is semantically identical to `2.0` and
    must not be rejected just because the annotation says `float`."""
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "instruments": [
                    {"symbol": "A", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {
                "name": "mean_reversion",
                "parameters": {"entry_zscore": 2, "exit_zscore": 0.5},
            },
        }
    )
    assert cfg.strategy.parameters["entry_zscore"] == 2


def test_any_annotated_custom_strategy_parameter_does_not_crash() -> None:

    from quantlab.strategies.base import (
        BaseStrategy,
        register_strategy,
        validate_strategy_parameters,
    )

    @register_strategy("any_param_test")
    class _AnyParamStrategy(BaseStrategy):
        def __init__(self, threshold: Any = None) -> None:
            self.threshold = threshold

        def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame | None = None
        ) -> pd.DataFrame:
            return pd.DataFrame()

    # Must not raise -- an `Any`-annotated parameter is skipped, not
    # type-checked (and certainly not a crash).
    validate_strategy_parameters("any_param_test", {"threshold": 5})
    validate_strategy_parameters("any_param_test", {"threshold": "text"})


def test_time_series_momentum_zero_lookback_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("time_series_momentum", {"lookback_period": 0})


def test_time_series_momentum_skip_equal_lookback_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "time_series_momentum", {"lookback_period": 20, "skip_period": 20}
        )


def test_cross_sectional_momentum_overlapping_long_short_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "cross_sectional_momentum",
            {"top_fraction": 0.75, "bottom_fraction": 0.75, "long_short": True},
        )


def test_cross_sectional_momentum_unused_bottom_fraction_not_over_strict() -> None:
    config = _try_strategy(
        "cross_sectional_momentum",
        {"top_fraction": 0.75, "bottom_fraction": 0.75, "long_short": False},
    )
    assert config.strategy_name == "cross_sectional_momentum"


def test_trend_following_negative_windows_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("trend_following", {"fast_window": -5, "slow_window": -1})


def test_trend_following_negative_target_volatility_rejected() -> None:
    """Risk sizing is a portfolio concern, not a trend-strategy parameter."""
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("trend_following", {"target_volatility": -0.2})


def test_pairs_trading_adf_pvalue_out_of_range_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {"symbol_a": "AAA", "symbol_b": "BBB", "adf_pvalue_threshold": 1.5},
        )


def test_time_series_momentum_zero_volatility_window_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "time_series_momentum",
            {"signal_scaling": "volatility_adjusted", "volatility_window": 0},
        )


def test_time_series_momentum_zero_periods_per_year_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("time_series_momentum", {"periods_per_year": 0})


def test_cross_sectional_momentum_unknown_signal_scaling_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("cross_sectional_momentum", {"signal_scaling": "totally_bogus"})


def test_trend_following_zero_volatility_window_rejected() -> None:
    """`volatility_window` is not a `trend_following` parameter — sizing
    config belongs to the portfolio allocator, not the strategy."""
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("trend_following", {"volatility_window": 0})


def test_trend_following_zero_periods_per_year_rejected() -> None:
    """Annualisation belongs to the configured portfolio allocator."""
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("trend_following", {"periods_per_year": 0})


def test_mean_reversion_stop_below_entry_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "mean_reversion",
            {"entry_zscore": 2.0, "exit_zscore": 0.5, "stop_zscore": 1.0},
        )


def test_pairs_trading_entry_below_exit_rejected() -> None:
    """`pairs_trading` walks the same entry/exit/stop state machine as
    `mean_reversion` and must enforce the same z-score ordering."""
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {
                "symbol_a": "AAA",
                "symbol_b": "BBB",
                "entry_zscore": 0.5,
                "exit_zscore": 2.0,
            },
        )


def test_pairs_trading_stop_below_entry_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {
                "symbol_a": "AAA",
                "symbol_b": "BBB",
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "stop_zscore": 1.0,
            },
        )


def test_pairs_trading_zero_formation_window_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {"symbol_a": "AAA", "symbol_b": "BBB", "formation_window": 0},
        )


def test_pairs_trading_zero_zscore_window_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {"symbol_a": "AAA", "symbol_b": "BBB", "zscore_window": 0},
        )


def test_mean_reversion_negative_zscore_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("mean_reversion", {"entry_zscore": -1.0, "exit_zscore": -2.0})


def test_pairs_trading_negative_zscore_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {
                "symbol_a": "AAA",
                "symbol_b": "BBB",
                "entry_zscore": -1.0,
                "exit_zscore": -2.0,
            },
        )


def test_pairs_trading_formation_window_below_adf_floor_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "pairs_trading",
            {"symbol_a": "AAA", "symbol_b": "BBB", "formation_window": 2},
        )


def test_time_series_momentum_continuous_scaling_small_lookback_works() -> None:
    from quantlab.strategies.momentum import TimeSeriesMomentumStrategy

    strategy = TimeSeriesMomentumStrategy(
        lookback_period=10, skip_period=0, signal_scaling="continuous"
    )
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = pd.Series(np.linspace(100.0, 110.0, 100), index=idx)
    data = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "A",
            "close": prices,
            "adjusted_close": prices,
            "open": prices,
            "high": prices,
            "low": prices,
            "volume": 1000.0,
        }
    )
    signals = strategy.generate_signals(data)
    assert signals.shape == (100, 1)


def test_cross_sectional_momentum_non_binary_signal_scaling_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy("cross_sectional_momentum", {"signal_scaling": "continuous"})


def test_time_series_momentum_continuous_scaling_needs_lookback_ge_2() -> None:
    with pytest.raises(InvalidConfigurationError):
        _try_strategy(
            "time_series_momentum",
            {"lookback_period": 1, "skip_period": 0, "signal_scaling": "continuous"},
        )


def test_time_series_momentum_continuous_scaling_lookback_2_is_valid() -> None:
    config = _try_strategy(
        "time_series_momentum",
        {"lookback_period": 2, "skip_period": 0, "signal_scaling": "continuous"},
        portfolio={"allocator": "signal_proportional"},
    )
    assert config.strategy.parameters["lookback_period"] == 2


def test_cross_sectional_momentum_zero_top_fraction_accepted_for_short_only() -> None:
    config = _try_strategy(
        "cross_sectional_momentum",
        {"top_fraction": 0.0, "bottom_fraction": 0.25, "long_short": True},
    )
    assert config.strategy.parameters["top_fraction"] == 0.0


def test_mean_reversion_zero_entry_zscore_rejected_with_clear_message() -> None:
    with pytest.raises(InvalidConfigurationError, match=r"entry_zscore must be > 0\.0"):
        _try_strategy("mean_reversion", {"entry_zscore": 0.0, "exit_zscore": -0.5})


def test_pairs_trading_zero_entry_zscore_rejected_with_clear_message() -> None:
    """See `test_mean_reversion_zero_entry_zscore_rejected_with_clear_message`
    — `pairs_trading` walks the same z-score state machine and must give
    the same direct error naming `entry_zscore`."""
    with pytest.raises(InvalidConfigurationError, match=r"entry_zscore must be > 0\.0"):
        _try_strategy(
            "pairs_trading",
            {"symbol_a": "AAA", "symbol_b": "BBB", "entry_zscore": 0.0},
        )


def test_config_rejects_a_numpy_array_strategy_parameter() -> None:
    import numpy as np

    from quantlab.config import StrategyConfig

    with pytest.raises(ValidationError, match="numpy array"):
        StrategyConfig(name="buy_and_hold", parameters={"values": np.array([1.0, 2.0])})


def test_register_strategy_rejects_a_silent_duplicate_name() -> None:
    from quantlab.exceptions import StrategyError
    from quantlab.strategies.base import BaseStrategy, register_strategy

    name = "test_duplicate_registration_target"

    @register_strategy(name)
    class _First(BaseStrategy):
        def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame | None = None
        ) -> pd.DataFrame:
            raise NotImplementedError

    try:
        with pytest.raises(StrategyError, match="already registered"):

            @register_strategy(name)
            class _Second(BaseStrategy):
                def generate_signals(
                    self, data: pd.DataFrame, features: pd.DataFrame | None = None
                ) -> pd.DataFrame:
                    raise NotImplementedError

        # replace=True is the explicit opt-in and must succeed.
        @register_strategy(name, replace=True)
        class _Third(BaseStrategy):
            def generate_signals(
                self, data: pd.DataFrame, features: pd.DataFrame | None = None
            ) -> pd.DataFrame:
                raise NotImplementedError

    finally:
        from quantlab.strategies import base as base_mod

        base_mod._REGISTRY.pop(name, None)


def test_strategy_periods_per_year_injected_from_config() -> None:
    """Strategies that annualise a signal inherit the experiment frequency."""
    from quantlab.backtesting.runner import build_strategy_from_config

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "instruments": [
                    {"symbol": "BTCUSDT", "source": "binance", "calendar": "24/7"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
                "frequency": "1d",
            },
            "strategy": {
                "name": "time_series_momentum",
                "parameters": {"lookback_period": 20, "skip_period": 1},
            },
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    assert cfg.periods_per_year == 365
    strategy = build_strategy_from_config(cfg)
    assert strategy.periods_per_year == 365  # type: ignore[attr-defined]


def test_explicit_strategy_periods_per_year_still_wins() -> None:
    """An explicit strategy annualisation remains a deliberate override."""
    from quantlab.backtesting.runner import build_strategy_from_config

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "test",
            "data": {
                "instruments": [
                    {"symbol": "BTCUSDT", "source": "binance", "calendar": "24/7"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-06-01",
                "frequency": "1d",
            },
            "strategy": {
                "name": "time_series_momentum",
                "parameters": {
                    "lookback_period": 20,
                    "skip_period": 1,
                    "periods_per_year": 100,
                },
            },
            "portfolio": {"allocator": "equal_weight"},
            "execution": {},
            "backtest": {},
            "validation": {"method": "holdout"},
            "reproducibility": {"random_seed": 42},
        }
    )
    strategy = build_strategy_from_config(cfg)
    assert strategy.periods_per_year == 100  # type: ignore[attr-defined]


def test_strategy_without_periods_per_year_param_is_unaffected() -> None:
    """A strategy whose constructor doesn't accept `periods_per_year` (e.g.
    buy_and_hold) must build exactly as before — the injection only applies
    to strategies that declare the parameter."""
    from quantlab.backtesting.runner import build_strategy_from_config

    _, cfg = _rf_test_setup()
    strategy = build_strategy_from_config(cfg)
    assert not hasattr(strategy, "periods_per_year")


def test_min_usable_date_none_when_a_feature_is_always_nan() -> None:
    from quantlab.features.pipeline import FeaturePipeline

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=idx)

    pipe = FeaturePipeline()
    pipe.add("good", lambda d: d.rolling(3).mean())
    pipe.add("all_nan", lambda d: d * np.nan)
    pipe.fit(data)
    assert pipe.min_usable_date is None


def test_min_usable_date_still_works_when_all_features_are_valid() -> None:
    """Sanity check for the all-NaN handling above: the ordinary
    (all-features-valid) case must still resolve to the longer window's
    first-valid date."""
    from quantlab.features.pipeline import FeaturePipeline

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=idx)

    pipe = FeaturePipeline()
    pipe.add("short", lambda d: d.rolling(2).mean())
    pipe.add("long", lambda d: d.rolling(5).mean())
    pipe.fit(data)
    assert pipe.min_usable_date == idx[4]  # the longer window's first-valid date


def test_duplicate_feature_name_rejected() -> None:
    """Two features registered under the same name must be rejected — a
    silent overwrite would mean only one of the two columns actually
    appears in `transform()`'s output."""
    from quantlab.features.pipeline import FeaturePipeline

    pipe = FeaturePipeline()
    pipe.add("sma", lambda d: d)
    with pytest.raises(ValueError, match="already registered"):
        pipe.add("sma", lambda d: d * 2)


def test_config_rejects_nan_and_infinity_in_strategy_parameters() -> None:
    from quantlab.config import StrategyConfig

    with pytest.raises(ValidationError, match="NaN/Infinity"):
        StrategyConfig(name="x", parameters={"threshold": float("nan")})
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        StrategyConfig(name="x", parameters={"nested": {"deep": float("inf")}})
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        StrategyConfig(name="x", parameters={"values": [1.0, float("nan")]})
    # Ordinary, finite parameters remain unaffected.
    ok = StrategyConfig(name="x", parameters={"lookback": 20, "threshold": 0.5})
    assert ok.parameters == {"lookback": 20, "threshold": 0.5}
