"""Regression tests for strategy API, validation and pairs methodology."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError, StrategyError
from quantlab.strategies import (
    BaseStrategy,
    BuyAndHoldStrategy,
    PairsTradingStrategy,
    TrendFollowingStrategy,
    adf_pvalue,
    build_strategy,
    register_strategy,
    validate_strategy_parameters,
)
from quantlab.strategies.base import _unwrap_simple_type, strategy_parameter_names
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.momentum import (
    CrossSectionalMomentumStrategy,
    TimeSeriesMomentumStrategy,
)
from quantlab.strategies.pairs_trading import (
    _ols_coefficients,
    _ols_slope,
    _walk_pairs_positions,
    rolling_hedge_parameters,
)


def _config(
    strategy: str,
    parameters: Mapping[str, Any],
    *,
    symbols: list[str] | None = None,
    portfolio: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "strategy_hardening",
            "data": {
                "instruments": [
                    {"symbol": symbol, "source": "csv", "calendar": "XNYS"}
                    for symbol in (symbols or ["AAA", "BBB"])
                ],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "strategy": {"name": strategy, "parameters": dict(parameters)},
            "portfolio": dict(portfolio or {}),
        }
    )


def test_registry_rejects_invalid_names_classes_and_replace_flag() -> None:
    with pytest.raises(StrategyError, match="non-empty string"):
        register_strategy(cast(Any, "  "))
    with pytest.raises(StrategyError, match="boolean"):
        register_strategy("bad_replace", replace=cast(Any, 1))
    with pytest.raises(StrategyError, match="inherit BaseStrategy"):
        register_strategy("not_a_strategy")(cast(Any, object))


def test_build_strategy_requires_a_parameter_mapping() -> None:
    with pytest.raises(StrategyError, match="mapping"):
        build_strategy("buy_and_hold", cast(Any, []))
    with pytest.raises(StrategyError, match="names must be strings"):
        build_strategy("buy_and_hold", cast(Any, {1: "value"}))


def test_config_validation_does_not_instantiate_custom_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.base as strategy_base

    monkeypatch.setattr(strategy_base, "_REGISTRY", dict(strategy_base._REGISTRY))
    constructions = 0

    @register_strategy("side_effect_probe")
    class SideEffectProbe(BaseStrategy):
        def __init__(self, lookback: int = 5) -> None:
            nonlocal constructions
            constructions += 1
            self.lookback = lookback

        def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame | None = None
        ) -> pd.DataFrame:
            prices = self._prices(data)
            return self._validate_signals(prices * 0.0, prices)

    validate_strategy_parameters("side_effect_probe", {"lookback": 10})
    assert constructions == 0
    built = build_strategy("side_effect_probe", {"lookback": 10})
    assert isinstance(built, SideEffectProbe)
    assert built.lookback == 10
    assert constructions == 1


@pytest.mark.parametrize(
    "signals",
    [
        pd.DataFrame({"AAA": [0.0, np.inf]}),
        pd.DataFrame({"AAA": [0.0, 1.01]}),
        pd.DataFrame({"AAA": [0.0, 0.5]}, index=[0, 0]),
    ],
)
def test_signal_contract_rejects_invalid_values_and_axes(
    signals: pd.DataFrame,
) -> None:
    with pytest.raises(StrategyError):
        BaseStrategy._validate_signals(signals)


def test_signal_contract_requires_exact_reference_axes() -> None:
    reference = pd.DataFrame({"AAA": [100.0, 101.0]}, index=[0, 1])
    wrong_columns = pd.DataFrame({"BBB": [0.0, 1.0]}, index=[0, 1])
    with pytest.raises(StrategyError, match="exactly match"):
        BaseStrategy._validate_signals(wrong_columns, reference)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.inf])
def test_direct_strategy_data_requires_finite_positive_prices(
    bad_price: float,
) -> None:
    data = make_ohlcv("AAA", [100.0, 101.0])
    data.loc[data.index[-1], "adjusted_close"] = bad_price
    with pytest.raises(StrategyError):
        BuyAndHoldStrategy().generate_signals(data)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MeanReversionStrategy(entry_zscore=np.nan), "finite"),
        (lambda: MeanReversionStrategy(long_only=cast(Any, "false")), "boolean"),
        (
            lambda: TimeSeriesMomentumStrategy(lookback_period=cast(Any, True)),
            "integer",
        ),
        (
            lambda: CrossSectionalMomentumStrategy(long_short=cast(Any, 1)),
            "boolean",
        ),
        (lambda: TrendFollowingStrategy(long_only=cast(Any, 0)), "boolean"),
        (
            lambda: PairsTradingStrategy(
                "AAA", "BBB", dynamic_hedge_ratio=cast(Any, "true")
            ),
            "boolean",
        ),
    ],
)
def test_direct_constructors_reject_ambiguous_parameter_types(
    factory: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_strategy_parameters_are_immutable_and_defensively_copied() -> None:
    strategy = MeanReversionStrategy(lookback_period=20)
    with pytest.raises(AttributeError, match="immutable"):
        strategy.lookback_period = 30
    parameters = strategy.parameters()
    parameters["lookback_period"] = 99
    assert strategy.lookback_period == 20


def test_adf_inconclusive_is_explicit_and_never_passes_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.pairs_trading as pairs_module

    assert adf_pvalue(pd.Series(np.ones(25))) is None
    with pytest.raises(ValueError, match="below 1"):
        PairsTradingStrategy("AAA", "BBB", adf_pvalue_threshold=1.0)

    monkeypatch.setattr(pairs_module, "adf_pvalue", lambda series: None)
    strategy = PairsTradingStrategy("AAA", "BBB", formation_window=20, zscore_window=2)
    index = pd.date_range("2020-01-01", periods=25)
    a = pd.Series(np.linspace(100.0, 120.0, 25), index=index)
    b = pd.Series(np.linspace(50.0, 60.0, 25), index=index)
    assert not strategy._stationarity_gate(a, b).any()


def test_pairs_adf_uses_the_full_trailing_formation_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.pairs_trading as pairs_module

    observed_lengths: list[int] = []

    def _capture(series: pd.Series) -> float:
        observed_lengths.append(len(series))
        return 0.01

    monkeypatch.setattr(pairs_module, "adf_pvalue", _capture)
    strategy = PairsTradingStrategy("AAA", "BBB", formation_window=20, zscore_window=2)
    index = pd.date_range("2020-01-01", periods=25)
    b = pd.Series(np.linspace(50.0, 60.0, 25), index=index)
    a = 5.0 + 1.5 * b + pd.Series(np.sin(np.arange(25)), index=index)
    strategy._stationarity_gate(a, b)
    assert observed_lengths
    assert set(observed_lengths) == {20}


def test_static_pairs_adf_tests_the_same_fixed_relationship(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.pairs_trading as pairs_module

    original_ols = pairs_module._ols_coefficients
    regression_calls = 0

    def _counted_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        nonlocal regression_calls
        regression_calls += 1
        return original_ols(x, y)

    monkeypatch.setattr(pairs_module, "_ols_coefficients", _counted_ols)
    monkeypatch.setattr(pairs_module, "adf_pvalue", lambda series: 0.01)
    strategy = PairsTradingStrategy(
        "AAA",
        "BBB",
        formation_window=20,
        zscore_window=2,
        dynamic_hedge_ratio=False,
    )
    index = pd.date_range("2020-01-01", periods=25)
    b = pd.Series(np.linspace(50.0, 60.0, 25), index=index)
    a = 5.0 + 1.5 * b + pd.Series(np.sin(np.arange(25)), index=index)
    strategy._stationarity_gate(a, b)
    assert regression_calls == 1


def test_pairs_signals_convert_share_beta_to_dollar_neutral_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.pairs_trading as pairs_module

    def _parameters(
        a: pd.Series, b: pd.Series, window: int, dynamic: bool
    ) -> tuple[pd.Series, pd.Series]:
        return (
            pd.Series(0.0, index=a.index),
            pd.Series(2.0, index=a.index),
        )

    monkeypatch.setattr(pairs_module, "rolling_hedge_parameters", _parameters)
    monkeypatch.setattr(
        pairs_module,
        "rolling_zscore",
        lambda spread, window: pd.Series(-3.0, index=spread.index),
    )
    monkeypatch.setattr(
        PairsTradingStrategy,
        "_stationarity_gate",
        lambda self, a, b: np.ones(len(a), dtype=bool),
    )
    data = pd.concat(
        [make_ohlcv("AAA", np.full(25, 100.0)), make_ohlcv("BBB", np.full(25, 50.0))],
        ignore_index=True,
    )
    strategy = PairsTradingStrategy("AAA", "BBB", formation_window=20, zscore_window=2)
    last = strategy.generate_signals(data).iloc[-1]
    assert last["AAA"] == pytest.approx(1.0)
    assert last["BBB"] == pytest.approx(-1.0)


def test_pairs_undefined_zscore_closes_instead_of_resurrecting_position() -> None:
    positions = _walk_pairs_positions(
        np.array([-3.0, -3.0, np.nan, -3.0]),
        np.array([True, True, True, False]),
        entry=2.0,
        exit_=0.5,
        stop=4.0,
    )
    np.testing.assert_array_equal(positions, [1.0, 1.0, 0.0, 0.0])


def test_pairs_configuration_preserves_both_hedge_legs() -> None:
    parameters = {"symbol_a": "AAA", "symbol_b": "BBB"}
    good = _config(
        "pairs_trading",
        parameters,
        portfolio={
            "allocator": "signal_proportional",
            "maximum_weight": 1.0,
            "target_minimum_weight": 0.0,
        },
    )
    assert good.portfolio.allocator == "signal_proportional"

    with pytest.raises(InvalidConfigurationError, match="signal_proportional"):
        _config("pairs_trading", parameters)
    with pytest.raises(InvalidConfigurationError, match="maximum_weight"):
        _config(
            "pairs_trading",
            parameters,
            portfolio={"allocator": "signal_proportional", "maximum_weight": 0.6},
        )
    with pytest.raises(InvalidConfigurationError, match=">= 2"):
        _config(
            "pairs_trading",
            parameters,
            portfolio={
                "allocator": "signal_proportional",
                "target_maximum_positions": 1,
            },
        )


def test_signal_magnitude_scaling_rejects_incompatible_allocators() -> None:
    with pytest.raises(InvalidConfigurationError, match="preserves signal magnitude"):
        _config(
            "time_series_momentum",
            {
                "lookback_period": 20,
                "skip_period": 1,
                "signal_scaling": "continuous",
            },
            portfolio={"allocator": "equal_weight"},
        )
    with pytest.raises(InvalidConfigurationError, match="again"):
        _config(
            "time_series_momentum",
            {
                "lookback_period": 20,
                "skip_period": 1,
                "signal_scaling": "volatility_adjusted",
            },
            portfolio={"allocator": "inverse_volatility"},
        )


def test_cross_sectional_config_rejects_overlapping_single_asset_baskets() -> None:
    with pytest.raises(InvalidConfigurationError, match="distinct symbols"):
        _config(
            "cross_sectional_momentum",
            {
                "lookback_period": 20,
                "skip_period": 1,
                "top_fraction": 0.25,
                "bottom_fraction": 0.25,
                "long_short": True,
            },
            symbols=["AAA"],
        )


def test_trend_strategy_contains_direction_parameters_only() -> None:
    strategy = TrendFollowingStrategy(fast_window=10, slow_window=30)
    assert strategy.parameters() == {
        "fast_window": 10,
        "slow_window": 30,
        "long_only": True,
    }


def test_build_strategy_wraps_constructor_validation_errors() -> None:
    with pytest.raises(StrategyError, match="Invalid parameters"):
        build_strategy("trend_following", {"fast_window": "not an int"})


def test_strategy_parameter_names_rejects_unknown_strategy() -> None:
    with pytest.raises(StrategyError, match="Unknown strategy"):
        strategy_parameter_names("does_not_exist")


def test_validate_strategy_parameters_rejects_unknown_strategy_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        validate_strategy_parameters("does_not_exist", {})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_strategy_parameters(
            "trend_following", cast(Any, ["not", "a", "mapping"])
        )
    with pytest.raises(ValueError, match="must be strings"):
        validate_strategy_parameters("trend_following", cast(Any, {1: 2}))


def test_validate_strategy_parameters_reports_unresolvable_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.strategies.base as strategy_base

    monkeypatch.setattr(strategy_base, "_REGISTRY", dict(strategy_base._REGISTRY))

    @register_strategy("broken_annotation_probe")
    class BrokenAnnotationProbe(BaseStrategy):
        def __init__(
            self,
            value: NotARealType = 1,  # type: ignore[name-defined] # noqa: F821
        ) -> None:
            self.value = value
            self._freeze_parameters()

        def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame | None = None
        ) -> pd.DataFrame:
            raise NotImplementedError

    with pytest.raises(ValueError, match="Cannot resolve constructor annotations"):
        validate_strategy_parameters("broken_annotation_probe", {"value": 1})


def test_unwrap_simple_type_returns_none_for_ambiguous_annotations() -> None:
    assert _unwrap_simple_type(int | str) is None
    assert _unwrap_simple_type(dict[str, int]) is None


def test_prices_rejects_data_with_no_rows() -> None:
    empty = make_ohlcv("AAA", [100.0, 101.0]).iloc[0:0]
    with pytest.raises(StrategyError, match="at least one date and symbol"):
        BaseStrategy._prices(empty)


def test_validate_signals_rejects_non_dataframe_input() -> None:
    with pytest.raises(StrategyError, match="pandas DataFrame"):
        BaseStrategy._validate_signals(cast(Any, [1, 2, 3]))


def test_strategy_repr_lists_its_parameters() -> None:
    strategy = TrendFollowingStrategy(fast_window=10, slow_window=30, long_only=False)
    text = repr(strategy)
    assert text.startswith("TrendFollowingStrategy(")
    assert "fast_window=10" in text
    assert "slow_window=30" in text
    assert "long_only=False" in text


def test_time_series_momentum_volatility_adjusted_scaling_respects_bounds() -> None:
    prices = 100.0 + np.cumsum(np.sin(np.arange(120) / 5.0))
    data = make_ohlcv("AAA", prices)
    strategy = TimeSeriesMomentumStrategy(
        lookback_period=30,
        skip_period=1,
        long_only=False,
        signal_scaling="volatility_adjusted",
        volatility_window=10,
        periods_per_year=252,
    )
    signals = strategy.generate_signals(data)
    assert signals.shape == (120, 1)
    assert signals["AAA"].abs().le(1.0).all()


def test_cross_sectional_momentum_rejects_skip_not_smaller_than_lookback() -> None:
    with pytest.raises(ValueError, match="skip_period must be smaller"):
        CrossSectionalMomentumStrategy(lookback_period=10, skip_period=10)


def test_adf_pvalue_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        adf_pvalue(cast(Any, [1.0, 2.0, 3.0]))


def test_ols_coefficients_and_slope_handle_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        _ols_coefficients(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    intercept, slope = _ols_coefficients(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    assert np.isnan(intercept)
    assert np.isnan(slope)

    intercept, slope = _ols_coefficients(
        np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])
    )
    assert np.isnan(intercept)
    assert np.isnan(slope)

    assert _ols_slope(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) == (
        pytest.approx(2.0)
    )


def test_rolling_hedge_parameters_rejects_non_series_inputs() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        rolling_hedge_parameters(
            cast(Any, [1.0, 2.0]), pd.Series([1.0, 2.0]), window=2, dynamic=True
        )


def test_walk_pairs_positions_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        _walk_pairs_positions(
            np.array([1.0, 2.0]), np.array([True]), entry=1.0, exit_=0.5, stop=None
        )


def test_pairs_strategy_allows_no_stop_and_rejects_missing_symbol() -> None:
    strategy = PairsTradingStrategy("AAA", "BBB", stop_zscore=None)
    assert strategy.stop_zscore is None
    data = make_ohlcv("AAA", [100.0] * 30)
    with pytest.raises(StrategyError, match="needs symbol"):
        strategy.generate_signals(data)
