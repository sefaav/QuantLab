"""Config-driven backtest assembly.

Turns an :class:`ExperimentConfig` plus market data into a
:class:`BacktestResult` by building the strategy, allocator and execution model
from the config — the single path used by the CLI and dashboard so a whole
experiment is reproducible from one YAML file.
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtesting.engine import BacktestEngine
from quantlab.backtesting.result import BacktestResult
from quantlab.config import ExperimentConfig
from quantlab.constants import (
    CALENDAR_DAYS_PER_YEAR,
    CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR,
    FREQUENCY_TO_PERIODS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
)
from quantlab.data.base import price_matrix, volume_matrix
from quantlab.data.calendar import is_247, uniform_calendar
from quantlab.data.validator import DataQualityReport
from quantlab.execution.execution_model import ExecutionModel
from quantlab.portfolio.allocator import PortfolioAllocator, build_allocator
from quantlab.strategies.base import (
    BaseStrategy,
    build_strategy,
    strategy_parameter_names,
)


def build_allocator_from_config(config: ExperimentConfig) -> PortfolioAllocator:
    """Instantiate the configured allocator, passing only the kwargs it accepts."""
    name = config.portfolio.allocator
    kwargs: dict[str, object] = {}
    if name == "inverse_volatility":
        kwargs = {
            "volatility_window": config.portfolio.volatility_window,
            "maximum_weight": config.portfolio.maximum_weight,
            "periods_per_year": config.periods_per_year,
        }
    elif name == "volatility_targeting":
        # PortfolioConfig._check_volatility_targeting_requires_target_volatility
        # guarantees this is set -- never an implicit default.
        assert config.portfolio.target_volatility is not None
        kwargs = {
            "target_volatility": config.portfolio.target_volatility,
            "volatility_window": config.portfolio.volatility_window,
            "maximum_leverage": config.portfolio.maximum_leverage,
            "periods_per_year": config.periods_per_year,
        }
    return build_allocator(name, **kwargs)


def build_strategy_from_config(config: ExperimentConfig) -> BaseStrategy:
    """Instantiate the configured strategy with its parameter dict.

    Injects the experiment's annualisation factor when the strategy accepts
    ``periods_per_year`` and the YAML does not override it.
    """
    parameters = dict(config.strategy_parameters)
    accepted = strategy_parameter_names(config.strategy_name)
    if "periods_per_year" in accepted and "periods_per_year" not in parameters:
        parameters["periods_per_year"] = config.periods_per_year
    return build_strategy(config.strategy_name, parameters)


def build_execution_from_config(
    config: ExperimentConfig, data: pd.DataFrame
) -> ExecutionModel:
    """Build the execution model, wiring ADV for volume-based slippage.

    Dollar ADV uses unadjusted prices and a trailing 21-day market window,
    shifted once so a fill never sees its own bar's volume. The configured
    frequency and market calendar determine the bars per day; an explicit
    metrics annualisation override does not alter this physical conversion.

    When instruments trade on different calendars, this window falls back to
    the equity (252-day, non-24/7) convention — a documented, accepted
    approximation, since it only sizes a nominal liquidity window for
    volume-based slippage rather than multiplying directly into reported
    metrics the way ``periods_per_year`` does.
    """
    adv: pd.DataFrame | float | None = None
    if config.execution.slippage_model.lower() in {"volume", "volume_based"}:
        shares = volume_matrix(data)
        price = price_matrix(data, adjusted=False)
        bar_dollar_volume = shares * price
        calendar = uniform_calendar(
            instrument.calendar for instrument in config.data.instruments
        )
        market_is_247 = calendar is not None and is_247(calendar)
        frequency_table = (
            CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR
            if market_is_247
            else FREQUENCY_TO_PERIODS_PER_YEAR
        )
        days_per_year = (
            CALENDAR_DAYS_PER_YEAR if market_is_247 else TRADING_DAYS_PER_YEAR
        )
        bars_per_day = frequency_table[str(config.frequency)] / days_per_year
        window = max(1, round(21 * bars_per_day))
        adv = (
            bar_dollar_volume.rolling(window, min_periods=1).mean().shift(1)
            * bars_per_day
        )
    return ExecutionModel.from_config(config.execution, average_daily_volume=adv)


def run_backtest_from_config(
    data: pd.DataFrame,
    config: ExperimentConfig,
    *,
    execution_delay: int = 0,
    data_quality_report: DataQualityReport | None = None,
) -> BacktestResult:
    """Assemble components from ``config`` and run the backtest.

    A configured holdout also attaches its train/validation/test metrics and
    test series to the returned result.

    Args:
        data: Canonical long OHLCV frame.
        config: Experiment configuration.
        execution_delay: Extra periods of execution delay, forwarded to
            :meth:`~quantlab.backtesting.engine.BacktestEngine.run` (the
            "acting on stale signals" stress scenario).
        data_quality_report: The report ``DataLoader.load()`` produced for
            ``data``. When provided, it is stored in result metadata.
    """
    strategy = build_strategy_from_config(config)
    allocator = build_allocator_from_config(config)
    execution_model = build_execution_from_config(config, data)
    result = BacktestEngine().run(
        data,
        strategy,
        allocator,
        execution_model,
        config,
        execution_delay=execution_delay,
    )
    if data_quality_report is not None:
        result.metadata["data_quality"] = data_quality_report.to_dict()

    if config.validation.method == "holdout":
        from quantlab.validation.holdout import run_holdout_report

        holdout = run_holdout_report(data, config, result)
        if holdout is not None:
            result.metadata["holdout_chronological_metrics"] = holdout.test_metrics
            result.metadata["holdout_report"] = holdout.to_metadata()
            result.holdout_test_returns = holdout.test_returns
            result.holdout_test_equity = holdout.test_equity
    return result
