"""Compact default parameter grids for walk-forward validation."""

from __future__ import annotations

from typing import Any

from quantlab.config import ExperimentConfig


def parameter_grid_for_config(config: ExperimentConfig) -> dict[str, list[Any]]:
    """Return the configured walk-forward grid or the strategy default.

    The copy prevents callers from mutating candidate lists stored inside the
    otherwise frozen experiment configuration.
    """
    configured = config.validation.parameter_grid
    if configured is not None:
        return {name: list(values) for name, values in configured.items()}
    return default_parameter_grid(config)


def default_parameter_grid(config: ExperimentConfig) -> dict[str, list[Any]]:
    """Return a compact, valid grid for the configured strategy regime.

    Continuous choices such as windows and thresholds are candidates for
    selection. Structural choices, such as long-only versus long/short,
    remain fixed so one walk-forward run evaluates one research hypothesis.
    Every optimized field includes its configured value.
    """
    parameters = config.strategy_parameters

    match config.strategy_name:
        case "buy_and_hold":
            return {}
        case "time_series_momentum":
            grid = _momentum_window_grid(parameters)
            if parameters.get("signal_scaling", "binary") == "volatility_adjusted":
                grid["volatility_window"] = _ordered_unique(
                    [21, 63, 126, int(parameters.get("volatility_window", 63))]
                )
            return grid
        case "cross_sectional_momentum":
            grid = _momentum_window_grid(parameters)
            grid.update(_cross_sectional_fraction_grid(config, parameters))
            return grid
        case "mean_reversion":
            configured_entry = float(parameters.get("entry_zscore", 2.0))
            exit_zscore = float(parameters.get("exit_zscore", 0.5))
            stop_zscore = parameters.get("stop_zscore", 4.0)
            maximum_entry = float(stop_zscore) if stop_zscore is not None else None
            return {
                "lookback_period": _ordered_unique(
                    [10, 20, 40, int(parameters.get("lookback_period", 20))]
                ),
                "entry_zscore": [
                    value
                    for value in _ordered_unique([1.5, 2.0, 2.5, configured_entry])
                    if value > exit_zscore
                    and (maximum_entry is None or value < maximum_entry)
                ],
            }
        case "trend_following":
            configured_fast = int(parameters.get("fast_window", 20))
            configured_slow = int(parameters.get("slow_window", 100))
            fast_candidates = [
                value
                for value in _ordered_unique([10, 20, 40, configured_fast])
                if value < configured_slow
            ]
            slow_candidates = [
                value
                for value in _ordered_unique([50, 100, 200, configured_slow])
                if value > max(fast_candidates)
            ]
            return {"fast_window": fast_candidates, "slow_window": slow_candidates}
        case "pairs_trading":
            configured_entry = float(parameters.get("entry_zscore", 2.0))
            exit_zscore = float(parameters.get("exit_zscore", 0.5))
            stop_zscore = parameters.get("stop_zscore", 4.0)
            maximum_entry = float(stop_zscore) if stop_zscore is not None else None
            return {
                "formation_window": _ordered_unique(
                    [126, 252, 504, int(parameters.get("formation_window", 252))]
                ),
                "zscore_window": _ordered_unique(
                    [21, 63, 126, int(parameters.get("zscore_window", 63))]
                ),
                "entry_zscore": [
                    value
                    for value in _ordered_unique([1.5, 2.0, 2.5, configured_entry])
                    if value > exit_zscore
                    and (maximum_entry is None or value < maximum_entry)
                ],
            }
        case _:
            return {}


def _momentum_window_grid(parameters: dict[str, Any]) -> dict[str, list[Any]]:
    """Build compatible lookback/skip candidates around configured values."""
    configured_lookback = int(parameters.get("lookback_period", 252))
    configured_skip = int(parameters.get("skip_period", 21))
    skip_candidates = [
        value
        for value in _ordered_unique([0, 21, configured_skip])
        if value < configured_lookback
    ]
    largest_skip = max(skip_candidates)
    lookback_candidates = [
        value
        for value in _ordered_unique([126, 189, 252, configured_lookback])
        if value > largest_skip
    ]
    return {"lookback_period": lookback_candidates, "skip_period": skip_candidates}


def _cross_sectional_fraction_grid(
    config: ExperimentConfig, parameters: dict[str, Any]
) -> dict[str, list[Any]]:
    """Return selection fractions that remain disjoint for this universe."""
    configured_top = float(parameters.get("top_fraction", 0.25))
    long_short = bool(parameters.get("long_short", False))
    configured_bottom = float(parameters.get("bottom_fraction", 0.25))
    universe_size = len(config.symbols)

    if long_short:
        # Every Cartesian top/bottom pair must remain valid.
        shared_cap = min(0.5, 1.0 - configured_top, 1.0 - configured_bottom)
        standard = [value for value in (0.10, 0.25, 0.50) if value <= shared_cap]
        top = _distinct_selection_fractions([configured_top, *standard], universe_size)
        bottom = _distinct_selection_fractions(
            [configured_bottom, *standard], universe_size
        )
        return {"top_fraction": top, "bottom_fraction": bottom}

    return {
        "top_fraction": _distinct_selection_fractions(
            [configured_top, 0.10, 0.25, 0.50], universe_size
        )
    }


def _distinct_selection_fractions(
    candidates: list[float], universe_size: int
) -> list[Any]:
    """Keep one fraction per effective cross-sectional position count."""
    selected_counts: set[int] = set()
    result: list[Any] = []
    for value in _ordered_unique(candidates):
        count = max(1, int(universe_size * value)) if value > 0.0 else 0
        if count not in selected_counts:
            selected_counts.add(count)
            result.append(value)
    return result


def _ordered_unique(values: list[Any]) -> list[Any]:
    """Deduplicate a short candidate list while preserving display order."""
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
