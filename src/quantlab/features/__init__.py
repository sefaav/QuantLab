"""Public feature-engineering API.

Most indicators are trailing. ``forward_returns`` is intentionally
forward-looking and is exposed only for research labels.
"""

from __future__ import annotations

from quantlab.features.correlation import correlation_matrix
from quantlab.features.cross_sectional import (
    cross_sectional_demean,
    cross_sectional_percentile,
    cross_sectional_rank,
    cross_sectional_zscore,
    select_top_bottom,
)
from quantlab.features.mean_reversion import (
    bollinger_bands,
    distance_to_moving_average,
    half_life,
    normalized_distance_to_mean,
    rolling_percentile_rank,
    rolling_zscore,
    rsi,
)
from quantlab.features.momentum import (
    cross_sectional_momentum_persistence,
    ma_crossover_signal,
    momentum,
    momentum_persistence,
    moving_average,
    price_above_ma,
    rate_of_change,
    volatility_adjusted_momentum,
)
from quantlab.features.pairs_diagnostics import (
    PairDiagnostics,
    compute_pair_diagnostics,
)
from quantlab.features.pipeline import FeaturePipeline, FeatureSpec
from quantlab.features.returns import (
    cumulative_returns,
    equity_curve,
    forward_returns,
    log_returns,
    simple_returns,
)
from quantlab.features.stationarity import (
    ADFResult,
    CointegrationResult,
    adf_test,
    cointegration_test,
    hurst_exponent,
)
from quantlab.features.technical import (
    donchian_position,
    efficiency_ratio,
    exponential_moving_average,
    macd,
    rolling_max,
    rolling_min,
)
from quantlab.features.volatility import (
    annualize_volatility,
    average_true_range,
    downside_volatility,
    ewma_volatility,
    historical_volatility,
    realized_volatility,
    rolling_beta,
    rolling_correlation,
)

__all__ = [  # noqa: RUF022 - grouped by category, not alphabetical, for readability
    # returns
    "simple_returns",
    "log_returns",
    "forward_returns",
    "cumulative_returns",
    "equity_curve",
    # momentum
    "momentum",
    "rate_of_change",
    "moving_average",
    "ma_crossover_signal",
    "price_above_ma",
    "volatility_adjusted_momentum",
    "momentum_persistence",
    "cross_sectional_momentum_persistence",
    # volatility
    "historical_volatility",
    "realized_volatility",
    "ewma_volatility",
    "downside_volatility",
    "average_true_range",
    "rolling_beta",
    "rolling_correlation",
    "annualize_volatility",
    # mean reversion
    "rolling_zscore",
    "distance_to_moving_average",
    "normalized_distance_to_mean",
    "rolling_percentile_rank",
    "rsi",
    "bollinger_bands",
    "half_life",
    # cross sectional
    "cross_sectional_rank",
    "cross_sectional_percentile",
    "cross_sectional_zscore",
    "cross_sectional_demean",
    "select_top_bottom",
    # technical
    "exponential_moving_average",
    "macd",
    "rolling_max",
    "rolling_min",
    "donchian_position",
    "efficiency_ratio",
    # correlation
    "correlation_matrix",
    # stationarity / cointegration
    "ADFResult",
    "CointegrationResult",
    "adf_test",
    "cointegration_test",
    "hurst_exponent",
    # pairs diagnostics
    "PairDiagnostics",
    "compute_pair_diagnostics",
    # pipeline
    "FeaturePipeline",
    "FeatureSpec",
]
