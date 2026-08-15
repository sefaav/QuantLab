"""QuantLab — Reproducible quantitative research and backtesting platform.

The public API is intentionally small at the top level; import submodules for
the full surface (``quantlab.backtesting.engine``, ``quantlab.strategies`` …).

This project is for educational and research purposes only. It is not
investment advice, and historical performance does not guarantee future
results.
"""

from __future__ import annotations

from quantlab.config import ExperimentConfig
from quantlab.exceptions import (
    BacktestError,
    DataDownloadError,
    DataValidationError,
    InvalidConfigurationError,
    QuantLabError,
)

__version__ = "0.1.0"

__all__ = [
    "BacktestError",
    "DataDownloadError",
    "DataValidationError",
    "ExperimentConfig",
    "InvalidConfigurationError",
    "QuantLabError",
    "__version__",
]
