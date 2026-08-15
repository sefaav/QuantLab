"""Custom exception hierarchy for QuantLab.

Having a small, explicit hierarchy lets callers catch broad categories
(``QuantLabError``) or precise failures (``DataValidationError``), and keeps
error messages actionable — never a bare ``Exception``.
"""

from __future__ import annotations


class QuantLabError(Exception):
    """Base class for every error raised deliberately by QuantLab."""


class DataDownloadError(QuantLabError):
    """A market-data source failed to download or returned nothing usable."""


class DataValidationError(QuantLabError):
    """Market data violated an invariant of the canonical schema.

    The message should tell the user *what* failed and *how* to fix it, e.g.::

        Found 14 duplicate timestamps for symbol SPY.
        Run the data cleaning pipeline or inspect the source file.
    """


class InvalidConfigurationError(QuantLabError):
    """An experiment configuration was structurally or semantically invalid."""


class BacktestError(QuantLabError):
    """The backtest engine could not complete a run."""


class StrategyError(QuantLabError):
    """A strategy produced signals that violate the strategy contract."""


class InsufficientDataError(QuantLabError):
    """Not enough observations to compute a feature, signal or metric."""
