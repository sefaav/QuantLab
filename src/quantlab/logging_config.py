"""Centralised logging setup.

Uses only Python's standard-library ``logging`` module. By default,
:func:`configure_logging` sends QuantLab log records to stderr and attempts to
write the rotating file ``logs/quantlab.log``. An unavailable log directory
does not prevent a command from running.

Security note: this module does not redact secrets. Callers must ensure that
API keys and other sensitive values are never included in log messages.
"""

from __future__ import annotations

import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from quantlab.constants import LOGGER_NAME, LOGS_DIR

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Prevent repeated configuration calls from stacking duplicate handlers.
_CONFIGURED = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return the package-level ``quantlab`` logger.

    Args:
        level: Logging level for the package logger (name or numeric value).
        log_file: Preferred log file. Defaults to ``logs/quantlab.log``.
        console: Whether to also emit records to stderr.

    Returns:
        The configured package logger. Child loggers are obtained with
        :func:`get_logger`.
    """
    global _CONFIGURED

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if _CONFIGURED:
        # Already wired; just adjust the level and return.
        return logger

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    target = log_file if log_file is not None else LOGS_DIR / "quantlab.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError as exc:
        message = f"File logging disabled because {target} is unavailable: {exc}"
        if console:
            logger.warning(message)
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to the root logger to avoid double-printing.
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``quantlab`` namespace.

    Args:
        name: Dotted suffix, typically ``__name__``. If ``None`` returns the
            package logger itself.

    Returns:
        A ``logging.Logger`` whose records flow through the configured handlers.
    """
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    # Normalise "quantlab.data.yahoo" and "some.module" to a child of the root.
    suffix = name.split(".", 1)[-1] if name.startswith(LOGGER_NAME) else name
    return logging.getLogger(LOGGER_NAME).getChild(suffix)
