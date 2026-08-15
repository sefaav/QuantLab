"""Tests for package logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NoReturn

import pytest

import quantlab.logging_config as logging_config
from quantlab.constants import LOGGER_NAME


def test_unwritable_log_file_does_not_abort_console_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file-system logging failure must not prevent a CLI command."""

    def refuse_file_handler(*args: object, **kwargs: object) -> NoReturn:
        raise PermissionError("read-only log directory")

    logger = logging.getLogger(LOGGER_NAME)
    original_handlers = logger.handlers[:]
    original_propagate = logger.propagate
    try:
        logger.handlers.clear()
        monkeypatch.setattr(logging_config, "_CONFIGURED", False)
        monkeypatch.setattr(logging_config, "RotatingFileHandler", refuse_file_handler)

        configured = logging_config.configure_logging(
            log_file=tmp_path / "unavailable" / "quantlab.log"
        )

        assert configured is logger
        assert any(
            type(handler) is logging.StreamHandler for handler in logger.handlers
        )
        assert "File logging disabled" in capsys.readouterr().err
    finally:
        logger.handlers.clear()
        logger.handlers.extend(original_handlers)
        logger.propagate = original_propagate
