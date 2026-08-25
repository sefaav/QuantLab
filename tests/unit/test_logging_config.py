"""Tests for package logging setup."""

from __future__ import annotations

import io
import logging
import sys
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
        # isinstance, not an exact type check: the console handler is
        # _CurrentStderrHandler, a StreamHandler subclass that always
        # resolves sys.stderr fresh on every emit instead of a snapshot
        # taken at construction time (see its docstring for why).
        assert any(
            isinstance(handler, logging.StreamHandler) for handler in logger.handlers
        )
        assert "File logging disabled" in capsys.readouterr().err
    finally:
        logger.handlers.clear()
        logger.handlers.extend(original_handlers)
        logger.propagate = original_propagate


def test_console_handler_survives_sys_stderr_being_swapped_and_closed() -> None:
    """A plain logging.StreamHandler() snapshots sys.stderr once at
    construction; if the process later replaces sys.stderr (e.g. pytest
    swaps in a fresh per-test capture object and closes the previous one --
    the console handler is only ever built once per process, gated on
    configure_logging's own _CONFIGURED latch), it would try to write to
    the now-closed old stream and log a swallowed "I/O operation on closed
    file" error instead of the real message. The console handler must
    resolve sys.stderr fresh on every emit instead."""
    handler = logging_config._CurrentStderrHandler()
    logger = logging.getLogger("quantlab-test-console-handler-survives")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    old_stderr = sys.stderr
    try:
        first = io.StringIO()
        sys.stderr = first
        logger.warning("first message")
        first.close()

        second = io.StringIO()
        sys.stderr = second
        logger.warning("second message")
    finally:
        sys.stderr = old_stderr
        logger.removeHandler(handler)

    output = second.getvalue()
    assert "second message" in output
    assert "Logging error" not in output
