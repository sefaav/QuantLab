"""Resumable-process checkpointing for long-running validation loops.

Lets an interrupted walk-forward, stress-test or sensitivity run continue from
where it left off instead of restarting, gated on the same provenance
guarantee already used by :func:`quantlab.backtesting.result.
load_previous_robustness_artifacts` (config + data_hash + code_hash +
dependency_versions, deliberately not git_dirty/git_commit — code_hash already
hashes current file contents, uncommitted changes included).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from quantlab.backtesting import engine as _engine
from quantlab.backtesting.result import _write_path_atomic
from quantlab.config import ExperimentConfig
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import BacktestError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

_LOCK_TIMEOUT_SECONDS = 30.0


def compute_provenance(
    config: ExperimentConfig, data: pd.DataFrame, **run_params: Any
) -> dict[str, Any]:
    """Fingerprint everything a checkpoint's validity depends on.

    ``run_params`` covers whatever a caller can override independently of
    ``config`` itself (train/validation/test windows, parameter grid,
    execution delay, ...) — those aren't captured by ``config`` alone but
    still change what a resumed run would compute.

    Calls ``_engine.<name>()`` through the module rather than importing
    ``_source_hash``/``_dependency_versions`` by name, so a test that
    monkeypatches ``quantlab.backtesting.engine._source_hash`` (the
    established pattern elsewhere in this codebase for pinning code_hash
    across two CLI invocations in one test) actually takes effect here too.
    """
    return {
        "config": config.model_dump(mode="json"),
        "data_hash": ParquetStorage.hash_frame(data),
        "code_hash": _engine._source_hash(),
        "dependency_versions": _engine._dependency_versions(),
        "run_params": run_params,
    }


def _lock_path(path: Path) -> Path:
    """Return the persistent sibling lock used to serialize checkpoint access."""
    resolved = path.resolve()
    return resolved.parent / f".{resolved.name}.lock"


@contextmanager
def _locked_checkpoint(path: Path) -> Iterator[None]:
    """Serialize reads and writes against one checkpoint file.

    A separate lock from `backtesting.result._locked_bundle`'s (the final
    bundle save) on purpose — checkpoint writes happen throughout a run,
    the bundle save happens once at the end; they're independent critical
    sections and don't need to contend with each other.
    """
    lock = FileLock(str(_lock_path(path)), timeout=_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except FileLockTimeout as exc:
        raise BacktestError(
            f"Timed out waiting to access the checkpoint at {path} after "
            f"{_LOCK_TIMEOUT_SECONDS:g} seconds; another process may still be "
            "using it."
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _read_payload(path: Path) -> dict[str, Any] | None:
    """Return the raw ``{"provenance", "progress", "state"}`` payload, or None."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        logger.warning("Checkpoint at %s could not be read (%s).", path, exc)
        return None
    if not isinstance(payload, dict) or "provenance" not in payload:
        logger.warning("Checkpoint at %s has an unexpected shape; ignoring it.", path)
        return None
    return payload


def load_checkpoint(path: Path, provenance: dict[str, Any]) -> Any | None:
    """Return the checkpointed state if it exists and provenance matches.

    Any problem — missing file, unreadable pickle, mismatched provenance —
    is treated as "nothing to resume" rather than raised, so a checkpoint
    can never block a run, only skip work it has already done.
    """
    with _locked_checkpoint(path):
        payload = _read_payload(path)
    if payload is None:
        return None
    if payload["provenance"] != provenance:
        logger.info(
            "Not resuming from %s: config, data, code or run parameters have "
            "changed since it was written.",
            path,
        )
        return None
    return payload["state"]


def save_checkpoint(
    path: Path, provenance: dict[str, Any], state: Any, progress: int
) -> None:
    """Atomically replace the checkpoint at ``path``, without ever regressing it.

    ``progress`` is an opaque, caller-supplied count (folds/scenarios/cells/
    candidates already done — not interpreted, only compared). Two processes
    racing on the same experiment (same provenance, same path) could
    otherwise have the less-advanced one overwrite the more-advanced one's
    checkpoint: the lock makes the read-compare-write atomic, and only
    writing when ``progress`` is at least as large as what's already on disk
    (for matching provenance) means the more-advanced state always wins,
    never silently regresses.
    """
    with _locked_checkpoint(path):
        existing = _read_payload(path)
        if (
            existing is not None
            and existing["provenance"] == provenance
            and existing.get("progress", -1) > progress
        ):
            logger.debug(
                "Not overwriting checkpoint at %s: on-disk progress (%s) is "
                "already ahead of this write (%s).",
                path,
                existing.get("progress"),
                progress,
            )
            return
        payload = {"provenance": provenance, "progress": progress, "state": state}
        _write_path_atomic(path, lambda tmp: tmp.write_bytes(pickle.dumps(payload)))


def clear_checkpoint(path: Path) -> None:
    """Remove a checkpoint, e.g. after its run completes or via ``--fresh``."""
    with _locked_checkpoint(path):
        path.unlink(missing_ok=True)
