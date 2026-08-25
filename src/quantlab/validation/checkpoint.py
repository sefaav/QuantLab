"""Resumable-process checkpointing for long-running validation loops.

Lets an interrupted walk-forward, stress-test or sensitivity run continue from
where it left off instead of restarting, gated on the same kind of provenance
guarantee already used by :func:`quantlab.backtesting.result.
load_previous_robustness_artifacts` (config + data_hash + generator_hash +
dependency_versions, deliberately not git_dirty/git_commit — generator_hash
already hashes current file contents, uncommitted changes included).

Checkpoint files are pickle, trusted-local-file-only: this module's
:class:`_RestrictedUnpickler` blocks the well-known process/OS/import-machinery
gadgets (``os.system``, ``subprocess``, ``eval``, ...), but that is a coarse
mitigation, not a sandbox — it cannot certify a file as safe, only make the
common attack surface smaller. Provenance checking cannot help here either:
the provenance dict is itself *inside* the pickled payload, so it is only
checked after unpickling has already run whatever an embedded ``__reduce__``
specifies. Never point a checkpoint path at a file from an untrusted source
(a shared filesystem, a downloaded or cloned experiment folder, ...).
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
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

# Modules that can reach the OS, the interpreter or the import machinery --
# never legitimately part of checkpointed validation state (DataFrames,
# dataclasses, plain containers, numbers). Blocking these closes off the
# common pickle remote-code-execution gadgets (os.system, subprocess.Popen,
# ...) without needing a full allowlist of every pandas/numpy internal class
# a DataFrame's pickle stream happens to touch, which is fragile and
# version-dependent.
_DANGEROUS_MODULE_PREFIXES = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "sys",
        "shutil",
        "socket",
        "importlib",
        "runpy",
        "ctypes",
        "multiprocessing",
        "pty",
        "code",
        "pickle",
    }
)
_DANGEROUS_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
)


class _RestrictedUnpickler(pickle.Unpickler):  # nosemgrep: avoid-pickle
    """Blocks well-known code-execution gadgets in an untrusted pickle.

    A checkpoint file lives in a local experiment output directory and is
    read back by quantlab itself later -- but nothing stops it from having
    been replaced in the meantime (a shared filesystem, a downloaded or
    cloned experiment folder, ...). Provenance can't gate this the way it
    gates everything else in this module: the provenance dict is itself
    *inside* the pickled payload, so it can only be checked after
    unpickling has already run whatever code an embedded ``__reduce__``
    specifies. See :func:`_read_payload`.
    """

    def find_class(self, module: str, name: str) -> Any:
        top_level = module.partition(".")[0]
        if top_level in _DANGEROUS_MODULE_PREFIXES:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle {module}.{name}: disallowed module."
            )
        if module == "builtins" and name in _DANGEROUS_BUILTINS:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle builtins.{name}: disallowed callable."
            )
        return super().find_class(module, name)


def compute_provenance(
    config: ExperimentConfig, data: pd.DataFrame, **run_params: Any
) -> dict[str, Any]:
    """Fingerprint everything a checkpoint's validity depends on.

    ``run_params`` covers whatever a caller can override independently of
    ``config`` itself (train/validation/test windows, parameter grid,
    execution delay, ...) — those aren't captured by ``config`` alone but
    still change what a resumed run would compute.

    Uses ``_generator_hash()``, not ``_source_hash()``: a checkpoint is
    resumed from inside a CLI command's own loop, so a change to the CLI's
    own orchestration (e.g. how it passes overrides into this run) can
    change what a resumed run computes even when the narrower
    computational-only hash is unchanged — the same reasoning
    ``load_previous_walk_forward_robustness``/``load_previous_robustness_
    artifacts`` (``quantlab.backtesting.result``) apply to saved-bundle
    reuse. See ``_generator_hash()``'s own docstring.

    Calls ``_engine.<name>()`` through the module rather than importing
    ``_generator_hash``/``_dependency_versions`` by name, so a test that
    monkeypatches ``quantlab.backtesting.engine._generator_hash`` (the
    established pattern elsewhere in this codebase for pinning a hash
    across two CLI invocations in one test) actually takes effect here too.
    """
    return {
        "config": config.model_dump(mode="json"),
        "data_hash": ParquetStorage.hash_frame(data),
        "generator_hash": _engine._generator_hash(),
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
    """Return the raw ``{"provenance", "progress", "state"}`` payload, or None.

    A checkpoint is only ever written by :func:`save_checkpoint` with this
    exact shape -- exactly these three keys, no more, no fewer -- so any
    deviation, including an extra key, means the file is corrupted, foreign,
    or stale (never simply "an older valid shape" this module needs to keep
    reading); treated as unreadable, same as any other malformed payload.
    ``state`` itself is opaque here (its shape is defined by whichever
    command wrote it) -- callers must validate it themselves before relying
    on it, see :func:`load_checkpoint`.
    """
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            # nosemgrep: avoid-pickle -- trusted-local-file-only by design,
            # gated by _RestrictedUnpickler above (blocks the well-known
            # code-execution gadgets); see this module's own docstring for
            # why a full sandbox isn't achievable here and a JSON-only
            # format isn't a drop-in replacement for the arbitrary
            # DataFrame/dataclass state this checkpoints.
            payload = _RestrictedUnpickler(handle).load()
    except Exception as exc:
        logger.warning("Checkpoint at %s could not be read (%s).", path, exc)
        return None
    if (
        not isinstance(payload, dict)
        or payload.keys() != {"provenance", "progress", "state"}
        or not isinstance(payload["provenance"], dict)
        or isinstance(payload["progress"], bool)
        or not isinstance(payload["progress"], int)
        or payload["progress"] < 0
    ):
        logger.warning("Checkpoint at %s has an unexpected shape; ignoring it.", path)
        return None
    return payload


def load_checkpoint(
    path: Path,
    provenance: dict[str, Any],
    *,
    validate: Callable[[Any, int], bool] | None = None,
) -> tuple[Any, int] | None:
    """Return the checkpointed ``(state, progress)`` if it exists and matches.

    Any problem — missing file, unreadable pickle, mismatched provenance,
    or a ``validate`` failure — is treated as "nothing to resume" rather
    than raised, so a checkpoint can never block a run, only skip work it
    has already done.

    ``progress`` is exactly what the caller last passed to
    :func:`save_checkpoint` — return it as-is rather than making the caller
    re-derive "how much is done" from ``state`` itself (e.g. via
    ``len(state)``), which silently under- or over-counts whenever one
    checkpointed unit doesn't correspond to exactly one element of ``state``
    (e.g. a single scenario block that appends several result rows at once).

    Args:
        path: Checkpoint file.
        provenance: Must equal what the checkpoint was saved with.
        validate: Optional callable receiving ``(state, progress)``,
            returning whether they form a valid pair *for this specific
            command* — e.g. ``state`` has the expected container shape, or
            ``progress`` does not exceed this run's own total unit count.
            This module has no way to know either on its own: ``state``'s
            shape and what "total" means are defined entirely by the
            caller. Without a check here, a caller blindly destructuring an
            unexpected ``state`` (a corrupted file, or one left over from a
            differently-shaped older version of the same command) fails
            with a confusing raw unpacking error instead of a clear refusal
            to resume.
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
    state, progress = payload["state"], payload["progress"]
    if validate is not None:
        try:
            is_valid = validate(state, progress)
        except Exception as exc:
            # `state` is untrusted, arbitrary checkpointed data by
            # definition -- a caller's validator inspecting its content
            # (e.g. comparing a corrupted value that turns out to be
            # `pd.NA`) can itself raise partway through, not just return
            # False. This module's own contract ("any problem ... is
            # treated as 'nothing to resume' rather than raised") has to
            # hold even then: a validator crashing must never propagate out
            # of a checkpoint read and abort the run it was meant to help.
            logger.warning(
                "Not resuming from %s: checkpointed state/progress raised "
                "%s while validating it; ignoring it.",
                path,
                exc,
            )
            return None
        if not is_valid:
            logger.warning(
                "Not resuming from %s: checkpointed state/progress failed "
                "this command's own validation; ignoring it.",
                path,
            )
            return None
    return state, progress


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
        # nosemgrep: avoid-pickle -- serializing our own trusted in-memory
        # state, not deserializing untrusted input; see _read_payload's own
        # nosemgrep comment and this module's docstring for the full
        # rationale (arbitrary DataFrame/dataclass state, trusted-local-file
        # only).
        _write_path_atomic(path, lambda tmp: tmp.write_bytes(pickle.dumps(payload)))


def clear_checkpoint(path: Path) -> None:
    """Remove a checkpoint, e.g. after its run completes or via ``--fresh``."""
    with _locked_checkpoint(path):
        path.unlink(missing_ok=True)
