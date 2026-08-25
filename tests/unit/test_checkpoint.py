"""Tests for the resumable-process checkpointing module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.config import ExperimentConfig
from quantlab.validation.checkpoint import (
    clear_checkpoint,
    compute_provenance,
    load_checkpoint,
    save_checkpoint,
)


def test_save_and_load_round_trips_state(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1, 2, 3]}, progress=3)
    assert load_checkpoint(path, provenance) == ({"folds": [1, 2, 3]}, 3)


def test_load_returns_none_when_file_is_absent(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    provenance = compute_provenance(sample_config, synthetic_panel)
    assert load_checkpoint(tmp_path / "missing.pkl", provenance) is None


def test_load_returns_none_when_provenance_no_longer_matches(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1]}, progress=1)

    changed_config = sample_config.revalidated_copy(
        update={
            "execution": sample_config.execution.revalidated_copy(
                update={"commission_bps": 99.0}
            )
        }
    )
    new_provenance = compute_provenance(changed_config, synthetic_panel)
    assert load_checkpoint(path, new_provenance) is None


def test_load_returns_none_when_run_params_change(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """run_params (e.g. a walk-forward grid) aren't part of ``config`` itself,
    but still invalidate a checkpoint when they change between attempts."""
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(
        sample_config, synthetic_panel, train_window=300, parameter_grid={"a": [1, 2]}
    )
    save_checkpoint(path, provenance, {"folds": [1]}, progress=1)

    different_provenance = compute_provenance(
        sample_config,
        synthetic_panel,
        train_window=300,
        parameter_grid={"a": [1, 2, 3]},
    )
    assert load_checkpoint(path, different_provenance) is None


def test_load_returns_none_when_file_is_corrupted(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(b"not a pickle")
    provenance = compute_provenance(sample_config, synthetic_panel)
    assert load_checkpoint(path, provenance) is None


def test_load_returns_none_when_payload_has_an_unexpected_shape(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    import pickle

    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps(["not", "a", "dict"]))
    provenance = compute_provenance(sample_config, synthetic_panel)
    assert load_checkpoint(path, provenance) is None


def test_load_returns_none_when_payload_is_missing_the_state_key(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """A payload that passes the old, incomplete shape check (only checking
    for "provenance") but is missing "state" or "progress" must still be
    treated as an unreadable checkpoint -- never a raw KeyError, which would
    break this module's own documented contract that a bad checkpoint can
    only skip work, never crash the caller."""
    import pickle

    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    path.write_bytes(pickle.dumps({"provenance": provenance}))
    assert load_checkpoint(path, provenance) is None


def test_load_returns_none_when_progress_is_negative(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    import pickle

    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    path.write_bytes(
        pickle.dumps({"provenance": provenance, "progress": -1, "state": {}})
    )
    assert load_checkpoint(path, provenance) is None


def test_load_returns_none_when_payload_has_an_extra_key(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """The payload shape is exactly {"provenance", "progress", "state"} --
    an extra key means the file is corrupted, foreign, or from an
    incompatible version, not "an older valid shape plus something new" to
    tolerate."""
    import pickle

    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    path.write_bytes(
        pickle.dumps(
            {
                "provenance": provenance,
                "progress": 1,
                "state": {},
                "extra_key": "unexpected",
            }
        )
    )
    assert load_checkpoint(path, provenance) is None


def test_load_checkpoint_rejects_state_failing_a_caller_supplied_validator(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """A command-specific `validate` callback lets a caller refuse to resume
    from a checkpoint whose state/progress doesn't match its own expected
    shape (e.g. a wrong container type, or progress exceeding this run's
    own total unit count) -- this module has no way to know either on its
    own, since `state`'s shape and what "total" means are caller-defined."""
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1, 2, 3]}, progress=3)

    # Correctly shaped and within a generous bound -- must still load.
    assert load_checkpoint(
        path, provenance, validate=lambda state, progress: progress <= 10
    ) == ({"folds": [1, 2, 3]}, 3)

    # A validator that rejects this specific state/progress pair must be
    # honoured, exactly like a provenance mismatch.
    assert (
        load_checkpoint(path, provenance, validate=lambda state, progress: False)
        is None
    )
    assert (
        load_checkpoint(
            path, provenance, validate=lambda state, progress: progress <= 1
        )
        is None
    )


def test_clear_checkpoint_is_idempotent_on_an_absent_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.pkl"
    clear_checkpoint(path)  # must not raise
    clear_checkpoint(path)


def test_clear_checkpoint_removes_an_existing_file(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1]}, progress=1)
    assert path.is_file()
    clear_checkpoint(path)
    assert not path.is_file()


def test_save_does_not_regress_a_more_advanced_checkpoint(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """Simulates two processes racing on the same experiment: the
    less-advanced one's write must not clobber the more-advanced one's."""
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1, 2, 3, 4, 5]}, progress=5)
    save_checkpoint(path, provenance, {"folds": [1, 2]}, progress=2)
    assert load_checkpoint(path, provenance) == ({"folds": [1, 2, 3, 4, 5]}, 5)


def test_save_overwrites_when_progress_advances(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1]}, progress=1)
    save_checkpoint(path, provenance, {"folds": [1, 2]}, progress=2)
    assert load_checkpoint(path, provenance) == ({"folds": [1, 2]}, 2)


def test_save_overwrites_stale_provenance_regardless_of_progress(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """The monotonic guard only applies to a *matching* provenance — a new
    run (different provenance) must always be able to start fresh, even if
    the old, no-longer-relevant checkpoint had a higher progress count."""
    path = tmp_path / "checkpoint.pkl"
    old_provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, old_provenance, {"folds": list(range(10))}, progress=10)

    changed_config = sample_config.revalidated_copy(
        update={
            "execution": sample_config.execution.revalidated_copy(
                update={"commission_bps": 99.0}
            )
        }
    )
    new_provenance = compute_provenance(changed_config, synthetic_panel)
    save_checkpoint(path, new_provenance, {"folds": [1]}, progress=1)
    assert load_checkpoint(path, new_provenance) == ({"folds": [1]}, 1)


def test_load_refuses_a_malicious_os_system_payload(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """A checkpoint file's provenance can only be checked *after*
    unpickling -- it's itself inside the pickled payload -- so a hostile
    file (a shared filesystem, a downloaded experiment folder, ...) must
    never get to run arbitrary code just by being read. Provenance
    mismatch alone can't be the guard here; the unpickler itself must
    refuse the dangerous class before it's ever instantiated."""
    import os
    import pickle

    class Evil:
        def __reduce__(self) -> tuple[object, ...]:
            return (os.system, ("echo pwned > pwned.txt",))

    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps({"provenance": {}, "progress": 1, "state": Evil()}))

    assert load_checkpoint(path, {}) is None
    assert not (tmp_path / "pwned.txt").exists()


def test_load_refuses_a_malicious_eval_payload(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    import pickle

    class Evil:
        def __reduce__(self) -> tuple[object, ...]:
            return (eval, ("1 + 1",))

    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps({"provenance": {}, "progress": 1, "state": Evil()}))

    assert load_checkpoint(path, {}) is None


def test_load_still_round_trips_a_dataframe_containing_state(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    """The restricted unpickler must not collaterally break legitimate
    checkpointed state -- a DataFrame's pickle stream touches many
    pandas/numpy internal classes, none of which are on the blocklist."""
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    frame = pd.DataFrame(
        {"a": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2)
    )
    save_checkpoint(path, provenance, {"weights": frame}, progress=1)

    state, progress = load_checkpoint(path, provenance)  # type: ignore[misc]
    pd.testing.assert_frame_equal(state["weights"], frame)
    assert progress == 1
