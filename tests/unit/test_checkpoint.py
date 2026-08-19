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
    assert load_checkpoint(path, provenance) == {"folds": [1, 2, 3]}


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
    assert load_checkpoint(path, provenance) == {"folds": [1, 2, 3, 4, 5]}


def test_save_overwrites_when_progress_advances(
    tmp_path: Path, sample_config: ExperimentConfig, synthetic_panel: pd.DataFrame
) -> None:
    path = tmp_path / "checkpoint.pkl"
    provenance = compute_provenance(sample_config, synthetic_panel)
    save_checkpoint(path, provenance, {"folds": [1]}, progress=1)
    save_checkpoint(path, provenance, {"folds": [1, 2]}, progress=2)
    assert load_checkpoint(path, provenance) == {"folds": [1, 2]}


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
    assert load_checkpoint(path, new_provenance) == {"folds": [1]}
