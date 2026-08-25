"""Backtest result container.

:class:`BacktestResult` bundles the managed outputs of a run and knows how to
summarise, plot, render to HTML and save itself in the reproducible directory
layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from quantlab.config import ExperimentConfig
from quantlab.constants import GENERATED_REPORTS_DIR
from quantlab.exceptions import BacktestError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

#: Optional files cleared before saving unless a caller has verified that an
#: existing artefact is still compatible with the new result.
_OPTIONAL_ARTIFACTS = (
    "benchmark.csv",
    "holdout_test_returns.csv",
    "holdout_test_equity.csv",
    "walk_forward_results.csv",
    "walk_forward_oos_returns.csv",
    "walk_forward_oos_equity.csv",
    "stress_tests.csv",
    "bootstrap_summary.csv",
    "permutation_test.csv",
    "sensitivity.csv",
)

_SAVE_IN_PROGRESS_MARKER = ".quantlab-save-in-progress"
_SAVE_LOCK_TIMEOUT_SECONDS = 30.0
_VALIDATION_ARTIFACT_INDEX = {
    "walk_forward_results.csv": False,
    "walk_forward_oos_returns.csv": True,
    "walk_forward_oos_equity.csv": True,
    "stress_tests.csv": False,
    "bootstrap_summary.csv": False,
    "permutation_test.csv": False,
    "sensitivity.csv": False,
}
_VALIDATION_ARTIFACTS = frozenset(_VALIDATION_ARTIFACT_INDEX)

#: The four on-demand robustness techniques, each normally run and saved by
#: its own separate CLI command — the CSV filename each one's `robustness`
#: dict key round-trips through, used by `load_previous_robustness_artifacts`
#: to recover what a sibling command already saved.
_ROBUSTNESS_ARTIFACT_FILES: dict[str, str] = {
    "stress_tests": "stress_tests.csv",
    "bootstrap": "bootstrap_summary.csv",
    "permutation_test": "permutation_test.csv",
    "sensitivity": "sensitivity.csv",
}

#: The subset of `_ROBUSTNESS_ARTIFACT_FILES` keys whose CLI command also
#: records its own effective run parameters (n_iterations/block_size/etc.,
#: including any CLI override) in `metadata.json` -- stress-test has no
#: such override, so it has no entry here. Used by
#: `load_previous_robustness_artifacts` to recover a sibling command's
#: parameters alongside its CSV, not just the numbers with no record of
#: what produced them.
_ROBUSTNESS_RUN_PARAMS_KEYS: dict[str, str] = {
    "bootstrap": "bootstrap_run_params",
    "permutation_test": "permutation_test_run_params",
    "sensitivity": "sensitivity_run_params",
}


def _bundle_lock_path(output_directory: Path) -> Path:
    """Return the persistent sibling lock used to serialize bundle saves."""
    resolved = output_directory.resolve()
    return resolved.parent / f".{resolved.name}.quantlab-save.lock"


@contextmanager
def _locked_bundle(output_directory: Path) -> Iterator[None]:
    """Serialize reads and writes that participate in a bundle save."""
    lock = FileLock(
        str(_bundle_lock_path(output_directory)),
        timeout=_SAVE_LOCK_TIMEOUT_SECONDS,
    )
    try:
        lock.acquire()
    except FileLockTimeout as exc:
        raise BacktestError(
            "Timed out waiting to save the result bundle at "
            f"{output_directory} after {_SAVE_LOCK_TIMEOUT_SECONDS:g} seconds; "
            "another process may still be writing the same experiment."
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _write_path_atomic(path: Path, writer: Callable[[Path], object]) -> None:
    """Write one file beside its destination, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, document: str) -> None:
    """Atomically replace one UTF-8 text file."""
    _write_path_atomic(
        path, lambda temporary: temporary.write_text(document, encoding="utf-8")
    )


def _write_csv_atomic(
    value: pd.Series | pd.DataFrame,
    path: Path,
    *,
    index: bool = True,
) -> None:
    """Atomically replace one CSV file."""
    _write_path_atomic(path, lambda temporary: value.to_csv(temporary, index=index))


@dataclass
class BacktestResult:
    """All outputs of a single backtest."""

    config: ExperimentConfig
    equity_curve: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series | None
    positions: pd.DataFrame
    weights: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Extra series kept for reporting / gross-vs-net comparison.
    gross_returns: pd.Series | None = None
    gross_equity: pd.Series | None = None
    turnover: pd.Series | None = None
    # Post-constraint targets before stateful rebalancing and turnover caps.
    # Walk-forward concatenates them before applying those stateful steps.
    target_weights: pd.DataFrame | None = None
    # Populated only when `validation.method == "holdout"`: the
    # out-of-sample test block's own returns/equity, saved separately from the
    # full-sample series so the holdout evidence is reproducible from disk.
    holdout_test_returns: pd.Series | None = None
    holdout_test_equity: pd.Series | None = None
    # Rendering failures recorded by `save()` after numeric artefacts are saved.
    save_warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Human-facing views
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Return a formatted text summary of headline metrics."""
        m = self.metrics
        lines = [
            f"Experiment : {self.config.experiment_name}",
            f"Strategy   : {self.config.strategy_name}",
            f"Symbols    : {', '.join(self.config.symbols)}",
            f"Period     : {self.equity_curve.index.min():%Y-%m-%d} "
            f"-> {self.equity_curve.index.max():%Y-%m-%d}",
            "-" * 48,
            f"Total return       : {m.get('total_return', 0):>10.2%}",
            f"CAGR               : {m.get('cagr', 0):>10.2%}",
            f"Volatility (ann.)  : {m.get('annualized_volatility', 0):>10.2%}",
            f"Sharpe             : {m.get('sharpe_ratio', 0):>10.2f}",
            f"Sortino            : {m.get('sortino_ratio', 0):>10.2f}",
            f"Calmar             : {m.get('calmar_ratio', 0):>10.2f}",
            f"Max drawdown       : {m.get('max_drawdown', 0):>10.2%}",
            f"Hit rate (non-zero periods): {m.get('hit_rate', 0):>5.2%}",
            f"Total costs (currency units): {self.total_costs():>7.2f}",
            f"Number of trades   : {self.number_of_trades():>10d}",
        ]
        if self.benchmark_returns is not None and "beta" in m:
            lines += [
                "-" * 48,
                f"Beta               : {m.get('beta', 0):>10.2f}",
                f"Alpha (ann.)       : {m.get('alpha', 0):>10.2%}",
                f"Information ratio  : {m.get('information_ratio', 0):>10.2f}",
            ]
        return "\n".join(lines)

    def total_costs(self) -> float:
        """Total transaction cost in currency (sum of trade-log costs)."""
        if len(self.trades) and "total_cost" in self.trades.columns:
            return float(self.trades["total_cost"].sum())
        return 0.0

    def total_cost_fraction(self) -> float:
        """Total cost as a fraction of equity, summed over time."""
        if "total" in self.costs.columns:
            return float(self.costs["total"].sum())
        return 0.0

    def number_of_trades(self) -> int:
        """Number of individual fills recorded in the trade log."""
        return len(self.trades)

    def gross_net_comparison(self) -> dict[str, float]:
        """Return gross/net performance, cost drag and currency costs."""
        out: dict[str, float] = {
            "net_total_return": self.metrics.get("total_return", 0.0),
            "net_sharpe": self.metrics.get("sharpe_ratio", 0.0),
            "total_cost": self.total_costs(),
        }
        if self.gross_returns is not None:
            from quantlab.risk.metrics import sharpe_ratio

            out["gross_sharpe"] = sharpe_ratio(
                self.gross_returns,
                risk_free_rate=self.config.risk_free_rate,
                periods_per_year=self.config.periods_per_year,
            )
        if self.gross_equity is not None and len(self.gross_equity) > 1:
            out["gross_total_return"] = float(
                self.gross_equity.iloc[-1] / self.gross_equity.iloc[0] - 1.0
            )
            out["cost_drag"] = out["gross_total_return"] - out["net_total_return"]
        return out

    # ------------------------------------------------------------------ #
    # Rendering (delegates to the reporting module)
    # ------------------------------------------------------------------ #
    def plot(self) -> Any:
        """Return an interactive equity/drawdown figure.

        Returns a ``plotly.graph_objects.Figure``; typed ``Any`` because plotly
        is an optional, lazily-imported dependency with no bundled type stubs.
        """
        from quantlab.reporting.charts import equity_and_drawdown_figure

        return equity_and_drawdown_figure(self)

    def to_html(
        self,
        output_path: str | Path | None = None,
        *,
        robustness: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        figures: dict[str, str] | None = None,
    ) -> str:
        """Render a self-contained HTML research report.

        Args:
            output_path: If given, the HTML is also written here.
            robustness: Extra sections (e.g. walk-forward / stress tables) to
                fold into the Robustness section. A holdout train/validation/
                test table is added automatically whenever
                ``metadata["holdout_report"]`` is present, merged with
                whatever is passed here.
            warnings: If given, any chart that fails to render is appended
                here instead of only being logged (see `report_figures`).
            figures: Optional pre-rendered chart data URIs to reuse.
        """
        from quantlab.reporting.html_report import render_html_report

        merged = self._merged_robustness(robustness)
        return render_html_report(
            self,
            output_path,
            robustness=merged,
            warnings=warnings,
            figures=figures,
        )

    def _merged_robustness(
        self, robustness: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Add the saved holdout split to caller-supplied robustness tables."""
        holdout_meta = self.metadata.get("holdout_report")
        if not holdout_meta:
            return robustness
        # Two-way train/test holdouts omit validation fields entirely.
        blocks = [
            ("Train", holdout_meta["train_metrics"], holdout_meta["train_period"]),
        ]
        if "validation_metrics" in holdout_meta:
            blocks.append(
                (
                    "Validation",
                    holdout_meta["validation_metrics"],
                    holdout_meta["validation_period"],
                )
            )
        # Labeled plainly "Test", not "out-of-sample": whether it's
        # genuinely OOS depends on parameters having been fixed before
        # looking at it, a property of the user's workflow this table
        # can't verify (see quantlab.validation.holdout's module docstring).
        blocks.append(
            (
                "Test",
                holdout_meta["test_metrics"],
                holdout_meta["test_period"],
            )
        )
        holdout_table = pd.DataFrame(
            [
                {
                    "Block": block,
                    "Period": f"{period[0][:10]} to {period[1][:10]}",
                    "Sharpe": metrics.get("sharpe_ratio", float("nan")),
                    "CAGR": metrics.get("cagr", float("nan")),
                    "Max Drawdown": metrics.get("max_drawdown", float("nan")),
                }
                for block, metrics, period in blocks
            ]
        )
        merged = dict(robustness) if robustness else {}
        merged.setdefault("holdout_split", holdout_table)
        return merged

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(
        self,
        output_directory: str | Path | None = None,
        *,
        robustness: dict[str, Any] | None = None,
        keep_artifacts: Iterable[str] = (),
        validation_artifacts: Mapping[str, pd.Series | pd.DataFrame] | None = None,
    ) -> Path:
        """Persist the result's managed outputs to a reproducible directory.

        Writes ``config.yaml``, ``metadata.json``, ``metrics.json`` and CSVs for
        the equity curve, benchmark, trades, positions and costs, plus a
        ``figures/`` folder and an HTML report.

        Args:
            output_directory: Destination. Defaults to
                ``reports/generated/<experiment_name>/``.
            robustness: Extra sections (e.g. walk-forward / stress tables) to
                fold into the saved report's Robustness section.
            keep_artifacts: Filenames from ``_OPTIONAL_ARTIFACTS`` to spare
                from the pre-save cleanup below — for a caller that has
                already verified a file left by a *different* command still
                describes this exact config/result (e.g. ``quantlab report``
                reusing a still-valid prior ``walk-forward`` run's CSVs) and
                will not be rewriting it itself this call. Prefer
                :func:`save_with_walk_forward_reuse` or
                :func:`save_with_robustness_reuse` over passing this
                directly — both already implement the provenance check this
                parameter exists to make safe.
            validation_artifacts: Walk-forward and stress-test CSV values to
                write as part of this save. Their checksums are added to the
                metadata only after the atomic file replacements succeed.

        Returns:
            The output directory path.

        Notes:
            A cross-process lock serializes saves to the same destination. The
            in-progress marker additionally distinguishes an interrupted
            multi-file save from a complete bundle.
        """
        out = Path(
            output_directory
            if output_directory is not None
            else GENERATED_REPORTS_DIR / self.config.experiment_name
        )
        artifacts = dict(validation_artifacts or {})
        unsupported = set(artifacts) - _VALIDATION_ARTIFACTS
        if unsupported:
            raise BacktestError(
                "Unsupported validation artefact filename(s): "
                f"{sorted(unsupported)}. Expected only "
                f"{sorted(_VALIDATION_ARTIFACTS)}."
            )
        invalid_values = [
            name
            for name, value in artifacts.items()
            if not isinstance(value, (pd.Series, pd.DataFrame))
        ]
        if invalid_values:
            raise BacktestError(
                "Validation artefacts must be pandas Series or DataFrames; "
                f"invalid value(s): {sorted(invalid_values)}."
            )
        out.mkdir(parents=True, exist_ok=True)
        with _locked_bundle(out):
            return self._save_locked(
                out,
                robustness=robustness,
                keep_artifacts=keep_artifacts,
                validation_artifacts=artifacts,
            )

    def _save_locked(
        self,
        out: Path,
        *,
        robustness: dict[str, Any] | None,
        keep_artifacts: Iterable[str],
        validation_artifacts: Mapping[str, pd.Series | pd.DataFrame],
    ) -> Path:
        """Write a bundle while its cross-process save lock is held."""
        figures_dir = out / "figures"
        if figures_dir.is_symlink():
            raise BacktestError(
                f"Refusing to save figures through symbolic link {figures_dir}."
            )
        if figures_dir.exists() and not figures_dir.is_dir():
            raise BacktestError(
                f"Cannot save figures because {figures_dir} is not a directory."
            )

        # A marker makes an interrupted multi-file save distinguishable from a
        # complete bundle. It is removed only after every owned file is ready.
        save_marker = out / _SAVE_IN_PROGRESS_MARKER
        _write_text_atomic(save_marker, f"pid={os.getpid()}\n")

        # Remove optional artefacts that this run may no longer produce.
        keep = set(keep_artifacts) | set(validation_artifacts)
        for name in _OPTIONAL_ARTIFACTS:
            if name not in keep:
                (out / name).unlink(missing_ok=True)
        # A failed render must not leave an older report looking current.
        (out / "report.html").unlink(missing_ok=True)
        # Remove only QuantLab-managed charts. User files in the same directory
        # are outside this bundle writer's ownership and must be preserved.
        figures_dir.mkdir(parents=True, exist_ok=True)
        from quantlab.reporting.charts import managed_report_figure_filenames

        for filename in managed_report_figure_filenames():
            (figures_dir / filename).unlink(missing_ok=True)

        _write_path_atomic(out / "config.yaml", self.config.to_yaml)
        _write_csv_atomic(self.equity_curve.rename("equity"), out / "equity_curve.csv")
        _write_csv_atomic(self.returns.rename("return"), out / "returns.csv")
        if self.benchmark_returns is not None:
            _write_csv_atomic(
                self.benchmark_returns.rename("benchmark_return"),
                out / "benchmark.csv",
            )
        _write_csv_atomic(self.trades, out / "trades.csv", index=False)
        _write_csv_atomic(self.positions, out / "positions.csv")
        _write_csv_atomic(self.weights, out / "weights.csv")
        _write_csv_atomic(self.costs, out / "costs.csv")
        # Holdout out-of-sample evidence, saved separately so it is
        # reproducible from disk without re-running the backtest.
        if self.holdout_test_returns is not None:
            _write_csv_atomic(
                self.holdout_test_returns.rename("return"),
                out / "holdout_test_returns.csv",
            )
        if self.holdout_test_equity is not None:
            _write_csv_atomic(
                self.holdout_test_equity.rename("equity"),
                out / "holdout_test_equity.csv",
            )

        for name, value in validation_artifacts.items():
            _write_csv_atomic(
                value,
                out / name,
                index=_VALIDATION_ARTIFACT_INDEX[name],
            )
        # Recomputed from whatever _VALIDATION_ARTIFACTS files actually exist
        # in `out` now (freshly written above, or kept as-is via
        # keep_artifacts), never merged with whatever this call started
        # with -- otherwise a save that legitimately drops an artefact (not
        # re-supplied and not kept) would leave a stale checksum in
        # metadata.json for a file the pre-save cleanup above just deleted,
        # even though nothing currently on disk matches it any more.
        self.metadata["walk_forward_csv_checksums"] = {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in _VALIDATION_ARTIFACTS
            if (out / name).is_file()
        }

        # Render once, then reuse the same images on disk and in the HTML.
        self.save_warnings = []
        rendered_figures: dict[str, str] = {}
        try:
            from quantlab.reporting.charts import report_figures

            rendered_figures = report_figures(self, self.save_warnings)
        except Exception as exc:  # pragma: no cover - rendering is optional
            msg = f"Could not render figures: {exc}"
            logger.warning(msg)
            self.save_warnings.append(msg)
        try:
            from quantlab.reporting.charts import save_figures

            save_figures(
                self,
                out / "figures",
                self.save_warnings,
                rendered=rendered_figures,
            )
        except Exception as exc:  # pragma: no cover - rendering is optional
            msg = f"Could not save figures: {exc}"
            logger.warning(msg)
            self.save_warnings.append(msg)
        try:
            # Embedded chart failures append to the same warning collector.
            self.to_html(
                out / "report.html",
                robustness=robustness,
                warnings=self.save_warnings,
                figures=rendered_figures,
            )
        except Exception as exc:  # pragma: no cover - rendering is optional
            msg = f"Could not render HTML report: {exc}"
            logger.warning(msg)
            self.save_warnings.append(msg)

        # Explicit, persisted methodology marker: different CLI commands in
        # walk-forward mode save fundamentally different `self` objects to
        # the same experiment directory -- `quantlab walk-forward`'s own
        # `report` handler saves a full-sample result with OOS evidence
        # only attached as metadata, while `stress-test`/`bootstrap`/
        # `permutation`/`sensitivity`/`robustness` save the OOS-stitched
        # result itself (`wf.oos_result`) -- so metrics.json/metadata.json
        # can mean two different things depending on which command ran
        # last. Recording which one `self.metrics` actually is removes the
        # ambiguity for anyone reading the bundle later, without needing to
        # know the CLI's own save conventions.
        from quantlab.reporting.research_summary import out_of_sample_scope

        self.metadata["result_scope"] = out_of_sample_scope(self) or "full_sample"
        self.metadata["save_warnings"] = self.save_warnings
        _write_text_atomic(
            out / "metrics.json",
            json.dumps(
                _sanitize_non_finite_floats(self.metrics),
                indent=2,
                default=_json_default,
                allow_nan=False,
            ),
        )
        _write_text_atomic(
            out / "metadata.json",
            json.dumps(
                _sanitize_non_finite_floats(self.metadata),
                indent=2,
                default=_json_default,
                allow_nan=False,
            ),
        )

        save_marker.unlink(missing_ok=True)
        logger.info("Saved backtest result to %s", out)
        return out


def _load_metadata_json(metadata_path: Path, *, exp_dir: Path) -> dict[str, Any] | None:
    """Return a prior bundle's parsed ``metadata.json``, or ``None`` to refuse reuse.

    A hand-edited or partially-written ``metadata.json`` must never crash
    reuse detection with a raw ``json.JSONDecodeError`` -- malformed
    metadata is exactly the case reuse must refuse, the same as a missing
    file or a mismatched config/data/code hash.
    """
    try:
        return dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Not reusing prior metadata in %s: %s could not be parsed as "
            "valid JSON (%s).",
            exp_dir,
            metadata_path.name,
            exc,
        )
        return None


def load_previous_walk_forward_robustness(
    exp_dir: Path, result: BacktestResult
) -> dict[str, Any] | None:
    """Load prior walk-forward tables only when their provenance still matches.

    Provenance is config + data_hash + generator_hash + dependency_versions,
    not git_dirty/git_commit -- same reasoning as
    `load_previous_robustness_artifacts`: `generator_hash` already hashes
    current file contents, uncommitted changes included, so it alone gives
    the guarantee needed here. A separate git_dirty/git_commit gate on top
    would be strictly redundant once generator_hash matches, and actively
    wrong: `git status`/`git_dirty` cover the *whole* repository, not just
    the files generator_hash is scoped to, so an unrelated uncommitted
    change elsewhere (docs, configs, tests, ...) would refuse a `report`
    regeneration even though the exact same generator code, config and data
    produced it -- exactly the ordinarily-uncommitted development session
    this reuse mechanism exists to serve. Required CSVs must exist and
    match any recorded checksums; otherwise the function logs why and
    returns ``None``.
    """
    save_marker = exp_dir / _SAVE_IN_PROGRESS_MARKER
    if save_marker.exists() or save_marker.is_symlink():
        logger.warning(
            "Not reusing walk-forward metadata for %s: a prior bundle save "
            "did not complete (%s is still present).",
            exp_dir,
            save_marker.name,
        )
        return None

    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    old_metadata = _load_metadata_json(metadata_path, exp_dir=exp_dir)
    if old_metadata is None or "walk_forward_oos_metrics" not in old_metadata:
        return None

    old_snapshot = old_metadata.get("walk_forward_config_snapshot")
    if old_snapshot is None or old_snapshot != result.config.model_dump(mode="json"):
        logger.warning(
            "Not reusing walk-forward metadata for %s: the config that "
            "produced it no longer matches the config used to regenerate "
            "this report.",
            exp_dir,
        )
        return None

    old_data_hash = old_metadata.get("data_hash")
    new_data_hash = result.metadata.get("data_hash")
    if old_data_hash is None or old_data_hash != new_data_hash:
        logger.warning(
            "Not reusing walk-forward metadata for %s: the data used to "
            "produce it no longer matches the data used to regenerate this "
            "report (data_hash differs), even though the config is "
            "unchanged.",
            exp_dir,
        )
        return None

    # generator_hash, not code_hash: this gates reuse of a *saved bundle*
    # (walk-forward CSVs/metadata), so it must also catch a change to the
    # CLI's own orchestration of how that bundle gets assembled or reused
    # -- code_hash deliberately excludes cli.py (see
    # `quantlab.backtesting.engine._source_hash`'s docstring) and would
    # miss exactly that.
    old_generator_hash = old_metadata.get("generator_hash")
    new_generator_hash = result.metadata.get("generator_hash")
    if old_generator_hash is None or old_generator_hash != new_generator_hash:
        logger.warning(
            "Not reusing walk-forward metadata for %s: the quantlab source "
            "code that produced it no longer matches the source code used "
            "to regenerate this report (generator_hash differs or is "
            "missing), even though the config and data are unchanged.",
            exp_dir,
        )
        return None

    old_deps = old_metadata.get("dependency_versions")
    new_deps = result.metadata.get("dependency_versions")
    if old_deps is None or old_deps != new_deps:
        logger.warning(
            "Not reusing walk-forward metadata for %s: the dependency "
            "versions (numpy/pandas/etc.) used to produce it no longer "
            "match the versions used to regenerate this report, even "
            "though the config and data are unchanged.",
            exp_dir,
        )
        return None

    required = [
        exp_dir / "walk_forward_results.csv",
        exp_dir / "walk_forward_oos_returns.csv",
        exp_dir / "walk_forward_oos_equity.csv",
    ]
    if not all(p.is_file() for p in required):
        logger.warning(
            "Not reusing walk-forward metadata for %s: one or more "
            "walk-forward artefact files are missing.",
            exp_dir,
        )
        return None

    # Checksums detect edits or corruption after the original run. Recorded
    # unconditionally alongside walk_forward_oos_metrics at every save (see
    # save()), so their absence here is itself a red flag -- an incomplete
    # or tampered metadata.json -- not a free pass to skip verification.
    old_checksums = old_metadata.get("walk_forward_csv_checksums")
    if not old_checksums:
        logger.warning(
            "Not reusing walk-forward metadata for %s: no CSV checksums "
            "were recorded, so the required artefacts' integrity can't be "
            "verified before reuse.",
            exp_dir,
        )
        return None
    stress_path = exp_dir / "stress_tests.csv"
    # Stress results are optional but checked when present.
    checksummed = [*required, stress_path] if stress_path.is_file() else required
    # A recorded stress checksum means the file existed originally.
    if stress_path.name in old_checksums and not stress_path.is_file():
        logger.warning(
            "Not reusing walk-forward metadata for %s: stress_tests.csv "
            "was present when its checksum was recorded but is missing "
            "now (the file was deleted since).",
            exp_dir,
        )
        return None
    for path in checksummed:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if old_checksums.get(path.name) != actual:
            logger.warning(
                "Not reusing walk-forward metadata for %s: %s no longer "
                "matches the checksum recorded at walk-forward time "
                "(the file was modified or corrupted on disk since).",
                exp_dir,
                path.name,
            )
            return None

    result.metadata["walk_forward_oos_metrics"] = old_metadata[
        "walk_forward_oos_metrics"
    ]
    # Preserve provenance so later report-only saves can validate it again.
    result.metadata["walk_forward_config_snapshot"] = old_snapshot

    # Preserve the parameter grid and effective windows from the original
    # walk-forward run when regenerating only the report.
    old_parameter_grid = old_metadata.get("walk_forward_parameter_grid")
    if old_parameter_grid is not None:
        result.metadata["walk_forward_parameter_grid"] = old_parameter_grid

    old_windows = old_metadata.get("walk_forward_windows")
    if old_windows is not None:
        result.metadata["walk_forward_windows"] = old_windows

    result.metadata["walk_forward_run_timestamp"] = old_metadata.get(
        "walk_forward_run_timestamp",
        old_metadata.get("run_timestamp"),
    )

    if old_checksums:
        result.metadata["walk_forward_csv_checksums"] = old_checksums
    robustness: dict[str, Any] = {
        "walk_forward": pd.read_csv(exp_dir / "walk_forward_results.csv")
    }
    if stress_path.is_file():
        robustness["stress_tests"] = pd.read_csv(stress_path)
    return robustness


def save_with_walk_forward_reuse(result: BacktestResult, exp_dir: str | Path) -> Path:
    """Save a result while preserving compatible walk-forward artefacts.

    Also preserves compatible bootstrap/permutation-test/sensitivity
    artefacts, exactly like `save_with_robustness_reuse` does for those
    commands' own saves -- otherwise `quantlab report` (which calls this,
    not that) would delete still-valid evidence a `bootstrap`/
    `permutation-test`/`sensitivity` run had just saved, since neither of
    those techniques is itself part of "walk-forward" evidence.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    # Keep the provenance read and the following save under one lock; otherwise
    # another process could replace the validated CSVs between those two steps.
    with _locked_bundle(exp_dir):
        robustness = load_previous_walk_forward_robustness(exp_dir, result)
        keep_artifacts: set[str] = set()
        if robustness is not None:
            keep_artifacts = {
                "walk_forward_results.csv",
                "walk_forward_oos_returns.csv",
                "walk_forward_oos_equity.csv",
            }
            if "stress_tests" in robustness:
                keep_artifacts.add("stress_tests.csv")
        # Sibling on-demand techniques (bootstrap/permutation-test/
        # sensitivity), each saved by its own separate CLI command --
        # `load_previous_walk_forward_robustness` only ever knows about
        # walk-forward's own CSVs plus stress_tests, so a technique it
        # doesn't recognise (e.g. bootstrap_summary.csv) would otherwise be
        # silently deleted by this save's cleanup, even when still valid.
        previous_robustness_artifacts = load_previous_robustness_artifacts(
            exp_dir, result
        )
        # `robustness`'s own keys (from walk-forward reuse above) win on any
        # overlap -- same precedence `save_with_robustness_reuse` gives a
        # freshly-computed technique over a merely-recovered one.
        merged_robustness = {**previous_robustness_artifacts, **(robustness or {})}
        validation_artifacts: dict[str, pd.Series | pd.DataFrame] = {}
        for key, frame in previous_robustness_artifacts.items():
            filename = _ROBUSTNESS_ARTIFACT_FILES[key]
            if filename not in keep_artifacts:
                validation_artifacts[filename] = frame
        return result._save_locked(
            exp_dir,
            robustness=merged_robustness or None,
            keep_artifacts=keep_artifacts,
            validation_artifacts=validation_artifacts,
        )


def load_previous_robustness_artifacts(
    exp_dir: Path, result: BacktestResult
) -> dict[str, Any]:
    """Recover prior stress/bootstrap/permutation/sensitivity artefacts.

    Each on-demand robustness technique is run and saved by its own CLI
    command; without this, running `stress-test` then `bootstrap` against
    the same experiment would silently delete `stress_tests.csv`, since
    `result.save()`'s pre-save cleanup removes any `_OPTIONAL_ARTIFACTS`
    file the current call doesn't re-supply. Applies to holdout/plain
    backtests; walk-forward has its own, longer-lived
    `load_previous_walk_forward_robustness`. Callers merge the result under
    their own freshly computed values, so a stale entry never survives a
    technique actually being recomputed.

    Provenance is config + data_hash + generator_hash + dependency_versions,
    not git_dirty/git_commit (unlike `load_previous_walk_forward_robustness`):
    `generator_hash` already hashes current file contents including
    uncommitted changes, so it alone gives the guarantee needed here,
    without refusing reuse in an ordinarily-uncommitted working session.
    Uses `generator_hash`, not the narrower `code_hash`, because this CSV
    bundle's shape and reuse decision are themselves partly determined by
    CLI orchestration code (`code_hash` deliberately excludes `cli.py`; see
    `quantlab.backtesting.engine._source_hash`'s docstring).

    Returns an empty dict, rather than raising, whenever provenance can't
    be confirmed to still match ``result``.
    """
    save_marker = exp_dir / _SAVE_IN_PROGRESS_MARKER
    if save_marker.exists() or save_marker.is_symlink():
        logger.warning(
            "Not reusing prior robustness artefacts in %s: a prior bundle "
            "save did not complete (%s is still present).",
            exp_dir,
            save_marker.name,
        )
        return {}

    metadata_path = exp_dir / "metadata.json"
    config_path = exp_dir / "config.yaml"
    if not metadata_path.is_file() or not config_path.is_file():
        return {}
    old_metadata = _load_metadata_json(metadata_path, exp_dir=exp_dir)
    if old_metadata is None:
        return {}

    try:
        old_config = ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        logger.warning(
            "Not reusing prior robustness artefacts in %s: %s could not be "
            "parsed as a valid config (%s).",
            exp_dir,
            config_path.name,
            exc,
        )
        return {}
    if old_config.model_dump(mode="json") != result.config.model_dump(mode="json"):
        logger.warning(
            "Not reusing prior robustness artefacts in %s: the config that "
            "produced them no longer matches the config used for this run.",
            exp_dir,
        )
        return {}

    required_matches = (
        ("data_hash", "the data used to produce them no longer matches this run's"),
        (
            "generator_hash",
            "the quantlab source code that produced them no longer matches "
            "this run's (generator_hash differs or is missing)",
        ),
        (
            "dependency_versions",
            "the dependency versions (numpy/pandas/etc.) that produced them "
            "no longer match this run's",
        ),
    )
    for key, reason in required_matches:
        old_value = old_metadata.get(key)
        if old_value is None or old_value != result.metadata.get(key):
            logger.warning(
                "Not reusing prior robustness artefacts in %s: %s.",
                exp_dir,
                reason,
            )
            return {}

    # Checksums detect edits or corruption after the original run. Recorded
    # unconditionally for every validation artefact at save time (see
    # _save_locked) -- these robustness CSVs go through that same path --
    # so their absence here is itself a red flag, not a free pass to skip
    # verification, mirroring load_previous_walk_forward_robustness above.
    old_checksums = old_metadata.get("walk_forward_csv_checksums") or {}
    recovered: dict[str, Any] = {}
    for key, filename in _ROBUSTNESS_ARTIFACT_FILES.items():
        path = exp_dir / filename
        if not path.is_file():
            continue
        recorded = old_checksums.get(filename)
        if recorded is None:
            logger.warning(
                "Not reusing %s from %s: no checksum was recorded for it, "
                "so its integrity can't be verified before reuse.",
                filename,
                exp_dir,
            )
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded != actual:
            logger.warning(
                "Not reusing %s from %s: it no longer matches the checksum "
                "recorded when it was saved (the file was modified or "
                "corrupted on disk since).",
                filename,
                exp_dir,
            )
            continue
        try:
            recovered[key] = pd.read_csv(path)
        except Exception as exc:
            logger.warning(
                "Not reusing %s from %s: the file could not be read (%s).",
                filename,
                exp_dir,
                exc,
            )
            continue
        # Recover this technique's own effective run parameters alongside
        # its CSV -- without this, e.g. bootstrap_summary.csv could survive
        # a later permutation-test save while metadata.json's
        # bootstrap_run_params (n_iterations, block_size, including any CLI
        # override) silently disappears, leaving the surviving numbers with
        # no record of what actually produced them. Never overwrites a key
        # already present: when this technique is the one being freshly
        # recomputed this run, its own fresh run params (set by the CLI
        # command before this call) must win, exactly like `{**previous,
        # **robustness}` already lets a fresh DataFrame win over a
        # recovered one.
        run_params_key = _ROBUSTNESS_RUN_PARAMS_KEYS.get(key)
        if (
            run_params_key is not None
            and run_params_key in old_metadata
            and run_params_key not in result.metadata
        ):
            result.metadata[run_params_key] = old_metadata[run_params_key]
    return recovered


def save_with_robustness_reuse(
    result: BacktestResult,
    exp_dir: str | Path,
    *,
    robustness: dict[str, Any],
    validation_artifacts: Mapping[str, pd.Series | pd.DataFrame] | None = None,
) -> Path:
    """Save a result while preserving compatible sibling robustness artefacts.

    Used by each individual robustness CLI command (stress-test, bootstrap,
    permutation-test, sensitivity) instead of calling `result.save()`
    directly, so that e.g. running `bootstrap` after `stress-test` on the
    same experiment directory keeps the earlier stress-test evidence in the
    regenerated report instead of silently deleting it. ``robustness`` and
    ``validation_artifacts`` are this call's freshly computed results (using
    the same `_ROBUSTNESS_ARTIFACT_FILES`/CSV-filename keys); a technique
    genuinely being recomputed here always overrides whatever a prior save
    left behind for that same technique.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    new_validation_artifacts = (
        dict(validation_artifacts) if validation_artifacts else {}
    )
    # Keep the provenance read and the following save under one lock; otherwise
    # another process could replace the reused CSVs between those two steps.
    with _locked_bundle(exp_dir):
        previous = load_previous_robustness_artifacts(exp_dir, result)
        merged_robustness = {**previous, **robustness}
        merged_validation_artifacts = dict(new_validation_artifacts)
        for key, frame in previous.items():
            filename = _ROBUSTNESS_ARTIFACT_FILES[key]
            if filename not in merged_validation_artifacts:
                merged_validation_artifacts[filename] = frame
        return result._save_locked(
            exp_dir,
            robustness=merged_robustness or None,
            keep_artifacts=set(),
            validation_artifacts=merged_validation_artifacts,
        )


def _json_default(obj: object) -> object:
    """JSON serialiser for numpy/pandas/date scalars."""
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    return str(obj)


def _sanitize_non_finite_floats(value: object) -> object:
    """Recursively replace non-finite Python and NumPy floats with ``None``."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _sanitize_non_finite_floats(value.item())
    if isinstance(value, dict):
        return {k: _sanitize_non_finite_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite_floats(v) for v in value]
    return value
