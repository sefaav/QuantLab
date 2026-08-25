"""Command-line interface for QuantLab.

Commands:

    quantlab download          --config configs/momentum_sp500.yaml
    quantlab backtest          --config configs/momentum_sp500.yaml
    quantlab walk-forward      --config configs/momentum_sp500.yaml
    quantlab stress-test       --config configs/momentum_sp500.yaml
    quantlab bootstrap         --config configs/momentum_sp500.yaml
    quantlab permutation-test  --config configs/momentum_sp500.yaml
    quantlab sensitivity       --config configs/momentum_sp500.yaml
    quantlab robustness        --config configs/momentum_sp500.yaml
    quantlab report            --experiment cross_sectional_momentum_etfs
    quantlab dashboard

stress-test/bootstrap/permutation-test/sensitivity run one technique each,
applying any matching --n-iterations/--block-size/--param-x etc. override;
robustness runs every robustness.* technique enabled in the config in one
pass, with no CLI overrides. All five branch on validation.method: 'holdout'
(or unset) evaluates a plain backtest, 'walk_forward' re-runs the whole
walk-forward selection process (never a plain backtest standing in for it).

Commands display their main pipeline steps and convert expected QuantLab
errors into non-zero process exit codes. Package logs are written to the
configured QuantLab log directory.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

import typer

from quantlab.constants import GENERATED_REPORTS_DIR
from quantlab.exceptions import InsufficientDataError, QuantLabError
from quantlab.logging_config import configure_logging, get_logger
from quantlab.progress import ProgressReporter

if TYPE_CHECKING:
    # Only for type checking: the CLI otherwise lazy-imports heavy modules
    # inside each command so `quantlab --help` stays fast.
    import pandas as pd

    from quantlab.backtesting.result import BacktestResult
    from quantlab.config import ExperimentConfig
    from quantlab.data.validator import DataQualityReport
    from quantlab.validation.walk_forward import WalkForwardResult

app = typer.Typer(
    add_completion=False,
    help="QuantLab — reproducible quantitative research and backtesting.",
    no_args_is_help=True,
)
logger = get_logger(__name__)

_CONFIG_OPTION = typer.Option(
    None, "--config", "-c", help="Path to the experiment YAML config."
)
_SHIPPED_CONFIG_OPTION = typer.Option(
    None,
    "--shipped-config",
    help=(
        "Name of a config bundled with the installed package (e.g. "
        "'demo_offline' or 'demo_offline.yaml') — works even without a full "
        "repo checkout (no local configs/ directory). Mutually exclusive "
        "with --config."
    ),
)


def _echo_step(message: str) -> None:
    """Print a pipeline step to the console and package log.

    The ASCII marker remains compatible with legacy Windows console encodings.
    """
    typer.secho(f"-> {message}", fg=typer.colors.CYAN)
    logger.info(message)


def _make_cli_progress_callback(title: str) -> Callable[[int, int], None] | None:
    """Build an ``on_progress(done, total)`` callback for a live terminal line.

    Text/ETA come from the same `quantlab.progress.ProgressReporter` the
    dashboard uses, so a walk-forward/stress-test/sensitivity run gives a
    consistent, already-tuned estimate on both interfaces. Returns ``None``
    when stdout isn't a terminal (piped/redirected output, CI logs): a
    carriage-return-redrawn line only makes sense on a real tty, and callers
    already treat ``on_progress=None`` as "don't report progress".
    """
    if not sys.stdout.isatty():
        return None
    reporter = ProgressReporter(title)

    def _on_progress(done: int, total: int) -> None:
        text = reporter.text(done, total)
        # \r (not an ANSI clear-line code) plus padding for portability
        # across terminals that don't interpret escape sequences; padding
        # overwrites any leftover characters from a longer previous line.
        sys.stdout.write(f"\r  {text}".ljust(100))
        sys.stdout.flush()
        if total > 0 and done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return _on_progress


def _resolve_config_path(config: Path | None, shipped_config: str | None) -> Path:
    """Resolve exactly one explicit or package-bundled configuration.

    ``--config`` is never silently redirected to a same-named bundled file;
    package configurations require the explicit ``--shipped-config`` option.

    Raises:
        QuantLabError: If neither or both options are provided, or if the
            requested bundled configuration does not exist.
    """
    from quantlab.constants import CONFIGS_DIR

    if (config is None) == (shipped_config is None):
        raise QuantLabError(
            "Exactly one of --config PATH or --shipped-config NAME is required."
        )
    if shipped_config is not None:
        raw_name = shipped_config.strip()
        windows_path = PureWindowsPath(raw_name)
        posix_path = PurePosixPath(raw_name)
        invalid_name = (
            not raw_name
            or raw_name != shipped_config
            or "/" in raw_name
            or "\\" in raw_name
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or posix_path.is_absolute()
            or raw_name in {".", ".."}
        )
        suffix = Path(raw_name).suffix.lower()
        if invalid_name or (suffix and suffix not in {".yaml", ".yml"}):
            raise QuantLabError(
                f"Invalid shipped config name {shipped_config!r}: use a simple "
                "YAML filename such as 'demo_offline' or 'demo_offline.yaml'."
            )
        name = raw_name if suffix else f"{raw_name}.yaml"
        configs_root = CONFIGS_DIR.resolve()
        bundled = (configs_root / name).resolve()
        if bundled.parent != configs_root or not bundled.is_relative_to(configs_root):
            raise QuantLabError(
                f"Invalid shipped config name {shipped_config!r}: the resolved "
                "path must remain directly inside the bundled-config directory."
            )
        if not bundled.is_file():
            available = sorted(
                {
                    path.stem
                    for pattern in ("*.yaml", "*.yml")
                    for path in CONFIGS_DIR.glob(pattern)
                }
            )
            raise QuantLabError(
                f"No shipped config named {shipped_config!r} (looked for "
                f"{bundled}). Available: {available}."
            )
        return bundled
    assert config is not None  # guaranteed by the exactly-one check above
    return config


def _load_config(config_path: Path) -> ExperimentConfig:
    from quantlab.config import ExperimentConfig

    _echo_step(f"Loading config {config_path}")
    return ExperimentConfig.from_yaml(config_path)


def _echo_data_warnings(report: DataQualityReport, *, limit: int = 10) -> None:
    """Print a limited number of data-quality warnings.

    The complete warning list is persisted in the saved report and metadata.
    """
    if not report.warnings:
        return
    limit = max(0, limit)
    typer.secho(f"  data warnings: {len(report.warnings)}", fg=typer.colors.YELLOW)
    for message in report.warnings[:limit]:
        typer.secho(f"    - {message}", fg=typer.colors.YELLOW)
    if len(report.warnings) > limit:
        typer.secho(
            f"    ... and {len(report.warnings) - limit} more "
            "(included in the saved report and metadata).",
            fg=typer.colors.YELLOW,
        )


def _echo_save_outcome(result: BacktestResult, success_message: str) -> None:
    """Print a success message or a partial-rendering warning."""
    if result.save_warnings:
        typer.secho(
            f"[WARN] {success_message} — but with {len(result.save_warnings)} "
            "warning(s) (report/figures may be incomplete or missing; see log):",
            fg=typer.colors.YELLOW,
        )
        for warning in result.save_warnings:
            typer.secho(f"  - {warning}", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"[OK] {success_message}", fg=typer.colors.GREEN)


def _echo_parameter_grid(strategy_name: str, grid: dict[str, list[Any]]) -> None:
    """Describe the effective walk-forward search without overstating it."""
    if grid:
        combinations = 1
        for values in grid.values():
            combinations *= len(values)
        typer.echo(
            f"  parameter grid: {combinations} combination(s) across "
            f"{len(grid)} parameter(s)"
        )
        return
    if strategy_name == "buy_and_hold":
        typer.echo(
            "  parameter grid: 1 combination (buy_and_hold has no "
            "parameters to optimize; using the configured strategy as-is)"
        )
        return
    typer.echo(
        "  parameter grid: 1 combination (no optimization dimensions; "
        "using the configured strategy parameters as-is)"
    )


def _run_active_validation(
    cfg: ExperimentConfig,
    data: pd.DataFrame,
    report: DataQualityReport,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[BacktestResult, WalkForwardResult | None]:
    """Run whichever process ``cfg.validation.method`` names.

    The single place every robustness command resolves "what result / returns
    series applies right now": a plain backtest for 'holdout' (or unset), or
    the full walk-forward process for 'walk_forward' — never a plain backtest
    silently standing in when walk-forward is configured.

    Args:
        cfg: The loaded, validated experiment config.
        data: Canonical long OHLCV frame already loaded for ``cfg``.
        report: Data-quality report from loading ``data``, attached to a
            plain backtest's metadata.
        checkpoint_path: Forwarded to ``WalkForwardValidator.run()`` when the
            walk-forward branch is taken, so an interrupted run resumes
            instead of starting over. Ignored for a plain backtest, which is
            fast enough not to need it.

    Returns:
        ``(result, None)`` for a plain backtest, or ``(wf.oos_result, wf)``
        for walk-forward, so callers can branch on the second element to
        reach the walk-forward-aware variant of a technique.
    """
    if cfg.validation.method == "walk_forward":
        from quantlab.validation.walk_forward import (
            WalkForwardValidator,
            resolve_walk_forward_windows,
        )

        grid = _default_grid(cfg)
        train_window, validation_window, test_window = resolve_walk_forward_windows(cfg)
        _echo_step("Running walk-forward validation")
        _echo_parameter_grid(cfg.strategy_name, grid)
        wf = WalkForwardValidator(cfg).run(
            data,
            parameter_grid=grid,
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            expanding=cfg.validation.expanding,
            on_progress=_make_cli_progress_callback("Walk-forward"),
            checkpoint_path=checkpoint_path,
        )
        if wf.oos_result is None:
            raise InsufficientDataError(
                "No walk-forward folds fit the available history and configured "
                f"windows (train={train_window}, validation={validation_window}, "
                f"test={test_window})."
            )
        # WalkForwardValidator.run() has no data_quality_report parameter of
        # its own, so the saved OOS result would otherwise never carry the
        # data-quality warnings a plain backtest attaches below -- attach it
        # here instead, so a walk-forward report can surface the same
        # gap/frequency-mismatch evidence a plain backtest's report would.
        wf.oos_result.metadata["data_quality"] = report.to_dict()
        return wf.oos_result, wf

    from quantlab.backtesting.runner import run_backtest_from_config

    _echo_step(f"Running backtest '{cfg.experiment_name}'")
    result = run_backtest_from_config(data, cfg, data_quality_report=report)
    return result, None


def _walk_forward_validation_artifacts(
    wf: WalkForwardResult,
) -> dict[str, pd.Series | pd.DataFrame]:
    """CSV artifacts describing a walk-forward run, matching ``walk-forward``."""
    return {
        "walk_forward_results.csv": wf.summary_table(),
        "walk_forward_oos_returns.csv": wf.oos_returns.rename("return"),
        "walk_forward_oos_equity.csv": wf.oos_equity.rename("equity"),
    }


def _attach_walk_forward_evidence(
    result: BacktestResult,
    wf: WalkForwardResult | None,
    validation_artifacts: dict[str, object],
    robustness_extra: dict[str, object],
) -> None:
    """Fold walk-forward fold evidence into a save, when the mode is active."""
    if wf is None:
        return
    validation_artifacts.update(_walk_forward_validation_artifacts(wf))
    robustness_extra["walk_forward"] = wf.summary_table()
    # `result` (and its already-complete `.metrics`, computed via the same
    # full trade-log/benchmark/metrics pipeline as any BacktestResult) *is*
    # `wf.oos_result` at every caller of this function -- reuse it rather
    # than wf.oos_metrics()'s separate M.compute_metrics(oos_returns,
    # oos_equity) recomputation, which only has the stitched return/equity
    # series to work from and so omits benchmark comparisons, gross/net,
    # turnover and every other metric the full pipeline already derived.
    result.metadata["walk_forward_oos_metrics"] = dict(result.metrics)


def _compute_stress_tests(
    data: pd.DataFrame,
    cfg: ExperimentConfig,
    wf: WalkForwardResult | None,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Stress scenarios for the active validation method.

    Walk-forward mode re-runs the whole selection process per scenario
    (:func:`~quantlab.validation.robustness.run_walk_forward_stress_tests`)
    instead of reusing :func:`~quantlab.validation.robustness.run_stress_tests`
    plain-backtest variant — Robustness evidence must never silently come
    from a different validation method than the one currently in effect.
    """
    if wf is not None:
        from quantlab.validation.robustness import run_walk_forward_stress_tests

        return run_walk_forward_stress_tests(
            data, cfg, wf, on_progress=on_progress, checkpoint_path=checkpoint_path
        )
    from quantlab.validation.robustness import run_stress_tests

    return run_stress_tests(
        data, cfg, on_progress=on_progress, checkpoint_path=checkpoint_path
    )


def _compute_bootstrap(
    cfg: ExperimentConfig,
    result: BacktestResult,
    *,
    n_iterations: int | None = None,
    block_size: int | None = None,
) -> pd.DataFrame:
    """Block-bootstrap the active result's returns (mode-agnostic).

    Bootstrap resamples already-realised returns and optimizes nothing, so
    the same function applies unchanged whether ``result.returns`` came from
    a plain backtest or a walk-forward's stitched OOS series.
    """
    from quantlab.validation.bootstrap import bootstrap_returns

    effective_n_iterations = (
        n_iterations
        if n_iterations is not None
        else cfg.robustness.bootstrap.n_iterations
    )
    effective_block_size = (
        block_size if block_size is not None else cfg.robustness.bootstrap.block_size
    )
    boot = bootstrap_returns(
        result.returns,
        n_iterations=effective_n_iterations,
        block_size=effective_block_size,
        seed=cfg.random_seed,
        periods_per_year=cfg.periods_per_year,
        initial_capital=cfg.initial_capital,
        risk_free_rate=cfg.risk_free_rate,
    )
    # Record what was actually used, not just what the YAML says -- a CLI
    # override (--n-iterations/--block-size) would otherwise leave the saved
    # metadata silently describing a different run than the one that
    # actually produced these numbers.
    result.metadata["bootstrap_run_params"] = {
        "n_iterations": effective_n_iterations,
        "block_size": effective_block_size,
    }
    return boot.summary()


def _compute_permutation_test(
    cfg: ExperimentConfig,
    result: BacktestResult,
    *,
    n_iterations: int | None = None,
) -> dict[str, float]:
    """Random-sign Monte Carlo permutation test (mode-agnostic, see bootstrap)."""
    from quantlab.validation.robustness import monte_carlo_permutation

    effective_n_iterations = (
        n_iterations
        if n_iterations is not None
        else cfg.robustness.permutation_test.n_iterations
    )
    outcome = monte_carlo_permutation(
        result.returns,
        n_iterations=effective_n_iterations,
        seed=cfg.random_seed,
        periods_per_year=cfg.periods_per_year,
        risk_free_rate=cfg.risk_free_rate,
    )
    # See _compute_bootstrap: record the effective (possibly CLI-overridden)
    # parameter, not just the YAML default.
    result.metadata["permutation_test_run_params"] = {
        "n_iterations": effective_n_iterations,
    }
    return outcome


def _resolve_sensitivity_axes(
    cfg: ExperimentConfig,
    param_x: str | None,
    values_x: str | None,
    param_y: str | None,
    values_y: str | None,
) -> tuple[str, list[Any], str, list[Any]]:
    """Resolve sensitivity axes: CLI options override the YAML config.

    Raises:
        QuantLabError: If only some of the 4 CLI options are given, or if
            none are given and ``robustness.sensitivity.parameters`` is unset
            — sensitivity has no default grid, unlike walk-forward.
    """
    from quantlab.validation.parameter_grid import parse_parameter_grid_values

    cli_given = (param_x, values_x, param_y, values_y)
    if any(v is not None for v in cli_given):
        if param_x is None or values_x is None or param_y is None or values_y is None:
            raise QuantLabError(
                "--param-x, --values-x, --param-y and --values-y must all be "
                "given together, or all omitted to use "
                "robustness.sensitivity.parameters from the config."
            )
        return (
            param_x,
            parse_parameter_grid_values(values_x),
            param_y,
            parse_parameter_grid_values(values_y),
        )
    configured = cfg.robustness.sensitivity.parameters
    if configured is None:
        raise QuantLabError(
            "No sensitivity axes given: pass --param-x/--values-x/--param-y/"
            "--values-y, or set robustness.sensitivity.parameters (exactly 2 "
            "keys) in the config."
        )
    (x_name, x_values), (y_name, y_values) = list(configured.items())
    return x_name, x_values, y_name, y_values


def _compute_sensitivity(
    data: pd.DataFrame,
    cfg: ExperimentConfig,
    wf: WalkForwardResult | None,
    parameter_x: str,
    values_x: list[Any],
    parameter_y: str,
    values_y: list[Any],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Two-parameter sensitivity sweep for the active validation method.

    Walk-forward mode re-runs the whole selection process per grid cell
    (:func:`~quantlab.validation.parameter_sensitivity.
    run_walk_forward_parameter_sensitivity`) instead of the plain
    single-backtest variant, for the same reason as stress tests above.
    ``on_progress``/``checkpoint_path`` only apply to that walk-forward
    variant — the plain one has no ``on_progress`` either (already judged
    fast enough).
    """
    if wf is not None:
        from quantlab.validation.parameter_sensitivity import (
            run_walk_forward_parameter_sensitivity,
        )

        return run_walk_forward_parameter_sensitivity(
            data,
            cfg,
            parameter_x,
            values_x,
            parameter_y,
            values_y,
            on_progress=on_progress,
            checkpoint_path=checkpoint_path,
        )
    from quantlab.validation.parameter_sensitivity import run_parameter_sensitivity

    return run_parameter_sensitivity(
        data, cfg, parameter_x, values_x, parameter_y, values_y
    )


@app.command()
def download(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    force: bool = typer.Option(
        False,
        "--force",
        help="Ignore the existing cache and download remote data again.",
    ),
) -> None:
    """Acquire the market data required by an experiment.

    Yahoo and Binance data are downloaded when necessary and stored in the
    local Parquet cache. CSV data are loaded from local files and are not
    written to that cache.

    A separately configured symbol benchmark is acquired as well when it is
    not already part of the tradable universe. Use ``--force`` to refresh
    remote data even when the existing cache already covers the requested
    period.
    """
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))

        from quantlab.data.loader import DataLoader

        symbols = list(cfg.symbols)
        if (
            str(cfg.benchmark_kind) == "symbol"
            and cfg.benchmark_symbol
            and cfg.benchmark_symbol not in symbols
        ):
            symbols.append(cfg.benchmark_symbol)

        source = str(cfg.data_source)

        if source == "csv":
            _echo_step(f"Loading {len(symbols)} symbol(s) from local CSV files")
        elif force:
            _echo_step(
                f"Refreshing {len(symbols)} symbol(s) from {source} "
                "(ignoring the existing cache)"
            )
        else:
            _echo_step(
                f"Acquiring {len(symbols)} symbol(s) from {source} "
                "(using the cache when available)"
            )

        data = DataLoader().download(cfg, force=force)
        symbol_count = int(data["symbol"].nunique())

        if source == "csv":
            typer.secho(
                f"[OK] Loaded {len(data)} rows for "
                f"{symbol_count} symbol(s) from CSV files.",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"[OK] Data available for {symbol_count} symbol(s) "
                f"({len(data)} rows returned).",
                fg=typer.colors.GREEN,
            )

    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def backtest(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output directory. By default, uses QuantLab's generated-reports "
            "directory (reports/generated/ in a checkout; ~/.quantlab after a "
            "regular installation)."
        ),
    ),
) -> None:
    """Run a backtest and save its artefact bundle.

    This command does not compute walk-forward or stress-test artefacts. Saving
    over an existing experiment directory removes those earlier optional files.
    """
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        from quantlab.backtesting.runner import run_backtest_from_config
        from quantlab.data.loader import DataLoader

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        _echo_step(f"Running backtest '{cfg.experiment_name}'")
        result = run_backtest_from_config(data, cfg, data_quality_report=report)

        _echo_step("Saving results")
        out_dir = result.save(output)

        typer.echo("")
        typer.echo(result.summary())
        typer.echo("")
        _echo_save_outcome(result, f"Saved to {out_dir}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="walk-forward")
def walk_forward(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard any existing checkpoint and start over.",
    ),
) -> None:
    """Run walk-forward validation and robustness tests."""
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        from quantlab.backtesting.runner import run_backtest_from_config
        from quantlab.data.loader import DataLoader
        from quantlab.validation.checkpoint import clear_checkpoint
        from quantlab.validation.robustness import stress_test_checkpoint_paths
        from quantlab.validation.walk_forward import (
            WalkForwardValidator,
            resolve_walk_forward_windows,
        )

        out = GENERATED_REPORTS_DIR / cfg.experiment_name
        wf_checkpoint = out / ".checkpoint_walk_forward.pkl"
        stress_checkpoint = out / ".checkpoint_stress_test.pkl"
        if fresh:
            clear_checkpoint(wf_checkpoint)
            for path in stress_test_checkpoint_paths(stress_checkpoint):
                clear_checkpoint(path)

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        _echo_step("Running walk-forward validation")
        grid = _default_grid(cfg)
        _echo_parameter_grid(cfg.strategy_name, grid)
        validator = WalkForwardValidator(cfg)
        train_window, validation_window, test_window = resolve_walk_forward_windows(cfg)
        if not (
            cfg.validation.train_window
            and cfg.validation.validation_window
            and cfg.validation.test_window
        ):
            # Apply the documented CLI defaults and make every fallback visible.
            typer.secho(
                "  using default window(s) "
                f"(train={train_window}, validation={validation_window}, "
                f"test={test_window}) — set validation.train_window / "
                "validation_window / test_window explicitly to override.",
                fg=typer.colors.YELLOW,
            )
        wf = validator.run(
            data,
            parameter_grid=grid,
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            expanding=cfg.validation.expanding,
            on_progress=_make_cli_progress_callback("Walk-forward"),
            checkpoint_path=wf_checkpoint,
        )

        if not wf.folds:
            raise InsufficientDataError(
                "No walk-forward folds fit the available history and configured "
                f"windows (train={train_window}, validation={validation_window}, "
                f"test={test_window})."
            )

        _echo_step("Running stress tests")
        # Walk-forward-OOS-aware, matching the dedicated `stress-test` command
        # and the dashboard for this same config -- never a plain full-sample
        # re-run of the scenarios, which would silently mix a different
        # validation methodology into the same "walk_forward" evidence bundle
        # (see _compute_stress_tests). wf.folds non-empty is not itself a
        # guarantee wf.oos_result exists (see its docstring: None only when
        # no fold produced any OOS weights) -- run_walk_forward_stress_tests
        # requires it, so fall back to the full-sample scenarios in that
        # edge case rather than fail the whole command, the same graceful
        # degradation the OOS-metrics fallback just below already applies.
        stress = _compute_stress_tests(
            data,
            cfg,
            wf if wf.oos_result is not None else None,
            on_progress=_make_cli_progress_callback("Stress tests"),
            checkpoint_path=stress_checkpoint,
        )

        # Attach OOS metrics to a fresh full-sample result before saving one bundle.
        result = run_backtest_from_config(data, cfg, data_quality_report=report)
        # Reuse wf.oos_result.metrics (built via the same full trade-log/
        # benchmark/metrics pipeline as `result` itself) rather than
        # wf.oos_metrics()'s separate, less complete M.compute_metrics(
        # oos_returns, oos_equity) recomputation -- `wf.folds` non-empty
        # (checked above) is not itself a guarantee oos_result exists (see
        # its docstring), so still fall back for that edge case.
        oos = (
            dict(wf.oos_result.metrics)
            if wf.oos_result is not None
            else wf.oos_metrics(cfg.periods_per_year, cfg.risk_free_rate)
        )
        result.metadata["walk_forward_oos_metrics"] = oos
        result.metadata["walk_forward_parameter_grid"] = grid
        result.metadata["walk_forward_windows"] = {
            "train_window": train_window,
            "validation_window": validation_window,
            "test_window": test_window,
            "expanding": cfg.validation.expanding,
        }
        # Capture the walk-forward configuration so later report regeneration can
        # reject OOS artefacts produced by a different configuration.
        result.metadata["walk_forward_config_snapshot"] = cfg.model_dump(mode="json")
        # Preserve the original walk-forward timestamp separately from timestamps
        # created by later report-only regenerations.
        result.metadata["walk_forward_run_timestamp"] = result.metadata["run_timestamp"]
        # Save the numerical bundle and its validation artefacts under one marker
        # and cross-process lock. BacktestResult computes checksums only after each
        # CSV has been atomically replaced.
        result.save(
            out,
            robustness={
                "walk_forward": wf.summary_table(),
                "stress_tests": stress,
            },
            validation_artifacts={
                "walk_forward_results.csv": wf.summary_table(),
                "walk_forward_oos_returns.csv": wf.oos_returns.rename("return"),
                "walk_forward_oos_equity.csv": wf.oos_equity.rename("equity"),
                "stress_tests.csv": stress,
            },
        )

        _echo_save_outcome(
            result, f"Walk-forward done ({len(wf.folds)} folds). Saved to {out}"
        )
        typer.echo(
            f"  OOS Sharpe: {oos.get('sharpe_ratio', 0):.2f} | "
            f"OOS CAGR: {oos.get('cagr', 0):.2%}"
        )
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="stress-test")
def stress_test(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard any existing checkpoint and start over.",
    ),
) -> None:
    """Run stress-test scenarios (commission/slippage/delay/reduced universe).

    Re-runs the whole walk-forward process per scenario when
    validation.method is 'walk_forward', instead of a single backtest — see
    `quantlab robustness --help` for why.
    """
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        from quantlab.data.loader import DataLoader
        from quantlab.validation.checkpoint import clear_checkpoint
        from quantlab.validation.robustness import stress_test_checkpoint_paths

        out = GENERATED_REPORTS_DIR / cfg.experiment_name
        wf_checkpoint = out / ".checkpoint_walk_forward.pkl"
        stress_checkpoint = out / ".checkpoint_stress_test.pkl"
        if fresh:
            clear_checkpoint(wf_checkpoint)
            for path in stress_test_checkpoint_paths(stress_checkpoint):
                clear_checkpoint(path)

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        result, wf = _run_active_validation(
            cfg, data, report, checkpoint_path=wf_checkpoint
        )
        _echo_step("Running stress tests")
        stress = _compute_stress_tests(
            data,
            cfg,
            wf,
            on_progress=_make_cli_progress_callback("Stress tests"),
            checkpoint_path=stress_checkpoint,
        )

        validation_artifacts: dict[str, Any] = {"stress_tests.csv": stress}
        robustness_extra: dict[str, Any] = {"stress_tests": stress}
        _attach_walk_forward_evidence(
            result, wf, validation_artifacts, robustness_extra
        )

        _echo_step("Saving results")
        from quantlab.backtesting.result import save_with_robustness_reuse

        out_dir = save_with_robustness_reuse(
            result,
            out,
            robustness=robustness_extra,
            validation_artifacts=validation_artifacts,
        )
        typer.echo("")
        typer.echo(stress.to_string(index=False))
        typer.echo("")
        _echo_save_outcome(result, f"Saved to {out_dir}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def bootstrap(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    n_iterations: int | None = typer.Option(
        None,
        "--n-iterations",
        min=1,
        help="Override robustness.bootstrap.n_iterations from the config.",
    ),
    block_size: int | None = typer.Option(
        None,
        "--block-size",
        min=1,
        help="Override robustness.bootstrap.block_size from the config.",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard any existing walk-forward checkpoint and start over.",
    ),
) -> None:
    """Block-bootstrap the active result's returns (backtest or walk-forward OOS)."""
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        from quantlab.data.loader import DataLoader
        from quantlab.validation.checkpoint import clear_checkpoint

        wf_checkpoint = (
            GENERATED_REPORTS_DIR / cfg.experiment_name / ".checkpoint_walk_forward.pkl"
        )
        if fresh:
            clear_checkpoint(wf_checkpoint)

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        result, wf = _run_active_validation(
            cfg, data, report, checkpoint_path=wf_checkpoint
        )
        _echo_step("Running bootstrap")
        summary = _compute_bootstrap(
            cfg, result, n_iterations=n_iterations, block_size=block_size
        )

        validation_artifacts: dict[str, Any] = {"bootstrap_summary.csv": summary}
        robustness_extra: dict[str, Any] = {"bootstrap": summary}
        _attach_walk_forward_evidence(
            result, wf, validation_artifacts, robustness_extra
        )

        _echo_step("Saving results")
        from quantlab.backtesting.result import save_with_robustness_reuse

        out_dir = save_with_robustness_reuse(
            result,
            GENERATED_REPORTS_DIR / cfg.experiment_name,
            robustness=robustness_extra,
            validation_artifacts=validation_artifacts,
        )
        typer.echo("")
        typer.echo(summary.to_string(index=False))
        typer.echo("")
        _echo_save_outcome(result, f"Saved to {out_dir}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="permutation-test")
def permutation_test(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    n_iterations: int | None = typer.Option(
        None,
        "--n-iterations",
        min=1,
        help="Override robustness.permutation_test.n_iterations from the config.",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard any existing walk-forward checkpoint and start over.",
    ),
) -> None:
    """Random-sign Monte Carlo permutation test on the active result's returns."""
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        from quantlab.data.loader import DataLoader
        from quantlab.validation.checkpoint import clear_checkpoint

        wf_checkpoint = (
            GENERATED_REPORTS_DIR / cfg.experiment_name / ".checkpoint_walk_forward.pkl"
        )
        if fresh:
            clear_checkpoint(wf_checkpoint)

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        result, wf = _run_active_validation(
            cfg, data, report, checkpoint_path=wf_checkpoint
        )
        _echo_step("Running Monte Carlo permutation test")
        outcome = _compute_permutation_test(cfg, result, n_iterations=n_iterations)
        import pandas as pd

        summary = pd.DataFrame([outcome])

        validation_artifacts: dict[str, Any] = {"permutation_test.csv": summary}
        robustness_extra: dict[str, Any] = {"permutation_test": summary}
        _attach_walk_forward_evidence(
            result, wf, validation_artifacts, robustness_extra
        )

        _echo_step("Saving results")
        from quantlab.backtesting.result import save_with_robustness_reuse

        out_dir = save_with_robustness_reuse(
            result,
            GENERATED_REPORTS_DIR / cfg.experiment_name,
            robustness=robustness_extra,
            validation_artifacts=validation_artifacts,
        )
        typer.echo("")
        typer.echo(f"  real Sharpe : {outcome['real_sharpe']:.2f}")
        typer.echo(f"  p-value     : {outcome['p_value']:.4f}")
        typer.echo("")
        _echo_save_outcome(result, f"Saved to {out_dir}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def sensitivity(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    param_x: str | None = typer.Option(
        None, "--param-x", help="First swept strategy parameter."
    ),
    values_x: str | None = typer.Option(
        None, "--values-x", help="Comma-separated candidate values for --param-x."
    ),
    param_y: str | None = typer.Option(
        None, "--param-y", help="Second swept strategy parameter."
    ),
    values_y: str | None = typer.Option(
        None, "--values-y", help="Comma-separated candidate values for --param-y."
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard any existing checkpoint and start over.",
    ),
) -> None:
    """Two-parameter sensitivity sweep, scored on the active validation method.

    Give all four of --param-x/--values-x/--param-y/--values-y together to
    override robustness.sensitivity.parameters from the config, or omit all
    four to use that config section (there is no default grid — at least
    one source of axes is required).
    """
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        x_name, x_values, y_name, y_values = _resolve_sensitivity_axes(
            cfg, param_x, values_x, param_y, values_y
        )
        from quantlab.data.loader import DataLoader
        from quantlab.validation.checkpoint import clear_checkpoint

        out = GENERATED_REPORTS_DIR / cfg.experiment_name
        wf_checkpoint = out / ".checkpoint_walk_forward.pkl"
        sensitivity_checkpoint = out / ".checkpoint_sensitivity.pkl"
        if fresh:
            clear_checkpoint(wf_checkpoint)
            clear_checkpoint(sensitivity_checkpoint)

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        result, wf = _run_active_validation(
            cfg, data, report, checkpoint_path=wf_checkpoint
        )
        _echo_step(
            f"Running parameter sensitivity ({x_name} x {y_name}, "
            f"{len(x_values) * len(y_values)} combinations)"
        )
        sens = _compute_sensitivity(
            data,
            cfg,
            wf,
            x_name,
            x_values,
            y_name,
            y_values,
            on_progress=_make_cli_progress_callback("Sensitivity"),
            checkpoint_path=sensitivity_checkpoint,
        )

        # Record the axes actually used, not just that a sweep ran -- these
        # may have come from a CLI override rather than
        # robustness.sensitivity.parameters in the saved config.yaml.
        result.metadata["sensitivity_run_params"] = {
            "parameter_x": x_name,
            "values_x": x_values,
            "parameter_y": y_name,
            "values_y": y_values,
        }

        validation_artifacts: dict[str, Any] = {"sensitivity.csv": sens}
        robustness_extra: dict[str, Any] = {"sensitivity": sens}
        _attach_walk_forward_evidence(
            result, wf, validation_artifacts, robustness_extra
        )

        _echo_step("Saving results")
        from quantlab.backtesting.result import save_with_robustness_reuse

        out_dir = save_with_robustness_reuse(
            result,
            out,
            robustness=robustness_extra,
            validation_artifacts=validation_artifacts,
        )
        typer.echo("")
        typer.echo(sens.to_string(index=False))
        typer.echo("")
        _echo_save_outcome(result, f"Saved to {out_dir}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def robustness(
    config: Path | None = _CONFIG_OPTION,
    shipped_config: str | None = _SHIPPED_CONFIG_OPTION,
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Discard any existing checkpoint and start over.",
    ),
) -> None:
    """Run every robustness.* technique enabled in the config, in one pass.

    Reads only the YAML config (no per-technique CLI overrides — use the
    dedicated stress-test/bootstrap/permutation-test/sensitivity commands for
    that). Each technique branches on validation.method exactly like its
    dedicated command, calling the identical underlying function.
    """
    configure_logging()
    try:
        cfg = _load_config(_resolve_config_path(config, shipped_config))
        from quantlab.data.loader import DataLoader
        from quantlab.validation.checkpoint import clear_checkpoint
        from quantlab.validation.robustness import stress_test_checkpoint_paths

        out = GENERATED_REPORTS_DIR / cfg.experiment_name
        wf_checkpoint = out / ".checkpoint_walk_forward.pkl"
        stress_checkpoint = out / ".checkpoint_stress_test.pkl"
        sensitivity_checkpoint = out / ".checkpoint_sensitivity.pkl"
        if fresh:
            clear_checkpoint(wf_checkpoint)
            for path in stress_test_checkpoint_paths(stress_checkpoint):
                clear_checkpoint(path)
            clear_checkpoint(sensitivity_checkpoint)

        _echo_step("Loading and validating data")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)

        result, wf = _run_active_validation(
            cfg, data, report, checkpoint_path=wf_checkpoint
        )
        validation_artifacts: dict[str, Any] = {}
        robustness_extra: dict[str, Any] = {}
        _attach_walk_forward_evidence(
            result, wf, validation_artifacts, robustness_extra
        )

        ran_any = False
        if cfg.robustness.stress_test.enabled:
            ran_any = True
            _echo_step("Running stress tests")
            stress = _compute_stress_tests(
                data,
                cfg,
                wf,
                on_progress=_make_cli_progress_callback("Stress tests"),
                checkpoint_path=stress_checkpoint,
            )
            validation_artifacts["stress_tests.csv"] = stress
            robustness_extra["stress_tests"] = stress
            typer.echo("")
            typer.echo(stress.to_string(index=False))

        if cfg.robustness.bootstrap.enabled:
            ran_any = True
            _echo_step("Running bootstrap")
            boot_summary = _compute_bootstrap(cfg, result)
            validation_artifacts["bootstrap_summary.csv"] = boot_summary
            robustness_extra["bootstrap"] = boot_summary
            typer.echo("")
            typer.echo(boot_summary.to_string(index=False))

        if cfg.robustness.permutation_test.enabled:
            ran_any = True
            _echo_step("Running Monte Carlo permutation test")
            outcome = _compute_permutation_test(cfg, result)
            import pandas as pd

            permutation_summary = pd.DataFrame([outcome])
            validation_artifacts["permutation_test.csv"] = permutation_summary
            robustness_extra["permutation_test"] = permutation_summary
            typer.echo("")
            typer.echo(f"  real Sharpe : {outcome['real_sharpe']:.2f}")
            typer.echo(f"  p-value     : {outcome['p_value']:.4f}")

        if cfg.robustness.sensitivity.enabled:
            ran_any = True
            parameters = cfg.robustness.sensitivity.parameters
            if parameters is None:
                raise QuantLabError(
                    "robustness.sensitivity.enabled is true but "
                    "robustness.sensitivity.parameters is not set."
                )
            (x_name, x_values), (y_name, y_values) = list(parameters.items())
            _echo_step(f"Running parameter sensitivity ({x_name} x {y_name})")
            sens = _compute_sensitivity(
                data,
                cfg,
                wf,
                x_name,
                x_values,
                y_name,
                y_values,
                on_progress=_make_cli_progress_callback("Sensitivity"),
                checkpoint_path=sensitivity_checkpoint,
            )
            validation_artifacts["sensitivity.csv"] = sens
            robustness_extra["sensitivity"] = sens
            typer.echo("")
            typer.echo(sens.to_string(index=False))

        if not ran_any:
            typer.secho(
                "  no robustness.* technique is enabled in this config — "
                "nothing to run. Set e.g. robustness.bootstrap.enabled: true.",
                fg=typer.colors.YELLOW,
            )

        _echo_step("Saving results")
        from quantlab.backtesting.result import save_with_robustness_reuse

        out_dir = save_with_robustness_reuse(
            result,
            out,
            robustness=robustness_extra,
            validation_artifacts=validation_artifacts,
        )
        typer.echo("")
        _echo_save_outcome(result, f"Saved to {out_dir}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def report(
    experiment: str = typer.Option(
        ...,
        "--experiment",
        "-e",
        help=(
            "Regenerate a report from a previously saved experiment. If none exists, "
            "run a bundled QuantLab config with the same experiment name."
        ),
    ),
) -> None:
    """Regenerate a report from a saved experiment or matching bundled config."""
    configure_logging()
    try:
        reports_root = GENERATED_REPORTS_DIR.resolve()
        exp_dir = (GENERATED_REPORTS_DIR / experiment).resolve()
        if not exp_dir.is_relative_to(reports_root):
            raise QuantLabError(
                f"Invalid --experiment {experiment!r}: must not escape the "
                f"generated-reports directory ({GENERATED_REPORTS_DIR})."
            )
        config_path = exp_dir / "config.yaml"
        if not config_path.is_file():
            # Fall back to a shipped config of the same name.
            from quantlab.constants import CONFIGS_DIR

            candidates = sorted(
                path
                for pattern in ("*.yaml", "*.yml")
                for path in CONFIGS_DIR.glob(pattern)
            )
            for candidate in candidates:
                from quantlab.config import ExperimentConfig

                if ExperimentConfig.from_yaml(candidate).experiment_name == experiment:
                    config_path = candidate
                    break
        if not config_path.is_file():
            raise QuantLabError(
                f"No saved or bundled config found for experiment {experiment!r}. "
                "Run `quantlab backtest` first or check the experiment name."
            )
        cfg = _load_config(config_path)
        if cfg.experiment_name != experiment:
            raise QuantLabError(
                f"Config {config_path} declares experiment_name="
                f"{cfg.experiment_name!r}, but --experiment was {experiment!r}."
            )
        from quantlab.backtesting.result import save_with_walk_forward_reuse
        from quantlab.backtesting.runner import run_backtest_from_config
        from quantlab.data.loader import DataLoader

        _echo_step("Reloading data and re-running for the report")
        data, report = DataLoader().load(cfg)
        _echo_data_warnings(report)
        result = run_backtest_from_config(data, cfg, data_quality_report=report)

        # A report-only run does not recompute walk-forward validation.
        # Reuse earlier OOS artefacts only when their provenance checks pass.
        out = save_with_walk_forward_reuse(result, exp_dir)
        _echo_save_outcome(result, f"Report at {out / 'report.html'}")
    except QuantLabError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    from importlib.util import find_spec

    configure_logging()

    if find_spec("streamlit") is None:
        typer.secho(
            "[ERROR] Streamlit is not installed. Install it with: "
            'python -m pip install -e ".[dashboard]"',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    if not app_path.is_file():
        typer.secho(
            f"[ERROR] Dashboard entry point not found: {app_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    _echo_step(f"Launching Streamlit dashboard: {app_path}")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.secho(
            f"[ERROR] Streamlit exited with status {exc.returncode}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=exc.returncode) from exc


def _default_grid(cfg: ExperimentConfig) -> dict[str, list[Any]]:
    """Return the configured or default grid shared by research interfaces."""
    from quantlab.validation.parameter_grid import parameter_grid_for_config

    return parameter_grid_for_config(cfg)


if __name__ == "__main__":  # pragma: no cover
    app()
