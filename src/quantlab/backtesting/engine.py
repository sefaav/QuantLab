"""Backtest engine.

Orchestrates the exact ordered pipeline:

1-2. load & validate data (done by the caller / DataLoader)
3.   compute features (inside the strategy)
4.   generate signals at t
5.   transform signals into target weights
6.   apply constraints
7.   shift weights before computing returns  ← look-ahead barrier
8.   compute turnover
9.   compute gross returns
10.  compute costs
11.  subtract costs
12.  update portfolio value
13.  record trades
14.  compute metrics
15.  produce the result

Signal → allocation → execution → accounting are distinct, independently
tested steps, kept separate rather than mixed together so each stage's
correctness can be verified on its own.
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import time
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata
from numbers import Integral
from pathlib import Path

import pandas as pd

from quantlab.backtesting.accounting import (
    compute_asset_returns,
    portfolio_metrics_from_accounting,
    run_accounting,
)
from quantlab.backtesting.benchmark import build_benchmark
from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.trade_log import build_trade_log
from quantlab.config import BenchmarkKind, ExperimentConfig
from quantlab.constants import (
    CALENDAR_DAYS_PER_YEAR,
    CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR,
    FREQUENCY_TO_PERIODS_PER_YEAR,
    SYMBOL,
    TIMESTAMP,
    TRADING_DAYS_PER_YEAR,
)
from quantlab.data.base import price_matrix, volume_matrix
from quantlab.data.calendar import is_247, uniform_calendar
from quantlab.data.closures import DAILY_FREQUENCY, tradable_mask_for
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import BacktestError, QuantLabError
from quantlab.execution.execution_model import ExecutionModel
from quantlab.execution.orders import (
    shift_respecting_tradability,
    validate_execution_frame,
)
from quantlab.execution.slippage import (
    ConstantSlippageModel,
    SlippageModel,
    VolumeBasedSlippageModel,
)
from quantlab.logging_config import get_logger
from quantlab.portfolio.allocator import PortfolioAllocator, build_allocator
from quantlab.portfolio.constraints import constraints_from_config
from quantlab.portfolio.rebalancing import rebalance_and_cap_turnover
from quantlab.portfolio.volatility_targeting import apply_volatility_target
from quantlab.risk.metrics import compute_metrics
from quantlab.strategies.base import (
    BaseStrategy,
    build_strategy,
    strategy_parameter_names,
)

logger = get_logger(__name__)

#: Numerical dependencies recorded in result metadata for reproducibility.
_TRACKED_PACKAGES = (
    "quantlab",
    "pandas",
    "numpy",
    "pydantic",
    "scipy",
    "statsmodels",
    # Directly drives calendar/session/settlement results (holidays,
    # sessions, closures) -- a version bump can change backtest output the
    # same way a pandas/numpy bump can, so it belongs in provenance too.
    "pandas-market-calendars",
)


def _git_commit_hash() -> str | None:
    """Return the current Git commit hash, or ``None`` when unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _git_is_dirty() -> bool | None:
    """Return whether tracked files differ from HEAD, or ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        return bool(completed.stdout.strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _dependency_versions() -> dict[str, str]:
    """Installed versions of quantlab and its key numerical dependencies."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


#: Cached hashes and the fingerprints used to validate each, keyed by which
#: `_hash_source_tree()` caller computed them ("source" / "generator").
_source_hash_cache: dict[str, tuple[tuple[tuple[str, int], ...], str, float]] = {}

#: Maximum cache lifetime when file mtimes appear unchanged.
_SOURCE_HASH_TTL_SECONDS = 60.0

#: Never part of either hash: the dashboard UI is imported by neither a
#: notebook cell, the computational backtest path, nor CLI orchestration.
_DASHBOARD_TOP_LEVEL_PART = "dashboard"


def _hash_source_tree(
    *, cache_key: str, excluded_files: frozenset[str] = frozenset()
) -> str:
    r"""Return a SHA-256 hash of a scoped subset of QuantLab's Python sources.

    Always excludes `dashboard/` (see `_DASHBOARD_TOP_LEVEL_PART`); callers
    additionally exclude specific files via ``excluded_files``. POSIX
    relative paths keep the hash platform-independent. A path/mtime
    fingerprint avoids rereading unchanged files, while a short TTL bounds
    staleness when a synchronisation tool preserves mtimes. ``cache_key``
    keeps this cache and the differently-scoped one(s) other callers use
    from colliding.
    """
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for path in root.rglob("*.py")
        if path.relative_to(root).parts[0] != _DASHBOARD_TOP_LEVEL_PART
        and path.relative_to(root).name not in excluded_files
    )
    fingerprint = tuple(
        (path.relative_to(root).as_posix(), path.stat().st_mtime_ns) for path in paths
    )
    now = time.monotonic()
    cached = _source_hash_cache.get(cache_key)
    if (
        cached is not None
        and now - cached[2] < _SOURCE_HASH_TTL_SECONDS
        and fingerprint == cached[0]
    ):
        return cached[1]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    value = digest.hexdigest()
    _source_hash_cache[cache_key] = (fingerprint, value, now)
    return value


def _source_hash() -> str:
    """Return a hash scoped to the modules that affect *computed results*.

    Excludes `dashboard/` and `cli.py` -- editing either changes nothing a
    notebook or a plain backtest computes, so this must stay stable across
    such edits: recorded as ``code_hash`` for informational/reproducibility
    purposes, and compared against a notebook's own stored ``code_hash`` to
    decide whether it needs rebuilding (see ``scripts/build_notebooks.py``,
    ``tests/unit/test_notebooks.py``). Never use this to gate reuse of a
    saved *artifact bundle* (walk-forward CSVs, robustness CSVs, a
    checkpoint) -- that's what `_generator_hash()` is for.
    """
    return _hash_source_tree(cache_key="source", excluded_files=frozenset({"cli.py"}))


#: Outside src/quantlab/ entirely (never reached by `_hash_source_tree`'s own
#: rglob), but this script makes the exact same save/reuse decision as
#: `cli.py` (see its own module docstring: "Compatible walk-forward
#: artefacts from an earlier run are reused only when ... provenance checks
#: still pass") -- must be covered by `_generator_hash()` for the same
#: reason `cli.py` itself is.
_GENERATOR_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "generate_report.py"
)


def _generator_hash() -> str:
    """Return a hash scoped to everything that can affect a *saved bundle*.

    Unlike `_source_hash()`, includes `cli.py`: the CLI orchestrates how a
    computed result becomes the files on disk (which CSVs get written, how
    they're assembled, how reuse itself is decided) -- an unchanged
    `_source_hash()` does not guarantee an unchanged bundle if only that
    orchestration logic changed. Used wherever a decision reuses or resumes
    a previously *saved* artifact rather than merely reporting provenance:
    `quantlab.backtesting.result.load_previous_walk_forward_robustness`,
    `load_previous_robustness_artifacts`, and
    `quantlab.validation.checkpoint.compute_provenance`.

    Also folds in `_GENERATOR_SCRIPT`'s own content: it lives outside
    `src/quantlab/` entirely, so `_hash_source_tree`'s scan can never reach
    it on its own. Silently omitted (not an error) when absent -- e.g. an
    installed wheel that doesn't bundle dev-only `scripts/`.
    """
    tree_hash = _hash_source_tree(cache_key="generator")
    digest = hashlib.sha256(tree_hash.encode("utf-8"))
    if _GENERATOR_SCRIPT.is_file():
        digest.update(_GENERATOR_SCRIPT.read_bytes())
    return digest.hexdigest()


def _qualified_class_name(instance: object) -> str:
    """Exact class identity (module + qualname), for verification.

    Distinguishes a subclass that overrides behaviour (e.g. a custom
    CommissionModel subclass that always charges zero despite reporting the
    same ``commission_bps``, or a strategy subclass that overrides signal
    generation without changing any constructor parameter) from the exact
    class ``config.yaml`` alone would build. Neither an ``isinstance`` check
    nor a friendly type-family label (``"constant"`` vs ``"volume"``) can
    tell such a subclass apart from its base class -- both pass identically,
    silently missing the override -- but this always can.
    """
    cls = type(instance)
    return f"{cls.__module__}.{cls.__qualname__}"


def _hash_adv(adv: pd.DataFrame | float | None) -> str:
    """Deterministic identity for a volume-based slippage model's ADV.

    A DataFrame is hashed via the same content hash already used for data
    provenance elsewhere (``ParquetStorage.hash_frame``); a scalar or
    ``None`` is hashed via its own repr. Either way this is compact and
    deterministic, unlike embedding the whole ADV matrix into
    metadata.json -- but still lets two differently-built ADV arrays (e.g.
    a manipulated or stale one smuggled in via a custom ``ExecutionModel``)
    be told apart for ``config_yaml_reflects_execution``.
    """
    if isinstance(adv, pd.DataFrame):
        return ParquetStorage.hash_frame(adv)
    return hashlib.sha256(repr(adv).encode("utf-8")).hexdigest()


def _describe_slippage(model: SlippageModel) -> dict[str, object]:
    """Describe a slippage model's own parameters, for metadata/reporting.

    Introspects the actual instance rather than ``config.execution``, for
    the same reason ``_build_metadata`` reads commission/spread from the
    real ``ExecutionModel``: a custom slippage model passed directly to
    :class:`BacktestEngine` need not match the YAML-configured one.
    ``slippage_class`` (exact class identity, see
    :func:`_qualified_class_name`) is always included, even for a
    recognised base class, so a subclass overriding behaviour without
    changing any reported parameter is still distinguishable -- the
    friendly ``slippage_model`` label alone (``"constant"``/``"volume"``)
    passes identically for such a subclass, via ``isinstance``.
    """
    identity = {"slippage_class": _qualified_class_name(model)}
    if isinstance(model, ConstantSlippageModel):
        return {
            **identity,
            "slippage_model": "constant",
            "slippage_bps": model.slippage_bps,
        }
    if isinstance(model, VolumeBasedSlippageModel):
        return {
            **identity,
            "slippage_model": "volume",
            "slippage_bps": model.base_slippage_bps,
            "impact_coefficient": model.impact_coefficient,
            "average_daily_volume_hash": _hash_adv(model.average_daily_volume),
        }
    return {**identity, "slippage_model": type(model).__name__}


def _effective_component_parameters(instance: object) -> tuple[dict[str, object], bool]:
    """Best-effort snapshot of a component's *actual* constructor values.

    docs/api.md documents using ``BacktestEngine`` directly with a custom
    strategy/allocator instance -- ``config.yaml`` in the saved bundle is
    still whatever ``ExperimentConfig`` the caller happened to pass
    alongside it, which need not match the object actually used (e.g. a
    strategy built with ``lookback_period=10`` passed alongside a config
    whose own ``strategy.parameters.lookback_period`` says 252). This
    mirrors ``strategy_parameter_names()``'s introspection of
    ``__init__``'s signature, then reads each named parameter back off the
    instance as an attribute -- the convention every built-in strategy/
    allocator follows. Only JSON-safe scalars are kept: a custom component
    could store anything under a matching attribute name (e.g. a whole
    ``average_daily_volume`` DataFrame), and dumping that into
    metadata.json would be unserialisable or enormous rather than useful.

    Returns the best-effort snapshot, plus whether *every* constructor
    parameter could actually be captured. A parameter with no matching
    attribute, or a non-scalar value, is omitted from the snapshot -- and
    makes the second return value ``False``. A caller comparing two
    snapshots for verification purposes must treat ``False`` as "unverified
    entirely", never silently pass on whatever subset happened to match:
    two different uncaptured values (e.g. two different callables, or two
    different DataFrames) would otherwise compare as equal simply because
    neither made it into the dict.
    """
    try:
        signature = inspect.signature(type(instance).__init__)
    except (TypeError, ValueError):
        return {}, False
    values: dict[str, object] = {}
    fully_captured = True
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            continue
        if not hasattr(instance, parameter.name):
            fully_captured = False
            continue
        value = getattr(instance, parameter.name)
        if value is None or isinstance(value, (bool, int, float, str)):
            values[parameter.name] = value
        else:
            fully_captured = False
    return values, fully_captured


def _reference_strategy(config: ExperimentConfig) -> BaseStrategy | None:
    """Best-effort reconstruction of the strategy ``config`` alone would build.

    Mirrors :func:`~quantlab.backtesting.runner.build_strategy_from_config`
    (duplicated rather than imported: :mod:`quantlab.backtesting.runner`
    itself imports :class:`BacktestEngine` from this module, so importing it
    back here would create a cycle). Returns ``None`` rather than raising
    when construction fails -- an unbuildable reference is "not verified",
    never fabricated into a false positive or negative.
    """
    parameters = dict(config.strategy_parameters)
    accepted = strategy_parameter_names(config.strategy_name)
    if "periods_per_year" in accepted and "periods_per_year" not in parameters:
        parameters["periods_per_year"] = config.periods_per_year
    try:
        return build_strategy(config.strategy_name, parameters)
    except QuantLabError:
        return None


def _reference_allocator(config: ExperimentConfig) -> PortfolioAllocator | None:
    """Best-effort reconstruction of the allocator ``config`` alone would build.

    Mirrors :func:`~quantlab.backtesting.runner.build_allocator_from_config`,
    duplicated for the same import-cycle reason as :func:`_reference_strategy`.
    """
    name = config.portfolio.allocator
    kwargs: dict[str, object] = {}
    if name == "inverse_volatility":
        kwargs = {
            "volatility_window": config.portfolio.volatility_window,
            "maximum_weight": config.portfolio.maximum_weight,
            "periods_per_year": config.periods_per_year,
        }
    elif name == "volatility_targeting":
        if config.portfolio.target_volatility is None:
            return None
        kwargs = {
            "target_volatility": config.portfolio.target_volatility,
            "volatility_window": config.portfolio.volatility_window,
            "maximum_leverage": config.portfolio.maximum_leverage,
            "periods_per_year": config.periods_per_year,
        }
    try:
        return build_allocator(name, **kwargs)
    except QuantLabError:
        return None


def _reference_execution_model(
    config: ExperimentConfig, data: pd.DataFrame
) -> ExecutionModel | None:
    """Best-effort reconstruction of the execution model ``config`` alone would build.

    Mirrors :func:`~quantlab.backtesting.runner.build_execution_from_config`,
    duplicated for the same import-cycle reason as :func:`_reference_strategy`.
    Unlike that function's simple, config-only construction, this one also
    operates on ``data`` (pivoting, rolling ADV) for volume-based slippage --
    a runtime DataFrame whose shape isn't guaranteed by config validation the
    way ``config.strategy``/``config.portfolio`` are, so a broad ``except
    Exception`` (not just ``QuantLabError``) is deliberate: this is a
    best-effort verification a direct-API caller's own custom ``data`` slice
    must never be able to crash, only leave "unverified".
    """
    try:
        adv: pd.DataFrame | float | None = None
        if config.execution.slippage_model.lower() in {"volume", "volume_based"}:
            shares = volume_matrix(data)
            price = price_matrix(data, adjusted=False)
            bar_dollar_volume = shares * price
            calendar = uniform_calendar(
                instrument.calendar for instrument in config.data.instruments
            )
            market_is_247 = calendar is not None and is_247(calendar)
            frequency_table = (
                CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR
                if market_is_247
                else FREQUENCY_TO_PERIODS_PER_YEAR
            )
            days_per_year = (
                CALENDAR_DAYS_PER_YEAR if market_is_247 else TRADING_DAYS_PER_YEAR
            )
            bars_per_day = frequency_table[str(config.frequency)] / days_per_year
            window = max(1, round(21 * bars_per_day))
            adv = (
                bar_dollar_volume.rolling(window, min_periods=1).mean().shift(1)
                * bars_per_day
            )
        return ExecutionModel.from_config(config.execution, average_daily_volume=adv)
    except Exception:
        return None


def _effective_execution_summary(model: ExecutionModel) -> dict[str, object]:
    """Return the same commission/spread/slippage snapshot recorded in metadata.

    A single dict makes the real object and a rebuilt reference directly
    comparable with ``==``, the same pattern
    ``config_yaml_reflects_strategy``/``config_yaml_reflects_allocator`` use.
    Includes ``commission_class``/``spread_class`` (exact class identity,
    see :func:`_qualified_class_name`), not just ``commission_bps``/
    ``spread_bps``: a commission/spread subclass overriding the actual cost
    calculation while still reporting the same bps value would otherwise
    compare as an identical match.
    """
    return {
        "commission_class": _qualified_class_name(model.commission),
        "commission_bps": model.commission.commission_bps,
        "spread_class": _qualified_class_name(model.spread),
        "spread_bps": model.spread.spread_bps,
        **_describe_slippage(model.slippage),
    }


class BacktestEngine:
    """Vectorised backtest engine with a delayed-execution barrier.

    Weights are always shifted before returns are computed (see ``run``'s
    step 7), preventing the common look-ahead leak of acting on a signal the
    same period it was formed. This does not, by itself, make an arbitrary
    custom strategy leak-free: a strategy that reads future rows out of
    ``data`` directly, or otherwise builds a signal non-causally, is outside
    what this barrier can catch -- causal feature and signal construction
    remains the strategy's own responsibility.
    """

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        allocator: PortfolioAllocator,
        execution_model: ExecutionModel,
        config: ExperimentConfig,
        *,
        execution_delay: int = 0,
    ) -> BacktestResult:
        """Run a backtest and return a :class:`BacktestResult`.

        Args:
            data: Canonical long OHLCV frame (may also carry a benchmark
                symbol outside ``config.symbols``, see below).
            strategy: Signal generator.
            allocator: Turns signals into target portfolio weights.
            execution_model: Cost model (commission/spread/slippage).
            config: Experiment configuration.
            execution_delay: Additional bars between target formation and
                execution. Turnover, costs, trades and returns are recomputed
                from the delayed weights.
        """
        started = time.perf_counter()
        if not isinstance(data, pd.DataFrame):
            raise BacktestError("data must be a pandas DataFrame.")
        missing_columns = [
            column for column in (TIMESTAMP, SYMBOL) if column not in data.columns
        ]
        if missing_columns:
            raise BacktestError(
                "Market data is missing required column(s): "
                f"{missing_columns}. Expected canonical long OHLCV data."
            )
        if data.empty:
            raise BacktestError("Cannot backtest on empty data.")
        if isinstance(execution_delay, bool) or not isinstance(
            execution_delay, Integral
        ):
            raise BacktestError("execution_delay must be a non-negative integer.")
        if execution_delay < 0:
            raise BacktestError("execution_delay must be a non-negative integer.")
        delay = int(execution_delay)

        # Retain benchmark-only rows for comparison but exclude them from trading.
        tradable_symbols = set(config.symbols)
        tradable_data = data[data[SYMBOL].isin(tradable_symbols)].reset_index(drop=True)
        present_symbols = set(tradable_data[SYMBOL].unique())
        missing_symbols = [
            symbol for symbol in config.symbols if symbol not in present_symbols
        ]
        if missing_symbols:
            raise BacktestError(
                "Market data is missing configured tradable symbol(s): "
                f"{missing_symbols}. Refusing to backtest a silently reduced universe."
            )

        prices = price_matrix(tradable_data, adjusted=True)
        if prices.shape[0] < 2:
            raise BacktestError(
                f"Need at least 2 dates to backtest; got {prices.shape[0]}."
            )
        asset_returns = compute_asset_returns(prices)

        # Require the strategy to cover the exact tradable panel.
        signals = strategy.generate_signals(tradable_data)
        signals = strategy._validate_signals(signals, prices)

        # Convert signals to target weights.
        allocated = validate_execution_frame(
            allocator.allocate(signals, tradable_data), name="allocator output"
        )
        if not allocated.index.equals(prices.index) or not allocated.columns.equals(
            prices.columns
        ):
            raise BacktestError(
                "allocator output must have exactly the price-matrix index and "
                "columns; missing or extra dates/symbols must not be filled "
                "silently."
            )
        target_weights = allocated

        # Apply portfolio-level volatility targeting only when the allocator
        # does not already perform that scaling.
        if (
            config.portfolio.target_volatility is not None
            and config.portfolio.allocator != "volatility_targeting"
        ):
            target_weights = apply_volatility_target(
                target_weights,
                asset_returns.reindex(
                    index=target_weights.index, columns=target_weights.columns
                ),
                config.portfolio.target_volatility,
                window=config.portfolio.volatility_window,
                maximum_leverage=config.portfolio.maximum_leverage,
                periods_per_year=config.periods_per_year,
            )

        # Enforce portfolio constraints.
        constraints = constraints_from_config(config.portfolio)
        constrained = constraints.apply(target_weights)

        # Apply the shared stateful rebalancing/turnover pipeline once over
        # the full index so its state remains continuous. A closed symbol
        # (per its own calendar) never trades that date -- only meaningful at
        # daily granularity, see quantlab.data.closures. Only bother computing
        # a mask when the instruments actually span more than one calendar:
        # for a single-calendar experiment this must be a provable no-op
        # (tradable=None, byte-identical to a raw schedule/turnover-cap run
        # with no tradability awareness at all), since a calendar library's
        # real holiday set need not match every quirk of whatever data
        # happens to be loaded for a lone calendar.
        tradable = None
        symbol_calendars = {
            instrument.symbol: instrument.calendar
            for instrument in config.data.instruments
        }
        shared_calendar = uniform_calendar(symbol_calendars.values())
        if config.data.frequency == DAILY_FREQUENCY and shared_calendar is None:
            tradable = tradable_mask_for(
                pd.DatetimeIndex(prices.index), config.symbols, symbol_calendars
            )
        held_weights = rebalance_and_cap_turnover(
            constrained, config.portfolio, tradable=tradable, calendar=shared_calendar
        )
        if delay > 0:
            if tradable is not None:
                # A raw row-count shift would delay execution onto a date a
                # symbol can't actually trade on (see
                # shift_respecting_tradability's docstring) -- exactly the
                # same bug the mandatory look-ahead-barrier shift below
                # avoids, so the extra configured delay must avoid it too.
                held_weights = shift_respecting_tradability(
                    held_weights, delay, tradable
                ).fillna(0.0)
            else:
                held_weights = held_weights.shift(delay).fillna(0.0)

        # Accounting contains the one-period look-ahead barrier.
        accounting = run_accounting(
            held_weights,
            asset_returns,
            execution_model,
            config.initial_capital,
            tradable=tradable,
        )

        # Align the benchmark to the simulated portfolio dates.
        benchmark_data = (
            data if config.benchmark_kind is BenchmarkKind.SYMBOL else tradable_data
        )
        benchmark_returns = build_benchmark(
            benchmark_data,
            pd.DatetimeIndex(prices.index),
            benchmark_symbol=config.benchmark_symbol,
            benchmark_calendar=config.benchmark_calendar,
            first_asset_symbol=config.symbols[0],
            risk_free_rate=config.backtest.risk_free_rate,
            periods_per_year=config.periods_per_year,
            kind=str(config.benchmark_kind),
        )

        # The actually-supplied execution_model is the single source of truth
        # for what was charged -- not config.execution, which merely
        # describes the YAML default and can legitimately differ when a
        # caller uses BacktestEngine directly with a custom ExecutionModel
        # (see docs/api.md's "Extension points"). Accounting already used
        # this exact model; the trade log's per-fill breakdown must agree
        # with it too, or the equity curve, trade log and report could each
        # describe a different reality.
        trades = build_trade_log(
            accounting.executed_weights,
            accounting.weight_changes,
            accounting.equity,
            price_matrix(tradable_data, adjusted=False),
            commission_bps=execution_model.commission.commission_bps,
            spread_bps=execution_model.spread.spread_bps,
            slippage_model=execution_model.slippage,
            slippage_equity=accounting.equity_for_costs,
        )

        # Compute performance, risk and portfolio metrics.
        metrics = compute_metrics(
            accounting.net_returns,
            accounting.equity,
            benchmark_returns=benchmark_returns,
            risk_free_rate=config.backtest.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
        metrics.update(
            portfolio_metrics_from_accounting(accounting, config.periods_per_year)
        )
        metrics["number_of_trades"] = float(len(trades))
        metrics["average_trade_size"] = (
            float(trades["traded_notional"].mean()) if len(trades) else 0.0
        )
        # Fractional cost summed over time; currency costs live in the trade log.
        metrics["total_cost_fraction"] = float(accounting.costs.total.sum())

        calculation_elapsed = time.perf_counter() - started
        logger.info(
            "Backtest '%s' completed in %.3fs (Sharpe=%.2f, CAGR=%.2f%%).",
            config.experiment_name,
            calculation_elapsed,
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("cagr", 0.0) * 100.0,
        )

        # Assemble the immutable run result.
        return BacktestResult(
            config=config,
            equity_curve=accounting.equity,
            returns=accounting.net_returns,
            benchmark_returns=benchmark_returns,
            positions=accounting.executed_weights,
            weights=held_weights,
            target_weights=constrained,
            signals=signals,
            trades=trades,
            costs=accounting.costs.to_frame(),
            metrics=metrics,
            metadata=self._build_metadata(
                config,
                tradable_data,
                calculation_elapsed,
                data,
                strategy,
                allocator,
                execution_model,
            ),
            gross_returns=accounting.gross_returns,
            gross_equity=accounting.gross_equity,
            turnover=accounting.turnover,
        )

    @staticmethod
    def _build_metadata(
        config: ExperimentConfig,
        tradable_data: pd.DataFrame,
        calculation_elapsed: float,
        full_data: pd.DataFrame,
        strategy: BaseStrategy,
        allocator: PortfolioAllocator,
        execution_model: ExecutionModel,
    ) -> dict[str, object]:
        """Build metadata used to compare and audit runs.

        ``data_hash`` covers the full input, including a separate benchmark,
        while ``n_rows`` counts only tradable rows. Hashes and dependency
        versions detect differences but do not recreate a full environment.
        ``elapsed_seconds`` measures calculation through metrics; metadata
        hashing and result serialisation are excluded.

        ``strategy``/``allocator``/``execution_model`` are read from the
        objects actually passed to :meth:`run`, not from ``config`` --
        docs/api.md documents using :class:`BacktestEngine` directly with a
        custom strategy, allocator or execution-model instance, which need
        not match ``config``'s own YAML-derived settings. Recording
        ``config``'s values here instead would let accounting (which always
        uses the real objects), the trade log and this metadata each
        describe a different reality for the exact same run.

        ``strategy_parameters``/``allocator_parameters`` are the same kind
        of best-effort actual-object snapshot (see
        ``_effective_component_parameters``), covering what commission/
        spread/slippage don't: `config.yaml` in the saved bundle is still
        whatever config the caller passed, which may not match the real
        object -- not only via a declared value (e.g. a config saying
        ``lookback_period: 252`` alongside a strategy actually built with
        ``lookback_period=10``), but also via an *undeclared* parameter
        left at its constructor default (e.g. the config never mentions
        ``lookback_period`` at all, defaulting it to 20, while the real
        object was built with 10 -- comparing only declared keys would miss
        this entirely, since the mismatched key is never even examined).
        ``config_yaml_reflects_strategy``/``config_yaml_reflects_allocator``/
        ``config_yaml_reflects_execution`` each compare the *complete*
        effective parameter set -- plus the exact class identity, see
        ``_qualified_class_name`` -- against a reference object rebuilt from
        ``config`` alone (mirroring the same factory each ordinary CLI/
        dashboard run uses, see ``_reference_strategy``/
        ``_reference_allocator``/``_reference_execution_model``). A subclass
        that overrides behaviour without changing any reported parameter
        (e.g. a strategy subclass that always stays in cash, or a
        commission subclass that always charges zero) is caught by the
        class-identity check even when every parameter still matches. A
        reference that cannot even be built, or whose parameters -- or the
        real object's own -- could not be *fully* captured (see
        ``_effective_component_parameters``'s second return value) counts
        as unverified, never as a silent pass on a partial match. A report
        claiming the whole run is reproducible from config.yaml must
        require all three (see ``render_html_report``'s footer).
        """
        strategy_parameters, strategy_captured = _effective_component_parameters(
            strategy
        )
        reference_strategy = _reference_strategy(config)
        config_yaml_reflects_strategy = False
        if reference_strategy is not None:
            reference_strategy_parameters, reference_captured = (
                _effective_component_parameters(reference_strategy)
            )
            config_yaml_reflects_strategy = (
                strategy_captured
                and reference_captured
                and _qualified_class_name(strategy)
                == _qualified_class_name(reference_strategy)
                and strategy_parameters == reference_strategy_parameters
            )
        allocator_parameters, allocator_captured = _effective_component_parameters(
            allocator
        )
        reference_allocator = _reference_allocator(config)
        config_yaml_reflects_allocator = False
        if reference_allocator is not None:
            reference_allocator_parameters, reference_allocator_captured = (
                _effective_component_parameters(reference_allocator)
            )
            config_yaml_reflects_allocator = (
                allocator_captured
                and reference_allocator_captured
                and _qualified_class_name(allocator)
                == _qualified_class_name(reference_allocator)
                and allocator_parameters == reference_allocator_parameters
            )
        execution_effective = _effective_execution_summary(execution_model)
        reference_execution = _reference_execution_model(config, full_data)
        config_yaml_reflects_execution = (
            reference_execution is not None
            and execution_effective == _effective_execution_summary(reference_execution)
        )
        return {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "experiment_name": config.experiment_name,
            "strategy": strategy.name,
            "strategy_parameters": strategy_parameters,
            "config_yaml_reflects_strategy": config_yaml_reflects_strategy,
            "allocator": allocator.name,
            "allocator_parameters": allocator_parameters,
            "config_yaml_reflects_allocator": config_yaml_reflects_allocator,
            "config_yaml_reflects_execution": config_yaml_reflects_execution,
            **execution_effective,
            "symbols": config.symbols,
            "start_date": str(config.start_date),
            "end_date": str(config.end_date),
            "random_seed": config.random_seed,
            "n_rows": len(tradable_data),
            "elapsed_seconds": round(calculation_elapsed, 4),
            "periods_per_year": config.periods_per_year,
            "data_hash": ParquetStorage.hash_frame(full_data),
            "git_commit": _git_commit_hash(),
            "git_dirty": _git_is_dirty(),
            "dependency_versions": _dependency_versions(),
            "code_hash": _source_hash(),
            "generator_hash": _generator_hash(),
        }
