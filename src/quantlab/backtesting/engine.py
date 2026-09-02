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

import numpy as np
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
    EPSILON,
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
    executed_weights,
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
from quantlab.portfolio.constraints import ConstraintTouch, constraints_from_config
from quantlab.portfolio.rebalancing import (
    apply_rebalancing,
    rebalance_and_cap_turnover,
    rebalance_dates,
)
from quantlab.portfolio.volatility_targeting import apply_volatility_target
from quantlab.risk.metrics import compute_metrics
from quantlab.strategies.base import (
    BaseStrategy,
    SignalReasons,
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

        # Engine-injected context (never a user-configured constructor
        # hyperparameter, see BaseStrategy.symbol_calendars's own
        # docstring), set before any strategy method that might compute a
        # rolling-window feature is called, so every native-calendar call
        # site (quantlab.features.native_calendar.compute_native_then_
        # align) can compute on each symbol's own calendar rather than a
        # closure-padded combined timeline.
        symbol_calendars = {
            instrument.symbol: instrument.calendar
            for instrument in config.data.instruments
        }
        strategy.symbol_calendars = symbol_calendars

        # Require the strategy to cover the exact tradable panel.
        signals = strategy.generate_signals(tradable_data)
        signals = strategy._validate_signals(signals, prices)

        # Optional, strategy-specific explanation of `signals` -- a pure,
        # deterministic recomputation from the SAME tradable_data (see
        # BaseStrategy.explain_signals's own docstring), never a cache of
        # the call above. None for strategies that don't implement it
        # (the default): the generic strategy_signal reason still works,
        # just without a strategy-specific sub-code.
        raw_signal_reasons = strategy.explain_signals(tradable_data)
        signal_reasons: SignalReasons | None = (
            None
            if raw_signal_reasons is None
            else BaseStrategy._validate_signal_reasons(
                raw_signal_reasons.detail_code, raw_signal_reasons.details, prices
            )
        )

        # Diagnostic decision proxy (optional, see BaseStrategy.decision_
        # signal's own docstring): defaults to the raw signal itself, which
        # is already a faithful decision proxy for every built-in strategy
        # except pairs_trading (whose raw signal mixes a discrete decision
        # with mechanical price/beta rescaling). Used ONLY for trigger
        # detection and position_strategy_origin tracking below -- never
        # substituted for `signals` in allocation, constraints or execution.
        raw_decision_proxy = strategy.decision_signal(tradable_data)
        decision_proxy = (
            signals
            if raw_decision_proxy is None
            else BaseStrategy._validate_decision_signal(raw_decision_proxy, prices)
        )

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

        # Fully-desired weights, post-allocation and post-volatility-target
        # but pre-constraint -- captured under its own stable name before
        # `constrained` (below) overwrites what `target_weights` means, so
        # the trade log can later tell "didn't reach its desired size" (a
        # constraint) apart from "the desired size itself changed" (signal/
        # rebalance/vol-target). See build_trade_log's own docstring.
        desired_target = target_weights

        # Enforce portfolio constraints. apply_with_provenance runs the
        # exact same computation as apply() (see its own docstring) and
        # additionally records, per constraint, which cells it actually
        # changed -- real provenance from the real computation, not a
        # parallel reconstruction.
        constraints = constraints_from_config(config.portfolio)
        constrained, constraint_touches = constraints.apply_with_provenance(
            target_weights
        )

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
        shared_calendar = uniform_calendar(symbol_calendars.values())
        if config.data.frequency == DAILY_FREQUENCY and shared_calendar is None:
            # `prices.columns` (not `config.symbols`) is the order every
            # other frame in this method is built and validated against
            # (`allocated.columns.equals(prices.columns)` above, and
            # transitively `constrained`/`held_weights`/the diagnostic
            # frames below) -- `pivot_field` sorts symbols alphabetically,
            # which need not match the instrument declaration order in
            # `config.symbols`. `executed_weights(..., tradable=tradable)`
            # (used throughout this method) requires its `tradable` frame
            # to have EXACTLY matching columns, in the same order, so
            # `tradable` must be built against `prices.columns` here too.
            tradable = tradable_mask_for(
                pd.DatetimeIndex(prices.index), list(prices.columns), symbol_calendars
            )
            # Position-group-coherent tradability (e.g. pairs_trading's two
            # legs): a per-column tradable mask alone only guarantees each
            # LEG's own eligibility independently -- a declared group must
            # be eligible to move as ONE unit, on a date every member is
            # tradable, never a date only some of its legs are (no legging
            # risk is modeled). `_rebalance_tradability_aware`'s per-column
            # "pending debt, retried every day" scheduling then operates on
            # this group-collapsed mask directly. A symbol never in any
            # declared group keeps its own independent tradability.
            groups = strategy.position_groups()
            if groups:
                for group in groups:
                    members = [symbol for symbol in group if symbol in tradable.columns]
                    if len(members) > 1:
                        group_tradable = tradable[members].all(axis=1)
                        for symbol in members:
                            tradable[symbol] = group_tradable

        # Diagnostic (not real-execution) frames used only to attribute a
        # trade's reason -- signals/allocated/desired_target/constrained are
        # recomputed fresh every row (never forward-filled the way
        # held_weights is), so comparing them "yesterday vs today" would
        # measure normal day-to-day drift, not "since the last rebalance
        # decision". Resampling them the same way rebalance_and_cap_turnover
        # itself samples `constrained` (via apply_rebalancing, using the
        # same frequency/calendar) gives them the same rebalance-date
        # cadence as held_weights, so a plain shift(1) inside
        # build_trade_log correctly reads "value as of the previous
        # rebalance decision". Never used to recompute an executed weight, cost or
        # PnL figure -- only build_trade_log's reason classifier reads
        # these. Centralised in one local helper so these frames can never
        # drift out of sync with each other or with held_weights' own shift.
        def _rebalance_diagnostic_frame(frame: pd.DataFrame) -> pd.DataFrame:
            return apply_rebalancing(
                frame, config.portfolio.rebalance_frequency, calendar=shared_calendar
            )

        # `rebalanced_signal` is built from `decision_proxy`, not `signals`
        # directly: this is what prevents `strategy_signal` from firing on
        # a strategy's purely mechanical rescaling (e.g. pairs_trading's
        # price/beta drift at constant discrete state) that isn't a real
        # new decision. `rebalanced_constrained` (new) is used only to
        # build `target_episode_id` below.
        rebalanced_signal = _rebalance_diagnostic_frame(decision_proxy)
        rebalanced_allocated = _rebalance_diagnostic_frame(allocated)
        rebalanced_desired = _rebalance_diagnostic_frame(desired_target)
        rebalanced_constrained = _rebalance_diagnostic_frame(constrained)

        # `target_episode_id`: a cell-level, monotonically increasing
        # integer identifying which upstream decision produced the target
        # currently being chased at each rebalance date -- the real
        # identity a turnover-capped/tradability-deferred debt is scoped
        # to (see rebalancing.py's `episode_id`), never a bare calendar
        # counter (row-level) nor the target's own numeric value (two
        # distinct decisions can coincidentally produce the same number).
        # Increments a cell's counter only when a REAL upstream event
        # concerns that cell: the exact same three diagnostic comparisons
        # `_classify_reason` uses for trigger detection (signal/allocator/
        # vol-target changed since the last rebalance), or -- when none of
        # those fired -- the pre-turnover target itself still drifting
        # (the same condition that drives the position_rescaling fallback
        # below), which is how a continuously-rescaling target (pairs_
        # trading) still gets a fresh episode per genuine change. A plain
        # periodic re-sampling of the SAME still-pursued target increments
        # nothing, so a turnover/tradability debt against it survives
        # across rebalance dates as required.
        def _changed_since_last_rebalance(frame: pd.DataFrame) -> pd.DataFrame:
            values = frame.to_numpy()
            previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
            return pd.DataFrame(
                np.abs(values - previous) > EPSILON,
                index=frame.index,
                columns=frame.columns,
            )

        signal_changed = _changed_since_last_rebalance(rebalanced_signal)
        allocator_changed = _changed_since_last_rebalance(rebalanced_allocated)
        vol_target_changed = _changed_since_last_rebalance(rebalanced_desired)
        constrained_changed = _changed_since_last_rebalance(rebalanced_constrained)
        no_trigger = ~(signal_changed | allocator_changed | vol_target_changed)
        new_episode_this_row = (
            signal_changed
            | allocator_changed
            | vol_target_changed
            | (no_trigger & constrained_changed)
        )
        target_episode_id = new_episode_this_row.astype(int).cumsum()

        # When weight drift is enabled, `apply_weight_drift` (below, via
        # run_accounting) is the SOLE place `maximum_turnover` is applied --
        # capping it here too would make this decision-level step produce
        # an INTERMEDIATE, not-yet-fully-walked target (e.g. 0.2 while the
        # true schedule target is 1.0), which the drift layer would then
        # treat as "the" target, capable of trading the portfolio BACKWARD
        # toward that stale intermediate value even while organic price
        # drift has already carried it past it. Scheduling and
        # tradability-aware pending-debt-carrying are UNCHANGED (`None`
        # is already this module's own "uncapped" convention, not a
        # separate code path) -- only the cap itself is turned off here.
        decision_portfolio_config = (
            config.portfolio.revalidated_copy(update={"maximum_turnover": None})
            if config.portfolio.model_weight_drift
            else config.portfolio
        )
        held_weights, turnover_provenance = rebalance_and_cap_turnover(
            constrained,
            decision_portfolio_config,
            tradable=tradable,
            calendar=shared_calendar,
            episode_id=target_episode_id,
            return_provenance=True,
        )

        def _apply_extra_delay(frame: pd.DataFrame) -> pd.DataFrame:
            if delay <= 0:
                return frame
            if tradable is not None:
                # A raw row-count shift would delay execution onto a date a
                # symbol can't actually trade on (see
                # shift_respecting_tradability's docstring) -- the mandatory
                # look-ahead-barrier shift below avoids exactly this, and the
                # extra configured delay must avoid it too.
                # Applied identically to held_weights and every reason
                # frame, so they all stay aligned to the same decision date.
                return shift_respecting_tradability(frame, delay, tradable).fillna(0.0)
            return frame.shift(delay).fillna(0.0)

        held_weights = _apply_extra_delay(held_weights)
        desired_target_aligned = _apply_extra_delay(desired_target)
        constrained_aligned = _apply_extra_delay(constrained)
        rebalanced_signal = _apply_extra_delay(rebalanced_signal)
        rebalanced_allocated = _apply_extra_delay(rebalanced_allocated)
        rebalanced_desired = _apply_extra_delay(rebalanced_desired)

        # constraint_touches (from apply_with_provenance above) is already
        # at the same raw daily cadence as `constrained` itself -- not a
        # rebalance-sampled diagnostic frame -- so it only needs the same
        # delay+executed_weights alignment as executed_constrained, never
        # _rebalance_diagnostic_frame. Boolean frames are round-tripped
        # through float so they can reuse the exact same real numeric
        # functions, then thresholded back to bool (exact, since the only
        # values ever produced are 0.0/1.0).
        def _align_bool(frame: pd.DataFrame) -> pd.DataFrame:
            # `executed_weights` is built for *weights*, where a closed row
            # correctly repeats the last tradable row's value (frozen, no
            # reallocation while closed). Applied to a boolean flag, that
            # same repetition would keep the flag True for every row a
            # column stays closed after it lands True once -- wrong for a
            # flag, which must describe THIS row's own event, never a
            # carried-forward one. AND with `tradable` (when given) so a
            # closed row's flag is always False, matching
            # apply_weight_drift's own documented precondition; with no
            # tradable mask at all, every row is implicitly tradable and
            # this repetition concern cannot arise.
            flag = (
                executed_weights(
                    _apply_extra_delay(frame.astype(float)), tradable=tradable
                )
                > 0.5
            )
            return flag & tradable if tradable is not None else flag

        # Genuine scheduled rebalance dates, aligned onto the executed
        # timeline the same way held_weights becomes executed (extra
        # delay, then the mandatory look-ahead-barrier shift) --
        # apply_weight_drift's own anchor detection needs this, not just
        # value-diffing `executed` against its own previous row: a
        # rebalance whose freshly-decided target happens to numerically
        # match the immediately preceding one (a constant-target
        # schedule, or an unchanged signal) must still be treated as a
        # real trade back to target, never silently absorbed into
        # ongoing drift. See apply_weight_drift's own docstring.
        schedule_dates = rebalance_dates(
            pd.DatetimeIndex(constrained.index),
            config.portfolio.rebalance_frequency,
            calendar=shared_calendar,
        )
        is_rebalance_date_frame = pd.DataFrame(
            np.broadcast_to(
                constrained.index.isin(schedule_dates)[:, None], constrained.shape
            ),
            index=constrained.index,
            columns=constrained.columns,
        )
        # Per-column, NOT collapsed with `.any(axis=1)`: a closed
        # instrument must never be forced to anchor (and therefore trade)
        # just because some OTHER instrument's own schedule/value-change
        # fires the same row -- see apply_weight_drift's own docstring.
        # `_align_bool` already routes this per-column through the exact
        # same tradability-aware shift real weight values get, so a
        # closed column's own flag correctly stays tied to ITS OWN next
        # genuinely tradable session, never another column's.
        rebalance_date = _align_bool(is_rebalance_date_frame)

        aligned_constraint_touches: dict[str, ConstraintTouch] = {
            name: ConstraintTouch(
                touched=_align_bool(touch.touched),
                before=executed_weights(
                    _apply_extra_delay(touch.before), tradable=tradable
                ),
                after=executed_weights(
                    _apply_extra_delay(touch.after), tradable=tradable
                ),
                direct=_align_bool(touch.direct),
            )
            for name, touch in constraint_touches.items()
        }
        # Split each redistribution-capable constraint's provenance into
        # two entries -- the base name keeps only the directly-clipped
        # cells, "*_redistribution" the touched-but-not-direct ones -- so
        # build_trade_log can attribute each honestly, never a single
        # winning constraint. Both entries share the SAME before/after
        # (the real value immediately around this constraint's own
        # effect); only which cells count as "touched" differs. Every
        # other constraint (no redistribution concept, `direct == touched`
        # by construction) passes through as a single entry unchanged.
        redistribution_capable = frozenset(
            {"maximum_weight", "minimum_weight", "maximum_positions"}
        )
        executed_constraint_touches: dict[str, ConstraintTouch] = {}
        for name, touch in aligned_constraint_touches.items():
            if name in redistribution_capable:
                executed_constraint_touches[name] = ConstraintTouch(
                    touched=touch.direct,
                    before=touch.before,
                    after=touch.after,
                    direct=touch.direct,
                )
                redistribution_mask = touch.touched & ~touch.direct
                executed_constraint_touches[f"{name}_redistribution"] = ConstraintTouch(
                    touched=redistribution_mask,
                    before=touch.before,
                    after=touch.after,
                    direct=pd.DataFrame(
                        False, index=touch.touched.index, columns=touch.touched.columns
                    ),
                )
            else:
                executed_constraint_touches[name] = touch

        # Real, cell-level turnover-cap/tradability provenance from
        # rebalancing.py, aligned to executed_weights' index the same way
        # as everything else above.
        executed_turnover_actively_limited = _align_bool(
            turnover_provenance.turnover_actively_limited
        )
        executed_turnover_touched = _align_bool(turnover_provenance.turnover_touched)
        executed_tradability_touched = _align_bool(
            turnover_provenance.tradability_touched
        )
        executed_tradability_compliance_limited = _align_bool(
            turnover_provenance.tradability_compliance_limited
        )

        # Two seeds built from `decision_proxy` (never `signal_reasons.
        # detail_code.notna()`, which would wrongly gate detection on
        # whether a strategy-specific detail happens to exist -- a real
        # transition without one must still be detected):
        # - `trigger_detail_seed`: a plain, unbounded row-index pointer to
        #   the MOST RECENT transition (any magnitude change > EPSILON),
        #   never cleared -- consulted only when `strategy_signal` is
        #   itself the trigger, so a stale pointer on non-firing rows is
        #   harmless; reading the strategy's own detail_code/details AT
        #   that exact row (rather than forward-filling the text itself)
        #   means a detail-less transition correctly clears any earlier
        #   detail rather than letting it leak forward.
        # - `position_origin_seed`: a REGIME-based (flat/long/short via
        #   sign) pointer for `position_strategy_origin` -- a continuous
        #   signal's own magnitude drift never creates a new origin, only
        #   a flat<->non-flat regime change does; a transition into flat
        #   clears it, a transition out of flat (fresh entry or reversal)
        #   replaces it. Independent of `trigger_detail_seed`: a
        #   downstream layer holding the executed weight flat does not,
        #   by itself, move this seed, since it only ever looks at
        #   `decision_proxy`.
        decision_values = decision_proxy.to_numpy()
        row_index_1based = np.arange(1, len(signals.index) + 1, dtype=float)
        decision_prev_values = np.vstack(
            [np.zeros((1, decision_values.shape[1])), decision_values[:-1]]
        )
        has_transition_magnitude = (
            np.abs(decision_values - decision_prev_values) > EPSILON
        )
        trigger_detail_seed = (
            pd.DataFrame(
                np.where(has_transition_magnitude, row_index_1based[:, None], np.nan),
                index=signals.index,
                columns=signals.columns,
            )
            .ffill()
            .fillna(0.0)
        )

        def _regime(values: np.ndarray) -> np.ndarray:
            return np.where(values > EPSILON, 1, np.where(values < -EPSILON, -1, 0))

        regime_now = _regime(decision_values)
        regime_prev = np.vstack(
            [np.zeros((1, regime_now.shape[1]), dtype=int), regime_now[:-1]]
        )
        has_regime_transition = regime_now != regime_prev
        position_origin_candidate = np.where(
            has_regime_transition,
            np.where(regime_now == 0, 0.0, row_index_1based[:, None]),
            np.nan,
        )
        position_origin_seed = (
            pd.DataFrame(
                position_origin_candidate, index=signals.index, columns=signals.columns
            )
            .ffill()
            .fillna(0.0)
        )

        def _source_row(seed: pd.DataFrame) -> np.ndarray:
            executed_positions = executed_weights(
                _apply_extra_delay(_rebalance_diagnostic_frame(seed)),
                tradable=tradable,
            )
            return executed_positions.round().to_numpy().astype(int) - 1

        trigger_source_row = _source_row(trigger_detail_seed)
        position_origin_source_row = _source_row(position_origin_seed)

        def _gather(raw: pd.DataFrame, source_row: np.ndarray) -> pd.DataFrame:
            raw_values = raw.to_numpy(dtype=object)
            gathered = np.full(source_row.shape, None, dtype=object)
            for column_index in range(source_row.shape[1]):
                valid = source_row[:, column_index] >= 0
                gathered[valid, column_index] = raw_values[
                    source_row[valid, column_index], column_index
                ]
            return pd.DataFrame(
                gathered, index=raw.index, columns=raw.columns, dtype=object
            )

        executed_strategy_reason_code: pd.DataFrame | None = None
        executed_strategy_reason_details: pd.DataFrame | None = None
        if signal_reasons is not None:
            executed_strategy_reason_code = _gather(
                signal_reasons.detail_code, trigger_source_row
            )
            executed_strategy_reason_details = _gather(
                signal_reasons.details, trigger_source_row
            )
            executed_position_strategy_origin_code = _gather(
                signal_reasons.detail_code, position_origin_source_row
            )
            executed_position_strategy_origin_details = _gather(
                signal_reasons.details, position_origin_source_row
            )
        else:
            _empty_object = pd.DataFrame(
                None, index=signals.index, columns=signals.columns, dtype=object
            )
            executed_position_strategy_origin_code = _empty_object
            executed_position_strategy_origin_details = _empty_object.copy()

        # Origin timestamp: independent of whether explain_signals() exists
        # at all -- a strategy without strategy-specific detail codes still
        # gets a temporally correct origin (point 2: the temporal tracking
        # of position_strategy_origin is not gated on detail_code
        # existing).
        _dates = signals.index.to_numpy()
        _origin_gathered = np.full(
            position_origin_source_row.shape, pd.NaT, dtype=object
        )
        for _column_index in range(position_origin_source_row.shape[1]):
            _valid = position_origin_source_row[:, _column_index] >= 0
            _origin_gathered[_valid, _column_index] = _dates[
                position_origin_source_row[_valid, _column_index]
            ]
        executed_position_strategy_origin_timestamp = pd.DataFrame(
            _origin_gathered, index=signals.index, columns=signals.columns, dtype=object
        )

        # Accounting contains the one-period look-ahead barrier. A
        # strategy's stop_loss_pct/take_profit_pct/position_groups() are
        # read generically here (default None/None/None disables the
        # check entirely) -- the mechanism itself lives in accounting.py,
        # operating on the REAL executed position, never this strategy's
        # raw `signals` above.
        accounting = run_accounting(
            held_weights,
            asset_returns,
            execution_model,
            config.initial_capital,
            tradable=tradable,
            stop_loss_pct=strategy.stop_loss_pct,
            take_profit_pct=strategy.take_profit_pct,
            position_groups=strategy.position_groups(),
            model_weight_drift=config.portfolio.model_weight_drift,
            maximum_weight=config.portfolio.maximum_weight,
            # Same combined-cap convention as rebalancing.py's own
            # gross_cap = min(gross_caps): maximum_leverage always has a
            # real (non-None) value, so it must be folded in here too, or
            # the drift-compliance LP would silently miss it whenever
            # maximum_gross_exposure itself is left unset.
            maximum_gross_exposure=(
                min(
                    config.portfolio.maximum_gross_exposure,
                    config.portfolio.maximum_leverage,
                )
                if config.portfolio.maximum_gross_exposure is not None
                else config.portfolio.maximum_leverage
            ),
            maximum_net_exposure=config.portfolio.maximum_net_exposure,
            long_only=config.portfolio.long_only,
            rebalance_date=rebalance_date,
            maximum_turnover=config.portfolio.maximum_turnover,
        )

        # `maximum_turnover` is enforced at the decision level above ONLY
        # when weight drift is disabled -- when it's enabled,
        # `apply_weight_drift` is the sole place it's actually applied (see
        # `decision_portfolio_config` above), so the decision-level
        # `turnover_provenance` computed from it is empty in that case.
        # OR-merging in `accounting`'s own turnover provenance keeps the
        # trade log's `turnover_cap` attribution accurate either way -- a
        # no-op when drift is disabled (accounting's own frames are then
        # all-``False`` placeholders), the real signal when it's enabled.
        # Already at `accounting.executed_weights`' own index -- no further
        # alignment needed.
        executed_turnover_actively_limited = (
            executed_turnover_actively_limited
            | accounting.drift_turnover_actively_limited
        )
        executed_turnover_touched = (
            executed_turnover_touched | accounting.drift_turnover_touched
        )

        # Real, row-broadcast provenance: forced liquidation affects every
        # column simultaneously and unconditionally once the portfolio is
        # ruined -- a legitimate broadcast, not an approximation (see
        # AccountingResult.ruined's own docstring). Already at
        # accounting.executed_weights' own final index -- no further
        # alignment needed.
        executed_forced_liquidation = pd.DataFrame(
            dict.fromkeys(accounting.executed_weights.columns, accounting.ruined),
            index=accounting.executed_weights.index,
        )

        # Align every reason-attribution frame to accounting.executed_weights'
        # own index using the exact same shift function run_accounting uses
        # internally, so they can never misalign with the trade log's own
        # date index. executed_desired/executed_constrained are real
        # pipeline frames (just re-aligned, not resampled); the *_diag ones
        # are the rebalance-sampled diagnostic frames from above.
        executed_desired = executed_weights(desired_target_aligned, tradable=tradable)
        executed_constrained = executed_weights(constrained_aligned, tradable=tradable)
        executed_signal_diag = executed_weights(rebalanced_signal, tradable=tradable)
        executed_allocated_diag = executed_weights(
            rebalanced_allocated, tradable=tradable
        )
        executed_desired_diag = executed_weights(rebalanced_desired, tradable=tradable)

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
            executed_desired=executed_desired,
            executed_constrained=executed_constrained,
            executed_signal_diag=executed_signal_diag,
            executed_allocated_diag=executed_allocated_diag,
            executed_desired_diag=executed_desired_diag,
            tradable=tradable,
            executed_strategy_reason_code=executed_strategy_reason_code,
            executed_strategy_reason_details=executed_strategy_reason_details,
            constraint_provenance=executed_constraint_touches,
            executed_turnover_actively_limited=executed_turnover_actively_limited,
            executed_turnover_touched=executed_turnover_touched,
            executed_tradability_touched=executed_tradability_touched,
            executed_tradability_compliance_limited=(
                executed_tradability_compliance_limited
            ),
            executed_forced_liquidation=executed_forced_liquidation,
            executed_stop_loss_triggered=accounting.stop_loss_triggered,
            executed_take_profit_triggered=accounting.take_profit_triggered,
            executed_drift_compliance_forced=accounting.drift_compliance_forced,
            executed_drift_compliance_pending=accounting.drift_compliance_pending,
            executed_position_strategy_origin_timestamp=(
                executed_position_strategy_origin_timestamp
            ),
            executed_position_strategy_origin_code=(
                executed_position_strategy_origin_code
            ),
            executed_position_strategy_origin_details=(
                executed_position_strategy_origin_details
            ),
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
