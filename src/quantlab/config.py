"""Experiment configuration models.

Configs are the single reproducible description of an experiment. They are
Pydantic models so that loading is validated and mistakes fail loudly and
early.

We model configs in their natural nested form (it maps cleanly to the
domain) and expose flat, read-only convenience accessors on
:class:`ExperimentConfig` so both nested and flat access work.
"""

from __future__ import annotations

import itertools
import math
import re
import types
from collections.abc import Iterable, Mapping
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin

import numpy as np
import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from quantlab.constants import (
    CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    FREQUENCY_TO_PERIODS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
)
from quantlab.exceptions import InvalidConfigurationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

#: Safe filename characters for symbols and experiment names. ``^`` and ``=``
#: are included because Yahoo identifiers commonly use them; path separators
#: and other unsafe characters remain forbidden.
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9._=^-]*$")

#: Windows device names remain reserved even when followed by an extension.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _validate_path_component(value: str, *, field_name: str) -> None:
    """Raise ``ValueError`` if ``value`` isn't safe to use as a path segment."""
    if not _SAFE_PATH_COMPONENT.match(value):
        raise ValueError(
            f"Invalid {field_name} {value!r}: only letters, digits, '.', "
            "'-', '_', '^' and '=' are allowed (must also start with a "
            "letter, digit or '^') — anything else, e.g. a path separator, "
            "could escape the intended directory when used to build a "
            "file path."
        )
    stem = value.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(
            f"Invalid {field_name} {value!r}: {stem!r} is a Windows reserved "
            "device name (CON/PRN/AUX/NUL/COM1-9/LPT1-9) — using it as a "
            "file or directory name addresses the actual OS device instead "
            "of a normal file, silently discarding anything written to it."
        )
    if value.endswith((".", " ")):
        # Windows strips trailing dots and spaces, which would make distinct
        # configured names resolve to the same filesystem entry.
        raise ValueError(
            f"Invalid {field_name} {value!r}: must not end with '.' or a "
            "space — Windows silently strips a trailing dot/space from a "
            "path component, so this would collide with the same name "
            "minus that trailing character on that platform."
        )


#: Scalar configuration values with an unambiguous JSON representation.
#: Dates are accepted because ``model_dump(mode="json")`` serialises them to ISO.
_JSON_SAFE_SCALARS = (bool, int, str, date, type(None))


def _is_numeric_annotation(annotation: Any) -> bool:
    """Return whether a model-field annotation represents an int or float."""
    if annotation in (int, float):
        return True
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        return any(member in (int, float) for member in get_args(annotation))
    return False


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct a YAML mapping and fail on a repeated key."""
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _reject_non_json_safe(value: Any, *, path: str) -> None:
    """Reject values that cannot be represented safely in saved metadata.

    Accepted values are ``None``, booleans, integers, finite floats, strings,
    dates, lists, dictionaries with string keys, and nested Pydantic models
    (each validates its own fields recursively via
    ``_reject_non_json_safe_fields``, so re-walking its internals here would
    be redundant — this mirrors the same skip already applied to a model
    stored directly in a field, generalised to one nested inside a list or
    dict too, e.g. ``DataConfig.instruments: list[InstrumentConfig]``).
    Array-like objects and unordered containers must be converted explicitly
    by the caller.
    """
    if isinstance(value, BaseModel):
        return
    if isinstance(value, _JSON_SAFE_SCALARS):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{path}: NaN/Infinity are not permitted (got {value!r})")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{path}: dict keys must be strings for a JSON-safe "
                    f"config value, got {key!r} ({type(key).__name__})."
                )
            _reject_non_json_safe(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _reject_non_json_safe(item, path=f"{path}[{i}]")
        return
    if isinstance(value, np.generic):
        raise ValueError(
            f"{path}: a numpy scalar ({type(value).__name__}, value "
            f"{value!r}) is not a JSON-safe config value — convert "
            f"explicitly with e.g. `.item()` before assigning it here."
        )
    if isinstance(value, np.ndarray):
        raise ValueError(
            f"{path}: a numpy array is not a JSON-safe config value — "
            "convert explicitly with `.tolist()` before assigning it here."
        )
    if isinstance(value, (pd.Series, pd.DataFrame)):
        raise ValueError(
            f"{path}: a {type(value).__name__} is not a JSON-safe config "
            "value — convert explicitly (e.g. `.tolist()`/`.to_dict()`) "
            "before assigning it here."
        )
    if isinstance(value, (tuple, set, frozenset)):
        raise ValueError(
            f"{path}: a {type(value).__name__} is not a JSON-safe config "
            "value — convert explicitly to a `list` before assigning it "
            "here."
        )
    raise ValueError(
        f"{path}: {type(value).__name__} is not a JSON-safe config value "
        f"(got {value!r}) — config values must be one of None/bool/int/"
        "finite-float/str/list/dict-with-string-keys. Convert explicitly to "
        "one of those before assigning it here."
    )


class MissingValuePolicy(StrEnum):
    """How the cleaner treats missing canonical market data."""

    DROP = "drop"
    FORWARD_FILL = "forward_fill"
    RAISE = "raise"
    NONE = "none"


class RebalanceFrequency(StrEnum):
    """Recognised rebalancing cadences.

    ``custom`` is reserved for a future user-defined schedule and is rejected
    by :class:`PortfolioConfig` until that schedule can be configured.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


class ValidationMethod(StrEnum):
    """Supported out-of-sample validation schemes."""

    HOLDOUT = "holdout"
    WALK_FORWARD = "walk_forward"


class DataFrequency(StrEnum):
    """Bar frequencies understood by QuantLab."""

    D1 = "1d"
    H1 = "1h"
    W1 = "1w"
    MO1 = "1mo"


class DataSourceName(StrEnum):
    """Market-data sources supported by the loader."""

    YAHOO = "yahoo"
    BINANCE = "binance"
    CSV = "csv"


class OptimizationMetric(StrEnum):
    """Metrics available for walk-forward parameter selection."""

    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    TOTAL_RETURN = "total_return"


class SlippageModelName(StrEnum):
    """Supported slippage models."""

    CONSTANT = "constant"
    VOLUME = "volume"


class BenchmarkKind(StrEnum):
    """Supported benchmark constructions."""

    SYMBOL = "symbol"
    EQUAL_WEIGHT = "equal_weight"
    FIRST_ASSET = "first_asset"
    CASH = "cash"


class _StrictModel(BaseModel):
    """Validated config model with forbidden extra keys and frozen attributes.

    ``frozen=True`` prevents field reassignment, but Python lists and
    dictionaries stored inside a field remain mutable. Callers must treat the
    complete config tree as immutable and create changes with
    :meth:`revalidated_copy` rather than mutating nested containers in place.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def revalidated_copy(self, *, update: Mapping[str, Any] | None = None) -> Self:
        """Copy with ``update`` applied, re-running every validator.

        Pydantic's regular ``model_copy(update=...)`` does not validate the
        updated values. This helper rebuilds the model from a fresh dump so
        field bounds, relationship checks, and JSON-safety checks all run.

        The returned model does not preserve ``model_fields_set`` because the
        full dump makes defaults indistinguishable from explicitly supplied
        fields. QuantLab does not rely on that Pydantic bookkeeping value.
        """
        data = self.model_dump(mode="python")
        if update:
            data.update(update)
        return type(self).model_validate(data)

    @model_validator(mode="before")
    @classmethod
    def _reject_bool_for_numeric_fields(cls, data: Any) -> Any:
        """Reject booleans before Pydantic coerces them to 0/1 numbers."""
        if not isinstance(data, Mapping):
            return data
        for name, field in cls.model_fields.items():
            if (
                name in data
                and isinstance(data[name], (bool, np.bool_))
                and _is_numeric_annotation(field.annotation)
            ):
                raise ValueError(
                    f"{name} must be numeric, not a boolean (got {data[name]!r})."
                )
        return data

    @model_validator(mode="after")
    def _reject_non_json_safe_fields(self) -> _StrictModel:
        """Globally forbid NaN, Infinity, and non-JSON-safe values."""
        for name, value in self.__dict__.items():
            if isinstance(value, BaseModel):
                continue
            _reject_non_json_safe(value, path=f"{type(self).__name__}.{name}")
        return self


#: Frequencies supported by each remote backend. CSV frequency is checked
#: against observed timestamps after the file is loaded.
_SOURCE_SUPPORTED_FREQUENCIES: dict[DataSourceName, frozenset[DataFrequency]] = {
    DataSourceName.YAHOO: frozenset(
        {DataFrequency.D1, DataFrequency.H1, DataFrequency.W1, DataFrequency.MO1}
    ),
    DataSourceName.BINANCE: frozenset(
        {DataFrequency.D1, DataFrequency.H1, DataFrequency.W1}
    ),
}


def compatible_frequencies_for_sources(
    sources: Iterable[DataSourceName],
) -> set[DataFrequency]:
    """Return the frequencies compatible with every remote source in ``sources``.

    ``csv`` is neutral: its real frequency depends on the timestamps actually
    present in the file, checked after loading by
    :class:`~quantlab.data.validator.DataValidator` (``_check_declared_frequency``),
    not known in advance the way a remote API's capabilities are. A
    ``sources`` collection containing only ``csv`` (or empty) is therefore
    compatible with every :class:`DataFrequency`.

    This is the single source of truth for frequency compatibility — both
    :class:`DataConfig`'s validator and the dashboard's frequency picker call
    it directly, so the two can never diverge.
    """
    remote = [source for source in sources if source is not DataSourceName.CSV]
    if not remote:
        return set(DataFrequency)
    tables = [_SOURCE_SUPPORTED_FREQUENCIES[source] for source in remote]
    return set.intersection(*(set(table) for table in tables))


class InstrumentConfig(_StrictModel):
    """A single instrument: its symbol, data source, and trading calendar.

    The unit of configuration for a tradable asset or a benchmark. Each
    instrument resolves its own source and calendar explicitly — nothing here
    is inferred at run time; any auto-detection happens upstream (e.g. in the
    dashboard) before this model is constructed.
    """

    symbol: str
    source: DataSourceName
    calendar: str = Field(
        description=(
            "'24/7' for a continuous market, or any calendar name accepted "
            "by pandas_market_calendars (e.g. 'XNYS', 'XHKG', 'CME_Equity')."
        )
    )

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        """Normalise the symbol and make it safe as a CSV/cache filename."""
        sym = value.strip().upper()
        if not sym:
            raise ValueError("symbol must not be empty.")
        _validate_path_component(sym, field_name="symbol")
        return sym

    @field_validator("calendar")
    @classmethod
    def _validate_calendar(cls, value: str) -> str:
        # Local import: quantlab.data imports quantlab.config at module load
        # time (via data/loader.py), so a top-level import here would cycle.
        from quantlab.data.calendar import validate_calendar_name

        calendar = value.strip()
        if not calendar:
            raise ValueError("calendar must not be empty.")
        validate_calendar_name(calendar)
        return calendar

    @model_validator(mode="after")
    def _check_binance_is_always_247(self) -> InstrumentConfig:
        if self.source is DataSourceName.BINANCE and self.calendar != "24/7":
            raise ValueError(
                f"calendar {self.calendar!r} is not permitted with source "
                f"'binance' for symbol {self.symbol!r} — every Binance "
                "symbol genuinely trades 24/7, so there is no real "
                "instrument this combination could correctly describe. "
                "Set calendar: '24/7'."
            )
        return self


class DataConfig(_StrictModel):
    """Data-acquisition settings (the ``data:`` block)."""

    instruments: list[InstrumentConfig] = Field(
        min_length=1,
        description="Tradable instruments, each with its own symbol/source/calendar.",
    )
    start_date: date
    end_date: date
    frequency: DataFrequency = Field(
        default=DataFrequency.D1, description="Bar frequency."
    )
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.DROP
    forward_fill_limit: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum consecutive price bars filled per symbol when "
            "missing_value_policy is 'forward_fill'."
        ),
    )
    use_bundled_demo_data: bool = Field(
        default=False,
        description=(
            "For a csv-sourced instrument, use QuantLab's bundled synthetic "
            "demo files when every requested symbol is absent from the "
            "local raw-data directory (a partial match is always a hard "
            "error, never a mix of local and bundled data). Must be "
            "enabled explicitly."
        ),
    )

    @property
    def symbols(self) -> list[str]:
        """Normalised list of tickers, one per instrument."""
        return [instrument.symbol for instrument in self.instruments]

    @model_validator(mode="after")
    def _check_unique_symbols(self) -> DataConfig:
        """Reject the same symbol appearing in more than one instrument.

        Price series are indexed by symbol alone downstream (the canonical
        schema's SYMBOL column, price_matrix columns) — two instruments for
        the same ticker would be an unresolvable collision, not a case to
        silently support.
        """
        seen: set[str] = set()
        duplicates: set[str] = set()
        for instrument in self.instruments:
            if instrument.symbol in seen:
                duplicates.add(instrument.symbol)
            seen.add(instrument.symbol)
        if duplicates:
            raise ValueError(
                f"Duplicate symbol(s) {sorted(duplicates)} across `instruments` "
                "— each symbol may appear as only one instrument (one source, "
                "one calendar) per experiment."
            )
        return self

    @model_validator(mode="after")
    def _check_dates(self) -> DataConfig:
        if self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be on or after "
                f"start_date ({self.start_date})."
            )
        if self.end_date == self.start_date and self.frequency is not DataFrequency.H1:
            raise ValueError(
                "start_date and end_date may be equal only for intraday "
                "frequency '1h'; longer bar frequencies need a wider date range."
            )
        return self

    @model_validator(mode="after")
    def _check_frequency_supported_by_every_instrument(self) -> DataConfig:
        sources = {instrument.source for instrument in self.instruments}
        allowed = compatible_frequencies_for_sources(sources)
        if self.frequency not in allowed:
            remote = sorted(
                (source for source in sources if source is not DataSourceName.CSV),
                key=str,
            )
            details = ", ".join(
                f"{source}: {sorted(_SOURCE_SUPPORTED_FREQUENCIES[source])}"
                for source in remote
            )
            raise ValueError(
                f"frequency '{self.frequency}' is not supported by every "
                "instrument's source. Supported frequencies per remote "
                f"source: {details or '(none)'}."
            )
        return self

    @model_validator(mode="after")
    def _check_intraday_frequency_requires_uniform_calendar(self) -> DataConfig:
        """Reject a sub-daily mixed-calendar universe.

        Verified-closure handling (:mod:`quantlab.data.closures`) only
        operates at daily frequency -- an intraday, mixed-calendar universe
        (e.g. Yahoo/XNYS '1h' alongside Binance/24-7 '1h') has no equivalent
        mechanism, so a held equity inevitably hits an hour the other
        calendar trades but it has no return for, failing deep inside
        accounting with a confusing error instead of at config load. Until a
        genuine per-session intraday timeline exists, refuse this
        combination explicitly.
        """
        if self.frequency is DataFrequency.H1:
            calendars = {instrument.calendar for instrument in self.instruments}
            if len(calendars) > 1:
                raise ValueError(
                    "Intraday frequency '1h' does not support a "
                    f"mixed-calendar universe ({sorted(calendars)}) -- "
                    "verified-closure handling only operates at daily "
                    "frequency, so a held equity would inevitably hit an "
                    "hour another calendar trades but it has no return for. "
                    "Use a single shared calendar for '1h', or daily "
                    "frequency for a mixed-calendar universe."
                )
        return self

    @model_validator(mode="after")
    def _note_mixed_calendars_use_native_calendar_features(self) -> DataConfig:
        """Informational: a mixed-calendar universe uses native-calendar features.

        Every built-in strategy's own rolling-window signal (momentum
        lookback, technical indicator, mean-reversion indicator, a pairs
        spread's own hedge fit) and the ADV window are computed on each
        instrument's own native session dates before aligning back onto
        the combined timeline (see :func:`quantlab.features.
        native_calendar.compute_native_then_align`) -- a session-bound
        instrument sharing a combined timeline with an always-open one
        (e.g. equities alongside crypto) is not diluted by the always-open
        instrument's own extra sessions in what actually gets traded. Not
        yet covered: the inverse-volatility/volatility-targeting
        allocators, and most Strategy Explorer diagnostics (see
        docs/limitations.md for the precise, still-open list). Kept at
        ``logger.warning`` (surfaced in the dashboard sidebar) purely so a
        user configuring a mixed-calendar portfolio is still made aware of
        both what is covered and what remains open.
        """
        calendars = {instrument.calendar for instrument in self.instruments}
        if len(calendars) > 1:
            logger.warning(
                "Instruments span more than one calendar (%s): every "
                "built-in strategy's own rolling-window signal (momentum "
                "lookback, technical indicator, mean-reversion indicator, "
                "a pairs spread's own hedge fit) and the ADV window are "
                "computed on each instrument's own native calendar before "
                "aligning onto the combined timeline. Not yet covered: "
                "the inverse-volatility/volatility-targeting allocators, "
                "and most Strategy Explorer diagnostics. See "
                "docs/limitations.md.",
                sorted(calendars),
            )
        return self

    @model_validator(mode="after")
    def _check_forward_fill_limit_is_relevant(self) -> DataConfig:
        if (
            self.missing_value_policy is not MissingValuePolicy.FORWARD_FILL
            and self.forward_fill_limit != 1
        ):
            raise ValueError(
                "forward_fill_limit can differ from its default only when "
                "missing_value_policy is 'forward_fill'."
            )
        return self

    @model_validator(mode="after")
    def _check_bundled_demo_data_requires_a_csv_instrument(self) -> DataConfig:
        """Restrict the bundled synthetic-data fallback to csv instruments."""
        if self.use_bundled_demo_data and not any(
            instrument.source is DataSourceName.CSV for instrument in self.instruments
        ):
            raise ValueError(
                "use_bundled_demo_data: true has no effect without at least "
                "one csv-sourced instrument — the bundled-demo-CSV fallback "
                "is only ever consulted by the csv loader."
            )
        return self


class StrategyConfig(_StrictModel):
    """Strategy selection and its free-form parameter dictionary."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: Price series used to generate signals -- execution/costs always use
    #: the raw close regardless of this setting (see docs/data_pipeline.md);
    #: this only controls what a strategy's own generate_signals() sees.
    signal_price_type: Literal["adjusted_close", "close"] = "adjusted_close"

    @model_validator(mode="after")
    def _reject_price_type_in_parameters(self) -> StrategyConfig:
        if "price_type" in self.parameters:
            raise ValueError(
                "strategy.parameters must not set 'price_type' directly -- "
                "use strategy.signal_price_type instead, so the value the "
                "strategy actually uses always matches what is recorded in "
                "resolved_config."
            )
        return self


class PortfolioConfig(_StrictModel):
    """Portfolio construction and constraint settings."""

    allocator: str = Field(default="equal_weight")
    maximum_weight: float | None = Field(default=None, gt=0.0, le=1.0)
    # Non-convex minimum-size and position-count limits apply to target
    # portfolios. Turnover-limited transitional weights may temporarily differ.
    target_minimum_weight: float | None = Field(default=None, ge=0.0)
    maximum_gross_exposure: float | None = Field(default=None, gt=0.0)
    maximum_net_exposure: float | None = Field(default=None, ge=0.0)
    target_maximum_positions: int | None = Field(default=None, gt=0)
    long_only: bool = False
    # Volatility targeting
    target_volatility: float | None = Field(default=None, gt=0.0)
    volatility_window: int = Field(default=63, gt=1)
    maximum_leverage: float = Field(default=1.0, gt=0.0)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    # Maximum L1 weight change allowed on any single row. A rebalance whose
    # full target exceeds this in one step lands partially and keeps
    # closing the remaining gap over subsequent rows, so a whole rebalance
    # can take several rows to fully execute.
    maximum_turnover: float | None = Field(default=None, gt=0.0)
    # Evolve executed weights forward by organic price drift between real
    # trades (see quantlab.backtesting.accounting.apply_weight_drift)
    # instead of holding them constant until the next scheduled rebalance --
    # a real portfolio's weights genuinely do drift with each asset's own
    # price move between trades; holding them constant is only ever exactly
    # correct when the schedule itself rebalances every single period. A
    # genuinely scheduled rebalance still always trades toward its
    # freshly-decided target regardless of drift -- landing there in one
    # row unless maximum_turnover caps the move, in which case it lands
    # partially and keeps closing the gap on subsequent rows. `False`
    # remains available as an explicit legacy/reproducibility escape
    # hatch, not the standard path.
    model_weight_drift: bool = True

    @model_validator(mode="after")
    def _reject_unimplemented_custom_rebalancing(self) -> PortfolioConfig:
        """Reject the reserved custom cadence before backtest execution."""
        if self.rebalance_frequency is RebalanceFrequency.CUSTOM:
            raise ValueError(
                "rebalance_frequency 'custom' is not implemented. Choose "
                "daily, weekly, monthly, or quarterly."
            )
        return self

    @model_validator(mode="after")
    def _check_weight_bounds(self) -> PortfolioConfig:
        if (
            self.target_minimum_weight is not None
            and self.maximum_weight is not None
            and self.target_minimum_weight > self.maximum_weight
        ):
            raise ValueError(
                f"target_minimum_weight ({self.target_minimum_weight}) must "
                f"not exceed maximum_weight ({self.maximum_weight})."
            )
        return self

    @model_validator(mode="after")
    def _check_volatility_targeting_requires_target_volatility(self) -> PortfolioConfig:
        """Reject an implicit target rather than silently defaulting to 12%.

        ``volatility_targeting`` sizes leverage directly off this number --
        a value a user reading the YAML would never see must never drive
        real leverage decisions, so it must always be explicit.
        """
        if self.allocator == "volatility_targeting" and self.target_volatility is None:
            raise ValueError(
                "allocator 'volatility_targeting' requires an explicit "
                "target_volatility (e.g. 0.12 for 12% annualised) — there is "
                "no implicit default."
            )
        return self

    @model_validator(mode="after")
    def _warn_if_positions_and_weight_cap_infeasible(self) -> PortfolioConfig:
        """Warn when the position/weight caps can't reach the gross ceiling.

        E.g. ``target_maximum_positions=2`` with ``maximum_weight=0.30`` can
        never exceed 60% gross. This warns rather than rejects because the
        resulting under-investment may be intentional.
        """
        if (
            self.target_maximum_positions is not None
            and self.maximum_weight is not None
        ):
            achievable = self.target_maximum_positions * self.maximum_weight
            ceiling = min(self.maximum_gross_exposure or 1.0, self.maximum_leverage)
            if achievable < ceiling - 1e-9:
                logger.warning(
                    "target_maximum_positions (%d) x maximum_weight (%.4f) = "
                    "%.4f can never reach the gross-exposure ceiling (%.4f) "
                    "— this portfolio will structurally under-invest "
                    "regardless of the signal, unless that is intentional.",
                    self.target_maximum_positions,
                    self.maximum_weight,
                    achievable,
                    ceiling,
                )
        return self


class ExecutionConfig(_StrictModel):
    """Transaction-cost parameters in basis points."""

    commission_bps: float = Field(default=0.0, ge=0.0)
    spread_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    slippage_model: SlippageModelName = SlippageModelName.CONSTANT
    # Optional volume-based slippage coefficient.
    impact_coefficient: float = Field(default=0.1, ge=0.0)


class BacktestConfig(_StrictModel):
    """Accounting and benchmark settings (the ``backtest:`` block)."""

    initial_capital: float = Field(default=100_000.0, gt=0.0)
    benchmark_kind: BenchmarkKind = BenchmarkKind.SYMBOL
    benchmark: InstrumentConfig | None = Field(
        default=None,
        description=(
            "Benchmark instrument (symbol/source/calendar), only valid when "
            "benchmark_kind='symbol'. If its symbol is already a tradable "
            "instrument under data.instruments, its source and calendar must "
            "match exactly — the already-loaded series is reused rather than "
            "downloaded twice."
        ),
    )
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    periods_per_year: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_benchmark_configuration(self) -> BacktestConfig:
        if (
            self.benchmark_kind is not BenchmarkKind.SYMBOL
            and self.benchmark is not None
        ):
            raise ValueError("benchmark is only valid when benchmark_kind='symbol'.")
        return self


class ValidationConfig(_StrictModel):
    """Out-of-sample validation settings."""

    method: ValidationMethod = ValidationMethod.HOLDOUT
    # Chronological holdout ratios. Train ratio inferred as remainder.
    validation_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)
    test_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)
    # Walk-forward windows in number of periods.
    train_window: int | None = Field(default=None, gt=0)
    validation_window: int | None = Field(default=None, gt=0)
    test_window: int | None = Field(default=None, gt=0)
    # Advance between consecutive folds' train windows. None defaults to
    # test_window (contiguous, non-overlapping test blocks). A smaller step
    # overlaps test blocks for denser evaluation; step must not exceed
    # test_window (see walk_forward_windows()'s docstring for why).
    step: int | None = Field(default=None, gt=0)
    expanding: bool = True
    optimization_metric: OptimizationMetric = OptimizationMetric.SHARPE
    # Optional strategy-parameter candidates for walk-forward selection. When
    # omitted, QuantLab supplies a compact strategy-specific default grid.
    parameter_grid: dict[str, list[Any]] | None = None

    @model_validator(mode="after")
    def _check_ratios(self) -> ValidationConfig:
        val = self.validation_ratio or 0.0
        test = self.test_ratio or 0.0
        if val + test >= 1.0:
            raise ValueError(
                "validation_ratio + test_ratio must leave room for training "
                f"(got {val} + {test} >= 1.0)."
            )
        if self.method is ValidationMethod.HOLDOUT and val > 0.0 and test <= 0.0:
            raise ValueError(
                "validation_ratio has no effect without a positive test_ratio "
                "when validation.method is 'holdout'."
            )
        if self.method is ValidationMethod.WALK_FORWARD and self.step is not None:
            # Mirrors resolve_walk_forward_windows()'s own default (126) --
            # kept duplicated rather than imported since that function lives
            # in quantlab.validation.walk_forward, which itself imports this
            # config module.
            effective_test_window = self.test_window or 126
            if self.step > effective_test_window:
                raise ValueError(
                    f"step ({self.step}) must not exceed test_window "
                    f"({effective_test_window}) -- a larger step would skip "
                    "dates between consecutive folds' test blocks entirely."
                )
        if self.method is ValidationMethod.WALK_FORWARD and (
            self.validation_ratio is not None or self.test_ratio is not None
        ):
            raise ValueError(
                "validation_ratio and test_ratio apply only to method 'holdout'; "
                "remove them when validation.method is 'walk_forward'."
            )
        if self.parameter_grid is not None:
            if self.method is not ValidationMethod.WALK_FORWARD:
                raise ValueError(
                    "parameter_grid applies only to validation.method 'walk_forward'."
                )
            for name, candidates in self.parameter_grid.items():
                if not name.strip():
                    raise ValueError("parameter_grid names must be non-empty strings.")
                if not candidates:
                    raise ValueError(
                        f"parameter_grid.{name} must contain at least one candidate."
                    )
                if name == "price_type":
                    raise ValueError(
                        "parameter_grid must not include 'price_type' -- it is a "
                        "structural choice, set strategy.signal_price_type instead."
                    )
                # Caught here, at YAML load time, rather than deep inside
                # walk_forward.py's own execution-time duplicate check --
                # same "reject at the door" convention as StressTestSettings.
                _no_duplicate_values(candidates, name=f"parameter_grid.{name}")
        if self.method is not ValidationMethod.WALK_FORWARD:
            window_fields = {
                "train_window": self.train_window,
                "validation_window": self.validation_window,
                "test_window": self.test_window,
                "step": self.step,
            }
            provided = sorted(k for k, v in window_fields.items() if v is not None)
            if provided:
                verb = "apply" if len(provided) > 1 else "applies"
                pronoun = "them" if len(provided) > 1 else "it"
                raise ValueError(
                    f"{', '.join(provided)} {verb} only to validation.method "
                    f"'walk_forward'; remove {pronoun} or set method: walk_forward."
                )
        return self


class ReproducibilityConfig(_StrictModel):
    """Determinism controls. One seed drives every stochastic step."""

    random_seed: int = Field(default=42, ge=0)


def _no_duplicate_values(values: list[Any], *, name: str) -> list[Any]:
    """Reject a candidate list containing the same value more than once."""
    seen: list[Any] = []
    for value in values:
        if value in seen:
            raise ValueError(f"{name} must not contain duplicate values.")
        seen.append(value)
    return values


class StressTestSettings(_StrictModel):
    """Scenario magnitudes for the ``robustness stress-test`` / orchestrator run.

    Each list is a set of scenario magnitudes to evaluate independently
    (one row per value) -- an empty list disables that scenario TYPE
    entirely, the same "empty means nothing to run" convention as
    ``validation.parameter_grid``. Defaults reproduce exactly the fixed
    scenario set this project ran before these became configurable.
    """

    enabled: bool = False
    commission_multipliers: list[StrictFloat] = Field(
        default_factory=lambda: [2.0, 5.0]
    )
    slippage_multipliers: list[StrictFloat] = Field(default_factory=lambda: [2.0])
    execution_delays: list[StrictInt] = Field(default_factory=lambda: [1])
    best_days_removed: list[StrictInt] = Field(default_factory=lambda: [10])
    #: Number of symbols dropped from the tail of the universe, per
    #: scenario. A scenario whose universe is too small for that count is
    #: recorded with status="failed", never silently omitted.
    reduce_universe_by: list[StrictInt] = Field(default_factory=lambda: [1])

    @model_validator(mode="after")
    def _check_magnitudes(self) -> StressTestSettings:
        # Multipliers below are genuine stress scenarios (elevated costs),
        # not arbitrary perturbations -- 1.0 or below would silently test
        # cheaper-than-baseline execution instead. Delays/day-counts/removed
        # symbols are plain positive counts.
        for name, values, gt in (
            ("commission_multipliers", self.commission_multipliers, 1.0),
            ("slippage_multipliers", self.slippage_multipliers, 1.0),
            ("execution_delays", self.execution_delays, 0),
            ("best_days_removed", self.best_days_removed, 0),
            ("reduce_universe_by", self.reduce_universe_by, 0),
        ):
            for value in values:
                if value <= gt:
                    raise ValueError(f"{name} entries must be greater than {gt}.")
            _no_duplicate_values(values, name=name)
        return self


class BootstrapSettings(_StrictModel):
    """Settings for a block-bootstrap resample of realised returns."""

    enabled: bool = False
    n_iterations: int = Field(default=1000, gt=0)
    block_size: int = Field(default=1, gt=0)
    #: Central percentile interval width reported by BootstrapResult.summary()
    #: (0.90 -> the 5th/95th percentiles).
    confidence_level: float = Field(default=0.90, gt=0.0, lt=1.0)


class PermutationTestSettings(_StrictModel):
    """Settings for the random-sign Monte Carlo permutation test."""

    enabled: bool = False
    n_iterations: int = Field(default=1000, gt=0)


class SensitivitySettings(_StrictModel):
    """Two-parameter sweep for the parameter-sensitivity heatmap.

    ``parameters`` must name exactly two strategy parameters (the sweep's x
    and y axes) with their candidate values; unlike ``validation.
    parameter_grid``, there is no default here — sensitivity has no
    meaningful "no parameters selected" fallback.
    """

    enabled: bool = False
    parameters: dict[str, list[Any]] | None = None

    @model_validator(mode="after")
    def _check_parameters_shape(self) -> SensitivitySettings:
        if self.parameters is None:
            if self.enabled:
                raise ValueError(
                    "robustness.sensitivity.enabled is true but parameters is "
                    "not set -- sensitivity has no meaningful default (unlike "
                    "validation.parameter_grid), so the x/y axes and their "
                    "candidate values must be given explicitly."
                )
            return self
        if len(self.parameters) != 2:
            raise ValueError(
                "robustness.sensitivity.parameters must name exactly 2 "
                f"parameters (the x and y axes), got {len(self.parameters)}."
            )
        for name, candidates in self.parameters.items():
            if not name.strip():
                raise ValueError(
                    "robustness.sensitivity.parameters names must be non-empty strings."
                )
            if not candidates:
                raise ValueError(
                    f"robustness.sensitivity.parameters.{name} must "
                    "contain at least one candidate value."
                )
            if name == "price_type":
                raise ValueError(
                    "robustness.sensitivity.parameters must not include "
                    "'price_type' -- it is a structural choice, set "
                    "strategy.signal_price_type instead."
                )
            # Caught here, at YAML load time, rather than deep inside
            # parameter_sensitivity.py's own execution-time duplicate
            # check -- same "reject at the door" convention as
            # StressTestSettings/ValidationConfig.parameter_grid.
            _no_duplicate_values(
                candidates, name=f"robustness.sensitivity.parameters.{name}"
            )
        return self


class RobustnessConfig(_StrictModel):
    """Optional CLI/dashboard robustness-suite settings (``robustness:``).

    Absent from YAML means every technique stays disabled — running a
    backtest or walk-forward is unaffected either way.
    """

    stress_test: StressTestSettings = Field(default_factory=StressTestSettings)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    permutation_test: PermutationTestSettings = Field(
        default_factory=PermutationTestSettings
    )
    sensitivity: SensitivitySettings = Field(default_factory=SensitivitySettings)


class OutputConfig(_StrictModel):
    """Where and what a run saves (``output:``).

    ``directory`` overrides the default ``reports/generated/<experiment_
    name>`` location used by every CLI command (``backtest``'s own
    ``--output`` flag still takes priority when given). The two artefact
    toggles skip only the presentation layer -- metrics.json/trades.csv/
    equity_curve.csv and every other numeric artefact are always written
    regardless. ``quantlab report`` does NOT depend on those saved
    artefacts to regenerate the HTML report later: it reloads the data and
    re-runs the backtest from this same config, and only reuses previously
    saved walk-forward/stress/bootstrap/permutation-test/sensitivity
    EVIDENCE when its own provenance check against the fresh run still
    passes.
    """

    directory: str | None = None
    save_html_report: bool = True
    save_figures: bool = True

    @field_validator("directory")
    @classmethod
    def _reject_blank_directory(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "output.directory must not be empty or whitespace-only -- omit "
                "the field entirely to use the default location instead."
            )
        return value


class ExperimentConfig(_StrictModel):
    """Top-level, reproducible description of a single experiment.

    Composes the sub-configs mirroring the YAML layout. Flat, read-only
    accessor properties are also provided for convenience.
    """

    experiment_name: str
    data: DataConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig
    )
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    #: Optional overrides for the HTML report's auto-generated research
    #: question/hypothesis text (quantlab.reporting.research_summary). When
    #: unset, that text is synthesized from the strategy name/config as
    #: before -- unchanged behaviour for every experiment that doesn't set
    #: these.
    research_question: str | None = None
    hypothesis: str | None = None

    @field_validator("research_question", "hypothesis")
    @classmethod
    def _reject_blank_research_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "research_question/hypothesis must not be empty or "
                "whitespace-only -- omit the field entirely to use the "
                "auto-generated text instead."
            )
        return value

    @field_validator("experiment_name")
    @classmethod
    def _validate_experiment_name(cls, value: str) -> str:
        """Reject names that are unsafe as report-directory components."""
        _validate_path_component(value, field_name="experiment_name")
        return value

    @model_validator(mode="after")
    def _check_component_names(self) -> ExperimentConfig:
        """Validate registered components and strategy-specific parameters.

        Imports stay local because the portfolio and strategy registries import
        this module while they are initialised.
        """
        from quantlab.portfolio.allocator import available_allocators
        from quantlab.strategies import (
            available_strategies,
            strategy_parameter_names,
            strategy_sweepable_parameter_names,
            validate_strategy_parameters,
        )

        allocators = available_allocators()
        if self.portfolio.allocator not in allocators:
            raise ValueError(
                f"Unknown portfolio.allocator '{self.portfolio.allocator}'. "
                f"Registered allocators: {allocators}."
            )
        strategies = available_strategies()
        if self.strategy.name not in strategies:
            raise ValueError(
                f"Unknown strategy.name '{self.strategy.name}'. "
                f"Registered strategies: {strategies}."
            )
        # Validate parameters without constructing the strategy.
        validate_strategy_parameters(self.strategy.name, self.strategy.parameters)
        parameter_grid = self.validation.parameter_grid
        if parameter_grid is not None:
            accepted = strategy_parameter_names(self.strategy.name)
            unknown = sorted(set(parameter_grid) - accepted)
            if unknown:
                raise ValueError(
                    f"Unknown validation.parameter_grid key(s) {unknown} for "
                    f"strategy '{self.strategy.name}'. Accepted parameters: "
                    f"{sorted(accepted)}."
                )
            names = list(parameter_grid)
            for values in itertools.product(*(parameter_grid[name] for name in names)):
                validate_strategy_parameters(
                    self.strategy.name,
                    {
                        **self.strategy.parameters,
                        **dict(zip(names, values, strict=True)),
                    },
                )
        sensitivity_parameters = self.robustness.sensitivity.parameters
        if sensitivity_parameters is not None:
            # Boolean/structural parameters (e.g. long_only) are excluded:
            # sweeping one changes which other parameters are even
            # meaningful, so sensitivity treats them as fixed, matching the
            # default walk-forward grid's own rule.
            accepted = strategy_sweepable_parameter_names(self.strategy.name)
            unknown = sorted(set(sensitivity_parameters) - accepted)
            if unknown:
                raise ValueError(
                    f"Unknown or unsweepable robustness.sensitivity.parameters "
                    f"key(s) {unknown} for strategy '{self.strategy.name}'. "
                    f"Accepted parameters: {sorted(accepted)}."
                )
            # Same value-combination check as validation.parameter_grid above:
            # a name being sweepable doesn't mean every candidate value (or
            # combination across the two axes) is actually valid for this
            # strategy (e.g. lookback_period: [0], or a combination this
            # strategy's own validator rejects) -- catch it here, at config
            # load, rather than only once the sensitivity sweep actually runs.
            sensitivity_names = list(sensitivity_parameters)
            for values in itertools.product(
                *(sensitivity_parameters[name] for name in sensitivity_names)
            ):
                validate_strategy_parameters(
                    self.strategy.name,
                    {
                        **self.strategy.parameters,
                        **dict(zip(sensitivity_names, values, strict=True)),
                    },
                )
        # A pairs strategy can trade only symbols present in the loaded universe.
        if self.strategy.name == "pairs_trading":
            symbol_a = str(self.strategy.parameters["symbol_a"]).strip().upper()
            symbol_b = str(self.strategy.parameters["symbol_b"]).strip().upper()
            missing = [s for s in (symbol_a, symbol_b) if s not in self.data.symbols]
            if missing:
                raise ValueError(
                    f"pairs_trading symbol(s) {missing} not in data.symbols "
                    f"{self.data.symbols}. Add them to data.symbols so they "
                    "are actually downloaded/loaded."
                )
            if self.portfolio.allocator != "signal_proportional":
                raise ValueError(
                    "pairs_trading requires portfolio.allocator "
                    "'signal_proportional' so the relative dollar-notional "
                    "hedge ratio implied by beta and current prices is preserved."
                )
            if (
                self.portfolio.maximum_weight is not None
                and self.portfolio.maximum_weight < 1.0
            ):
                raise ValueError(
                    "pairs_trading does not support portfolio.maximum_weight "
                    "because independently capping its legs changes the hedge ratio."
                )
            if (
                self.portfolio.target_minimum_weight is not None
                and self.portfolio.target_minimum_weight > 0.0
            ):
                raise ValueError(
                    "pairs_trading does not support target_minimum_weight because "
                    "dropping one small leg breaks the pair hedge."
                )
            if (
                self.portfolio.target_maximum_positions is not None
                and self.portfolio.target_maximum_positions < 2
            ):
                raise ValueError(
                    "pairs_trading requires target_maximum_positions >= 2."
                )
            if self.portfolio.long_only:
                raise ValueError("pairs_trading requires portfolio.long_only=false.")
        if self.strategy.name == "cross_sectional_momentum":
            parameters = self.strategy.parameters
            top_fraction = float(parameters.get("top_fraction", 0.25))
            long_short = bool(parameters.get("long_short", False))
            bottom_fraction = (
                float(parameters.get("bottom_fraction", 0.25)) if long_short else 0.0
            )
            available = len(self.data.symbols)
            top_count = max(1, int(available * top_fraction)) if top_fraction else 0
            bottom_count = (
                max(1, int(available * bottom_fraction)) if bottom_fraction else 0
            )
            if top_count + bottom_count > available:
                raise ValueError(
                    "cross_sectional_momentum needs enough distinct symbols for "
                    f"its top/bottom selections ({top_count} + {bottom_count} "
                    f"requested, {available} configured)."
                )
            scaling = parameters.get("signal_scaling", "binary")
            if scaling != "binary" and self.portfolio.allocator == "equal_weight":
                raise ValueError(
                    "Non-binary cross_sectional_momentum signals require an "
                    "allocator that preserves signal magnitude; equal_weight "
                    "keeps only signs."
                )
        if self.strategy.name == "time_series_momentum":
            scaling = self.strategy.parameters.get("signal_scaling", "binary")
            if scaling != "binary" and self.portfolio.allocator == "equal_weight":
                raise ValueError(
                    "Non-binary time_series_momentum signals require an allocator "
                    "that preserves signal magnitude; equal_weight keeps only signs."
                )
            if scaling == "volatility_adjusted" and self.portfolio.allocator in {
                "inverse_volatility",
                "volatility_targeting",
            }:
                raise ValueError(
                    "volatility_adjusted time_series_momentum must not be combined "
                    "with an allocator that applies inverse-volatility sizing again."
                )
        return self

    @model_validator(mode="after")
    def _check_mixed_calendars_require_explicit_periods_per_year(
        self,
    ) -> ExperimentConfig:
        """Forbid inferring one annualisation factor from multiple calendars.

        `DataConfig` alone can't see `backtest.periods_per_year`, so this
        cross-field check lives here rather than on a sub-config.
        """
        calendars = {instrument.calendar for instrument in self.data.instruments}
        if len(calendars) > 1 and self.backtest.periods_per_year is None:
            raise ValueError(
                "Mixed market calendars require an explicit "
                "backtest.periods_per_year — QuantLab cannot infer one "
                "annualisation factor when instruments trade on different "
                f"calendars ({sorted(calendars)})."
            )
        return self

    @model_validator(mode="after")
    def _check_benchmark_matches_overlapping_instrument(self) -> ExperimentConfig:
        """A benchmark that duplicates a tradable symbol must match it exactly.

        Downstream data is keyed by symbol alone, so an inconsistent
        source/calendar override for an overlapping benchmark would be
        ambiguous — reject it here rather than silently picking one.
        """
        benchmark = self.backtest.benchmark
        if benchmark is None:
            return self
        for instrument in self.data.instruments:
            if instrument.symbol != benchmark.symbol:
                continue
            if (
                instrument.source != benchmark.source
                or instrument.calendar != benchmark.calendar
            ):
                raise ValueError(
                    f"backtest.benchmark symbol {benchmark.symbol!r} is "
                    f"already a tradable instrument with source="
                    f"{instrument.source!r}, calendar={instrument.calendar!r}; "
                    "the benchmark override must match exactly or be omitted "
                    "(the tradable instrument's data is then reused automatically)."
                )
            break
        return self

    @model_validator(mode="after")
    def _check_benchmark_frequency_supported_by_its_own_source(
        self,
    ) -> ExperimentConfig:
        """An external benchmark's source must support the configured frequency too.

        `DataConfig._check_frequency_supported_by_every_instrument` only sees
        `data.instruments` — a benchmark that reuses a tradable instrument's
        data is already covered there (and cross-checked for source/calendar
        consistency by `_check_benchmark_matches_overlapping_instrument`
        above), but a benchmark outside the tradable universe has its own,
        otherwise-unchecked source. Without this, e.g. frequency: '1mo' with
        an external Binance benchmark would be silently accepted here only
        to fail later, confusingly, at download time.
        """
        benchmark = self.backtest.benchmark
        if benchmark is None or benchmark.symbol in self.data.symbols:
            return self
        allowed = compatible_frequencies_for_sources([benchmark.source])
        if self.data.frequency not in allowed:
            supported = _SOURCE_SUPPORTED_FREQUENCIES.get(
                benchmark.source, set(DataFrequency)
            )
            raise ValueError(
                f"frequency '{self.data.frequency}' is not supported by "
                f"backtest.benchmark's source {benchmark.source!r} (symbol "
                f"{benchmark.symbol!r}). Supported frequencies: "
                f"{sorted(supported)}."
            )
        return self

    # ----------------------------------------------------------------- #
    # Construction helpers
    # ----------------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file.

        Args:
            path: Path to a YAML config file.

        Returns:
            A validated :class:`ExperimentConfig`.

        Raises:
            InvalidConfigurationError: If the file is missing, unreadable, or
                fails validation.
        """
        path = Path(path)
        if not path.is_file():
            raise InvalidConfigurationError(
                f"Config file not found: {path}. "
                f"Check the path or create it under configs/."
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise InvalidConfigurationError(
                f"Could not read YAML config {path}: {exc}"
            ) from exc
        try:
            raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
        except yaml.YAMLError as exc:
            raise InvalidConfigurationError(
                f"Could not parse YAML config {path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise InvalidConfigurationError(
                f"Config {path} must be a YAML mapping, got {type(raw).__name__}."
            )
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Validate a plain dict into an :class:`ExperimentConfig`."""
        try:
            return cls.model_validate(data)
        except Exception as exc:  # pydantic.ValidationError and friends
            raise InvalidConfigurationError(
                f"Invalid experiment configuration: {exc}"
            ) from exc

    def to_yaml(self, path: str | Path) -> Path:
        """Serialise this config to YAML (used to snapshot each run).

        Args:
            path: Destination file.

        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    # ----------------------------------------------------------------- #
    # Derived values
    # ----------------------------------------------------------------- #
    @property
    def periods_per_year(self) -> int:
        """Resolve annualisation factor.

        Priority: explicit ``backtest.periods_per_year`` > frequency lookup >
        daily default. The frequency lookup distinguishes 24/7 markets from
        session-bound ones, off the calendar every instrument shares — a
        validator guarantees a single shared calendar whenever
        ``backtest.periods_per_year`` isn't set explicitly.
        """
        if self.backtest.periods_per_year is not None:
            return self.backtest.periods_per_year
        # Local import: avoids a top-level quantlab.data <-> quantlab.config cycle.
        from quantlab.data.calendar import is_247, uniform_calendar

        calendar = uniform_calendar(
            instrument.calendar for instrument in self.data.instruments
        )
        assert calendar is not None  # enforced by _check_mixed_calendars_...
        table = (
            CRYPTO_FREQUENCY_TO_PERIODS_PER_YEAR
            if is_247(calendar)
            else FREQUENCY_TO_PERIODS_PER_YEAR
        )
        return table.get(self.data.frequency, TRADING_DAYS_PER_YEAR)

    # ----------------------------------------------------------------- #
    # Flat convenience accessors
    # ----------------------------------------------------------------- #
    @property
    def data_source(self) -> str:
        """Data source label: a single name, or ``"mixed (a, b)"``."""
        sources = sorted(
            {str(instrument.source) for instrument in self.data.instruments}
        )
        if len(sources) == 1:
            return sources[0]
        return f"mixed ({', '.join(sources)})"

    @property
    def symbols(self) -> list[str]:
        """Normalised list of tickers to load."""
        return list(self.data.symbols)

    @property
    def start_date(self) -> date:
        """Inclusive start date of the experiment."""
        return self.data.start_date

    @property
    def end_date(self) -> date:
        """Inclusive end date of the experiment."""
        return self.data.end_date

    @property
    def frequency(self) -> str:
        """Bar frequency (e.g. ``"1d"``)."""
        return self.data.frequency

    @property
    def strategy_name(self) -> str:
        """Registered strategy name to instantiate."""
        return self.strategy.name

    @property
    def strategy_parameters(self) -> dict[str, Any]:
        """Free-form parameter dict passed to the strategy constructor."""
        return dict(self.strategy.parameters)

    @property
    def initial_capital(self) -> float:
        """Starting equity for the backtest."""
        return self.backtest.initial_capital

    @property
    def risk_free_rate(self) -> float:
        """Annual risk-free rate used by excess-return metrics."""
        return self.backtest.risk_free_rate

    @property
    def benchmark_symbol(self) -> str | None:
        """Symbol to compare performance against, if any."""
        return self.backtest.benchmark.symbol if self.backtest.benchmark else None

    @property
    def benchmark_source(self) -> DataSourceName | None:
        """Data source of the benchmark instrument, if any."""
        return self.backtest.benchmark.source if self.backtest.benchmark else None

    @property
    def benchmark_calendar(self) -> str | None:
        """Calendar of the benchmark instrument, if any."""
        return self.backtest.benchmark.calendar if self.backtest.benchmark else None

    @property
    def benchmark_kind(self) -> BenchmarkKind:
        """Benchmark construction used for relative metrics."""
        return self.backtest.benchmark_kind

    @property
    def benchmark_label(self) -> str | None:
        """Human-readable benchmark name, or ``None`` when disabled."""
        kind = self.backtest.benchmark_kind
        if kind is BenchmarkKind.SYMBOL:
            return self.benchmark_symbol
        if kind is BenchmarkKind.EQUAL_WEIGHT:
            return "Equal weight"
        if kind is BenchmarkKind.FIRST_ASSET:
            return self.symbols[0]
        return "Cash"

    @property
    def commission_bps(self) -> float:
        """Commission in basis points of traded notional."""
        return self.execution.commission_bps

    @property
    def slippage_bps(self) -> float:
        """Slippage in basis points of traded notional."""
        return self.execution.slippage_bps

    @property
    def spread_bps(self) -> float:
        """Full quoted spread in basis points."""
        return self.execution.spread_bps

    @property
    def rebalance_frequency(self) -> str:
        """Configured rebalancing cadence as a string."""
        return str(self.portfolio.rebalance_frequency)

    @property
    def random_seed(self) -> int:
        """Seed applied to every stochastic step for reproducibility."""
        return self.reproducibility.random_seed
