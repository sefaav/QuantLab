"""Execution-price derivation, isolated from the accounting engine.

The overnight/intraday return decomposition used by
``execution_timing="next_bar_open"`` (see :mod:`quantlab.backtesting.accounting`)
needs an *adjusted* open price -- the canonical schema carries a raw
``open``, but adjustment factors (splits, dividends) are only ever applied
to ``close``/``adjusted_close`` today. This module isolates how an adjusted
open is derived behind :class:`ExecutionPriceSource`, so the accounting
engine depends on the *protocol*, never a specific formula -- a future
point-in-time / corporate-actions pipeline is a drop-in replacement
implementation, requiring no change to the execution-timing model itself.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Mapping
from typing import Protocol

import numpy as np
import pandas as pd

from quantlab.config import ExecutionTiming, ExperimentConfig
from quantlab.constants import ADJUSTED_CLOSE, CLOSE, OPEN
from quantlab.data.base import pivot_field
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import BacktestError
from quantlab.features._validation import numeric_pandas, same_axes

#: Fingerprint for the built-in default -- never produced by
#: `execution_price_source_fingerprint` for an actual custom instance, so
#: it can never collide with one.
DEFAULT_EXECUTION_PRICE_SOURCE_FINGERPRINT = (
    "default (RatioApproximatedExecutionPrices)"
)


def _content_aware_repr(value: object) -> str:
    """``repr(value)``, except a DataFrame/Series/ndarray is hashed whole.

    Pandas/NumPy truncate their own ``repr`` for a large object (showing
    only the head/tail rows) -- using that truncated text directly would
    let two instances holding different DATA in the same-shaped middle
    rows collide on an identical fingerprint. Hashing every element
    instead of the printed text avoids that blind spot -- delegates to
    :meth:`~quantlab.data.storage.ParquetStorage.hash_frame`, the same
    content-hash convention already used for data provenance elsewhere
    (see ``engine.py``'s own ``_hash_adv``), which also folds in dtype/
    column schema so two same-shaped, same-VALUED frames of different
    dtypes don't collide either.
    """
    if isinstance(value, pd.DataFrame):
        return (
            f"DataFrame(shape={value.shape}, hash={ParquetStorage.hash_frame(value)})"
        )
    if isinstance(value, pd.Series):
        digest = ParquetStorage.hash_frame(value.to_frame())
        return f"Series(shape={value.shape}, hash={digest})"
    if isinstance(value, np.ndarray):
        content_digest = hashlib.sha256(value.tobytes()).hexdigest()
        return (
            f"ndarray(shape={value.shape}, dtype={value.dtype}, hash={content_digest})"
        )
    if isinstance(value, Mapping):
        # Recurse so a DataFrame/Series/ndarray NESTED inside a dict
        # attribute is content-hashed too, not just a top-level one --
        # sorted by each entry's own serialized form so two otherwise-
        # identical instances never collide-or-diverge purely from
        # dict insertion order or a non-natively-orderable key type
        # (e.g. mixed str/int keys, which plain `sorted(dict.items())`
        # would raise `TypeError` on).
        entries = sorted(
            (_content_aware_repr(k), _content_aware_repr(v)) for k, v in value.items()
        )
        return "{" + ", ".join(f"{k}: {v}" for k, v in entries) + "}"
    if isinstance(value, list | tuple):
        kind = "list" if isinstance(value, list) else "tuple"
        return f"{kind}([{', '.join(_content_aware_repr(item) for item in value)}])"
    return repr(value)


def _instance_state(instance: object) -> dict[str, object] | None:
    """Best-effort per-instance state, from ``__dict__`` AND ``__slots__``.

    Always merges both sources (never one OR the other, gated on which
    succeeds first) -- a leaf class can hold a real, if empty, ``__dict__``
    while its actual attributes live entirely in an inherited ``__slots__``
    mixin, so trusting ``__dict__`` alone would report "no state" for such
    an instance and collapse two differently-configured instances onto the
    same fingerprint. Walks the full MRO for slots, since a subclass's own
    ``__slots__`` never includes its parents'; an unset slot (e.g. one a
    constructor never reached before raising) is excluded rather than
    crashing this best-effort fingerprint on it. ``None`` only when the
    class exposes neither (e.g. ``__slots__ = ()``).
    """
    state: dict[str, object] = {}
    with contextlib.suppress(TypeError):
        state.update(vars(instance))
    for klass in type(instance).__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name.startswith("__"):
                continue
            try:
                state[name] = getattr(instance, name)
            except AttributeError:
                continue
    return state or None


def execution_price_source_fingerprint(execution_price_source: object | None) -> str:
    """Deterministic identity for a caller-supplied ``ExecutionPriceSource``.

    Class name alone (``type(x).__module__ + type(x).__qualname__``) is not
    enough: two DIFFERENT instances of the SAME class with different
    constructor parameters (e.g. two ``RatioApproximatedExecutionPrices``
    subclasses tuned with different dividend-adjustment thresholds) would
    otherwise fingerprint identically, letting a walk-forward checkpoint
    (or ``BacktestEngine``'s own ``config_yaml_reflects_execution``
    provenance) silently treat them as the same source even though they
    price the adjusted open differently. Folds in the instance's own
    per-attribute state (see :func:`_instance_state` -- ``__dict__`` or,
    for a ``__slots__``-only class, every slot across its full MRO; each
    value passed through :func:`_content_aware_repr`, not JSON-encoded,
    since an arbitrary attribute -- e.g. a DataFrame, or one nested
    inside a dict -- need not be JSON-serializable). A class exposing
    neither falls back to the class name alone.
    Best-effort, like ``engine.py``'s own ``_hash_adv`` -- not a
    cryptographic guarantee for a source that hides meaningful state
    outside both its ``__dict__`` and its ``__slots__`` (e.g. a closure
    captured over a plain function, or a value computed dynamically by a
    property rather than stored on the instance at all). An explicit,
    state-free ``RatioApproximatedExecutionPrices()`` instance -- never a
    subclass -- fingerprints identically to ``None``, since it is
    behaviorally identical to the default this module would have
    constructed internally either way.
    """
    if execution_price_source is None:
        return DEFAULT_EXECUTION_PRICE_SOURCE_FINGERPRINT
    cls = type(execution_price_source)
    state = _instance_state(execution_price_source)
    # An explicitly-passed instance of the exact built-in default class
    # (never a subclass, which could override `adjusted_open` or carry
    # real state) with no per-instance state at all is behaviorally
    # identical to the implicit `None` default this module constructs
    # internally -- collapsing it onto the SAME sentinel keeps
    # `WalkForwardValidator.rescore_with_costs`'s cache-staleness check
    # and a walk-forward checkpoint's own provenance comparison from
    # treating two numerically-identical runs as mismatched merely
    # because one spelled the default out explicitly.
    if cls is RatioApproximatedExecutionPrices and not state:
        return DEFAULT_EXECUTION_PRICE_SOURCE_FINGERPRINT
    class_name = f"{cls.__module__}.{cls.__qualname__}"
    state_repr = (
        repr({key: _content_aware_repr(value) for key, value in sorted(state.items())})
        if state is not None
        else repr(None)
    )
    digest = hashlib.sha256(state_repr.encode("utf-8")).hexdigest()[:16]
    return f"{class_name}:{digest}"


class ExecutionPriceSource(Protocol):
    """Derives the price series `next_bar_open` execution needs."""

    def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the ``dates × symbols`` adjusted-open matrix for ``data``."""
        ...


class RatioApproximatedExecutionPrices:
    """``adjusted_open = open * (adjusted_close / close)``, per bar.

    An approximation, not an independently-verified corporate-actions
    pipeline -- QuantLab has none today (``adjusted_close`` is itself taken
    directly from the data vendor, e.g. Yahoo's own "Adj Close"; see
    ``quantlab.data.yahoo``). This ratio is exact for a pure stock split
    (which scales every OHLC field of that bar identically), but only
    approximate for a cash-dividend adjustment: the true ex-dividend price
    drop happens once, at that day's open, not proportionally throughout
    the bar the way this ratio implicitly assumes. The resulting error's
    order of magnitude is the dividend yield of that one bar -- not
    independently verified to be strictly bounded or non-accumulating
    across other bars, only assumed small for an ordinary cash dividend;
    treat as a known, disclosed imprecision rather than a proven error
    bound. Only used for a daily bar -- see
    ``ExperimentConfig._check_next_bar_open_requires_a_daily_frequency``
    for why a resampled weekly/monthly bar is rejected outright instead
    of silently applying this same approximation to a temporally
    incoherent open/close pair.
    """

    def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the ratio-approximated adjusted-open matrix for ``data``."""
        open_matrix = pivot_field(data, OPEN)
        close_matrix = pivot_field(data, CLOSE)
        adjusted_close_matrix = pivot_field(data, ADJUSTED_CLOSE)
        # All three pivoted independently from the same `data` -- validate
        # every one against the others (not just close vs. open) before any
        # positional/elementwise arithmetic: a silent axis mismatch here
        # (e.g. adjusted_close's columns in a different order) would
        # silently compute a wrong ratio for every symbol, not raise.
        same_axes(
            open_matrix,
            close_matrix,
            adjusted_close_matrix,
            names=("close", "adjusted_close"),
        )
        close_values = close_matrix.to_numpy(dtype=float)
        # A zero raw close makes this bar's ratio undefined -- masked to
        # NaN (never raised here) for the SAME reason `same_bar_close`
        # tolerates a missing/undefined return on a cell that was never
        # actually held: `run_accounting`'s own `missing_overnight` check
        # (accounting.py) already raises, but only where the cell is
        # actually held before or after that day's trade -- matching
        # `missing_held_returns`'s identical held-cell-only tolerance for
        # the undecomposed close-to-close formula. Raising unconditionally
        # here (on ANY zero-close cell in the whole tradable panel,
        # including a symbol never held that day, or held on a different
        # day entirely) would make `next_bar_open` strictly LESS tolerant
        # of ordinary data quality issues than `same_bar_close` -- exactly
        # backwards, since `next_bar_open` is one of only two timings a
        # real deployment can ever use. Explicitly masked to NaN (not left
        # to a natural 0/0 or x/0 division, which can just as easily
        # produce +-inf) since `run_accounting` separately rejects any
        # Infinity in `overnight_returns` unconditionally, regardless of
        # whether that cell is ever held -- only a genuine NaN gets the
        # correct, held-cell-scoped tolerance.
        open_values = open_matrix.to_numpy(dtype=float)
        adjusted_close_values = adjusted_close_matrix.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = adjusted_close_values / close_values
            adjusted_open = open_values * ratio
        # A zero raw OPEN or a zero ADJUSTED CLOSE (with a nonzero raw
        # close) is just as undefined as a zero raw close, but silently
        # escapes a mask that only checks `close_values`: `open * ratio`
        # or `open * (0/close)` is exactly `0.0`, not NaN/inf, either way,
        # so it would otherwise pass through as a seemingly-valid
        # adjusted_open of 0.0 -- which then makes `overnight_returns =
        # 0.0 / prices.shift(1) - 1 == -1.0` exactly, tripping
        # `run_accounting`'s unconditional "not at or below -1.0" check
        # regardless of whether the cell is ever held. `adjusted_close`
        # is the ratio's own numerator, so a zero there is exactly as
        # undefined as a zero denominator (`close`) -- masked to NaN for
        # the identical reason and by the identical mechanism as the
        # zero-close case above -- same held-cell-scoped tolerance via
        # `missing_overnight`, never an unconditional raise here.
        #
        # A NEGATIVE price is masked the same way, for a different reason:
        # `DataValidator.load`'s default (non-``strict``) path only warns
        # and records `invalid_price_count` on a non-positive price, it
        # does not raise, so a corrupt negative tick can genuinely reach
        # here. Unlike zero, a negative `close`/`open`/`adjusted_close`
        # does not reliably produce a NaN/inf on its own -- `open * ratio`
        # can come out as an ordinary-looking (wrong) finite number for
        # some sign combinations, which would otherwise silently feed a
        # plausible but meaningless execution price into `overnight_
        # returns` with no error at all.
        invalid_price = (
            (close_values <= 0.0)
            | (open_values <= 0.0)
            | (adjusted_close_values <= 0.0)
        )
        adjusted_open[invalid_price] = np.nan
        return pd.DataFrame(
            adjusted_open, index=open_matrix.index, columns=open_matrix.columns
        )


def execution_price_source_is_consulted(config: ExperimentConfig) -> bool:
    """Whether ``config`` would ever actually call an ``execution_price_source``.

    Only ``next_bar_open`` needs the adjusted-open matrix at all --
    :func:`adjusted_open_for_config`/:func:`overnight_returns_for_config`/
    :func:`reference_prices_for_config` all early-return without calling
    it otherwise. The single source of truth for "is a non-default
    source here actually load-bearing" -- both
    :meth:`~quantlab.backtesting.engine.BacktestEngine.run`'s and
    :class:`~quantlab.validation.walk_forward.WalkForwardValidator`'s own
    ``config_yaml_reflects_execution`` metadata call this directly, so a
    custom source paired with ``same_bar_close``/``immediate_after_close``
    (provably dormant, zero effect on the run's numbers) is never flagged
    as making either result unreproducible from config.yaml alone.
    """
    return config.execution.execution_timing == ExecutionTiming.NEXT_BAR_OPEN


def _validate_adjusted_open(
    adjusted_open: object, price_source: object
) -> pd.DataFrame:
    """Validate a caller-supplied source's own ``adjusted_open()`` return value.

    A custom :class:`ExecutionPriceSource`'s return value crosses this
    module's public boundary unchecked otherwise -- the wrong type (e.g.
    a bare ``ndarray``, which has no ``.reindex``) previously surfaced
    only as a confusing, far-downstream ``AttributeError`` inside
    :func:`overnight_returns_for_config`/:func:`reference_prices_for_config`,
    rather than a clear error naming the actual misbehaving source.
    Reuses :func:`~quantlab.features._validation.numeric_pandas` (unique/
    sorted index and columns, numeric non-boolean dtype, no Infinity,
    strictly positive where defined -- the same invariant
    :class:`RatioApproximatedExecutionPrices` itself already documents
    masking a zero/negative/undefined ratio to NaN to uphold) rather than
    re-deriving those same checks here.
    """
    class_name = type(price_source).__name__
    if not isinstance(adjusted_open, pd.DataFrame):
        raise BacktestError(
            f"{class_name}.adjusted_open(...) must return a pandas "
            f"DataFrame, got {type(adjusted_open).__name__}."
        )
    try:
        numeric_pandas(adjusted_open, name="adjusted_open", strictly_positive=True)
    except (TypeError, ValueError) as exc:
        raise BacktestError(
            f"{class_name}.adjusted_open(...) returned an invalid "
            f"adjusted-open matrix: {exc}"
        ) from exc
    return adjusted_open


def adjusted_open_for_config(
    config: ExperimentConfig,
    tradable_data: pd.DataFrame,
    *,
    execution_price_source: ExecutionPriceSource | None = None,
) -> pd.DataFrame | None:
    """Return the adjusted-open matrix, or ``None`` outside ``next_bar_open``.

    A caller needing BOTH :func:`overnight_returns_for_config` and
    :func:`reference_prices_for_config` (e.g.
    :meth:`~quantlab.backtesting.engine.BacktestEngine.run`) computes it
    once here and passes the result to each via their own
    ``adjusted_open`` parameter, instead of each independently paying for
    :meth:`ExecutionPriceSource.adjusted_open`'s full-panel pivots.
    """
    if not execution_price_source_is_consulted(config):
        return None
    price_source = execution_price_source or RatioApproximatedExecutionPrices()
    return _validate_adjusted_open(
        price_source.adjusted_open(tradable_data), price_source
    )


def overnight_returns_for_config(
    config: ExperimentConfig,
    tradable_data: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    execution_price_source: ExecutionPriceSource | None = None,
    adjusted_open: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Return the ``overnight_returns`` matrix ``run_accounting`` needs, or ``None``.

    Shared by :meth:`~quantlab.backtesting.engine.BacktestEngine.run` and
    every walk-forward/robustness accounting call site in
    :mod:`quantlab.validation.walk_forward`, so a single backtest and a
    walk-forward study over the same config always derive
    ``overnight_returns`` from the exact same formula.

    ``execution_price_source`` is threaded through from both callers:
    ``BacktestEngine.run`` accepts its own ``execution_price_source``
    parameter and forwards it here, and ``WalkForwardValidator``'s own
    constructor accepts the identical parameter and forwards it to every
    one of its internal call sites (fold selection, OOS stitching, and
    ``rescore_with_costs``) -- a custom, e.g. more accurate point-in-time/
    corporate-actions-aware, ``ExecutionPriceSource`` given to a
    ``BacktestEngine`` run is honoured identically by a
    ``WalkForwardValidator`` constructed with the same source over the
    same config.

    Only ``next_bar_open`` needs the overnight/intraday decomposition
    (see ``accounting.py``'s own ``_gross_returns_and_cost_basis``):
    ``immediate_after_close`` is causally ordered in a continuous 24/7
    market using the exact same plain close-to-close formula as
    ``same_bar_close`` (there is no session gap to separate overnight
    from intraday) -- not a claim that it is fully realistic, since it
    still models no execution latency -- and ``same_bar_close`` itself is
    the backtest-only idealization this field defaults to. ``None`` here
    is a strict no-op downstream.

    Args:
        config: Experiment configuration.
        tradable_data: The canonical OHLCV frame ``prices`` was pivoted
            from (adjusted close).
        prices: That same adjusted-close price matrix (``dates ×
            symbols``).
        execution_price_source: Derives the adjusted-open matrix.
            Defaults to :class:`RatioApproximatedExecutionPrices`.
        adjusted_open: A caller-precomputed ``execution_price_source.
            adjusted_open(tradable_data)``, reused as-is instead of
            recomputing it -- both this function and
            :func:`reference_prices_for_config` derive it from the exact
            same call, so a caller needing both (e.g.
            :meth:`~quantlab.backtesting.engine.BacktestEngine.run`)
            computes it once and passes it to each rather than paying for
            the derivation twice. ``None`` (default) computes it fresh.
            Validated here either way (see :func:`_validate_adjusted_open`)
            -- a caller-supplied value crosses this module's public
            boundary exactly like a freshly-derived one, so it must not
            be trusted unchecked just because it arrived pre-computed.
    """
    if not execution_price_source_is_consulted(config):
        return None
    price_source = execution_price_source or RatioApproximatedExecutionPrices()
    if adjusted_open is None:
        adjusted_open = price_source.adjusted_open(tradable_data)
    adjusted_open = _validate_adjusted_open(adjusted_open, price_source)
    return (
        adjusted_open.reindex(index=prices.index, columns=prices.columns)
        / prices.shift(1)
        - 1.0
    )


def reference_prices_for_config(
    config: ExperimentConfig,
    tradable_data: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    execution_price_source: ExecutionPriceSource | None = None,
    adjusted_open: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Return ``(reference_prices, needs_shift)`` for the trade log's own fill price.

    A fill appearing on row ``T`` (``executed[T] != executed[T-1]``) was
    decided from data through close(T-1) either way, but the price it was
    actually FILLED at depends on ``execution_timing``:
    ``same_bar_close``/``immediate_after_close`` fill at that same
    close(T-1) -- ``prices`` itself, lagged by one row inside
    ``build_trade_log`` (``needs_shift=True``, its existing default).
    ``next_bar_open`` fills at open(T) instead -- the SAME adjusted-open
    matrix :func:`overnight_returns_for_config` derives its own overnight
    leg from, used as-is on row T with no further shift
    (``needs_shift=False``), never the row-T-1 close.

    ``adjusted_open``: see :func:`overnight_returns_for_config`'s
    identical parameter -- pass the same precomputed matrix to both to
    avoid deriving it twice. Validated here either way, also matching
    that function's own identical treatment.
    """
    if not execution_price_source_is_consulted(config):
        return prices, True
    price_source = execution_price_source or RatioApproximatedExecutionPrices()
    if adjusted_open is None:
        adjusted_open = price_source.adjusted_open(tradable_data)
    adjusted_open = _validate_adjusted_open(adjusted_open, price_source)
    return adjusted_open.reindex(index=prices.index, columns=prices.columns), False
