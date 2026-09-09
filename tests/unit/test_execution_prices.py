"""Tests for the isolated execution-price abstraction.

See ``quantlab.backtesting.execution_prices``: the accounting engine's
overnight/intraday return decomposition (``execution_timing="next_bar_open"``)
depends on the ``ExecutionPriceSource`` protocol, never a hardcoded formula,
so a future, more accurate implementation is a drop-in replacement. These
tests exercise the shipped approximation, ``RatioApproximatedExecutionPrices``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

import quantlab.backtesting.execution_prices as execution_prices_module
from quantlab.backtesting.execution_prices import (
    DEFAULT_EXECUTION_PRICE_SOURCE_FINGERPRINT,
    RatioApproximatedExecutionPrices,
    execution_price_source_fingerprint,
)


def _canonical_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_adjusted_open_is_exact_for_a_pure_split() -> None:
    """A pure stock split scales open/close/adjusted_close proportionally.

    On a pre-split historical bar, adjusted_close is the vendor's
    retroactively-halved close (a 2:1 split), while raw open/close stay
    at their original, un-adjusted values. The correct adjusted open is
    exactly the same halving applied to the raw open, since a split
    scales an entire bar's OHLC identically -- and that is exactly what
    ``open * (adjusted_close / close)`` recovers.
    """
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "AAA",
                "open": 95.0,
                "high": 101.0,
                "low": 94.0,
                "close": 100.0,
                "adjusted_close": 50.0,  # 2:1 split adjustment applied
                "volume": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2020-01-02"),
                "symbol": "AAA",
                "open": 51.0,
                "high": 52.0,
                "low": 49.0,
                "close": 50.0,
                "adjusted_close": 50.0,  # post-split day: no adjustment needed
                "volume": 2_000.0,
            },
        ]
    )
    result = RatioApproximatedExecutionPrices().adjusted_open(data)
    assert result.loc[pd.Timestamp("2020-01-01"), "AAA"] == pytest.approx(47.5)
    assert result.loc[pd.Timestamp("2020-01-02"), "AAA"] == pytest.approx(51.0)


def test_adjusted_open_dividend_case_is_deterministic_but_not_asserted_exact() -> None:
    """A cash-dividend adjustment is only approximated by this ratio.

    The true ex-dividend price drop happens once, at the open, not
    proportionally throughout the bar the way this formula implicitly
    assumes -- so this test only pins down the formula's own documented,
    deterministic behavior, never a "true" point-in-time value QuantLab
    has no independent way to compute today.
    """
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adjusted_close": 98.0,  # a $2 dividend adjustment
                "volume": 1_000.0,
            }
        ]
    )
    result = RatioApproximatedExecutionPrices().adjusted_open(data)
    # Documented formula: open * (adjusted_close / close) = 100 * 0.98 = 98.0.
    assert result.loc[pd.Timestamp("2020-01-01"), "AAA"] == pytest.approx(98.0)


def test_adjusted_open_masks_a_zero_raw_close_to_nan_instead_of_raising() -> None:
    """A zero raw close makes the ratio undefined for that one bar -- masked
    to NaN, never raised unconditionally here: `same_bar_close` only ever
    validates a return on a cell that was actually HELD
    (`missing_held_returns` in accounting.py), and `run_accounting`'s own
    `missing_overnight` check applies that identical held-cell-only
    tolerance to this NaN downstream. Raising here regardless of whether
    the cell is ever held would make `next_bar_open` -- one of only two
    timings a real deployment can ever use -- strictly LESS tolerant of
    an ordinary data-quality blip than the backtest-only `same_bar_close`.
    A neighboring, well-formed bar/symbol must be unaffected."""
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "AAA",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 0.0,
                "adjusted_close": 0.0,
                "volume": 0.0,
            },
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "BBB",
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "adjusted_close": 50.0,
                "volume": 100.0,
            },
        ]
    )
    result = RatioApproximatedExecutionPrices().adjusted_open(data)
    assert pd.isna(result.loc[pd.Timestamp("2020-01-01"), "AAA"])
    assert not pd.isna(result.loc[pd.Timestamp("2020-01-01"), "BBB"])
    assert result.loc[pd.Timestamp("2020-01-01"), "BBB"] == pytest.approx(50.0)
    # Never +-inf either: `run_accounting` separately rejects any Infinity
    # in `overnight_returns` unconditionally (regardless of held state),
    # so a masked cell must be an actual NaN, not a natural 0/0 or x/0
    # division result that could just as easily come out +-inf.
    assert not np.isinf(cast(float, result.loc[pd.Timestamp("2020-01-01"), "AAA"]))


def test_adjusted_open_masks_a_zero_raw_open_to_nan_too() -> None:
    """A zero raw OPEN (with a nonzero close) silently escapes a mask that
    only checks close: `0 * (adjusted_close / close)` is exactly `0.0`,
    not NaN/inf, which would then make `overnight_returns = 0.0 /
    prices.shift(1) - 1 == -1.0` exactly -- tripping run_accounting's
    unconditional "not at or below -1.0" check regardless of whether the
    cell is ever held. Must be masked to NaN by the identical mechanism as
    a zero close."""
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "AAA",
                "open": 0.0,
                "high": 10.0,
                "low": 0.0,
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 0.0,
            },
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "BBB",
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "adjusted_close": 50.0,
                "volume": 100.0,
            },
        ]
    )
    result = RatioApproximatedExecutionPrices().adjusted_open(data)
    assert pd.isna(result.loc[pd.Timestamp("2020-01-01"), "AAA"])
    assert not np.isinf(cast(float, result.loc[pd.Timestamp("2020-01-01"), "AAA"]))
    assert not pd.isna(result.loc[pd.Timestamp("2020-01-01"), "BBB"])
    assert result.loc[pd.Timestamp("2020-01-01"), "BBB"] == pytest.approx(50.0)


def test_adjusted_open_masks_a_negative_close_to_nan_too() -> None:
    """A negative price (a corrupt tick) is reachable in practice: the
    default, non-strict `DataValidator.load` path only warns and records
    `invalid_price_count` on a non-positive price, it does not raise. A
    negative close/open/adjusted_close does not reliably produce a NaN or
    inf on its own the way zero does -- `open * (adjusted_close / close)`
    can come out as an ordinary-looking (wrong) finite number for some
    sign combinations -- so it must be masked explicitly, by the same
    mechanism as the zero-price cases above, rather than silently feeding
    a plausible but meaningless execution price downstream."""
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "AAA",
                "open": 10.0,
                "high": 10.0,
                "low": -5.0,
                "close": -5.0,
                "adjusted_close": -5.0,
                "volume": 0.0,
            },
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": "BBB",
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "adjusted_close": 50.0,
                "volume": 100.0,
            },
        ]
    )
    result = RatioApproximatedExecutionPrices().adjusted_open(data)
    assert pd.isna(result.loc[pd.Timestamp("2020-01-01"), "AAA"])
    assert not np.isinf(cast(float, result.loc[pd.Timestamp("2020-01-01"), "AAA"]))
    assert not pd.isna(result.loc[pd.Timestamp("2020-01-01"), "BBB"])
    assert result.loc[pd.Timestamp("2020-01-01"), "BBB"] == pytest.approx(50.0)


def test_adjusted_open_rejects_a_misaligned_adjusted_close_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open/close/adjusted_close are pivoted independently from the same
    `data`, so real, well-formed input can never actually misalign them
    (all three share `data`'s own timestamp/symbol pairs) -- this proves
    the same_axes() guard itself actually fires, using a monkeypatched
    pivot_field to simulate the axis mismatch it exists to catch (e.g. a
    future point-in-time adjusted_close source computed on a different
    row set than open/close)."""
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": symbol,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "adjusted_close": 9.5,
                "volume": 100.0,
            }
            for symbol in ("AAA", "BBB")
        ]
    )
    real_pivot_field = execution_prices_module.pivot_field

    def _pivot_field_dropping_a_symbol_for_adjusted_close(
        frame: pd.DataFrame, field: str
    ) -> pd.DataFrame:
        wide = real_pivot_field(frame, field)
        if field == "adjusted_close":
            return wide.drop(columns=["BBB"])
        return wide

    monkeypatch.setattr(
        execution_prices_module,
        "pivot_field",
        _pivot_field_dropping_a_symbol_for_adjusted_close,
    )
    with pytest.raises(ValueError, match="adjusted_close must have the same columns"):
        RatioApproximatedExecutionPrices().adjusted_open(data)


def test_execution_price_source_is_part_of_the_public_backtesting_api() -> None:
    """``ExecutionPriceSource``/``RatioApproximatedExecutionPrices`` are
    documented as the extension point for a custom next_bar_open price
    source (see docs/api.md, docs/backtesting.md) -- they must therefore
    actually be importable from the public ``quantlab.backtesting``
    package, not only from the internal ``execution_prices`` submodule."""
    import quantlab.backtesting as backtesting_package

    assert (
        backtesting_package.ExecutionPriceSource
        is execution_prices_module.ExecutionPriceSource
    )
    assert (
        backtesting_package.RatioApproximatedExecutionPrices
        is execution_prices_module.RatioApproximatedExecutionPrices
    )
    assert "ExecutionPriceSource" in backtesting_package.__all__
    assert "RatioApproximatedExecutionPrices" in backtesting_package.__all__


def test_adjusted_open_multi_symbol_shape() -> None:
    data = _canonical_frame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-01"),
                "symbol": symbol,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "adjusted_close": 9.5,
                "volume": 100.0,
            }
            for symbol in ("AAA", "BBB")
        ]
    )
    result = RatioApproximatedExecutionPrices().adjusted_open(data)
    assert list(result.columns) == ["AAA", "BBB"]
    assert result.loc[pd.Timestamp("2020-01-01"), "AAA"] == pytest.approx(9.5)
    assert result.loc[pd.Timestamp("2020-01-01"), "BBB"] == pytest.approx(9.5)


def test_execution_price_source_fingerprint_default_is_a_stable_sentinel() -> None:
    assert (
        execution_price_source_fingerprint(None)
        == DEFAULT_EXECUTION_PRICE_SOURCE_FINGERPRINT
    )
    fingerprint_again = execution_price_source_fingerprint(None)
    assert execution_price_source_fingerprint(None) == fingerprint_again


def test_execution_price_source_fingerprint_explicit_default_matches_none() -> None:
    """Regression test: an explicit, state-free ``RatioApproximatedExecutionPrices()``
    is behaviorally identical to the implicit ``None`` default -- fingerprinting it
    differently would make ``WalkForwardValidator.rescore_with_costs``'s cache check
    (and a walk-forward checkpoint's own provenance comparison) spuriously treat two
    numerically-identical runs as mismatched merely because one spelled the default
    out explicitly. A SUBCLASS must still fingerprint distinctly, since it could
    override ``adjusted_open`` or carry real state."""
    assert execution_price_source_fingerprint(
        RatioApproximatedExecutionPrices()
    ) == execution_price_source_fingerprint(None)

    class _CustomSubclass(RatioApproximatedExecutionPrices):
        pass

    assert execution_price_source_fingerprint(
        _CustomSubclass()
    ) != execution_price_source_fingerprint(None)


def test_execution_price_source_fingerprint_distinguishes_different_instances() -> None:
    """Regression test: two instances of the SAME class with different
    constructor parameters must fingerprint differently -- a checkpoint
    (or config_yaml_reflects_execution) keyed only on the class name would
    otherwise let a walk-forward run resume under a source that prices
    the adjusted open differently from the one that built the checkpoint,
    silently mixing two incompatible datasets into one run."""

    class _TunedExecutionPrices:
        def __init__(self, dividend_threshold: float) -> None:
            self.dividend_threshold = dividend_threshold

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

    low = _TunedExecutionPrices(dividend_threshold=0.01)
    high = _TunedExecutionPrices(dividend_threshold=0.05)
    same_as_low = _TunedExecutionPrices(dividend_threshold=0.01)

    fingerprint_low = execution_price_source_fingerprint(low)
    fingerprint_high = execution_price_source_fingerprint(high)
    fingerprint_same = execution_price_source_fingerprint(same_as_low)

    assert fingerprint_low != fingerprint_high
    assert fingerprint_low == fingerprint_same
    assert fingerprint_low != DEFAULT_EXECUTION_PRICE_SOURCE_FINGERPRINT
    # Both still identify the same underlying class, just with a distinct
    # per-configuration suffix -- not two unrelated-looking strings.
    assert fingerprint_low.split(":")[0] == fingerprint_high.split(":")[0]


def test_execution_price_source_fingerprint_distinguishes_large_dataframes() -> None:
    """Regression test: two instances holding a large DataFrame attribute
    that differs only in a MIDDLE row (outside pandas' own truncated
    repr, which shows only head/tail rows) must still fingerprint
    differently -- hashing the truncated repr text instead of the actual
    data would silently collide two sources that price the adjusted open
    from genuinely different tables."""

    class _TableBackedExecutionPrices:
        def __init__(self, table: pd.DataFrame) -> None:
            self.table = table

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

    n = 200
    table_a = pd.DataFrame({"a": range(n), "b": range(n)})
    table_b = table_a.copy()
    table_b.loc[100, "b"] = 999_999.0  # a middle row, outside the repr window
    assert repr(table_a) == repr(table_b), (
        "test setup assumption broken: pandas no longer truncates this size"
    )

    fingerprint_a = execution_price_source_fingerprint(
        _TableBackedExecutionPrices(table_a)
    )
    fingerprint_b = execution_price_source_fingerprint(
        _TableBackedExecutionPrices(table_b)
    )
    assert fingerprint_a != fingerprint_b


def test_execution_price_source_fingerprint_falls_back_to_class_name_with_no_dict() -> (
    None
):
    """A class with no `__dict__` (e.g. `__slots__` only) can't be
    introspected further -- must still return a stable, class-identifying
    fingerprint rather than raising."""

    class _SlottedExecutionPrices:
        __slots__ = ()

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

    fingerprint = execution_price_source_fingerprint(_SlottedExecutionPrices())
    assert "_SlottedExecutionPrices" in fingerprint
    assert fingerprint == execution_price_source_fingerprint(_SlottedExecutionPrices())


def test_execution_price_source_fingerprint_distinguishes_slotted_instances() -> None:
    """Regression test: a ``__slots__``-only class (no ``__dict__`` at
    all) still carries real, meaningful per-instance state in its
    slots -- two differently-configured instances of such a class must
    not fingerprint identically just because ``vars()`` raises
    ``TypeError`` on them."""

    class _SlottedTunedExecutionPrices:
        __slots__ = ("threshold",)

        def __init__(self, threshold: float) -> None:
            self.threshold = threshold

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

    low = _SlottedTunedExecutionPrices(0.01)
    high = _SlottedTunedExecutionPrices(0.05)
    same_as_low = _SlottedTunedExecutionPrices(0.01)

    fingerprint_low = execution_price_source_fingerprint(low)
    fingerprint_high = execution_price_source_fingerprint(high)
    fingerprint_same = execution_price_source_fingerprint(same_as_low)
    assert fingerprint_low != fingerprint_high
    assert fingerprint_low == fingerprint_same


def test_execution_price_source_fingerprint_sees_inherited_slots_on_a_dict_class() -> (
    None
):
    """Regression test: a LEAF class that doesn't itself declare
    ``__slots__`` still gets a real (if empty) ``__dict__`` from Python,
    even when every actual attribute lives in an inherited ``__slots__``
    mixin -- checking ``__dict__`` alone (and stopping there, since
    ``vars()`` succeeds) would silently report "no state" for such an
    instance and collapse two differently-configured instances onto the
    same fingerprint, exactly the collision this function exists to
    prevent."""

    class _SlottedMixin:
        __slots__ = ("threshold",)

        def __init__(self, threshold: float) -> None:
            self.threshold = threshold

    class _MixedStateExecutionPrices(_SlottedMixin):
        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

    low = _MixedStateExecutionPrices(0.01)
    high = _MixedStateExecutionPrices(0.99)
    same_as_low = _MixedStateExecutionPrices(0.01)

    fingerprint_low = execution_price_source_fingerprint(low)
    fingerprint_high = execution_price_source_fingerprint(high)
    fingerprint_same = execution_price_source_fingerprint(same_as_low)
    assert fingerprint_low != fingerprint_high
    assert fingerprint_low == fingerprint_same


def test_execution_price_source_fingerprint_distinguishes_nested_dataframes() -> None:
    """Regression test: a DataFrame nested inside a dict-valued
    attribute (e.g. a per-symbol table of corrective factors) must still
    be content-hashed, not merely ``repr()``'d as a dict containing an
    opaque pandas object -- two instances differing only in a middle
    row of a large nested table (outside pandas' own truncated repr
    window) must not fingerprint identically."""

    class _TableMapExecutionPrices:
        def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
            self.tables = tables

        def adjusted_open(self, data: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

    n = 200
    table_a = pd.DataFrame({"a": range(n), "b": range(n)})
    table_b = table_a.copy()
    table_b.loc[100, "b"] = 999_999.0

    fingerprint_a = execution_price_source_fingerprint(
        _TableMapExecutionPrices({"x": table_a})
    )
    fingerprint_b = execution_price_source_fingerprint(
        _TableMapExecutionPrices({"x": table_b})
    )
    assert fingerprint_a != fingerprint_b
