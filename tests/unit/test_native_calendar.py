"""Direct tests for `compute_native_then_align`, the shared helper that
computes a rolling-window feature on each symbol's own native calendar
before aligning it back onto a closure-padded combined timeline."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from quantlab.features.native_calendar import compute_native_then_align


def _mean_of_last(window: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda p: p.rolling(window, min_periods=window).mean()


def test_symbol_calendars_none_short_circuits_to_compute_fn() -> None:
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3)
    )
    calls = []

    def compute_fn(p: pd.DataFrame) -> pd.DataFrame:
        calls.append(p)
        return p * 2.0

    result = compute_native_then_align(
        compute_fn, prices, None, pd.DatetimeIndex(prices.index)
    )

    pd.testing.assert_frame_equal(result, prices * 2.0)
    assert len(calls) == 1


def test_no_calendar_for_any_column_short_circuits() -> None:
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3)
    )
    result = compute_native_then_align(
        lambda p: p * 2.0, prices, {"ZZZ": "XNYS"}, pd.DatetimeIndex(prices.index)
    )
    pd.testing.assert_frame_equal(result, prices * 2.0)


def test_no_actual_closure_in_range_short_circuits_byte_identical() -> None:
    """A calendar is configured, but every row in `prices.index` happens to
    be a real session on it (e.g. only business days present) -- must take
    the fast path (compute once on the whole frame), byte-identical to the
    single-calendar vectorized call."""
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    result = compute_native_then_align(
        _mean_of_last(2), prices, {"AAA": "XNYS"}, pd.DatetimeIndex(prices.index)
    )
    expected = prices.rolling(2, min_periods=2).mean()
    pd.testing.assert_frame_equal(result, expected)


def test_uniform_calendar_fast_path_covers_multiple_columns_sharing_it() -> None:
    """The uniform-calendar short-circuit (`len(calendars) ==
    len(prices.columns) and uniform_calendar(...) is not None`) exists
    precisely for a genuine multi-column, single-calendar universe (e.g.
    several XNYS equities together, per this module's own docstring) --
    a single column trivially satisfies "every column shares one
    calendar" without exercising more than one, so this test uses three.
    Confirms both the byte-for-byte result AND that `compute_fn` is
    called exactly ONCE on the full multi-column frame, never once per
    column."""
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0, 4.0],
            "BBB": [10.0, 20.0, 30.0, 40.0],
            "CCC": [100.0, 200.0, 300.0, 400.0],
        },
        index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    calls: list[pd.DataFrame] = []

    def compute_fn(p: pd.DataFrame) -> pd.DataFrame:
        calls.append(p.copy())
        return p.rolling(2, min_periods=2).mean()

    result = compute_native_then_align(
        compute_fn,
        prices,
        {"AAA": "XNYS", "BBB": "XNYS", "CCC": "XNYS"},
        pd.DatetimeIndex(prices.index),
    )

    expected = prices.rolling(2, min_periods=2).mean()
    pd.testing.assert_frame_equal(result, expected)
    assert len(calls) == 1
    assert list(calls[0].columns) == ["AAA", "BBB", "CCC"]


def test_native_computation_removes_closure_dilution() -> None:
    """The core fix: AAA (XNYS) is closed over a weekend shared with BTC
    (24/7) on the same combined timeline. A 3-period rolling mean computed
    on AAA's own native (session-only) dates must differ from -- and be
    more accurate than -- the same rolling mean computed directly on the
    closure-padded combined timeline."""
    dates = pd.DatetimeIndex(
        [
            "2024-01-04",  # Thu (AAA open)
            "2024-01-05",  # Fri (AAA open)
            "2024-01-06",  # Sat (AAA closed, verified XNYS weekend)
            "2024-01-07",  # Sun (AAA closed, verified XNYS weekend)
            "2024-01-08",  # Mon (AAA open)
            "2024-01-09",  # Tue (AAA open)
        ]
    )
    # AAA's padded series: flat-filled (last real close) on the weekend,
    # exactly as `insert_verified_closure_bars` produces in production.
    aaa = pd.Series([10.0, 12.0, 12.0, 12.0, 16.0, 20.0], index=dates)
    btc = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=dates)
    prices = pd.DataFrame({"AAA": aaa, "BTC": btc})

    result = compute_native_then_align(
        _mean_of_last(3),
        prices,
        {"AAA": "XNYS", "BTC": "24/7"},
        pd.DatetimeIndex(prices.index),
    )

    # Native AAA rolling(3): only real sessions Thu/Fri/Mon/Tue contribute.
    # Warm-up needs 3 native observations, so only Mon (Thu,Fri,Mon) and Tue
    # (Fri,Mon,Tue) are defined; the weekend rows forward-fill from Friday's
    # own (still-NaN, insufficient-warmup) native result, so they stay NaN.
    assert pd.isna(result.loc["2024-01-04", "AAA"])
    assert pd.isna(result.loc["2024-01-05", "AAA"])
    assert pd.isna(result.loc["2024-01-06", "AAA"])
    assert pd.isna(result.loc["2024-01-07", "AAA"])
    assert result.loc["2024-01-08", "AAA"] == pytest.approx((10.0 + 12.0 + 16.0) / 3.0)
    assert result.loc["2024-01-09", "AAA"] == pytest.approx((12.0 + 16.0 + 20.0) / 3.0)

    # The naive diluted computation (directly on the padded frame) would
    # have given a materially different, wrong answer for Monday.
    diluted = prices.rolling(3, min_periods=3).mean()
    assert diluted.loc["2024-01-08", "AAA"] != pytest.approx(
        result.loc["2024-01-08", "AAA"]
    )

    # BTC has no closures at all -- untouched, byte-identical to a plain
    # vectorized computation on its own full series.
    expected_btc = btc.rolling(3, min_periods=3).mean()
    pd.testing.assert_series_equal(result["BTC"], expected_btc, check_names=False)


def test_symbol_with_no_calendar_entry_computed_directly() -> None:
    """A column absent from `symbol_calendars` is treated as always open --
    computed directly, never sliced.

    Regression test: the original version of this test used a calendar
    with zero actual closures in range (or no calendar at all), which hits
    `compute_native_then_align`'s own EARLIER short-circuits (a calendar-
    less universe, or one where nothing genuinely closed, both return
    before ever reaching the per-column loop this test means to exercise)
    -- it happened to pass, but for the wrong reason, never actually
    running the `calendar is None` branch it claimed to cover. AAA (XNYS)
    below has a genuine weekend closure, forcing the function past both
    short-circuits into the per-column loop; ZZZ, absent from
    `symbol_calendars` entirely, must then be reached by THAT loop and
    take its own `calendar is None` branch."""
    dates = pd.DatetimeIndex(
        [
            "2024-01-04",  # Thu (AAA open)
            "2024-01-05",  # Fri (AAA open)
            "2024-01-06",  # Sat (AAA closed, verified XNYS weekend)
            "2024-01-07",  # Sun (AAA closed, verified XNYS weekend)
            "2024-01-08",  # Mon (AAA open)
        ]
    )
    aaa = pd.Series([10.0, 12.0, 12.0, 12.0, 16.0], index=dates)
    zzz = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    prices = pd.DataFrame({"AAA": aaa, "ZZZ": zzz})

    result = compute_native_then_align(
        _mean_of_last(2), prices, {"AAA": "XNYS"}, pd.DatetimeIndex(prices.index)
    )

    # ZZZ has no calendar entry at all -- computed directly on its own
    # full (unsliced) column, byte-identical to a plain vectorized call.
    expected_zzz = prices[["ZZZ"]].rolling(2, min_periods=2).mean()["ZZZ"]
    pd.testing.assert_series_equal(result["ZZZ"], expected_zzz, check_names=False)
