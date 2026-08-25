"""Regression tests for loader, resampler, and cache hardening."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quantlab.data.storage as storage_module
from quantlab.config import ExperimentConfig
from quantlab.constants import OHLCV_COLUMNS
from quantlab.data.base import MarketDataSource
from quantlab.data.loader import DataLoader, build_source
from quantlab.data.resampler import resample_ohlcv
from quantlab.data.storage import ParquetStorage
from quantlab.exceptions import DataDownloadError, DataValidationError


def _ohlcv(
    timestamps: pd.DatetimeIndex | list[pd.Timestamp] | list[str],
    *,
    symbol: str = "AAA",
    volume: float = 100.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": volume,
        }
    )


def test_hourly_247_cache_requires_safely_closed_hours_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = pd.Timestamp("2026-08-08 15:40:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: now)
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")

    yesterday = pd.date_range("2026-08-07", periods=24, freq="h")
    storage.write_symbol(
        _ohlcv(yesterday, symbol="BTCUSDT"),
        "binance",
        "BTCUSDT",
        "1h",
        calendar="24/7",
    )
    assert not storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2026, 8, 7),
        date(2026, 8, 8),
        calendar="24/7",
    )

    safely_closed_today = pd.date_range("2026-08-08", periods=15, freq="h")
    storage.write_symbol(
        _ohlcv(safely_closed_today, symbol="BTCUSDT"),
        "binance",
        "BTCUSDT",
        "1h",
        calendar="24/7",
    )
    assert storage.cache_covers(
        "binance",
        "BTCUSDT",
        "1h",
        date(2026, 8, 7),
        date(2026, 8, 8),
        calendar="24/7",
    )


@pytest.mark.parametrize(
    ("frequency", "calendar", "bar_timestamp"),
    [
        ("1h", "24/7", pd.Timestamp("2024-01-03 10:00:00")),
        ("1d", "XNYS", pd.Timestamp("2024-01-03")),  # Wednesday
        ("1w", "XNYS", pd.Timestamp("2024-01-03")),  # week ending Friday
        ("1mo", "XNYS", pd.Timestamp("2024-01-15")),  # month ending Jan 31
    ],
)
def test_drop_still_open_bars_requires_posting_lag_past_bucket_close(
    frequency: str,
    calendar: str,
    bar_timestamp: pd.Timestamp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bar must stay hidden from the served view not just while its own
    bucket is open, but until this frequency's posting-lag tolerance has
    *also* elapsed past that bucket's close -- the same threshold every
    cache-coverage check already uses to decide a bar is safe to trust (see
    ``_posting_lag_for``). Checked at the exact boundary instants (not just
    a second to either side of them) -- ``_drop_still_open_bars`` compares
    with ``<=``, so a bar becomes safe the instant ``now`` reaches
    ``bucket_end + posting_lag``, not only strictly after it -- across
    hourly, daily, weekly and monthly frequencies."""
    from quantlab.data.calendar import bar_bucket_end
    from quantlab.data.storage import _drop_still_open_bars, _posting_lag_for

    frame = _ohlcv([bar_timestamp])
    bucket_end = bar_bucket_end(
        pd.Series([bar_timestamp]), frequency, calendar=calendar
    ).iloc[0]
    posting_lag = _posting_lag_for(frequency)
    epsilon = pd.Timedelta(seconds=1)
    safe_at = bucket_end + posting_lag

    boundaries = [
        (bucket_end - epsilon, 0, "just before bucket close"),
        (bucket_end, 0, "exactly at bucket close"),
        (bucket_end + epsilon, 0, "just after close, before posting lag"),
        (safe_at - epsilon, 0, "just before posting lag elapses"),
        (safe_at, 1, "exactly at the posting-lag boundary"),
        (safe_at + epsilon, 1, "just after posting lag elapses"),
    ]
    for now, expected_len, label in boundaries:
        monkeypatch.setattr(storage_module, "_utc_now", lambda now=now: now)
        served = _drop_still_open_bars(frame, frequency, calendar=calendar)
        assert len(served) == expected_len, label


def test_equity_hourly_cache_requires_the_final_requested_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2026-01-20 12:00:00")
    )
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    days = pd.date_range("2024-01-02", "2024-01-05", freq="B")
    hours = [day + pd.Timedelta(hours=hour) for day in days for hour in range(9, 16)]
    storage.write_symbol(_ohlcv(hours), "yahoo", "AAA", "1h", calendar="XNYS")

    assert not storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 8), calendar="XNYS"
    )

    monday = [pd.Timestamp("2024-01-08") + pd.Timedelta(hours=h) for h in range(9, 16)]
    storage.write_symbol(_ohlcv(monday), "yahoo", "AAA", "1h", calendar="XNYS")
    assert storage.cache_covers(
        "yahoo", "AAA", "1h", date(2024, 1, 2), date(2024, 1, 8), calendar="XNYS"
    )


def test_resampler_rejects_a_finer_target() -> None:
    daily = _ohlcv(pd.date_range("2024-01-01", periods=3, freq="D"))
    with pytest.raises(DataValidationError, match="finer"):
        resample_ohlcv(daily, "1h")


def test_resampler_preserves_unknown_volume_and_canonical_schema() -> None:
    daily = _ohlcv(
        pd.date_range("2024-01-01", periods=5, freq="D"), volume=float("nan")
    )
    weekly = resample_ohlcv(daily, "1w", source_frequency="1d")
    assert list(weekly.columns) == list(OHLCV_COLUMNS)
    assert weekly["volume"].isna().all()
    assert weekly["close"].notna().all()


def test_resampler_returns_a_canonical_empty_frame() -> None:
    empty = pd.DataFrame(columns=OHLCV_COLUMNS)
    result = resample_ohlcv(empty, "1w", source_frequency="1d")
    assert result.empty
    assert list(result.columns) == list(OHLCV_COLUMNS)


def test_resampler_rejects_duplicates_and_unknown_frequencies() -> None:
    frame = _ohlcv(["2024-01-01", "2024-01-01"])
    with pytest.raises(DataValidationError, match="duplicate"):
        resample_ohlcv(frame, "1w", source_frequency="1d")
    with pytest.raises(DataValidationError, match="Unsupported resampling"):
        resample_ohlcv(frame.iloc[[0]], "2h", source_frequency="1h")


def test_resampler_without_a_calendar_splits_a_utc_midnight_crossing_session() -> None:
    """Baseline (documents current default behaviour, not the desired one):
    with no ``calendar`` passed, two hourly bars from the SAME XASX session
    (which opens before UTC midnight of its own label date under daylight
    saving) land in two different daily output bars, purely because they
    fall on two different raw UTC calendar days."""
    from quantlab.data.calendar import sessions

    schedule = sessions("XASX", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
    market_open = schedule.iloc[0]["market_open"]
    assert market_open.date() < schedule.index[0].date()  # confirms the crossing

    bars = pd.DatetimeIndex([market_open, market_open + pd.Timedelta(hours=1)])
    data = _ohlcv(bars, symbol="BHP")
    daily = resample_ohlcv(data, "1d", source_frequency="1h")
    assert len(daily) == 2


def test_resampler_with_calendar_merges_a_utc_midnight_crossing_session() -> None:
    """The fix: passing the instrument's own calendar groups by its real
    trading session instead of a raw UTC period, so the same two bars from
    test_resampler_without_a_calendar_splits_a_utc_midnight_crossing_session
    merge into the one daily bar they actually belong to."""
    from quantlab.data.calendar import sessions

    schedule = sessions("XASX", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
    session_date = schedule.index[0]
    market_open = schedule.iloc[0]["market_open"]

    bars = pd.DatetimeIndex([market_open, market_open + pd.Timedelta(hours=1)])
    data = pd.DataFrame(
        {
            "timestamp": bars,
            "symbol": "BHP",
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.5, 10.0],
            "close": [10.2, 10.8],
            "adjusted_close": [10.2, 10.8],
            "volume": [100.0, 200.0],
        }
    )
    daily = resample_ohlcv(data, "1d", source_frequency="1h", calendar="XASX")
    assert len(daily) == 1
    assert daily["timestamp"].iloc[0] == pd.Timestamp(session_date)
    assert daily["open"].iloc[0] == 10.0  # first chronologically, not first per UTC day
    assert daily["close"].iloc[0] == 10.8  # last chronologically
    assert daily["volume"].iloc[0] == 300.0


def test_resampler_weekly_xnys_produces_one_bar_per_monday_sunday_week() -> None:
    """A full XNYS week (Tue-Fri, since Monday Jan 1 is a holiday) plus the
    following Mon-Fri week must resample into exactly one weekly bar each,
    Monday-labelled -- ``Period(freq="W-MON")`` means "week *ending* on
    Monday" (Tuesday..Monday), not "week starting on Monday", so grouping by
    it would wrongly split a Mon-Fri week into a lone Monday bar plus a
    Tue-Fri remainder attributed to the next week."""
    dates = pd.bdate_range("2024-01-02", "2024-01-12")  # Tue-Fri, then Mon-Fri
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "open": range(len(dates)),
            "high": range(len(dates)),
            "low": range(len(dates)),
            "close": range(len(dates)),
            "adjusted_close": range(len(dates)),
            "volume": 100.0,
        }
    ).astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "adjusted_close": float,
        }
    )

    for cal in (None, "XNYS"):
        weekly = resample_ohlcv(
            data, "1w", source_frequency="1d", **({"calendar": cal} if cal else {})
        )
        assert len(weekly) == 2
        assert weekly["timestamp"].tolist() == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-08"),
        ]
        # Week 1 (Jan 2-5, 4 rows: opens 0-3) must stay a single bar, not
        # split into a lone Monday (there is none -- Jan 1 is a holiday) and
        # a remainder.
        assert weekly["open"].iloc[0] == 0.0
        assert weekly["close"].iloc[0] == 3.0
        assert weekly["open"].iloc[1] == 4.0
        assert weekly["close"].iloc[1] == 8.0


def test_resampler_weekly_with_utc_midnight_crossing_calendar_aligns_to_monday() -> (
    None
):
    """Same Monday..Sunday week-boundary bug, but for a calendar whose own
    daily bars are timestamped via session_labels (real trading-session
    dates that can themselves cross UTC midnight) rather than raw calendar
    dates -- weekly resampling on top of that must still group Monday
    through Sunday, not Tuesday through Monday."""
    from quantlab.data.calendar import sessions

    schedule = sessions("XASX", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-14"))
    session_dates = list(schedule.index)
    opens = [schedule.loc[d, "market_open"] for d in session_dates]
    data = pd.DataFrame(
        {
            "timestamp": opens,
            "symbol": "BHP",
            "open": range(len(opens)),
            "high": range(len(opens)),
            "low": range(len(opens)),
            "close": range(len(opens)),
            "adjusted_close": range(len(opens)),
            "volume": 100.0,
        }
    ).astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "adjusted_close": float,
        }
    )

    weekly = resample_ohlcv(data, "1w", source_frequency="1d", calendar="XASX")
    # Every output timestamp must be a Monday, and every session must land
    # in the week containing its own real session date.
    assert (pd.DatetimeIndex(weekly["timestamp"]).dayofweek == 0).all()
    expected_weeks = sorted(
        {d.normalize() - pd.Timedelta(days=d.dayofweek) for d in session_dates}
    )
    assert weekly["timestamp"].tolist() == expected_weeks


def test_resampler_weekly_with_a_non_western_trading_week_does_not_split_it() -> None:
    """A calendar whose trading week isn't Monday-Sunday (XSAU trades
    Sunday-Thursday) must never have its own real trading week split
    across two output bars just because a fixed ISO week boundary glues
    Sunday to the previous week instead of the Monday-Thursday sessions it
    actually trades alongside."""
    from quantlab.data.calendar import sessions

    schedule = sessions("XSAU", pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-11"))
    dates = list(schedule.index)
    assert [d.strftime("%a") for d in dates] == ["Sun", "Mon", "Tue", "Wed", "Thu"]
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "X",
            "open": range(len(dates)),
            "high": range(len(dates)),
            "low": range(len(dates)),
            "close": range(len(dates)),
            "adjusted_close": range(len(dates)),
            "volume": 100.0,
        }
    ).astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "adjusted_close": float,
        }
    )

    weekly = resample_ohlcv(data, "1w", source_frequency="1d", calendar="XSAU")
    # All five sessions (Sun-Thu) are one real trading week -- must produce
    # exactly one bar, not split Sunday off into its own.
    assert len(weekly) == 1
    assert weekly["timestamp"].iloc[0] == pd.Timestamp("2024-01-07")
    assert weekly["open"].iloc[0] == 0.0
    assert weekly["close"].iloc[0] == 4.0


def _ohlcv_frame(dates: list[str], *, symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )


def test_resample_infers_daily_source_frequency_across_a_weekend_gap() -> None:
    """Two adjacent daily bars, Friday then Monday, are 3 raw calendar days
    apart -- a literal-gap inference would misread that as a 3-day source
    frequency and then refuse to resample it even to '1d' itself, since a
    3-day source is (wrongly) 'coarser' than a 1-day target."""
    data = _ohlcv_frame(["2024-01-05", "2024-01-08"])  # Friday, Monday
    out = resample_ohlcv(data, "1d")
    assert len(out) == 2


def test_resample_infers_monthly_source_frequency_from_a_31_day_gap() -> None:
    """Two adjacent monthly bars (Jan 1st, Feb 1st) are a genuine 31 raw
    calendar days apart, while FREQUENCY_TIMEDELTA['1mo'] is a fixed nominal
    30 days -- a literal-gap comparison would treat the observed 31-day
    source as coarser than the 30-day '1mo' target and refuse even a
    monthly-to-monthly no-op resample."""
    data = _ohlcv_frame(["2024-01-01", "2024-02-01"])
    out = resample_ohlcv(data, "1mo")
    assert len(out) == 2


def test_weekly_cache_covers_a_non_western_trading_week_without_a_false_gap(
    tmp_path: Path,
) -> None:
    """A single weekly bar labeled at the start of an XSAU (Sunday-Thursday)
    trading week must be recognised as complete for a request spanning only
    that same week -- grouping the gap check by a fixed Monday-Sunday ISO
    week would split that one real week across two ISO periods and report a
    perfectly complete cache as having an internal gap, forcing a spurious
    re-download."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    storage.write_symbol(
        _ohlcv_frame(["2024-01-07"], symbol="AAA"),  # a Sunday, week start
        "yahoo",
        "AAA",
        "1w",
        calendar="XSAU",
    )
    assert storage.cache_covers(
        "yahoo", "AAA", "1w", date(2024, 1, 7), date(2024, 1, 11), calendar="XSAU"
    )


class _PartialSource(MarketDataSource):
    name = "partial"

    def download(
        self,
        symbols: list[str],
        start: date,
        end: date,
        frequency: str,
        *,
        calendar: str = "XNYS",
    ) -> pd.DataFrame:
        del symbols, start, end, frequency, calendar
        return _ohlcv(["2024-01-03"])


def test_forced_download_returns_the_persisted_merged_frame(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    storage.write_symbol(
        _ohlcv(["2023-12-31", "2024-01-01"]), "partial", "AAA", "1d", calendar="24/7"
    )
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "forced_consistency",
            "data": {
                "instruments": [{"symbol": "AAA", "source": "csv", "calendar": "24/7"}],
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    loader = DataLoader(storage=storage)
    returned = loader._download_symbol(
        _PartialSource(), config.data.instruments[0], config, force=True
    )
    persisted = storage.read_symbol("partial", "AAA", "1d", calendar="24/7")
    assert persisted is not None
    pd.testing.assert_frame_equal(returned, persisted)
    assert returned["timestamp"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-12-31",
        "2024-01-03",
    ]


def test_write_symbol_never_drops_still_open_bars_from_the_persisted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache key has no calendar component, so the same file must never
    depend on which calendar happened to write last. A Wednesday bar is
    already settled under XNYS (its close, plus the daily posting-lag
    tolerance, is well before UTC midnight the next day) but still 'open'
    under a 24/7 calendar (whose daily bucket, plus that same tolerance,
    settles even later) -- writing it under either calendar must never let
    that calendar's own settlement opinion decide whether the bar survives
    in the FILE itself, which would silently delete another experiment's
    already-settled data."""
    write_time = pd.Timestamp("2024-01-03 22:00:00")  # after XNYS close
    monkeypatch.setattr(storage_module, "_utc_now", lambda: write_time)
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")

    bar = _ohlcv(["2024-01-03"], symbol="AAA")
    storage.write_symbol(bar, "yahoo", "AAA", "1d", calendar="XNYS")
    # A second experiment on the same (source, symbol, frequency) key, using
    # a 24/7 calendar, writes next -- it must not purge XNYS's already-
    # settled bar from the shared file just because 24/7 still considers it
    # provisional.
    storage.write_symbol(bar.iloc[:0], "yahoo", "AAA", "1d", calendar="24/7")

    raw = pd.read_parquet(storage._cache_path("yahoo", "AAA", "1d"))
    assert len(raw) == 1

    # Each caller's own read still applies its own settlement opinion --
    # only the persisted file is calendar-independent. XNYS's close (~21:00
    # UTC) plus the 12h daily posting lag has passed by 10:00 the next day;
    # 24/7's bucket (open until UTC midnight) plus that same lag hasn't.
    read_time = pd.Timestamp("2024-01-04 10:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: read_time)
    xnys_view = storage.read_symbol("yahoo", "AAA", "1d", calendar="XNYS")
    always_open_view = storage.read_symbol("yahoo", "AAA", "1d", calendar="24/7")
    assert xnys_view is not None
    assert len(xnys_view) == 1
    assert always_open_view is not None
    assert always_open_view.empty


def test_cache_covers_forces_a_refresh_of_a_bar_settled_after_it_was_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_symbol's own docstring says a still-open bar is 'naturally
    superseded once its real, closed value is next downloaded' -- but
    nothing forces that next download to actually happen: coverage-checking
    only checks presence, not whether the bar was still forming when it was
    fetched. Without the per-row `_fetched_at` provenance write_symbol
    always sets, once enough wall time passes for the bar to look 'settled'
    from read_symbol's point of view, cache_covers would report the range
    as fully covered forever, silently serving the stale provisional OHLCV
    value fetched while the market was still open."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    bar_date = pd.Timestamp("2024-01-16")

    # Fetched at 19:00 UTC -- XNYS (closes ~21:00 UTC in winter) is still
    # open, so this bar is genuinely provisional at write time.
    still_open_now = pd.Timestamp("2024-01-16 19:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: still_open_now)
    storage.write_symbol(
        _ohlcv([bar_date], symbol="AAPL"), "yahoo", "AAPL", "1d", calendar="XNYS"
    )

    served_before_close = storage.read_symbol("yahoo", "AAPL", "1d", calendar="XNYS")
    assert served_before_close is not None
    assert served_before_close.empty  # correctly masked while still open

    # Two days later: the bar has genuinely settled by now, but nothing has
    # re-downloaded it -- the cached value is still the stale provisional one.
    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2024-01-18 12:00:00")
    )
    covers = storage.cache_covers(
        "yahoo", "AAPL", "1d", bar_date.date(), bar_date.date(), calendar="XNYS"
    )
    assert covers is False  # must force a redownload, never silently accept


def test_cache_staleness_check_is_per_row_not_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-file `last_write_time` would be wrongly refreshed by any write
    to the file, even one touching an unrelated, older date range -- making
    a genuinely still-provisional bar look freshly confirmed. Provenance
    must be tracked per row: a write to one date must never mark a
    *different* date's bar as freshly written."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    provisional_date = pd.Timestamp("2024-01-16")

    still_open_now = pd.Timestamp("2024-01-16 19:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: still_open_now)
    storage.write_symbol(
        _ohlcv([provisional_date], symbol="AAPL"),
        "yahoo",
        "AAPL",
        "1d",
        calendar="XNYS",
    )

    # A later write to a completely different, older date -- must not reset
    # the provisional bar's own fetch time.
    unrelated_write_time = pd.Timestamp("2024-01-17 08:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: unrelated_write_time)
    older_date = pd.Timestamp("2024-01-02")
    storage.write_symbol(
        _ohlcv([older_date], symbol="AAPL"),
        "yahoo",
        "AAPL",
        "1d",
        calendar="XNYS",
        replace_start=older_date.date(),
        replace_end=older_date.date(),
    )

    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2024-01-18 12:00:00")
    )
    covers = storage.cache_covers(
        "yahoo",
        "AAPL",
        "1d",
        provisional_date.date(),
        provisional_date.date(),
        calendar="XNYS",
    )
    assert covers is False


def test_cache_staleness_check_still_finds_a_stale_bar_buried_mid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking only the newest (frontier) row would stop catching a stale
    bar the moment a newer bar is appended -- the stale bar becomes an
    'internal' row and is silently never revisited again."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    provisional_date = pd.Timestamp("2024-01-16")

    still_open_now = pd.Timestamp("2024-01-16 19:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: still_open_now)
    storage.write_symbol(
        _ohlcv([provisional_date], symbol="BTC"), "yahoo", "BTC", "1d", calendar="XNYS"
    )

    # A genuinely new bar is appended the next day -- the old provisional
    # bar is now buried mid-file, no longer the frontier.
    next_day_close = pd.Timestamp("2024-01-17 22:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: next_day_close)
    storage.write_symbol(
        _ohlcv([pd.Timestamp("2024-01-17")], symbol="BTC"),
        "yahoo",
        "BTC",
        "1d",
        calendar="XNYS",
    )

    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2024-01-20 12:00:00")
    )
    covers = storage.cache_covers(
        "yahoo",
        "BTC",
        "1d",
        provisional_date.date(),
        provisional_date.date(),
        calendar="XNYS",
    )
    assert covers is False


def test_cache_staleness_respects_the_posting_lag_not_just_bucket_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bar fetched moments after its bucket closed, but before the
    posting-lag tolerance (12h for daily bars) has elapsed, is still fair
    game to be a provider's not-yet-finalised value -- it must still be
    treated as needing a refresh once the full lag has since passed without
    one, not accepted as final the instant the bucket itself closed."""
    from quantlab.data.calendar import daily_equity_bucket_settlement

    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    bar_date = pd.Timestamp("2024-02-01")
    bucket_end = daily_equity_bucket_settlement(bar_date, calendar="XNYS")

    just_after_close = bucket_end + pd.Timedelta(minutes=1)
    monkeypatch.setattr(storage_module, "_utc_now", lambda: just_after_close)
    storage.write_symbol(
        _ohlcv([bar_date], symbol="EARLY"), "yahoo", "EARLY", "1d", calendar="XNYS"
    )

    # Well past the 12h posting lag, still no refresh.
    after_posting_lag = bucket_end + pd.Timedelta(hours=13)
    monkeypatch.setattr(storage_module, "_utc_now", lambda: after_posting_lag)
    covers = storage.cache_covers(
        "yahoo", "EARLY", "1d", bar_date.date(), bar_date.date(), calendar="XNYS"
    )
    assert covers is False


def test_cache_staleness_check_ignores_a_stale_row_outside_the_requested_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row's own bucket can sit entirely outside the requested range (an
    older bar the frontier has since moved past) -- a narrow re-download of
    the requested range can never refresh that unrelated row, so it must
    never keep the cache permanently 'not covering' requests that don't
    touch it. Only a stale row that overlaps the requested range should
    force a refresh (see the sibling test below)."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")

    # Jan 16: fetched while still provisional -- stays stale forever, but
    # it's outside every later request for Jan 17 alone.
    still_open_now = pd.Timestamp("2024-01-16 19:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: still_open_now)
    storage.write_symbol(
        _ohlcv([pd.Timestamp("2024-01-16")], symbol="AAA"),
        "yahoo",
        "AAA",
        "1d",
        calendar="XNYS",
    )

    # Jan 17: fetched properly, well after its own close + posting lag.
    proper_time = pd.Timestamp("2024-01-18 10:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: proper_time)
    storage.write_symbol(
        _ohlcv([pd.Timestamp("2024-01-17")], symbol="AAA"),
        "yahoo",
        "AAA",
        "1d",
        calendar="XNYS",
    )

    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2024-01-20 12:00:00")
    )
    # A request for Jan 17 only is unaffected by Jan 16's stale row.
    assert storage.cache_covers(
        "yahoo", "AAA", "1d", date(2024, 1, 17), date(2024, 1, 17), calendar="XNYS"
    )
    # A request that actually includes Jan 16 still catches its staleness.
    assert not storage.cache_covers(
        "yahoo", "AAA", "1d", date(2024, 1, 16), date(2024, 1, 17), calendar="XNYS"
    )


def test_cache_covers_rejects_a_v3_row_with_unknown_provenance(
    tmp_path: Path,
) -> None:
    """A v3-format cache file's per-row provenance guarantee is only real if
    a row lacking it (e.g. one written directly via :meth:`ParquetStorage.
    save`, bypassing :meth:`write_symbol`; production never does this) is
    treated as unverified rather than silently trusted -- otherwise the
    guarantee is bypassable by any code path that skips ``write_symbol``."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    bar_date = pd.Timestamp("2024-01-16")
    path = storage._cache_path("yahoo", "AAA", "1d")
    storage.save(_ohlcv([bar_date], symbol="AAA"), path)

    assert not storage.cache_covers(
        "yahoo", "AAA", "1d", bar_date.date(), bar_date.date(), calendar="XNYS"
    )


def test_cache_staleness_check_survives_an_empty_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty incoming write (e.g. a provider returning zero new rows for
    an already-covered range) must not crash the staleness check, and must
    not disturb the existing rows' own provenance."""
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    bar_date = pd.Timestamp("2024-01-16")

    still_open_now = pd.Timestamp("2024-01-16 19:00:00")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: still_open_now)
    storage.write_symbol(
        _ohlcv([bar_date], symbol="AAPL"), "yahoo", "AAPL", "1d", calendar="XNYS"
    )

    empty = _ohlcv([], symbol="AAPL").astype(
        {"timestamp": "datetime64[ns]", "symbol": "object"}
    )
    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2024-01-17 08:00:00")
    )
    storage.write_symbol(empty, "yahoo", "AAPL", "1d", calendar="XNYS")

    monkeypatch.setattr(
        storage_module, "_utc_now", lambda: pd.Timestamp("2024-01-18 12:00:00")
    )
    covers = storage.cache_covers(
        "yahoo", "AAPL", "1d", bar_date.date(), bar_date.date(), calendar="XNYS"
    )
    assert covers is False


def test_forced_replacement_purges_a_session_bar_crossing_utc_midnight(
    tmp_path: Path,
) -> None:
    """A naive [UTC midnight, next UTC midnight) replace window would miss a
    genuine session bar for a calendar whose local session opens before UTC
    midnight of its own label date (e.g. XASX under daylight saving,
    UTC+11) -- the stale bar would silently survive a forced replacement
    meant to purge it."""
    from quantlab.data.calendar import sessions

    schedule = sessions("XASX", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
    session_date = schedule.index[0]
    market_open = schedule.iloc[0]["market_open"]
    assert market_open.date() < session_date.date()  # confirms the crossing scenario

    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    stale = _ohlcv([market_open], symbol="BHP")
    storage.write_symbol(stale, "asx", "BHP", "1d", calendar="XASX")

    fresh = _ohlcv([market_open], symbol="BHP", volume=200.0)
    storage.write_symbol(
        fresh,
        "asx",
        "BHP",
        "1d",
        calendar="XASX",
        replace_start=session_date.date(),
        replace_end=session_date.date(),
    )
    raw = pd.read_parquet(storage._cache_path("asx", "BHP", "1d"))
    assert len(raw) == 1
    assert raw["volume"].iloc[0] == 200.0


def test_storage_deduplicates_the_first_write_and_validates_its_symbol(
    tmp_path: Path,
) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    duplicate = _ohlcv(["2024-01-01", "2024-01-01"])
    storage.write_symbol(duplicate, "yahoo", "AAA", "1d", calendar="XNYS")
    cached = storage.read_symbol("yahoo", "AAA", "1d", calendar="XNYS")
    assert cached is not None
    assert len(cached) == 1

    with pytest.raises(DataValidationError, match="rows for"):
        storage.write_symbol(
            _ohlcv(["2024-01-02"], symbol="BBB"), "yahoo", "AAA", "1d", calendar="XNYS"
        )


def test_storage_uses_a_versioned_cache_namespace(tmp_path: Path) -> None:
    from quantlab.data.storage import _CACHE_FORMAT_VERSION

    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    path = storage.write_symbol(
        _ohlcv(["2024-01-01"]), "yahoo", "AAA", "1d", calendar="XNYS"
    )
    # Asserted against the live constant, not a hardcoded literal, so this
    # test documents "the cache is namespaced by version" without needing an
    # update on every future version bump.
    assert path.relative_to(tmp_path / "cache").parts[0] == _CACHE_FORMAT_VERSION


def test_atomic_save_keeps_the_previous_file_when_serialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    path = storage.save(pd.DataFrame({"value": [1]}), tmp_path / "frame.parquet")

    def fail_to_parquet(self: pd.DataFrame, *_args: Any, **_kwargs: Any) -> None:
        del self
        raise OSError("simulated failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    with pytest.raises(OSError, match="simulated failure"):
        storage.save(pd.DataFrame({"value": [2]}), path)
    assert storage.load(path)["value"].tolist() == [1]
    assert not list(tmp_path.glob(".*.tmp"))


def test_loader_refuses_a_symbol_with_zero_usable_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _ohlcv(["2024-01-01", "2024-01-02"], symbol="BBB").to_csv(
        raw / "AAA.csv", index=False
    )
    config = ExperimentConfig.from_dict(
        {
            "experiment_name": "mislabeled_csv",
            "data": {
                "instruments": [{"symbol": "AAA", "source": "csv", "calendar": "24/7"}],
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    with pytest.raises(DataValidationError, match="reduced universe"):
        DataLoader(raw_dir=raw).load(config)


def test_build_source_distinguishes_csv_from_remote_sources() -> None:
    with pytest.raises(DataDownloadError, match="handled directly"):
        build_source("csv")


def test_loader_preserves_a_falsy_injected_storage(tmp_path: Path) -> None:
    class FalsyStorage(ParquetStorage):
        def __bool__(self) -> bool:
            return False

    storage = FalsyStorage(tmp_path / "cache", tmp_path / "metadata")
    assert DataLoader(storage=storage).storage is storage


def test_metadata_names_are_case_safe_and_windows_device_safe(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path / "cache", tmp_path / "metadata")
    upper = storage.write_metadata("NUL", {"value": np.float32(1.0)})
    lower = storage.write_metadata("nul", {"value": np.float32(2.0)})
    assert upper != lower
    assert upper.is_file()
    assert lower.is_file()
