"""Trading-calendar and bar-settlement helpers.

Canonical market-data timestamps are timezone-naive UTC. Every named calendar
(any name accepted by ``pandas_market_calendars``, e.g. ``"XNYS"``, ``"XHKG"``,
``"CME_Equity"``) settles bars against its own maintained schedule; the
special sentinel ``"24/7"`` is a continuous calendar handled without
consulting ``pandas_market_calendars`` at all.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay

from quantlab.exceptions import DataValidationError

_TWENTY_FOUR_SEVEN = "24/7"


def is_247(calendar: str) -> bool:
    """Return whether ``calendar`` is the continuous (non-exchange) sentinel."""
    return calendar == _TWENTY_FOUR_SEVEN


@functools.cache
def _mcal_calendar(name: str) -> mcal.MarketCalendar:
    """Return the cached ``pandas_market_calendars`` calendar for ``name``.

    Never call with ``"24/7"`` — check :func:`is_247` first.
    """
    try:
        return mcal.get_calendar(name)
    except RuntimeError as exc:
        raise DataValidationError(f"Unknown market calendar {name!r}: {exc}") from exc


def validate_calendar_name(calendar: str) -> None:
    """Raise ``DataValidationError`` if ``calendar`` is not a usable name."""
    if is_247(calendar):
        return
    _mcal_calendar(calendar)


def uniform_calendar(calendars: Iterable[str]) -> str | None:
    """Return the shared calendar name if every value is identical, else ``None``."""
    values = set(calendars)
    if len(values) == 1:
        return next(iter(values))
    return None


@functools.cache
def business_day_offset(calendar: str) -> CustomBusinessDay:
    """Return the cached session offset (holidays included) for ``calendar``.

    Only meaningful for a non-24/7 calendar — callers check :func:`is_247` first.
    """
    return cast("CustomBusinessDay", _mcal_calendar(calendar).holidays())


# Kept as a plain module constant for ergonomics: XNYS remains the default and
# by far the most common calendar in the codebase and its tests.
XNYS_BUSINESS_DAY: CustomBusinessDay = business_day_offset("XNYS")


def _normalise_utc_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the UTC calendar day for a timestamp."""
    value = pd.Timestamp(ts)
    if pd.isna(value):
        raise DataValidationError("A valid timestamp is required for settlement.")
    if value.tzinfo is not None:
        value = value.tz_convert("UTC").tz_localize(None)
    return value.normalize()


def holidays_between(
    calendar: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Return ``calendar``'s holiday dates within the inclusive range."""
    start_day = _normalise_utc_day(start)
    end_day = _normalise_utc_day(end)
    if start_day > end_day:
        raise ValueError("start must be on or before end.")

    offset = business_day_offset(calendar)
    # The runtime attribute exists, but the pandas stub does not expose it.
    raw_holidays = pd.DatetimeIndex(np.asarray(getattr(offset, "holidays")))  # noqa: B009
    return raw_holidays[(raw_holidays >= start_day) & (raw_holidays <= end_day)]


def session_weekmask(calendar: str) -> str:
    """Return ``calendar``'s own weekday pattern, for ``numpy.busday_count``.

    Never assume Monday-Friday: a calendar such as XSAU trades Sunday-
    Thursday, so any business-day arithmetic must use its real weekmask
    (e.g. ``numpy.busday_count``'s default weekmask is Monday-Friday and
    would silently miscount for such a calendar).
    """
    offset = business_day_offset(calendar)
    # The runtime attribute exists, but the pandas stub does not expose it.
    return cast(str, getattr(offset, "weekmask"))  # noqa: B009


_SCHEDULE_START = pd.Timestamp("1950-01-01")
_SCHEDULE_END = pd.Timestamp("2075-12-31")


@functools.cache
def _schedule_for(calendar: str) -> pd.DataFrame:
    """Return a process-cached open/close schedule for ``calendar``."""
    schedule: pd.DataFrame = _mcal_calendar(calendar).schedule(
        start_date=_SCHEDULE_START, end_date=_SCHEDULE_END
    )
    return schedule


def sessions(calendar: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return ``calendar``'s sessions with timezone-naive UTC session times.

    Always has ``market_open``/``market_close``. A calendar with an official
    intraday break (e.g. XHKG's lunch recess) also has ``break_start``/
    ``break_end``, normalized the same way -- never left tz-aware while the
    other two columns are tz-naive, which would silently corrupt any
    comparison between them. A calendar without a break simply has no break
    columns, exactly as ``pandas_market_calendars`` reports it.
    """
    start_day = _normalise_utc_day(start)
    end_day = _normalise_utc_day(end)
    if start_day > end_day:
        raise ValueError("start must be on or before end.")

    if start_day >= _SCHEDULE_START and end_day <= _SCHEDULE_END:
        schedule = _schedule_for(calendar).loc[start_day:end_day].copy()
    else:
        schedule = (
            _mcal_calendar(calendar)
            .schedule(start_date=start_day, end_date=end_day)
            .copy()
        )

    for column in ("market_open", "market_close", "break_start", "break_end"):
        if column in schedule.columns:
            schedule[column] = pd.to_datetime(
                schedule[column], utc=True
            ).dt.tz_localize(None)
    schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    return schedule


def has_session_break(calendar: str) -> bool:
    """Return whether ``calendar`` has an official intraday break (recess).

    ``24/7`` and most equity/futures calendars never do; a handful (e.g.
    XHKG's lunch recess) do, and a session's tradable time is then two
    disjoint intervals, not one continuous ``[market_open, market_close)``
    block -- see :func:`sessions`.
    """
    if is_247(calendar):
        return False
    return "break_start" in _schedule_for(calendar).columns


def session_labels(calendar: str, ts: pd.Series) -> pd.Series:
    """Return each timestamp's real trading-session date, index-aligned to ``ts``.

    Never a naive UTC calendar-day boundary (``ts.dt.date`` /
    ``ts.dt.normalize()``): a large positive UTC offset (e.g. Sydney,
    +10/+11) can push a session's local-morning open into the *previous* UTC
    calendar day, silently splitting one real session's bars across two
    different "days" and corrupting anything that groups or compares by day
    for such a calendar. ``ts`` must be sorted ascending.

    Matches each timestamp to the first session whose close has not yet
    happened at that instant (smallest ``market_close >= ts``) -- the
    session actually "in progress or next up" at that moment, covering the
    whole gap since the previous session's close. This is deliberately not
    a match against the nearest ``market_open``: for a calendar whose
    sessions open many hours after UTC midnight (e.g. XNYS, opening at
    13:30 UTC), a daily bar conventionally timestamped at UTC midnight of
    its own trading date is *closer in raw time* to the previous session's
    open than to its own -- "nearest open" would silently misattribute
    every such bar to the day before, merging or dropping bars under any
    grouping built on this label. Matching on the close instead correctly
    keeps a midnight-of-D bar on D (D's own close is still hours away, the
    previous session's is a full day-plus behind) while still handling a
    genuine intraday bar the same way "nearest" did (mid-session, at-open,
    or well-before-open all still resolve to their own session, since its
    close is the next one due) and still correctly handling a session that
    opens before UTC midnight of its own label date.
    """
    # A forward search needs the *next* session's close on or after every
    # row, including the last one -- e.g. a genuinely post-market timestamp
    # on `ts.max()`'s own calendar date can fall after that date's own
    # close, needing the following session. Padding the fetch window well
    # past any realistic holiday closure (a market closed 10+ consecutive
    # calendar days is not a real case any calendar here models) guarantees
    # one is always available; sessions() only returns real session rows,
    # so the extra padding costs a few unused trailing rows, never a wrong
    # match.
    schedule = sessions(calendar, ts.min(), ts.max() + pd.Timedelta(days=10))
    if schedule.empty:
        return cast("pd.Series", ts.dt.normalize())
    # `ts` and `schedule["market_close"]` come from independent sources (a
    # caller's own data vs. pandas_market_calendars' schedule) that need not
    # share the same datetime64 resolution (e.g. Parquet-cached data can
    # come back as `[ms]` while the schedule is `[us]`) -- merge_asof
    # requires an exact dtype match on its join keys, not just "both
    # datetime64", so both sides are normalised to `[ns]` (this project's
    # own canonical resolution) right before the merge.
    lookup = pd.merge_asof(
        pd.DataFrame({"ts": ts.to_numpy(dtype="datetime64[ns]")}),
        pd.DataFrame(
            {
                "session": schedule.index.to_numpy(),
                "close": schedule["market_close"].to_numpy(dtype="datetime64[ns]"),
            }
        ).sort_values("close"),
        left_on="ts",
        right_on="close",
        direction="forward",
    )
    return pd.Series(lookup["session"].to_numpy(), index=ts.index)


def _market_close_for_session(calendar: str, day: pd.Timestamp) -> pd.Timestamp | None:
    """Return a session's timezone-naive UTC close on ``calendar``, or ``None``."""
    schedule = _schedule_for(calendar)
    if _SCHEDULE_START <= day <= _SCHEDULE_END:
        if day not in schedule.index:
            return None
        raw_close: Any = schedule.at[day, "market_close"]
    else:
        one_day: pd.DataFrame = _mcal_calendar(calendar).schedule(
            start_date=day, end_date=day
        )
        if one_day.empty:
            return None
        raw_close = one_day.iloc[0]["market_close"]

    close = pd.Timestamp(raw_close)
    if close.tzinfo is None:
        raise DataValidationError(
            f"{calendar} returned a timezone-naive close for {day.date()}."
        )
    return close.tz_convert("UTC").tz_localize(None)


# Nominal observed spacing used by frequency validation. Calendar-month
# settlement is calculated separately and never relies on the 30-day value.
FREQUENCY_TIMEDELTA: dict[str, pd.Timedelta] = {
    "1d": pd.Timedelta(days=1),
    "1D": pd.Timedelta(days=1),
    "1h": pd.Timedelta(hours=1),
    "1H": pd.Timedelta(hours=1),
    "1w": pd.Timedelta(weeks=1),
    "1W": pd.Timedelta(weeks=1),
    "1mo": pd.Timedelta(days=30),
    "1M": pd.Timedelta(days=30),
}

MONTHLY_FREQUENCIES = frozenset({"1mo", "1M"})
PERIODIC_FREQUENCIES = frozenset({"1w", "1W", "1mo", "1M"})
DAILY_FREQUENCIES = frozenset({"1d", "1D"})


def last_trading_day_on_or_before(
    ts: pd.Timestamp, *, calendar: str = "XNYS"
) -> pd.Timestamp:
    """Return the latest trading day on or before ``ts``, normalized."""
    day = _normalise_utc_day(ts)
    if is_247(calendar):
        return day
    return pd.Timestamp(business_day_offset(calendar).rollback(day)).normalize()


def first_trading_day_on_or_after(
    ts: pd.Timestamp, *, calendar: str = "XNYS"
) -> pd.Timestamp:
    """Return the earliest trading day on or after ``ts``, normalized."""
    day = _normalise_utc_day(ts)
    if is_247(calendar):
        return day
    return pd.Timestamp(business_day_offset(calendar).rollforward(day)).normalize()


def is_session_day(calendar: str, days: pd.DatetimeIndex) -> np.ndarray:
    """Return a boolean array: whether each (UTC, normalized) day is a session.

    ``"24/7"`` treats every day as a session. For a named exchange calendar, a
    day is a session iff it is in that calendar's own schedule from
    ``pandas_market_calendars`` -- never a hardcoded Monday-Friday
    assumption, which would be wrong for a calendar whose weekend falls on
    different days (e.g. XSAU's Friday-Saturday weekend, Sunday-Thursday
    trading week: a hardcoded weekday check would mark a real XSAU Sunday
    session closed and a real XSAU Friday closure open).
    """
    normalized = pd.DatetimeIndex([_normalise_utc_day(day) for day in days])
    if is_247(calendar):
        return np.ones(len(normalized), dtype=bool)
    if len(normalized) == 0:
        return np.zeros(0, dtype=bool)
    start_day, end_day = normalized.min(), normalized.max()
    if start_day >= _SCHEDULE_START and end_day <= _SCHEDULE_END:
        session_index = _schedule_for(calendar).index
    else:
        session_index = (
            _mcal_calendar(calendar)
            .schedule(start_date=start_day, end_date=end_day)
            .index
        )
    return np.asarray(normalized.isin(session_index), dtype=bool)


def daily_equity_bucket_settlement(
    ts: pd.Timestamp, *, calendar: str = "XNYS"
) -> pd.Timestamp:
    """Return the close boundary for an equity daily bar dated ``ts``.

    A genuine session on ``calendar`` uses its scheduled close. A non-session
    date has no exchange close, so it uses the following UTC midnight as a
    conservative boundary instead of inventing a close time.
    """
    day = _normalise_utc_day(ts)
    close = _market_close_for_session(calendar, day)
    return close if close is not None else day + pd.Timedelta(days=1)


_WEEKDAY_ABBREVIATIONS = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}


@functools.cache
def _structural_trading_weekdays(calendar: str) -> frozenset[int]:
    """Return the weekday numbers (Mon=0..Sun=6) this calendar trades on.

    Derived from the offset's ``weekmask`` alone -- the calendar's fixed
    weekly pattern, independent of any specific date's holidays. A one-off
    holiday (e.g. Thanksgiving, a Thursday) must never be mistaken for the
    week's structural boundary the way a genuine weekend is.
    """
    weekmask: str = cast("Any", business_day_offset(calendar)).weekmask
    return frozenset(_WEEKDAY_ABBREVIATIONS[name] for name in weekmask.split())


def weekly_bucket_start(ts: pd.Timestamp, *, calendar: str = "XNYS") -> pd.Timestamp:
    """Return the label date of the trading week containing ``ts``.

    The mirror of :func:`weekly_bucket_settlement`: the first day of the
    same structural trading-weekday run (e.g. Sunday for XSAU, Monday for
    XNYS), not an intraday instant -- for labelling a resampled weekly bar
    the same way every other resampling target is labelled, a plain
    calendar-day date. Always names the *same* week as
    ``weekly_bucket_settlement`` for the same ``ts`` (a rest day anchors to
    the week that just ended in both, never a different week from one
    function to the other), so grouping by either gives the same partition.
    """
    day = _normalise_utc_day(ts)
    if is_247(calendar):
        return day - pd.Timedelta(days=day.dayofweek)

    trading_weekdays = _structural_trading_weekdays(calendar)
    cursor = day
    if cursor.dayofweek not in trading_weekdays:
        # Same rest-day anchoring as weekly_bucket_settlement: a rest day
        # belongs to the week that just ended, not the one about to start.
        while cursor.dayofweek not in trading_weekdays:
            cursor = cursor - pd.Timedelta(days=1)
    while (cursor - pd.Timedelta(days=1)).dayofweek in trading_weekdays:
        cursor = cursor - pd.Timedelta(days=1)
    return cursor


def weekly_bucket_settlement(
    ts: pd.Timestamp, *, calendar: str = "XNYS"
) -> pd.Timestamp:
    """Return the close of the trading week containing ``ts``."""
    day = _normalise_utc_day(ts)
    if is_247(calendar):
        week_start = day - pd.Timedelta(days=day.dayofweek)
        return week_start + pd.Timedelta(weeks=1)

    # The trading week containing `ts` is bounded by *this calendar's own*
    # structural rest days, never a fixed Monday-Sunday window -- wrong for
    # a non-Western trading week (e.g. XSAU trades Sunday-Thursday, so
    # Monday is mid-week, not the start). Membership uses the weekmask only
    # (not `is_session_day`, which also excludes one-off holidays) so a
    # mid-week holiday like Thanksgiving never gets mistaken for the
    # boundary of the week. A rest day anchors backward onto the week that
    # just ended (a Saturday belongs to the week whose last trading weekday
    # was Friday, exactly as it does for XNYS today); a trading weekday
    # walks forward to the last trading weekday of its own run. Only once
    # that structural end date is found does `last_trading_day_on_or_before`
    # resolve an actual holiday landing exactly on it (e.g. a market-holiday
    # Friday rolls back to Thursday's real close).
    trading_weekdays = _structural_trading_weekdays(calendar)
    cursor = day
    if cursor.dayofweek in trading_weekdays:
        while (cursor + pd.Timedelta(days=1)).dayofweek in trading_weekdays:
            cursor = cursor + pd.Timedelta(days=1)
    else:
        while cursor.dayofweek not in trading_weekdays:
            cursor = cursor - pd.Timedelta(days=1)
    cursor = last_trading_day_on_or_before(cursor, calendar=calendar)
    return daily_equity_bucket_settlement(cursor, calendar=calendar)


def monthly_bucket_settlement(
    ts: pd.Timestamp, *, calendar: str = "XNYS"
) -> pd.Timestamp:
    """Return the close of the calendar month containing ``ts``."""
    day = _normalise_utc_day(ts)
    month_start = pd.Timestamp(year=day.year, month=day.month, day=1)
    next_month_start = month_start + pd.DateOffset(months=1)
    if is_247(calendar):
        return next_month_start

    last_session = last_trading_day_on_or_before(
        next_month_start - pd.Timedelta(days=1), calendar=calendar
    )
    return daily_equity_bucket_settlement(last_session, calendar=calendar)


def bar_bucket_end(
    ts: pd.Series, frequency: str, *, calendar: str = "XNYS"
) -> pd.Series:
    """Return each bar's settlement instant as timezone-naive UTC."""
    if frequency not in FREQUENCY_TIMEDELTA:
        raise DataValidationError(
            f"Unsupported frequency {frequency!r}; expected one of "
            f"{sorted(FREQUENCY_TIMEDELTA)}."
        )

    timestamps = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
    if timestamps.isna().any():
        raise DataValidationError("Bar timestamps must not be missing.")

    if frequency in MONTHLY_FREQUENCIES:

        def settle_monthly(value: Any) -> pd.Timestamp:
            return monthly_bucket_settlement(pd.Timestamp(value), calendar=calendar)

        return timestamps.map(settle_monthly)
    if frequency in PERIODIC_FREQUENCIES:

        def settle_weekly(value: Any) -> pd.Timestamp:
            return weekly_bucket_settlement(pd.Timestamp(value), calendar=calendar)

        return timestamps.map(settle_weekly)
    if frequency in DAILY_FREQUENCIES and not is_247(calendar):

        def settle_daily(value: Any) -> pd.Timestamp:
            return daily_equity_bucket_settlement(
                pd.Timestamp(value), calendar=calendar
            )

        return timestamps.map(settle_daily)
    return timestamps + FREQUENCY_TIMEDELTA[frequency]
