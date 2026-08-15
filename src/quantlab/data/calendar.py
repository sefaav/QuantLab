"""Trading-calendar and bar-settlement helpers.

Canonical market-data timestamps are timezone-naive UTC. Equity settlement
uses the maintained XNYS schedule; continuous markets use calendar periods.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay

from quantlab.exceptions import DataValidationError

_XNYS = mcal.get_calendar("XNYS")

# XNYS sessions, including exchange holidays and exceptional closures.
XNYS_BUSINESS_DAY: CustomBusinessDay = _XNYS.holidays()


def _normalise_utc_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the UTC calendar day for a timestamp."""
    value = pd.Timestamp(ts)
    if pd.isna(value):
        raise DataValidationError("A valid timestamp is required for settlement.")
    if value.tzinfo is not None:
        value = value.tz_convert("UTC").tz_localize(None)
    return value.normalize()


def xnys_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return XNYS holiday dates within the inclusive range."""
    start_day = _normalise_utc_day(start)
    end_day = _normalise_utc_day(end)
    if start_day > end_day:
        raise ValueError("start must be on or before end.")

    # The runtime attribute exists, but the pandas stub does not expose it.
    raw_holidays = pd.DatetimeIndex(
        np.asarray(getattr(XNYS_BUSINESS_DAY, "holidays"))  # noqa: B009
    )
    return raw_holidays[(raw_holidays >= start_day) & (raw_holidays <= end_day)]


_SCHEDULE_START = pd.Timestamp("1950-01-01")
_SCHEDULE_END = pd.Timestamp("2075-12-31")


@functools.lru_cache(maxsize=1)
def _xnys_schedule() -> pd.DataFrame:
    """Return a process-cached XNYS open/close schedule."""
    schedule: pd.DataFrame = _XNYS.schedule(
        start_date=_SCHEDULE_START, end_date=_SCHEDULE_END
    )
    return schedule


def xnys_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return XNYS sessions with timezone-naive UTC open and close times."""
    start_day = _normalise_utc_day(start)
    end_day = _normalise_utc_day(end)
    if start_day > end_day:
        raise ValueError("start must be on or before end.")

    if start_day >= _SCHEDULE_START and end_day <= _SCHEDULE_END:
        schedule = _xnys_schedule().loc[start_day:end_day].copy()
    else:
        schedule = _XNYS.schedule(start_date=start_day, end_date=end_day).copy()

    for column in ("market_open", "market_close"):
        schedule[column] = pd.to_datetime(schedule[column], utc=True).dt.tz_localize(
            None
        )
    schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    return schedule


def _market_close_for_session(day: pd.Timestamp) -> pd.Timestamp | None:
    """Return a session's timezone-naive UTC close, or ``None``."""
    schedule = _xnys_schedule()
    if _SCHEDULE_START <= day <= _SCHEDULE_END:
        if day not in schedule.index:
            return None
        raw_close: Any = schedule.at[day, "market_close"]
    else:
        one_day: pd.DataFrame = _XNYS.schedule(start_date=day, end_date=day)
        if one_day.empty:
            return None
        raw_close = one_day.iloc[0]["market_close"]

    close = pd.Timestamp(raw_close)
    if close.tzinfo is None:
        raise DataValidationError(
            f"XNYS returned a timezone-naive close for {day.date()}."
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
    ts: pd.Timestamp, *, is_247_market: bool
) -> pd.Timestamp:
    """Return the latest trading day on or before ``ts``, normalized."""
    day = _normalise_utc_day(ts)
    if is_247_market:
        return day
    return pd.Timestamp(XNYS_BUSINESS_DAY.rollback(day)).normalize()


def first_trading_day_on_or_after(
    ts: pd.Timestamp, *, is_247_market: bool
) -> pd.Timestamp:
    """Return the earliest trading day on or after ``ts``, normalized."""
    day = _normalise_utc_day(ts)
    if is_247_market:
        return day
    return pd.Timestamp(XNYS_BUSINESS_DAY.rollforward(day)).normalize()


def daily_equity_bucket_settlement(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the close boundary for an equity daily bar dated ``ts``.

    Genuine XNYS sessions use their scheduled close. A non-session date has
    no exchange close, so it uses the following UTC midnight as a conservative
    boundary instead of inventing a 16:00 close.
    """
    day = _normalise_utc_day(ts)
    close = _market_close_for_session(day)
    return close if close is not None else day + pd.Timedelta(days=1)


def weekly_bucket_settlement(ts: pd.Timestamp, *, is_247_market: bool) -> pd.Timestamp:
    """Return the close of the calendar week containing ``ts``."""
    day = _normalise_utc_day(ts)
    week_start = day - pd.Timedelta(days=day.dayofweek)
    if is_247_market:
        return week_start + pd.Timedelta(weeks=1)

    last_session = last_trading_day_on_or_before(
        week_start + pd.Timedelta(days=6), is_247_market=False
    )
    return daily_equity_bucket_settlement(last_session)


def monthly_bucket_settlement(ts: pd.Timestamp, *, is_247_market: bool) -> pd.Timestamp:
    """Return the close of the calendar month containing ``ts``."""
    day = _normalise_utc_day(ts)
    month_start = pd.Timestamp(year=day.year, month=day.month, day=1)
    next_month_start = month_start + pd.DateOffset(months=1)
    if is_247_market:
        return next_month_start

    last_session = last_trading_day_on_or_before(
        next_month_start - pd.Timedelta(days=1), is_247_market=False
    )
    return daily_equity_bucket_settlement(last_session)


def bar_bucket_end(ts: pd.Series, frequency: str, *, is_247_market: bool) -> pd.Series:
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
            return monthly_bucket_settlement(
                pd.Timestamp(value), is_247_market=is_247_market
            )

        return timestamps.map(settle_monthly)
    if frequency in PERIODIC_FREQUENCIES:

        def settle_weekly(value: Any) -> pd.Timestamp:
            return weekly_bucket_settlement(
                pd.Timestamp(value), is_247_market=is_247_market
            )

        return timestamps.map(settle_weekly)
    if frequency in DAILY_FREQUENCIES and not is_247_market:

        def settle_daily(value: Any) -> pd.Timestamp:
            return daily_equity_bucket_settlement(pd.Timestamp(value))

        return timestamps.map(settle_daily)
    return timestamps + FREQUENCY_TIMEDELTA[frequency]
