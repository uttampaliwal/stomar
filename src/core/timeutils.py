"""Market-time helpers.

The NSE trades in Asia/Kolkata (IST, UTC+05:30, no DST). Every daily
boundary that gates trading — the order-budget ledger files, risk-state
day/week keys — must be computed in IST so that behaviour never depends on
the host machine's local timezone. On a UTC VPS, ``date.today()`` rolls at
05:30 IST, which lands mid-session and silently shifts the daily budget
and loss-limit windows.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """Current time as a tz-aware Asia/Kolkata datetime."""
    return datetime.now(IST)


def ist_today() -> date:
    """Today's date according to the market timezone (Asia/Kolkata)."""
    return now_ist().date()


def iso_week_key(d: date | None = None) -> str:
    """ISO year-week key (e.g. ``2026-W34``) used for weekly loss windows."""
    d = d if d is not None else ist_today()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def parse_quote_timestamp(ts: object) -> datetime | None:
    """Parse a quote timestamp into a tz-aware UTC datetime.

    Naive timestamps are interpreted as Asia/Kolkata — every NSE quote
    source used by this project reports IST wall-clock time. Assuming UTC
    instead made live IST quotes look 5h30m stale and blocked execution.

    Returns None when the value cannot be parsed at all.
    """
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, date):
        dt = datetime(ts.year, ts.month, ts.day)
    else:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(timezone.utc)
