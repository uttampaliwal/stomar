"""NSE trading calendar (P2.3).

The pipeline previously assumed Mon-Fri are trading days, which caused
false "missed day" backfills on holidays (Diwali, Holi, etc.). This
module holds the official NSE holiday list and provides ``is_trading_day``.

Holidays are best-effort from NSE's published annual calendars. The
fallback remains Mon-Fri, so an unexpected new holiday only costs one
extra fetch cycle — it can never stop trading.

Usage:
    from src.core.calendar import is_trading_day, trading_days_since
    is_trading_day(datetime(2026, 3, 3))  # Holi -> False
"""

from datetime import date, datetime, timedelta

# Official NSE holiday calendar (2025 & 2026), as published by NSE.
# Format: (month, day)
_NSE_HOLIDAYS_2025 = {
    (1, 1),   # New Year
    (1, 26),  # Republic Day
    (2, 26),  # Maha Shivratri
    (3, 3),   # Holi
    (3, 31),  # Id-Ul-Fitr
    (4, 10),  # Shri Mahavir Jayanti
    (4, 14),  # Dr. B.R. Ambedkar Jayanti
    (4, 18),  # Good Friday
    (5, 12),  # Id-Ul-Adha
    (8, 15),  # Independence Day
    (8, 27),  # Muharram
    (10, 2),  # Mahatma Gandhi Jayanti
    (10, 21),  # Diwali (Laxmi Puja)
    (10, 22),  # Diwali Balipratipada
    (11, 5),  # Guru Nanak Jayanti
    (12, 25),  # Christmas
}

_NSE_HOLIDAYS_2026 = {
    (1, 26),  # Republic Day
    (2, 17),  # Maha Shivratri
    (3, 3),   # Holi
    (3, 20),  # Id-Ul-Fitr
    (4, 3),   # Shri Mahavir Jayanti
    (4, 14),  # Dr. B.R. Ambedkar Jayanti
    (5, 1),   # Maharashtra Day
    (5, 26),  # Id-Ul-Adha
    (8, 15),  # Independence Day
    (8, 18),  # Muharram
    (10, 2),  # Mahatma Gandhi Jayanti
    (10, 20),  # Diwali (Laxmi Puja)
    (10, 21),  # Diwali Balipratipada
    (11, 4),  # Guru Nanak Jayanti
    (12, 25),  # Christmas
}

_HOLIDAYS_BY_YEAR = {
    2025: _NSE_HOLIDAYS_2025,
    2026: _NSE_HOLIDAYS_2026,
}


def is_trading_day(dt: date | datetime) -> bool:
    """True if NSE was/is open on this date (weekday and not a holiday)."""
    if isinstance(dt, datetime):
        dt = dt.date()
    if dt.weekday() >= 5:
        return False
    holidays = _HOLIDAYS_BY_YEAR.get(dt.year, set())
    return (dt.month, dt.day) not in holidays


def trading_days_since(start_date: str, end_date: str) -> list[str]:
    """All NSE trading days in [start_date, end_date] inclusive, as strings."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days
