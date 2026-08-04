"""Tests for src/core/calendar.py (P2.3 NSE holidays)."""

from datetime import date

from src.core.calendar import is_trading_day, trading_days_since


class TestIsTradingDay:
    def test_weekdays_are_trading_days(self):
        assert is_trading_day(date(2026, 8, 3))   # Monday
        assert is_trading_day(date(2026, 8, 4))   # Tuesday

    def test_weekends_are_closed(self):
        assert not is_trading_day(date(2026, 8, 1))   # Saturday
        assert not is_trading_day(date(2026, 8, 2))   # Sunday

    def test_diwali_holiday_2025(self):
        assert not is_trading_day(date(2025, 10, 21))  # Diwali (Laxmi Puja)

    def test_holi_holiday_2026(self):
        assert not is_trading_day(date(2026, 3, 3))  # Holi

    def test_independence_day_2025(self):
        assert not is_trading_day(date(2025, 8, 15))

    def test_day_after_holiday_is_trading_day(self):
        assert is_trading_day(date(2025, 10, 20))  # day before Diwali
        assert is_trading_day(date(2025, 10, 23))  # after Diwali Balipratipada


class TestTradingDaysSince:
    def test_excludes_holidays_and_weekends(self):
        days = trading_days_since("2025-10-20", "2025-10-27")
        # Oct 20 Mon, 21 Tue (Diwali — excluded), 22 Wed (excluded),
        # 23 Thu, 24 Fri, 25-26 weekend, 27 Mon
        assert days == ["2025-10-20", "2025-10-23", "2025-10-24", "2025-10-27"]

    def test_empty_range(self):
        assert trading_days_since("2026-08-01", "2026-08-02") == []
