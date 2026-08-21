"""Tests for market-timezone helpers (Asia/Kolkata day/week boundaries)."""

from datetime import date, datetime, timezone

from src.core.timeutils import (
    IST,
    ist_today,
    iso_week_key,
    now_ist,
    parse_quote_timestamp,
)


def test_now_and_today_are_ist():
    now = now_ist()
    assert now.tzinfo is not None
    assert now.utcoffset().total_seconds() == 5.5 * 3600
    assert isinstance(ist_today(), date)


def test_iso_week_key_format():
    key = iso_week_key(date(2026, 8, 21))
    assert key == "2026-W34"


def test_parse_naive_timestamp_assumes_ist_not_utc():
    # A naive IST wall-clock quote from "now" must NOT look 5h30m old
    # (the old UTC assumption made live NSE quotes appear stale).
    naive = datetime.now(IST).replace(tzinfo=None).isoformat()
    parsed = parse_quote_timestamp(naive)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
    age = datetime.now(timezone.utc) - parsed
    assert abs(age.total_seconds()) < 60


def test_parse_tz_aware_timestamp_passthrough():
    ts = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    assert parse_quote_timestamp(ts) == ts


def test_parse_z_suffix():
    parsed = parse_quote_timestamp("2026-08-21T09:15:00Z")
    assert parsed is not None
    assert parsed.hour == 9


def test_parse_garbage_returns_none():
    assert parse_quote_timestamp("") is None
    assert parse_quote_timestamp("not-a-date") is None
    assert parse_quote_timestamp(None) is None
