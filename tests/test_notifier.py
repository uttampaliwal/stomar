"""Tests for src/core/notifier.py — Telegram/email channels + daily summary (B7).

Never hits the network: requests and smtplib are mocked or unconfigured.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_external_channels(monkeypatch):
    """Ensure no real channels are configured during tests."""
    for var in (
        "STOMAR_TELEGRAM_BOT_TOKEN",
        "STOMAR_TELEGRAM_CHAT_ID",
        "STOMAR_SMTP_HOST",
        "STOMAR_SMTP_USER",
        "STOMAR_SMTP_PASSWORD",
        "STOMAR_SMTP_TO",
    ):
        monkeypatch.delenv(var, raising=False)


def test_send_message_skips_when_nothing_configured():
    from src.core.notifier import send_message
    assert send_message("Subject", "Body") is False


def test_telegram_skips_when_no_config(monkeypatch):
    from src.core import notifier
    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: pytest.fail("should not call"))
    assert notifier._send_telegram("hello") is False


def test_telegram_posts_payload_and_returns_true(monkeypatch):
    from src.core import notifier
    monkeypatch.setenv("STOMAR_TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("STOMAR_TELEGRAM_CHAT_ID", "42")
    calls = {}

    class FakeResp:
        status_code = 200
        text = "ok"

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(notifier.requests, "post", fake_post)
    assert notifier._send_telegram("x" * 5000) is True
    assert "bot" + "tok123" in calls["url"]
    assert calls["json"]["chat_id"] == "42"
    # long bodies truncated to the Telegram limit
    assert len(calls["json"]["text"]) <= notifier._TELEGRAM_MAX_CHARS


def test_telegram_retries_once_then_fails(monkeypatch):
    from src.core import notifier
    monkeypatch.setenv("STOMAR_TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("STOMAR_TELEGRAM_CHAT_ID", "42")
    import requests as real_requests

    def fake_post(*a, **k):
        raise real_requests.RequestException("down")

    monkeypatch.setattr(notifier.requests, "post", fake_post)
    monkeypatch.setattr(notifier.time, "sleep", lambda s: None)
    assert notifier._send_telegram("hi") is False


def test_telegram_api_error_returns_false(monkeypatch):
    from src.core import notifier
    monkeypatch.setenv("STOMAR_TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("STOMAR_TELEGRAM_CHAT_ID", "42")

    class FakeResp:
        status_code = 401
        text = "Unauthorized"

    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: FakeResp())
    assert notifier._send_telegram("hi") is False


def test_notify_critical_dispatches_to_channels(monkeypatch):
    from src.core import notifier
    captured = {}
    monkeypatch.setattr(
        notifier, "send_message",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )
    notifier.notify("pipeline_failure", severity="critical", details="boom")
    assert captured["subject"] == "pipeline_failure"
    assert "boom" in captured["body"]
    assert "Traceback" not in captured["body"]


def test_notify_includes_traceback_for_exception(monkeypatch):
    from src.core import notifier
    captured = {}
    monkeypatch.setattr(
        notifier, "send_message",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )
    try:
        raise ValueError("kaboom")
    except ValueError as e:
        notifier.notify("drift", severity="critical", details="x", exc=e)
    assert "kaboom" in captured["body"]
    assert "Traceback" in captured["body"]


def test_daily_summary_builds_digest(monkeypatch):
    from src.core import notifier
    captured = {}
    monkeypatch.setattr(
        notifier, "send_message",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )
    health = {
        "timestamp": "2026-08-04T15:45:00",
        "status": "completed",
        "paper_days": 1,
        "kill_switch_active": False,
        "data_freshness_days": {"RELIANCE.NS": 0, "TCS.NS": 9, "INFY.NS": None},
        "model_age_days": {"RELIANCE.NS": 1, "TCS.NS": 2, "INFY.NS": None},
    }
    ok = notifier.send_daily_summary(
        health, daily={"decisions": 5, "errors": 0}, paper={"trades": 2}
    )
    assert ok is True
    assert captured["subject"] == "Daily summary"
    body = captured["body"]
    assert "2026-08-04" in body
    assert "Decisions: 5" in body
    assert "Errors: 0" in body
    assert "Paper trades: 2" in body
    assert "Paper days: 1" in body
    assert "TCS.NS" in body  # stale > 5 days flagged
    assert "INFY.NS" in body  # missing model flagged
    assert "RELIANCE.NS" not in body  # fresh ticker not flagged


def test_daily_summary_flags_kill_switch_and_skips(monkeypatch):
    from src.core import notifier
    captured = {}
    monkeypatch.setattr(
        notifier, "send_message",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )
    health = {
        "timestamp": "2026-08-04T15:45:00",
        "status": "completed",
        "paper_days": 3,
        "kill_switch_active": True,
        "data_freshness_days": {"RELIANCE.NS": 0},
        "model_age_days": {"RELIANCE.NS": 1},
    }
    notifier.send_daily_summary(health, daily={"decisions": 0, "skipped": True},
                                paper={"trades": 0, "skipped": True})
    body = captured["body"]
    assert "KILL SWITCH ACTIVE" in body
    assert "already ran today (skipped)" in body
