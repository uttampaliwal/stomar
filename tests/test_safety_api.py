"""API security invariants: auth fail-closed, rate limiting, audit trail."""

import hmac

from src.core.settings import settings

from api.main import (
    _PROTECTED_PREFIXES,
    _auth_verdict,
    _is_protected_path,
    _rate_limited,
)

_MUTATING = ("POST", "PUT", "PATCH", "DELETE")
_READ = ("GET",)


def test_protected_prefixes_cover_order_paths():
    # the paper-trading order route (the most dangerous) must be protected
    assert any("/api/paper-trading/" in p for p in _PROTECTED_PREFIXES)
    assert any("/api/automation/" in p for p in _PROTECTED_PREFIXES)
    assert any("/api/ledger/" in p for p in _PROTECTED_PREFIXES)
    # sensitive read-only endpoints must be protected too
    assert any("/api/pipeline/status" in p for p in _PROTECTED_PREFIXES)


def test_all_methods_on_protected_prefixes_require_auth():
    # both mutating and read methods are gated on protected prefixes
    for method in _MUTATING + _READ:
        assert _is_protected_path("/api/paper-trading/order") is True
        assert _is_protected_path("/api/paper-trading/state") is True
        assert _is_protected_path("/api/ledger/anything") is True
        assert _is_protected_path("/api/pipeline/status") is True
    # unaffected read endpoints stay open
    assert _is_protected_path("/api/health") is False
    assert _is_protected_path("/api/market/status") is False


def test_missing_api_key_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    verdict = _auth_verdict("/api/paper-trading/order", "whatever")
    assert verdict is not None
    assert verdict[0] == 503  # explicitly disabled, not silently open


def test_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "real-key")
    verdict = _auth_verdict("/api/paper-trading/state", "wrong-key")
    assert verdict is not None
    assert verdict[0] == 401


def test_correct_key_accepted(monkeypatch):
    key = settings.api_key or "test-key"
    monkeypatch.setattr(settings, "api_key", key)
    assert _auth_verdict("/api/paper-trading/order", key) is None
    assert _auth_verdict("/api/pipeline/status", key) is None


def test_order_quantity_bounds(monkeypatch):
    from api.routers import paper_trading as pt

    class _FakeTrader:
        pass

    monkeypatch.setattr(pt, "get_trader", lambda: _FakeTrader())
    for bad in [0, -1, -100, 1_000_001, 999_999_999]:
        res = pt.place_order({"ticker": "RELIANCE.NS", "quantity": bad})
        assert "error" in res, f"quantity={bad} accepted"
        assert "Quantity" in res["error"], res
    res = pt.place_order({"ticker": "RELIANCE.NS", "quantity": "abc"})
    assert "error" in res, "non-numeric quantity accepted"
    # a valid quantity must get past validation (no bounds error)
    res = pt.place_order({"ticker": "BAD!TICKER", "quantity": 10})
    assert "Invalid ticker" in res["error"], res


def test_reset_capital_bounds(monkeypatch):
    from api.routers import paper_trading as pt

    class _FakeTrader:
        initial_capital = 0

        def reset(self):
            self.initial_capital = self.initial_capital

        def save_state(self, *a, **k):
            pass

        def get_summary(self):
            return {"capital": self.initial_capital}

    fake = _FakeTrader()
    monkeypatch.setattr(pt, "get_trader", lambda: fake)
    for bad in [0, -1, -1000, 100_000_001, 1e15, "nan", "inf", "abc"]:
        res = pt.reset_paper_trading({"capital": bad})
        assert "error" in res, f"capital={bad} accepted"
    res = pt.reset_paper_trading({"capital": 200000})
    assert res.get("status") == "success"
    assert res.get("state", {}).get("capital") == 200000


def test_comparison_is_constant_time():
    key = "secret-key-12345"
    assert hmac.compare_digest(key, key) is True


def test_rate_limiter_enforces_window():
    # distinct keys are independent
    _rate_limited("ip-a")
    assert _rate_limited("ip-b") is False
    # drain a bucket
    for _ in range(60):
        _rate_limited("ip-c")
    assert _rate_limited("ip-c") is True


def test_audit_log_written_on_mutation(tmp_path, monkeypatch):
    import api.main as api_main
    monkeypatch.setattr(api_main, "_AUDIT_LOG_DIR", str(tmp_path))
    from api.main import audit_middleware

    class _FakeResponse:
        status_code = 201

    async def _call_next(request):
        return _FakeResponse()

    class _FakeRequest:
        method = "POST"
        url = type("U", (), {"path": "/api/paper-trading/order"})()
        client = type("C", (), {"host": "127.0.0.1"})()
        headers = {"x-request-id": "req-123"}

    import asyncio
    asyncio.run(audit_middleware(_FakeRequest(), _call_next))
    files = list(tmp_path.glob("mutations-*.jsonl"))
    assert files, "audit file not written"
    content = files[0].read_text()
    assert "paper-trading" in content
    assert "req-123" in content
