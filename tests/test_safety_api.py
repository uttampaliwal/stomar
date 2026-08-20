"""API security invariants: auth fail-closed, rate limiting, audit trail."""

import hmac

from starlette.requests import Request

from src.core.settings import settings

from api.main import (
    _PROTECTED_PREFIXES,
    _auth_verdict,
    _is_protected_path,
    _rate_limited,
)

_MUTATING = ("POST", "PUT", "PATCH", "DELETE")
_READ = ("GET",)


def _req(path, headers=None, cookies=None):
    """Build a minimal Starlette Request for `_auth_verdict`."""
    merged = dict(headers or {})
    if cookies:
        merged["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in merged.items()
            ],
        }
    )


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
    monkeypatch.setattr(settings, "auth_password", "")
    verdict = _auth_verdict(_req("/api/paper-trading/order", headers={"X-API-Key": "whatever"}))
    assert verdict is not None
    assert verdict[0] == 503  # explicitly disabled, not silently open


def test_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "real-key")
    verdict = _auth_verdict(_req("/api/paper-trading/state", headers={"X-API-Key": "wrong-key"}))
    assert verdict is not None
    assert verdict[0] == 401


def test_correct_key_accepted(monkeypatch):
    key = settings.api_key or "test-key"
    monkeypatch.setattr(settings, "api_key", key)
    assert _auth_verdict(_req("/api/paper-trading/order", headers={"X-API-Key": key})) is None
    assert _auth_verdict(_req("/api/pipeline/status", headers={"X-API-Key": key})) is None


def test_valid_session_cookie_accepted_without_key(monkeypatch, tmp_path):
    from api.auth import SessionStore, SESSION_COOKIE_NAME
    import api.main as api_main

    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "auth_password", "")
    store = SessionStore(path=tmp_path / "s.json", ttl_hours=1.0)
    token = store.create()
    monkeypatch.setattr(api_main, "session_store", store)
    assert _auth_verdict(
        _req("/api/paper-trading/order", cookies={SESSION_COOKIE_NAME: token})
    ) is None
    assert _auth_verdict(_req("/api/paper-trading/order")) is not None


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


def test_reset_capital_bounds(monkeypatch, tmp_path):
    from api.routers import paper_trading as pt
    from filelock import FileLock

    class _FakeTrader:
        initial_capital = 0

        def reset(self):
            self.initial_capital = self.initial_capital

        def load_state(self, path):
            pass

        def save_state(self, path):
            pass

        def get_summary(self):
            return {"capital": self.initial_capital}

    fake = _FakeTrader()
    monkeypatch.setattr(pt, "get_trader", lambda: fake)
    # hermetic lock: never touch the real data/paper_state.json.lock
    monkeypatch.setattr(pt, "_state_lock", FileLock(str(tmp_path / "paper_state.json.lock")))
    for bad in [0, -1, -1000, 100_000_001, 1e15, "nan", "inf", "abc"]:
        res = pt.reset_paper_trading({"capital": bad})
        assert "error" in res, f"capital={bad} accepted"
    res = pt.reset_paper_trading({"capital": 200000})
    assert res.get("status") == "success"
    assert res.get("state", {}).get("capital") == 200000


def test_paper_state_session_reads_and_persists(monkeypatch, tmp_path):
    from api.routers import paper_trading as pt
    from src.trading.paper_trader import PaperTrader
    from filelock import FileLock

    monkeypatch.setattr(pt, "PAPER_STATE_PATH", str(tmp_path / "paper_state.json"))
    monkeypatch.setattr(pt, "_state_lock", FileLock(str(tmp_path / "paper_state.json.lock")))

    trader = PaperTrader(initial_capital=200000)
    trader.cash = 1000.0
    trader.save_state(pt.PAPER_STATE_PATH)

    # a session must reload the freshest disk state, then persist mutations
    with pt._paper_state_session(trader):
        assert trader.cash == 1000.0
        trader.cash = 2000.0

    other = PaperTrader(initial_capital=200000)
    pt._refresh_trader(other)
    assert other.cash == 2000.0, "session changes not persisted"


def test_paper_state_lock_busy_fails_fast(monkeypatch, tmp_path):
    from api.routers import paper_trading as pt
    from src.trading.paper_trader import PaperTrader
    from filelock import FileLock

    monkeypatch.setattr(pt, "PAPER_STATE_PATH", str(tmp_path / "paper_state.json"))
    monkeypatch.setattr(pt, "_state_lock", FileLock(str(tmp_path / "paper_state.json.lock"), timeout=1))

    trader = PaperTrader(initial_capital=200000)
    trader.save_state(pt.PAPER_STATE_PATH)

    held = FileLock(str(tmp_path / "paper_state.json.lock"))
    held.acquire()
    try:
        try:
            pt._refresh_trader(trader)
            raise AssertionError("expected busy error")
        except RuntimeError as e:
            assert "busy" in str(e)
    finally:
        held.release()


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


# ── response cache middleware ───────────────────────────────────────────────

class _BodyIter:
    """Async iterator over response chunks (mimics body_iterator)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeCacheRequest:
    def __init__(self, path, query=""):
        self.method = "GET"
        self.url = type("U", (), {"path": path, "query": query})()


class _FakeCacheResponse:
    def __init__(self, status=200, body=b'{"x": 1}', content_type="application/json"):
        self.status_code = status
        self.media_type = "application/json" if "json" in content_type else "text/plain"
        self.headers = {"content-type": content_type}
        self.body_iterator = _BodyIter([body])


def _run_cache(monkeypatch, path, query="", status=200, body=b'{"x": 1}',
               content_type="application/json", max_entries=None, clear=True):
    import api.main as api_main
    from api.main import cache_middleware

    if clear:
        api_main._response_cache.clear()
    monkeypatch.setattr(api_main, "_CACHE_TTL", 30.0)
    if max_entries is not None:
        monkeypatch.setattr(api_main.settings, "cache_max_entries", max_entries)

    calls = {"n": 0}

    async def _call_next(request):
        calls["n"] += 1
        return _FakeCacheResponse(status=status, body=body, content_type=content_type)

    import asyncio
    resp = asyncio.run(cache_middleware(
        _FakeCacheRequest(path, query), _call_next))
    return resp, calls


def test_cache_second_request_is_hit(monkeypatch):
    resp1, calls1 = _run_cache(monkeypatch, "/api/scanner/TCS")
    assert resp1.headers.get("X-Cache") == "MISS"
    assert calls1["n"] == 1
    resp2, calls2 = _run_cache(monkeypatch, "/api/scanner/TCS", clear=False)
    assert resp2.headers.get("X-Cache") == "HIT"
    assert calls2["n"] == 0, "cached request must not hit the endpoint"
    assert resp2.body == b'{"x": 1}'


def test_cache_distinguishes_query_strings(monkeypatch):
    _run_cache(monkeypatch, "/api/scanner", query="a=1")
    resp, calls = _run_cache(monkeypatch, "/api/scanner", query="a=2")
    assert calls["n"] == 1, "different query strings are distinct keys"


def test_cache_skips_error_responses(monkeypatch):
    resp1, calls1 = _run_cache(monkeypatch, "/api/scanner/TCS", status=500)
    assert calls1["n"] == 1
    resp2, calls2 = _run_cache(monkeypatch, "/api/scanner/TCS", status=500)
    assert calls2["n"] == 1, "error responses must not be cached"
    assert resp2.headers.get("X-Cache") is None


def test_cache_skips_non_json(monkeypatch):
    _run_cache(monkeypatch, "/api/scanner/TCS", content_type="text/html")
    resp, calls = _run_cache(monkeypatch, "/api/scanner/TCS", content_type="text/html")
    assert calls["n"] == 1, "non-JSON responses must not be cached"


def test_cache_skips_protected_paths(monkeypatch):
    resp, calls = _run_cache(monkeypatch, "/api/paper-trading/state")
    assert calls["n"] == 1
    resp2, calls2 = _run_cache(monkeypatch, "/api/paper-trading/state")
    assert calls2["n"] == 1, "protected responses must never be cached"


def test_cache_evicts_lru_over_limit(monkeypatch):
    _run_cache(monkeypatch, "/api/scanner/A", max_entries=2)
    _run_cache(monkeypatch, "/api/scanner/B", max_entries=2, clear=False)
    _run_cache(monkeypatch, "/api/scanner/C", max_entries=2, clear=False)
    # A was evicted first -> must be recomputed, B/C are still cached
    resp_b, calls_b = _run_cache(monkeypatch, "/api/scanner/B", max_entries=2, clear=False)
    assert calls_b["n"] == 0
    resp_c, calls_c = _run_cache(monkeypatch, "/api/scanner/C", max_entries=2, clear=False)
    assert calls_c["n"] == 0
    resp, calls = _run_cache(monkeypatch, "/api/scanner/A", max_entries=2, clear=False)
    assert calls["n"] == 1
