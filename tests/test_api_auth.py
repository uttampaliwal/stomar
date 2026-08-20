"""Integration tests for API authentication, rate limiting, and caching.

test_safety_api.py exercises the auth primitives (``_auth_verdict``,
``_rate_limited``) directly; this file drives the real middleware stack
through FastAPI's TestClient so a regression in middleware ordering or
registration shows up as an actual HTTP failure.

The cache tests register throwaway JSON routes on a cacheable prefix so
the assertions are deterministic and never touch the network or the
production data directory.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app

_TEST_KEY = "test-secret-key"
_ORIGINAL_CACHEABLE = list(api_main._CACHEABLE_PREFIXES)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_key", _TEST_KEY)
    api_main._rate_buckets.clear()
    api_main._response_cache.clear()
    with TestClient(app) as c:
        yield c
    api_main._rate_buckets.clear()
    api_main._response_cache.clear()


class TestAuthGate:
    def test_health_is_open_without_key(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_open_read_endpoint_works_without_key(self, client):
        r = client.get("/api/market/status")
        assert r.status_code != 401

    def test_protected_without_key_returns_401(self, client):
        r = client.get("/api/risk-guard/status")
        assert r.status_code == 401
        assert "api key" in r.text.lower()

    def test_protected_wrong_key_returns_401(self, client):
        r = client.get("/api/risk-guard/status", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_protected_valid_key_passes(self, client):
        r = client.get("/api/risk-guard/status", headers={"X-API-Key": _TEST_KEY})
        assert r.status_code == 200

    def test_protected_fails_closed_when_key_not_configured(self, client, monkeypatch):
        # env=dev with no credentials at all -> disabled, never silently open
        monkeypatch.setattr(api_main.settings, "api_key", "")
        monkeypatch.setattr(api_main.settings, "auth_password", "")
        r = client.get("/api/risk-guard/status")
        assert r.status_code == 503

    def test_all_read_protected_prefixes_gated(self, client):
        for path in ("/api/paper-trading/state", "/api/ledger/summary",
                     "/api/pipeline/status", "/api/automation/decisions"):
            assert client.get(path).status_code == 401, path

    def test_mutating_endpoints_gated_too(self, client):
        assert client.post("/api/automation/run").status_code == 401
        assert client.post("/api/automation/run", headers={"X-API-Key": "bad"}).status_code == 401

    def test_options_preflight_on_protected_path_not_rejected(self, client):
        # CORS must be the outermost middleware; otherwise a browser preflight
        # without X-API-Key is 401'd by auth before CORS answers it.
        r = client.options(
            "/api/paper-trading/order",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


class TestRateLimitIntegration:
    def test_mutating_burst_blocked_after_limit(self, client, monkeypatch):
        # keep the endpoint side-effect free during the burst
        from services.risk_guard import risk_guard as rg

        monkeypatch.setattr(rg, "update_equity", lambda equity: None)
        headers = {"X-API-Key": _TEST_KEY}
        statuses = [
            client.post("/api/risk-guard/equity", json={"equity": 1_000_000.0}, headers=headers).status_code
            for _ in range(api_main._RATE_LIMIT_MAX + 1)
        ]
        assert statuses[: api_main._RATE_LIMIT_MAX] == [200] * api_main._RATE_LIMIT_MAX
        assert statuses[-1] == 429

    def test_reads_are_not_rate_limited(self, client):
        headers = {"X-API-Key": _TEST_KEY}
        for _ in range(api_main._RATE_LIMIT_MAX + 5):
            r = client.get("/api/risk-guard/status", headers=headers)
            assert r.status_code == 200


class TestCacheMiddlewareIntegration:
    @pytest.fixture
    def cacheable_client(self, client, monkeypatch):
        def _err():
            from fastapi import Response

            return Response(content='{"detail":"boom"}', status_code=500,
                            media_type="application/json")

        # register the literal path BEFORE the {x} catch-all
        app.get("/api/__cachetest__/err")(_err)
        app.get("/api/__cachetest__/{x}")(lambda x: {"x": x})
        monkeypatch.setattr(
            api_main, "_CACHEABLE_PREFIXES", _ORIGINAL_CACHEABLE + ["/api/__cachetest__"]
        )
        monkeypatch.setattr(api_main, "_CACHE_TTL", 60.0)
        return client

    def test_miss_then_hit(self, cacheable_client):
        r1 = cacheable_client.get("/api/__cachetest__/A")
        assert r1.status_code == 200
        assert r1.headers.get("X-Cache") == "MISS"
        r2 = cacheable_client.get("/api/__cachetest__/A")
        assert r2.status_code == 200
        assert r2.headers.get("X-Cache") == "HIT"

    def test_distinct_query_strings_are_distinct_keys(self, cacheable_client):
        a = cacheable_client.get("/api/__cachetest__/A", params={"q": "1"})
        b = cacheable_client.get("/api/__cachetest__/A", params={"q": "2"})
        assert a.headers.get("X-Cache") == "MISS"
        assert b.headers.get("X-Cache") == "MISS"

    def test_error_responses_never_cached(self, cacheable_client):
        r1 = cacheable_client.get("/api/__cachetest__/err")
        r2 = cacheable_client.get("/api/__cachetest__/err")
        assert r1.status_code == 500
        assert r1.headers.get("X-Cache") is None
        assert r2.headers.get("X-Cache") is None

    def test_protected_paths_never_cached_even_if_cacheable(self, client, monkeypatch):
        # regression guard: a cached 200 under a protected prefix would
        # bypass the API key check entirely
        monkeypatch.setattr(
            api_main, "_CACHEABLE_PREFIXES", _ORIGINAL_CACHEABLE + ["/api/risk-guard"]
        )
        monkeypatch.setattr(api_main, "_CACHE_TTL", 60.0)
        headers = {"X-API-Key": _TEST_KEY}
        r1 = client.get("/api/risk-guard/status", headers=headers)
        r2 = client.get("/api/risk-guard/status", headers=headers)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.headers.get("X-Cache") is None
        assert r2.headers.get("X-Cache") is None
