"""Integration tests for browser session authentication (finding #3).

Drives the real middleware stack through FastAPI's TestClient: login →
cookie → protected access → logout, plus API-key back-compat and the
fail-closed production guard.

Note: like the other api tests, importing ``api.main`` requires the market
data store, which is single-writer — stop the dev server before running
this file.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.routers.auth as api_auth_router
from api.auth import SESSION_COOKIE_NAME, SessionStore
from api.main import app

_TEST_PASSWORD = "correct horse battery staple"
_TEST_KEY = "test-secret-key"


@pytest.fixture(autouse=True)
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_key", "")
    monkeypatch.setattr(api_main.settings, "auth_password", _TEST_PASSWORD)
    store = SessionStore(path=tmp_path / "sessions.json", ttl_hours=12.0)
    monkeypatch.setattr(api_main, "session_store", store)
    monkeypatch.setattr(api_auth_router, "session_store", store)
    api_auth_router._login_attempts.clear()
    api_main._rate_buckets.clear()
    with TestClient(app) as c:
        yield c
    api_main._rate_buckets.clear()
    api_auth_router._login_attempts.clear()


class TestLoginFlow:
    def test_login_requires_password_configured(self, client, monkeypatch):
        monkeypatch.setattr(api_main.settings, "auth_password", "")
        r = client.post("/api/auth/login", json={"password": "x"})
        assert r.status_code == 503

    def test_wrong_password_401(self, client):
        r = client.post("/api/auth/login", json={"password": "wrong"})
        assert r.status_code == 401

    def test_empty_password_401(self, client):
        r = client.post("/api/auth/login", json={"password": ""})
        assert r.status_code == 401

    def test_login_sets_httponly_cookie(self, client):
        r = client.post("/api/auth/login", json={"password": _TEST_PASSWORD})
        assert r.status_code == 200
        assert r.cookies.get(SESSION_COOKIE_NAME)

    def test_me_anonymous_then_authenticated(self, client):
        assert client.get("/api/auth/me").json() == {"authenticated": False}
        client.post("/api/auth/login", json={"password": _TEST_PASSWORD})
        assert client.get("/api/auth/me").json() == {"authenticated": True}

    def test_session_grants_protected_access(self, client):
        client.post("/api/auth/login", json={"password": _TEST_PASSWORD})
        r = client.get("/api/risk-guard/status")
        assert r.status_code == 200

    def test_logout_invalidates_session(self, client):
        client.post("/api/auth/login", json={"password": _TEST_PASSWORD})
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").json() == {"authenticated": False}
        assert client.get("/api/risk-guard/status").status_code == 401

    def test_no_session_no_key_still_fails_closed(self, client):
        r = client.get("/api/risk-guard/status")
        assert r.status_code == 401

    def test_api_key_still_accepted_with_sessions_enabled(self, client, monkeypatch):
        monkeypatch.setattr(api_main.settings, "api_key", _TEST_KEY)
        r = client.get("/api/risk-guard/status", headers={"X-API-Key": _TEST_KEY})
        assert r.status_code == 200

    def test_login_rate_limited(self, client):
        for _ in range(10):
            r = client.post("/api/auth/login", json={"password": "wrong"})
            assert r.status_code == 401
        r = client.post("/api/auth/login", json={"password": _TEST_PASSWORD})
        assert r.status_code == 429


class TestFailClosedProduction:
    def test_production_requires_some_credential(self, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(api_main.settings, "env", "production")
            m.setattr(api_main.settings, "api_key", "")
            m.setattr(api_main.settings, "auth_password", "")
            with pytest.raises(RuntimeError, match="refusing to start"):
                importlib.reload(api_main)

    def test_production_with_password_boots(self, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(api_main.settings, "env", "production")
            m.setattr(api_main.settings, "api_key", "")
            m.setattr(api_main.settings, "auth_password", "pw")
            importlib.reload(api_main)
