"""Login/logout/session endpoints for browser authentication.

``POST /api/auth/login`` exchanges the configured password
(``STOMAR_AUTH_PASSWORD``) for an HttpOnly session cookie. The password is
verified server-side only — it never reaches the browser bundle, unlike the
old ``VITE_STOMAR_API_KEY`` which was shipped to every client.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Body, Request, Response
from fastapi.responses import JSONResponse

from api.auth import SESSION_COOKIE_NAME, session_store
from src.core.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Login attempts are cheap for an attacker to brute-force; a strict per-IP
# limiter runs here (in addition to the API-wide mutation limiter).
_LOGIN_MAX_PER_WINDOW = 10
_LOGIN_WINDOW = 60.0
_login_attempts: dict[str, list[float]] = {}


def _login_rate_limited(request: Request) -> bool:
    key = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _login_attempts.setdefault(key, [])
    bucket[:] = [ts for ts in bucket if now - ts < _LOGIN_WINDOW]
    if len(bucket) >= _LOGIN_MAX_PER_WINDOW:
        return True
    bucket.append(now)
    return False


def _session_cookie_options(secure: bool) -> dict:
    return {
        "key": SESSION_COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure": secure,
        "path": "/",
        "max_age": int(settings.session_ttl_hours * 3600),
    }


@router.post("/login")
def login(request: Request, response: Response,
          body: dict = Body(default={})):
    if not settings.auth_password:
        return JSONResponse(
            status_code=503,
            content={"detail": "authentication not configured; set STOMAR_AUTH_PASSWORD"},
        )
    if _login_rate_limited(request):
        return JSONResponse(status_code=429, content={"detail": "too many login attempts"})

    password = str(body.get("password", ""))
    # settings.auth_password is the configured plaintext secret; compare in
    # constant time so timing does not leak password length/prefix.
    if password and hmac_equals(password, settings.auth_password):
        token = session_store.create()
        opts = _session_cookie_options(secure=settings.env == "production")
        response.set_cookie(value=token, **opts)
        logger.info("session created for client %s", request.client.host if request.client else "?")
        return {"ok": True}
    logger.warning("failed login attempt for client %s",
                   request.client.host if request.client else "?")
    return JSONResponse(status_code=401, content={"detail": "invalid password"})


def hmac_equals(a: str, b: str) -> bool:
    import hmac as _hmac
    return _hmac.compare_digest(a.encode(), b.encode())


@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        session_store.destroy(token)
    response.delete_cookie(
        SESSION_COOKIE_NAME, path="/",
        httponly=True, samesite="lax",
        secure=settings.env == "production",
    )
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return {"authenticated": session_store.validate(token)}
