"""Browser session authentication for the StoMar API.

Replaces the frontend's reliance on the shared X-API-Key header: the API key
is still accepted for non-browser clients (CLI, scripts, curl), but browser
clients authenticate with a password and receive an opaque, HttpOnly,
server-side session cookie. The password never leaves the server; sessions
persist across restarts in ``data/sessions.json`` so a restart does not log
everyone out.

Security properties:

* passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib, 200k iterations,
  per-password random salt) — the plaintext password is never stored;
* session tokens are 256-bit CSPRNG values stored only server-side; the
  cookie carries the token, not the session data;
* the cookie is HttpOnly + SameSite=Lax and Secure in production, so
  JavaScript cannot read it and it is not sent cross-site;
* failed logins are rate-limited per IP in the API middleware;
* sessions expire after ``settings.session_ttl_hours`` (default 12h).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path

from src.core.constants import DATA_DIR
from src.core.secure_io import atomic_write_json
from src.core.settings import settings

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "stomar_session"
_PBKDF2_ITERATIONS = 200_000
_SESSION_FILE = Path(DATA_DIR) / "sessions.json"


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Return (salt_hex, hash_hex) for *password* (PBKDF2-HMAC-SHA256)."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Constant-time verification of *password* against stored values."""
    try:
        _, candidate = hash_password(password, salt_hex)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate, hash_hex)


def _session_path() -> Path:
    return _SESSION_FILE


class SessionStore:
    """File-backed session store. Opaque tokens → expiry timestamps."""

    def __init__(self, path: str | Path | None = None,
                 ttl_hours: float = 12.0):
        self._path = Path(path) if path is not None else _session_path()
        self._ttl_hours = ttl_hours
        self._lock = threading.RLock()
        self._sessions: dict[str, float] = {}
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._sessions = {
                    k: float(v) for k, v in data.items()
                }
        except (OSError, ValueError, TypeError) as exc:
            logger.error("failed to load sessions file %s: %s", self._path, exc)

    def _persist(self):
        try:
            atomic_write_json(self._path, self._sessions)
        except OSError as exc:
            logger.error("failed to persist sessions: %s", exc)

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        expires = time.time() + self._ttl_hours * 3600
        with self._lock:
            self._sessions[token] = expires
            self._persist()
        return token

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        now = time.time()
        with self._lock:
            expires = self._sessions.get(token)
            if expires is None:
                return False
            if expires <= now:
                del self._sessions[token]
                self._persist()
                return False
            return True

    def destroy(self, token: str | None):
        if not token:
            return
        with self._lock:
            if self._sessions.pop(token, None) is not None:
                self._persist()

    def prune(self) -> int:
        """Drop expired sessions; returns how many were removed."""
        now = time.time()
        with self._lock:
            expired = [t for t, exp in self._sessions.items() if exp <= now]
            for t in expired:
                del self._sessions[t]
            if expired:
                self._persist()
            return len(expired)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Module-level singleton: one store per process, shared by the auth router
# and the middleware. TTL comes from settings at import time.
session_store = SessionStore(ttl_hours=settings.session_ttl_hours)
