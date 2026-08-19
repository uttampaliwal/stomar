"""Unit tests for password hashing and the file-backed SessionStore.

No ``api.main`` import here, so this file runs even while the dev server
holds the market data store lock.
"""

import time

from api.auth import (
    SessionStore,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_round_trip(self):
        salt, digest = hash_password("s3cret")
        assert verify_password("s3cret", salt, digest)

    def test_wrong_password_rejected(self):
        salt, digest = hash_password("s3cret")
        assert not verify_password("nope", salt, digest)

    def test_random_salts(self):
        _, d1 = hash_password("same")
        _, d2 = hash_password("same")
        assert d1 != d2

    def test_corrupt_storage_rejected(self):
        assert not verify_password("s3cret", "zz", "ab" * 32)
        assert not verify_password("s3cret", "00", "")


class TestSessionStore:
    def test_create_and_validate(self, tmp_path):
        s = SessionStore(path=tmp_path / "s.json", ttl_hours=1.0)
        token = s.create()
        assert s.validate(token)
        assert not s.validate("bogus")
        assert not s.validate(None)

    def test_destroy(self, tmp_path):
        s = SessionStore(path=tmp_path / "s.json", ttl_hours=1.0)
        token = s.create()
        s.destroy(token)
        assert not s.validate(token)

    def test_ttl_expiry(self, tmp_path):
        s = SessionStore(path=tmp_path / "s.json", ttl_hours=0.0)
        token = s.create()
        time.sleep(0.01)
        assert not s.validate(token)

    def test_persists_across_restart(self, tmp_path):
        path = tmp_path / "s.json"
        s1 = SessionStore(path=path, ttl_hours=1.0)
        token = s1.create()
        s2 = SessionStore(path=path, ttl_hours=1.0)
        assert s2.validate(token)

    def test_prune(self, tmp_path):
        s = SessionStore(path=tmp_path / "s.json", ttl_hours=0.0)
        s.create()
        s.create()
        assert s.count() == 2
        assert s.prune() == 2
        assert s.count() == 0
