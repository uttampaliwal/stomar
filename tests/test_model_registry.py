"""Tests for src/model_registry.py."""

import json
import os

import pytest

from src.model_registry import ModelRegistry, ModelVersion


@pytest.fixture
def registry(tmp_path):
    """Create a temp registry for tests."""
    return ModelRegistry(registry_dir=str(tmp_path / "registry"))


# ── ModelVersion ──

class TestModelVersion:
    def test_defaults(self):
        v = ModelVersion(ticker="T", version=1, model_path="m.pt", feature_hash="h1")
        assert v.status == "staging"
        assert v.created_at
        assert v.promoted_at is None
        assert v.metrics == {}


# ── ModelRegistry.register ──

class TestRegister:
    def test_register_first(self, registry):
        v = registry.register("TEST.NS", "/path/model.pt", "abc123")
        assert v.version == 1
        assert v.status == "staging"
        assert v.ticker == "TEST.NS"

    def test_register_incremental(self, registry):
        registry.register("TEST.NS", "/m1.pt", "h1")
        v2 = registry.register("TEST.NS", "/m2.pt", "h2")
        assert v2.version == 2

    def test_register_with_metrics(self, registry):
        v = registry.register("TEST.NS", "/m.pt", "h1", metrics={"acc": 0.53})
        assert v.metrics["acc"] == 0.53

    def test_persists_to_disk(self, registry):
        registry.register("TEST.NS", "/m.pt", "h1")
        path = registry._registry_path("TEST.NS")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1


# ── ModelRegistry.promote ──

class TestPromote:
    def test_promote_staging(self, registry):
        registry.register("TEST.NS", "/m.pt", "h1")
        v = registry.promote("TEST.NS", 1)
        assert v.status == "production"
        assert v.promoted_at is not None

    def test_promote_archives_old(self, registry):
        registry.register("TEST.NS", "/m1.pt", "h1")
        registry.promote("TEST.NS", 1)
        registry.register("TEST.NS", "/m2.pt", "h2")
        registry.promote("TEST.NS", 2)

        versions = registry.list_versions("TEST.NS")
        prod = [v for v in versions if v.status == "production"]
        archived = [v for v in versions if v.status == "archived"]
        assert len(prod) == 1
        assert prod[0].version == 2
        assert len(archived) == 1
        assert archived[0].version == 1

    def test_promote_non_staging_fails(self, registry):
        registry.register("TEST.NS", "/m.pt", "h1")
        registry.promote("TEST.NS", 1)
        with pytest.raises(ValueError, match="not 'staging'"):
            registry.promote("TEST.NS", 1)

    def test_promote_nonexistent_fails(self, registry):
        with pytest.raises(ValueError, match="not found"):
            registry.promote("TEST.NS", 99)


# ── ModelRegistry.get_production ──

class TestGetProduction:
    def test_returns_production(self, registry):
        registry.register("TEST.NS", "/m.pt", "h1")
        registry.promote("TEST.NS", 1)
        prod = registry.get_production("TEST.NS")
        assert prod is not None
        assert prod.version == 1
        assert prod.status == "production"

    def test_no_production(self, registry):
        registry.register("TEST.NS", "/m.pt", "h1")
        prod = registry.get_production("TEST.NS")
        assert prod is None


# ── ModelRegistry.get_version ──

class TestGetVersion:
    def test_get_existing(self, registry):
        registry.register("TEST.NS", "/m.pt", "h1")
        v = registry.get_version("TEST.NS", 1)
        assert v is not None
        assert v.version == 1

    def test_get_nonexistent(self, registry):
        v = registry.get_version("TEST.NS", 99)
        assert v is None


# ── ModelRegistry.list_versions ──

class TestListVersions:
    def test_empty(self, registry):
        assert registry.list_versions("TEST.NS") == []

    def test_lists_all(self, registry):
        registry.register("TEST.NS", "/m1.pt", "h1")
        registry.register("TEST.NS", "/m2.pt", "h2")
        registry.register("TEST.NS", "/m3.pt", "h3")
        versions = registry.list_versions("TEST.NS")
        assert len(versions) == 3


# ── ModelRegistry.rollback ──

class TestRollback:
    def test_rollback_to_archived(self, registry):
        registry.register("TEST.NS", "/m1.pt", "h1")
        registry.promote("TEST.NS", 1)
        registry.register("TEST.NS", "/m2.pt", "h2")
        registry.promote("TEST.NS", 2)

        rolled = registry.rollback("TEST.NS", 1)
        assert rolled.version == 1
        assert rolled.status == "production"

        prod = registry.get_production("TEST.NS")
        assert prod.version == 1

    def test_rollback_nonexistent_fails(self, registry):
        with pytest.raises(ValueError, match="not found"):
            registry.rollback("TEST.NS", 99)


# ── ModelRegistry.get_summary ──

class TestGetSummary:
    def test_empty_summary(self, registry):
        summary = registry.get_summary()
        assert summary == {}

    def test_summary_with_versions(self, registry):
        registry.register("TEST.NS", "/m1.pt", "h1")
        registry.register("TEST.NS", "/m2.pt", "h2")
        registry.promote("TEST.NS", 1)
        summary = registry.get_summary()
        assert "TEST.NS" in summary
        assert summary["TEST.NS"]["total_versions"] == 2
        assert summary["TEST.NS"]["production"] == 1
