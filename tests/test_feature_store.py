"""Tests for src/feature_store.py."""

import os

import pytest

from src import feature_store
from src.feature_store import (
    compute_feature_hash,
    register_feature_version,
    load_feature_version,
    load_latest_version,
    verify_feature_compatibility,
    list_versions,
)


@pytest.fixture(autouse=True)
def temp_feature_dir(tmp_path):
    """Redirect feature store to temp dir for tests."""
    original_dir = feature_store.FEATURE_VERSIONS_DIR
    feature_store.FEATURE_VERSIONS_DIR = str(tmp_path / "feature_versions")
    yield
    feature_store.FEATURE_VERSIONS_DIR = original_dir


# ── compute_feature_hash ──

class TestComputeFeatureHash:
    def test_deterministic(self):
        cols = ["rsi", "macd", "sma_20"]
        h1 = compute_feature_hash(cols)
        h2 = compute_feature_hash(cols)
        assert h1 == h2

    def test_order_independent(self):
        h1 = compute_feature_hash(["a", "b", "c"])
        h2 = compute_feature_hash(["c", "a", "b"])
        assert h1 == h2

    def test_different_for_different_cols(self):
        h1 = compute_feature_hash(["a", "b"])
        h2 = compute_feature_hash(["a", "c"])
        assert h1 != h2

    def test_12_char_hex(self):
        h = compute_feature_hash(["x", "y"])
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_with_transformations(self):
        h1 = compute_feature_hash(["a", "b"], {"a": "log"})
        h2 = compute_feature_hash(["a", "b"], {"a": "sqrt"})
        assert h1 != h2


# ── register_feature_version ──

class TestRegisterFeatureVersion:
    def test_creates_file(self):
        h = compute_feature_hash(["rsi", "macd"])
        record = register_feature_version(h, ["rsi", "macd"], description="test")
        assert record["hash"] == h
        assert record["n_features"] == 2
        assert os.path.exists(os.path.join(feature_store.FEATURE_VERSIONS_DIR, f"{h}.json"))

    def test_creates_latest(self):
        h = compute_feature_hash(["rsi"])
        register_feature_version(h, ["rsi"])
        assert os.path.exists(os.path.join(feature_store.FEATURE_VERSIONS_DIR, "latest.json"))

    def test_load_roundtrip(self):
        cols = ["rsi", "macd", "sma_20"]
        h = compute_feature_hash(cols)
        register_feature_version(h, cols, description="roundtrip test")
        loaded = load_feature_version(h)
        assert loaded["hash"] == h
        assert loaded["columns"] == sorted(cols)
        assert loaded["description"] == "roundtrip test"


# ── load_feature_version ──

class TestLoadFeatureVersion:
    def test_load_existing(self):
        h = compute_feature_hash(["a"])
        register_feature_version(h, ["a"])
        record = load_feature_version(h)
        assert record["hash"] == h

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_feature_version("nonexistent12")


# ── load_latest_version ──

class TestLoadLatestVersion:
    def test_load_latest(self):
        h1 = compute_feature_hash(["a", "b"])
        register_feature_version(h1, ["a", "b"])
        h2 = compute_feature_hash(["x", "y"])
        register_feature_version(h2, ["x", "y"])
        latest = load_latest_version()
        assert latest["hash"] == h2

    def test_no_versions_raises(self):
        with pytest.raises(FileNotFoundError, match="No feature versions"):
            load_latest_version()


# ── verify_feature_compatibility ──

class TestVerifyFeatureCompatibility:
    def test_compatible(self):
        cols = ["rsi", "macd"]
        h = compute_feature_hash(cols)
        register_feature_version(h, cols)
        result = verify_feature_compatibility(h, h)
        assert result["compatible"] is True

    def test_incompatible_different_hash(self):
        h1 = compute_feature_hash(["rsi", "macd"])
        h2 = compute_feature_hash(["rsi", "sma_20"])
        register_feature_version(h1, ["rsi", "macd"])
        register_feature_version(h2, ["rsi", "sma_20"])
        result = verify_feature_compatibility(h1, h2)
        assert result["compatible"] is False
        assert "missing_in_current" in result
        assert "extra_in_current" in result

    def test_model_hash_not_found(self):
        h = compute_feature_hash(["a"])
        register_feature_version(h, ["a"])
        result = verify_feature_compatibility("nonexistent123", h)
        assert result["compatible"] is False
        assert "error" in result


# ── list_versions ──

class TestListVersions:
    def test_empty(self):
        assert list_versions() == []

    def test_lists_all(self):
        h1 = compute_feature_hash(["a"])
        h2 = compute_feature_hash(["b"])
        register_feature_version(h1, ["a"])
        register_feature_version(h2, ["b"])
        versions = list_versions()
        assert len(versions) == 2
