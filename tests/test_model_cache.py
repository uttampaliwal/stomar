"""Tests for the model cache memory management in src/models/model.py.

Each cached model set is 200-500 MB; the cache must evict by memory budget
(oldest first), not just by entry count.
"""

import sys
import time

import pytest
import torch

from src.models import model as m


class _FakeBooster:
    def __init__(self, raw: bytes):
        self._raw = raw

    def save_raw(self):
        return self._raw


class _FakeXgb:
    """Mimics xgboost's sklearn wrapper surface used by the estimator."""

    def get_booster(self):
        return _FakeBooster(b"x" * 4096)


class _FakeLgbBooster:
    def model_to_string(self):
        return "y" * 8192


class _FakeLgb:
    def __init__(self):
        self.booster_ = _FakeLgbBooster()


class TestEstimateModelBytes:
    def test_torch_module_uses_parameter_bytes(self):
        net = torch.nn.Linear(100, 100)
        expected = sum(p.numel() * p.element_size() for p in net.parameters())
        assert m._estimate_model_bytes(net) == expected

    def test_xgboost_uses_booster_dump(self):
        assert m._estimate_model_bytes(_FakeXgb()) == 4096

    def test_lightgbm_uses_model_to_string(self):
        assert m._estimate_model_bytes(_FakeLgb()) == 8192

    def test_fallback_is_getsizeof(self):
        obj = object()
        assert m._estimate_model_bytes(obj) == sys.getsizeof(obj)


class TestEviction:
    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        m._model_cache.clear()
        yield
        m._model_cache.clear()

    def _fill(self, sizes, now):
        for i, size in enumerate(sizes):
            m._model_cache[f"k{i}"] = (now + i, size, ())

    def test_evicts_oldest_until_under_budget(self, monkeypatch):
        monkeypatch.setattr(m, "_model_cache_budget_bytes", lambda: 10_000)
        now = time.time()
        self._fill([4_000, 4_000, 4_000], now)
        m._evict_model_cache(now, now - 1)  # nothing stale
        total = sum(s for _ts, s, _mdl in m._model_cache.values())
        assert total <= 10_000
        assert "k0" not in m._model_cache  # oldest evicted first
        assert set(m._model_cache) == {"k1", "k2"}

    def test_evicts_oldest_until_under_count_cap(self, monkeypatch):
        monkeypatch.setattr(m, "_model_cache_budget_bytes", lambda: 1 << 30)
        monkeypatch.setattr(m, "_MODEL_CACHE_MAX", 2)
        now = time.time()
        self._fill([1, 1, 1, 1], now)
        m._evict_model_cache(now, now - 1)
        assert len(m._model_cache) == 2
        assert set(m._model_cache) == {"k2", "k3"}

    def test_stale_entries_evicted_regardless_of_budget(self):
        now = time.time()
        m._model_cache["fresh"] = (now, 1, ())
        m._model_cache["stale"] = (now - 7200, 1, ())
        m._evict_model_cache(now, now - 3600)
        assert set(m._model_cache) == {"fresh"}
