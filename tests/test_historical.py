"""Tests for multi-threaded historical ingestion."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.historical import (
    backfill_universe,
    fetch_ticker_with_fallback,
    fetch_universe,
)
from src.data.resilience import RateLimiter, throttle


def _make_df(n=50, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=n)
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=dates,
    )


def _yf_ok(ticker, period, interval):
    df = _make_df()
    df.index.name = "date"
    return df


def _yf_fail(ticker, period, interval):
    raise RuntimeError("yf down")


def _nse_ok(ticker, period, interval):
    df = _make_df(n=30)
    df.index.name = "date"
    return df


class TestFetchWithFallback:
    def test_primary_source_success(self):
        result = fetch_ticker_with_fallback("RELIANCE.NS", sources=[_yf_ok])
        assert result["error"] is None
        assert result["rows"] == 50
        assert result["source"] == "_yf_ok"

    def test_falls_back_when_primary_fails(self):
        result = fetch_ticker_with_fallback("RELIANCE.NS", sources=[_yf_fail, _nse_ok])
        assert result["error"] is None
        assert result["rows"] == 30
        assert result["source"] == "_nse_ok"

    def test_all_sources_fail_reports_error(self):
        result = fetch_ticker_with_fallback("RELIANCE.NS", sources=[_yf_fail])
        assert result["df"] is None
        assert result["error"] is not None
        assert "yf down" in result["error"]

    def test_deduplicates_index(self):
        def _dup(ticker, period, interval):
            df = _make_df(n=10)
            df.index.name = "date"
            return pd.concat([df, df.iloc[[9]]])  # duplicate last row

        result = fetch_ticker_with_fallback("X.NS", sources=[_dup])
        assert result["rows"] == 10


class TestFetchUniverse:
    def test_parallel_fetch_returns_all_tickers(self, monkeypatch):
        import src.data.historical as historical

        calls = []

        def fake_fetch(ticker, period, interval):
            calls.append(ticker)
            return {
                "df": _make_df(), "source": "_fake",
                "ticker": ticker, "rows": 50, "error": None,
            }

        monkeypatch.setattr(historical, "fetch_ticker_with_fallback", fake_fetch)
        results = fetch_universe(
            ["RELIANCE", "TCS", "INFY"], period="1y", workers=3, as_yf=True
        )
        assert set(results) == {"RELIANCE.NS", "TCS.NS", "INFY.NS"}
        assert all(r["error"] is None for r in results.values())
        assert sorted(calls) == ["INFY.NS", "RELIANCE.NS", "TCS.NS"]

    def test_universe_name_expands(self, monkeypatch):
        import src.data.historical as historical

        monkeypatch.setattr(
            historical, "fetch_ticker_with_fallback",
            lambda ticker, period, interval: {
                "df": _make_df(), "source": "_fake",
                "ticker": ticker, "rows": 50, "error": None,
            },
        )
        results = fetch_universe("nifty_50", workers=4)
        assert len(results) == 50
        assert all(t.endswith(".NS") for t in results)


class TestBackfill:
    def test_backfill_persists_to_store(self, tmp_path, monkeypatch):
        from src.data.store import MarketDataStore

        import src.data.historical as historical

        monkeypatch.setattr(
            historical, "fetch_ticker_with_fallback",
            lambda ticker, period, interval: {
                "df": _make_df(), "source": "_yf_ok",
                "ticker": ticker, "rows": 50, "error": None,
            },
        )
        store = MarketDataStore(str(tmp_path / "backfill.db"))
        summary = backfill_universe(
            ["RELIANCE", "TCS"], workers=2, store=store, save_parquet=False
        )
        assert summary["total"] == 2
        assert summary["ok"] == 2
        assert summary["failed"] == 0
        assert summary["rows_upserted"] == 100
        assert set(store.available_symbols()) == {"RELIANCE.NS", "TCS.NS"}
        store.close()

    def test_backfill_tracks_failures(self, tmp_path, monkeypatch):
        from src.data.store import MarketDataStore

        import src.data.historical as historical

        monkeypatch.setattr(
            historical, "fetch_ticker_with_fallback",
            lambda ticker, period, interval: {
                "df": None, "source": None, "ticker": ticker,
                "rows": 0, "error": "nope",
            },
        )
        store = MarketDataStore(str(tmp_path / "backfill2.db"))
        summary = backfill_universe(["BROKEN.NS"], workers=1, store=store, save_parquet=False)
        assert summary["failed"] == 1
        assert summary["failures"]["BROKEN.NS"] == "nope"
        store.close()


class TestRateLimiter:
    def test_rate_limiter_paces_calls(self):
        limiter = RateLimiter(rate=20.0, burst=5)
        start = time.monotonic()
        for _ in range(5):
            with limiter.acquire():
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # burst allows first 5 instantly

    def test_rate_limiter_throttles_after_burst(self):
        limiter = RateLimiter(rate=50.0, burst=2)
        for _ in range(2):
            with limiter.acquire():
                pass
        start = time.monotonic()
        with limiter.acquire():
            pass
        elapsed = time.monotonic() - start
        assert elapsed >= 1.0 / 50.0

    def test_rate_limiter_rejects_bad_rate(self):
        with pytest.raises(ValueError):
            RateLimiter(rate=0)

    def test_throttle_decorator(self):
        @throttle(rate=100.0, burst=10, key="test")
        def fast_fn():
            return 42

        assert fast_fn() == 42
