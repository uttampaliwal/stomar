"""Tests for data_fetcher module."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMarketStatus:
    def test_market_status_returns_valid_string(self):
        from src.data.data_fetcher import get_market_status
        status = get_market_status()
        assert status in ("Open", "Closed", "Closed (Weekend)")

    def test_market_status_is_string(self):
        from src.data.data_fetcher import get_market_status
        status = get_market_status()
        assert isinstance(status, str)


class TestFetchStockData:
    def test_fetch_returns_dataframe(self):
        import pandas as pd
        from src.data.data_fetcher import fetch_stock_data
        df = fetch_stock_data("RELIANCE.NS", period="1y")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_has_expected_columns(self):
        from src.data.data_fetcher import fetch_stock_data
        df = fetch_stock_data("RELIANCE.NS", period="1y")
        expected = ["open", "high", "low", "close", "volume"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_fetch_invalid_ticker_raises(self):
        from src.data.data_fetcher import fetch_stock_data
        with pytest.raises(ValueError):
            fetch_stock_data("INVALID_TICKER_XYZ.NS", period="1y", force_refresh=True)


class TestTickerFetchLock:
    """Concurrent fetch_stock_data calls for one ticker must be serialized."""

    def test_same_ticker_calls_are_serialized(self, monkeypatch):
        import threading
        import time
        import src.data.data_fetcher as dfm

        calls = []
        state = {"active": 0, "max_active": 0}
        state_lock = threading.Lock()

        def slow_fetch(*args, **kwargs):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.1)
            calls.append(args[0])
            with state_lock:
                state["active"] -= 1
            return None

        monkeypatch.setattr(dfm, "_fetch_stock_data_locked", slow_fetch)

        threads = [
            threading.Thread(target=dfm.fetch_stock_data, args=("LOCK.NS",), kwargs={"force_refresh": True})
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 3
        assert state["max_active"] == 1

    def test_different_tickers_run_in_parallel(self, monkeypatch):
        import threading
        import time
        import src.data.data_fetcher as dfm

        state = {"active": 0, "max_active": 0}
        state_lock = threading.Lock()

        def slow_fetch(*args, **kwargs):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.15)
            with state_lock:
                state["active"] -= 1
            return None

        monkeypatch.setattr(dfm, "_fetch_stock_data_locked", slow_fetch)

        threads = [
            threading.Thread(target=dfm.fetch_stock_data, args=(f"T{i}.NS",), kwargs={"force_refresh": True})
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["max_active"] == 3
