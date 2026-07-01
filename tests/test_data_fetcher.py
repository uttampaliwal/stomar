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
