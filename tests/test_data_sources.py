"""Tests for src/data_sources.py."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data_sources import (
    YFinanceSource,
    NSEArchiveSource,
    fetch_with_fallback,
)


def _make_df(n=100, start="2024-01-01"):
    """Create clean OHLCV DataFrame."""
    dates = pd.bdate_range(start, periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


# ── YFinanceSource ──

class TestYFinanceSource:
    @patch("src.data_sources.yf")
    def test_fetch_returns_df(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_df(50)
        mock_yf.Ticker.return_value = mock_ticker

        src = YFinanceSource()
        df = src.fetch("TEST.NS", period="1y")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    @patch("src.data_sources.yf")
    def test_fetch_empty_raises(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        src = YFinanceSource()
        with pytest.raises(ValueError, match="empty"):
            src.fetch("TEST.NS")

    def test_validate_good(self):
        src = YFinanceSource()
        assert src.validate(_make_df(50)) is True

    def test_validate_too_short(self):
        src = YFinanceSource()
        assert src.validate(_make_df(5)) is False

    def test_validate_missing_cols(self):
        src = YFinanceSource()
        df = pd.DataFrame({"close": [100, 101]})
        assert src.validate(df) is False


# ── NSEArchiveSource ──

class TestNSEArchiveSource:
    @patch("requests.Session")
    def test_fetch_network_error_raises(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Connection refused")
        mock_session_cls.return_value = mock_session

        src = NSEArchiveSource()
        with pytest.raises(ValueError, match="failed"):
            src.fetch("TEST.NS")

    def test_validate_good(self):
        src = NSEArchiveSource()
        assert src.validate(_make_df(50)) is True

    def test_validate_missing_cols(self):
        src = NSEArchiveSource()
        df = pd.DataFrame({"close": [100, 101]})
        assert src.validate(df) is False


# ── fetch_with_fallback ──

class TestFetchWithFallback:
    @patch("src.data_sources.yf")
    def test_primary_success(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_df(100)
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_with_fallback("TEST.NS", period="1y")
        assert "df" in result
        assert result["source"] == "YFinanceSource"
        assert result["validation"]["passed"]

    def test_all_sources_fail(self):
        class FailSource:
            def fetch(self, *a, **kw):
                raise ValueError("fail")
            def validate(self, df):
                return False

        with pytest.raises(ValueError, match="All data sources failed"):
            fetch_with_fallback("TEST.NS", sources=[FailSource()])

    @patch("src.data_sources.yf")
    def test_fallback_to_secondary(self, mock_yf):
        # Primary fails
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("yfinance down")
        mock_yf.Ticker.return_value = mock_ticker

        # Secondary succeeds
        mock_source = MagicMock()
        mock_source.fetch.return_value = _make_df(50)
        mock_source.validate.return_value = True
        mock_source.__class__.__name__ = "MockSource"

        result = fetch_with_fallback("TEST.NS", sources=[YFinanceSource(), mock_source])
        assert result["source"] == "MockSource"

    @patch("src.data_sources.yf")
    def test_returns_metadata(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_df(50)
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_with_fallback("TEST.NS")
        assert "timestamp" in result
        assert "validation" in result
        assert "source" in result

    @patch("src.data_sources.yf")
    def test_validation_catches_bad_data(self, mock_yf):
        # Primary returns bad data (fails validation)
        bad_df = pd.DataFrame({"close": [100] * 5})
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = bad_df
        mock_yf.Ticker.return_value = mock_ticker

        # Secondary returns good data
        mock_source = MagicMock()
        mock_source.fetch.return_value = _make_df(50)
        mock_source.validate.return_value = True
        mock_source.__class__.__name__ = "MockSource"

        result = fetch_with_fallback("TEST.NS", sources=[YFinanceSource(), mock_source])
        assert result["source"] == "MockSource"
