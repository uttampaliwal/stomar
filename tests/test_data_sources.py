"""Tests for src/data_sources.py."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data_sources import (
    DataSource,
    YFinanceSource,
    NSEArchiveSource,
    fetch_with_fallback,
)


def _make_df(n=100, start="2024-01-01"):
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


# --- DataSource base class ---

def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()


def test_datasource_validate_true_for_valid_df():
    ds = YFinanceSource()
    df = _make_df(20)
    assert ds.validate(df) is True


def test_datasource_validate_false_for_none():
    ds = YFinanceSource()
    assert ds.validate(None) is False


def test_datasource_validate_false_for_empty():
    ds = YFinanceSource()
    assert ds.validate(pd.DataFrame()) is False


def test_datasource_validate_false_for_short_df():
    ds = YFinanceSource()
    df = _make_df(5)
    assert ds.validate(df) is False


# --- YFinanceSource ---

def test_yfinance_validate_requires_ohlcv_columns():
    ds = YFinanceSource()
    df = _make_df(20)
    df = df.drop(columns=["volume"])
    assert ds.validate(df) is False


def test_yfinance_fetch_calls_ticker():
    ds = YFinanceSource()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _make_df(50)
    with patch("src.data_sources.yf.Ticker", return_value=mock_ticker) as mock_cls:
        result = ds.fetch("RELIANCE.NS", period="1y")
        mock_cls.assert_called_once_with("RELIANCE.NS")
        mock_ticker.history.assert_called_once_with(period="1y", interval="1d")
        assert len(result) == 50


def test_yfinance_fetch_raises_on_empty():
    ds = YFinanceSource()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("src.data_sources.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError, match="empty data"):
            ds.fetch("BAD.NS")


def test_yfinance_fetch_lowercases_columns():
    ds = YFinanceSource()
    mock_ticker = MagicMock()
    df = _make_df(20)
    df.columns = [c.upper() for c in df.columns]
    mock_ticker.history.return_value = df
    with patch("src.data_sources.yf.Ticker", return_value=mock_ticker):
        result = ds.fetch("TEST.NS")
        assert all(c.islower() for c in result.columns)


# --- NSEArchiveSource ---

def test_nse_archive_validate_true_for_valid_df():
    ds = NSEArchiveSource()
    df = _make_df(20)
    assert ds.validate(df) is True


def test_nse_archive_validate_false_for_missing_columns():
    ds = NSEArchiveSource()
    df = _make_df(20).drop(columns=["open", "high"])
    assert ds.validate(df) is False


# --- fetch_with_fallback ---

def test_fetch_with_fallback_uses_primary():
    mock_source = MagicMock(spec=DataSource)
    mock_source.validate.return_value = True
    mock_source.fetch.return_value = _make_df(50)
    mock_source.__class__.__name__ = "MockSource"

    result = fetch_with_fallback("TEST.NS", sources=[mock_source])
    assert "df" in result
    assert result["source"] == "MockSource"
    assert "validation" in result
    assert "timestamp" in result


def test_fetch_with_fallback_falls_back_on_validation_failure():
    primary = MagicMock(spec=DataSource)
    primary.validate.return_value = False
    primary.__class__.__name__ = "Primary"

    fallback = MagicMock(spec=DataSource)
    fallback.validate.return_value = True
    fallback.fetch.return_value = _make_df(50)
    fallback.__class__.__name__ = "Fallback"

    result = fetch_with_fallback("TEST.NS", sources=[primary, fallback])
    assert result["source"] == "Fallback"


def test_fetch_with_fallback_raises_on_all_failure():
    bad_source = MagicMock(spec=DataSource)
    bad_source.fetch.side_effect = ValueError("fail")
    bad_source.__class__.__name__ = "Bad"

    with pytest.raises(ValueError, match="All data sources failed"):
        fetch_with_fallback("TEST.NS", sources=[bad_source])


def test_fetch_with_fallback_returns_validation_info():
    mock_source = MagicMock(spec=DataSource)
    mock_source.validate.return_value = True
    mock_source.fetch.return_value = _make_df(50)
    mock_source.__class__.__name__ = "Mock"

    result = fetch_with_fallback("TEST.NS", sources=[mock_source])
    assert isinstance(result["validation"], dict)
    assert "date_range" in result["validation"]
