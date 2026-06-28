"""Shared test fixtures for StoMar test suite."""
import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_prices():
    """Generate sample OHLCV price data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100000, 1000000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def sample_returns():
    """Generate sample daily returns array."""
    np.random.seed(42)
    return np.random.randn(252) * 0.015  # ~1.5% daily vol


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve DataFrame."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n)
    equity = 100000 * np.cumprod(1 + np.random.randn(n) * 0.01)
    return pd.DataFrame({"date": dates, "equity": equity})


@pytest.fixture
def sample_feature_df():
    """Generate sample DataFrame with required features for ensemble testing."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    feature_cols = [
        "close", "volume", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "rsi", "macd", "macd_signal",
        "bb_width", "atr", "obv", "volume_ratio",
    ]
    data = {}
    for col in feature_cols:
        if col == "close":
            data[col] = close
        elif col == "volume":
            data[col] = np.random.randint(100000, 1000000, n)
        else:
            data[col] = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df
