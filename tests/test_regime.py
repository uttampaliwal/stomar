"""Tests for src/regime.py — market regime detection."""

import numpy as np
import pandas as pd
import pytest

from src.regime import detect_regime


def _make_price_series(n=200, trend="up"):
    dates = pd.bdate_range("2024-01-01", periods=n)
    np.random.seed(42)
    if trend == "up":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    elif trend == "down":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 - 0.1)
    else:
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.Series(close, index=dates)


def _make_ohlc(n=200, trend="up"):
    prices = _make_price_series(n, trend)
    dates = prices.index
    np.random.seed(99)
    return pd.DataFrame({
        "open": prices.values - np.random.rand(n) * 0.5,
        "high": prices.values + np.random.rand(n) * 1.0,
        "low": prices.values - np.random.rand(n) * 1.0,
        "close": prices.values,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)


class TestDetectRegime:
    def test_returns_dict(self):
        prices = _make_price_series(200)
        result = detect_regime(prices)
        assert isinstance(result, dict)

    def test_has_regime_key(self):
        prices = _make_price_series(200)
        result = detect_regime(prices)
        assert "regime" in result

    def test_has_confidence_key(self):
        prices = _make_price_series(200)
        result = detect_regime(prices)
        assert "confidence" in result

    def test_valid_regime_values(self):
        prices = _make_price_series(200)
        result = detect_regime(prices)
        assert result["regime"] in ("Bull", "Bear", "Sideways", "Unknown")

    def test_confidence_in_range(self):
        prices = _make_price_series(200)
        result = detect_regime(prices)
        assert 0 <= result["confidence"] <= 100

    def test_has_indicators(self):
        prices = _make_price_series(200)
        result = detect_regime(prices)
        assert "indicators" in result

    def test_bullish_uptrend(self):
        prices = _make_price_series(200, trend="up")
        result = detect_regime(prices)
        assert result["regime"] in ("Bull", "Sideways")

    def test_bearish_downtrend(self):
        prices = _make_price_series(200, trend="down")
        result = detect_regime(prices)
        assert result["regime"] in ("Bear", "Sideways")

    def test_insufficient_data(self):
        prices = _make_price_series(10)
        result = detect_regime(prices)
        assert result["regime"] == "Unknown"
        assert result["confidence"] == 0

    def test_with_ohlc(self):
        ohlc = _make_ohlc(200)
        prices = ohlc["close"]
        result = detect_regime(prices, ohlc=ohlc)
        assert isinstance(result, dict)
        assert "regime" in result

    def test_lookback_parameter(self):
        prices = _make_price_series(200)
        result = detect_regime(prices, lookback=100)
        assert isinstance(result, dict)
