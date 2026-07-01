"""Tests for src/multitimeframe.py — multi-timeframe analysis."""

import numpy as np
import pandas as pd
import pytest

from src.signals.multitimeframe import get_tf_signal, get_combined_signal


def _make_ohlcv(n=200, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    return df


class TestGetTfSignal:
    def test_returns_dict(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert isinstance(result, dict)

    def test_has_signal_key(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert "signal" in result

    def test_has_strength_key(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert "strength" in result

    def test_strength_is_numeric(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert isinstance(result["strength"], (int, float))

    def test_signal_valid_values(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert result["signal"] in ("Bullish", "Bearish", "Neutral")

    def test_small_df_returns_neutral(self):
        df = _make_ohlcv(5)
        result = get_tf_signal(df)
        assert result["signal"] == "Neutral"

    def test_empty_df_returns_neutral(self):
        result = get_tf_signal(pd.DataFrame())
        assert result["signal"] == "Neutral"

    def test_has_direction_key(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert "direction" in result
        assert result["direction"] in (-1, 0, 1)

    def test_has_indicators_key(self):
        df = _make_ohlcv(200)
        result = get_tf_signal(df)
        assert "indicators" in result


class TestGetCombinedSignal:
    def test_returns_dict(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert isinstance(result, dict)

    def test_has_signal_key(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert "signal" in result

    def test_has_direction_key(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert "direction" in result

    def test_has_confidence_key(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert "confidence" in result

    def test_signal_valid_values(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert result["signal"] in ("Bullish", "Bearish", "Neutral")

    def test_has_timeframes_key(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert "timeframes" in result

    def test_empty_data_returns_neutral(self):
        result = get_combined_signal({})
        assert result["signal"] == "Neutral"
        assert result["confidence"] == 0.0

    def test_multiple_timeframes(self):
        mtf_data = {
            "daily": _make_ohlcv(200),
            "1h": _make_ohlcv(200),
            "15m": _make_ohlcv(200),
        }
        result = get_combined_signal(mtf_data)
        assert len(result["timeframes"]) == 3

    def test_weighted_score_exists(self):
        mtf_data = {"daily": _make_ohlcv(200)}
        result = get_combined_signal(mtf_data)
        assert "weighted_score" in result
