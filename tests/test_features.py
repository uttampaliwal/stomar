"""Tests for feature engineering."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFeatureEngineering:
    """Test technical indicator computation and feature generation."""

    def test_add_technical_indicators_preserves_length(self, sample_prices):
        from src.data.features import add_technical_indicators
        result = add_technical_indicators(sample_prices, ticker="TEST.NS")
        # Should not drastically reduce row count
        assert len(result) > len(sample_prices) * 0.8

    def test_add_technical_indicators_has_expected_columns(self, sample_prices):
        from src.data.features import add_technical_indicators
        result = add_technical_indicators(sample_prices, ticker="TEST.NS")
        expected = ["sma_10", "sma_20", "ema_12", "rsi", "macd", "bb_width", "atr"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_target_direction_is_binary(self, sample_prices):
        from src.data.features import add_technical_indicators
        result = add_technical_indicators(sample_prices, ticker="TEST.NS")
        if "target_direction" in result.columns:
            valid = result["target_direction"].dropna()
            unique_vals = set(valid.unique())
            assert unique_vals <= {0, 1}, f"target_direction has non-binary values: {unique_vals}"

    def test_no_infinite_values(self, sample_prices):
        from src.data.features import add_technical_indicators
        result = add_technical_indicators(sample_prices, ticker="TEST.NS")
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(result[numeric_cols].values).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in features"
