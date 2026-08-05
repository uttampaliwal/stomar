"""Tests for feature engineering."""
import numpy as np
import pandas as pd
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


class TestTripleBarrierLabels:
    """Vectorized triple-barrier labels must match the reference semantics."""

    def _naive(self, close, pp, lp, mh):
        n = len(close)
        labels = pd.Series(2, index=close.index, dtype=int)
        for i in range(n - 1):
            up = close.iloc[i] * (1 + pp)
            lo = close.iloc[i] * (1 - lp)
            for j in range(i + 1, min(i + mh + 1, n)):
                if close.iloc[j] >= up:
                    labels.iloc[i] = 1
                    break
                elif close.iloc[j] <= lo:
                    labels.iloc[i] = 0
                    break
        return labels

    def test_matches_naive_reference_random(self):
        from src.data.features import _triple_barrier_labels
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = int(rng.integers(1, 300))
            prices = pd.Series(np.round(rng.uniform(50, 3000, n), 2))
            pp = float(rng.uniform(0.001, 0.05))
            lp = float(rng.uniform(0.001, 0.05))
            mh = int(rng.integers(1, 15))
            got = _triple_barrier_labels(prices, pp, lp, mh)
            want = self._naive(prices, pp, lp, mh)
            assert got.equals(want)

    def test_up_then_down_hit_within_window(self):
        from src.data.features import _triple_barrier_labels
        prices = pd.Series([100, 103, 97, 96, 95, 104, 100])
        got = _triple_barrier_labels(prices, 0.02, 0.02, 5)
        # row 0: up at j=1 (103 >= 102), label 1
        # row 1: down at j=2 (97 <= 100.94), label 0
        assert got.iloc[0] == 1
        assert got.iloc[1] == 0

    def test_time_barrier_when_neither_hit(self):
        from src.data.features import _triple_barrier_labels
        prices = pd.Series([100, 100.5, 100.4, 100.3, 100.2, 100.1, 99])
        got = _triple_barrier_labels(prices, 0.02, 0.02, 5)
        assert got.iloc[0] == 2

    def test_down_barrier_hit_first(self):
        from src.data.features import _triple_barrier_labels
        prices = pd.Series([100, 99, 98, 102, 103, 104])
        got = _triple_barrier_labels(prices, 0.02, 0.02, 5)
        assert got.iloc[0] == 0  # 98 <= 98 (j=2) before 102 >= 102 (j=3)

    def test_last_bar_always_time_barrier(self):
        from src.data.features import _triple_barrier_labels
        prices = pd.Series([100, 103])
        got = _triple_barrier_labels(prices, 0.02, 0.02, 5)
        assert got.iloc[-1] == 2

    def test_small_series(self):
        from src.data.features import _triple_barrier_labels
        assert _triple_barrier_labels(pd.Series([100.0])).iloc[0] == 2
        assert _triple_barrier_labels(pd.Series([], dtype=float)).empty

    def test_preserves_index(self):
        from src.data.features import _triple_barrier_labels
        idx = pd.date_range("2025-01-01", periods=10, freq="D")
        prices = pd.Series(np.linspace(100, 120, 10), index=idx)
        got = _triple_barrier_labels(prices, 0.02, 0.02, 3)
        assert list(got.index) == list(idx)


class TestPrepareLstmDataDeprecation:
    def test_alias_warns_and_delegates(self, sample_prices):
        import warnings

        from src.data.features import (
            prepare_lstm_data,
            prepare_lstm_data_simple_split,
        )

        feature_cols = ["close", "volume"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            X, y, scaler = prepare_lstm_data(
                sample_prices, feature_cols, seq_length=5, train_ratio=0.7
            )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

        X2, y2, _ = prepare_lstm_data_simple_split(
            sample_prices, feature_cols, seq_length=5, train_ratio=0.7
        )
        assert X.shape == X2.shape
        assert y.shape == y2.shape
