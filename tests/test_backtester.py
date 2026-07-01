"""Tests for backtester module — focus on no-data-leakage verification."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestWalkForwardSplit:
    """Test that walk_forward_split never leaks test data into training."""

    def test_no_overlap_between_train_and_test(self, sample_prices):
        from src.trading.backtester import walk_forward_split
        splits = walk_forward_split(sample_prices, train_years=1, test_years=1, step_months=3)
        for split in splits:
            train_idx = set(split["train"])
            test_idx = set(split["test"])
            assert len(train_idx & test_idx) == 0, "Train and test indices overlap!"

    def test_test_follows_train(self, sample_prices):
        from src.trading.backtester import walk_forward_split
        splits = walk_forward_split(sample_prices, train_years=1, test_years=1, step_months=3)
        for split in splits:
            max_train = max(split["train"])
            min_test = min(split["test"])
            assert max_train < min_test, "Test data starts before train data ends!"

    def test_chronological_order(self, sample_prices):
        from src.trading.backtester import walk_forward_split
        splits = walk_forward_split(sample_prices, train_years=1, test_years=1, step_months=3)
        for split in splits:
            assert list(split["train"]) == sorted(split["train"]), "Train indices not sorted"
            assert list(split["test"]) == sorted(split["test"]), "Test indices not sorted"

    def test_splits_are_generated(self):
        from src.trading.backtester import walk_forward_split
        np.random.seed(42)
        n = 600
        dates = pd.bdate_range("2021-01-01", periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({"close": close}, index=dates)
        splits = walk_forward_split(df, train_years=1, test_years=1, step_months=3)
        assert len(splits) > 0, "No splits generated"

    def test_short_data_returns_empty(self):
        from src.trading.backtester import walk_forward_split
        short_df = pd.DataFrame({"close": range(50)}, index=pd.bdate_range("2024-01-01", periods=50))
        splits = walk_forward_split(short_df, train_years=3, test_years=1, step_months=6)
        assert len(splits) == 0, "Should return empty for data shorter than train+test"


class TestComputeMetrics:
    """Test backtesting metrics computation."""

    def test_basic_metrics_computation(self, sample_equity_curve):
        from src.trading.backtester import compute_metrics
        metrics = compute_metrics(sample_equity_curve, [])
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_empty_equity_returns_empty(self):
        from src.trading.backtester import compute_metrics
        metrics = compute_metrics(pd.DataFrame(), [])
        assert metrics == {}

    def test_short_equity_returns_empty(self):
        from src.trading.backtester import compute_metrics
        df = pd.DataFrame({"date": [1], "equity": [100]})
        metrics = compute_metrics(df, [])
        assert metrics == {}

    def test_sharpe_uses_correct_risk_free_rate(self, sample_equity_curve):
        from src.trading.backtester import compute_metrics
        from src.core.constants import RISK_FREE_RATE
        metrics = compute_metrics(sample_equity_curve, [], risk_free_rate=RISK_FREE_RATE)
        assert metrics["sharpe_ratio"] is not None
