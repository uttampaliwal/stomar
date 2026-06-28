"""Additional tests for backtester module to increase coverage."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_df(n=300):
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100000, 1000000, n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df


class TestWalkForwardSplit:
    def test_step_months_controls_gap(self):
        from src.backtester import walk_forward_split
        df = _make_df(600)
        splits = walk_forward_split(df, train_years=1, test_years=1, step_months=3)
        if len(splits) >= 2:
            gap = (splits[1]["train"][0] - splits[0]["train"][0]).days
            assert gap >= 60  # ~3 months

    def test_large_train_years_no_splits(self):
        from src.backtester import walk_forward_split
        df = _make_df(100)
        splits = walk_forward_split(df, train_years=5, test_years=2)
        assert len(splits) == 0

    def test_all_splits_have_train_and_test(self):
        from src.backtester import walk_forward_split
        df = _make_df(600)
        splits = walk_forward_split(df, train_years=1, test_years=1, step_months=6)
        for s in splits:
            assert "train" in s
            assert "test" in s
            assert len(s["train"]) > 0
            assert len(s["test"]) > 0


class TestComputeMetrics:
    def test_sharpe_ratio_range(self, sample_equity_curve):
        from src.backtester import compute_metrics
        metrics = compute_metrics(sample_equity_curve, [])
        assert -10 < metrics["sharpe_ratio"] < 10

    def test_max_drawdown_negative(self, sample_equity_curve):
        from src.backtester import compute_metrics
        metrics = compute_metrics(sample_equity_curve, [])
        assert metrics["max_drawdown"] <= 0

    def test_sortino_ratio(self, sample_equity_curve):
        from src.backtester import compute_metrics
        metrics = compute_metrics(sample_equity_curve, [])
        assert "sortino_ratio" in metrics

    def test_total_return(self, sample_equity_curve):
        from src.backtester import compute_metrics
        metrics = compute_metrics(sample_equity_curve, [])
        assert "total_return" in metrics

    def test_annualized_return(self, sample_equity_curve):
        from src.backtester import compute_metrics
        metrics = compute_metrics(sample_equity_curve, [])
        assert "annualized_return" in metrics

    def test_win_rate_with_trades(self):
        from src.backtester import compute_metrics
        eq = pd.DataFrame({"date": pd.bdate_range("2023-01-01", periods=10),
                           "equity": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107]})
        trades = [{"pnl": 10}, {"pnl": -5}, {"pnl": 15}, {"pnl": -2}]
        metrics = compute_metrics(eq, trades)
        assert metrics["win_rate"] == 50.0
        assert metrics["total_trades"] == 4

    def test_profit_factor(self):
        from src.backtester import compute_metrics
        eq = pd.DataFrame({"date": pd.bdate_range("2023-01-01", periods=5),
                           "equity": [100, 101, 102, 103, 104]})
        trades = [{"pnl": 100}, {"pnl": 100}, {"pnl": -50}]
        metrics = compute_metrics(eq, trades)
        assert metrics["profit_factor"] > 0

    def test_single_point_equity(self):
        from src.backtester import compute_metrics
        eq = pd.DataFrame({"date": [1], "equity": [100]})
        metrics = compute_metrics(eq, [])
        assert metrics == {}

    def test_zero_volatility(self):
        from src.backtester import compute_metrics
        eq = pd.DataFrame({"date": pd.bdate_range("2023-01-01", periods=5),
                           "equity": [100, 100, 100, 100, 100]})
        metrics = compute_metrics(eq, [])
        assert metrics["annualized_volatility"] == 0.0

    def test_custom_risk_free_rate(self, sample_equity_curve):
        from src.backtester import compute_metrics
        m1 = compute_metrics(sample_equity_curve, [], risk_free_rate=0.0)
        m2 = compute_metrics(sample_equity_curve, [], risk_free_rate=0.1)
        assert m1["sharpe_ratio"] != m2["sharpe_ratio"]
