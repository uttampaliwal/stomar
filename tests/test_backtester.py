"""Tests for backtester module — focus on no-data-leakage verification."""
import numpy as np
import pandas as pd
import pytest
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


class TestSimulatedMetrics:
    """Pipeline evaluate-gate key mapping (simulated_sharpe / simulated_max_drawdown)."""

    def test_maps_ratio_and_drawdown_fraction(self):
        from src.trading.backtester import _attach_simulated_metrics
        metrics = _attach_simulated_metrics(
            {}, {"sharpe_ratio": 1.234, "max_drawdown": -12.34}
        )
        assert metrics["simulated_sharpe"] == 1.234
        assert metrics["simulated_max_drawdown"] == pytest.approx(0.1234)

    def test_ignores_missing_keys(self):
        from src.trading.backtester import _attach_simulated_metrics
        metrics = _attach_simulated_metrics({}, {"total_return": 0.5})
        assert "simulated_sharpe" not in metrics
        assert "simulated_max_drawdown" not in metrics

    def test_preserves_existing_metrics(self):
        from src.trading.backtester import _attach_simulated_metrics
        metrics = _attach_simulated_metrics(
            {"ensemble_accuracy": 0.55}, {"sharpe_ratio": 0.9, "max_drawdown": -5.0}
        )
        assert metrics["ensemble_accuracy"] == 0.55
        assert metrics["simulated_max_drawdown"] == pytest.approx(0.05)


# ── next-open execution semantics ─────────────────────────────────────────

def _sig(direction, close, open_):
    return {"direction": direction, "confidence": 0.9,
            "price": close, "open": open_, "actual": direction}


def test_simple_backtest_buys_at_next_open_not_signal_close():
    """Signals are known by the prior close; fills must reference this
    bar's OPEN, not the close that decided the label."""
    from src.trading.backtester import run_simple_backtest

    signals = {"TEST.NS": {
        "2026-01-05": _sig(1, close=110.0, open_=100.0),  # buy signal day
        "2026-01-06": _sig(0, close=90.0, open_=105.0),   # exit signal day
    }}
    stats, portfolio = run_simple_backtest(
        df_feat=None, signals=signals, initial_capital=100_000, slippage=0.0
    )

    buys = [t for t in portfolio.trades if t["action"] == "BUY"]
    sells = [t for t in portfolio.trades if t["action"] == "SELL"]
    assert buys, "expected a buy fill"
    assert buys[0]["price"] == 100.0  # OPEN of the buy-signal bar
    assert buys[0]["price"] != 110.0  # ...not its close
    assert sells, "expected an exit fill"
    assert sells[0]["price"] == 105.0  # OPEN of the exit-signal bar


def test_simple_backtest_legacy_signals_fall_back_to_close():
    """Signal dicts without an 'open' key keep working (close reference)."""
    from src.trading.backtester import run_simple_backtest

    signals = {"TEST.NS": {
        "2026-01-05": {"direction": 1, "price": 100.0},
    }}
    stats, portfolio = run_simple_backtest(
        df_feat=None, signals=signals, initial_capital=100_000, slippage=0.0
    )
    buys = [t for t in portfolio.trades if t["action"] == "BUY"]
    assert buys and buys[0]["price"] == 100.0
