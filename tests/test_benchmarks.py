"""Tests for benchmark comparison module."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBuyAndHold:
    def test_cumulative_return(self):
        from src.benchmarks import buy_and_hold
        returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
        actuals = np.array([1, 0, 1, 1, 0])
        result = buy_and_hold(actuals, returns)
        expected_cum = np.prod(1 + returns) - 1
        assert abs(result["cumulative_return"] - expected_cum) < 1e-10

    def test_returns_all_keys(self):
        from src.benchmarks import buy_and_hold
        result = buy_and_hold(np.array([1, 0]), np.array([0.01, -0.01]))
        expected = ["name", "cumulative_return", "sharpe", "accuracy",
                    "n_trades", "annualized_return"]
        for k in expected:
            assert k in result

    def test_name(self):
        from src.benchmarks import buy_and_hold
        result = buy_and_hold(np.array([1]), np.array([0.01]))
        assert result["name"] == "Buy & Hold"


class TestMomentum20D:
    def test_signals_positive_momentum(self):
        from src.benchmarks import momentum_20d
        returns = np.array([0.01] * 25)  # Always up
        result = momentum_20d(returns)
        assert result["accuracy"] >= 0.8

    def test_returns_all_keys(self):
        from src.benchmarks import momentum_20d
        result = momentum_20d(np.array([0.01] * 30))
        expected = ["name", "cumulative_return", "sharpe", "accuracy",
                    "n_trades", "annualized_return"]
        for k in expected:
            assert k in result

    def test_name(self):
        from src.benchmarks import momentum_20d
        result = momentum_20d(np.array([0.01] * 30))
        assert result["name"] == "20D Momentum"


class TestAlwaysLong:
    def test_accuracy_is_fraction_positive(self):
        from src.benchmarks import always_long
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        result = always_long(returns)
        assert result["accuracy"] == float(np.mean(returns > 0))


class TestRandomStrategy:
    def test_deterministic(self):
        from src.benchmarks import random_strategy
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 6)
        r1 = random_strategy(returns, random_state=42)
        r2 = random_strategy(returns, random_state=42)
        assert r1["cumulative_return"] == r2["cumulative_return"]


class TestSmaCrossover:
    def test_bullish_trend(self):
        from src.benchmarks import sma_crossover
        returns = np.array([0.01] * 60)
        result = sma_crossover(returns)
        assert result["cumulative_return"] > 0


class TestCompareToBenchmarks:
    def test_returns_list_sorted_by_sharpe(self):
        from src.benchmarks import compare_to_benchmarks
        returns = np.random.randn(100) * 0.01
        strategy_returns = returns * 0.8  # correlated
        results = compare_to_benchmarks(strategy_returns, returns=returns)
        assert isinstance(results, list)
        assert len(results) >= 5
        sharpes = [r["sharpe"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_your_strategy_first(self):
        from src.benchmarks import compare_to_benchmarks
        returns = np.random.randn(100) * 0.01
        results = compare_to_benchmarks(returns * 0.5, returns=returns)
        # Your strategy should be in the list
        names = [r["name"] for r in results]
        assert "Your Strategy" in names
