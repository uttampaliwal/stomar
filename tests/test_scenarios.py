"""Tests for regime-conditional and backtest scenarios modules."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_df(n=500):
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close_vals = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_vals = close_vals + abs(np.random.randn(n) * 0.5)
    low_vals = close_vals - abs(np.random.randn(n) * 0.5)
    open_vals = close_vals + np.random.randn(n) * 0.2
    volume = np.random.randint(100000, 1000000, n)

    df = pd.DataFrame({
        "close": close_vals, "high": high_vals, "low": low_vals,
        "open": open_vals, "volume": volume,
    }, index=dates)
    return df


class TestRegimeAllocation:
    def test_bull_allocation(self):
        from src.regime_strategy import get_regime_allocation
        result = get_regime_allocation("Bull")
        assert result["equity_pct"] > 0.5
        assert result["strategy"] == "momentum"

    def test_bear_allocation(self):
        from src.regime_strategy import get_regime_allocation
        result = get_regime_allocation("Bear")
        assert result["equity_pct"] < 0.5
        assert result["strategy"] == "defensive"

    def test_sideways_allocation(self):
        from src.regime_strategy import get_regime_allocation
        result = get_regime_allocation("Sideways")
        assert 0.3 <= result["equity_pct"] <= 0.7

    def test_low_confidence_reduces(self):
        from src.regime_strategy import get_regime_allocation
        r1 = get_regime_allocation("Bull", confidence=0.8)
        r2 = get_regime_allocation("Bull", confidence=0.2)
        assert r2["equity_pct"] < r1["equity_pct"]


class TestRegimePositionSize:
    def test_bull_larger(self):
        from src.regime_strategy import regime_adjusted_position_size
        r1 = regime_adjusted_position_size(0.10, "Bull")
        r2 = regime_adjusted_position_size(0.10, "Bear")
        assert r1["position_size_pct"] > r2["position_size_pct"]

    def test_vol_adjustment(self):
        from src.regime_strategy import regime_adjusted_position_size
        r1 = regime_adjusted_position_size(0.10, "Bull", current_vol=0.10)
        r2 = regime_adjusted_position_size(0.10, "Bull", current_vol=0.30)
        assert r1["position_size_pct"] > r2["position_size_pct"]


class TestBacktestRegimeStrategy:
    def test_returns_all_keys(self):
        from src.regime_strategy import backtest_regime_strategy
        df = _make_df()
        result = backtest_regime_strategy(df)
        expected = ["regime", "equity_allocation", "strategy_return",
                    "strategy_annualized", "strategy_sharpe",
                    "buy_hold_return", "buy_hold_annualized", "excess_return"]
        for k in expected:
            assert k in result


class TestScenarios:
    def test_buy_and_hold(self):
        from src.scenarios import scenario_buy_and_hold
        df = _make_df()
        result = scenario_buy_and_hold(df)
        assert result["name"] == "Buy & Hold"
        assert "total_return" in result
        assert "sharpe" in result

    def test_momentum(self):
        from src.scenarios import scenario_momentum
        df = _make_df()
        result = scenario_momentum(df, lookback=20)
        assert "Momentum" in result["name"]
        assert result["n_trades"] >= 0

    def test_mean_reversion(self):
        from src.scenarios import scenario_mean_reversion
        df = _make_df()
        result = scenario_mean_reversion(df, window=20)
        assert "Mean Reversion" in result["name"]

    def test_volatility_target(self):
        from src.scenarios import scenario_volatility_target
        df = _make_df()
        result = scenario_volatility_target(df, target_vol=0.15)
        assert "Vol Target" in result["name"]

    def test_sma_crossover(self):
        from src.scenarios import scenario_sma_crossover
        df = _make_df()
        result = scenario_sma_crossover(df, short_window=10, long_window=50)
        assert "SMA" in result["name"]

    def test_run_all_scenarios(self):
        from src.scenarios import run_all_scenarios
        df = _make_df()
        results = run_all_scenarios(df)
        assert len(results) >= 10
        # Should be sorted by Sharpe
        sharpes = [r["sharpe"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)
