"""Comprehensive tests for the advanced portfolio optimizer.

Tests cover:
    - NSE cost model with exact Indian market rates
    - Black-Litterman model (equilibrium, views, diagnostics)
    - Mean-CVaR optimization (95% and 99% confidence)
    - Portfolio constraints engine
    - Efficient frontier generation
    - Cost-adjusted return calculations
    - Full optimization pipeline integration
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──


@pytest.fixture
def sample_prices():
    """Multi-asset price DataFrame for optimizer testing."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2024-01-01", periods=n)
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    data = {}
    for i, t in enumerate(tickers):
        base = 100 + i * 50
        data[t] = base + np.cumsum(np.random.randn(n) * (1.0 + i * 0.2))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_prices_large():
    """Larger asset set for realistic optimizer testing."""
    np.random.seed(123)
    n = 252
    dates = pd.bdate_range("2024-01-01", periods=n)
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "KOTAKBANK.NS",
        "BAJFINANCE.NS",
        "LT.NS",
    ]
    data = {}
    for i, t in enumerate(tickers):
        mu = np.random.uniform(-0.0002, 0.0005)
        sigma = np.random.uniform(0.01, 0.025)
        data[t] = 100 * np.exp(np.cumsum(np.random.normal(mu, sigma, n)))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def cost_model():
    from src.trading.optimizer_advanced import NSECostModel

    return NSECostModel()


@pytest.fixture
def bl_model():
    from src.trading.optimizer_advanced import BlackLittermanModel

    return BlackLittermanModel()


@pytest.fixture
def sample_returns_matrix():
    np.random.seed(42)
    return np.random.randn(252, 5) * 0.015


# ── NSE Cost Model Tests ──


class TestNSECostModel:
    def test_buy_side_costs(self, cost_model):
        costs = cost_model.calculate_trade_cost(100_000, "buy")
        assert costs["brokerage"] > 0
        assert costs["stt"] > 0
        assert costs["stamp_duty"] > 0
        assert costs["gst"] > 0
        assert costs["total"] > 0
        assert costs["effective_rate"] > 0

    def test_sell_side_costs(self, cost_model):
        costs = cost_model.calculate_trade_cost(100_000, "sell")
        assert costs["brokerage"] > 0
        assert costs["stt"] > 0
        assert costs["stamp_duty"] == 0.0
        assert costs["gst"] > 0
        assert costs["total"] > 0

    def test_stt_rate(self, cost_model):
        costs = cost_model.calculate_trade_cost(100_000, "buy")
        expected_stt = 100_000 * 0.001
        assert abs(costs["stt"] - expected_stt) < 0.01

    def test_exchange_charge_rate(self, cost_model):
        costs = cost_model.calculate_trade_cost(100_000, "buy")
        expected = 100_000 * 0.0000345
        assert abs(costs["exchange_charge"] - expected) < 0.001

    def test_gst_applied_to_brokerage_and_exchange(self, cost_model):
        costs = cost_model.calculate_trade_cost(100_000, "buy")
        expected_gst = (100_000 * 0.0003 + 100_000 * 0.0000345) * 0.18
        assert abs(costs["gst"] - expected_gst) < 0.01

    def test_round_trip_cost_rate(self, cost_model):
        rate = cost_model.total_round_trip_rate()
        # STT on both sides (0.2%) + exchange + SEBI + stamp + GST + slippage
        assert rate > 0.002
        assert rate < 0.02

    def test_zero_trade_value(self, cost_model):
        costs = cost_model.calculate_trade_cost(0, "buy")
        assert costs["total"] == 0.0
        assert costs["effective_rate"] == 0.0

    def test_slippage_with_adv(self, cost_model):
        costs_high_adv = cost_model.calculate_trade_cost(100_000, "buy", 0.05, 10_000_000)
        costs_low_adv = cost_model.calculate_trade_cost(100_000, "buy", 0.05, 100_000)
        assert costs_low_adv["slippage"] > costs_high_adv["slippage"]

    def test_costs_scale_linearly(self, cost_model):
        c1 = cost_model.calculate_trade_cost(50_000, "buy")
        c2 = cost_model.calculate_trade_cost(100_000, "buy")
        ratio = c2["total"] / c1["total"]
        assert 1.5 < ratio < 2.5

    def test_effective_rate_reasonable(self, cost_model):
        costs = cost_model.calculate_trade_cost(100_000, "buy")
        # NSE delivery costs should be ~0.2-0.5% per side
        assert 0.001 < costs["effective_rate"] < 0.01


# ── Black-Litterman Tests ──


class TestBlackLittermanModel:
    def test_no_views_returns_market(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        result = bl_model.compute(market_w, cov, [], [])
        np.testing.assert_array_almost_equal(result["weights"], market_w)
        assert result["views_applied"] is False
        assert result["n_views"] == 0

    def test_single_view_shifts_weights(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.15)]
        confidences = [0.8]
        result = bl_model.compute(market_w, cov, views, confidences)
        assert result["views_applied"] is True
        assert result["weights"][0] > market_w[0]

    def test_weights_sum_to_one(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.10), (1, -0.05)]
        confidences = [0.7, 0.5]
        result = bl_model.compute(market_w, cov, views, confidences)
        assert abs(np.sum(result["weights"]) - 1.0) < 0.01

    def test_no_negative_weights(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.20)]
        confidences = [0.9]
        result = bl_model.compute(market_w, cov, views, confidences)
        assert np.all(result["weights"] >= 0)

    def test_high_confidence_view_dominates(self, bl_model):
        n = 3
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.30)]
        result_high = bl_model.compute(market_w, cov, views, [0.95])
        result_low = bl_model.compute(market_w, cov, views, [0.10])
        assert result_high["weights"][0] > result_low["weights"][0]

    def test_posterior_return_differs_from_equilibrium(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.20), (2, -0.10)]
        confidences = [0.8, 0.6]
        result = bl_model.compute(market_w, cov, views, confidences)
        assert not np.allclose(result["expected_return"], result["posterior_return"])

    def test_diagnostics_present(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.10)]
        confidences = [0.7]
        result = bl_model.compute(market_w, cov, views, confidences)
        assert "view_impact" in result
        assert "avg_confidence" in result
        assert "posterior_variance" in result

    def test_multiple_views(self, bl_model):
        n = 5
        market_w = np.ones(n) / n
        cov = np.eye(n) * 0.0004
        views = [(0, 0.15), (1, 0.10), (2, -0.05)]
        confidences = [0.8, 0.6, 0.5]
        result = bl_model.compute(market_w, cov, views, confidences)
        assert result["n_views"] == 3
        assert result["views_applied"] is True


# ── Mean-CVaR Tests ──


class TestMeanCVaROptimizer:
    def test_returns_dict_with_required_keys(self, sample_returns_matrix):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        result = mean_cvar_optimize(sample_returns_matrix, 0.95)
        assert "weights" in result
        assert "cvar" in result
        assert "return" in result
        assert "volatility" in result
        assert "success" in result

    def test_weights_sum_to_one(self, sample_returns_matrix):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        result = mean_cvar_optimize(sample_returns_matrix, 0.95)
        assert abs(np.sum(result["weights"]) - 1.0) < 0.01

    def test_weights_non_negative(self, sample_returns_matrix):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        result = mean_cvar_optimize(sample_returns_matrix, 0.95)
        assert np.all(result["weights"] >= 0)

    def test_cvar_95_vs_99(self, sample_returns_matrix):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        r95 = mean_cvar_optimize(sample_returns_matrix, 0.95)
        r99 = mean_cvar_optimize(sample_returns_matrix, 0.99)
        # 99% CVaR should be worse (more negative) than 95% CVaR
        assert r99["cvar"] <= r95["cvar"]

    def test_max_weight_constraint(self, sample_returns_matrix):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        result = mean_cvar_optimize(sample_returns_matrix, 0.95, max_stock_weight=0.30)
        assert np.all(result["weights"] <= 0.30 + 1e-6)

    def test_accepts_dataframe(self, sample_prices):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        returns = sample_prices.pct_change().dropna()
        result = mean_cvar_optimize(returns, 0.95)
        assert result["success"] in (True, False)

    def test_cvar_is_negative(self, sample_returns_matrix):
        from src.trading.optimizer_advanced import mean_cvar_optimize

        result = mean_cvar_optimize(sample_returns_matrix, 0.95)
        assert result["cvar"] <= 0


# ── Constraint Engine Tests ──


class TestConstraintsEngine:
    def test_stock_cap_enforced(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(
            sample_prices,
            max_stock_weight=0.20,
        )
        for key in ["max_sharpe", "min_variance", "black_litterman"]:
            weights = result[key]["weights"]
            assert np.all(weights <= 0.20 + 1e-6), f"{key} violates stock cap"

    def test_sector_cap_enforced(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(
            sample_prices,
            max_sector_weight=0.40,
        )
        # Check IT sector weight (TCS.NS + INFY.NS)
        constraints = result["constraints"]
        sector_map = constraints["sector_mapping"]
        tickers = result["tickers"]
        for key in ["max_sharpe", "min_variance", "black_litterman"]:
            weights = result[key]["weights"]
            it_weight = sum(weights[i] for i, t in enumerate(tickers) if sector_map.get(t) == "IT")
            assert it_weight <= 0.40 + 1e-6, f"{key} IT sector weight {it_weight:.4f} > 40%"

    def test_weights_sum_to_one(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        for key in ["max_sharpe", "min_variance", "black_litterman", "mean_cvar_95", "mean_cvar_99"]:
            assert abs(np.sum(result[key]["weights"]) - 1.0) < 0.01, f"{key} weights don't sum to 1"

    def test_no_negative_weights(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        for key in ["max_sharpe", "min_variance", "black_litterman", "mean_cvar_95", "mean_cvar_99"]:
            assert np.all(result[key]["weights"] >= -1e-6), f"{key} has negative weights"


# ── Efficient Frontier Tests ──


class TestEfficientFrontier:
    def test_frontier_has_points(self, sample_prices):
        from src.trading.optimizer_advanced import compute_efficient_frontier

        result = compute_efficient_frontier(sample_prices, n_points=10)
        assert len(result["frontier"]) > 0

    def test_frontier_monotonic_volatility(self, sample_prices):
        from src.trading.optimizer_advanced import compute_efficient_frontier

        result = compute_efficient_frontier(sample_prices, n_points=15)
        vols = [p["volatility"] for p in result["frontier"]]
        # Volatility should generally increase along the frontier
        assert vols[-1] >= vols[0]

    def test_cost_adjusted_return_lower(self, sample_prices):
        from src.trading.optimizer_advanced import compute_efficient_frontier

        result = compute_efficient_frontier(sample_prices, n_points=5)
        for point in result["frontier"]:
            assert point["cost_adjusted_return"] <= point["return"] + 1e-6

    def test_cost_model_present(self, sample_prices):
        from src.trading.optimizer_advanced import compute_efficient_frontier

        result = compute_efficient_frontier(sample_prices, n_points=5)
        assert "cost_model" in result
        assert "round_trip_rate" in result["cost_model"]


# ── Cost-Adjusted Returns Tests ──


class TestCostAdjustedReturns:
    def test_gross_vs_net(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        ca = result["cost_adjusted"]
        assert ca["max_sharpe"]["net_return"] <= ca["max_sharpe"]["gross_return"]
        assert ca["black_litterman"]["net_return"] <= ca["black_litterman"]["gross_return"]

    def test_transaction_costs_positive(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        ca = result["cost_adjusted"]
        assert ca["max_sharpe"]["transaction_costs"] >= 0
        assert ca["black_litterman"]["transaction_costs"] >= 0

    def test_cost_analysis_structure(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        assert "cost_analysis" in result
        for key in ["max_sharpe", "min_variance", "black_litterman"]:
            assert "total" in result["cost_analysis"][key]
            assert "turnover" in result["cost_analysis"][key]


# ── Integration Tests ──


class TestOptimizePortfolioAdvanced:
    def test_returns_all_keys(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        expected_keys = [
            "tickers",
            "max_sharpe",
            "min_variance",
            "black_litterman",
            "mean_cvar_95",
            "mean_cvar_99",
            "cost_adjusted",
            "efficient_frontier",
            "interactive_frontier",
            "returns_stats",
            "cost_analysis",
            "correlation",
            "constraints",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_with_views(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        views = [(0, 0.15), (1, 0.10)]
        confidences = [0.8, 0.6]
        result = optimize_portfolio_advanced(sample_prices, views, confidences)
        assert result["black_litterman"]["n_views"] == 2

    def test_with_excluded_stocks(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(
            sample_prices,
            excluded_stocks=["TCS.NS"],
            max_stock_weight=0.40,
        )
        # TCS.NS weight should be zero or near-zero
        tickers = result["tickers"]
        tcs_idx = tickers.index("TCS.NS")
        assert result["max_sharpe"]["weights"][tcs_idx] < 0.01

    def test_different_confidence_levels(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        # 99% CVaR should generally be more conservative than 95%
        assert result["mean_cvar_99"]["cvar"] <= result["mean_cvar_95"]["cvar"]

    def test_correlation_matrix_shape(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        n = len(result["tickers"])
        corr = result["correlation"]
        for t1 in result["tickers"]:
            for t2 in result["tickers"]:
                assert t1 in corr and t2 in corr[t1]

    def test_returns_stats_structure(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        stats = result["returns_stats"]
        assert "mean_daily" in stats
        assert "annualized" in stats
        for t in result["tickers"]:
            assert t in stats["mean_daily"]
            assert t in stats["annualized"]
            # Annualized should be ~252x daily (tolerance accounts for rounding to 6 decimals)
            daily = stats["mean_daily"][t]
            annual = stats["annualized"][t]
            expected_annual = daily * 252
            if abs(expected_annual) > 0.01:
                assert abs(annual - expected_annual) / abs(expected_annual) < 0.01
            else:
                assert abs(annual - expected_annual) < 0.001

    def test_constraints_recorded(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(
            sample_prices,
            max_stock_weight=0.25,
            max_sector_weight=0.45,
        )
        c = result["constraints"]
        assert c["max_stock_weight"] == 0.25
        assert c["max_sector_weight"] == 0.45

    def test_efficient_frontier_has_points(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        assert len(result["efficient_frontier"]) > 0

    def test_interactive_frontier_structure(self, sample_prices):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        result = optimize_portfolio_advanced(sample_prices)
        interactive = result["interactive_frontier"]
        assert "max_sharpe" in interactive
        assert "min_variance" in interactive
        assert "black_litterman" in interactive
        assert "frontier" in interactive


# ── Edge Cases ──


class TestEdgeCases:
    def test_minimum_assets(self):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        np.random.seed(42)
        n = 60
        dates = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.DataFrame(
            {
                "A.NS": 100 + np.cumsum(np.random.randn(n) * 0.5),
                "B.NS": 200 + np.cumsum(np.random.randn(n) * 0.8),
                "C.NS": 150 + np.cumsum(np.random.randn(n) * 0.6),
            },
            index=dates,
        )
        result = optimize_portfolio_advanced(prices)
        assert len(result["tickers"]) == 3
        assert abs(np.sum(result["max_sharpe"]["weights"]) - 1.0) < 0.01

    def test_identical_prices(self):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.DataFrame(
            {
                "A.NS": np.ones(n) * 100,
                "B.NS": np.ones(n) * 100,
            },
            index=dates,
        )
        result = optimize_portfolio_advanced(prices)
        # Should not crash, weights should be valid
        assert abs(np.sum(result["max_sharpe"]["weights"]) - 1.0) < 0.01

    def test_very_volatile_prices(self):
        from src.trading.optimizer_advanced import optimize_portfolio_advanced

        np.random.seed(42)
        n = 252
        dates = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.DataFrame(
            {
                "A.NS": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.05)),
                "B.NS": 200 * np.exp(np.cumsum(np.random.randn(n) * 0.05)),
            },
            index=dates,
        )
        result = optimize_portfolio_advanced(prices)
        assert "max_sharpe" in result
