"""Tests for cross-sectional ranking module."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_stock_data():
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)

    stocks = {}
    for ticker in ["RELIANCE", "TCS", "HDFCBANK"]:
        close = pd.Series(
            100 + np.cumsum(np.random.randn(n) * 0.5),
            index=dates
        )
        high = close + abs(np.random.randn(n) * 0.5)
        low = close - abs(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.2
        volume = np.random.randint(100000, 1000000, n)

        stocks[ticker] = pd.DataFrame({
            "close": close, "high": high, "low": low,
            "open": open_, "volume": volume,
        }, index=dates)

    return stocks


class TestMomentumScore:
    def test_returns_all_windows(self):
        from src.ranking import compute_momentum_score
        close = pd.Series(range(100), dtype=float)
        result = compute_momentum_score(close, windows=[5, 20])
        assert "mom_5d" in result
        assert "mom_20d" in result
        assert "momentum_combined" in result

    def test_positive_momentum(self):
        from src.ranking import compute_momentum_score
        close = pd.Series(range(100), dtype=float)
        result = compute_momentum_score(close)
        assert result["momentum_combined"] > 0


class TestVolatilityScore:
    def test_returns_all_keys(self):
        from src.ranking import compute_volatility_score
        returns = pd.Series(np.random.randn(100) * 0.01)
        result = compute_volatility_score(returns)
        assert "volatility" in result
        assert "vol_score" in result

    def test_lower_vol_higher_score(self):
        from src.ranking import compute_volatility_score
        low_vol = pd.Series(np.random.randn(100) * 0.005)
        high_vol = pd.Series(np.random.randn(100) * 0.05)
        r1 = compute_volatility_score(low_vol)
        r2 = compute_volatility_score(high_vol)
        assert r1["vol_score"] > r2["vol_score"]


class TestVolumeScore:
    def test_returns_all_keys(self):
        from src.ranking import compute_volume_score
        volume = pd.Series(np.random.randint(100000, 1000000, 100))
        result = compute_volume_score(volume)
        assert "relative_volume" in result
        assert "volume_score" in result


class TestTechnicalScore:
    def test_returns_all_keys(self):
        from src.ranking import compute_technical_score
        stocks = _make_stock_data()
        result = compute_technical_score(stocks["RELIANCE"])
        expected = ["rsi", "rsi_score", "macd", "macd_signal", "macd_score",
                    "price_vs_sma20", "price_vs_sma50", "sma_score",
                    "bb_position", "bb_score", "technical_combined"]
        for k in expected:
            assert k in result

    def test_scores_in_range(self):
        from src.ranking import compute_technical_score
        stocks = _make_stock_data()
        result = compute_technical_score(stocks["RELIANCE"])
        for k in ["rsi_score", "macd_score", "sma_score", "bb_score", "technical_combined"]:
            assert 0 <= result[k] <= 100


class TestRankStocks:
    def test_returns_sorted_list(self):
        from src.ranking import rank_stocks
        stocks = _make_stock_data()
        rankings = rank_stocks(stocks)
        assert len(rankings) == 3
        scores = [r["composite_score"] for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_has_rank_and_percentile(self):
        from src.ranking import rank_stocks
        stocks = _make_stock_data()
        rankings = rank_stocks(stocks)
        for r in rankings:
            assert "rank" in r
            assert "percentile" in r
            assert 1 <= r["rank"] <= 3


class TestGetRecommendation:
    def test_top_stock_strong_buy(self):
        from src.ranking import rank_stocks, get_recommendation
        stocks = _make_stock_data()
        rankings = rank_stocks(stocks)
        top_ticker = rankings[0]["ticker"]
        rec = get_recommendation(rankings, top_ticker)
        assert rec["recommendation"] in ["STRONG BUY", "BUY"]

    def test_unknown_ticker(self):
        from src.ranking import get_recommendation
        rec = get_recommendation([], "UNKNOWN")
        assert rec["recommendation"] == "HOLD"


class TestFactorAnalysis:
    def test_returns_correlations(self):
        from src.ranking import factor_analysis
        stocks = _make_stock_data()
        result = factor_analysis(stocks)
        assert "factor_correlations" in result
        assert "best_factor" in result
        assert result["n_stocks"] == 3
