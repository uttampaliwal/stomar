"""Tests for src/ranking.py — fundamental ranking."""

import numpy as np
import pandas as pd

from src.signals.ranking import (
    rank_stocks,
    get_recommendation,
    fetch_fundamentals,
    fundamental_score,
    clear_fundamental_cache,
)


def _make_df(n=120, close_start=100):
    """Synthetic OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = close_start + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.rand(n) * 0.5,
        "high": close + np.abs(np.random.randn(n) * 0.5),
        "low": close - np.abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(100000, 500000, n),
    }, index=dates)


class TestFundamentalScore:
    def test_good_fundamentals(self):
        fund = {
            "pe_ratio": 12.0, "pb_ratio": 1.5, "roe": 0.20,
            "roce": 0.25, "dividend_yield": 0.03, "debt_to_equity": 50,
            "profit_margin": 0.15,
        }
        score = fundamental_score(fund)
        assert 50 < score <= 100

    def test_bad_fundamentals(self):
        fund = {
            "pe_ratio": 80.0, "pb_ratio": 15.0, "roe": 0.02,
            "roce": 0.03, "dividend_yield": 0.0, "debt_to_equity": 300,
            "profit_margin": 0.01,
        }
        score = fundamental_score(fund)
        assert 0 <= score < 50

    def test_neutral_empty(self):
        score = fundamental_score({})
        assert 0 <= score <= 100  # all zeros → debt factor gives high score

    def test_result_in_range(self):
        fund = {
            "pe_ratio": 25.0, "pb_ratio": 3.0, "roe": 0.12,
            "roce": 0.15, "dividend_yield": 0.01, "debt_to_equity": 100,
            "profit_margin": 0.08,
        }
        score = fundamental_score(fund)
        assert 0 <= score <= 100

    def test_high_roe_beats_low(self):
        high = fundamental_score({"roe": 0.30})
        low = fundamental_score({"roe": 0.05})
        assert high > low

    def test_low_pe_beats_high(self):
        low_pe = fundamental_score({"pe_ratio": 8.0})
        high_pe = fundamental_score({"pe_ratio": 50.0})
        assert low_pe > high_pe


class TestFetchFundamentals:
    def test_returns_dict_keys(self):
        clear_fundamental_cache()
        # Use a known ticker — but this test is offline-safe via cache mock
        from unittest.mock import patch
        mock_info = {
            "trailingPE": 20.0, "priceToBook": 2.5, "returnOnCapitalEmployed": 0.15,
            "returnOnEquity": 0.18, "dividendYield": 0.02, "debtToEquity": 80,
            "marketCap": 1e12, "profitMargins": 0.12,
        }
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = mock_info
            result = fetch_fundamentals("TEST.NS")
            assert result["pe_ratio"] == 20.0
            assert result["roe"] == 0.18
            assert result["debt_to_equity"] == 80.0

    def test_caches_result(self):
        clear_fundamental_cache()
        from unittest.mock import patch
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {"trailingPE": 15.0}
            fetch_fundamentals("CACHE.NS")
            fetch_fundamentals("CACHE.NS")
            assert mock_ticker.call_count == 1

    def test_handles_missing_fields(self):
        clear_fundamental_cache()
        from unittest.mock import patch
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {}
            result = fetch_fundamentals("EMPTY.NS")
            assert result["pe_ratio"] == 0.0
            assert result["roe"] == 0.0

    def test_clear_cache(self):
        clear_fundamental_cache()
        from unittest.mock import patch
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {"trailingPE": 10.0}
            fetch_fundamentals("CLEAR.NS")
            clear_fundamental_cache()
            fetch_fundamentals("CLEAR.NS")
            assert mock_ticker.call_count == 2


class TestRankStocksWithFundamentals:
    def test_basic_ranking(self):
        data = {"A.NS": _make_df(120, 100), "B.NS": _make_df(120, 120)}
        rankings = rank_stocks(data)
        assert len(rankings) == 2
        assert rankings[0]["composite_score"] >= rankings[1]["composite_score"]

    def test_includes_fundamental_fields(self):
        data = {"A.NS": _make_df(120, 100)}
        rankings = rank_stocks(data, use_fundamentals=False)
        assert "fundamental_score" in rankings[0]
        assert "fundamentals" in rankings[0]

    def test_use_fundamentals_flag(self):
        from unittest.mock import patch
        data = {"TEST.NS": _make_df(120, 100)}
        mock_fund = {
            "pe_ratio": 10.0, "roe": 0.25, "roce": 0.20,
            "dividend_yield": 0.03, "debt_to_equity": 40,
            "profit_margin": 0.15, "pb_ratio": 1.2, "market_cap": 1e12,
        }
        with patch("src.signals.ranking.fetch_fundamentals", return_value=mock_fund):
            rankings = rank_stocks(data, use_fundamentals=True)
            assert rankings[0]["fundamental_score"] > 50

    def test_skips_short_data(self):
        data = {"SHORT.NS": _make_df(30, 100), "LONG.NS": _make_df(120, 100)}
        rankings = rank_stocks(data)
        assert len(rankings) == 1
        assert rankings[0]["ticker"] == "LONG.NS"

    def test_ranking_has_percentile(self):
        data = {f"T{i}.NS": _make_df(120, 100 + i * 10) for i in range(5)}
        rankings = rank_stocks(data)
        assert all("percentile" in r for r in rankings)
        assert all("rank" in r for r in rankings)


class TestGetRecommendation:
    def test_top_stock_strong_buy(self):
        rankings = [
            {"ticker": "A.NS", "rank": 1, "percentile": 90, "composite_score": 0.8},
            {"ticker": "B.NS", "rank": 2, "percentile": 50, "composite_score": 0.5},
        ]
        rec = get_recommendation(rankings, "A.NS")
        assert rec["recommendation"] == "STRONG BUY"

    def test_bottom_stock_sell(self):
        rankings = [
            {"ticker": "A.NS", "rank": 1, "percentile": 90, "composite_score": 0.8},
            {"ticker": "B.NS", "rank": 2, "percentile": 10, "composite_score": 0.3},
        ]
        rec = get_recommendation(rankings, "B.NS")
        assert rec["recommendation"] in ("SELL", "STRONG SELL")

    def test_unknown_ticker(self):
        rec = get_recommendation([], "NOPE.NS")
        assert rec["recommendation"] == "HOLD"


class TestFactorAnalysis:
    def test_returns_correlations(self):
        data = {f"T{i}.NS": _make_df(120, 100 + i * 10) for i in range(5)}
        from src.signals.ranking import factor_analysis
        result = factor_analysis(data)
        assert "factor_correlations" in result
        assert "n_stocks" in result
        assert result["n_stocks"] == 5

    def test_insufficient_data(self):
        data = {"A.NS": _make_df(120, 100)}
        from src.signals.ranking import factor_analysis
        result = factor_analysis(data)
        assert result["insufficient_data"] is True
