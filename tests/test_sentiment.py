"""Tests for src/sentiment.py — multi-source sentiment analysis."""

import os
from unittest.mock import MagicMock, patch


from src.sentiment import (
    _deduplicate,
    _fetch_google_news,
    _fetch_moneycontrol,
    _fetch_screener,
    _fetch_economic_times,
    _fetch_yahoo_news,
    analyze_sentiment,
    fetch_news_headlines,
    get_stock_sentiment,
    SOURCE_WEIGHTS,
)


class TestSourceWeights:
    def test_all_sources_have_weights(self):
        assert "Yahoo Finance" in SOURCE_WEIGHTS
        assert "MoneyControl" in SOURCE_WEIGHTS
        assert "Economic Times" in SOURCE_WEIGHTS
        assert "Screener.in" in SOURCE_WEIGHTS
        assert "Google News" in SOURCE_WEIGHTS

    def test_weights_are_positive(self):
        for source, weight in SOURCE_WEIGHTS.items():
            assert weight > 0, f"{source} weight should be positive"
            assert weight <= 1.0, f"{source} weight should be <= 1.0"


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        articles = [
            {"title": "Reliance posts strong Q3 results", "source": "A"},
            {"title": "Reliance posts strong Q3 results", "source": "B"},
        ]
        result = _deduplicate(articles)
        assert len(result) == 1

    def test_keeps_different_articles(self):
        articles = [
            {"title": "Reliance posts strong Q3 results", "source": "A"},
            {"title": "TCS beats earnings estimates", "source": "B"},
        ]
        result = _deduplicate(articles)
        assert len(result) == 2

    def test_filters_short_titles(self):
        articles = [
            {"title": "Hi", "source": "A"},
            {"title": "Reliance posts strong Q3 results this quarter", "source": "B"},
        ]
        result = _deduplicate(articles)
        assert len(result) == 1
        assert "Reliance" in result[0]["title"]


class TestFetchYahooNews:
    def test_returns_list(self):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = [
                {"title": "Test headline", "summary": "Summary", "publisher": "Yahoo", "providerPublishTime": 1700000000}
            ]
            result = _fetch_yahoo_news("TEST.NS")
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["source"] == "Yahoo"

    def test_handles_empty_news(self):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = _fetch_yahoo_news("TEST.NS")
            assert result == []

    def test_handles_exception(self):
        with patch("yfinance.Ticker", side_effect=Exception("Network error")):
            result = _fetch_yahoo_news("TEST.NS")
            assert result == []


class TestFetchGoogleNews:
    def test_handles_exception(self):
        with patch("requests.get", side_effect=Exception("Timeout")):
            result = _fetch_google_news("TEST.NS")
            assert result == []


class TestFetchMoneyControl:
    def test_returns_list(self):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '<a class="news_title" href="/test">Reliance gains 2%</a>'
            mock_get.return_value = mock_resp
            result = _fetch_moneycontrol("TEST.NS")
            assert isinstance(result, list)

    def test_handles_exception(self):
        with patch("requests.get", side_effect=Exception("Timeout")):
            result = _fetch_moneycontrol("TEST.NS")
            assert result == []


class TestFetchEconomicTimes:
    def test_handles_exception(self):
        with patch("requests.get", side_effect=Exception("Timeout")):
            result = _fetch_economic_times("TEST.NS")
            assert result == []


class TestFetchScreener:
    def test_handles_exception(self):
        with patch("requests.get", side_effect=Exception("Timeout")):
            result = _fetch_screener("TEST.NS")
            assert result == []


class TestFetchHeadlines:
    def test_deduplicates_across_sources(self):
        with patch("src.sentiment._fetch_yahoo_news") as mock_yahoo, \
             patch("src.sentiment._fetch_google_news") as mock_google, \
             patch("src.sentiment._fetch_moneycontrol") as mock_mc, \
             patch("src.sentiment._fetch_economic_times") as mock_et, \
             patch("src.sentiment._fetch_screener") as mock_sc:
            mock_yahoo.return_value = [{"title": "Same headline about stock", "source": "Yahoo"}]
            mock_google.return_value = [{"title": "Same headline about stock", "source": "Google"}]
            mock_mc.return_value = [{"title": "Different headline about results", "source": "MC"}]
            mock_et.return_value = []
            mock_sc.return_value = []
            result = fetch_news_headlines("TEST.NS")
            assert len(result) == 2

    def test_respects_max_articles(self):
        with patch("src.sentiment._fetch_yahoo_news") as mock_yahoo, \
             patch("src.sentiment._fetch_google_news") as mock_google, \
             patch("src.sentiment._fetch_moneycontrol") as mock_mc, \
             patch("src.sentiment._fetch_economic_times") as mock_et, \
             patch("src.sentiment._fetch_screener") as mock_sc:
            mock_yahoo.return_value = [{"title": f"Headline {i} about the stock market today", "source": "Y"} for i in range(20)]
            mock_google.return_value = []
            mock_mc.return_value = []
            mock_et.return_value = []
            mock_sc.return_value = []
            result = fetch_news_headlines("TEST.NS", max_articles=5)
            assert len(result) <= 5


class TestAnalyzeSentiment:
    def test_empty_articles(self):
        result = analyze_sentiment([])
        assert result["score"] == 0.0
        assert result["label"] == "Neutral"
        assert result["source_breakdown"] == {}

    def test_returns_all_keys(self):
        articles = [{"title": "Stock rallies on strong earnings", "source": "Yahoo"}]
        with patch("src.sentiment.get_finbert") as mock_finbert:
            mock_pipe = MagicMock()
            mock_pipe.return_value = [{"label": "positive", "score": 0.9}]
            mock_finbert.return_value = mock_pipe
            result = analyze_sentiment(articles)
            assert "score" in result
            assert "weighted_score" in result
            assert "source_breakdown" in result
            assert "sources_used" in result

    def test_positive_articles_give_positive_score(self):
        articles = [
            {"title": "Stock surges on great results", "source": "Yahoo"},
            {"title": "Strong buy rating maintained", "source": "MoneyControl"},
        ]
        with patch("src.sentiment.get_finbert") as mock_finbert:
            mock_pipe = MagicMock()
            mock_pipe.return_value = [
                {"label": "positive", "score": 0.9},
                {"label": "positive", "score": 0.85},
            ]
            mock_finbert.return_value = mock_pipe
            result = analyze_sentiment(articles)
            assert result["score"] > 0
            assert result["label"] == "Positive"

    def test_negative_articles_give_negative_score(self):
        articles = [
            {"title": "Stock crashes on poor guidance", "source": "ET"},
        ]
        with patch("src.sentiment.get_finbert") as mock_finbert:
            mock_pipe = MagicMock()
            mock_pipe.return_value = [{"label": "negative", "score": 0.95}]
            mock_finbert.return_value = mock_pipe
            result = analyze_sentiment(articles)
            assert result["score"] < 0
            assert result["label"] == "Negative"

    def test_source_breakdown_populated(self):
        articles = [
            {"title": "Headline from Yahoo", "source": "Yahoo Finance"},
            {"title": "Headline from MC", "source": "MoneyControl"},
        ]
        with patch("src.sentiment.get_finbert") as mock_finbert:
            mock_pipe = MagicMock()
            mock_pipe.return_value = [
                {"label": "positive", "score": 0.8},
                {"label": "negative", "score": 0.7},
            ]
            mock_finbert.return_value = mock_pipe
            result = analyze_sentiment(articles)
            assert "Yahoo Finance" in result["source_breakdown"]
            assert "MoneyControl" in result["source_breakdown"]

    def test_handles_finbert_exception(self):
        articles = [{"title": "Test headline", "source": "Yahoo"}]
        with patch("src.sentiment.get_finbert", side_effect=Exception("Model load failed")):
            result = analyze_sentiment(articles)
            assert result["score"] == 0.0
            assert "error" in result


class TestGetStockSentiment:
    def test_returns_cached_result(self, tmp_path):
        import json
        cache_dir = tmp_path / "data"
        cache_dir.mkdir()
        cache_file = cache_dir / "sentiment_TEST_NS.json"
        cached = {"score": 0.5, "label": "Positive", "ticker": "TEST.NS"}
        with open(cache_file, "w") as f:
            json.dump(cached, f)

        with patch("src.sentiment._sentiment_cache_dir", str(cache_dir)):
            result = get_stock_sentiment("TEST.NS")
            assert result["score"] == 0.5

    def test_fetches_when_no_cache(self, tmp_path):
        cache_dir = str(tmp_path / "empty_cache")
        os.makedirs(cache_dir, exist_ok=True)

        import src.sentiment as mod
        old_dir = mod._sentiment_cache_dir
        mod._sentiment_cache_dir = cache_dir
        try:
            with patch.object(mod, "fetch_news_headlines", return_value=[{"title": "Test headline about the stock market", "source": "Yahoo"}]) as mock_fetch, \
                 patch.object(mod, "analyze_sentiment", return_value={"score": 0.3, "label": "Positive"}) as mock_analyze:
                result = get_stock_sentiment("TEST.NS")
                assert result["ticker"] == "TEST.NS"
                mock_fetch.assert_called_once()
        finally:
            mod._sentiment_cache_dir = old_dir


class TestMultiSourceAggregation:
    def test_weighted_score_differs_from_unweighted(self):
        """Source-weighted score should differ when sources have different weights."""
        articles = [
            {"title": "Very positive headline", "source": "Reddit"},
            {"title": "Slightly positive headline", "source": "Yahoo Finance"},
        ]
        with patch("src.sentiment.get_finbert") as mock_finbert:
            mock_pipe = MagicMock()
            # Reddit: high positive, Yahoo: low positive
            mock_pipe.return_value = [
                {"label": "positive", "score": 0.95},
                {"label": "positive", "score": 0.55},
            ]
            mock_finbert.return_value = mock_pipe
            result = analyze_sentiment(articles)
            # Yahoo (weight=1.0) gets more influence than Reddit (weight=0.5)
            assert result["weighted_score"] != result["score"]
