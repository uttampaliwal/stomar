"""Sentiment analysis module with multi-source NLP.

Fetches headlines from multiple sources (Yahoo Finance, Google News,
MoneyControl, Economic Times, Screener.in) and analyzes sentiment
using FinBERT with source-weighted aggregation.

Usage:
    from src.sentiment import get_stock_sentiment
    result = get_stock_sentiment("RELIANCE.NS")
"""

import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_finbert_pipeline = None
_sentiment_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Source weights: more authoritative sources get higher weight
SOURCE_WEIGHTS = {
    "Yahoo Finance": 1.0,
    "MoneyControl": 0.9,
    "Economic Times": 0.85,
    "Screener.in": 0.8,
    "Google News": 0.7,
    "Reddit": 0.5,
    "StockTwits": 0.4,
}


def get_finbert():
    """Lazy-load FinBERT pipeline with timeout fallback."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        import concurrent.futures

        def _load():
            from transformers import pipeline
            return pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                max_length=512,
                truncation=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_load)
            _finbert_pipeline = future.result(timeout=30)
    except Exception as e:
        logger.warning("FinBERT load failed (%s), using keyword fallback", e)
        _finbert_pipeline = "fallback"

    return _finbert_pipeline


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

def _fetch_yahoo_news(ticker: str, max_articles: int = 15) -> list:
    """Fetch from yfinance news endpoint."""
    articles = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            for item in news[:max_articles]:
                title = item.get("title", "")
                summary = item.get("summary", "")
                pub_time = item.get("providerPublishTime")
                source = item.get("publisher", "")
                if title:
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "source": source or "Yahoo Finance",
                        "time": pd.Timestamp.fromtimestamp(pub_time) if pub_time else pd.NaT,
                    })
    except Exception as e:
        logger.debug("Yahoo news fetch failed for %s: %s", ticker, e)
    return articles


def _fetch_google_news(ticker: str, max_articles: int = 15) -> list:
    """Fetch from Google News RSS."""
    articles = []
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    rss_url = f"https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:max_articles]:
                title = item.find("title")
                pub = item.find("pubDate")
                if title is not None and title.text:
                    articles.append({
                        "title": title.text.strip(),
                        "summary": "",
                        "source": "Google News",
                        "time": pd.to_datetime(pub.text) if pub is not None and pub.text else pd.NaT,
                    })
    except Exception as e:
        logger.debug("Google News fetch failed for %s: %s", ticker, e)
    return articles


def _fetch_moneycontrol(ticker: str, max_articles: int = 10) -> list:
    """Fetch from MoneyControl news page."""
    articles = []
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id={symbol}&duression=Y"
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        if resp.status_code == 200:
            from html.parser import HTMLParser

            class MCHTMLParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.articles = []
                    self._in_title = False
                    self._current = {}

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "a" and "news_title" in attrs_dict.get("class", ""):
                        self._in_title = True
                        self._current = {"href": attrs_dict.get("href", "")}

                def handle_data(self, data):
                    if self._in_title:
                        self._current["title"] = data.strip()

                def handle_endtag(self, tag):
                    if tag == "a" and self._in_title:
                        self._in_title = False
                        if self._current.get("title"):
                            self.articles.append({
                                "title": self._current["title"],
                                "summary": "",
                                "source": "MoneyControl",
                                "time": pd.NaT,
                            })

            parser = MCHTMLParser()
            parser.feed(resp.text)
            articles = parser.articles[:max_articles]
    except Exception as e:
        logger.debug("MoneyControl fetch failed for %s: %s", ticker, e)
    return articles


def _fetch_economic_times(ticker: str, max_articles: int = 10) -> list:
    """Fetch from Economic Times search."""
    articles = []
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://economictimes.indiatimes.com/topic/{symbol.lower()}"
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        if resp.status_code == 200:
            import re
            # Extract article titles from ET page
            titles = re.findall(r'<a[^>]*href="(/[^"]*)"[^>]*>([^<]+)</a>', resp.text)
            seen = set()
            for href, title in titles:
                title = title.strip()
                if len(title) > 20 and title not in seen and symbol.lower() in title.lower():
                    seen.add(title)
                    articles.append({
                        "title": title,
                        "summary": "",
                        "source": "Economic Times",
                        "time": pd.NaT,
                    })
                    if len(articles) >= max_articles:
                        break
    except Exception as e:
        logger.debug("ET fetch failed for %s: %s", ticker, e)
    return articles


def _fetch_screener(ticker: str, max_articles: int = 10) -> list:
    """Fetch from Screener.in news page."""
    articles = []
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    url = f"https://www.screener.in/company/{symbol}/news/"
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        if resp.status_code == 200:
            import re
            titles = re.findall(r'<li[^>]*>.*?<a[^>]*>([^<]+)</a>', resp.text, re.DOTALL)
            for title in titles:
                title = title.strip()
                if len(title) > 15:
                    articles.append({
                        "title": title,
                        "summary": "",
                        "source": "Screener.in",
                        "time": pd.NaT,
                    })
                    if len(articles) >= max_articles:
                        break
    except Exception as e:
        logger.debug("Screener fetch failed for %s: %s", ticker, e)
    return articles


def _deduplicate(articles: list) -> list:
    """Remove duplicate articles by title similarity."""
    seen = set()
    unique = []
    for a in articles:
        key = a["title"][:60].lower().strip()
        if key not in seen and len(key) > 10:
            seen.add(key)
            unique.append(a)
    return unique


# ---------------------------------------------------------------------------
# Multi-source aggregation
# ---------------------------------------------------------------------------

def fetch_news_headlines(ticker: str, max_articles: int = 30) -> list:
    """Fetch news from all sources and deduplicate.

    Returns list of article dicts with title, summary, source, time.
    """
    all_articles = []

    all_articles.extend(_fetch_yahoo_news(ticker, max_articles=15))
    all_articles.extend(_fetch_google_news(ticker, max_articles=15))
    all_articles.extend(_fetch_moneycontrol(ticker, max_articles=10))
    all_articles.extend(_fetch_economic_times(ticker, max_articles=10))
    all_articles.extend(_fetch_screener(ticker, max_articles=10))

    return _deduplicate(all_articles)[:max_articles]


# ---------------------------------------------------------------------------
# Keyword-based sentiment fallback (when FinBERT unavailable)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "surge", "rally", "gain", "profit", "bull", "rise", "jump", "high",
    "record", "growth", "strong", "upgrade", "outperform", "buy", "boost",
    "dividend", "expansion", "recovery", "optimism", "beat", "exceed",
}
_NEGATIVE_WORDS = {
    "crash", "loss", "bear", "fall", "drop", "decline", "plunge", "low",
    "weak", "downgrade", "underperform", "sell", "fear", "risk", "debt",
    "recession", "slowdown", "warning", "miss", "lawsuit", "fraud",
}


def _keyword_sentiment(articles: list) -> dict:
    """Simple keyword-based sentiment analysis."""
    pos = 0
    neg = 0
    neu = 0
    scores = []

    for article in articles:
        title = (article.get("title", "") + " " + article.get("summary", "")).lower()
        words = set(title.split())
        p = len(words & _POSITIVE_WORDS)
        n = len(words & _NEGATIVE_WORDS)
        if p > n:
            val = min(1.0, p * 0.2)
            pos += 1
        elif n > p:
            val = -min(1.0, n * 0.2)
            neg += 1
        else:
            val = 0
            neu += 1
        scores.append(val)

    avg = float(np.mean(scores)) if scores else 0.0
    label = "Positive" if avg > 0.1 else ("Negative" if avg < -0.1 else "Neutral")

    return {
        "score": round(avg, 4),
        "weighted_score": round(avg, 4),
        "label": label,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total": len(articles),
        "source_breakdown": {},
        "sources_used": ["keyword"],
    }


def analyze_sentiment(articles: list) -> dict:
    """Run FinBERT sentiment analysis on articles.

    Returns dict with aggregate score, label, per-source breakdown,
    and source-weighted score.
    """
    if not articles:
        return {
            "score": 0.0, "label": "Neutral",
            "positive": 0, "negative": 0, "neutral": 0,
            "source_breakdown": {},
            "weighted_score": 0.0,
        }

    try:
        pipe = get_finbert()
        texts = [a["title"] for a in articles if a.get("title")]
        if not texts:
            return {
                "score": 0.0, "label": "Neutral",
                "positive": 0, "negative": 0, "neutral": 0,
                "source_breakdown": {},
                "weighted_score": 0.0,
            }

        # Keyword fallback if FinBERT unavailable
        if pipe == "fallback":
            return _keyword_sentiment(articles)

        results = pipe(texts)

        scores = []
        pos = 0
        neg = 0
        neu = 0
        source_scores = {}
        source_counts = {}

        for article, result in zip(articles, results):
            label = result["label"]
            score = result["score"]
            source = article.get("source", "Unknown")

            if label == "positive":
                val = score
                pos += 1
            elif label == "negative":
                val = -score
                neg += 1
            else:
                val = 0
                neu += 1

            scores.append(val)

            if source not in source_scores:
                source_scores[source] = []
                source_counts[source] = 0
            source_scores[source].append(val)
            source_counts[source] += 1

        avg_score = float(np.mean(scores)) if scores else 0.0

        # Source-weighted score
        weighted_scores = []
        for source, s_scores in source_scores.items():
            weight = SOURCE_WEIGHTS.get(source, 0.5)
            source_avg = float(np.mean(s_scores))
            weighted_scores.append(source_avg * weight)
        weighted_score = float(np.mean(weighted_scores)) if weighted_scores else avg_score

        # Determine label from weighted score
        if weighted_score > 0.15:
            label = "Positive"
        elif weighted_score < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        # Source breakdown
        source_breakdown = {}
        for source, s_scores in source_scores.items():
            source_breakdown[source] = {
                "count": source_counts[source],
                "avg_score": round(float(np.mean(s_scores)), 4),
                "weight": SOURCE_WEIGHTS.get(source, 0.5),
            }

        return {
            "score": round(avg_score, 4),
            "weighted_score": round(weighted_score, 4),
            "label": label,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "total": len(texts),
            "source_breakdown": source_breakdown,
            "sources_used": list(source_scores.keys()),
        }
    except Exception as e:
        logger.error("Sentiment analysis failed: %s", e)
        return {
            "score": 0.0, "weighted_score": 0.0, "label": "Neutral",
            "positive": 0, "negative": 0, "neutral": 0,
            "source_breakdown": {}, "error": str(e),
        }


def get_stock_sentiment(ticker: str) -> dict:
    """Main entry point: fetch from all sources + analyze with FinBERT.

    Uses 30-min disk cache. Returns comprehensive sentiment dict.
    """
    cache_file = os.path.join(
        _sentiment_cache_dir, f"sentiment_{ticker.replace('.', '_')}.json"
    )

    # Check cache
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 1800:
            import json
            with open(cache_file) as f:
                return json.load(f)

    articles = fetch_news_headlines(ticker)
    result = analyze_sentiment(articles)
    result["ticker"] = ticker
    result["articles"] = [
        {"title": a["title"], "source": a["source"]}
        for a in articles[:8]
    ]

    # Persist cache
    try:
        import json
        os.makedirs(_sentiment_cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, default=str)
    except Exception:
        pass

    return result
