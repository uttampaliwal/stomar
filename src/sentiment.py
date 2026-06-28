import yfinance as yf
import numpy as np
import pandas as pd
import warnings
import os
import time
import requests

warnings.filterwarnings("ignore")

_finbert_pipeline = None
_sentiment_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
    return _finbert_pipeline


def fetch_news_headlines(ticker: str, max_articles: int = 25) -> list:
    articles = []
    symbol = ticker.replace(".NS", "").replace(".BO", "")

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
                        "source": source,
                        "time": pd.Timestamp.fromtimestamp(pub_time) if pub_time else pd.NaT,
                    })
    except Exception:
        pass

    if len(articles) < 3:
        rss_urls = [
            f"https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-IN&gl=IN&ceid=IN:en",
        ]
        try:
            import xml.etree.ElementTree as ET
            for rss_url in rss_urls:
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
        except Exception:
            pass

    seen = set()
    unique = []
    for a in articles:
        key = a["title"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique[:max_articles]


def analyze_sentiment(articles: list) -> dict:
    if not articles:
        return {"score": 0.0, "label": "Neutral", "positive": 0, "negative": 0, "neutral": 0}

    try:
        pipe = get_finbert()
        texts = [a["title"] for a in articles if a.get("title")]
        if not texts:
            return {"score": 0.0, "label": "Neutral", "positive": 0, "negative": 0, "neutral": 0}

        results = pipe(texts)

        scores = []
        pos = 0
        neg = 0
        neu = 0
        for r in results:
            label = r["label"]
            score = r["score"]
            if label == "positive":
                scores.append(score)
                pos += 1
            elif label == "negative":
                scores.append(-score)
                neg += 1
            else:
                scores.append(0)
                neu += 1

        avg_score = float(np.mean(scores)) if scores else 0.0
        if avg_score > 0.15:
            label = "Positive"
        elif avg_score < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "score": round(avg_score, 4),
            "label": label,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "total": len(texts),
        }
    except Exception as e:
        return {"score": 0.0, "label": "Neutral", "positive": 0, "negative": 0, "neutral": 0, "error": str(e)}


def get_stock_sentiment(ticker: str) -> dict:
    cache_file = os.path.join(_sentiment_cache_dir, f"sentiment_{ticker.replace('.', '_')}.json")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 1800:
            import json
            with open(cache_file) as f:
                return json.load(f)

    articles = fetch_news_headlines(ticker)
    result = analyze_sentiment(articles)
    result["ticker"] = ticker
    result["articles"] = articles[:5]

    try:
        import json
        with open(cache_file, "w") as f:
            json.dump(result, f, default=str)
    except Exception:
        pass

    return result
