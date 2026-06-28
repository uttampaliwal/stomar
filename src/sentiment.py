import yfinance as yf
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

_sentiment_pipeline = None


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            max_length=128,
            truncation=True,
        )
    return _sentiment_pipeline


def fetch_news_sentiment(ticker: str, max_articles: int = 20) -> float:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0.0

        articles = news[:max_articles]
        texts = []
        for article in articles:
            title = article.get("title", "")
            summary = article.get("summary", "")
            texts.append(f"{title}. {summary}" if summary else title)

        if not texts:
            return 0.0

        pipe = get_sentiment_pipeline()
        results = pipe(texts)

        scores = []
        for r in results:
            score = r["score"]
            if r["label"] == "NEGATIVE":
                score = -score
            scores.append(score)

        return float(np.mean(scores))
    except Exception:
        return 0.0


def get_sentiment_for_date_range(ticker: str, days: int = 30) -> pd.Series:
    scores = {}
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return pd.Series(dtype=float)

        pipe = get_sentiment_pipeline()
        for article in news[:50]:
            title = article.get("title", "")
            if not title:
                continue

            pub_time = article.get("providerPublishTime")
            if pub_time is None:
                continue

            date = pd.Timestamp.fromtimestamp(pub_time).normalize()
            result = pipe(title)[0]
            score = result["score"]
            if result["label"] == "NEGATIVE":
                score = -score
            scores[date] = scores.get(date, []) + [score]

        daily_avg = {d: np.mean(s) for d, s in scores.items()}
        series = pd.Series(daily_avg).sort_index()
        return series
    except Exception:
        return pd.Series(dtype=float)
