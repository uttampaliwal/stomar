from api.routers.insights import enrichment


def test_enrichment_returns_summary(monkeypatch):
    def fake_fetch_stock_data(ticker, period, interval, force_refresh=False):
        import pandas as pd
        return pd.DataFrame({"close": [100.0, 101.0]})

    def fake_get_live_price(ticker):
        return 101.5

    def fake_get_stock_sentiment(ticker):
        return {"score": 0.2, "label": "Positive", "total": 3, "articles": [{"title": "test"}]}

    monkeypatch.setattr("api.routers.insights.fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr("api.routers.insights.get_live_price", fake_get_live_price)
    monkeypatch.setattr("api.routers.insights.get_stock_sentiment", fake_get_stock_sentiment)

    result = enrichment("RELIANCE.NS")

    assert result["ticker"] == "RELIANCE.NS"
    assert result["price"] == 101.5
    assert result["change_pct"] == 1.0
    assert result["sentiment_label"] == "Positive"
