"""Cross-sectional stock ranking module.

Ranks stocks relative to each other using multiple factors:
- Momentum (short, medium, long-term)
- Value (P/E, P/B, dividend yield)
- Quality (ROE, profit margin, debt-to-equity)
- Volatility (lower is better for risk-adjusted)
- Volume (relative volume as liquidity proxy)
"""
import numpy as np
import pandas as pd


def compute_momentum_score(close_series, windows=[5, 20, 60, 120]):
    """Compute momentum score at multiple windows.

    Args:
        close_series: Series of close prices indexed by date
        windows: List of lookback windows

    Returns:
        Dict with momentum scores
    """
    scores = {}
    for w in windows:
        if len(close_series) > w:
            ret = (close_series.iloc[-1] / close_series.iloc[-w]) - 1
            scores[f"mom_{w}d"] = float(ret)
        else:
            scores[f"mom_{w}d"] = 0.0

    # Combined momentum: weighted average
    weights = {5: 0.1, 20: 0.2, 60: 0.3, 120: 0.4}
    combined = sum(scores.get(f"mom_{w}d", 0) * wt
                   for w, wt in weights.items() if f"mom_{w}d" in scores)
    scores["momentum_combined"] = float(combined)

    return scores


def compute_volatility_score(returns, window=20):
    """Compute volatility score (lower vol = higher score).

    Args:
        returns: Series of daily returns
        window: Rolling window for vol

    Returns:
        Dict with volatility score
    """
    if len(returns) < window:
        return {"volatility": 0.0, "vol_score": 50.0}

    vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
    return {"volatility": float(vol), "vol_score": float(100 - min(vol * 200, 100))}


def compute_volume_score(volume, window=20):
    """Compute volume score (higher relative volume = higher score).

    Args:
        volume: Series of volume
        window: Rolling window

    Returns:
        Dict with volume score
    """
    if len(volume) < window:
        return {"relative_volume": 1.0, "volume_score": 50.0}

    avg_vol = volume.rolling(window).mean().iloc[-1]
    current_vol = volume.iloc[-1]

    if avg_vol > 0:
        rel_vol = current_vol / avg_vol
    else:
        rel_vol = 1.0

    # Score: 50 for normal, higher for higher volume
    vol_score = min(50 + (rel_vol - 1) * 50, 100)
    vol_score = max(vol_score, 0)

    return {"relative_volume": float(rel_vol), "volume_score": float(vol_score)}


def compute_technical_score(df):
    """Compute technical analysis score.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dict with technical scores
    """
    close = df["close"]
    scores = {}

    # RSI
    if len(close) >= 14:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
        # RSI score: 50 is neutral, higher for oversold (buy signal)
        scores["rsi"] = current_rsi
        scores["rsi_score"] = float(max(0, min(100, 100 - abs(current_rsi - 50) * 2)))
    else:
        scores["rsi"] = 50
        scores["rsi_score"] = 50

    # MACD
    if len(close) >= 26:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_val = float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0
        signal_val = float(signal.iloc[-1]) if not np.isnan(signal.iloc[-1]) else 0
        scores["macd"] = macd_val
        scores["macd_signal"] = signal_val
        scores["macd_score"] = float(60 if macd_val > signal_val else 40)
    else:
        scores["macd"] = 0
        scores["macd_signal"] = 0
        scores["macd_score"] = 50

    # Price vs SMA
    if len(close) >= 50:
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        scores["price_vs_sma20"] = float((current / sma20 - 1) * 100) if sma20 > 0 else 0
        scores["price_vs_sma50"] = float((current / sma50 - 1) * 100) if sma50 > 0 else 0
        # Score: above SMA = good
        scores["sma_score"] = float(60 if current > sma20 and sma20 > sma50 else 40)
    else:
        scores["price_vs_sma20"] = 0
        scores["price_vs_sma50"] = 0
        scores["sma_score"] = 50

    # Bollinger Band position
    if len(close) >= 20:
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_pos = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        scores["bb_position"] = float(bb_pos) if not np.isnan(bb_pos) else 0.5
        # Score: middle is best (mean reversion)
        scores["bb_score"] = float(max(0, 100 - abs(bb_pos - 0.5) * 200))
    else:
        scores["bb_position"] = 0.5
        scores["bb_score"] = 50

    # Combined technical score
    scores["technical_combined"] = float(np.mean([
        scores.get("rsi_score", 50),
        scores.get("macd_score", 50),
        scores.get("sma_score", 50),
        scores.get("bb_score", 50),
    ]))

    return scores


def rank_stocks(stock_data, weights=None):
    """Rank stocks using multi-factor model.

    Args:
        stock_data: Dict of {ticker: DataFrame} with OHLCV data
        weights: Dict of factor weights (momentum, volatility, volume, technical)

    Returns:
        List of ranking dicts sorted by composite score
    """
    if weights is None:
        weights = {
            "momentum": 0.3,
            "volatility": 0.2,
            "volume": 0.1,
            "technical": 0.4,
        }

    rankings = []

    for ticker, df in stock_data.items():
        if len(df) < 60:
            continue

        close = df["close"]
        returns = close.pct_change().dropna()
        volume = df["volume"]

        mom = compute_momentum_score(close)
        vol = compute_volatility_score(returns)
        vol_score = compute_volume_score(volume)
        tech = compute_technical_score(df)

        composite = (
            mom["momentum_combined"] * weights["momentum"] +
            vol["vol_score"] / 100 * weights["volatility"] +
            vol_score["volume_score"] / 100 * weights["volume"] +
            tech["technical_combined"] / 100 * weights["technical"]
        )

        rankings.append({
            "ticker": ticker,
            "composite_score": float(composite),
            "momentum": mom,
            "volatility": vol,
            "volume": vol_score,
            "technical": tech,
            "current_price": float(close.iloc[-1]),
            "daily_return": float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0,
        })

    rankings.sort(key=lambda x: x["composite_score"], reverse=True)

    for i, r in enumerate(rankings):
        r["rank"] = i + 1
        r["percentile"] = float((len(rankings) - i) / len(rankings) * 100) if rankings else 50

    return rankings


def get_recommendation(rankings, ticker):
    """Get buy/hold/sell recommendation for a stock based on rank.

    Args:
        rankings: List of ranking dicts from rank_stocks()
        ticker: Stock ticker to get recommendation for

    Returns:
        Dict with recommendation, rank, and reasoning
    """
    stock = next((r for r in rankings if r["ticker"] == ticker), None)
    if stock is None:
        return {"recommendation": "HOLD", "reasoning": "No data available"}

    rank = stock["rank"]
    total = len(rankings)
    percentile = stock["percentile"]

    if percentile >= 80:
        rec = "STRONG BUY"
        reasoning = f"Top {100 - percentile:.0f}% — strong momentum, good technicals"
    elif percentile >= 60:
        rec = "BUY"
        reasoning = f"Top {100 - percentile:.0f}% — above average across factors"
    elif percentile >= 40:
        rec = "HOLD"
        reasoning = f"Middle {100 - percentile:.0f}% — average performance"
    elif percentile >= 20:
        rec = "SELL"
        reasoning = f"Bottom {percentile:.0f}% — below average factors"
    else:
        rec = "STRONG SELL"
        reasoning = f"Bottom {percentile:.0f}% — weak across all factors"

    return {
        "recommendation": rec,
        "rank": rank,
        "total": total,
        "percentile": percentile,
        "composite_score": stock["composite_score"],
        "reasoning": reasoning,
    }


def factor_analysis(stock_data):
    """Analyze which factors drive returns across stocks.

    Args:
        stock_data: Dict of {ticker: DataFrame}

    Returns:
        Dict with factor correlations and importance
    """
    data = []
    for ticker, df in stock_data.items():
        if len(df) < 60:
            continue
        close = df["close"]
        returns = close.pct_change().dropna()
        mom = compute_momentum_score(close)
        vol = compute_volatility_score(returns)
        tech = compute_technical_score(df)

        data.append({
            "ticker": ticker,
            "return_20d": float((close.iloc[-1] / close.iloc[-min(20, len(close))] - 1)),
            "momentum": mom["momentum_combined"],
            "volatility": vol["volatility"],
            "technical": tech["technical_combined"],
        })

    if len(data) < 3:
        return {"factor_correlations": {}, "insufficient_data": True}

    df = pd.DataFrame(data).set_index("ticker")

    correlations = {}
    for factor in ["momentum", "volatility", "technical"]:
        if factor in df.columns:
            corr = df["return_20d"].corr(df[factor])
            correlations[factor] = float(corr) if not np.isnan(corr) else 0.0

    return {
        "factor_correlations": correlations,
        "best_factor": max(correlations, key=lambda k: abs(correlations[k])) if correlations else "none",
        "n_stocks": len(data),
    }
