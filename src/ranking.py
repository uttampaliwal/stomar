"""Cross-sectional stock ranking module.

Ranks stocks relative to each other using multiple factors:
- Momentum (short, medium, long-term)
- Fundamental (P/E, P/B, ROCE, dividend yield, debt-to-equity)
- Quality (ROE, profit margin, debt-to-equity)
- Volatility (lower is better for risk-adjusted)
- Volume (relative volume as liquidity proxy)
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fundamental data cache (in-memory, per-session)
_fundamental_cache: dict[str, dict] = {}


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data for a stock via yfinance.

    Returns dict with pe_ratio, pb_ratio, roce, roe, dividend_yield,
    debt_to_equity, market_cap, profit_margin.

    Falls back to 0.0 for unavailable fields.
    """
    if ticker in _fundamental_cache:
        return _fundamental_cache[ticker]

    defaults = {
        "pe_ratio": 0.0, "pb_ratio": 0.0, "roce": 0.0, "roe": 0.0,
        "dividend_yield": 0.0, "debt_to_equity": 0.0, "market_cap": 0.0,
        "profit_margin": 0.0,
    }

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "pe_ratio": float(info.get("trailingPE", 0.0) or 0.0),
            "pb_ratio": float(info.get("priceToBook", 0.0) or 0.0),
            "roce": float(info.get("returnOnCapitalEmployed", 0.0) or 0.0),
            "roe": float(info.get("returnOnEquity", 0.0) or 0.0),
            "dividend_yield": float(info.get("dividendYield", 0.0) or 0.0),
            "debt_to_equity": float(info.get("debtToEquity", 0.0) or 0.0),
            "market_cap": float(info.get("marketCap", 0.0) or 0.0),
            "profit_margin": float(info.get("profitMargins", 0.0) or 0.0),
        }

        _fundamental_cache[ticker] = result
        return result

    except Exception as e:
        logger.warning("Failed to fetch fundamentals for %s: %s", ticker, e)
        _fundamental_cache[ticker] = defaults
        return defaults


def clear_fundamental_cache():
    """Clear the in-memory fundamental cache."""
    _fundamental_cache.clear()


def fundamental_score(fundamentals: dict) -> float:
    """Score a stock on fundamental factors (0–100 scale).

    Positive factors (higher = better): roe, roce, dividend_yield, profit_margin
    Negative factors (lower = better): pe_ratio, pb_ratio, debt_to_equity

    Returns composite fundamental score 0–100.
    """
    scores = []

    # P/E: lower is better (but > 0). Score 100 at PE=5, 0 at PE=60
    pe = fundamentals.get("pe_ratio", 0.0)
    if pe > 0:
        pe_score = max(0, min(100, (60 - pe) / 55 * 100))
        scores.append(("pe", pe_score, 0.15))

    # P/B: lower is better. Score 100 at PB=0.5, 0 at PB=10
    pb = fundamentals.get("pb_ratio", 0.0)
    if pb > 0:
        pb_score = max(0, min(100, (10 - pb) / 9.5 * 100))
        scores.append(("pb", pb_score, 0.10))

    # ROE: higher is better. Score 0 at 0%, 100 at 30%
    roe = fundamentals.get("roe", 0.0)
    if isinstance(roe, (int, float)) and np.isfinite(roe):
        roe_pct = roe * 100 if abs(roe) <= 1 else roe
        roe_score = max(0, min(100, roe_pct / 30 * 100))
        scores.append(("roe", roe_score, 0.20))

    # ROCE: higher is better. Score 0 at 0%, 100 at 30%
    roce = fundamentals.get("roce", 0.0)
    if isinstance(roce, (int, float)) and np.isfinite(roce):
        roce_pct = roce * 100 if abs(roce) <= 1 else roce
        roce_score = max(0, min(100, roce_pct / 30 * 100))
        scores.append(("roce", roce_score, 0.15))

    # Dividend yield: higher is better. Score 0 at 0%, 100 at 5%
    div = fundamentals.get("dividend_yield", 0.0)
    if isinstance(div, (int, float)) and np.isfinite(div):
        div_pct = div * 100 if abs(div) <= 1 else div
        div_score = max(0, min(100, div_pct / 5 * 100))
        scores.append(("dividend", div_score, 0.10))

    # Debt-to-equity: lower is better. Score 100 at 0, 0 at 200
    dte = fundamentals.get("debt_to_equity", 0.0)
    if isinstance(dte, (int, float)) and np.isfinite(dte):
        dte_score = max(0, min(100, (200 - dte) / 200 * 100))
        scores.append(("debt", dte_score, 0.15))

    # Profit margin: higher is better. Score 0 at 0%, 100 at 30%
    pm = fundamentals.get("profit_margin", 0.0)
    if isinstance(pm, (int, float)) and np.isfinite(pm):
        pm_pct = pm * 100 if abs(pm) <= 1 else pm
        pm_score = max(0, min(100, pm_pct / 30 * 100))
        scores.append(("profit_margin", pm_score, 0.15))

    if not scores:
        return 50.0  # neutral if no data

    total_weight = sum(w for _, _, w in scores)
    return sum(s * w for _, s, w in scores) / total_weight


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


def rank_stocks(stock_data, weights=None, use_fundamentals=False, ml_signals=None):
    """Rank stocks using multi-factor model.

    Args:
        stock_data: Dict of {ticker: DataFrame} with OHLCV data
        weights: Dict of factor weights (momentum, volatility, volume, technical, fundamental, ml_signal)
        use_fundamentals: If True, fetch and include fundamental scores
        ml_signals: Dict of {ticker: float} ML/meta-controller signals (-1.0 to +1.0).
            Positive = bullish, negative = bearish. If provided, adds ml_signal factor.

    Returns:
        List of ranking dicts sorted by composite score
    """
    has_ml = ml_signals and len(ml_signals) > 0

    if weights is None:
        if has_ml:
            weights = {
                "momentum": 0.25,
                "volatility": 0.15,
                "volume": 0.10,
                "technical": 0.35,
                "ml_signal": 0.15,
            }
        else:
            weights = {
                "momentum": 0.3,
                "volatility": 0.2,
                "volume": 0.1,
                "technical": 0.4,
            }

    if use_fundamentals and "fundamental" not in weights:
        # Redistribute weights to include fundamental
        n = len(weights) + 1
        base = 1.0 / n
        weights = {**weights, "fundamental": base}
        # Scale others down proportionally
        scale = (1.0 - base) / sum(v for k, v in weights.items() if k != "fundamental")
        weights = {k: v * scale if k != "fundamental" else base for k, v in weights.items()}

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

        ml_score = 50.0
        if has_ml and ticker in ml_signals:
            # Convert ML signal (-1 to +1) to score (0 to 100)
            ml_signal_val = ml_signals[ticker]
            ml_score = float(max(0, min(100, (ml_signal_val + 1) / 2 * 100)))
            composite += ml_score / 100 * weights.get("ml_signal", 0)

        fund_score = 50.0
        fund_data = {}
        if use_fundamentals:
            fund_data = fetch_fundamentals(ticker)
            fund_score = fundamental_score(fund_data)
            composite += fund_score / 100 * weights["fundamental"]

        rankings.append({
            "ticker": ticker,
            "composite_score": float(composite),
            "momentum": mom,
            "volatility": vol,
            "volume": vol_score,
            "technical": tech,
            "ml_score": float(ml_score),
            "fundamental_score": float(fund_score),
            "fundamentals": fund_data,
            "current_price": float(close.iloc[-1]),
            "daily_return": float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0,
        })

    rankings.sort(key=lambda x: x["composite_score"], reverse=True)

    for i, r in enumerate(rankings):
        r["rank"] = i + 1
        r["percentile"] = float((len(rankings) - i) / len(rankings) * 100) if rankings else 50

    return rankings


def get_recommendation(rankings, ticker):
    """Get buy/hold/sell recommendation for a stock based on rank and ML signal.

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
    ml_score = stock.get("ml_score", 50.0)

    # ML signal direction overrides percentile if strong enough
    ml_direction = "BUY" if ml_score > 60 else ("SELL" if ml_score < 40 else "NEUTRAL")

    if ml_direction == "BUY" and percentile >= 40:
        # ML agrees with above-average ranking
        rec = "STRONG BUY"
        reasoning = f"Rank #{rank}/{total} + ML bullish ({ml_score:.0f})"
    elif ml_direction == "SELL" and percentile <= 60:
        # ML agrees with below-average ranking
        rec = "STRONG SELL"
        reasoning = f"Rank #{rank}/{total} + ML bearish ({ml_score:.0f})"
    elif ml_direction == "BUY" and percentile < 40:
        # ML bullish but ranking is low — conflicting
        rec = "BUY"
        reasoning = f"ML bullish ({ml_score:.0f}) despite rank #{rank}/{total}"
    elif ml_direction == "SELL" and percentile > 60:
        # ML bearish but ranking is high — conflicting
        rec = "SELL"
        reasoning = f"ML bearish ({ml_score:.0f}) despite rank #{rank}/{total}"
    elif percentile >= 80:
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
        "ml_score": ml_score,
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
