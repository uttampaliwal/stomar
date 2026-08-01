import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import time
import logging

from src.core.constants import DATA_DIR

logger = logging.getLogger(__name__)

MTF_CACHE_DIR = DATA_DIR
CACHE_VERSION = 1  # Bump to invalidate old caches
os.makedirs(MTF_CACHE_DIR, exist_ok=True)


def fetch_mtf_data(ticker: str) -> dict:
    return _fetch_mtf_data_impl(ticker)


def _fetch_mtf_data_impl(ticker: str) -> dict:
    ticker_clean = ticker.replace('.', '_')
    cache_prefix = os.path.join(MTF_CACHE_DIR, f"mtf_{ticker_clean}")

    # Try loading from parquet cache (one file per timeframe)
    cache_files = {tf: f"{cache_prefix}_{tf}.parquet" for tf in ("15m", "1h", "daily", "weekly")}
    all_exist = all(os.path.exists(f) for f in cache_files.values())
    if all_exist:
        youngest = max(os.path.getmtime(f) for f in cache_files.values())
        if time.time() - youngest < 3600:
            try:
                return {tf: pd.read_parquet(path) for tf, path in cache_files.items()}
            except Exception:
                pass  # Corrupted cache, re-fetch

    # NOTE: legacy mtf_*.pkl pickle caches are deliberately NOT loaded here.
    # pd.read_pickle() on untrusted files is a deserialization risk. Legacy
    # caches are migrated once by scripts/migrate_legacy_models.py; until
    # then the data is simply re-fetched.

    data = {}
    intervals = {
        "15m": {"period": "60d", "interval": "15m"},
        "1h": {"period": "730d", "interval": "1h"},
        "daily": {"period": "5y", "interval": "1d"},
        "weekly": {"period": "10y", "interval": "1wk"},
    }

    stock = yf.Ticker(ticker)
    for tf, params in intervals.items():
        try:
            df = stock.history(period=params["period"], interval=params["interval"])
            if df is not None and len(df) > 20:
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                data[tf] = _add_tf_indicators(df)
        except Exception:
            continue

    if data:
        _save_mtf_cache(cache_prefix, data)
    return data


def _save_mtf_cache(cache_prefix: str, data: dict) -> None:
    """Save MTF data as individual parquet files per timeframe."""
    try:
        for tf, df in data.items():
            df.to_parquet(f"{cache_prefix}_{tf}.parquet")
    except Exception:
        pass


def _add_tf_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    df["sma_10"] = ta.trend.sma_indicator(c, window=10)
    df["sma_20"] = ta.trend.sma_indicator(c, window=20)
    df["sma_50"] = ta.trend.sma_indicator(c, window=50)
    df["ema_12"] = ta.trend.ema_indicator(c, window=12)
    df["ema_26"] = ta.trend.ema_indicator(c, window=26)
    df["rsi"] = ta.momentum.rsi(c, window=14)
    df["macd"] = ta.trend.macd_diff(c)
    df["macd_signal"] = ta.trend.macd_signal(c)
    df["bb_high"] = ta.volatility.bollinger_hband(c)
    df["bb_low"] = ta.volatility.bollinger_lband(c)
    df["atr"] = ta.volatility.average_true_range(h, lo, c, window=14)
    df["adx"] = ta.trend.adx(h, lo, c, window=14)
    df["stoch_k"] = ta.momentum.stoch(h, lo, c)
    df["stoch_d"] = ta.momentum.stoch_signal(h, lo, c)
    vol_sum = v.rolling(20).sum()
    df["vwap"] = ((v * (h + lo + c) / 3).rolling(20).sum() / vol_sum.replace(0, np.nan))
    df["obv"] = ta.volume.on_balance_volume(c, v)
    df["returns"] = c.pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    return df


def get_tf_signal(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 20:
        return {"direction": 0, "strength": 0, "signal": "Neutral", "indicators": {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    close = latest["close"]

    bullish = 0
    total = 0
    indicators = {}

    if "sma_10" in df.columns and pd.notna(latest.get("sma_10")) and pd.notna(latest.get("sma_20")):
        total += 1
        if latest["sma_10"] > latest["sma_20"]:
            bullish += 1
        indicators["SMA Cross"] = "Bullish" if latest["sma_10"] > latest["sma_20"] else "Bearish"

    if "ema_12" in df.columns and pd.notna(latest.get("ema_12")) and pd.notna(latest.get("ema_26")):
        total += 1
        if latest["ema_12"] > latest["ema_26"]:
            bullish += 1
        indicators["EMA Cross"] = "Bullish" if latest["ema_12"] > latest["ema_26"] else "Bearish"

    if "rsi" in df.columns and pd.notna(latest.get("rsi")):
        total += 1
        if latest["rsi"] < 30:
            bullish += 1
            indicators["RSI"] = f"Oversold ({latest['rsi']:.0f})"
        elif latest["rsi"] > 70:
            indicators["RSI"] = f"Overbought ({latest['rsi']:.0f})"
        else:
            bullish += 0.5
            indicators["RSI"] = f"Neutral ({latest['rsi']:.0f})"

    if "macd" in df.columns and pd.notna(latest.get("macd")) and pd.notna(latest.get("macd_signal")):
        total += 1
        if latest["macd"] > latest["macd_signal"]:
            bullish += 1
        indicators["MACD"] = "Bullish" if latest["macd"] > latest["macd_signal"] else "Bearish"

    if "stoch_k" in df.columns and pd.notna(latest.get("stoch_k")) and pd.notna(latest.get("stoch_d")):
        total += 1
        if latest["stoch_k"] > latest["stoch_d"]:
            bullish += 1
        indicators["Stochastic"] = "Bullish" if latest["stoch_k"] > latest["stoch_d"] else "Bearish"

    if "adx" in df.columns and pd.notna(latest.get("adx")):
        total += 1
        if latest["adx"] > 25:
            indicators["ADX"] = f"Strong Trend ({latest['adx']:.0f})"
            bullish += 1 if latest.get("close", 0) > prev.get("close", 0) else 0
        else:
            indicators["ADX"] = f"Weak ({latest['adx']:.0f})"

    if "bb_high" in df.columns and pd.notna(latest.get("bb_high")) and pd.notna(latest.get("bb_low")):
        total += 1
        if close < latest["bb_low"]:
            bullish += 1
            indicators["Bollinger"] = "Below Lower Band"
        elif close > latest["bb_high"]:
            indicators["Bollinger"] = "Above Upper Band"
        else:
            bullish += 0.5
            indicators["Bollinger"] = "Inside Bands"

    if "vwap" in df.columns and pd.notna(latest.get("vwap")):
        total += 1
        if close > latest["vwap"]:
            bullish += 1
        indicators["VWAP"] = "Above" if close > latest["vwap"] else "Below"

    strength = bullish / max(total, 1)
    if strength > 0.6:
        direction = 1
        signal = "Bullish"
    elif strength < 0.4:
        direction = -1
        signal = "Bearish"
    else:
        direction = 0
        signal = "Neutral"

    return {"direction": direction, "strength": round(strength, 3), "signal": signal, "indicators": indicators}


def get_combined_signal(mtf_data: dict) -> dict:
    tf_weights = {"15m": 0.15, "1h": 0.20, "daily": 0.40, "weekly": 0.25}
    weighted_score = 0
    tf_signals = {}

    for tf, weight in tf_weights.items():
        if tf in mtf_data:
            sig = get_tf_signal(mtf_data[tf])
            tf_signals[tf] = sig
            weighted_score += sig["direction"] * sig["strength"] * weight

    if abs(weighted_score) > 0.15:
        final_dir = 1 if weighted_score > 0 else -1
    else:
        final_dir = 0

    labels = {1: "Bullish", -1: "Bearish", 0: "Neutral"}
    confidence = min(abs(weighted_score) / 0.5, 1.0) * 100

    return {
        "direction": final_dir,
        "signal": labels[final_dir],
        "confidence": round(confidence, 1),
        "weighted_score": round(weighted_score, 3),
        "timeframes": tf_signals,
    }
