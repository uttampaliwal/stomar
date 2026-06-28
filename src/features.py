import pandas as pd
import numpy as np
import ta
import os
import time
import json


def add_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    cache_file = os.path.join(cache_dir, f"sentiment_{ticker.replace('.','_')}.json")

    score = 0.0
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            score = data.get("score", 0.0)
        except Exception:
            pass

    df["sentiment_score"] = score
    return df


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    cache_file = os.path.join(cache_dir, "fii_dii.parquet")

    fii_net = 0.0
    dii_net = 0.0
    if os.path.exists(cache_file):
        try:
            flow_df = pd.read_parquet(cache_file)
            if len(flow_df) > 0:
                fii_net = float(flow_df.iloc[0]["fii_net"])
                dii_net = float(flow_df.iloc[0]["dii_net"])
        except Exception:
            pass

    df["fii_net"] = fii_net
    df["dii_net"] = dii_net
    df["flow_signal"] = 1.0 if fii_net > 0 and dii_net > 0 else (-1.0 if fii_net < -500 else 0.0)
    return df


def add_pcr_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    cache_file = os.path.join(cache_dir, "options_pcr.json")

    pcr = 1.0
    max_pain = 0.0
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            pcr = data.get("pcr_oi", 1.0)
            max_pain = data.get("max_pain", 0.0)
        except Exception:
            pass

    df["pcr"] = pcr
    df["max_pain"] = max_pain
    return df


def add_multitimeframe_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    try:
        from src.multitimeframe import fetch_mtf_data, get_combined_signal
        mtf = fetch_mtf_data(ticker)
        if mtf:
            signal = get_combined_signal(mtf)
            df["mtf_signal"] = signal["direction"]
            df["mtf_confidence"] = signal["confidence"]
        else:
            df["mtf_signal"] = 0
            df["mtf_confidence"] = 0
    except Exception:
        df["mtf_signal"] = 0
        df["mtf_confidence"] = 0
    return df


def add_technical_indicators(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["sma_10"] = ta.trend.sma_indicator(close, window=10)
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)
    df["rsi"] = ta.momentum.rsi(close, window=14)
    df["macd"] = ta.trend.macd_diff(close)
    df["macd_signal"] = ta.trend.macd_signal(close)
    df["bb_high"] = ta.volatility.bollinger_hband(close)
    df["bb_low"] = ta.volatility.bollinger_lband(close)
    df["bb_width"] = df["bb_high"] - df["bb_low"]
    df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["volume_sma"] = ta.trend.sma_indicator(volume, window=20)
    df["volume_ratio"] = volume / df["volume_sma"].replace(0, np.nan)

    # Price position in range
    df["high_low_pct"] = (high - low) / close * 100
    df["close_open_pct"] = (close - df["open"]) / df["open"] * 100
    df["close_position"] = (close - low) / (high - low + 1e-8)

    # Returns at multiple horizons
    for d in [1, 2, 3, 5, 10, 20]:
        df[f"returns_{d}d"] = close.pct_change(d)

    # Volatility
    for d in [5, 10, 20]:
        df[f"volatility_{d}d"] = df["returns_1d"].rolling(d).std()

    # Lagged returns
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["returns_1d"].shift(lag)

    # Calendar features
    if df.index.dtype == "datetime64[ns]" or isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["day_of_month"] = df.index.day

    # Target
    df["target"] = close.shift(-1) / close - 1
    df["target_direction"] = (df["target"] > 0).astype(int)

    if ticker:
        df = add_sentiment_features(df, ticker)
        df = add_flow_features(df)
        df = add_pcr_features(df)
        df = add_multitimeframe_features(df, ticker)

    return df


def prepare_lstm_data(
    df: pd.DataFrame, feature_cols: list, seq_length: int = 60
):
    from sklearn.preprocessing import MinMaxScaler

    data = df[feature_cols].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length : i])
        y.append(scaled[i, 0])

    return np.array(X), np.array(y), scaler
