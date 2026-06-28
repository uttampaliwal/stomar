import pandas as pd
import numpy as np
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    df["high_low_pct"] = (high - low) / close * 100
    df["close_open_pct"] = (close - df["open"]) / df["open"] * 100

    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)

    df["volatility_10d"] = df["returns_1d"].rolling(10).std()
    df["volatility_20d"] = df["returns_1d"].rolling(20).std()

    df["target"] = close.shift(-1) / close - 1
    df["target_direction"] = (df["target"] > 0).astype(int)

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
