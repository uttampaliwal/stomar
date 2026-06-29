import pandas as pd
import numpy as np
import ta
import os
import json
import logging

logger = logging.getLogger(__name__)


def add_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add sentiment features with point-in-time discipline.

    The sentiment score from today's news is only assigned to the most recent
    row. Historical rows receive 0.0 because that sentiment was not yet
    available on those dates. This prevents look-ahead bias in backtests.
    """
    df = df.copy()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    cache_file = os.path.join(cache_dir, f"sentiment_{ticker.replace('.','_')}.json")

    score = 0.0
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            score = data.get("score", 0.0)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read sentiment cache for %s: %s", ticker, e)

    # Point-in-time: only the most recent row gets the score.
    # Historical rows stay 0.0 because this sentiment was not available then.
    df["sentiment_score"] = 0.0
    if len(df) > 0:
        df.iloc[-1, df.columns.get_loc("sentiment_score")] = score
    return df


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FII/DII flow features with point-in-time discipline.

    FII/DII data for day T is only available after market close on day T,
    so it should only be joined to day T+1's price. We achieve this by
    joining on date and then shifting by 1 day.
    """
    df = df.copy()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    cache_file = os.path.join(cache_dir, "fii_dii.parquet")

    # Default: all zeros
    df["fii_net"] = 0.0
    df["dii_net"] = 0.0
    df["flow_signal"] = 0.0

    if not os.path.exists(cache_file):
        return df

    try:
        flow_df = pd.read_parquet(cache_file)
        if len(flow_df) == 0:
            return df

        flow_df["date"] = pd.to_datetime(flow_df["date"])
        flow_df = flow_df.set_index("date").sort_index()

        # Keep only the columns we need
        flow_df = flow_df[["fii_net", "dii_net"]]

        # Shift by 1 day: FII/DII data for day T joins to day T+1
        flow_df = flow_df.shift(1)

        # Join on date index (left join preserves all price dates)
        df = df.join(flow_df, how="left")

        # Fill missing dates with 0
        df["fii_net"] = df["fii_net"].fillna(0.0)
        df["dii_net"] = df["dii_net"].fillna(0.0)
        df["flow_signal"] = df.apply(
            lambda r: 1.0 if r["fii_net"] > 0 and r["dii_net"] > 0
            else (-1.0 if r["fii_net"] < -500 else 0.0),
            axis=1,
        )
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Failed to load flow features: %s", e)

    return df


def add_pcr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add options PCR features with point-in-time discipline.

    PCR is computed during market hours and only available after market close.
    Assign only to the most recent row to prevent look-ahead bias.
    """
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
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read PCR cache: %s", e)

    # Point-in-time: only the most recent row gets the PCR value.
    # Historical rows stay at defaults because PCR was not available then.
    df["pcr"] = 1.0
    df["max_pain"] = 0.0
    if len(df) > 0:
        df.iloc[-1, df.columns.get_loc("pcr")] = pcr
        df.iloc[-1, df.columns.get_loc("max_pain")] = max_pain
    return df


def add_multitimeframe_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add multi-timeframe features with point-in-time discipline.

    MTF signals are computed from current data. Assign only to the most
    recent row to prevent look-ahead bias in backtests.
    """
    df = df.copy()
    try:
        from src.multitimeframe import fetch_mtf_data, get_combined_signal
        mtf = fetch_mtf_data(ticker)
        if mtf:
            signal = get_combined_signal(mtf)
            mtf_dir = signal["direction"]
            mtf_conf = signal["confidence"]
        else:
            mtf_dir = 0
            mtf_conf = 0
    except Exception as e:
        logger.warning("Failed to compute MTF features for %s: %s", ticker, e)
        mtf_dir = 0
        mtf_conf = 0

    # Point-in-time: only the most recent row gets the MTF signal.
    # Historical rows stay at 0 because this signal was not available then.
    df["mtf_signal"] = 0.0
    df["mtf_confidence"] = 0.0
    if len(df) > 0:
        df.iloc[-1, df.columns.get_loc("mtf_signal")] = float(mtf_dir)
        df.iloc[-1, df.columns.get_loc("mtf_confidence")] = float(mtf_conf)
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

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)

    # CCI
    df["cci"] = ta.trend.cci(high, low, close, window=20)

    # MFI (Money Flow Index)
    df["mfi"] = ta.volume.money_flow_index(high, low, close, volume, window=14)

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()

    # VWAP (intraday approximation using daily data)
    typical_price = (high + low + close) / 3
    df["vwap"] = (typical_price * volume).cumsum() / volume.cumsum()

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

    # Interaction features
    df["rsi_x_volume"] = df["rsi"] * df["volume_ratio"]
    df["macd_x_bb_width"] = df["macd"] * df["bb_width"]
    df["adx_x_volatility"] = df["adx"] * df["volatility_20d"]
    df["momentum_x_vol"] = df["returns_5d"] * df["volatility_10d"]

    # Rolling correlation of returns and volume
    df["return_vol_corr"] = df["returns_1d"].rolling(20).corr(df["volume_ratio"])

    # Rolling skewness and kurtosis of returns
    df["return_skew"] = df["returns_1d"].rolling(20).skew()
    df["return_kurt"] = df["returns_1d"].rolling(20).kurt()

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
    df: pd.DataFrame, feature_cols: list, seq_length: int = 60,
    train_ratio: float = 0.8,
):
    """Prepare LSTM data with proper train/test split to prevent scaler leakage.

    The scaler is fit ONLY on the training portion of the data, then applied
    to both train and test. This prevents information from the test set
    leaking into the training preprocessing.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        seq_length: Sequence length for LSTM input
        train_ratio: Fraction of data to use for training (default 0.8)

    Returns:
        (X, y, scaler) where X is (N, seq_length, n_features), y is (N,),
        and scaler is fitted on training data only.
    """
    from sklearn.preprocessing import MinMaxScaler

    data = df[feature_cols].dropna().values

    # Split into train/test BEFORE fitting scaler
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    # Transform all data with the training-fitted scaler
    scaled = scaler.transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length : i])
        y.append(scaled[i, 0])

    return np.array(X), np.array(y), scaler
