import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.model import build_lstm, build_xgb_model, save_models

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "close", "volume", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "bb_width", "atr", "obv", "volume_ratio",
    "high_low_pct", "close_open_pct", "returns_1d", "returns_5d", "returns_20d",
    "volatility_10d", "volatility_20d",
]

SEQ_LENGTH = 60
EPOCHS = 30
BATCH_SIZE = 32


def train_for_ticker(ticker: str, force_retrain: bool = False):
    print(f"\n{'='*50}")
    print(f"Training models for {ticker}")
    print(f"{'='*50}")

    df = fetch_stock_data(ticker, period="3y", force_refresh=force_retrain)
    print(f"Fetched {len(df)} rows of data")

    df_feat = add_technical_indicators(df)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After feature engineering: {len(df_feat)} rows")

    lstm_features = [c for c in FEATURE_COLS if c in df_feat.columns]
    xgb_features = lstm_features + ["target_direction"]

    # --- XGBoost Training ---
    print("\nTraining XGBoost...")
    xgb_data = df_feat[xgb_features].dropna()
    X_xgb = xgb_data[lstm_features]
    y_xgb = xgb_data["target_direction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_xgb, y_xgb, test_size=0.2, shuffle=False
    )

    xgb_model = build_xgb_model()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Test Accuracy: {acc:.4f}")

    # --- LSTM Training ---
    print("\nTraining LSTM...")
    lstm_data = df_feat[lstm_features].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(lstm_data)

    X_lstm, y_lstm = [], []
    for i in range(SEQ_LENGTH, len(scaled)):
        X_lstm.append(scaled[i - SEQ_LENGTH : i])
        y_lstm.append(scaled[i, 0])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

    lstm_model = build_lstm(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
    lstm_model.fit(
        X_train_lstm,
        y_train_lstm,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_lstm, y_test_lstm),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3
            ),
        ],
    )

    lstm_pred = lstm_model.predict(X_test_lstm)
    lstm_direction = (lstm_pred.flatten() > lstm_data[-len(lstm_pred):, 0].mean()).astype(int)
    lstm_actual = (y_test_lstm > scaled[:len(y_test_lstm), 0].mean()).astype(int)
    lstm_acc = accuracy_score(lstm_actual, lstm_direction)
    print(f"LSTM Direction Accuracy: {lstm_acc:.4f}")

    save_models(lstm_model, xgb_model, scaler, lstm_features, ticker)
    print(f"Models saved for {ticker}")

    return {"xgb_accuracy": acc, "lstm_accuracy": lstm_acc}


def train_multiple_stocks(tickers: list, force_retrain: bool = False):
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = train_for_ticker(ticker, force_retrain)
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            results[ticker] = {"error": str(e)}
    return results
