import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import tensorflow as tf

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_lstm(input_shape: tuple) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_xgb_model():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )


def save_models(
    lstm_model: tf.keras.Model,
    xgb_model,
    scaler: MinMaxScaler,
    feature_cols: list,
    ticker: str,
):
    ticker_clean = ticker.replace(".", "_")
    lstm_model.save(os.path.join(MODELS_DIR, f"{ticker_clean}_lstm.keras"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f"{ticker_clean}_xgb.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker_clean}_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{ticker_clean}_features.pkl"))


def load_models(ticker: str):
    ticker_clean = ticker.replace(".", "_")
    lstm = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f"{ticker_clean}_lstm.keras")
    )
    xgb = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_scaler.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_features.pkl"))
    return lstm, xgb, scaler, features


def models_exist(ticker: str) -> bool:
    ticker_clean = ticker.replace(".", "_")
    paths = [
        os.path.join(MODELS_DIR, f"{ticker_clean}_lstm.keras"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_xgb.pkl"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_scaler.pkl"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_features.pkl"),
    ]
    return all(os.path.exists(p) for p in paths)
