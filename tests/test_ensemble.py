"""Tests for ensemble module."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_test_ensemble():
    """Create a test ensemble with all model types on correct device."""
    from src.model import StockLSTM, StockGRU, StockTransformer, build_xgb_model, DEVICE
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    np.random.seed(42)
    n = 100
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(100000, 1000000, n)

    feature_cols = ["close", "volume", "sma_10", "sma_20", "sma_50",
                    "ema_12", "ema_26", "rsi", "macd", "macd_signal",
                    "bb_width", "atr", "obv", "volume_ratio"]
    data = {}
    for col in feature_cols:
        if col == "close":
            data[col] = close
        elif col == "volume":
            data[col] = volume
        else:
            data[col] = close + np.random.randn(n) * 0.5
    df = pd.DataFrame(data, index=dates)

    scaler = MinMaxScaler()
    scaler.fit_transform(df[feature_cols].values)

    input_dim = len(feature_cols)
    lstm = StockLSTM(input_dim=input_dim).to(DEVICE)
    gru = StockGRU(input_dim=input_dim).to(DEVICE)
    transformer = StockTransformer(input_dim=input_dim).to(DEVICE)
    xgb = build_xgb_model()

    X_xgb = df[feature_cols].iloc[:-1]
    y_xgb = (df["close"].iloc[1:].values > df["close"].iloc[:-1].values).astype(int)
    xgb.fit(X_xgb, y_xgb, verbose=False)

    return lstm, gru, transformer, xgb, scaler, feature_cols, df


class TestEnsembleDirection:
    def test_ensemble_output_is_binary(self):
        from src.ensemble import predict_ensemble
        lstm, gru, transformer, xgb, scaler, feature_cols, df = _make_test_ensemble()

        direction, confidence, details = predict_ensemble(
            lstm, gru, transformer, xgb, scaler, feature_cols, df
        )
        assert direction in (0, 1), f"Direction should be 0 or 1, got {direction}"
        assert 0 <= confidence <= 100, f"Confidence should be 0-100, got {confidence}"


class TestEnsembleConfidence:
    def test_confidence_is_non_negative(self):
        from src.ensemble import predict_ensemble
        lstm, gru, transformer, xgb, scaler, feature_cols, df = _make_test_ensemble()

        direction, confidence, details = predict_ensemble(
            lstm, gru, transformer, xgb, scaler, feature_cols, df
        )
        assert confidence >= 0
