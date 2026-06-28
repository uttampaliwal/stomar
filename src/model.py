import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def build_lstm(input_dim: int) -> StockLSTM:
    return StockLSTM(input_dim=input_dim).to(DEVICE)


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
    lstm_model: StockLSTM,
    xgb_model,
    scaler: MinMaxScaler,
    feature_cols: list,
    ticker: str,
):
    ticker_clean = ticker.replace(".", "_")
    torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, f"{ticker_clean}_lstm.pt"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f"{ticker_clean}_xgb.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker_clean}_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{ticker_clean}_features.pkl"))
    joblib.dump(lstm_model.lstm.input_size, os.path.join(MODELS_DIR, f"{ticker_clean}_lstm_dim.pkl"))


def load_models(ticker: str):
    ticker_clean = ticker.replace(".", "_")
    input_dim = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_lstm_dim.pkl"))
    lstm = StockLSTM(input_dim=input_dim).to(DEVICE)
    lstm.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, f"{ticker_clean}_lstm.pt"),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    lstm.eval()
    xgb = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_scaler.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, f"{ticker_clean}_features.pkl"))
    return lstm, xgb, scaler, features


def models_exist(ticker: str) -> bool:
    ticker_clean = ticker.replace(".", "_")
    paths = [
        os.path.join(MODELS_DIR, f"{ticker_clean}_lstm.pt"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_xgb.pkl"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_scaler.pkl"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_features.pkl"),
        os.path.join(MODELS_DIR, f"{ticker_clean}_lstm_dim.pkl"),
    ]
    return all(os.path.exists(p) for p in paths)
