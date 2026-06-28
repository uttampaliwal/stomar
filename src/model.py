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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
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


class StockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class StockTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 60, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.2, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(d_model, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_lstm(input_dim: int) -> StockLSTM:
    return StockLSTM(input_dim=input_dim).to(DEVICE)


def build_gru(input_dim: int) -> StockGRU:
    return StockGRU(input_dim=input_dim).to(DEVICE)


def build_transformer(input_dim: int) -> StockTransformer:
    return StockTransformer(input_dim=input_dim).to(DEVICE)


def build_xgb_model():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.05,
        random_state=42, eval_metric="logloss",
    )


def save_models(lstm, gru, transformer, xgb, scaler, feature_cols, ticker):
    ticker_clean = ticker.replace(".", "_")
    base = lambda name: os.path.join(MODELS_DIR, f"{ticker_clean}_{name}")
    torch.save(lstm.state_dict(), base("lstm.pt"))
    torch.save(gru.state_dict(), base("gru.pt"))
    torch.save(transformer.state_dict(), base("transformer.pt"))
    joblib.dump(xgb, base("xgb.pkl"))
    joblib.dump(scaler, base("scaler.pkl"))
    joblib.dump(feature_cols, base("features.pkl"))
    joblib.dump(lstm.lstm.input_size, base("lstm_dim.pkl"))


def load_models(ticker: str):
    ticker_clean = ticker.replace(".", "_")
    base = lambda name: os.path.join(MODELS_DIR, f"{ticker_clean}_{name}")
    input_dim = joblib.load(base("lstm_dim.pkl"))

    lstm = StockLSTM(input_dim=input_dim).to(DEVICE)
    lstm.load_state_dict(torch.load(base("lstm.pt"), map_location=DEVICE, weights_only=True))
    lstm.eval()

    gru = StockGRU(input_dim=input_dim).to(DEVICE)
    gru.load_state_dict(torch.load(base("gru.pt"), map_location=DEVICE, weights_only=True))
    gru.eval()

    transformer = StockTransformer(input_dim=input_dim).to(DEVICE)
    transformer.load_state_dict(torch.load(base("transformer.pt"), map_location=DEVICE, weights_only=True))
    transformer.eval()

    xgb = joblib.load(base("xgb.pkl"))
    scaler = joblib.load(base("scaler.pkl"))
    features = joblib.load(base("features.pkl"))
    return lstm, gru, transformer, xgb, scaler, features


def models_exist(ticker: str) -> bool:
    ticker_clean = ticker.replace(".", "_")
    base = lambda name: os.path.join(MODELS_DIR, f"{ticker_clean}_{name}")
    exts = ["lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl", "scaler.pkl", "features.pkl", "lstm_dim.pkl"]
    return all(os.path.exists(base(e)) for e in exts)
