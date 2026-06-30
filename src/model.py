import warnings

import torch
import torch.nn as nn
import joblib
import os
import time

from src.constants import MODELS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: dict[str, tuple[float, tuple]] = {}
_MODEL_CACHE_TTL = 3600  # 1 hour


class StockLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class StockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
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
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 max_seq_len: int = 120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.2, batch_first=True, norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.norm(x)
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
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=1.0, random_state=42, eval_metric="logloss",
    )


def build_lgb_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=30,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )


def save_models(lstm, gru, transformer, xgb, scaler, feature_cols, ticker, lgb_model=None):
    ticker_clean = ticker.replace(".", "_")
    def base(name):
        return os.path.join(MODELS_DIR, f"{ticker_clean}_{name}")
    torch.save(lstm.state_dict(), base("lstm.pt"))
    torch.save(gru.state_dict(), base("gru.pt"))
    torch.save(transformer.state_dict(), base("transformer.pt"))
    joblib.dump(xgb, base("xgb.pkl"))
    if lgb_model is not None:
        joblib.dump(lgb_model, base("lgb.pkl"))
    joblib.dump(scaler, base("scaler.pkl"))
    joblib.dump(feature_cols, base("features.pkl"))
    joblib.dump(lstm.lstm.input_size, base("lstm_dim.pkl"))


def load_models(ticker: str):
    cache_key = ticker
    if cache_key in _model_cache:
        ts, models = _model_cache[cache_key]
        if time.time() - ts < _MODEL_CACHE_TTL:
            return models
        del _model_cache[cache_key]

    import warnings
    ticker_clean = ticker.replace(".", "_")
    def base(name):
        return os.path.join(MODELS_DIR, f"{ticker_clean}_{name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb = joblib.load(base("xgb.pkl"))
        scaler = joblib.load(base("scaler.pkl"))
        features = joblib.load(base("features.pkl"))

        lgb_model = None
        lgb_path = base("lgb.pkl")
        if os.path.exists(lgb_path):
            lgb_model = joblib.load(lgb_path)

    result = lstm, gru, transformer, xgb, scaler, features, lgb_model
    _model_cache[cache_key] = (time.time(), result)
    return result


def models_exist(ticker: str) -> bool:
    ticker_clean = ticker.replace(".", "_")
    def base(name):
        return os.path.join(MODELS_DIR, f"{ticker_clean}_{name}")
    exts = ["lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl", "scaler.pkl", "features.pkl", "lstm_dim.pkl"]
    return all(os.path.exists(base(e)) for e in exts)


def promote_model(ticker: str, feature_hash: str = None) -> dict:
    """Promote a trained model to production.

    Copies model files to production paths and records metadata.
    Full registry integration added in Task 6.

    Args:
        ticker: Stock ticker
        feature_hash: Feature version hash for compatibility tracking

    Returns:
        Dict with status, ticker, model_path
    """
    import shutil
    ticker_clean = ticker.replace(".", "_")
    dest_dir = os.path.join(MODELS_DIR, "production")
    os.makedirs(dest_dir, exist_ok=True)

    src_dir = MODELS_DIR
    copied = []
    for ext in ["lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl", "scaler.pkl", "features.pkl", "lstm_dim.pkl"]:
        src = os.path.join(src_dir, f"{ticker_clean}_{ext}")
        dst = os.path.join(dest_dir, f"{ticker_clean}_{ext}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(ext)

    meta = {
        "status": "promoted",
        "ticker": ticker,
        "feature_hash": feature_hash,
        "files_promoted": copied,
    }
    import json
    meta_path = os.path.join(dest_dir, f"{ticker_clean}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
