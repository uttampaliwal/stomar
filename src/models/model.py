import logging
import torch
import torch.nn as nn
import joblib
import os
import time
import json

from src.core.constants import MODELS_DIR
from src.models.artifacts import ArtifactBundle, ArtifactVerificationError, has_predict, has_transform

os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple[float, tuple]] = {}
_MODEL_CACHE_TTL = 3600  # 1 hour
_MODEL_CACHE_MAX = 10  # Max cached model sets (each ~200-500 MB)

# Feature schema version. Bump whenever FEATURE_COLS changes meaning.
FEATURE_SCHEMA_VERSION = "4"

# File layout of a per-ticker bundle.
_TICKER_ARTIFACTS = [
    "lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl", "scaler.pkl",
    "features.pkl", "lstm_dim.pkl", "lgb.pkl", "cat.pkl",
]

# ── deprecated legacy loader (never used by the runtime path) ────────────────
#
# Kept only so old code that imported it fails loudly instead of silently
# deserializing. New code must go through ArtifactBundle.


def safe_joblib_load(path: str, expected_key: str = None):  # pragma: no cover - legacy API
    raise RuntimeError(
        "safe_joblib_load() is disabled: it deserialized before verification. "
        "Use src.models.artifacts.ArtifactBundle instead."
    )


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


def build_catboost_model():
    """CatBoost classifier (5th ensemble member; ordered boosting, robust to
    feature interactions and categorical leakage)."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _base(root: str, ticker: str, name: str) -> str:
    ticker_clean = ticker.replace(".", "_")
    return os.path.join(root, f"{ticker_clean}_{name}")


def save_models(lstm, gru, transformer, xgb, scaler, feature_cols, ticker,
                lgb_model=None, cat_model=None, *,
                model_version: str = "1",
                feature_schema_version: str = FEATURE_SCHEMA_VERSION,
                training_dataset_hash: str = "",
                root: str = MODELS_DIR):
    """Persist the full ticker bundle plus a SHA-256 manifest.

    Nothing is loadable later unless every file matches the manifest.
    ``lstm``/``gru``/``transformer`` may be None (tree-only bundles);
    only artifacts that exist are written and manifested.
    """
    ticker_clean = ticker.replace(".", "_")
    def base(name):
        return os.path.join(root, f"{ticker_clean}_{name}")

    if lstm is not None:
        torch.save(lstm.state_dict(), base("lstm.pt"))
        torch.save(gru.state_dict(), base("gru.pt"))
        torch.save(transformer.state_dict(), base("transformer.pt"))
        joblib.dump(lstm.lstm.input_size, base("lstm_dim.pkl"))
    joblib.dump(xgb, base("xgb.pkl"))
    if lgb_model is not None:
        joblib.dump(lgb_model, base("lgb.pkl"))
    if cat_model is not None:
        joblib.dump(cat_model, base("cat.pkl"))
    joblib.dump(scaler, base("scaler.pkl"))
    joblib.dump(feature_cols, base("features.pkl"))

    files = []
    for ext in _TICKER_ARTIFACTS:
        if os.path.exists(base(ext)):
            files.append(f"{ticker_clean}_{ext}")
    ArtifactBundle.create(
        root, ticker_clean,
        {name: "" for name in files},
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        training_dataset_hash=training_dataset_hash,
    )
    return files


def load_cat_model(ticker: str, root: str = MODELS_DIR):
    """Load the optional CatBoost model (5th ensemble member) via manifest.

    Returns None when the model file is absent (legacy / tree-only models).
    Raises ArtifactVerificationError if present but unverifiable.
    """
    try:
        bundle = ArtifactBundle.for_ticker(root, ticker)
    except ArtifactVerificationError:
        return None
    name = f"{ticker.replace('.', '_')}_cat.pkl"
    if name not in bundle.listed_files():
        return None
    cat = bundle.load_joblib(name, type_check=has_predict)
    return cat


def load_models(ticker: str, root: str = MODELS_DIR):
    """Load a ticker's model bundle — every artifact hash-verified first.

    Raises ArtifactVerificationError when the bundle is missing, tampered,
    or when a file changed after verification. This is the ONLY runtime
    entry point for loading ticker models.
    """
    cache_key = f"{ticker}:{root}"
    if cache_key in _model_cache:
        ts, models = _model_cache[cache_key]
        if time.time() - ts < _MODEL_CACHE_TTL:
            return models
        del _model_cache[cache_key]

    bundle = ArtifactBundle.for_ticker(root, ticker)
    ticker_clean = ticker.replace(".", "_")

    input_dim = bundle.load_joblib(f"{ticker_clean}_lstm_dim.pkl")

    lstm = StockLSTM(input_dim=input_dim).to(DEVICE)
    lstm.load_state_dict(bundle.load_torch(f"{ticker_clean}_lstm.pt"))
    lstm.eval()

    gru = StockGRU(input_dim=input_dim).to(DEVICE)
    gru.load_state_dict(bundle.load_torch(f"{ticker_clean}_gru.pt"))
    gru.eval()

    transformer = StockTransformer(input_dim=input_dim).to(DEVICE)
    transformer.load_state_dict(bundle.load_torch(f"{ticker_clean}_transformer.pt"))
    transformer.eval()

    xgb = bundle.load_joblib(f"{ticker_clean}_xgb.pkl", type_check=has_predict)
    scaler = bundle.load_joblib(f"{ticker_clean}_scaler.pkl", type_check=has_transform)
    features = bundle.load_joblib(f"{ticker_clean}_features.pkl")

    lgb_model = None
    lgb_name = f"{ticker_clean}_lgb.pkl"
    if lgb_name in bundle.listed_files():
        lgb_model = bundle.load_joblib(lgb_name, type_check=has_predict)

    result = lstm, gru, transformer, xgb, scaler, features, lgb_model
    # Evict oldest entries if cache is full
    if len(_model_cache) >= _MODEL_CACHE_MAX:
        oldest_key = min(_model_cache, key=lambda k: _model_cache[k][0])
        del _model_cache[oldest_key]
        logger.debug("Evicted model cache: %s", oldest_key)
    _model_cache[cache_key] = (time.time(), result)
    return result


def model_feature_cols(models: dict) -> list[str]:
    """Feature list to use for inference with a loaded model bundle.

    ALWAYS prefers the feature list stored with the model: the scaler and
    tree models were fitted on exactly that column set, and the current
    trainer.FEATURE_COLS may have evolved (new/removed factors) since
    training. Falling back to the current schema would silently break
    shape alignment or feed columns the model never saw.
    """
    from src.models.trainer import FEATURE_COLS
    stored = models.get("features") if isinstance(models, dict) else None
    if isinstance(stored, list) and stored:
        return stored
    return FEATURE_COLS


def models_exist(ticker: str, root: str = MODELS_DIR) -> bool:
    """True only when the full verified bundle (incl. manifest) is present."""
    try:
        bundle = ArtifactBundle.for_ticker(root, ticker)
    except ArtifactVerificationError:
        return False
    ticker_clean = ticker.replace(".", "_")
    required = ["lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl",
                "scaler.pkl", "features.pkl", "lstm_dim.pkl"]
    return all(f"{ticker_clean}_{ext}" in bundle.listed_files() for ext in required)


def promote_model(ticker: str, feature_hash: str = None, metrics: dict = None,
                  approval_state: str = "approved") -> dict:
    """Promote a trained bundle to the production directory.

    Copies the manifest together with every artifact so the promoted
    bundle remains hash-verified, and records the promotion in the model
    registry (see src.models.model_registry).

    Args:
        ticker: Stock ticker
        feature_hash: Feature version hash for compatibility tracking
        metrics: Performance metrics recorded at promotion time
        approval_state: "approved" or "rejected"

    Returns:
        Dict with status, ticker, model_path, model_version
    """
    import shutil
    from src.models.model_registry import ModelRegistry

    src_bundle = ArtifactBundle.for_ticker(MODELS_DIR, ticker)
    ticker_clean = ticker.replace(".", "_")
    dest_dir = os.path.join(MODELS_DIR, "production")
    os.makedirs(dest_dir, exist_ok=True)

    copied = []
    for name in src_bundle.listed_files():
        src = src_bundle.file_path(name)
        dst = os.path.join(dest_dir, name)
        shutil.copy2(src, dst)
        copied.append(name)
    shutil.copy2(src_bundle.manifest_path, os.path.join(dest_dir, f"{ticker_clean}_manifest.json"))

    # Re-verify the promoted bundle byte-for-byte.
    promoted = ArtifactBundle.for_ticker(dest_dir, ticker)
    if not promoted.listed_files():
        raise ArtifactVerificationError("promoted bundle is empty")

    registry = ModelRegistry()
    record = registry.register(
        ticker=ticker,
        model_path=os.path.join(dest_dir, f"{ticker_clean}_xgb.pkl"),
        feature_hash=feature_hash or promoted.feature_schema_version,
        metrics={
            "model_version": promoted.model_version,
            "feature_schema_version": promoted.feature_schema_version,
            "training_dataset_hash": promoted.training_dataset_hash,
            **(metrics or {}),
        },
        notes=f"approval_state={approval_state}",
    )

    meta_path = os.path.join(dest_dir, f"{ticker_clean}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "status": "promoted",
            "ticker": ticker,
            "feature_hash": feature_hash,
            "files_promoted": copied,
            "model_version": promoted.model_version,
            "registry_version": record.version,
            "approval_state": approval_state,
        }, f, indent=2)

    logger.info("Promoted %s v%s to production (registry version %d)",
                ticker, promoted.model_version, record.version)
    return {
        "status": "promoted",
        "ticker": ticker,
        "feature_hash": feature_hash,
        "files_promoted": copied,
        "model_version": promoted.model_version,
        "registry_version": record.version,
        "approval_state": approval_state,
    }


def load_production_models(ticker: str):
    """Load the production bundle (models/production) if one exists."""
    prod_dir = os.path.join(MODELS_DIR, "production")
    if os.path.isdir(prod_dir) and models_exist(ticker, root=prod_dir):
        return load_models(ticker, root=prod_dir)
    raise ArtifactVerificationError(f"no verified production bundle for {ticker}")
