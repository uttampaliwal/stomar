"""Ensemble prediction with stacked meta-learner and regime routing.

Combines 5 base models (LSTM, GRU, Transformer, XGBoost, LightGBM) using
either equal weights or a learned LogisticRegression meta-learner.

Usage:
    from src.ensemble import predict_ensemble, train_meta_learner

    # Train meta-learner on out-of-sample data
    meta_model = train_meta_learner(meta_X_train, y_train)

    # Predict with meta-learner
    direction, confidence, details = predict_ensemble(
        lstm, gru, transformer, xgb, scaler, feature_cols, df,
        meta_model=meta_model,
    )
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.model import DEVICE

logger = logging.getLogger(__name__)


# ── Regime-Conditional Weights ──

REGIME_WEIGHTS = {
    "Bull": {
        "lstm": 0.15, "gru": 0.15, "transformer": 0.2,
        "xgb": 0.25, "lgb": 0.25,
    },
    "Bear": {
        "lstm": 0.25, "gru": 0.25, "transformer": 0.15,
        "xgb": 0.15, "lgb": 0.20,
    },
    "Sideways": {
        "lstm": 0.20, "gru": 0.20, "transformer": 0.20,
        "xgb": 0.20, "lgb": 0.20,
    },
}


def get_regime_weights(regime: str) -> dict:
    """Return ensemble weights for the given market regime.

    Falls back to equal weight for unknown regimes.
    """
    if regime in REGIME_WEIGHTS:
        return REGIME_WEIGHTS[regime]
    return {"lstm": 0.2, "gru": 0.2, "transformer": 0.2, "xgb": 0.2, "lgb": 0.2}


# ── Meta-Learner ──

def train_meta_learner(meta_X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train a stacked meta-learner on base model outputs.

    Args:
        meta_X: (N, 5) array of base model probabilities [lstm, gru, tf, xgb, lgb]
        y: (N,) binary labels (0 or 1)

    Returns:
        Fitted sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    pipe.fit(meta_X, y)

    coefs = pipe.named_steps["clf"].coef_[0]
    model_names = ["lstm", "gru", "transformer", "xgb", "lgb"]
    logger.info("Meta-learner trained. Coefficients: %s",
                dict(zip(model_names, coefs.round(3))))

    return pipe


def predict_with_metalearner(meta_model: Pipeline, meta_X: np.ndarray) -> np.ndarray:
    """Predict probabilities using the meta-learner.

    Args:
        meta_model: Fitted Pipeline from train_meta_learner()
        meta_X: (N, 5) array of base model probabilities

    Returns:
        (N,) array of probabilities for class 1
    """
    return meta_model.predict_proba(meta_X)[:, 1]


def evaluate_metalearner(meta_X_test: np.ndarray, y_test: np.ndarray,
                         meta_model: Pipeline,
                         equal_weight_preds: np.ndarray = None) -> dict:
    """Evaluate meta-learner vs equal-weight baseline.

    Args:
        meta_X_test: (N, 5) test features
        y_test: (N,) true labels
        meta_model: Fitted meta-learner
        equal_weight_preds: (N,) equal-weight ensemble predictions (optional)

    Returns:
        Dict with accuracy comparison
    """
    meta_probs = predict_with_metalearner(meta_model, meta_X_test)
    meta_preds = (meta_probs > 0.5).astype(int)
    meta_acc = accuracy_score(y_test, meta_preds)

    result = {
        "meta_learner_accuracy": meta_acc,
        "meta_learner_correct": int(meta_preds.sum()),
        "total_samples": len(y_test),
    }

    if equal_weight_preds is not None:
        ew_acc = accuracy_score(y_test, equal_weight_preds)
        result["equal_weight_accuracy"] = ew_acc
        result["improvement"] = meta_acc - ew_acc
        result["improvement_pct"] = (meta_acc - ew_acc) / max(ew_acc, 0.001) * 100

    return result


def save_meta_model(meta_model: Pipeline, path: str):
    """Save meta-learner to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta_model, f)
    logger.info("Meta-learner saved to %s", path)


def load_meta_model(path: str) -> Pipeline:
    """Load meta-learner from disk."""
    with open(path, "rb") as f:
        meta_model = pickle.load(f)
    logger.info("Meta-learner loaded from %s", path)
    return meta_model


# ── Ensemble Prediction ──

def predict_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
                     recent_weights=None, lgb_model=None,
                     meta_model=None, regime=None):
    """Run ensemble prediction on latest data.

    Args:
        meta_model: Optional fitted meta-learner Pipeline. When provided,
            uses learned weights instead of manual combination.
        regime: Optional regime string ("Bull"/"Bear"/"Sideways").
            When provided with meta_model=None, uses regime-specific weights.
    """
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    latest_data = df_feat[feature_cols].dropna()
    if len(latest_data) < 60:
        return None, None, {}

    latest_scaled = scaler.transform(latest_data.values[-60:])
    inp = torch.tensor(latest_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    lstm.eval()
    gru.eval()
    transformer.eval()
    with torch.no_grad():
        pred_lstm = lstm(inp).item()
        pred_gru = gru(inp).item()
        pred_transformer = transformer(inp).item()

    current_scaled = latest_scaled[-1, 0]

    dir_lstm = 1 if pred_lstm > current_scaled else 0
    dir_gru = 1 if pred_gru > current_scaled else 0
    dir_transformer = 1 if pred_transformer > current_scaled else 0

    xgb_input = latest_data.iloc[-1:][feature_cols]
    xgb_prob = xgb.predict_proba(xgb_input)[0]
    dir_xgb = int(xgb.predict(xgb_input)[0])

    dir_lgb = 0
    lgb_prob = [0.5, 0.5]
    if lgb_model is not None:
        lgb_prob = lgb_model.predict_proba(xgb_input)[0]
        dir_lgb = int(lgb_model.predict(xgb_input)[0])

    # Build model probability vector
    meta_X = np.array([[
        float(xgb_prob[1]),   # xgb P(up)
        float(lgb_prob[1]),   # lgb P(up)
        float(dir_lstm),      # lstm direction
        float(dir_gru),       # gru direction
        float(dir_transformer),  # transformer direction
    ]])

    # Use meta-learner if available
    if meta_model is not None:
        ensemble_prob = float(predict_with_metalearner(meta_model, meta_X)[0])
        weights_used = "meta_learner"
    else:
        # Use regime weights or provided weights
        if recent_weights is not None:
            weights = recent_weights
        elif regime is not None:
            weights = get_regime_weights(regime)
        else:
            n_models = 5
            w = 1.0 / n_models
            weights = {"lstm": w, "gru": w, "transformer": w, "xgb": w, "lgb": w}

        dl_prob = (dir_lstm * weights.get("lstm", 0.2) +
                   dir_gru * weights.get("gru", 0.2) +
                   dir_transformer * weights.get("transformer", 0.2))

        xgb_weighted = xgb_prob[1] * weights.get("xgb", 0.2)
        lgb_weighted = lgb_prob[1] * weights.get("lgb", 0.2)

        ensemble_prob = dl_prob + xgb_weighted + lgb_weighted
        total_weight = sum(weights.values())
        if total_weight > 0:
            ensemble_prob = ensemble_prob / total_weight
        weights_used = weights

    ensemble_dir = 1 if ensemble_prob > 0.5 else 0
    confidence = abs(ensemble_prob - 0.5) * 2 * 100

    details = {
        "lstm_dir": dir_lstm, "lstm_pred": pred_lstm,
        "gru_dir": dir_gru, "gru_pred": pred_gru,
        "transformer_dir": dir_transformer, "transformer_pred": pred_transformer,
        "xgb_dir": dir_xgb, "xgb_prob_up": float(xgb_prob[1]),
        "lgb_dir": dir_lgb, "lgb_prob_up": float(lgb_prob[1]),
        "ensemble_prob": float(ensemble_prob),
        "weights": weights_used,
    }
    return ensemble_dir, confidence, details


def backtest_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
                      seq_length=60, lgb_model=None, meta_model=None, regime=None):
    """Backtest ensemble over historical data.

    Args:
        meta_model: Optional meta-learner for learned weights.
        regime: Optional regime string for regime-conditional routing.
    """
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols].dropna().values
    scaled = scaler.transform(data)

    results = []
    for i in range(seq_length, len(scaled)):
        inp = torch.tensor(scaled[i - seq_length:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prev_close = scaled[i - 1, 0]
        actual_close = scaled[i, 0]
        actual_dir = 1 if actual_close > prev_close else 0

        lstm.eval()
        gru.eval()
        transformer.eval()
        with torch.no_grad():
            p_lstm = lstm(inp).item()
            p_gru = gru(inp).item()
            p_tf = transformer(inp).item()

        xgb_inp = df_feat[feature_cols].iloc[[i - 1]]
        xgb_p = xgb.predict_proba(xgb_inp)[0][1]

        lgb_p = 0.5
        if lgb_model is not None:
            lgb_p = lgb_model.predict_proba(xgb_inp)[0][1]

        d_lstm = 1 if p_lstm > prev_close else 0
        d_gru = 1 if p_gru > prev_close else 0
        d_tf = 1 if p_tf > prev_close else 0

        # Build meta features
        meta_X = np.array([[xgb_p, lgb_p, d_lstm, d_gru, d_tf]])

        if meta_model is not None:
            final_prob = float(predict_with_metalearner(meta_model, meta_X)[0])
        else:
            if regime is not None:
                weights = get_regime_weights(regime)
            else:
                weights = {"lstm": 0.2, "gru": 0.2, "transformer": 0.2, "xgb": 0.2, "lgb": 0.2}

            dl_prob = (d_lstm * weights.get("lstm", 0.2) +
                       d_gru * weights.get("gru", 0.2) +
                       d_tf * weights.get("transformer", 0.2))
            xgb_weighted = xgb_p * weights.get("xgb", 0.2)
            lgb_weighted = lgb_p * weights.get("lgb", 0.2)
            final_prob = dl_prob + xgb_weighted + lgb_weighted
            total_weight = sum(weights.values())
            if total_weight > 0:
                final_prob = final_prob / total_weight

        final = 1 if final_prob > 0.5 else 0
        dl_ens = (d_lstm + d_gru + d_tf) / 3
        dl_dir = 1 if dl_ens > 0.5 else 0
        xgb_dir = int(xgb.predict(xgb_inp)[0])

        results.append({
            "actual": actual_dir,
            "lstm": d_lstm, "gru": d_gru, "transformer": d_tf,
            "dl_ensemble": dl_dir, "xgb": xgb_dir, "lgb": 1 if lgb_p > 0.5 else 0,
            "final_ensemble": final,
        })

    return results
