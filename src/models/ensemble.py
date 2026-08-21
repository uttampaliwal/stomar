"""Ensemble prediction with stacked meta-learner and regime routing.

Combines 6 base models (LSTM, GRU, Transformer, XGBoost, LightGBM, CatBoost)
using either equal weights, static regime weights, learned weights from a
LogisticRegression meta-learner, or dynamic weights driven by each model's
rolling 30-day performance *within the detected market regime*.

The meta-controller layer is the Bayesian/stacking ensemble: base-model
probability outputs are stacked, and the meta-learner (or dynamic regime-
conditional weights) maps them to a final probability with a Wilson
confidence interval and a conviction label.

Usage:
    from src.models.ensemble import (
        predict_ensemble, train_meta_learner, regime_adjusted_ensemble,
    )

    # Dynamic regime-conditional prediction
    decision = regime_adjusted_ensemble(ticker, models, df_feat)
    # {"signal": "BUY", "probability_up": 0.84, "confidence": 68.0,
    #  "conviction": "HIGH", "confidence_interval": [0.61, 0.95], ...}
"""

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.models.model import DEVICE
from src.models.artifacts import has_fit_predict

logger = logging.getLogger(__name__)

# Model key order used across the ensemble (probabilities vector layout)
MODEL_NAMES = ["lstm", "gru", "transformer", "xgb", "lgb", "cat"]
# Meta-learner column order (catboost appended last so legacy 5-feature
# meta models remain compatible).
META_COLUMNS = ["xgb", "lgb", "lstm", "gru", "transformer", "cat"]

WILSON_Z = 1.645  # 90% confidence

# The conviction interval is computed over the trailing fraction of
# backtest rows only. _backtest_rows spans the FULL history (mostly the
# models' own training window); scoring all of it reports in-sample
# accuracy as if it were out-of-sample and inflates every confidence
# interval. Mirroring the trainer's ~80/20 chronological split, the last
# 25% of rows is an approximately out-of-sample window (margin included).
APPROX_OOS_FRACTION = 0.25


def approx_oos_rows(backtest_rows: list[dict]) -> list[dict]:
    """Trailing approximately-out-of-sample slice of backtest rows.

    See APPROX_OOS_FRACTION. Returns at least one row for non-empty input
    so tiny histories still yield an interval (a wide, honest one).
    """
    if not backtest_rows:
        return []
    n_oos = max(1, int(len(backtest_rows) * APPROX_OOS_FRACTION))
    return backtest_rows[-n_oos:]


# ── Regime-Conditional Static Weights ──

REGIME_WEIGHTS = {
    "Bull": {
        "lstm": 0.12, "gru": 0.12, "transformer": 0.16,
        "xgb": 0.20, "lgb": 0.20, "cat": 0.20,
    },
    "Bear": {
        "lstm": 0.20, "gru": 0.20, "transformer": 0.12,
        "xgb": 0.12, "lgb": 0.16, "cat": 0.20,
    },
    "Sideways": {
        "lstm": 0.17, "gru": 0.17, "transformer": 0.17,
        "xgb": 0.17, "lgb": 0.16, "cat": 0.16,
    },
}


def get_regime_weights(regime: str) -> dict:
    """Return ensemble weights for the given market regime.

    Falls back to equal weight for unknown regimes.
    """
    if regime in REGIME_WEIGHTS:
        return REGIME_WEIGHTS[regime]
    return {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}


# ── Meta-Learner ──

def train_meta_learner(meta_X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train a stacked meta-learner on base model outputs.

    Args:
        meta_X: (N, K) array of base model probabilities in META_COLUMNS order
            (K = 6 with CatBoost, 5 for legacy pipelines)
        y: (N,) binary labels (0 or 1)

    Returns:
        Fitted sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(meta_X, y)

    coefs = pipe.named_steps["clf"].coef_[0]
    n_features = getattr(pipe, "n_features_in_", len(META_COLUMNS))
    model_names = META_COLUMNS[: min(n_features, len(META_COLUMNS))]
    logger.info("Meta-learner trained. Coefficients: %s",
                dict(zip(model_names, coefs[: len(model_names)].round(3))))

    return pipe


def predict_with_metalearner(meta_model: Pipeline, meta_X: np.ndarray) -> np.ndarray:
    """Predict probabilities using the meta-learner.

    Args:
        meta_model: Fitted Pipeline from train_meta_learner()
        meta_X: (N, K) array of base model probabilities. When the fitted
            meta-learner expects fewer columns than provided (legacy
            5-feature models), the leading columns are used.

    Returns:
        (N,) array of probabilities for class 1
    """
    n_features = getattr(meta_model, "n_features_in_", None)
    if n_features is not None and meta_X.shape[1] > n_features:
        meta_X = meta_X[:, :n_features]
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


def save_meta_model(meta_model: Pipeline, path: str, *,
                    feature_schema_version: str = "1",
                    training_dataset_hash: str = ""):
    """Save meta-learner to disk as a hash-verified artifact bundle."""
    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    import joblib
    from src.models.artifacts import ArtifactBundle
    joblib.dump(meta_model, path)
    ArtifactBundle.create(
        str(_Path(path).parent),
        _Path(path).stem,
        {_Path(path).name: ""},
        model_version="1",
        feature_schema_version=feature_schema_version,
        training_dataset_hash=training_dataset_hash,
    )
    logger.info("Meta-learner saved to %s", path)


def load_meta_model(path: str) -> Pipeline:
    """Load meta-learner from disk — hash-verified via the bundle manifest.

    Refuses to deserialize any file whose digest does not match the
    adjacent manifest.
    """
    from pathlib import Path as _Path
    from src.models.artifacts import ArtifactBundle
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Meta-learner file not found: {path}")
    bundle = ArtifactBundle.load(str(_Path(path).parent), _Path(path).stem)
    meta_model = bundle.load_joblib(_Path(path).name, type_check=has_fit_predict)
    if not hasattr(meta_model, "predict") or not hasattr(meta_model, "fit"):
        raise TypeError(f"Loaded object from {path} is not a valid sklearn estimator")
    logger.info("Meta-learner loaded from %s", path)
    return meta_model


# ── Ensemble Prediction ──

# Features that look into the future (leakage) — never allowed as model input.
# Legacy models were trained on some of these; at inference the column is
# neutralized with past-only values so the model still receives its expected
# input shape WITHOUT any future information. Retraining must exclude them
# entirely (see trainer.FEATURE_COLS).
BANNED_LOOKAHEAD_FEATURES = frozenset({"ichimoku_chikou"})


def _neutralize_lookahead_features(df_feat: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Replace banned look-ahead columns with past-only data and log loudly.

    Returns the (possibly shortened) feature list actually safe to use.
    """
    safe = []
    for col in feature_cols:
        if col in BANNED_LOOKAHEAD_FEATURES:
            if col in df_feat.columns:
                # close.shift(-26) is future data; close (shift 0) is the
                # latest past value available at decision time.
                df_feat[col] = df_feat["close"]
                logger.warning(
                    "look-ahead feature %r requested by model — neutralized "
                    "with past-only values; retrain without it", col,
                )
            continue
        safe.append(col)
    return safe


def predict_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
                     recent_weights=None, lgb_model=None, cat_model=None,
                     meta_model=None, regime=None):
    """Run ensemble prediction on latest data.

    Deep-learning base models (lstm/gru/transformer) may be None
    (tree-only bundles, #59); they are skipped and the remaining
    tree models drive the signal.

    Args:
        meta_model: Optional fitted meta-learner Pipeline. When provided,
            uses learned weights instead of manual combination.
        regime: Optional regime string ("Bull"/"Bear"/"Sideways").
            When provided with meta_model=None, uses regime-specific weights.
        cat_model: Optional CatBoost classifier (5th base model).
    """
    feature_cols = _neutralize_lookahead_features(df_feat, list(feature_cols))
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    latest_data = df_feat[feature_cols].dropna()
    if len(latest_data) < 60:
        return 0, 0.0, {"error": "insufficient_data"}

    latest_scaled = scaler.transform(latest_data.values[-60:])

    # Deep-learning base models may be absent (tree-only bundles, #59).
    has_dl = all(m is not None for m in (lstm, gru, transformer))
    if has_dl:
        inp = torch.tensor(latest_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        lstm.eval()
        gru.eval()
        transformer.eval()
        with torch.no_grad():
            pred_lstm = lstm(inp).item()
            pred_gru = gru(inp).item()
            pred_transformer = transformer(inp).item()

        current_scaled = latest_scaled[-1, 0]

        # Convert regression outputs to probabilities (sigmoid of price diff)
        diff_lstm = pred_lstm - current_scaled
        diff_gru = pred_gru - current_scaled
        diff_tf = pred_transformer - current_scaled
        prob_lstm = 1.0 / (1.0 + np.exp(-diff_lstm * 10))
        prob_gru = 1.0 / (1.0 + np.exp(-diff_gru * 10))
        prob_tf = 1.0 / (1.0 + np.exp(-diff_tf * 10))

        dir_lstm = 1 if pred_lstm > current_scaled else 0
        dir_gru = 1 if pred_gru > current_scaled else 0
        dir_transformer = 1 if pred_transformer > current_scaled else 0
    else:
        prob_lstm = prob_gru = prob_tf = 0.5
        dir_lstm = dir_gru = dir_transformer = 0
        pred_lstm = pred_gru = pred_transformer = 0.0

    xgb_input = latest_data.iloc[-1:][feature_cols]
    xgb_prob = xgb.predict_proba(xgb_input)[0]
    dir_xgb = int(xgb.predict(xgb_input)[0])

    dir_lgb = 0
    lgb_prob = [0.5, 0.5]
    if lgb_model is not None:
        lgb_prob = lgb_model.predict_proba(xgb_input)[0]
        dir_lgb = int(lgb_model.predict(xgb_input)[0])

    dir_cat = 0
    cat_prob = [0.5, 0.5]
    if cat_model is not None:
        cat_prob = cat_model.predict_proba(xgb_input)[0]
        dir_cat = int(cat_model.predict(xgb_input)[0])

    # Build model probability vector in META_COLUMNS order:
    # [xgb, lgb, lstm, gru, transformer, cat]
    meta_X = np.array([[
        float(xgb_prob[1]),   # xgb P(up)
        float(lgb_prob[1]),   # lgb P(up)
        float(prob_lstm),     # lstm P(up) -- continuous
        float(prob_gru),      # gru P(up) -- continuous
        float(prob_tf),       # transformer P(up) -- continuous
        float(cat_prob[1]),   # cat P(up)
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
            weights = {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}

        # Build available model probabilities and renormalize weights
        model_probs = []
        if has_dl:
            model_probs += [
                ("lstm", prob_lstm),
                ("gru", prob_gru),
                ("transformer", prob_tf),
            ]
        model_probs.append(("xgb", xgb_prob[1]))
        if lgb_model is not None:
            model_probs.append(("lgb", lgb_prob[1]))
        if cat_model is not None:
            model_probs.append(("cat", cat_prob[1]))

        total_w = sum(weights.get(name, 1.0 / len(MODEL_NAMES)) for name, _ in model_probs)
        if total_w > 0:
            ensemble_prob = sum(
                prob * weights.get(name, 1.0 / len(MODEL_NAMES)) / total_w
                for name, prob in model_probs
            )
        else:
            ensemble_prob = 0.5
        weights_used = {name: weights.get(name, 1.0 / len(MODEL_NAMES)) / total_w for name, _ in model_probs} if total_w > 0 else weights

    ensemble_dir = 1 if ensemble_prob > 0.5 else 0
    confidence = abs(ensemble_prob - 0.5) * 2 * 100

    details = {
        "lstm_dir": dir_lstm, "lstm_pred": pred_lstm, "lstm_prob": float(prob_lstm),
        "gru_dir": dir_gru, "gru_pred": pred_gru, "gru_prob": float(prob_gru),
        "transformer_dir": dir_transformer, "transformer_pred": pred_transformer, "transformer_prob": float(prob_tf),
        "xgb_dir": dir_xgb, "xgb_prob_up": float(xgb_prob[1]),
        "lgb_dir": dir_lgb, "lgb_prob_up": float(lgb_prob[1]),
        "cat_dir": dir_cat, "cat_prob_up": float(cat_prob[1]),
        "ensemble_prob": float(ensemble_prob),
        "weights": weights_used,
    }
    return ensemble_dir, confidence, details


def _backtest_rows(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
                   seq_length=60, lgb_model=None, cat_model=None):
    """Shared per-row backtest generator over historical data.

    Yields dicts with date, actual direction, per-model direction + prob_up,
    and the meta feature vector. Used by both backtest_ensemble() and the
    regime-conditional rolling performance tracker.
    """
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols].dropna()
    if len(data) < seq_length + 10:
        return
    scaled = scaler.transform(data.values)
    dates = data.index

    has_dl = all(m is not None for m in (lstm, gru, transformer))
    if has_dl:
        lstm.eval()
        gru.eval()
        transformer.eval()

    for i in range(seq_length, len(scaled)):
        try:
            prev_close = scaled[i - 1, 0]
            actual_close = scaled[i, 0]
            actual_dir = 1 if actual_close > prev_close else 0

            p_lstm = p_gru = p_tf = 0.0
            if has_dl:
                inp = torch.tensor(scaled[i - seq_length:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    p_lstm = lstm(inp).item()
                    p_gru = gru(inp).item()
                    p_tf = transformer(inp).item()

            xgb_inp = data.iloc[[i - 1]]
            xgb_p = xgb.predict_proba(xgb_inp)[0][1]

            lgb_p = 0.5
            if lgb_model is not None:
                lgb_p = lgb_model.predict_proba(xgb_inp)[0][1]

            cat_p = 0.5
            if cat_model is not None:
                cat_p = cat_model.predict_proba(xgb_inp)[0][1]

            # Convert DL regression to probabilities
            prob_lstm = 0.5
            prob_gru = 0.5
            prob_tf = 0.5
            if has_dl:
                diff_lstm = p_lstm - prev_close
                diff_gru = p_gru - prev_close
                diff_tf = p_tf - prev_close
                prob_lstm = 1.0 / (1.0 + np.exp(-diff_lstm * 10))
                prob_gru = 1.0 / (1.0 + np.exp(-diff_gru * 10))
                prob_tf = 1.0 / (1.0 + np.exp(-diff_tf * 10))

            yield {
                "date": dates[i],
                "actual": actual_dir,
                "lstm": 1 if p_lstm > prev_close else 0,
                "gru": 1 if p_gru > prev_close else 0,
                "transformer": 1 if p_tf > prev_close else 0,
                "xgb": int(xgb.predict(xgb_inp)[0]),
                "lgb": 1 if lgb_p > 0.5 else 0,
                "cat": 1 if cat_p > 0.5 else 0,
                "lstm_prob": float(prob_lstm),
                "gru_prob": float(prob_gru),
                "transformer_prob": float(prob_tf),
                "xgb_prob": float(xgb_p),
                "lgb_prob": float(lgb_p),
                "cat_prob": float(cat_p),
                "meta_X": [float(xgb_p), float(lgb_p), float(prob_lstm),
                           float(prob_gru), float(prob_tf), float(cat_p)],
            }
        except Exception as e:
            logger.debug("backtest row %d failed: %s", i, e)
            continue


def backtest_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
                      seq_length=60, lgb_model=None, cat_model=None, meta_model=None, regime=None):
    """Backtest ensemble over historical data.

    Args:
        meta_model: Optional meta-learner for learned weights.
        regime: Optional regime string for regime-conditional routing.
        cat_model: Optional CatBoost classifier.
    """
    rows = list(_backtest_rows(
        lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
        seq_length=seq_length, lgb_model=lgb_model, cat_model=cat_model,
    ))
    if not rows:
        return []

    if meta_model is not None:
        meta_probs = predict_with_metalearner(
            meta_model, np.asarray([r["meta_X"] for r in rows])
        )
    else:
        if regime is not None:
            weights = get_regime_weights(regime)
        else:
            weights = {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}

        meta_probs = []
        for r in rows:
            model_probs = [(n, r[f"{n}_prob"]) for n in MODEL_NAMES]
            total_w = sum(weights.get(n, 1.0 / len(MODEL_NAMES)) for n, _ in model_probs)
            if total_w > 0:
                meta_probs.append(sum(p * weights[n] / total_w for n, p in model_probs))
            else:
                meta_probs.append(0.5)
        meta_probs = np.asarray(meta_probs)

    results = []
    for r, final_prob in zip(rows, meta_probs):
        final = 1 if final_prob > 0.5 else 0
        dl_sum = r["lstm"] + r["gru"] + r["transformer"]
        dl_ens = dl_sum / 3
        dl_dir = 1 if dl_ens > 0.5 else 0
        if dl_sum == 0:  # tree-only bundle: no DL models to aggregate
            dl_dir = None
        results.append({
            "actual": r["actual"],
            "lstm": r["lstm"], "gru": r["gru"], "transformer": r["transformer"],
            "dl_ensemble": dl_dir, "xgb": r["xgb"], "lgb": r["lgb"],
            "cat": r["cat"],
            "final_ensemble": final,
        })

    return results


# ── Dynamic Regime-Conditional Weights (Meta-Controller) ──────────────────────

def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a proportion (used for confidence bounds).

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Z-score (1.645 -> 90% CI).

    Returns:
        (lower, upper) bounds.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half_width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width)))


def conviction_label(prob_up: float, ci_lower: float) -> str:
    """Map probability + interval lower bound to a conviction label."""
    if ci_lower >= 0.55 or prob_up >= 0.80:
        return "HIGH"
    if prob_up >= 0.62 or ci_lower >= 0.52:
        return "MEDIUM"
    return "LOW"


def compute_dynamic_regime_weights(
    backtest_rows: list[dict],
    regime_series: pd.Series,
    current_regime: str,
    window: int = 30,
    min_samples: int = 10,
    temperature: float = 8.0,
    model_names: list[str] | None = None,
) -> dict:
    """Rolling performance-based weights within the current regime.

    Each base model is scored by its directional accuracy over the last
    `window` calendar days *on days whose regime matches the current one*.
    Weights are a softmax of the excess-accuracy (acc - 0.5) at the given
    temperature, floored for stability and normalized to sum to 1.

    Models that produced constant probabilities (e.g. a CatBoost model that
    was never trained, so every value is a 0.5 placeholder) are excluded and
    the remaining weights are renormalized.

    Falls back to equal weights when there are too few regime-matching days.

    Returns a dict: {"weights": {...}, "accuracies": {...}, "n_samples": n,
    "excluded": [...], "method": "dynamic" | "fallback"}
    """
    names = model_names or MODEL_NAMES
    if not backtest_rows:
        return {
            "weights": {name: 1.0 / len(names) for name in names},
            "accuracies": {},
            "excluded": [],
            "n_samples": 0,
            "method": "fallback",
        }

    df = pd.DataFrame(backtest_rows).set_index("date")
    if regime_series is not None and len(regime_series) > 0:
        df = df.join(regime_series.rename("regime"), how="left")
    else:
        df["regime"] = None

    cutoff = df.index.max() - pd.Timedelta(days=window)
    sub = df[(df["regime"] == current_regime) & (df.index >= cutoff)]
    n_samples = len(sub)

    accuracies = {}
    if n_samples >= min_samples:
        # Drop models whose probability outputs never vary (placeholders)
        excluded = [
            name for name in names
            if sub[f"{name}_prob"].nunique() <= 1
        ]
        active = [name for name in names if name not in excluded]
        if active:
            for name in active:
                accuracies[name] = float(accuracy_score(sub["actual"], sub[name]))
            excess = np.clip(np.asarray([accuracies[n] for n in active]) - 0.5, 0.0, None)
            exp_w = np.exp(excess * temperature)
            weights = exp_w / exp_w.sum()
            weights = {name: round(float(w), 6) for name, w in zip(active, weights)}
            accuracies = {name: round(float(a), 4) for name, a in accuracies.items()}
            logger.info(
                "dynamic_regime_weights regime=%s n_samples=%d excluded=%s accuracies=%s",
                current_regime, n_samples, excluded, accuracies,
            )
            return {
                "weights": weights, "accuracies": accuracies,
                "excluded": excluded, "n_samples": n_samples,
                "method": "dynamic",
            }
    else:
        return {
            "weights": {name: 1.0 / len(names) for name in names},
            "accuracies": accuracies,
            "excluded": [],
            "n_samples": n_samples,
            "method": "fallback",
            "reason": f"only {n_samples} regime-matching days in window (need {min_samples})",
        }

    return {
        "weights": {name: 1.0 / len(names) for name in names},
        "accuracies": accuracies,
        "excluded": [],
        "n_samples": n_samples,
        "method": "fallback",
    }


def regime_adjusted_ensemble(
    ticker: str,
    models: dict,
    df_feat: pd.DataFrame,
    seq_length: int = 60,
    window: int = 30,
    use_meta: bool = True,
    regime_info: dict | None = None,
    backtest_rows: list[dict] | None = None,
) -> dict:
    """Dynamic meta-controller prediction with regime-based weighting.

    Pipeline:
        1. Detect the market regime (GMM/HMM on log returns + ATR).
        2. Backtest all 6 base models over the trailing window and score
           their accuracy on days in the current regime (rolling 30-day
           performance, regime-conditional).
        3. Weight the base-model probability outputs by those scores
           (or by the stacked meta-learner when one is fitted).
        4. Compute the ensemble probability and a Wilson confidence
            interval on the ensemble's APPROXIMATELY out-of-sample accuracy
            (the trailing APPROX_OOS_FRACTION of backtest rows — scoring the
            full history would present training-window accuracy as OOS).

    Args:
        ticker: Symbol (used for the optional meta-learner file lookup).
        models: Dict with keys lstm/gru/transformer/xgb/lgb/cat/scaler/features.
        df_feat: Feature DataFrame (needs OHLCV columns + FEATURE_COLS).
        seq_length: DL sequence length used at training time.
        window: Rolling performance window in calendar days.
        use_meta: Use the stacked meta-learner when available.
        regime_info: Precomputed detect_regime() output (skips refitting).
        backtest_rows: Precomputed _backtest_rows() output (skips recompute).

    Returns:
        Decision dict with signal, probability_up, confidence, conviction,
        confidence_interval, regime, weights, model breakdown and risk.
    """
    import os

    from src.models.model import MODELS_DIR
    from src.signals.regime_hmm import detect_regime

    ohlc = df_feat[["open", "high", "low", "close", "volume"]].dropna()
    if regime_info is None:
        regime_info = detect_regime(ohlc)
    regime_key = regime_info["regime_key"]

    # Active base models: skip CatBoost and any DL model absent from the
    # bundle (tree-only bundles, #59).
    active_models = [
        n for n in MODEL_NAMES
        if (n != "cat" or models.get("cat") is not None)
        and (n not in ("lstm", "gru", "transformer") or models.get(n) is not None)
    ]

    if backtest_rows is None:
        backtest_rows = list(_backtest_rows(
            models["lstm"], models["gru"], models["transformer"], models["xgb"],
            models["scaler"], models.get("features") or models.get("feature_cols"),
            df_feat, seq_length=seq_length,
            lgb_model=models.get("lgb"), cat_model=models.get("cat"),
        ))

    # Regime-conditional rolling performance -> dynamic weights
    regime_series = None
    if backtest_rows:
        from src.signals.regime_hmm import state_sequence
        try:
            regime_series = state_sequence(ohlc)
        except Exception as exc:
            logger.debug("state_sequence failed: %s", exc)

    dyn = compute_dynamic_regime_weights(
        backtest_rows, regime_series if regime_series is not None else pd.Series(dtype=str),
        regime_key, window=window, model_names=active_models,
    )

    # Optional stacked meta-learner (Bayesian-style ensemble layer)
    meta_model = None
    if use_meta:
        meta_path = os.path.join(MODELS_DIR, f"meta_{ticker.replace('.', '_')}.pkl")
        if os.path.exists(meta_path):
            try:
                meta_model = load_meta_model(meta_path)
            except Exception as exc:
                logger.debug("meta-learner load failed: %s", exc)

    # Final ensemble probability on the latest bar
    last = backtest_rows[-1] if backtest_rows else None
    if last is not None:
        meta_X = np.asarray([last["meta_X"]])
        if meta_model is not None:
            prob_up = float(predict_with_metalearner(meta_model, meta_X)[0])
            weight_method = "meta_learner"
            weights = None
        else:
            probs = np.asarray([last[f"{n}_prob"] for n in active_models])
            w = np.asarray([dyn["weights"][n] for n in active_models])
            total_w = float(w.sum())
            if total_w > 0:
                prob_up = float(probs @ w / total_w)
            else:
                prob_up = 0.5
            weight_method = dyn["method"]
            weights = dict(zip(active_models, [round(float(x / total_w), 4) for x in w])) if total_w > 0 else dict(zip(active_models, [round(float(x), 4) for x in w]))
    else:
        prob_up = 0.5
        weight_method = "fallback"
        weights = {n: 1.0 / len(active_models) for n in active_models}

    # Wilson CI on the ensemble's accuracy, restricted to the approximate
    # out-of-sample window (trailing rows the models did not train on).
    # Scoring the full history would present in-sample accuracy as OOS and
    # overstate conviction.
    n_correct = 0
    n_total = 0
    ci_basis = "insufficient_rows"
    if backtest_rows:
        oos_rows = approx_oos_rows(backtest_rows)
        ci_basis = f"approx_oos_tail_{len(oos_rows)}_rows"
        for r in oos_rows:
            if regime_series is not None and r["date"] in regime_series.index and \
               regime_series.loc[r["date"]] == regime_key:
                # ensemble direction from the same weighting logic
                meta_X = np.asarray([r["meta_X"]])
                if meta_model is not None:
                    p = float(predict_with_metalearner(meta_model, meta_X)[0])
                else:
                    probs = np.asarray([r[f"{n}_prob"] for n in active_models])
                    w = np.asarray([dyn["weights"][n] for n in active_models])
                    p = float(probs @ w / w.sum())
                n_correct += int((p > 0.5) == r["actual"])
                n_total += 1

    ci_lo, ci_hi = wilson_interval(n_correct, n_total)
    signal = "BUY" if prob_up >= 0.5 else "SELL"
    confidence = abs(prob_up - 0.5) * 2 * 100
    conviction = conviction_label(prob_up, ci_lo)

    return {
        "ticker": ticker,
        "signal": signal,
        "probability_up": round(float(prob_up), 4),
        "confidence": round(float(confidence), 2),
        "conviction": conviction,
        "confidence_interval": [round(ci_lo, 4), round(ci_hi, 4)],
        "confidence_interval_basis": {
            "window": ci_basis,
            "n_samples": n_total,
            "regime_filtered": True,
        },
        "method": weight_method,
        "weights": weights,
        "model_performance": dyn["accuracies"],
        "performance_window_days": window,
        "performance_n_samples": dyn["n_samples"],
        "excluded_models": dyn.get("excluded", []),
        "regime": {
            "label": regime_info["regime"],
            "key": regime_info["regime_key"],
            "probabilities": {k: round(float(v), 4) for k, v in regime_info["probabilities"].items()},
        },
        "risk": regime_info["risk"],
        "models": (
            {n: {"prob_up": round(float(last[f"{n}_prob"]), 4),
                 "direction": int(last[n])} for n in active_models}
            if last is not None else {}
        ),
    }
