"""Model interpretability: SHAP feature contributions with human-readable text.

Produces per-prediction explanations such as:

    "RSI_14 contributed +12% to the BUY signal (value 68.4, bullish range)"

SHAP values are computed with shap.TreeExplainer on the XGBoost model
(fastest available path); if the shap package is unavailable, xgboost's
native ``pred_contribs`` (identical SHAP values for tree models) is used as
a drop-in replacement, falling back to gain-based importance.

Usage:
    from src.signals.interpretability import explain_prediction
    explanation = explain_prediction(ticker, df_feat, feature_cols)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Human-readable labels for common factors
FEATURE_LABELS = {
    "rsi": "RSI_14",
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "stoch_k": "Stochastic %K",
    "stoch_d": "Stochastic %D",
    "williams_r": "Williams %R",
    "mfi": "Money Flow Index",
    "cci": "Commodity Channel Index",
    "adx": "ADX",
    "atr": "ATR",
    "sma_10": "10-day SMA",
    "sma_20": "20-day SMA",
    "sma_50": "50-day SMA",
    "ema_12": "12-day EMA",
    "ema_26": "26-day EMA",
    "bb_high": "Bollinger upper band",
    "bb_low": "Bollinger lower band",
    "bb_width": "Bollinger width",
    "obv": "On-Balance Volume",
    "cmf": "Chaikin Money Flow",
    "volume_ratio": "Volume ratio",
    "volume_sma": "Volume SMA",
    "supertrend": "Supertrend",
    "supertrend_dir": "Supertrend direction",
    "ichimoku_tenkan": "Ichimoku tenkan-sen",
    "ichimoku_kijun": "Ichimoku kijun-sen",
    "ichimoku_senkou_a": "Ichimoku senkou A",
    "ichimoku_senkou_b": "Ichimoku senkou B",
    "ichimoku_chikou": "Ichimoku chikou",
    "ofi": "Order Flow Imbalance",
    "vol_zscore": "Volatility Z-score",
    "pcr": "Put-Call Ratio",
    "pcr_slope": "PCR slope",
    "fii_net": "FII net flow",
    "dii_net": "DII net flow",
    "fii_momentum": "FII flow momentum",
    "mom_1m_voladj": "1M vol-adj momentum",
    "mom_3m_voladj": "3M vol-adj momentum",
    "mom_6m_voladj": "6M vol-adj momentum",
    "mom_12m_voladj": "12M vol-adj momentum",
    "rel_strength_1m": "Relative strength 1M vs NIFTY",
    "rel_strength_3m": "Relative strength 3M vs NIFTY",
    "high_low_spread": "High-Low range spread",
    "sentiment_score": "News sentiment",
    "mtf_signal": "Multi-timeframe signal",
    "mtf_confidence": "MTF confidence",
    "vwap": "VWAP",
    "close_position": "Close position in range",
    "high_low_pct": "Daily range %",
    "return_vol_corr": "Return-volume correlation",
}


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def explain_prediction_shap(
    ticker: str,
    df_feat: pd.DataFrame,
    feature_cols: list,
    top_n: int = 10,
    base_value: float | None = None,
) -> dict:
    """SHAP-based explanation with human-readable contributions.

    Args:
        ticker: Stock ticker symbol.
        df_feat: DataFrame with features.
        feature_cols: List of feature column names (model order).
        top_n: Number of top contributing features to report.
        base_value: Optional logit base value (avoids re-running the
            explainer when the caller already has it).

    Returns:
        Dict with SHAP values, contribution percentages, and a
        `summary` list of human-readable sentences.
    """
    from src.models.model import load_models, models_exist

    if not models_exist(ticker):
        return {"error": f"No trained models for {ticker}"}

    tup = load_models(ticker)
    xgb_model = tup[3]

    # Use the model's own training columns when available (legacy models
    # were trained with fewer features than the current FEATURE_COLS)
    model_cols = getattr(xgb_model, "feature_names_in_", None)
    ref_cols = list(model_cols) if model_cols is not None else feature_cols
    valid_cols = [c for c in ref_cols if c in df_feat.columns]
    # drop all-NaN columns (e.g. relative strength without an index) first,
    # then rows with any missing value
    X = df_feat[valid_cols].dropna(axis=1, how="all").dropna()
    if len(X) < 1:
        return {"error": "No valid data for explanation"}
    explained_cols = list(X.columns)

    latest = X.iloc[-1:]

    try:
        import shap

        explainer = shap.TreeExplainer(xgb_model)
        if base_value is None:
            ev = explainer.expected_value
            base_value = _coerce_base_value(ev)
        raw_values = explainer.shap_values(latest)
        method = "shap"
    except ImportError:
        logger.info("shap not installed — using xgboost pred_contribs (identical SHAP values)")
        method = "shap_pred_contribs"
        raw_values = _xgb_pred_contribs(xgb_model, latest)
        if base_value is None:
            # For binary logit, pred_contribs returns [base, f1..fn]
            base_value = float(raw_values[0, -1])
            raw_values = raw_values[:, :-1]

    if isinstance(raw_values, list):
        sv = np.asarray(raw_values[1]) if len(raw_values) > 1 else np.asarray(raw_values[0])
    else:
        sv = np.asarray(raw_values)
    if sv.ndim > 2:
        sv = sv[1] if sv.shape[0] > 1 else sv[0]
    if sv.ndim > 1:
        sv = sv.reshape(1, -1)

    contributions = []
    for i, feat in enumerate(explained_cols):
        if i >= sv.shape[1]:
            break
        contributions.append({"feature": feat, "shap_value": float(sv[0, i])})

    # Normalize to percentage contribution of the signal magnitude
    total_abs = sum(abs(c["shap_value"]) for c in contributions)
    for c in contributions:
        c["pct_contribution"] = round(abs(c["shap_value"]) / total_abs * 100, 2) if total_abs > 0 else 0.0

    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    top = contributions[:top_n]

    # Signal direction from the model's own decision
    try:
        prob_up = float(xgb_model.predict_proba(latest)[0][1])
    except Exception:
        prob_up = 0.5
    signal = "BUY" if prob_up >= 0.5 else "SELL"

    summary = _build_summary(top, latest, signal, base_value, method)

    return {
        "ticker": ticker,
        "method": method,
        "signal": signal,
        "probability_up": round(prob_up, 4),
        "base_value": round(float(base_value), 6),
        "top_features": top,
        "summary": summary,
        "total_features": len(explained_cols),
    }


def _coerce_base_value(ev) -> float:
    """Extract the log-odds base value from a shap expected_value.

    Binary tree models return a scalar, a 0-d array, a 1-element array, or a
    (1, n_classes) array depending on the shap/xgboost version.
    """
    arr = np.asarray(ev).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[1] if arr.size > 1 else arr[0])


def _xgb_pred_contribs(model, X: pd.DataFrame) -> np.ndarray:
    """Native SHAP contributions from xgboost (no shap dependency).

    Modern xgboost removed ``pred_contribs`` from the sklearn wrapper's
    ``predict``, so the booster path is used (identical SHAP values).
    """
    import xgboost as xgb

    dmatrix = xgb.DMatrix(X)
    return model.get_booster().predict(dmatrix, pred_contribs=True)


def _build_summary(
    top_contributions: list[dict],
    latest: pd.DataFrame,
    signal: str,
    base_value: float | None,
    method: str,
) -> list[str]:
    """Convert SHAP contributions into human-readable sentences."""
    sentences = [
        f"Model signals {signal} based on the most recent bar."
    ]
    for c in top_contributions[:8]:
        feat = c["feature"]
        label = feature_label(feat)
        pct = c["pct_contribution"]
        direction = "bullish" if c["shap_value"] > 0 else "bearish"
        magnitude = "strongly" if abs(c["shap_value"]) > 0.5 * max(
            (abs(x["shap_value"]) for x in top_contributions), default=1
        ) else "moderately"
        val = latest[feat].iloc[0] if feat in latest.columns else None
        val_str = f" (current value {float(val):.4g})" if val is not None and pd.notna(val) else ""
        sentences.append(
            f"{label} contributed {magnitude} to the {signal} signal, "
            f"pushing {direction} by {abs(c['shap_value']):.4f} ({pct:.1f}% of total){val_str}."
        )
    return sentences


def explain_prediction(ticker: str, df_feat: pd.DataFrame,
                       feature_cols: list, top_n: int = 10) -> dict:
    """Explain the current prediction with SHAP (or gain fallback).

    Thin wrapper that prefers SHAP and falls back to the gain-based
    explanation when the model is unavailable or computation fails.
    """
    try:
        return explain_prediction_shap(ticker, df_feat, feature_cols, top_n)
    except Exception as exc:
        logger.warning("SHAP explanation failed for %s: %s — gain fallback", ticker, exc)
        return _explain_gain(ticker, df_feat, feature_cols, top_n)


def _explain_gain(ticker: str, df_feat: pd.DataFrame,
                  feature_cols: list, top_n: int = 10) -> dict:
    """Gain-based explanation (no SHAP package required)."""
    try:
        from src.models.model import load_models, models_exist
    except ImportError:
        return {"error": "Models unavailable"}
    if not models_exist(ticker):
        return {"error": f"No trained models for {ticker}"}

    tup = load_models(ticker)
    xgb_model = tup[3]
    lgb_model = tup[6] if len(tup) > 6 else None

    model_cols = getattr(xgb_model, "feature_names_in_", None)
    ref_cols = list(model_cols) if model_cols is not None else feature_cols
    valid_cols = [c for c in ref_cols if c in df_feat.columns]
    xgb_imp = np.asarray(xgb_model.feature_importances_)
    valid_cols = valid_cols[: len(xgb_imp)]

    importance = xgb_imp.astype(float).copy()
    if lgb_model is not None:
        lgb_imp = np.asarray(lgb_model.feature_importances_)[: len(valid_cols)]
        importance = (importance + lgb_imp) / 2
    total = importance.sum()
    pct = importance / total * 100 if total > 0 else np.zeros_like(importance)

    order = np.argsort(importance)[::-1][:top_n]
    latest = df_feat.iloc[-1] if len(df_feat) else None
    top = []
    for i in order:
        feat = valid_cols[i]
        val = float(latest[feat]) if latest is not None and feat in latest.index and pd.notna(latest[feat]) else None
        top.append({
            "feature": feat,
            "shap_value": round(float(importance[i]) / max(total, 1e-9), 6),
            "pct_contribution": round(float(pct[i]), 2),
            "current_value": round(val, 6) if val is not None else None,
        })

    return {
        "ticker": ticker,
        "method": "gain",
        "signal": "BUY" if latest is not None and latest.get("target_direction", 1) == 1 else "SELL",
        "probability_up": None,
        "base_value": None,
        "top_features": top,
        "summary": [
            f"{feature_label(c['feature'])} contributed {c['pct_contribution']:.1f}% of model importance."
            for c in top[:8]
        ],
        "total_features": len(valid_cols),
    }
