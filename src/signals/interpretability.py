"""Model interpretability using SHAP values.

Provides feature importance and prediction explanations for the XGBoost
and LightGBM models in the ensemble. This is critical for trusting
trading decisions — you need to know WHY the model says BUY or SELL.

Usage:
    from src.signals.interpretability import explain_prediction
    explanation = explain_prediction(ticker, df_feat, feature_cols)
    # Returns top features driving the current prediction
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def explain_prediction(ticker: str, df_feat: pd.DataFrame,
                       feature_cols: list, top_n: int = 10) -> dict:
    """Explain the current prediction using feature importance.

    Uses tree-based model's built-in feature importance (gain) as a
    fast, dependency-free alternative to SHAP. For full SHAP analysis,
    install shap and use explain_prediction_shap().

    Args:
        ticker: Stock ticker symbol
        df_feat: DataFrame with features
        feature_cols: List of feature column names
        top_n: Number of top features to return

    Returns:
        Dict with feature importances, current values, and direction
    """
    try:
        from src.models.model import load_models, models_exist

        if not models_exist(ticker):
            return {"error": f"No trained models for {ticker}"}

        tup = load_models(ticker)
        xgb_model = tup[3]
        lgb_model = tup[6] if len(tup) > 6 else None

        # Get feature importance from XGBoost (gain-based)
        xgb_importance = xgb_model.feature_importances_
        valid_cols = [c for c in feature_cols if c in df_feat.columns]
        if len(xgb_importance) != len(valid_cols):
            valid_cols = valid_cols[:len(xgb_importance)]

        importance_df = pd.DataFrame({
            "feature": valid_cols,
            "xgb_importance": xgb_importance[:len(valid_cols)],
        })

        # Add LightGBM importance if available
        if lgb_model is not None:
            lgb_imp = lgb_model.feature_importances_
            importance_df["lgb_importance"] = lgb_imp[:len(valid_cols)]
            importance_df["importance"] = (
                importance_df["xgb_importance"] + importance_df["lgb_importance"]
            ) / 2
        else:
            importance_df["importance"] = importance_df["xgb_importance"]

        # Normalize
        total = importance_df["importance"].sum()
        if total > 0:
            importance_df["pct"] = importance_df["importance"] / total * 100
        else:
            importance_df["pct"] = 0

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        # Get current values of top features
        current_values = {}
        if len(df_feat) > 0:
            latest = df_feat.iloc[-1]
            for _, row in importance_df.head(top_n).iterrows():
                feat = row["feature"]
                if feat in latest.index:
                    val = latest[feat]
                    current_values[feat] = round(float(val), 4) if pd.notna(val) else None

        # Compute direction signals
        directions = {}
        for feat in list(current_values.keys())[:top_n]:
            val = current_values.get(feat)
            if val is None:
                continue
            # Simple heuristic: positive values suggest bullish, negative bearish
            if feat in ("rsi", "mfi", "stoch_k"):
                if val > 70:
                    directions[feat] = "overbought (bearish)"
                elif val < 30:
                    directions[feat] = "oversold (bullish)"
                else:
                    directions[feat] = "neutral"
            elif feat in ("adx",):
                directions[feat] = "strong trend" if val > 25 else "weak trend"
            elif feat in ("macd",):
                directions[feat] = "bullish" if val > 0 else "bearish"
            elif feat in ("volume_ratio",):
                directions[feat] = "high volume" if val > 1.5 else "normal"
            elif feat in ("volatility_5d", "volatility_10d", "volatility_20d"):
                directions[feat] = "high vol" if val > 0.02 else "low vol"
            else:
                directions[feat] = "positive" if val > 0 else "negative"

        top_features = importance_df.head(top_n).to_dict("records")

        return {
            "ticker": ticker,
            "top_features": top_features,
            "current_values": current_values,
            "directions": directions,
            "total_features": len(valid_cols),
        }

    except Exception as e:
        logger.warning(f"Interpretability failed for {ticker}: {e}")
        return {"error": str(e)}


def explain_prediction_shap(ticker: str, df_feat: pd.DataFrame,
                            feature_cols: list, top_n: int = 10) -> dict:
    """Full SHAP-based explanation (requires shap package).

    Falls back to built-in importance if shap is not installed.

    Args:
        ticker: Stock ticker symbol
        df_feat: DataFrame with features
        feature_cols: List of feature column names
        top_n: Number of top features to return

    Returns:
        Dict with SHAP values and feature contributions
    """
    try:
        import shap
    except ImportError:
        logger.info("shap not installed, using built-in importance")
        return explain_prediction(ticker, df_feat, feature_cols, top_n)

    try:
        from src.models.model import load_models, models_exist

        if not models_exist(ticker):
            return {"error": f"No trained models for {ticker}"}

        tup = load_models(ticker)
        xgb_model = tup[3]

        valid_cols = [c for c in feature_cols if c in df_feat.columns]
        X = df_feat[valid_cols].dropna()

        if len(X) < 1:
            return {"error": "No valid data for explanation"}

        # Use TreeExplainer for XGBoost (fast)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X.iloc[-1:])

        # Get top features by absolute SHAP value
        if isinstance(shap_values, list):
            sv = shap_values[1]  # class 1 (up)
        else:
            sv = shap_values

        sv = sv.flatten()
        feature_shap = list(zip(valid_cols, sv))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = [
            {"feature": feat, "shap_value": round(float(val), 6), "direction": "bullish" if val > 0 else "bearish"}
            for feat, val in feature_shap[:top_n]
        ]

        return {
            "ticker": ticker,
            "method": "shap",
            "top_features": top_features,
            "base_value": round(float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0]), 6),
            "total_features": len(valid_cols),
        }

    except Exception as e:
        logger.warning(f"SHAP explanation failed for {ticker}: {e}")
        return explain_prediction(ticker, df_feat, feature_cols, top_n)
