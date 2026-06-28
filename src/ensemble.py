import numpy as np
import torch
from src.model import DEVICE


def predict_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat, recent_weights=None):
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

    if recent_weights is None:
        weights = {"lstm": 0.25, "gru": 0.25, "transformer": 0.25, "xgb": 0.25}
    else:
        weights = recent_weights

    dl_prob = (dir_lstm * weights["lstm"] +
               dir_gru * weights["gru"] +
               dir_transformer * weights["transformer"])
    dl_dir = 1 if dl_prob > 0.5 else 0
    dl_confidence = abs(dl_prob - 0.5) * 2

    xgb_weighted = xgb_prob[1] * weights["xgb"]

    ensemble_prob = dl_prob + xgb_weighted
    total_weight = sum(weights.values())
    if total_weight > 0:
        ensemble_prob = ensemble_prob / total_weight

    ensemble_dir = 1 if ensemble_prob > 0.5 else 0
    confidence = abs(ensemble_prob - 0.5) * 2 * 100

    details = {
        "lstm_dir": dir_lstm, "lstm_pred": pred_lstm,
        "gru_dir": dir_gru, "gru_pred": pred_gru,
        "transformer_dir": dir_transformer, "transformer_pred": pred_transformer,
        "xgb_dir": dir_xgb, "xgb_prob_up": float(xgb_prob[1]),
        "ensemble_prob": float(ensemble_prob),
        "dl_prob": dl_prob,
        "weights": weights,
    }
    return ensemble_dir, confidence, details


def backtest_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat, seq_length=60):
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

        d_lstm = 1 if p_lstm > prev_close else 0
        d_gru = 1 if p_gru > prev_close else 0
        d_tf = 1 if p_tf > prev_close else 0

        ens = (d_lstm + d_gru + d_tf) / 3
        dl_dir = 1 if ens > 0.5 else 0
        xgb_dir = int(xgb.predict(xgb_inp)[0])

        final = 1 if (ens + xgb_p) / 2 > 0.5 else 0

        results.append({
            "actual": actual_dir,
            "lstm": d_lstm, "gru": d_gru, "transformer": d_tf,
            "dl_ensemble": dl_dir, "xgb": xgb_dir,
            "final_ensemble": final,
        })

    return results
