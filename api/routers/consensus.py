"""Consensus signal endpoint with Meta-Controller."""

import os
import logging
import warnings as _warnings
import joblib
from fastapi import APIRouter
from src.data.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.data.features import add_technical_indicators
from src.models.model import load_models, models_exist
from src.models.ensemble import predict_ensemble
from src.models.trainer import FEATURE_COLS
from src.signals.regime import detect_regime
from src.signals.sentiment import get_stock_sentiment
from api.utils import parallel_fetch

logger = logging.getLogger(__name__)
router = APIRouter()


def _load(ticker):
    result = load_models(ticker)
    lstm, gru, transformer, xgb, scaler, features, lgb = result
    return {"lstm": lstm, "gru": gru, "transformer": transformer, "xgb": xgb, "scaler": scaler, "features": features, "lgb": lgb}


def _load_meta_controller():
    mc_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "meta_controller.pkl")
    if os.path.exists(mc_path):
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                return joblib.load(mc_path)
        except Exception:
            pass
    return None


def _consensus_one(ticker, meta_controller):
    df = fetch_stock_data(ticker, period="1y")
    if df is None or df.empty or len(df) < 50:
        return None

    df_feat = add_technical_indicators(df.copy(), ticker)
    close = df["close"]

    # 1. Ensemble signal
    ensemble_signal = "N/A"
    ensemble_conf = 0.0
    if models_exist(ticker):
        try:
            m = _load(ticker)
            direction, confidence, details = predict_ensemble(
                m["lstm"], m["gru"], m["transformer"],
                m["xgb"], m["scaler"], FEATURE_COLS, df_feat,
                lgb_model=m["lgb"],
            )
            ensemble_signal = "BUY" if direction == 1 else "SELL"
            ensemble_conf = float(details.get("ensemble_prob", confidence / 100)) if details else float(confidence) / 100
        except Exception:
            pass

    # 2. Meta-Controller signal
    meta_signal = "N/A"
    meta_conf = 0.0
    if meta_controller is not None and meta_controller.model is not None:
        try:
            from src.trading.risk import calculate_var, calculate_cvar, calculate_sharpe
            from src.signals.volatility import forecast_volatility

            returns = close.pct_change().dropna()
            var_val = calculate_var(returns) if len(returns) > 30 else 0
            cvar_val = calculate_cvar(returns) if len(returns) > 30 else 0
            sharpe_val = calculate_sharpe(returns) if len(returns) > 30 else 0
            vol_fc = forecast_volatility(returns) if len(returns) > 30 else {"current_vol": 0}
            vol_fc_val = vol_fc.get("current_vol", 0) if isinstance(vol_fc, dict) else vol_fc

            regime_result = detect_regime(close, ohlc=df)
            regime = regime_result.get("regime", "Sideways")
            regime_bull = 1.0 if regime == "Bull" else 0.0
            regime_bear = 1.0 if regime == "Bear" else 0.0

            ens_dir = 1.0 if ensemble_signal == "BUY" else 0.0
            ens_conf_val = ensemble_conf if ensemble_conf > 0 else 0.5

            sent_score = 0.0
            fii_n = 0.0
            dii_n = 0.0
            pcr_val = 1.0
            mtf_val = 0.0
            try:
                sent_result = get_stock_sentiment(ticker)
                sent_score = sent_result.get("weighted_score", sent_result.get("score", 0.0))
            except Exception:
                pass
            try:
                from src.signals.flow import fetch_fii_dii, get_flow_sentiment, fetch_options_pcr
                fii_dii = fetch_fii_dii()
                if len(fii_dii) > 0:
                    fii_raw = float(fii_dii.iloc[0]["fii_net"])
                    dii_raw = float(fii_dii.iloc[0]["dii_net"])
                    flow_sent = get_flow_sentiment(fii_raw, dii_raw)
                    flow_map = {"Strong Bullish": 1.0, "Bullish": 0.5, "Positive": 0.5,
                                "Neutral": 0.0, "Negative": -0.5, "Bearish": -0.5,
                                "Strong Bearish": -1.0, "Divergent": 0.0}
                    fii_n = flow_map.get(flow_sent, 0.0)
                    dii_n = fii_n
                pcr_data = fetch_options_pcr()
                pcr_val = pcr_data.get("pcr_oi", 1.0)
            except Exception:
                pass
            try:
                from src.signals.multitimeframe import fetch_mtf_data, get_combined_signal
                mtf_raw = fetch_mtf_data(ticker)
                if mtf_raw:
                    mtf_combined = get_combined_signal(mtf_raw)
                    mtf_val = mtf_combined.get("direction", 0) * mtf_combined.get("confidence", 0) / 100.0
            except Exception:
                pass

            state = {
                "ensemble_direction": ens_dir,
                "ensemble_confidence": float(ens_conf_val),
                "sentiment_score": float(sent_score),
                "fii_net": float(fii_n),
                "dii_net": float(dii_n),
                "pcr": float(pcr_val),
                "mtf_signal": float(mtf_val),
                "regime_bull": regime_bull,
                "regime_bear": regime_bear,
                "var_95": float(var_val),
                "cvar_95": float(cvar_val),
                "sharpe": float(sharpe_val),
                "volatility_forecast": float(vol_fc_val),
                "fundamental_score": 0.0,
            }
            decision = meta_controller.decide(state)
            meta_signal = decision.get("action", "N/A")
            meta_conf = decision.get("confidence", 0.0)
        except Exception:
            meta_signal = "N/A"

    # 3. Regime
    try:
        regime_result = detect_regime(close, ohlc=df)
        regime = regime_result.get("regime", "Sideways")
    except Exception:
        regime = "N/A"

    # 4. 3-signal voting consensus
    vote_signals = [s for s in [ensemble_signal, meta_signal, regime] if s not in ("N/A", None, "Unknown")]
    buy_votes = sum(1 for s in vote_signals if s in ("BUY", "Bull"))
    sell_votes = sum(1 for s in vote_signals if s in ("SELL", "Bear"))

    if buy_votes >= 3 or (buy_votes >= 2 and meta_signal == "BUY"):
        consensus = "STRONG BUY"
    elif buy_votes >= 1 and sell_votes == 0 and meta_signal == "BUY":
        consensus = "BUY"
    elif sell_votes >= 3 or (sell_votes >= 2 and meta_signal == "SELL"):
        consensus = "STRONG SELL"
    elif sell_votes >= 1 and buy_votes == 0 and meta_signal == "SELL":
        consensus = "SELL"
    elif buy_votes >= 1 and sell_votes >= 1:
        consensus = "CONFLICTED"
    else:
        consensus = "HOLD"

    return {
        "ticker": ticker,
        "ensemble_signal": ensemble_signal,
        "ensemble_confidence": round(ensemble_conf, 4),
        "meta_signal": meta_signal,
        "meta_confidence": round(meta_conf, 4) if isinstance(meta_conf, float) else 0.0,
        "regime": regime,
        "consensus": consensus,
    }


@router.get("/")
def get_consensus():
    try:
        meta_controller = _load_meta_controller()
        raw = parallel_fetch(
            lambda t: _consensus_one(t, meta_controller),
            NSE_STOCKS,
            max_workers=8,
        )
        results = [v for v in raw.values() if v is not None]

        buy_count = sum(1 for r in results if r["consensus"] in ("STRONG BUY", "BUY"))
        sell_count = sum(1 for r in results if r["consensus"] in ("STRONG SELL", "SELL"))
        hold_count = sum(1 for r in results if r["consensus"] == "HOLD")
        conflicted = sum(1 for r in results if r["consensus"] == "CONFLICTED")

        return {
            "results": results,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "conflicted": conflicted,
            "scanned": len(results),
            "total": len(NSE_STOCKS),
        }
    except Exception as e:
        return {"error": str(e)}
