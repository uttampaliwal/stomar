import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.model import (
    build_lstm, build_gru, build_transformer,
    build_xgb_model, build_lgb_model, save_models, DEVICE,
)
from src.constants import DEFAULT_SEQ_LENGTH, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE
from src.logging_config import get_logger

warnings.filterwarnings("ignore")

logger = get_logger("trainer")

FEATURE_COLS = [
    "close", "volume", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "bb_width", "atr", "obv", "volume_ratio",
    "high_low_pct", "close_open_pct", "close_position",
    "returns_1d", "returns_2d", "returns_3d", "returns_5d", "returns_10d", "returns_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "day_of_week", "month", "quarter", "day_of_month",
    "sentiment_score", "fii_net", "dii_net", "flow_signal",
    "pcr", "mtf_signal", "mtf_confidence",
]

SEQ_LENGTH = DEFAULT_SEQ_LENGTH
EPOCHS = DEFAULT_EPOCHS
BATCH_SIZE = DEFAULT_BATCH_SIZE
LEARNING_RATE = DEFAULT_LEARNING_RATE


def _train_one_model(model, train_loader, X_val, y_val, model_name, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_loss = float("inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vp = model(X_val).squeeze()
            val_loss = criterion(vp, y_val).item()

        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("%s epoch=%d train_loss=%.6f val_loss=%.6f", model_name, epoch + 1, train_loss / len(train_loader), val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                logger.info("%s early_stop epoch=%d", model_name, epoch + 1)
                break

    return model


def train_for_ticker(ticker: str, force_retrain: bool = False):
    logger.info("training_start ticker=%s force_retrain=%s", ticker, force_retrain)

    df = fetch_stock_data(ticker, period="5y", force_refresh=force_retrain)
    logger.info("data_fetched ticker=%s rows=%d", ticker, len(df))

    df_feat = add_technical_indicators(df, ticker=ticker)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("features_computed ticker=%s rows=%d", ticker, len(df_feat))

    lstm_features = [c for c in FEATURE_COLS if c in df_feat.columns]

    # --- XGBoost ---
    logger.info("training_xgboost ticker=%s", ticker)
    xgb_data = df_feat[lstm_features + ["target_direction"]].dropna()
    X_xgb = xgb_data[lstm_features]
    y_xgb = xgb_data["target_direction"]
    up_c, dn_c = int(y_xgb.sum()), len(y_xgb) - int(y_xgb.sum())
    logger.info("xgb_class_distribution ticker=%s up=%d down=%d", ticker, up_c, dn_c)

    X_tr, X_te, y_tr, y_te = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)
    xgb_model = build_xgb_model()
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pr = xgb_model.predict(X_te)
    xgb_acc = accuracy_score(y_te, y_pr)
    logger.info("xgb_trained ticker=%s accuracy=%.4f", ticker, xgb_acc)

    # --- LightGBM ---
    logger.info("training_lightgbm ticker=%s", ticker)
    lgb_model = build_lgb_model()
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    y_pr_lgb = lgb_model.predict(X_te)
    lgb_acc = accuracy_score(y_te, y_pr_lgb)
    logger.info("lgb_trained ticker=%s accuracy=%.4f", ticker, lgb_acc)

    # --- Deep Learning Models ---
    logger.info("training_neural_networks ticker=%s", ticker)
    lstm_data = df_feat[lstm_features].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(lstm_data)

    X_lstm, y_lstm = [], []
    for i in range(SEQ_LENGTH, len(scaled)):
        X_lstm.append(scaled[i - SEQ_LENGTH : i])
        y_lstm.append(scaled[i, 0])
    X_lstm = np.array(X_lstm, dtype=np.float32)
    y_lstm = np.array(y_lstm, dtype=np.float32)

    split = int(len(X_lstm) * 0.8)
    X_tr_l = torch.tensor(X_lstm[:split]).to(DEVICE)
    y_tr_l = torch.tensor(y_lstm[:split]).to(DEVICE)
    X_te_l = torch.tensor(X_lstm[split:]).to(DEVICE)
    y_te_l = torch.tensor(y_lstm[split:]).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr_l, y_tr_l), batch_size=BATCH_SIZE, shuffle=True)
    input_dim = X_lstm.shape[2]

    lstm_model = _train_one_model(build_lstm(input_dim), loader, X_te_l, y_te_l, "LSTM")
    gru_model = _train_one_model(build_gru(input_dim), loader, X_te_l, y_te_l, "GRU")
    tf_model = _train_one_model(build_transformer(input_dim), loader, X_te_l, y_te_l, "Transformer")

    # Evaluate direction accuracy
    def eval_dir(model, X_te, y_te, scaled_data, split_idx):
        model.eval()
        with torch.no_grad():
            preds = model(X_te).squeeze().cpu().numpy()
        actual = y_lstm[split_idx:]
        inp_close = scaled_data[split_idx + SEQ_LENGTH - 1 : -1, 0]
        m = min(len(preds), len(inp_close), len(actual))
        preds, inp_close, actual = preds[:m], inp_close[:m], actual[:m]
        d_pred = (preds > inp_close).astype(int)
        d_act = (actual > inp_close).astype(int)
        return accuracy_score(d_act, d_pred)

    lstm_acc = eval_dir(lstm_model, X_te_l, y_te_l, scaled, split)
    gru_acc = eval_dir(gru_model, X_te_l, y_te_l, scaled, split)
    tf_acc = eval_dir(tf_model, X_te_l, y_te_l, scaled, split)

    logger.info("dl_direction_accuracy ticker=%s lstm=%.4f gru=%.4f transformer=%.4f", ticker, lstm_acc, gru_acc, tf_acc)

    # Ensemble backtest (TEST SET ONLY)
    from src.ensemble import backtest_ensemble
    test_start_idx = split + SEQ_LENGTH
    df_test = df_feat.iloc[test_start_idx:].copy()
    if len(df_test) > SEQ_LENGTH:
        bt = backtest_ensemble(lstm_model, gru_model, tf_model, xgb_model, scaler, lstm_features, df_test, lgb_model=lgb_model)
        if bt:
            final_dir = [r["final_ensemble"] for r in bt]
            actual_dir = [r["actual"] for r in bt]
            ensemble_acc = accuracy_score(actual_dir, final_dir)
            correct = int(np.sum(np.array(final_dir) == np.array(actual_dir)))
            logger.info("ensemble_test ticker=%s accuracy=%.4f correct=%d/%d", ticker, ensemble_acc, correct, len(final_dir))
        else:
            ensemble_acc = 0.0
    else:
        ensemble_acc = 0.0

    save_models(lstm_model, gru_model, tf_model, xgb_model, scaler, lstm_features, ticker, lgb_model=lgb_model)
    logger.info("training_complete ticker=%s xgb=%.4f lgb=%.4f lstm=%.4f gru=%.4f tf=%.4f ensemble=%.4f",
                ticker, xgb_acc, lgb_acc, lstm_acc, gru_acc, tf_acc, ensemble_acc)

    return {
        "xgb_accuracy": float(xgb_acc),
        "lgb_accuracy": float(lgb_acc),
        "lstm_accuracy": float(lstm_acc),
        "gru_accuracy": float(gru_acc),
        "transformer_accuracy": float(tf_acc),
        "ensemble_accuracy": float(ensemble_acc),
        "n_samples": len(df_feat),
    }


def train_multiple_stocks(tickers: list, force_retrain: bool = False):
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = train_for_ticker(ticker, force_retrain)
        except Exception as e:
            logger.error("training_failed ticker=%s error=%s", ticker, str(e))
            results[ticker] = {"error": str(e)}
    return results


def batch_train(tickers: list = None, force_retrain: bool = False) -> dict:
    """Train all configured tickers and return structured summary.

    Returns:
        {
            "success": ["RELIANCE.NS", ...],
            "failed": {"TCS.NS": "error msg"},
            "skipped": ["HDFCBANK.NS", ...],
            "duration_seconds": 123.4,
            "results": {ticker: result_dict, ...}
        }
    """
    import time

    if tickers is None:
        from src.data_fetcher import NSE_STOCKS
        tickers = list(NSE_STOCKS)

    start = time.time()
    results = train_multiple_stocks(tickers, force_retrain)
    elapsed = time.time() - start

    summary = {
        "success": [],
        "failed": {},
        "skipped": [],
        "duration_seconds": round(elapsed, 1),
        "results": results,
    }

    for ticker, result in results.items():
        if isinstance(result, dict) and "error" in result:
            summary["failed"][ticker] = result["error"]
        elif isinstance(result, dict) and result.get("skipped"):
            summary["skipped"].append(ticker)
        else:
            summary["success"].append(ticker)

    return summary
