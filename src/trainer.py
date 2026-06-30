import numpy as np
import os
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
    "stoch_k", "stoch_d", "williams_r", "cci", "mfi",
    "adx", "vwap",
]

SEQ_LENGTH = DEFAULT_SEQ_LENGTH
EPOCHS = DEFAULT_EPOCHS
BATCH_SIZE = DEFAULT_BATCH_SIZE
LEARNING_RATE = DEFAULT_LEARNING_RATE


def _collect_meta_features(lstm, gru, transformer, xgb, scaler, feature_cols,
                           df_feat, split_idx, seq_length, lgb_model=None):
    """Collect out-of-fold predictions from base models for meta-learner training.

    Returns (X_meta, y_meta) where X_meta is (N, 5) with columns:
    [xgb_prob_up, lgb_prob_up, lstm_prob, gru_prob, transformer_prob]
    and y_meta is binary direction labels.
    """
    feature_cols_valid = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols_valid].dropna()
    if len(data) < seq_length + split_idx + 10:
        return None

    scaled = scaler.transform(data.values)
    n_samples = len(scaled) - seq_length - split_idx
    if n_samples < 50:
        return None

    meta_X = []
    meta_y = []

    lstm.eval()
    gru.eval()
    transformer.eval()

    for i in range(split_idx + seq_length, len(scaled)):
        inp = torch.tensor(scaled[i - seq_length:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prev_close = scaled[i - 1, 0]
        actual_close = scaled[i, 0]
        actual_dir = 1 if actual_close > prev_close else 0

        with torch.no_grad():
            p_lstm = lstm(inp).item()
            p_gru = gru(inp).item()
            p_tf = transformer(inp).item()

        diff_lstm = p_lstm - prev_close
        diff_gru = p_gru - prev_close
        diff_tf = p_tf - prev_close
        prob_lstm = 1.0 / (1.0 + np.exp(-diff_lstm * 10))
        prob_gru = 1.0 / (1.0 + np.exp(-diff_gru * 10))
        prob_tf = 1.0 / (1.0 + np.exp(-diff_tf * 10))

        xgb_inp = data.iloc[[i - seq_length]]
        xgb_p = xgb.predict_proba(xgb_inp)[0][1]

        lgb_p = 0.5
        if lgb_model is not None:
            lgb_p = lgb_model.predict_proba(xgb_inp)[0][1]

        meta_X.append([xgb_p, lgb_p, prob_lstm, prob_gru, prob_tf])
        meta_y.append(actual_dir)

    return np.array(meta_X), np.array(meta_y)


def _train_one_model(model, train_loader, X_val, y_val, model_name, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
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
            if patience >= 8:
                logger.info("%s early_stop epoch=%d", model_name, epoch + 1)
                break

    return model


def _walk_forward_xgb(X, y, ticker, n_splits=5):
    """Walk-forward validation for XGBoost/LightGBM.

    Splits data into n_splits chronological folds. Each fold trains on all
    prior data and tests on the next chunk. Returns per-fold accuracies.
    """
    fold_size = len(X) // n_splits
    if fold_size < 50:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_tr, X_te, y_tr, y_te, []

    fold_accs = []
    best_acc = 0
    best_fold = 0

    for fold in range(n_splits):
        test_start = (fold + 1) * fold_size
        test_end = min(test_start + fold_size, len(X))
        if test_start >= len(X):
            break

        X_tr_fold = X.iloc[:test_start]
        y_tr_fold = y.iloc[:test_start]
        X_te_fold = X.iloc[test_start:test_end]
        y_te_fold = y.iloc[test_start:test_end]

        xgb_model = build_xgb_model()
        xgb_model.fit(X_tr_fold, y_tr_fold, eval_set=[(X_te_fold, y_te_fold)], verbose=False)
        y_pred = xgb_model.predict(X_te_fold)
        acc = accuracy_score(y_te_fold, y_pred)
        fold_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_fold = fold

    # Train final model on first (n_splits-1) folds, test on last fold
    final_test_start = (n_splits - 1) * fold_size
    if final_test_start >= len(X):
        X_tr_final, X_te_final, y_tr_final, y_te_final = train_test_split(X, y, test_size=0.2, shuffle=False)
    else:
        X_tr_final = X.iloc[:final_test_start]
        y_tr_final = y.iloc[:final_test_start]
        X_te_final = X.iloc[final_test_start:]
        y_te_final = y.iloc[final_test_start:]

    logger.info("walk_forward ticker=%s fold_accuracies=%s best_fold=%d",
                ticker, [round(a, 4) for a in fold_accs], best_fold)

    return X_tr_final, X_te_final, y_tr_final, y_te_final, fold_accs


def _walk_forward_dl(scaled, seq_length, n_splits=5):
    """Walk-forward split for deep learning models.

    Uses the first (n_splits-1) folds for training and the last fold for testing.
    Ensures test set has at least seq_length rows for ensemble backtest.
    """
    total = len(scaled)
    # Reduce n_splits if test set would be too small
    while n_splits > 2:
        fold_size = total // n_splits
        test_size = total - (n_splits - 1) * fold_size
        if test_size >= seq_length + 10:
            break
        n_splits -= 1

    fold_size = total // n_splits
    if fold_size < 10:
        split = int(total * 0.8)
        if split < seq_length + 10:
            return None, []
        return split, []

    # Use (n_splits-1) folds for training, last fold for testing
    split = (n_splits - 1) * fold_size
    if split < seq_length + 10:
        split = seq_length + 10
    return split, []


def train_for_ticker(ticker: str, force_retrain: bool = False):
    logger.info("training_start ticker=%s force_retrain=%s", ticker, force_retrain)

    df = fetch_stock_data(ticker, period="5y", force_refresh=force_retrain)
    logger.info("data_fetched ticker=%s rows=%d", ticker, len(df))

    df_feat = add_technical_indicators(df, ticker=ticker)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("features_computed ticker=%s rows=%d", ticker, len(df_feat))

    lstm_features = [c for c in FEATURE_COLS if c in df_feat.columns]

    # --- XGBoost (with walk-forward) ---
    logger.info("training_xgboost ticker=%s", ticker)
    xgb_data = df_feat[lstm_features + ["target_direction"]].dropna()
    X_xgb = xgb_data[lstm_features]
    y_xgb = xgb_data["target_direction"]
    up_c, dn_c = int(y_xgb.sum()), len(y_xgb) - int(y_xgb.sum())
    logger.info("xgb_class_distribution ticker=%s up=%d down=%d", ticker, up_c, dn_c)

    X_tr, X_te, y_tr, y_te, fold_accs = _walk_forward_xgb(X_xgb, y_xgb, ticker)
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

    # Walk-forward split for DL
    dl_split, dl_folds = _walk_forward_dl(lstm_data, SEQ_LENGTH)

    lstm_acc = 0.0
    gru_acc = 0.0
    tf_acc = 0.0
    ensemble_acc = 0.0
    lstm_model = None
    gru_model = None
    tf_model = None
    scaler = MinMaxScaler()

    if dl_split is not None and dl_split >= SEQ_LENGTH + 10:
        # Fit scaler ONLY on training data (fix leakage)
        train_data = lstm_data[:dl_split]
        scaler.fit(train_data)
        scaled = scaler.transform(lstm_data)

        X_lstm, y_lstm = [], []
        for i in range(SEQ_LENGTH, len(scaled)):
            X_lstm.append(scaled[i - SEQ_LENGTH : i])
            y_lstm.append(scaled[i, 0])
        X_lstm = np.array(X_lstm, dtype=np.float32)
        y_lstm = np.array(y_lstm, dtype=np.float32)

        split = dl_split_to_lstm_split(dl_split, seq_length=SEQ_LENGTH, total=len(scaled))
        if split > 0 and len(X_lstm) > split:
            X_tr_l = torch.tensor(X_lstm[:split]).to(DEVICE)
            y_tr_l = torch.tensor(y_lstm[:split]).to(DEVICE)
            X_te_l = torch.tensor(X_lstm[split:]).to(DEVICE)
            y_te_l = torch.tensor(y_lstm[split:]).to(DEVICE)

            loader = DataLoader(TensorDataset(X_tr_l, y_tr_l), batch_size=BATCH_SIZE, shuffle=False)
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
        logger.info("skipping_dl ticker=%s reason=insufficient_data rows=%d", ticker, len(df_feat))
        split = 0

    if lstm_model is not None:
        save_models(lstm_model, gru_model, tf_model, xgb_model, scaler, lstm_features, ticker, lgb_model=lgb_model)
    else:
        # Save tree models and scaler only (no DL models)
        import joblib
        ticker_clean = ticker.replace(".", "_")
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(xgb_model, os.path.join(models_dir, f"{ticker_clean}_xgb.pkl"))
        if lgb_model is not None:
            joblib.dump(lgb_model, os.path.join(models_dir, f"{ticker_clean}_lgb.pkl"))
        joblib.dump(scaler, os.path.join(models_dir, f"{ticker_clean}_scaler.pkl"))
        joblib.dump(lstm_features, os.path.join(models_dir, f"{ticker_clean}_features.pkl"))

    logger.info("training_complete ticker=%s xgb=%.4f lgb=%.4f lstm=%.4f gru=%.4f tf=%.4f ensemble=%.4f",
                ticker, xgb_acc, lgb_acc, lstm_acc, gru_acc, tf_acc, ensemble_acc)

    # --- Train Meta-Learner on out-of-fold predictions ---
    meta_model = None
    try:
        from src.ensemble import train_meta_learner, save_meta_model
        meta_features = _collect_meta_features(
            lstm_model, gru_model, tf_model, xgb_model, scaler,
            lstm_features, df_feat, split, SEQ_LENGTH, lgb_model,
        )
        if meta_features is not None:
            X_meta, y_meta = meta_features
            if len(X_meta) >= 50:
                meta_model = train_meta_learner(X_meta, y_meta)
                meta_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"meta_{ticker.replace('.', '_')}.pkl")
                save_meta_model(meta_model, meta_path)
                logger.info("meta_learner_trained ticker=%s samples=%d", ticker, len(X_meta))
    except Exception as e:
        logger.debug(f"Meta-learner training failed: {e}")

    return {
        "xgb_accuracy": float(xgb_acc),
        "lgb_accuracy": float(lgb_acc),
        "lstm_accuracy": float(lstm_acc),
        "gru_accuracy": float(gru_acc),
        "transformer_accuracy": float(tf_acc),
        "ensemble_accuracy": float(ensemble_acc),
        "n_samples": len(df_feat),
    }


def dl_split_to_lstm_split(dl_split, seq_length, total):
    """Convert a raw data index split to an LSTM sequence split.

    The dl_split is an index into the raw scaled data. LSTM sequences start
    at index seq_length, so the split point in X_lstm/y_lstm arrays is:
    """
    lstm_split = dl_split - seq_length
    lstm_split = max(0, min(lstm_split, total - seq_length))
    return lstm_split


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
