import pandas as pd
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
from src.model import build_lstm, build_xgb_model, save_models, DEVICE

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "close", "volume", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "bb_width", "atr", "obv", "volume_ratio",
    "high_low_pct", "close_open_pct", "close_position",
    "returns_1d", "returns_2d", "returns_3d", "returns_5d", "returns_10d", "returns_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "day_of_week", "month", "quarter", "day_of_month",
]

SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def train_for_ticker(ticker: str, force_retrain: bool = False):
    print(f"\n{'='*50}")
    print(f"Training models for {ticker}")
    print(f"{'='*50}")

    df = fetch_stock_data(ticker, period="5y", force_refresh=force_retrain)
    print(f"Fetched {len(df)} rows of data")

    df_feat = add_technical_indicators(df)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After feature engineering: {len(df_feat)} rows")

    lstm_features = [c for c in FEATURE_COLS if c in df_feat.columns]
    xgb_features = lstm_features + ["target_direction"]

    # --- XGBoost Training ---
    print("\nTraining XGBoost (balanced)...")
    xgb_data = df_feat[xgb_features].dropna()
    X_xgb = xgb_data[lstm_features]
    y_xgb = xgb_data["target_direction"]

    # Balance check
    up_count = int(y_xgb.sum())
    down_count = len(y_xgb) - up_count
    print(f"  Up days: {up_count}, Down days: {down_count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_xgb, y_xgb, test_size=0.2, shuffle=False
    )

    xgb_model = build_xgb_model()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    print(f"  XGBoost Test Accuracy: {acc:.4f}")
    print(f"  XGBoost Avg UP prob: {y_prob.mean():.4f}")

    # --- PyTorch LSTM Training ---
    print("\nTraining LSTM...")
    lstm_data = df_feat[lstm_features].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(lstm_data)

    X_lstm, y_lstm = [], []
    for i in range(SEQ_LENGTH, len(scaled)):
        X_lstm.append(scaled[i - SEQ_LENGTH : i])
        y_lstm.append(scaled[i, 0])

    X_lstm = np.array(X_lstm, dtype=np.float32)
    y_lstm = np.array(y_lstm, dtype=np.float32)

    split_idx = int(len(X_lstm) * 0.8)
    X_train_lstm = torch.tensor(X_lstm[:split_idx]).to(DEVICE)
    y_train_lstm = torch.tensor(y_lstm[:split_idx]).to(DEVICE)
    X_test_lstm = torch.tensor(X_lstm[split_idx:]).to(DEVICE)
    y_test_lstm = torch.tensor(y_lstm[split_idx:]).to(DEVICE)

    train_dataset = TensorDataset(X_train_lstm, y_train_lstm)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    lstm_model = build_lstm(input_dim=X_lstm.shape[2])
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        lstm_model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        lstm_model.eval()
        with torch.no_grad():
            val_preds = lstm_model(X_test_lstm).squeeze()
            val_loss = criterion(val_preds, y_test_lstm).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} - Train Loss: {train_loss/len(train_loader):.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Evaluate LSTM direction accuracy
    lstm_model.eval()
    with torch.no_grad():
        lstm_pred = lstm_model(X_test_lstm).squeeze().cpu().numpy()

    lstm_actual_vals = y_lstm[split_idx:]
    last_input_close = scaled[split_idx + SEQ_LENGTH - 1 : -1, 0]

    min_len = min(len(lstm_pred), len(last_input_close), len(lstm_actual_vals))
    lstm_pred = lstm_pred[:min_len]
    last_input_close = last_input_close[:min_len]
    lstm_actual_vals = lstm_actual_vals[:min_len]

    lstm_direction = (lstm_pred > last_input_close).astype(int)
    lstm_actual_dir = (lstm_actual_vals > last_input_close).astype(int)
    lstm_acc = accuracy_score(lstm_actual_dir, lstm_direction)
    correct = int(np.sum(lstm_direction == lstm_actual_dir))
    print(f"  LSTM Direction Accuracy: {lstm_acc:.4f} ({correct}/{min_len} correct)")

    save_models(lstm_model, xgb_model, scaler, lstm_features, ticker)
    print(f"Models saved for {ticker}")

    return {
        "xgb_accuracy": float(acc),
        "lstm_accuracy": float(lstm_acc),
        "lstm_correct": correct,
        "lstm_total": min_len,
        "n_samples": len(df_feat),
    }


def train_multiple_stocks(tickers: list, force_retrain: bool = False):
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = train_for_ticker(ticker, force_retrain)
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            results[ticker] = {"error": str(e)}
    return results
