# Model Training

Six base models per ticker, trained with walk-forward cross-validation,
combined by an ensemble, and steered by a meta-controller.

## Base models (`src/models/model.py`)

| Model | Architecture | Key params |
|---|---|---|
| StockLSTM | LSTM | hidden 128, 3 layers, dropout 0.3, fc 128→32→1 |
| StockGRU | GRU | hidden 64, 2 layers, dropout 0.2, fc 64→32→1 |
| StockTransformer | Transformer encoder | d_model 64, 4 heads, 3 layers, ff 256, learned positional encoding (max 120 steps) |
| XGBoost | Trees | 300 estimators, depth 4, lr 0.03, subsample 0.8, colsample 0.7 |
| LightGBM | Trees | 300 / depth 4 / lr 0.03, min_child_samples 30 |
| CatBoost | Trees | 300 iterations, depth 5, lr 0.05, l2 3.0 |

Device: `cuda` when available, else `cpu` (`src/models/model.py:15`) — CPU-only
torch is the default install; `scripts/setup_gpu.sh` / `.bat` swaps to a CUDA
build on NVIDIA machines (RTX 50-series → cu128).

## Trainer (`src/models/trainer.py`)

- Hyperparameters: SEQ_LENGTH=60, EPOCHS=40, BATCH_SIZE=32, LEARNING_RATE=0.001 (env-overridable).
- Input: 5 years of data, full SOTA feature pipeline + NIFTY index for relative strength; 58 training columns (`FEATURE_COLS`).
- **Trees**: purged & embargoed walk-forward (5 folds, embargo 5, horizon 1) via `PurgedGroupTimeSeriesSplit` (López de Prado); falls back to 80/20 chronological split when folds < 50 samples. OOS metrics: Sharpe/Sortino/Calmar/MaxDrawdown.
- **Deep learning**: 4 train folds + last test fold; scaler fit on training data only.
- Early stopping: ReduceLROnPlateau (factor 0.5, patience 3) + val-loss patience 8 epochs, best state restored.
- Adam, weight_decay 1e-5, MSELoss on next close (direction derived by comparison).
- Meta-learner per ticker: out-of-fold probability vector `[xgb, lgb, lstm, gru, transformer, cat]`, LogisticRegression trained when ≥50 samples → `models/meta_{TICKER}.pkl`.
- `batch_train()` trains all 20 tickers; `run_pipeline.py --train-all` triggers it.
- Model version "2", feature schema version "4".

## Ensemble (`src/models/ensemble.py`)

Combination priority:

1. **Meta-learner** — calibrated LogisticRegression (C=1.0, balanced) on stacked base probabilities.
2. **Dynamic regime-conditional weights** — softmax of excess accuracy (acc−0.5) at temperature 8.0 over trailing 30 days in the current regime (min 10 samples).
3. **Static regime weights** — e.g. Bull favors trees (0.20 each), Bear shifts weight to LSTM/GRU.
4. Equal weights (1/6) fallback.

`regime_adjusted_ensemble()` adds Wilson 90% confidence intervals on
regime-conditional OOS accuracy and conviction labels (HIGH/MEDIUM/LOW).
Confidence = `|prob − 0.5| × 2 × 100`. Requires ≥60 rows.

## Meta-controller (`src/models/meta_controller.py`)

The final decision layer — one action per ticker per day from **14 signals**:

`ensemble_direction, ensemble_confidence, sentiment_score, fii_net, dii_net,
pcr, mtf_signal, regime_bull, regime_bear, var_95, cvar_95, sharpe,
volatility_forecast, fundamental_score`

- Model: regularized LogisticRegression (C=0.1, balanced), optionally wrapped in `CalibratedClassifierCV`.
- Trained only with **≥500 resolved decisions** (via `TimeSeriesSplit`); backfill seeds this corpus.
- Decision thresholds: BUY ≥ 0.6, SELL ≤ 0.4, min confidence to trade 0.3; position size = `min(0.10, confidence × 0.10)`.
- Uncalibrated confidence discounted ×0.7.
- Rule-based fallback until trained; artifact `models/meta_controller.pkl` manifest-verified.

## Calibration (`src/models/calibration.py`)

- 5 confidence bins (width 0.2); per-bin mean confidence vs empirical accuracy from **resolved live decisions** (backfill excluded as in-sample).
- Metrics: **ECE** (expected calibration error) and **MCE** (max bin error) surfaced in `/api/ledger/calibration`.

## Artifacts & lifecycle

- Every bundle is manifest-verified before deserialization (see `Security-Model.md`): `models/{TICKER}_manifest.json` pins SHA-256 per file; feature-schema and training-dataset hashes recorded.
- Model registry (`models/registry/`): staging → production → archived lifecycle; `promote_model()` copies to `models/production/` and re-verifies; production inference reads only from there.
- Model cache: TTL 3600 s, max 10 bundles, 2048 MB memory budget.
- Retraining pipeline (`src/core/pipeline.py`) gates promotion on min OOS accuracy 0.50, min OOS Sharpe 0.0, max drawdown 0.30.
- Drift monitoring (`src/signals/monitoring.py`): performance drift >10% (critical) / >5% (warning), KS-test feature drift, prediction drift, data freshness → auto-enqueued retrain triggers (`data/monitoring/retrain_triggers.json`).
