# Architecture

System design, data flow, and module dependencies for StoMar.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                STREAMLIT DASHBOARD (19 tabs, app.py)             │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Predict  │Portfolio │ Backtest │ Scanner  │Consensus │      │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤      │
│  │Sentiment │Mkt Pulse│Optimizer │ Holdings │  Risk    │      │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤      │
│  │Volatility│ Ranking  │Scenarios │ Regime   │Monitoring│      │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤      │
│  │ Pipeline │Paper Trd │MF Tracker│ Ledger   │          │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
│         │           │           │           │                   │
│  ┌──────▼───────────▼──────────▼───────────▼──────┐            │
│  │              Orchestration Layer                 │            │
│  │  orchestrator.py  meta_controller.py  ledger.py │            │
│  └──────────────────┬──────────────────────────────┘            │
│                     │                                           │
│  ┌──────────────────▼──────────────────────────────┐            │
│  │              Signal Layer (14 modules)           │            │
│  │  ensemble  sentiment  flow  regime  volatility  │            │
│  │  ranking   risk  optimizer  mf_tracker  mtf     │            │
│  │  backtester  execution_quality  benchmarks      │            │
│  └──────────────────┬──────────────────────────────┘            │
│                     │                                           │
│  ┌──────────────────▼──────────────────────────────┐            │
│  │              Core Engine Layer                   │            │
│  │  trainer.py  features.py  model.py  backfill.py │            │
│  └──────────────────┬──────────────────────────────┘            │
│                     │                                           │
│  ┌──────────────────▼──────────────────────────────┐            │
│  │              Data Layer                          │            │
│  │  data_fetcher.py  data_sources.py  flow.py      │            │
│  │  data/ (parquet cache, SQLite ledger)            │            │
│  │  models/ (trained weights, meta-controller)      │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Map

```
app.py
├── src/data_fetcher.py     (no internal deps)
├── src/features.py
│   └── uses: sentiment.py, flow.py, multitimeframe.py
├── src/model.py            (no internal deps — pure PyTorch/sklearn)
├── src/trainer.py
│   ├── src/data_fetcher.py
│   ├── src/features.py
│   └── src/model.py
├── src/ensemble.py
│   └── src/model.py (for DEVICE)
├── src/backtester.py
│   └── src/portfolio.py
├── src/orchestrator.py
│   ├── src/data_fetcher.py
│   ├── src/features.py
│   ├── src/ensemble.py
│   ├── src/meta_controller.py
│   ├── src/ledger.py
│   ├── src/sentiment.py
│   ├── src/flow.py
│   ├── src/multitimeframe.py
│   ├── src/risk.py
│   ├── src/regime.py
│   ├── src/volatility.py
│   └── src/paper_trader.py
├── src/meta_controller.py
│   └── src/ledger.py (for training data)
├── src/ledger.py           (no internal deps — pure sqlite3)
├── src/paper_trader.py
│   └── src/risk_controls.py
├── src/portfolio.py        (no internal deps)
├── src/sentiment.py        (no internal deps — uses yfinance + HuggingFace)
├── src/flow.py             (no internal deps — uses requests + NSE APIs)
├── src/multitimeframe.py   (no internal deps — uses yfinance + ta)
├── src/risk.py             (no internal deps — pure numpy)
├── src/optimizer.py        (no internal deps — uses scipy)
├── src/holdings.py         (no internal deps — uses yfinance)
├── src/regime.py           (no internal deps — pure pandas/numpy)
├── src/ranking.py          (no internal deps — pure pandas/numpy)
├── src/volatility.py       (no internal deps — pure numpy)
├── src/backfill.py
│   ├── src/ledger.py
│   ├── src/data_fetcher.py
│   ├── src/features.py
│   ├── src/ensemble.py
│   └── src/regime.py
└── src/constants.py        (path constants, trading costs)
```

**Key insight:** Most `src/` modules are independent leaf nodes. The orchestrator is the only module with many internal dependencies — it ties everything together.

---

## Data Flow

### 1. Daily Autonomous Loop

```
python run_daily.py
    │
    ▼
DailyOrchestrator.run()                      [orchestrator.py]
    │
    ├── For each ticker:
    │   ├── fetch_stock_data(ticker)         [data_fetcher.py]
    │   ├── add_technical_indicators(df)     [features.py]
    │   │
    │   ├── Collect 14 signals:
    │   │   ├── predict_ensemble(...)        [ensemble.py]
    │   │   ├── get_stock_sentiment(...)     [sentiment.py]
    │   │   ├── get_flow_sentiment(...)      [flow.py]
    │   │   ├── fetch_options_pcr(...)       [flow.py]
    │   │   ├── get_combined_signal(...)     [multitimeframe.py]
    │   │   ├── detect_regime(...)           [regime.py]
    │   │   ├── compute_var/cvar/sharpe(...) [risk.py]
    │   │   ├── forecast_volatility(...)     [volatility.py]
    │   │   └── fetch_fundamentals(...)      [ranking.py]
    │   │
    │   ├── meta_controller.decide(state)    [meta_controller.py]
    │   │   └── 14 inputs → 1 decision (BUY/SELL/HOLD)
    │   │
    │   ├── ledger.log_decision(...)         [ledger.py]
    │   │
    │   └── (optional) paper_trader.on_bar() [paper_trader.py]
    │
    └── Returns summary dict
```

### 2. Meta-Controller Decision Flow

```
14 signal module outputs
    │
    ▼
MetaController.decide(state)                 [meta_controller.py]
    │
    ├── Build feature vector from state:
    │   ├── ensemble_direction, ensemble_confidence
    │   ├── sentiment_score, fii_net, dii_net
    │   ├── pcr, mtf_signal
    │   ├── regime_bull, regime_bear
    │   ├── var_95, cvar_95, sharpe
    │   ├── volatility_forecast
    │   └── fundamental_score
    │
    ├── LogisticRegression.predict_proba(features)
    │   └── Returns: action probabilities [sell_prob, hold_prob, buy_prob]
    │
    ├── Confidence scaling:
    │   ├── confidence = max(probabilities)
    │   ├── If confidence < 0.3 → HOLD (too uncertain)
    │   └── If confidence < 0.5 → HOLD (not confident enough)
    │
    ├── Position sizing:
    │   ├── base_size = 0.10 (10% of capital)
    │   ├── scaled_size = base_size * confidence
    │   └── capped at 0.20 (max 20% per position)
    │
    └── Returns: {"action", "confidence", "position_size", "reasoning"}
```

### 3. Training Pipeline

```
python run_daily.py --backfill
    │
    ▼
HistoricalBackfill.run()                     [backfill.py]
    │
    ├── For each day in lookback period:
    │   ├── Simulate what signals would have been
    │   ├── Log decision to ledger
    │   └── Compute actual outcome (next-day return)
    │
    └── Returns summary with accuracy metrics

    │
    ▼
MetaController.train(ledger)                 [meta_controller.py]
    │
    ├── Query ledger for all resolved decisions
    ├── Build feature matrix (14 features per decision)
    ├── Train LogisticRegression on features → outcomes
    ├── Save to models/meta_controller.pkl
    └── Returns: {"accuracy", "n_samples", "weights"}
```

### 4. Model Training Pipeline

```
train_for_ticker(ticker)                     [trainer.py]
    │
    ├── fetch_stock_data(ticker, "1y")       [data_fetcher.py]
    ├── add_technical_indicators(df)         [features.py] (48 features)
    ├── prepare_lstm_data(df, FEATURE_COLS)  [features.py] (sliding window)
    │
    ├── Walk-forward validation (5 folds):
    │   ├── For each fold:
    │   │   ├── Train 5 models on training split
    │   │   ├── Predict on test split
    │   │   └── Record accuracy
    │   └── Average accuracy across folds
    │
    ├── Train final models on full data
    ├── save_models(...)                     [model.py]
    │   └── Save 8 files to models/
    │
    └── (optional) Train meta-learner per ticker
```

---

## Model Architecture Details

### StockLSTM
```
Input (batch, 60, 48)
  → LSTM(input_dim=48, hidden=64, num_layers=2, dropout=0.3)
  → Linear(64, 32) → ReLU
  → Linear(32, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

### StockGRU
```
Input (batch, 60, 48)
  → GRU(input_dim=48, hidden=64, num_layers=2, dropout=0.3)
  → Linear(64, 32) → ReLU
  → Linear(32, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

### StockTransformer
```
Input (batch, 60, 48)
  → Linear(48, 64) → Positional Encoding
  → TransformerEncoder(d_model=64, nhead=4, num_layers=2, norm_first=True)
  → Linear(64, 32) → ReLU
  → Linear(32, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

### XGBoost
```
XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.03,
    min_child_weight=5, gamma=0.1, reg_lambda=1.0,
    subsample=0.8, colsample_bytree=0.8
)
Input: 48 features (flattened, no sequence)
Output: probability ∈ [0, 1]
```

### LightGBM
```
LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.03,
    min_child_samples=30, reg_alpha=0.5, reg_lambda=0.1,
    subsample=0.8, colsample_bytree=0.8
)
Input: 48 features (flattened, no sequence)
Output: probability ∈ [0, 1]
```

---

## Feature Engineering (48 Features)

### Technical Indicators (40)
| Feature | Source | Description |
|---------|--------|-------------|
| `rsi` | ta.RSIIndicator | Relative Strength Index (14) |
| `macd` | ta.MACD | MACD line |
| `macd_signal` | ta.MACD | MACD signal line |
| `macd_diff` | ta.MACD | MACD histogram |
| `bb_high` | ta.BollingerBands | Upper Bollinger Band |
| `bb_low` | ta.BollingerBands | Lower Bollinger Band |
| `bb_width` | computed | (High - Low) / Close |
| `sma_20` | ta.SMAIndicator | 20-day SMA |
| `sma_50` | ta.SMAIndicator | 50-day SMA |
| `sma_200` | ta.SMAIndicator | 200-day SMA |
| `ema_12` | ta.EMAIndicator | 12-day EMA |
| `ema_26` | ta.EMAIndicator | 26-day EMA |
| `adx` | ta.ADXIndicator | Average Directional Index |
| `atr` | ta.AverageTrueRange | Average True Range (14) |
| `obv` | ta.OnBalanceVolume | On-Balance Volume |
| `vwap` | ta.VolumeWeightedAveragePrice | VWAP |
| `stoch_k` | ta.StochasticOscillator | Stochastic %K |
| `stoch_d` | ta.StochasticOscillator | Stochastic %D |
| `williams_r` | ta.WilliamsRIndicator | Williams %R |
| `cci` | ta.CCIIndicator | Commodity Channel Index |
| `mfi` | ta.MFIIndicator | Money Flow Index |
| `roc` | ta.ROCIndicator | Rate of Change |
| `tsi` | ta.TSIIndicator | True Strength Index |
| `ichimoku_a` | ta.IchimokuIndicator | Ichimoku A |
| `ichimoku_b` | ta.IchimokuIndicator | Ichimoku B |
| `daily_return` | computed | Daily log return |
| `volatility_20` | computed | 20-day rolling std |
| `volatility_50` | computed | 50-day rolling std |
| `momentum_5` | computed | 5-day momentum |
| `momentum_10` | computed | 10-day momentum |
| `momentum_20` | computed | 20-day momentum |
| `volume_sma_20` | computed | 20-day volume MA |
| `price_sma_ratio` | computed | Price / SMA(20) |
| `high_low_ratio` | computed | High / Low |
| `open_close_ratio` | computed | Open / Close |
| `skew_20` | computed | 20-day rolling skew |
| `kurt_20` | computed | 20-day rolling kurtosis |
| `return_vol_corr` | computed | Return-volatility correlation |
| `sma_diff` | computed | (SMA20 - SMA50) / SMA50 |
| `ema_diff` | computed | (EMA12 - EMA26) / EMA26 |

### Alternative Data Features (8)
| Feature | Source | Description |
|---------|--------|-------------|
| `sentiment_score` | sentiment.py | FinBERT news sentiment (-1 to +1) |
| `fii_net` | flow.py | FII net buy/sell (₹ Cr) |
| `dii_net` | flow.py | DII net buy/sell (₹ Cr) |
| `flow_signal` | computed | Combined FII+DII signal |
| `pcr` | flow.py | Put-Call Ratio |
| `mtf_signal` | multitimeframe.py | Multi-timeframe combined signal |
| `mtf_confidence` | multitimeframe.py | Multi-timeframe confidence |
| `day_of_week` | calendar | Day of week (0-4) |

---

## Caching Strategy

| Cache Type | Location | TTL | Mechanism |
|-----------|----------|-----|-----------|
| Stock data | `data/*.parquet` | Until retrain | Parquet files |
| Sentiment | `data/sentiment_*.json` | 30 minutes | File-based |
| FII/DII | `data/fii_dii.parquet` | 24 hours | File-based |
| Options PCR | `data/options_pcr.json` | 1 hour | File-based |
| MTF data | `data/mtf_*.pkl` | 24 hours | File-based |
| Meta-controller | `models/meta_controller.pkl` | Until retrain | Pickle |

---

## State Management

### Persistent State (survives restarts)
| State | Location | Updated By |
|-------|----------|------------|
| Trading decisions | `data/stomar.db` | orchestrator.py |
| Paper trading positions | `data/paper_state.json` | paper_trader.py |
| Paper trading session | `data/paper_session.json` | paper_trader.py |
| MF tracker holdings | `data/mf_state.json` | mf_tracker.py |
| Meta-controller weights | `models/meta_controller.pkl` | meta_controller.py |

### Session State (Streamlit only)
| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `portfolio` | `Portfolio` | `Portfolio(100000)` | Shared portfolio across tabs |

---

## Error Handling Strategy

1. **Data fetch failures:** Caught at UI level, displayed as styled error card
2. **Model load failures:** Warning displayed, user prompted to retrain
3. **Sentiment failures:** FinBERT import try/except, keyword fallback
4. **NSE API failures:** FII/DII and PCR show "unavailable" when market closed
5. **Training failures:** Caught per-model, other models continue
6. **Meta-controller:** Falls back to ensemble if not trained
7. **Ledger:** Creates tables on first use, handles missing columns
