# Architecture

System design, data flow, and module dependencies for StoMar.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (app.py)                       │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │ Predict  │Portfolio │ Backtest │ Scanner  │Sentiment │      │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤      │
│  │ Mkt Pulse│Optimizer │ Holdings │  Risk    │          │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
│         │           │           │           │                   │
│    ┌────▼────┐ ┌────▼────┐ ┌───▼────┐ ┌───▼────┐              │
│    │ensemble │ │optimizer│ │holdings│ │ risk   │              │
│    └────┬────┘ └────┬────┘ └───┬────┘ └───┬────┘              │
│         │           │          │           │                    │
│  ┌──────▼───────────▼──────────▼───────────▼──────┐            │
│  │              Core Engine Layer                   │            │
│  │  trainer.py  backtester.py  sentiment.py  flow.py│           │
│  │  features.py  model.py  multitimeframe.py regime.py│         │
│  └──────────────────┬──────────────────────────────┘            │
│                     │                                           │
│  ┌──────────────────▼──────────────────────────────┐            │
│  │              Data Layer                          │            │
│  │  data_fetcher.py  (yfinance + NSE APIs)         │            │
│  │  models/           (saved .pt, .pkl, .json)     │            │
│  │  data/             (cached .parquet, .json)     │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Map

```
app.py
├── src/data_fetcher.py     (no internal deps)
├── src/features.py
│   └── uses: sentiment.py, flow.py, multitimeframe.py (for alt features)
├── src/model.py            (no internal deps — pure PyTorch/sklearn)
├── src/trainer.py
│   ├── src/data_fetcher.py
│   ├── src/features.py
│   └── src/model.py
├── src/ensemble.py
│   └── src/model.py (for DEVICE)
├── src/backtester.py
│   └── src/portfolio.py
├── src/portfolio.py        (no internal deps)
├── src/sentiment.py        (no internal deps — uses yfinance + HuggingFace)
├── src/flow.py             (no internal deps — uses requests + NSE APIs)
├── src/multitimeframe.py   (no internal deps — uses yfinance + ta)
├── src/risk.py             (no internal deps — pure numpy)
├── src/optimizer.py        (no internal deps — uses scipy)
├── src/holdings.py         (no internal deps — uses yfinance)
└── src/regime.py           (no internal deps — pure pandas/numpy)
```

**Key insight:** Most `src/` modules are independent leaf nodes. Only `trainer.py` and `backtester.py` have internal dependencies. This makes the system highly modular — any module can be tested or replaced in isolation.

---

## Data Flow

### 1. Training Pipeline

```
User clicks "Train" in UI
    │
    ▼
train_for_ticker(ticker)                    [trainer.py:76]
    │
    ├── fetch_stock_data(ticker, "2y")      [data_fetcher.py:17]
    │       │
    │       ├── Check cache (data/*.parquet)
    │       ├── If miss: yf.download(ticker)
    │       └── Save to parquet cache
    │
    ├── add_technical_indicators(df)         [features.py:88]
    │       │
    │       ├── 33 technical indicators (ta library)
    │       ├── Sentiment features (sentiment.py)
    │       ├── FII/DII flow features (flow.py)
    │       ├── Options PCR features (flow.py)
    │       └── Multi-timeframe features (multitimeframe.py)
    │
    ├── prepare_lstm_data(df, FEATURE_COLS)  [features.py:148]
    │       │
    │       └── MinMaxScaler + sliding window (60 days)
    │
    ├── Train 5 models in sequence:
    │   ├── StockLSTM   (PyTorch, 40 epochs, batch=32)
    │   ├── StockGRU    (PyTorch, 40 epochs, batch=32)
    │   ├── StockTransformer (PyTorch, 40 epochs, batch=32)
    │   ├── XGBClassifier (sklearn API, 100 estimators)
    │   └── LGBMClassifier (sklearn API, 100 estimators)
    │
    └── save_models(...)                    [model.py:105]
            │
            └── Save 8 files to models/:
                ├── {ticker}_lstm.pt
                ├── {ticker}_gru.pt
                ├── {ticker}_transformer.pt
                ├── {ticker}_xgb.pkl
                ├── {ticker}_lgb.pkl
                ├── {ticker}_scaler.pkl
                ├── {ticker}_features.pkl
                └── {ticker}_lstm_dim.pkl
```

### 2. Prediction Pipeline

```
User selects stock in Predictions tab
    │
    ▼
load_models(ticker)                         [model.py:119]
    │
    └── Returns: (lstm, gru, transformer, xgb, scaler, features, lgb)
    │
    ▼
predict_ensemble(...)                       [ensemble.py:6]
    │
    ├── LSTM forward pass   → lstm_dir, lstm_conf
    ├── GRU forward pass    → gru_dir, gru_conf
    ├── Transformer forward → trans_dir, trans_conf
    ├── XGBoost predict     → xgb_dir, xgb_conf
    ├── LightGBM predict    → lgb_dir, lgb_conf
    │
    └── Weighted vote (20% each):
        ├── direction = round(sum(weight * dir))
        └── confidence = sum(weight * conf) * 100
    │
    ▼
Returns: (ensemble_direction, confidence, details_dict)
```

### 3. Walk-Forward Backtest Pipeline

```
User clicks "Run Walk-Forward Backtest"
    │
    ▼
run_walk_forward_backtest(...)               [backtester.py:88]
    │
    ├── walk_forward_split(df)               [backtester.py:6]
    │       │
    │       └── Rolling windows:
    │           ├── Window 1: Train [2020-2023], Test [2023-2024]
    │           ├── Window 2: Train [2021-2024], Test [2024-2025]
    │           └── (step = 6 months)
    │
    ├── For each window:
    │   ├── Train all 5 models on training data
    │   ├── Predict on test data
    │   ├── Generate signals (BUY/SELL)
    │   ├── Simulate trades with Portfolio class
    │   └── Record per-day results
    │
    ├── compute_metrics(equity_curve, trades) [backtester.py:25]
    │       │
    │       └── Returns: accuracy, annual return, Sharpe, etc.
    │
    └── Returns: (metrics, portfolio, all_test_results)
```

### 4. Sentiment Pipeline

```
get_stock_sentiment(ticker)                 [sentiment.py:133]
    │
    ├── Check cache (data/sentiment_cache/*.json, TTL=30min)
    │
    ├── fetch_news_headlines(ticker)        [sentiment.py:28]
    │       │
    │       ├── Google News RSS search
    │       ├── Extract headlines + snippets
    │       └── Return list of article dicts
    │
    ├── analyze_sentiment(articles)         [sentiment.py:84]
    │       │
    │       ├── Load FinBERT pipeline (ProsusAI/finbert)
    │       ├── Classify each headline: positive/negative/neutral
    │       └── Compute weighted average score (-1 to +1)
    │
    └── Returns: {"score": float, "label": str, "articles": list}
```

### 5. Market Pulse Pipeline

```
tab_market_pulse()                          [app.py:895]
    │
    ├── FII/DII Flow
    │   └── fetch_fii_dii()                 [flow.py:29]
    │       ├── NSE API request
    │       └── Parse JSON → DataFrame
    │
    ├── Options PCR
    │   └── fetch_options_pcr()             [flow.py:87]
    │       ├── NSE options chain API
    │       ├── Compute PCR = Put OI / Call OI
    │       └── Find Max Pain strike price
    │
    └── Multi-Timeframe
        └── fetch_mtf_data(ticker)          [multitimeframe.py:13]
            ├── yfinance: 15m, 1h, 1d, 1wk data
            ├── Add 8 indicators per timeframe
            ├── Compute per-TF signal
            └── Weighted combination → combined signal
```

---

## Model Architecture Details

### StockLSTM (`model.py:12`)
```
Input (batch, 60, 41)
  → LSTM(input_dim=41, hidden=128, num_layers=2, dropout=0.2)
  → Linear(128, 64) → ReLU → Dropout(0.3)
  → Linear(64, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

### StockGRU (`model.py:30`)
```
Input (batch, 60, 41)
  → GRU(input_dim=41, hidden=128, num_layers=2, dropout=0.2)
  → Linear(128, 64) → ReLU → Dropout(0.3)
  → Linear(64, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

### StockTransformer (`model.py:48`)
```
Input (batch, 60, 41)
  → Linear(41, 64) → Positional Encoding
  → TransformerEncoder(d_model=64, nhead=4, num_layers=2)
  → Linear(64, 32) → ReLU → Dropout(0.3)
  → Linear(32, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

### XGBoost (`model.py:86`)
```
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
Input: 41 features (flattened, no sequence)
Output: probability ∈ [0, 1]
```

### LightGBM (`model.py:95`)
```
LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1
)
Input: 41 features (flattened, no sequence)
Output: probability ∈ [0, 1]
```

---

## Feature Engineering (41 Features)

### Technical Indicators (33)
| Feature | Source | Description |
|---------|--------|-------------|
| `rsi` | ta.RSIIndicator | Relative Strength Index (14) |
| `macd` | ta.MACD | MACD line |
| `macd_signal` | ta.MACD | MACD signal line |
| `macd_diff` | ta.MACD | MACD histogram |
| `bb_high` | ta.BollingerBands | Upper Bollinger Band |
| `bb_low` | ta.BollingerBands | Lower Bollinger Band |
| `bb_width` | computed | (High - Low) / Close |
| `sma_20` | ta.SMAIndicator | 20-day Simple Moving Average |
| `sma_50` | ta.SMAIndicator | 50-day Simple Moving Average |
| `sma_200` | ta.SMAIndicator | 200-day Simple Moving Average |
| `ema_12` | ta.EMAIndicator | 12-day Exponential MA |
| `ema_26` | ta.EMAIndicator | 26-day Exponential MA |
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

### Alternative Data Features (8)
| Feature | Source | Description |
|---------|--------|-------------|
| `sentiment_score` | sentiment.py | FinBERT news sentiment (-1 to +1) |
| `fii_net` | flow.py | FII net buy/sell (₹ Cr) |
| `dii_net` | flow.py | DII net buy/sell (₹ Cr) |
| `flow_signal` | computed | Combined FII+DII signal (1=bullish, 0=bearish) |
| `pcr` | flow.py | Put-Call Ratio |
| `mtf_signal` | multitimeframe.py | Multi-timeframe combined signal |
| `mtf_confidence` | multitimeframe.py | Multi-timeframe confidence |
| `day_of_week` | calendar | Day of week (0-4) |

---

## Caching Strategy

| Cache Type | Location | TTL | Mechanism |
|-----------|----------|-----|-----------|
| Stock data | `data/*.parquet` | Until retrain | `@st.cache_data` + parquet files |
| Sentiment | `data/sentiment_cache/*.json` | 30 minutes | File-based with timestamp check |
| FII/DII | `data/fii_dii/*.json` | 24 hours | File-based |
| Options PCR | `data/options/*.json` | 1 hour | File-based |
| MTF data | `data/mtf/*.json` | 24 hours | File-based |
| Streamlit cache | In-memory | 1 hour | `@st.cache_data(ttl=3600)` |

---

## State Management

### Session State Variables
| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `portfolio` | `Portfolio` | `Portfolio(100000)` | Shared portfolio across Backtest + Portfolio tabs |
| `last_ticker` | `str` | `NSE_STOCKS[0]` | Remembers last selected ticker |

### Widget Keys (Implicit State)
| Key | Tab | Widget |
|-----|-----|--------|
| `bt_ticker` | Backtest | Stock selector |
| `sent_ticker` | Sentiment | Stock selector |
| `tf_ticker` | Market Pulse | Stock selector |
| `opt_stocks` | Optimizer | Multi-select |
| `opt_period` | Optimizer | Period selector |
| `holdings_upload` | Holdings | File uploader |
| `risk_ticker` | Risk | Stock selector |
| `risk_period` | Risk | Period selector |

---

## Error Handling Strategy

1. **Data fetch failures:** Caught at UI level, displayed as styled error card
2. **Model load failures:** Warning displayed, user prompted to retrain
3. **Sentiment failures:** FinBERT import wrapped in try/except, falls back gracefully
4. **NSE API failures:** FII/DII and PCR show "unavailable" state when market closed
5. **Training failures:** Caught per-model, other models continue training

---

## File Size & Performance

| Metric | Value |
|--------|-------|
| Total source lines (src/) | ~2,322 |
| Total app.py lines | 1,258 |
| Model files per stock | 8 files (~5-20 MB total) |
| Training time per stock | ~30-90 seconds |
| Walk-forward backtest | ~2-5 minutes per stock |
| Sentiment analysis | ~5-15 seconds (cached 30 min) |
| Streamlit startup | ~3-5 seconds |
