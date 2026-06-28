# API Reference

Complete function signatures, parameters, return types, and usage for all modules.

---

## src/data_fetcher.py

### Constants
```python
NSE_STOCKS: list[str]  # 20 NSE tickers
DATA_DIR: str          # "data/" directory path
```

### `fetch_stock_data(ticker, period="2y", interval="1d", force_refresh=False) -> pd.DataFrame`
Fetches OHLCV stock data from yfinance with parquet caching.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | `str` | required | Yahoo Finance ticker (e.g., `"RELIANCE.NS"`) |
| `period` | `str` | `"2y"` | yfinance period string (`"6mo"`, `"1y"`, `"2y"`, `"5y"`) |
| `interval` | `str` | `"1d"` | Candle interval (`"1d"`, `"1h"`, `"15m"`) |
| `force_refresh` | `bool` | `False` | Bypass cache and re-download |

**Returns:** DataFrame with columns: `open, high, low, close, volume` (index = DatetimeIndex)

---

### `get_live_price(ticker) -> float`
Returns the most recent closing price.

---

### `get_market_status() -> str`
Returns `"Open"` or `"Closed"` based on current IST time.

---

## src/features.py

### `add_technical_indicators(df, ticker=None) -> pd.DataFrame`
Adds 41 features to the input DataFrame. Calls sub-functions for sentiment, flow, PCR, and MTF features.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | OHLCV data |
| `ticker` | `str` | `None` | Used for sentiment/MTF feature fetching |

**Returns:** DataFrame with original columns + 41 new feature columns

---

### `add_sentiment_features(df, ticker) -> pd.DataFrame`
Adds `sentiment_score` column from FinBERT analysis. Reads from cache file.

---

### `add_flow_features(df) -> pd.DataFrame`
Adds `fii_net`, `dii_net`, `flow_signal` columns from NSE FII/DII data.

---

### `add_pcr_features(df) -> pd.DataFrame`
Adds `pcr` column from NSE options chain data.

---

### `add_multitimeframe_features(df, ticker) -> pd.DataFrame`
Adds `mtf_signal`, `mtf_confidence` columns from multi-timeframe analysis.

---

### `prepare_lstm_data(df, feature_cols, seq_length=60) -> (np.array, np.array, MinMaxScaler)`
Creates sequences for LSTM/GRU/Transformer training.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | Feature-engineered DataFrame |
| `feature_cols` | `list[str]` | required | Column names to use |
| `seq_length` | `int` | `60` | Sliding window size (days) |

**Returns:** `(X, y, scaler)` where X.shape = `(samples, seq_length, n_features)`, y.shape = `(samples,)`

---

## src/model.py

### Constants
```python
MODELS_DIR: str   # "models/" directory path
DEVICE: torch.device  # "cuda" or "cpu"
```

### Classes

#### `StockLSTM(nn.Module)`
```python
__init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2)
forward(self, x) -> Tensor  # x: (batch, seq, features) → (batch, 1)
```

#### `StockGRU(nn.Module)`
```python
__init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2)
forward(self, x) -> Tensor
```

#### `StockTransformer(nn.Module)`
```python
__init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2)
forward(self, x) -> Tensor
```

### Builder Functions

#### `build_lstm(input_dim) -> StockLSTM`
#### `build_gru(input_dim) -> StockGRU`
#### `build_transformer(input_dim) -> StockTransformer`
#### `build_xgb_model() -> XGBClassifier`
```python
# Returns XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
```
#### `build_lgb_model() -> LGBMClassifier`
```python
# Returns LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
```

### `save_models(lstm, gru, transformer, xgb, scaler, feature_cols, ticker, lgb_model=None) -> None`
Saves all 7-8 model files to `models/` directory.

| Param | Type | Description |
|-------|------|-------------|
| `lstm` | `StockLSTM` | Trained LSTM model |
| `gru` | `StockGRU` | Trained GRU model |
| `transformer` | `StockTransformer` | Trained Transformer model |
| `xgb` | `XGBClassifier` | Trained XGBoost model |
| `scaler` | `MinMaxScaler` | Fitted scaler |
| `feature_cols` | `list[str]` | Feature column names |
| `ticker` | `str` | Stock ticker (used for filename) |
| `lgb_model` | `LGBMClassifier` | Optional LightGBM model |

---

### `load_models(ticker) -> tuple`
Loads all saved models for a ticker.

**Returns:** `(lstm, gru, transformer, xgb, scaler, feature_cols, lgb_model)`

| Index | Type | Description |
|-------|------|-------------|
| 0 | `StockLSTM` | LSTM model (on DEVICE) |
| 1 | `StockGRU` | GRU model (on DEVICE) |
| 2 | `StockTransformer` | Transformer model (on DEVICE) |
| 3 | `XGBClassifier` | XGBoost model |
| 4 | `MinMaxScaler` | Fitted scaler |
| 5 | `list[str]` | Feature column names |
| 6 | `LGBMClassifier` or `None` | LightGBM model |

---

### `models_exist(ticker) -> bool`
Checks if all required model files exist for a ticker.

**Required files:** `lstm.pt, gru.pt, transformer.pt, xgb.pkl, scaler.pkl, features.pkl, lstm_dim.pkl`
**Optional:** `lgb.pkl`

---

## src/trainer.py

### Constants
```python
FEATURE_COLS: list[str]  # 37 feature column names
SEQ_LENGTH: int = 60
EPOCHS: int = 40
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
```

### `train_for_ticker(ticker, force_retrain=False) -> dict`
Trains all 5 models for a single ticker.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | `str` | required | NSE ticker symbol |
| `force_retrain` | `bool` | `False` | Retrain even if models exist |

**Returns:**
```python
{
    "lstm_acc": float,
    "gru_acc": float,
    "transformer_acc": float,
    "xgb_acc": float,
    "lgb_acc": float,
    "ensemble_accuracy": float,
    "n_samples": int,
    "n_features": int,
}
```

---

### `train_multiple_stocks(tickers, force_retrain=False) -> dict`
Trains multiple tickers. Returns dict mapping ticker → training results.

---

## src/ensemble.py

### `predict_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat, recent_weights=None, lgb_model=None) -> tuple`
Runs all 5 models and combines predictions.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `lstm` | `StockLSTM` | required | LSTM model |
| `gru` | `StockGRU` | required | GRU model |
| `transformer` | `StockTransformer` | required | Transformer model |
| `xgb` | `XGBClassifier` | required | XGBoost model |
| `scaler` | `MinMaxScaler` | required | Fitted scaler |
| `feature_cols` | `list[str]` | required | Feature columns |
| `df_feat` | `pd.DataFrame` | required | Feature-engineered data |
| `recent_weights` | `np.array` | `None` | Optional custom weights |
| `lgb_model` | `LGBMClassifier` | `None` | Optional LightGBM |

**Returns:** `(ensemble_direction, confidence, details)`
```python
# ensemble_direction: int (1 = BUY, 0 = SELL)
# confidence: float (0-100, percentage)
# details: dict with keys:
#   "lstm_dir", "lstm_conf",
#   "gru_dir", "gru_conf",
#   "transformer_dir", "transformer_conf",
#   "xgb_dir", "xgb_conf",
#   "lgb_dir", "lgb_conf" (if lgb_model provided)
```

---

### `backtest_ensemble(lstm, gru, transformer, xgb, scaler, feature_cols, df_feat, seq_length=60, lgb_model=None) -> list[dict]`
Generates per-day predictions for backtesting.

**Returns:** List of dicts, one per day:
```python
{
    "date": str,          # "YYYY-MM-DD"
    "actual": int,        # 0 or 1
    "predicted": int,     # 0 or 1
    "ensemble_prob": float,
    "lstm": int,          # LSTM prediction (0 or 1)
    "gru": int,           # GRU prediction (0 or 1)
    "transformer": int,   # Transformer prediction (0 or 1)
    "xgb_prob": float,    # XGBoost probability
    "lgb_prob": float,    # LightGBM probability (if provided)
}
```

---

## src/backtester.py

### `walk_forward_split(df, train_years=3, test_years=1, step_months=6) -> list[dict]`
Splits data into rolling train/test windows.

**Returns:** List of window dicts:
```python
[
    {
        "train_start": datetime,
        "train_end": datetime,
        "test_start": datetime,
        "test_end": datetime,
    },
    ...
]
```

---

### `compute_metrics(equity_curve, trades, risk_free_rate=0.065) -> dict`
Computes portfolio performance metrics.

**Returns:**
```python
{
    "total_return": float,      # e.g., 0.25 = 25%
    "annual_return": float,
    "sharpe_ratio": float,
    "max_drawdown": float,      # percentage
    "win_rate": float,          # percentage
    "total_trades": int,
    "avg_trade_return": float,
}
```

---

### `run_walk_forward_backtest(ticker, df_feat, feature_cols, train_years=3, test_years=1, step_months=6, initial_capital=100000, position_pct=0.25, stop_loss_pct=0.05, take_profit_pct=0.15, max_positions=5, brokerage=0.0003, slippage=0.001) -> tuple`
Full walk-forward backtest with model training per window.

**Returns:** `(metrics, portfolio, all_test_results)`
```python
# metrics: dict with ensemble_accuracy, simulated_annual_return, etc.
# portfolio: Portfolio object with equity curve and trades
# all_test_results: list[dict] from backtest_ensemble per window
```

---

### `run_simple_backtest(df_feat, signals, initial_capital=100000, brokerage=0.0003, slippage=0.001) -> tuple`
Simple backtest with pre-computed signals.

**Returns:** `(stats, portfolio)`

---

### `generate_model_signals(ticker, df_feat, lstm, gru, transformer, xgb, scaler, feature_cols) -> dict`
Generates trading signals from trained models.

**Returns:** `{"signals": list[dict], "portfolio": Portfolio}`

---

## src/portfolio.py

### `Portfolio` Class

#### `__init__(self, initial_capital=100000)`
```python
self.initial_capital: float
self.cash: float
self.holdings: dict[ticker, {"qty": int, "avg_price": float}]
self.trades: list[dict]
self.equity_curve: list[dict]
```

#### `buy(ticker, price, quantity, date, brokerage=0.0003) -> bool`
Executes a buy order. Returns `True` if successful.

#### `sell(ticker, price, quantity, date, brokerage=0.0003) -> bool`
Executes a sell order. Returns `True` if successful.

#### `portfolio_value(prices=None) -> float`
Returns total portfolio value (cash + holdings).

#### `get_stats() -> dict`
```python
{
    "total_return": float,
    "sharpe_ratio": float,
    "max_drawdown": float,
    "win_rate": float,
    "cash_remaining": float,
    "holdings_value": float,
}
```

---

## src/sentiment.py

### `get_stock_sentiment(ticker) -> dict`
Full sentiment analysis pipeline with 30-min caching.

**Returns:**
```python
{
    "score": float,       # -1.0 to +1.0
    "label": str,         # "Positive", "Negative", "Neutral"
    "articles": list[dict],  # [{title, snippet, url, score}, ...]
    "n_articles": int,
}
```

---

### `get_finbert() -> pipeline`
Returns cached FinBERT sentiment pipeline (lazy-loaded).

---

### `fetch_news_headlines(ticker, max_articles=25) -> list[dict]`
Fetches headlines from Google News RSS.

**Returns:** `[{"title": str, "snippet": str, "url": str}, ...]`

---

### `analyze_sentiment(articles) -> dict`
Runs FinBERT on a list of articles.

**Returns:** `{"score": float, "positive": int, "negative": int, "neutral": int}`

---

## src/flow.py

### `fetch_fii_dii() -> pd.DataFrame`
Fetches today's FII/DII data from NSE API.

**Returns:** DataFrame with columns: `date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net`

---

### `fetch_options_pcr() -> dict`
Fetches NIFTY options chain and computes PCR + Max Pain.

**Returns:**
```python
{
    "pcr_oi": float,       # Put OI / Call OI
    "call_oi": int,        # Total call open interest
    "put_oi": int,         # Total put open interest
    "max_pain": float,     # Max pain strike price
    "symbol": str,         # "NIFTY"
}
```
**Note:** Returns `{"pcr_oi": 0}` when market is closed (NSE API returns 404).

---

### `get_flow_sentiment(fii_net, dii_net) -> str`
Classifies institutional flow sentiment.

**Returns:** One of:
- `"Strong Bullish"` — both FII and DII positive
- `"Mild Bullish"` — one positive, one negative
- `"Neutral"` — both near zero
- `"Mild Bearish"` — one strongly negative
- `"Strong Bearish"` — both negative

---

## src/multitimeframe.py

### `fetch_mtf_data(ticker) -> dict`
Fetches data across 4 timeframes (cached via `@st.cache_data`).

**Returns:**
```python
{
    "15m": pd.DataFrame,    # 15-minute OHLCV
    "1h": pd.DataFrame,     # 1-hour OHLCV
    "daily": pd.DataFrame,  # Daily OHLCV
    "weekly": pd.DataFrame, # Weekly OHLCV
}
```

---

### `get_combined_signal(mtf_data) -> dict`
Computes weighted signal across all timeframes.

**Returns:**
```python
{
    "signal": str,          # "Bullish", "Bearish", "Neutral"
    "confidence": float,    # 0-100
    "timeframes": {
        "15m": {"signal": str, "strength": float, "indicators": dict},
        "1h": {...},
        "daily": {...},
        "weekly": {...},
    }
}
```

**Timeframe weights:** Daily (40%), Weekly (30%), 1h (20%), 15m (10%)

---

## src/risk.py

### `kelly_criterion(win_rate, avg_win, avg_loss) -> float`
Computes optimal position size using Kelly Criterion.

```python
# f* = (p * b - q) / b
# where p = win_rate, q = 1-p, b = avg_win/avg_loss
```

---

### `calculate_var(returns, confidence=0.95) -> float`
Value at Risk at given confidence level.

---

### `calculate_cvar(returns, confidence=0.95) -> float`
Conditional VaR (expected shortfall).

---

### `calculate_sharpe(returns, risk_free_rate=0.065) -> float`
Annualized Sharpe ratio.

---

### `calculate_sortino(returns, risk_free_rate=0.065) -> float`
Annualized Sortino ratio (downside deviation only).

---

### `calculate_max_drawdown(equity_curve) -> float`
Maximum drawdown percentage.

---

### `calculate_calmar(returns, equity_curve) -> float`
Calmar ratio (annual return / max drawdown).

---

### `generate_risk_report(returns, equity_curve) -> dict`
Complete risk analysis.

**Returns:**
```python
{
    "annual_return": float,
    "annual_volatility": float,
    "sharpe": float,
    "sortino": float,
    "max_drawdown": float,
    "var_95": float,
    "cvar_95": float,
    "calmar": float,
    "positive_days_pct": float,
    "best_day": float,
    "worst_day": float,
    "skewness": float,
}
```

---

## src/optimizer.py

### `optimize_portfolio(prices, views=None, confidences=None, risk_free_rate=0.065) -> dict`
Full portfolio optimization with 3 strategies.

**Returns:**
```python
{
    "max_sharpe": {
        "weights": list[float],
        "return": float,
        "volatility": float,
        "sharpe": float,
    },
    "min_variance": {...},
    "black_litterman": {...},
    "efficient_frontier": [
        {"volatility": float, "return": float},
        ...
    ],
}
```

---

### `black_litterman(market_weights, cov_matrix, views, confidences, risk_aversion=2.5, tau=0.05) -> dict`
Black-Litterman model with investor views.

| Param | Type | Description |
|-------|------|-------------|
| `market_weights` | `np.array` | Market-cap implied weights |
| `cov_matrix` | `np.array` | Covariance matrix |
| `views` | `list` | View matrices (P, Q) |
| `confidences` | `list` | Confidence levels per view |

---

## src/holdings.py

### `parse_holdings_csv(filepath) -> pd.DataFrame`
Parses a Zerodha holdings CSV export.

**Returns:** DataFrame with columns: `name, quantity, avg_price, invested, current_value, pnl, return_pct`

---

### `compute_portfolio_stats(holdings_df) -> dict`
Computes comprehensive portfolio statistics.

**Returns:**
```python
{
    "total_invested": float,
    "total_current": float,
    "total_pnl": float,
    "total_return_pct": float,
    "categories": {
        "Gold": {"value": float, "weight": float, "return_pct": float},
        "Equity": {...},
        ...
    },
    "holdings": [
        {"name": str, "invested": float, "current_value": float,
         "pnl": float, "return_pct": float, "weight": float},
        ...
    ],
}
```

---

### Constants
```python
INDIAN_MF_MAP: dict[str, str]  # Maps 23 Indian MF names to Yahoo tickers
```

---

## src/regime.py

### `detect_regime(prices, lookback=200) -> dict`
Detects current market regime.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `prices` | `pd.Series` | required | Closing prices |
| `lookback` | `int` | `200` | SMA lookback period |

**Returns:**
```python
{
    "regime": str,          # "Bull", "Bear", "Sideways"
    "confidence": float,    # 0-100
    "rsi": float,
    "adx": float,
    "volatility": float,
    "recommendation": {
        "action": str,      # "Aggressive buying", "Defensive", etc.
        "allocation": str,  # "80-100% equity", "20-40% equity", etc.
        "risk_level": str,  # "Moderate-High", "Low-Moderate", etc.
    },
}
```

**Regime Rules:**
- **Bull:** Price > SMA(200), RSI > 50, ADX > 20
- **Bear:** Price < SMA(200), RSI < 50
- **Sideways:** Neither Bull nor Bear conditions met
