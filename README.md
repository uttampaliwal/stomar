# StoMar — Autonomous Quant Trading System

An autonomous quantitative trading system for the Indian NSE market. 5-model ML ensemble, meta-controller combining 14 signal modules, persistent SQLite ledger, paper trading with short selling, and a React dashboard with FastAPI backend.

[![CI](https://github.com/uttamkumar66/stomar/actions/workflows/ci.yml/badge.svg)](https://github.com/uttamkumar66/stomar/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd stomar

# 2. Create virtual environment (Python 3.12+)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd web && npm install && cd ..

# 5. Start both servers (FastAPI + React)
start_dev.bat

# 6. Train all models
python run_pipeline.py --train-all

# 7. Backfill history + train meta-controller
python run_daily.py --backfill

# 8. Start paper trading
python run_daily.py --paper-trade --capital 200000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                REACT DASHBOARD (19 pages, :5173)                  │
│  Scanner | Consensus | Ranking | Portfolio | Backtest | Risk     │
│  Sentiment | Market Pulse | Optimizer | Paper Trading | Ledger   │
│                      ↕ Vite proxy (/api)                         │
├─────────────────────────────────────────────────────────────────┤
│                  FASTAPI BACKEND (:8000)                          │
│  20 API routers | CORS | Response caching | Parallel fetch       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────────────┐
│                    META-CONTROLLER                               │
│  14 signal modules → 1 decision (contextual bandit)              │
│  Regime routing, confidence scaling, position sizing             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ signals from
┌───────────────────────────▼─────────────────────────────────────┐
│                     SIGNAL LAYER (14 modules)                    │
│  Ensemble | Sentiment | Flow | PCR | MTF | Regime | Volatility  │
│  Ranking | Risk | Fundamentals | Execution Quality | Monitoring  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ data from
┌───────────────────────────▼─────────────────────────────────────┐
│                       DATA LAYER                                 │
│  yfinance | NSE APIs | Google News | MoneyControl | ET           │
│  data/ (parquet cache) | models/ (trained weights)               │
│  data/stomar.db (SQLite ledger)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### 5-Model ML Ensemble

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| LSTM | Deep Learning (PyTorch) | hidden=64, dropout=0.3, fc=32 |
| GRU | Deep Learning (PyTorch) | hidden=64, dropout=0.3 |
| Transformer | Deep Learning (PyTorch) | d_model=64, nhead=4, norm_first |
| XGBoost | Gradient Boosting | depth=4, lr=0.03, min_child_weight=5 |
| LightGBM | Gradient Boosting | depth=4, lr=0.03, min_child_samples=30 |

- **Stacked meta-learner** learns optimal model combinations from base predictions
- **Regime-conditional routing** — Bull favors tree models, Bear favors DL
- **Walk-forward validation** — 5-fold chronological splits, no data leakage
- **48 technical features** (RSI, MACD, Bollinger, ATR, ADX, Stochastic, Williams %R, CCI, MFI, VWAP, skew/kurtosis)

### Meta-Controller (14 → 1 decision)

| Input Signal | Source | Weight |
|-------------|--------|--------|
| Ensemble direction + confidence | ML models | +3.03 (strongest) |
| Options PCR | NSE options chain | -0.62 |
| Ensemble confidence | ML models | +0.41 |
| Sharpe ratio | Risk module | +0.28 |
| Regime bull/bear | Regime detection | +0.28 |
| Sentiment score | 5-source NLP | — |
| FII/DII flow | NSE institutional | — |
| MTF signal | Multi-timeframe | — |
| VaR/CVaR | Risk module | — |
| Volatility forecast | GARCH | — |
| Fundamental score | Screener.in | — |

### Autonomous Daily Loop

```bash
python run_daily.py                          # Run all 20 NSE stocks
python run_daily.py --ticker RELIANCE.NS     # Specific tickers
python run_daily.py --backfill               # Backfill 1 year + train meta
python run_daily.py --train-meta             # Retrain meta-controller
python run_daily.py --paper-trade            # Auto-execute paper trades
python run_daily.py --dry-run                # Signals only, no ledger writes
```

### Paper Trading (with short selling)

- **Short positions** — SELL without existing position opens short
- **Cover** — BUY covers short position
- **Position sizing** — Meta-controller confidence scales trade size
- **State persistence** — `data/paper_state.json` survives restarts
- **NSE costs** — Brokerage, STT, stamp duty, exchange charges, GST

### Dashboard (19 pages)

| # | Page | Purpose |
|---|------|---------|
| 1 | Predictions | ML model predictions + candlestick + RSI + volume |
| 2 | Portfolio | Current holdings + allocation |
| 3 | Backtest | Walk-forward backtest with per-window breakdown |
| 4 | Scanner | Multi-stock screening (absolute direction) |
| 5 | **Consensus** | **Unified signal: Meta-Controller + Ensemble + Regime voting** |
| 6 | Sentiment | News sentiment analysis |
| 7 | Market Pulse | FII/DII flow, PCR, market health |
| 8 | Optimizer | Portfolio optimization + efficient frontier |
| 9 | Risk | VaR, CVaR, drawdown, Kelly criterion |
| 10 | Volatility | Multi-estimator volatility + forecast chart |
| 11 | Ranking | Cross-sectional stock ranking (relative quality) |
| 12 | Scenarios | What-if scenario comparison |
| 13 | Regime | Bull/Bear/Sideways detection + indicators |
| 14 | Correlation | Cross-asset correlation heatmap |
| 15 | Monitoring | System health + model drift |
| 16 | Pipeline | Retraining pipeline status |
| 17 | Paper Trading | Simulated trading engine |
| 18 | MF Tracker | Mutual fund NAV, XIRR, allocation |
| 19 | Ledger | Historical decisions, signal accuracy, P&L |

---

## Project Structure

```
stomar/
├── start_dev.bat              # Start FastAPI + React dev servers
├── run_daily.py               # Autonomous daily loop
├── run_pipeline.py            # Retraining pipeline
├── auto_pipeline.py           # Startup automation
├── schedule_pipeline.py       # Windows Task Scheduler
├── verify_system.py           # System verification
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── api/                       # FastAPI backend
│   ├── main.py                # App + CORS + response caching
│   ├── utils.py               # Parallel fetch utility
│   └── routers/               # 20 API routers
│       ├── predictions.py     # ML predictions + training
│       ├── scanner.py         # Multi-stock screening
│       ├── consensus.py       # Meta-Controller integration
│       ├── ranking.py         # Cross-sectional ranking
│       ├── backtest.py        # Walk-forward backtest
│       ├── risk.py            # VaR, CVaR, Sharpe, Kelly
│       ├── volatility.py      # Volatility analysis
│       ├── scenarios.py       # Scenario comparison
│       ├── regime.py          # Market regime detection
│       ├── correlation.py     # Correlation matrix
│       ├── optimizer.py       # Portfolio optimization
│       ├── sentiment.py       # News sentiment
│       ├── market.py          # Market health data
│       ├── portfolio.py       # Paper trading portfolio
│       ├── holdings.py        # Zerodha CSV import
│       ├── monitoring.py      # System health
│       ├── pipeline.py        # Retraining pipeline
│       ├── paper_trading.py   # Paper trading engine
│       ├── mf_tracker.py      # Mutual fund tracker
│       └── ledger.py          # Trading journal
├── web/                       # React frontend
│   ├── src/
│   │   ├── App.tsx            # Router with 19 routes
│   │   ├── components/        # Sidebar, ThemeProvider, UI components
│   │   ├── hooks/             # useApi, useDebouncedValue
│   │   ├── pages/             # 19 page components
│   │   └── lib/               # Utilities
│   ├── vite.config.ts         # Vite + proxy /api → :8000
│   └── package.json
├── src/                       # Python ML/trading logic
│   ├── constants.py           # Paths + global constants
│   ├── data_fetcher.py        # yfinance + NSE stock list
│   ├── data_sources.py        # Multi-source data with fallback
│   ├── features.py            # 48-feature engineering pipeline
│   ├── model.py               # 5 model architectures
│   ├── trainer.py             # Training + walk-forward validation
│   ├── ensemble.py            # Meta-learner + DL probability scaling
│   ├── backtester.py          # Walk-forward backtesting
│   ├── orchestrator.py        # Daily signal pipeline
│   ├── meta_controller.py     # Contextual bandit (14 → 1 decision)
│   ├── ledger.py              # SQLite trading journal
│   ├── paper_trader.py        # Paper trading + short selling
│   ├── sentiment.py           # 5-source sentiment + FinBERT
│   ├── flow.py                # FII/DII flow + options PCR
│   ├── multitimeframe.py      # 4-timeframe analysis
│   ├── risk.py                # VaR, CVaR, Sharpe, Kelly
│   ├── optimizer.py           # MVO, Black-Litterman
│   ├── holdings.py            # Zerodha CSV parser
│   ├── regime.py              # Bull/Bear/Sideways + ADX
│   ├── regime_strategy.py     # Regime-conditional signals
│   ├── mf_tracker.py          # Mutual fund NAV, XIRR
│   ├── monitoring.py          # System health monitoring
│   ├── volatility.py          # Volatility forecasting
│   ├── ranking.py             # Cross-sectional ranking
│   ├── scenarios.py           # Scenario analysis
│   └── pipeline.py            # Retraining pipeline
├── tests/                     # 601 tests
├── models/                    # Trained weights (gitignored)
│   ├── *.pt, *.pkl            # Per-ticker models
│   └── meta_controller.pkl    # Meta-controller
├── data/                      # Runtime data (gitignored)
│   ├── *.parquet              # OHLCV cache
│   ├── *.json                 # Sentiment/MTF cache
│   ├── stomar.db              # SQLite ledger
│   ├── paper_state.json       # Paper trading state
│   └── mf_state.json          # MF tracker state
└── docs/
    ├── ARCHITECTURE.md        # System design
    ├── API.md                 # Module interfaces
    ├── DESIGN.md              # UI/UX design system
    ├── DEPLOYMENT.md          # Setup guide
    ├── PLAN.md                # Development plan
    └── IMPROVEMENTS.md        # Roadmap
```

---

## CLI Usage

```bash
# Daily autonomous loop
python run_daily.py                              # All 20 NSE stocks
python run_daily.py --ticker RELIANCE.NS         # Specific tickers
python run_daily.py --backfill                   # Backfill 1 year
python run_daily.py --train-meta                 # Retrain meta-controller
python run_daily.py --paper-trade                # Auto paper trades
python run_daily.py --paper-trade --capital 500000

# Retraining pipeline
python run_pipeline.py --train-all               # Train all 20 stocks
python run_pipeline.py --train RELIANCE.NS       # Specific tickers
python run_pipeline.py --paper                   # Train + paper trade

# Windows scheduler
python schedule_pipeline.py                      # Install daily 4 PM IST
python schedule_pipeline.py --remove             # Remove task
python schedule_pipeline.py --run-now            # Run immediately
```

---

## Tests

```bash
# Run all 601 tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Lint check
ruff check src/ tests/ --output-format=concise
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + TypeScript + Tailwind CSS + Recharts (dark/light theme) |
| **Backend** | FastAPI + Uvicorn + Pydantic (CORS, response caching, parallel fetch) |
| **ML/DL** | PyTorch 2.0+, XGBoost 2.0+, LightGBM 4.0+ |
| **NLP** | HuggingFace Transformers (ProsusAI/finbert) |
| **Data** | yfinance, NSE India, Google News RSS, MoneyControl, ET, Screener.in |
| **Optimization** | SciPy, scikit-learn (Ledoit-Wolf) |
| **Technical Analysis** | `ta` library (30+ indicators) |
| **Storage** | SQLite (ledger), Parquet (data cache) |

---

## Important Disclaimers

- **Educational purposes only** — not financial advice
- Past performance does not guarantee future results
- Always do your own research before investing
- Start with paper trading before using real money
- Walk-forward accuracy varies: some stocks 55-72%, others 44-52%
- Meta-controller trained on backfill data — real accuracy may differ

---

## License

Licensed under the [Apache License 2.0](LICENSE).
