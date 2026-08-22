# StoMar — Autonomous Quant Trading System

An autonomous quantitative trading system for the Indian NSE market. 5-model ML ensemble, meta-controller combining 14 signal modules, persistent SQLite ledger, paper trading with short selling, model interpretability, and a React dashboard with FastAPI backend.

[![CI](https://github.com/uttampaliwal/stomar/actions/workflows/ci.yml/badge.svg)](https://github.com/uttampaliwal/stomar/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Quick Start

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager) and Node.js 22.22+ (see `web/.nvmrc`).
> `start_dev.sh` / `start_dev.bat` install both toolchains' dependencies automatically — you only need `uv` and Node.

```bash
# 1. Clone the repo
git clone https://github.com/uttampaliwal/stomar.git
cd stomar

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 3. Install Python + frontend dependencies and start both servers (FastAPI + React)
./start_dev.sh        # Linux/macOS
start_dev.bat         # Windows

# 4. Train all models
uv run run_pipeline.py --train-all

# 5. Backfill history + train meta-controller
uv run run_daily.py --backfill

# 6. Start paper trading
uv run run_daily.py --paper-trade --capital 200000
```

All Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
`uv sync` creates the `.venv` (Python 3.13, pinned via `.python-version`) and installs
everything reproducibly — no manual venv/pip management.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                REACT DASHBOARD (21 pages, :5173)                 │
│  Scanner | Consensus | Ranking | Portfolio | Backtest | Risk     │
│  Sentiment | Market Pulse | Optimizer | Paper Trading | Ledger   │
│                      ↕ Vite proxy (/api)                         │
├─────────────────────────────────────────────────────────────────┤
│                  FASTAPI BACKEND (:8000)                          │
│  24 API routers | CORS | Response caching | Parallel fetch      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────────────┐
│                    META-CONTROLLER                               │
│  14 signal modules → 1 decision (learned weighting)              │
│  Regime routing, confidence scaling, position sizing             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ signals from
┌───────────────────────────▼─────────────────────────────────────┐
│                     SIGNAL LAYER (16 modules)                    │
│  Ensemble | Sentiment | Flow | PCR | MTF | Regime | Volatility  │
│  Ranking | Risk | Fundamentals | Execution Quality | Monitoring  │
│  Interpretability | Significance | Benchmarks | Scenarios        │
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
- **48+ technical features** (RSI, MACD, Bollinger, ATR, ADX, Stochastic, Williams %R, CCI, MFI, VWAP, skew/kurtosis, triple-barrier labels)

### Meta-Controller (14 → 1 decision)

| Input Signal | Source | Learned Weight |
|-------------|--------|----------------|
| Ensemble direction | ML models | Strongest |
| Ensemble confidence | ML models | High |
| Options PCR | NSE options chain | Negative (contrarian) |
| Sharpe ratio | Risk module | Positive |
| Regime bull/bear | Regime detection | Conditional |
| Sentiment score | 5-source NLP | Variable |
| FII/DII flow | NSE institutional | Variable |
| MTF signal | Multi-timeframe | Variable |
| VaR/CVaR | Risk module | Negative (risk-off) |
| Volatility forecast | GARCH | Negative |
| Fundamental score | Screener.in | Positive |

### Model Interpretability

- **Feature importance** — XGBoost/LightGBM gain-based ranking of top features driving predictions
- **Prediction explanations** — Human-readable breakdown of why BUY/SELL/HOLD was chosen
- **Direction signals** — Overbought/oversold, trend strength, volume analysis for top features
- **SHAP support** — Optional SHAP-based explanations (install `shap` for full support)

### Triple-Barrier Labels

Advanced ML training targets that account for realistic trading conditions:
- **Upper barrier** — Profit-taking level (2% default)
- **Lower barrier** — Stop-loss level (2% default)
- **Vertical barrier** — Maximum holding period (5 days default)
- Labels: 1 = profit hit first, 0 = loss hit first, 2 = time expired

> **Note:** triple-barrier labels are computed in `features.py` (`tb_label`)
> and available as a training target, but the current supervised models are
> trained on the simple next-day direction target (`target_direction`).
> Switching training to `tb_label` is planned — see `docs/ROADMAP.md`.

### Autonomous Daily Loop

The recommended entry point is the **auto-pipeline** — one idempotent run
covers gap detection, backfill, training, today's decisions, paper trades,
and the Telegram summary:

```bash
uv run auto_pipeline.py                    # Everything, idempotent (safe to rerun)
uv run auto_pipeline.py --force            # Force even if already ran today
uv run auto_pipeline.py --setup            # One-command device setup: env + Telegram + GPU + scheduler
uv run auto_pipeline.py --setup-run        # Same, plus run the pipeline immediately
uv run auto_pipeline.py --capital 200000   # Paper starting capital
```

The lower-level single-day loop remains available:

```bash
uv run run_daily.py                          # Run all 20 NSE stocks
uv run run_daily.py --ticker RELIANCE.NS     # Specific tickers
uv run run_daily.py --backfill               # Backfill 1 year + train meta
uv run run_daily.py --train-meta             # Retrain meta-controller
uv run run_daily.py --paper-trade            # Auto-execute paper trades
uv run run_daily.py --dry-run                # Signals only, no ledger writes
```

### Paper Trading (with short selling)

- **Short positions** — SELL without existing position opens short
- **Cover** — BUY covers short position
- **Position sizing** — Meta-controller confidence scales trade size
- **State persistence** — `data/paper_state.json` survives restarts
- **NSE costs** — Brokerage, STT, stamp duty, exchange charges, GST
- **Risk controls** — Position limits, drawdown limits, kill switch, Kelly sizing

### Dashboard (21 pages)

| # | Page | Purpose |
|---|------|---------|
| 1 | Dashboard | Command center: pipeline health, recommendations, automation |
| 2 | Predictions | ML model predictions + candlestick + RSI + volume + **model explanations** |
| 3 | Portfolio | Current holdings + allocation |
| 4 | Backtest | Walk-forward backtest with per-window breakdown |
| 5 | Scanner | Multi-stock screening (absolute direction) |
| 6 | **Consensus** | **Unified signal: Meta-Controller + Ensemble + Regime voting** |
| 7 | Sentiment | News sentiment analysis |
| 8 | Market Pulse | FII/DII flow, PCR, market health |
| 9 | Optimizer | Portfolio optimization + efficient frontier |
| 10 | Risk | VaR, CVaR, drawdown, Kelly criterion |
| 11 | Volatility | Multi-estimator volatility + forecast chart |
| 12 | Ranking | Cross-sectional stock ranking (relative quality) |
| 13 | Scenarios | What-if scenario comparison |
| 14 | Regime | Bull/Bear/Sideways detection + indicators |
| 15 | Correlation | Cross-asset correlation heatmap |
| 16 | Monitoring | System health + model drift + data freshness |
| 17 | Pipeline | Retraining pipeline status |
| 18 | Paper Trading | Simulated trading engine |
| 19 | MF Tracker | Mutual fund NAV, XIRR, allocation |
| 20 | Ledger | Historical decisions, signal accuracy, P&L |
| 21 | **Wealth Goals** | **Monte-Carlo retirement projection + wealth advisor** |

---

## Documentation

| Doc | Covers |
|---|---|
| `docs/Architecture.md` | Module map, data flow, entry scripts, multi-device scheduling |
| `docs/API.md` | All 73 endpoints, auth, middleware, errors |
| `docs/Data-Pipeline.md` | Sources, DuckDB point-in-time store, features, retention |
| `docs/Model-Training.md` | Architectures, walk-forward training, ensemble, meta-controller |
| `docs/Risk-Engine.md` | RiskController, RiskGuard breakers, kill switch, execution gate |
| `docs/Paper-Trading.md` | PaperTrader, fills, NSE costs, tax, ledger, benchmarks |
| `docs/Live-Trading-Roadmap.md` | Phase status, live gate, readiness requirements |
| `docs/Security-Model.md` | API auth, artifact verification, tokens, secrets |
| `docs/DEPLOYMENT.md` | Native + Docker deployment, one-command device setup, multi-device state sync |
| `docs/ROADMAP.md` | Full phase tracker with acceptance criteria |
| `docs/REAL_MONEY_READINESS.md` | Live-money audit verdict |

---

## Project Structure

```
stomar/
├── start_dev.sh               # Start FastAPI + React dev servers (Linux/macOS)
├── start_dev.bat              # Start FastAPI + React dev servers (Windows)
├── run_daily.py               # Single-day loop (legacy scheduler target)
├── run_pipeline.py            # Retraining pipeline
├── auto_pipeline.py           # Recommended: gap detection + backfill + daily + paper + summary
├── schedule_pipeline.py       # Windows Task Scheduler
├── verify_system.py           # System verification
├── pyproject.toml             # Python deps + ruff/pytest config
├── uv.lock                    # Locked dependency graph (commit it)
├── .python-version            # Pinned Python 3.13
├── Dockerfile
├── api/                       # FastAPI backend
│   ├── main.py                # App + CORS + response caching
│   ├── utils.py               # Parallel fetch utility
│   └── routers/               # 24 API routers
│       ├── predictions.py     # ML predictions + feature importance + explanations
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
│       ├── ledger.py          # Trading journal
│       ├── insights.py        # Recommendations + enrichment
│       ├── automation.py      # Automation run + decisions
│       └── wealth.py          # Wealth strategies + Monte-Carlo + advisor
├── web/                       # React frontend
│   ├── src/
│   │   ├── App.tsx            # Router with 21 routes
│   │   ├── components/        # Sidebar, ThemeProvider, UI components
│   │   ├── hooks/             # useApi, useDebouncedValue
│   │   ├── pages/             # 21 page components
│   │   └── lib/               # Utilities + typed API shapes (api-types.ts)
│   ├── .nvmrc                 # Node 22
│   ├── vite.config.ts         # Vite + proxy /api → :8000
│   └── package.json
├── src/                       # Python ML/trading logic
│   ├── core/                  # Core infrastructure
│   │   ├── constants.py       # Paths + global constants + NSE cost calc
│   │   ├── logging_config.py  # Structured logging
│   │   ├── pipeline.py        # Retraining pipeline
│   │   └── backfill.py        # Historical backfill
│   ├── data/                  # Data layer
│   │   ├── data_fetcher.py    # yfinance + NSE stock list
│   │   ├── data_sources.py    # Multi-source data with fallback
│   │   ├── data_validation.py # Price data validation
│   │   ├── features.py        # 48+ feature engineering + triple-barrier labels
│   │   └── feature_store.py   # Feature versioning
│   ├── models/                # ML models
│   │   ├── model.py           # 5 model architectures + save/load
│   │   ├── trainer.py         # Training + walk-forward validation
│   │   ├── ensemble.py        # Meta-learner + DL probability scaling
│   │   ├── meta_controller.py # Learned signal weighting (14 → 1 decision)
│   │   └── model_registry.py  # Model versioning + lifecycle
│   ├── signals/               # Signal modules (16)
│   │   ├── orchestrator.py    # Daily signal pipeline
│   │   ├── sentiment.py       # 5-source sentiment + FinBERT
│   │   ├── flow.py            # FII/DII flow + options PCR
│   │   ├── multitimeframe.py  # 4-timeframe analysis
│   │   ├── regime.py          # Bull/Bear/Sideways + ADX
│   │   ├── regime_strategy.py # Regime-conditional signals
│   │   ├── volatility.py      # Volatility forecasting (EWMA, GARCH)
│   │   ├── ranking.py         # Cross-sectional ranking
│   │   ├── risk.py            # VaR, CVaR, Sharpe, Kelly
│   │   ├── scenarios.py       # Scenario analysis
│   │   ├── benchmarks.py      # Benchmark strategies
│   │   ├── significance.py    # Statistical significance (CPCV, deflated Sharpe)
│   │   ├── execution_quality.py # Execution quality metrics
│   │   ├── monitoring.py      # Model drift + data freshness
│   │   ├── interpretability.py # Feature importance + SHAP explanations
│   │   ├── alpha_research.py  # Alpha research pipeline
│   │   └── mf_tracker.py      # Mutual fund NAV, XIRR
│   └── trading/               # Trading infrastructure
│       ├── engine.py          # Event-driven execution engine
│       ├── paper_trader.py    # Paper trading + short selling
│       ├── portfolio.py       # Portfolio tracking + P&L
│       ├── backtester.py      # Walk-forward backtesting
│       ├── ledger.py          # SQLite trading journal
│       ├── risk_controls.py   # Pre-trade risk controls + kill switch
│       ├── optimizer.py       # MVO, Black-Litterman
│       └── holdings.py        # Zerodha CSV parser
├── tests/                     # 750 tests
├── models/                    # Trained weights (gitignored)
│   ├── *.pt, *.pkl            # Per-ticker models
│   └── meta_controller.pkl    # Meta-controller
├── data/                      # Runtime data (gitignored)
│   ├── *.parquet              # OHLCV cache
│   ├── *.json                 # Sentiment/MTF cache
│   ├── stomar.db              # SQLite ledger
│   ├── paper_state.json       # Paper trading state
│   └── mf_state.json          # MF tracker state
```

---

## API Endpoints

### Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions/{ticker}` | GET | ML prediction + candlestick data |
| `/api/predictions/{ticker}/feature-importance` | GET | Top 10 features by XGBoost importance |
| `/api/predictions/{ticker}/explain` | GET | Full prediction explanation with direction signals |

### Meta-Controller
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/consensus/` | GET | All tickers consensus |

### Paper Trading
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/paper-trading/state` | GET | Current portfolio state |
| `/api/paper-trading/order` | POST | Place paper order |
| `/api/paper-trading/positions` | GET | Open positions |
| `/api/paper-trading/trades` | GET | Trade history |
| `/api/paper-trading/close-position` | POST | Close/cover a position |
| `/api/paper-trading/reset` | POST | Reset paper trading account |

### Monitoring
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/monitoring/` | GET | System health + model status |
| `/api/pipeline/status` | GET | Training pipeline status |
| `/api/pipeline/run` | POST | Start orchestrator run |
| `/api/pipeline/run/status` | GET | Orchestrator run progress |
| `/api/pipeline/train/{ticker}` | POST | Train one ticker's models |

---

## CLI Usage

```bash
# Recommended: auto-pipeline (idempotent full run: backfill + daily + paper + summary)
uv run auto_pipeline.py                          # Everything (safe to rerun daily)
uv run auto_pipeline.py --force                  # Force even if already ran today
uv run auto_pipeline.py --setup-run              # Setup device + run (one command)

# Single-day loop (legacy)
uv run run_daily.py                              # All 20 NSE stocks
uv run run_daily.py --ticker RELIANCE.NS         # Specific tickers
uv run run_daily.py --backfill                   # Backfill 1 year
uv run run_daily.py --train-meta                 # Retrain meta-controller
uv run run_daily.py --paper-trade                # Auto paper trades
uv run run_daily.py --paper-trade --capital 500000

# Retraining pipeline
uv run run_pipeline.py --train-all               # Train all 20 stocks
uv run run_pipeline.py --train RELIANCE.NS       # Specific tickers
uv run run_pipeline.py --paper                   # Train + paper trade

# Scheduler (host cron / systemd timer / Windows Task Scheduler)
uv run schedule_pipeline.py                      # Install daily 15:45 IST + boot catch-up
uv run schedule_pipeline.py --remove             # Remove task
uv run schedule_pipeline.py --run-now            # Run immediately

# Readiness automation (live-money gate evidence, paper-only)
uv run scripts/readiness_watchdog.py             # One daily cycle (battery + supervised dry-run)
uv run scripts/readiness_watchdog.py --install   # Timers: Mon-Fri 16:10 + Sat deep replay
uv run scripts/readiness_watchdog.py --status    # Green-streak progress (x/14)
uv run scripts/readiness_report.py               # GO/NO-GO verdict across all requirements
```

---

## Docker (one-command startup)

```bash
cp .env.example .env    # set STOMAR_API_KEY, SMTP, PAPER_CAPITAL...
docker compose up -d --build
```

- `api` — FastAPI + bundled React SPA on port 8000, healthchecked, auto-restart.
- `scheduler` — cron daemon in-container running the tested `schedule_pipeline.py`
  (weekdays 15:45 IST auto-pipeline run + boot catch-up for missed days).
- State persists in named volumes `stomar-data` (SQLite/ledger/paper state/logs)
  and `stomar-models` (trained weights + meta-controller).
- CPU-only torch is baked into the image (see `pyproject.toml` `[tool.uv.sources]`);
  no GPU or CUDA runtime needed.

Verify: `curl http://localhost:8000/api/health` and `docker compose exec scheduler crontab -l`.

## Tests

```bash
# Run all 750 tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Lint check (whole repo, same as CI)
uv run ruff check .
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19 + TypeScript + Tailwind CSS + React Router 8 + Recharts (dark/light theme) |
| **Backend** | FastAPI + Uvicorn + Pydantic (CORS, response caching, parallel fetch) |
| **ML/DL** | PyTorch 2.0+, XGBoost 2.0+, LightGBM 4.0+ |
| **NLP** | HuggingFace Transformers (ProsusAI/finbert) |
| **Interpretability** | XGBoost feature importance + optional SHAP |
| **Data** | yfinance, NSE India, Google News RSS, MoneyControl, ET, Screener.in |
| **Optimization** | SciPy, scikit-learn (Ledoit-Wolf) |
| **Technical Analysis** | `ta` library (30+ indicators) |
| **Storage** | SQLite (ledger), Parquet (data cache) |
| **Testing** | pytest (750 tests), ruff (linting) |

---

## Key Improvements Over Basic Systems

| Feature | StoMar | Basic Systems |
|---------|--------|---------------|
| **ML Models** | 5-model ensemble + meta-learner | Single model (Random Forest/LSTM) |
| **Target Labels** | Triple-barrier (profit/loss/time) | Simple 1-day return direction |
| **Validation** | Walk-forward chronological splits | Random train/test split (leaky) |
| **Signal Integration** | 14 modules via regularized logistic meta-model | Single technical indicator |
| **Risk Management** | VaR, CVaR, Kelly, kill switch | Basic stop-loss |
| **Interpretability** | Feature importance + SHAP + explanations | None |
| **Cost Modeling** | Full NSE costs (STT, brokerage, GST) | Ignored or simplified |
| **Regime Detection** | Bull/Bear/Sideways with ADX | None |
| **Paper Trading** | Short selling + position sizing | Buy-only |

---

## Important Disclaimers

- **Educational purposes only** — not financial advice
- Past performance does not guarantee future results
- Always do your own research before investing
- Start with paper trading before using real money
- Walk-forward accuracy varies: some stocks 55-72%, others 44-52%
- Meta-controller trained on backfill data — real accuracy may differ
- Model interpretability helps understand predictions but doesn't guarantee accuracy

---

## License

Licensed under the [Apache License 2.0](LICENSE).
