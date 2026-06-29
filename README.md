# StoMar — Stock Market Prediction & Trading Suite

A quantitative analysis platform for the Indian NSE market. Built with a 5-model ML ensemble, stacked meta-learner with regime routing, walk-forward backtesting, Black-Litterman portfolio optimization, and a professional 17-tab Streamlit terminal.

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

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app.py
```

---

## Features

### 5-Model ML Ensemble + Meta-Learner

| Model | Type | Base Weight |
|-------|------|-------------|
| LSTM | Deep Learning (PyTorch) | 20% |
| GRU | Deep Learning (PyTorch) | 20% |
| Transformer | Deep Learning (PyTorch) | 20% |
| XGBoost | Gradient Boosting | 20% |
| LightGBM | Gradient Boosting | 20% |

- **Stacked meta-learner** learns optimal model combinations from base predictions
- **Regime-conditional routing** — Bull favors tree models, Bear favors DL, Sideways uses equal weights
- **41 technical features** per prediction (RSI, MACD, Bollinger Bands, ATR, ADX, etc.)

### Walk-Forward Backtesting
- **No data leakage:** Trains on 3 years, tests on 1 year, rolls forward
- **Out-of-sample metrics:** Accuracy, simulated annual return, Sharpe ratio
- **Brokerage + slippage** simulation (configurable slippage models)

### Alternative Data Integration
- **Multi-source Sentiment:** Yahoo Finance, Google News RSS, MoneyControl, Economic Times, Screener.in with source-weighted aggregation
- **FII/DII Flow:** NSE institutional flow data
- **Options PCR:** Put-Call Ratio + Max Pain from NSE options chain
- **Multi-Timeframe:** 15min / 1h / Daily / Weekly analysis

### Portfolio Management
- **Mean-Variance Optimization:** Max Sharpe + Min Variance portfolios
- **Black-Litterman:** Combines market equilibrium with custom views
- **Efficient Frontier** visualization
- **Ledoit-Wolf shrinkage** for stable covariance estimation

### Risk Management
- **VaR / CVaR** (Value at Risk, Conditional VaR)
- **Sharpe / Sortino / Calmar** ratios
- **Kelly Criterion** position sizing
- **Market Regime Detection** (Bull / Bear / Sideways)

### Event-Driven Engine & Paper Trading
- **Event-driven backtesting engine** with realistic order matching
- **Volume-weighted slippage** and adaptive slippation models
- **Paper trading** with state persistence
- **Execution quality analysis** (fill rates, latency, cost decomposition)

### Mutual Fund Tracker
- **NAV history** via yfinance (21 Indian MF schemes)
- **XIRR computation** per fund and portfolio level
- **Factor exposure analysis** (value, momentum, quality)
- **Concentration risk detection** and allocation breakdown
- **Benchmark comparison** vs Nifty 50

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit 1.58+ with custom CSS (glassmorphism, dark-first) |
| **ML/DL** | PyTorch 2.0+, XGBoost 2.0+, LightGBM 4.0+ |
| **NLP** | HuggingFace Transformers (ProsusAI/finbert) |
| **Data** | yfinance, NSE India, Google News RSS, MoneyControl, ET, Screener.in |
| **Optimization** | SciPy, scikit-learn (Ledoit-Wolf) |
| **Visualization** | Plotly (interactive charts) |
| **Technical Analysis** | `ta` library (30+ indicators) |

---

## Project Structure

```
stomar/
├── app.py                     # Main Streamlit app (17 tabs)
├── run_pipeline.py            # CLI entry point (train, paper trade)
├── run.bat                    # Windows launcher
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata + ruff/pytest config
├── Dockerfile                 # Container deployment
├── LICENSE                    # Apache 2.0
├── CONTRIBUTING.md            # Contribution guidelines
├── src/
│   ├── data_fetcher.py        # yfinance data + NSE stock list
│   ├── data_sources.py        # Multi-source data with fallback
│   ├── data_validation.py     # Data quality checks
│   ├── features.py            # 41-feature engineering pipeline
│   ├── feature_store.py       # Feature versioning + caching
│   ├── model.py               # 5 model architectures + save/load
│   ├── model_registry.py      # Model versioning
│   ├── trainer.py             # Training loop + batch training
│   ├── ensemble.py            # Meta-learner + regime routing
│   ├── backtester.py          # Walk-forward backtesting engine
│   ├── engine.py              # Event-driven backtesting engine
│   ├── portfolio.py           # Portfolio class (buy/sell/equity)
│   ├── sentiment.py           # 5-source sentiment analysis
│   ├── flow.py                # FII/DII flow + options PCR
│   ├── multitimeframe.py      # 4-timeframe analysis
│   ├── risk.py                # VaR, CVaR, Sharpe, Kelly, etc.
│   ├── risk_controls.py       # Slippage models, risk limits
│   ├── optimizer.py           # MVO, Black-Litterman, efficient frontier
│   ├── holdings.py            # Zerodha CSV parser + portfolio stats
│   ├── regime.py              # Bull/Bear/Sideways detection
│   ├── regime_strategy.py     # Regime-conditional trading signals
│   ├── mf_tracker.py          # Mutual fund NAV, XIRR, factors
│   ├── paper_trader.py        # Paper trading with state persistence
│   ├── execution_quality.py   # Execution quality analysis
│   ├── monitoring.py          # System health monitoring
│   ├── benchmarks.py          # Benchmark comparison
│   ├── alpha_research.py      # Alpha factor research
│   ├── volatility.py          # Volatility forecasting
│   ├── ranking.py             # Fundamental + technical ranking
│   ├── scenarios.py           # Scenario analysis
│   ├── significance.py        # Statistical significance tests
│   ├── constants.py           # Global constants
│   ├── logging_config.py      # Logging setup
│   └── __init__.py
├── tests/                     # 507 tests (20 files)
├── models/                    # Saved model weights (per ticker, gitignored)
├── data/                      # Cached data (gitignored)
└── docs/
    ├── ARCHITECTURE.md        # System design & data flow
    ├── DESIGN.md              # UI/UX design system
    ├── API.md                 # Module interfaces & function signatures
    ├── DEPLOYMENT.md          # Setup & configuration
    ├── PLAN.md                # Stage-by-stage development plan
    └── IMPROVEMENTS.md        # Feature requirements & roadmap
```

---

## 17-Tab Dashboard

| # | Tab | Purpose |
|---|-----|---------|
| 1 | Predictions | ML model predictions + confidence |
| 2 | Portfolio | Current holdings + allocation |
| 3 | Backtest | Walk-forward backtest results |
| 4 | Scanner | Multi-stock screening |
| 5 | Sentiment | News sentiment analysis |
| 6 | Market Pulse | FII/DII flow, PCR, market health |
| 7 | Optimizer | Portfolio optimization (MVO, Black-Litterman) |
| 8 | Holdings | Zerodha CSV import + stats |
| 9 | Risk | VaR, CVaR, drawdown, Kelly |
| 10 | Volatility | Volatility forecasting |
| 11 | Ranking | Fundamental + technical stock ranking |
| 12 | Scenarios | What-if scenario analysis |
| 13 | Regime Strategy | Regime-conditional trading signals |
| 14 | Monitoring | System health + model drift |
| 15 | Pipeline | Retraining pipeline status |
| 16 | Paper Trading | Simulated trading engine |
| 17 | MF Tracker | Mutual fund NAV, XIRR, allocation |

---

## CLI Usage

```bash
# Train all 20 NSE stocks
python run_pipeline.py --train-all

# Train specific tickers
python run_pipeline.py --train RELIANCE.NS TCS.NS

# Full pipeline with paper trading
python run_pipeline.py --paper

# Run specific tickers through pipeline
python run_pipeline.py RELIANCE.NS HDFCBANK.NS
```

---

## Tests

```bash
# Run all 507 tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Lint check
python -m ruff check src/ tests/ --output-format=concise
```

---

## Important Disclaimers

- **Educational purposes only** — not financial advice
- Past performance does not guarantee future results
- Always do your own research before investing
- Start with paper trading before using real money

---

## License

Licensed under the [Apache License 2.0](LICENSE).
