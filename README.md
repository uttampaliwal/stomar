# StoMar — Stock Market Prediction & Trading Suite

A state-of-the-art quantitative analysis platform for the Indian NSE market. Built with a 5-model ML ensemble, walk-forward backtesting, Black-Litterman portfolio optimization, and a professional 9-tab Streamlit terminal.

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd stomar

# 2. Create virtual environment (Python 3.14)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app.py
```

> **Note:** TensorFlow does NOT work on Python 3.14. This project uses **PyTorch** as the deep learning framework.

---

## Features

### 5-Model ML Ensemble
| Model | Type | Weight |
|-------|------|--------|
| LSTM | Deep Learning (PyTorch) | 20% |
| GRU | Deep Learning (PyTorch) | 20% |
| Transformer | Deep Learning (PyTorch) | 20% |
| XGBoost | Gradient Boosting | 20% |
| LightGBM | Gradient Boosting | 20% |

- **41 technical features** per prediction (RSI, MACD, Bollinger Bands, ATR, ADX, etc.)
- **Weighted voting** across all 5 models
- **Confidence scoring** based on model agreement

### Walk-Forward Backtesting
- **No data leakage:** Trains on 3 years, tests on 1 year, rolls forward
- **Out-of-sample metrics:** Accuracy, simulated annual return, Sharpe ratio
- **Per-window breakdown** with visual accuracy bars
- **Brokerage + slippage** simulation (0.03% brokerage, 0.1% slippage)

### Alternative Data Integration
- **News Sentiment:** FinBERT (ProsusAI/finbert) + Google News RSS, 30-min cache
- **FII/DII Flow:** NSE institutional flow data (real-time when market open)
- **Options PCR:** Put-Call Ratio + Max Pain from NSE options chain
- **Multi-Timeframe:** 15min / 1h / Daily / Weekly analysis with 8 indicators each

### Portfolio Management
- **Mean-Variance Optimization:** Max Sharpe + Min Variance portfolios
- **Black-Litterman:** Combines market equilibrium with custom views
- **Efficient Frontier** visualization
- **Ledoit-Wolf shrinkage** for stable covariance estimation

### Risk Management
- **VaR / CVaR** (Value at Risk, Conditional VaR)
- **Sharpe / Sortino / Calmar** ratios
- **Kelly Criterion** position sizing
- **Max Drawdown** analysis
- **Market Regime Detection** (Bull / Bear / Sideways)

### Holdings Tracker
- **Zerodha CSV import** — parses mutual fund holdings
- **Category allocation** (Gold, Silver, Debt, Equity, etc.)
- **Concentration risk** detection (Herfindahl index)
- **P&L tracking** with return calculation

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit 1.58+ with custom CSS (glassmorphism, dark-first design) |
| **ML/DL** | PyTorch 2.10+, XGBoost 2.0+, LightGBM 4.0+ |
| **NLP** | HuggingFace Transformers (ProsusAI/finbert) |
| **Data** | yfinance (NSE stock data), NSE website (FII/DII, options) |
| **Optimization** | SciPy (minimize), scikit-learn (Ledoit-Wolf) |
| **Visualization** | Plotly (interactive charts) |
| **Technical Analysis** | `ta` library (30+ indicators) |

---

## Project Structure

```
stomar/
├── app.py                  # Main Streamlit app (9 tabs, CSS, helpers)
├── requirements.txt        # Python dependencies
├── run.bat                 # Windows launcher
├── src/
│   ├── data_fetcher.py     # yfinance data + NSE stock list
│   ├── features.py         # 41-feature engineering pipeline
│   ├── model.py            # 5 model architectures + save/load
│   ├── trainer.py          # Training loop for all 5 models
│   ├── ensemble.py         # Weighted ensemble prediction
│   ├── backtester.py       # Walk-forward backtesting engine
│   ├── portfolio.py        # Portfolio class (buy/sell/equity)
│   ├── sentiment.py        # FinBERT + Google News sentiment
│   ├── flow.py             # FII/DII flow + options PCR
│   ├── multitimeframe.py   # 4-timeframe analysis
│   ├── risk.py             # VaR, CVaR, Sharpe, Kelly, etc.
│   ├── optimizer.py        # MVO, Black-Litterman, efficient frontier
│   ├── holdings.py         # Zerodha CSV parser + portfolio stats
│   └── regime.py           # Bull/Bear/Sideways detection
├── models/                 # Saved model weights (per ticker)
├── data/                   # Cached parquet, sentiment JSON, etc.
└── docs/                   # Project documentation
    ├── ARCHITECTURE.md     # System design & data flow
    ├── DESIGN.md           # UI/UX design system
    ├── API.md              # Module interfaces & function signatures
    └── DEPLOYMENT.md       # Setup & configuration
```

---

## Trained Stocks

The following stocks have been trained with the 5-model ensemble:

| Stock | Ensemble Accuracy | Walk-Forward Accuracy |
|-------|------------------|----------------------|
| RELIANCE.NS | 49.4% | 50.8% |
| TCS.NS | 45.2% | — |
| HDFCBANK.NS | 47.6% | — |

> Accuracy near 50% is expected for short-term stock prediction — even a 51% edge is profitable at scale.

---

## Important Disclaimers

- **Educational purposes only** — not financial advice
- Past performance does not guarantee future results
- Always do your own research before investing
- Start with paper trading before using real money
- The model does not account for black swan events, political news, or company-specific events

---

## License

Private project — not for redistribution.
