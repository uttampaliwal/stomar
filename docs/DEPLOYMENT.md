# Deployment Guide

Setup, configuration, and troubleshooting for StoMar.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.14 | 3.13+ should work, but 3.14 is tested |
| Node.js | 20+ | For React frontend build |
| pip | Latest | Comes with Python |
| Git | Latest | For cloning the repo |
| CUDA (optional) | 12.x | For GPU acceleration (PyTorch auto-detects) |

> **Important:** TensorFlow does NOT work on Python 3.14. StoMar uses PyTorch instead.

---

## Installation

### Windows

```powershell
# 1. Clone
git clone <repo-url>
cd stomar

# 2. Create virtual environment
python -m venv venv

# 3. Activate
venv\Scripts\activate

# 4. Install Python dependencies
pip install -e ".[dev]"

# 5. Install frontend dependencies
cd web && npm install && cd ..

# 6. Start both servers
start_dev.bat
```

### Linux / macOS

```bash
git clone <repo-url>
cd stomar
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cd web && npm install && cd ..
# Start FastAPI
uvicorn api.main:app --reload --port 8000 &
# Start React
cd web && npm run dev
```

---

## Dependencies

### Python (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `yfinance` | ≥0.2.44 | Yahoo Finance data (NSE stocks) |
| `pandas` | ≥2.2 | Data manipulation |
| `numpy` | ≥1.26 | Numerical operations |
| `scikit-learn` | ≥1.5 | ML utilities, preprocessing, metrics |
| `xgboost` | ≥2.0 | XGBoost gradient boosting |
| `lightgbm` | ≥4.0 | LightGBM gradient boosting |
| `torch` | ≥2.0 | Deep learning (LSTM, GRU, Transformer) |
| `ta` | ≥0.11 | Technical analysis indicators |
| `joblib` | ≥1.4 | Model serialization |
| `python-dotenv` | ≥1.0 | Environment variable loading |
| `transformers` | ≥4.30 | HuggingFace FinBERT for sentiment |
| `fastapi` | ≥0.111.0 | API backend framework |
| `uvicorn` | ≥0.30.0 | ASGI server |
| `pydantic` | ≥2.0 | Request/response validation |

### Frontend (`web/package.json`)

| Package | Purpose |
|---------|---------|
| `react` + `react-dom` | UI framework |
| `react-router-dom` | Client-side routing |
| `recharts` | Charts (candlestick, line, bar) |
| `lucide-react` | Icons |
| `tailwindcss` | Utility-first CSS |

### Optional GPU Support

For NVIDIA GPU acceleration:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

The app auto-detects CUDA and uses GPU if available. CPU-only works fine but is slower for training.

---

## Environment Variables

Create a `.env` file in the project root (optional):

```env
# No required env vars — app works out of the box
# These are optional:
HF_HOME=/path/to/huggingface/cache   # Override HuggingFace model cache
```

---

## Directory Structure After Setup

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
├── .env                       # Optional
├── api/                       # FastAPI backend
│   ├── main.py                # App + CORS + caching
│   └── routers/               # 20 API routers
├── web/                       # React frontend
│   ├── src/pages/             # 19 page components
│   └── vite.config.ts         # Vite config + /api proxy
├── src/                       # Source code (40 modules)
├── models/                    # Trained weights (gitignored)
│   ├── *.pt, *.pkl            # Per-ticker models
│   └── meta_controller.pkl    # Meta-controller
├── data/                      # Runtime data (gitignored)
│   ├── *.parquet              # OHLCV cache
│   ├── *.json                 # Sentiment/MTF cache
│   ├── stomar.db              # SQLite ledger
│   ├── paper_state.json       # Paper trading state
│   └── mf_state.json          # MF tracker state
└── docs/                      # This documentation
```

---

## Running the App

### Standard Launch (Windows)
```powershell
start_dev.bat
```
- React UI: `http://localhost:5173`
- FastAPI backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Manual Launch
```powershell
# Terminal 1: FastAPI
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: React
cd web && npm run dev
```

---

## First-Time Setup

1. **Start the app** — `start_dev.bat`
2. **Go to Predictions page** — Select a stock (e.g., RELIANCE.NS)
3. **Click "Train Model"** — Trains all 5 models (~30-90 seconds)
4. **View prediction** — Ensemble signal appears with confidence
5. **Try other pages** — Backtest, Scanner, Sentiment, Consensus, etc.

### Training Times (approximate)
| Stock | Time | Notes |
|-------|------|-------|
| RELIANCE.NS | ~45s | 1 year of daily data |
| TCS.NS | ~50s | Slightly more volatility |
| HDFCBANK.NS | ~40s | Stable data |

Walk-forward backtest takes ~2-5 minutes per stock (trains 4 NNs × multiple windows).

---

## CLI Commands

### Daily Autonomous Loop
```bash
# Run all 20 NSE stocks
python run_daily.py

# Specific tickers
python run_daily.py --ticker RELIANCE.NS TCS.NS

# Backfill 1 year of history + train meta-controller
python run_daily.py --backfill

# Backfill 6 months
python run_daily.py --backfill --days 126

# Retrain meta-controller only
python run_daily.py --train-meta

# Auto-execute paper trades
python run_daily.py --paper-trade

# Paper trade with custom capital
python run_daily.py --paper-trade --capital 500000

# Dry run (signals only, no ledger writes)
python run_daily.py --dry-run

# Custom ledger path
python run_daily.py --db custom.db
```

### Retraining Pipeline
```bash
# Train all 20 stocks
python run_pipeline.py --train-all

# Train specific tickers
python run_pipeline.py --train RELIANCE.NS TCS.NS

# Full pipeline with paper trading
python run_pipeline.py --paper
```

### Windows Scheduler
```bash
# Install daily task (4 PM IST)
python schedule_pipeline.py

# Custom time
python schedule_pipeline.py --time 16:00

# Run now
python schedule_pipeline.py --run-now

# Remove task
python schedule_pipeline.py --remove

# Check status
python schedule_pipeline.py --status
```

### System Verification
```bash
# Run end-to-end verification
python verify_system.py
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'fastapi'"
```powershell
pip install fastapi uvicorn
# Or ensure venv is activated
```

### "torch.cuda.is_available() = False"
- Normal if you don't have an NVIDIA GPU
- App works fine on CPU, just slower training
- To install CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu130`

### "Module not found" errors in React
```powershell
cd web
npm install
```

### NSE API returns 404 (FII/DII data)
- Normal when market is closed (weekends/holidays)
- FII/DII and Options PCR only work during market hours (9:15 AM - 3:30 PM IST)

### Sentiment analysis is slow
- First run downloads FinBERT model (~400MB)
- Subsequent runs use cache (30-min TTL)
- If slow, check internet connection

### App crashes on startup
1. Check Python version: `python --version`
2. Check Node.js version: `node --version`
3. Check all packages installed: `pip list`
4. Clear all caches: `rd /s /q __pycache__`

---

## Model Files Reference

Each trained stock creates 8 files in `models/`:

| File | Format | Size | Contents |
|------|--------|------|----------|
| `{ticker}_lstm.pt` | PyTorch | ~1-2 MB | LSTM weights |
| `{ticker}_gru.pt` | PyTorch | ~1-2 MB | GRU weights |
| `{ticker}_transformer.pt` | PyTorch | ~0.5-1 MB | Transformer weights |
| `{ticker}_xgb.pkl` | pickle | ~0.5 MB | XGBoost model |
| `{ticker}_lgb.pkl` | pickle | ~0.3 MB | LightGBM model |
| `{ticker}_scaler.pkl` | pickle | ~1 KB | MinMaxScaler |
| `{ticker}_features.pkl` | pickle | ~1 KB | Feature column names |
| `{ticker}_lstm_dim.pkl` | pickle | ~1 KB | Input dimension metadata |

**Total per stock:** ~5-20 MB

### To Retrain a Stock
1. Delete the model files: `del models\RELIANCE.NS_*.pkl models\RELIANCE.NS_*.pt`
2. Go to Predictions page
3. Select the stock
4. Click "Train Model"

Or just click "Train Model" — it retrains automatically when models exist (with `force_retrain=True`).

---

## Performance Tuning

### Speed Up Training
- Use GPU (NVIDIA CUDA) — 5-10x faster for LSTM/GRU/Transformer
- Reduce `EPOCHS` in `trainer.py` (default 40) — faster but less accurate
- Reduce `SEQ_LENGTH` in `trainer.py` (default 60) — faster but less context

### Speed Up UI
- Response cache (30s TTL) on expensive API endpoints
- In-memory cache (10min) on data fetcher, (1hr) on model loading
- Stale-while-revalidate in React `useApi` hook (30s)
- Debounced ticker selectors (400ms)

### Reduce Memory
- Train one stock at a time — models stay in memory
- React code-splits: recharts lazy-loaded (~407KB chunk)

---

## Git Workflow

### .gitignore Explained
```
__pycache__/     # Python bytecode cache
*.pyc            # Compiled Python files
.env             # Environment secrets
data/*.parquet   # Cached stock data (regenerable)
models/*.pkl     # Saved models (regenerable)
models/*.pt      # Saved PyTorch models
.venv/           # Virtual environment
venv/            # Virtual environment
*.egg-info/      # Python package metadata
.DS_Store        # macOS metadata
web/node_modules/ # Frontend dependencies
web/dist/        # Frontend build output
```

### What's Tracked vs Not Tracked
| Tracked | Not Tracked |
|---------|-------------|
| All `src/*.py` files | `data/*.parquet` (cache) |
| `api/*.py` | `models/*.pt, *.pkl` (trained models) |
| `web/src/**` | `web/node_modules/` |
| `requirements.txt` | `__pycache__/` |
| `docs/*.md` | `.env` |
| `start_dev.bat` | `venv/` |

---

## API Keys (Future)

Currently no API keys are required. All data sources are free:
- **yfinance:** Free, no key needed
- **NSE website:** Free, no key needed (public data)
- **Google News RSS:** Free, no key needed
- **HuggingFace FinBERT:** Free, auto-downloaded

If you want to add Zerodha Kite Connect (₹500/month) for live data + execution:
```env
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
```
