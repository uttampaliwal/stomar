# Deployment Guide

Setup, configuration, and troubleshooting for StoMar.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.14 | 3.13+ should work, but 3.14 is tested |
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

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python -m streamlit run app.py
```

### Linux / macOS

```bash
git clone <repo-url>
cd stomar
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

### Windows Batch Launcher

Double-click `run.bat` (after activating venv manually, or edit the bat file).

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.38 | Web UI framework |
| `yfinance` | ≥0.2.44 | Yahoo Finance data (NSE stocks) |
| `pandas` | ≥2.2 | Data manipulation |
| `numpy` | ≥1.26 | Numerical operations |
| `scikit-learn` | ≥1.5 | ML utilities, preprocessing, metrics |
| `xgboost` | ≥2.0 | XGBoost gradient boosting |
| `lightgbm` | ≥4.0 | LightGBM gradient boosting |
| `torch` | ≥2.0 | Deep learning (LSTM, GRU, Transformer) |
| `plotly` | ≥5.25 | Interactive charts |
| `ta` | ≥0.11 | Technical analysis indicators |
| `joblib` | ≥1.4 | Model serialization |
| `python-dotenv` | ≥1.0 | Environment variable loading |
| `transformers` | ≥4.30 | HuggingFace FinBERT for sentiment |

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
├── app.py
├── requirements.txt
├── run.bat
├── .env                    # Optional
├── src/                    # Source code (14 modules)
├── models/                 # Auto-created when you train
│   ├── RELIANCE.NS_lstm.pt
│   ├── RELIANCE.NS_gru.pt
│   ├── ...
├── data/                   # Auto-created for caching
│   ├── *.parquet           # Stock data cache
│   ├── sentiment_cache/    # Sentiment JSON cache
│   ├── fii_dii/            # FII/DII data cache
│   ├── options/            # Options chain cache
│   └── mtf/                # Multi-timeframe cache
└── docs/                   # This documentation
```

---

## Running the App

### Standard Launch
```powershell
python -m streamlit run app.py
```
Opens at: `http://localhost:8501`

### Custom Port
```powershell
python -m streamlit run app.py --server.port 8080
```

### Headless Mode (no browser auto-open)
```powershell
python -m streamlit run app.py --server.headless true
```

---

## First-Time Setup

1. **Launch the app** — `python -m streamlit run app.py`
2. **Go to Predictions tab** — Select a stock (e.g., RELIANCE.NS)
3. **Click "Train Model"** — Trains all 5 models (~30-90 seconds)
4. **View prediction** — Ensemble signal appears with confidence
5. **Try other tabs** — Backtest, Scanner, Sentiment, etc.

### Training Times (approximate)
| Stock | Time | Notes |
|-------|------|-------|
| RELIANCE.NS | ~45s | 2 years of daily data |
| TCS.NS | ~50s | Slightly more volatility |
| HDFCBANK.NS | ~40s | Stable data |

Walk-forward backtest takes ~2-5 minutes per stock (trains 4 NNs × multiple windows).

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```powershell
pip install streamlit
# Or ensure venv is activated
```

### "torch.cuda.is_available() = False"
- Normal if you don't have an NVIDIA GPU
- App works fine on CPU, just slower training
- To install CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu130`

### "ImportError: cannot import name 'run_simple_backtest'"
```powershell
# Clear cached .pyc files
rd /s /q __pycache__
rd /s /q src\__pycache__
python -m streamlit run app.py
```

### "AttributeError: 'DataFrame' object has no attribute 'iteritems'"
```powershell
pip install --upgrade pandas
```

### NSE API returns 404 (FII/DII data)
- Normal when market is closed (weekends/holidays)
- FII/DII and Options PCR only work during market hours (9:15 AM - 3:30 PM IST)

### Sentiment analysis is slow
- First run downloads FinBERT model (~400MB)
- Subsequent runs use cache (30-min TTL)
- If slow, check internet connection

### "use_container_width is deprecated"
- This is a Streamlit version issue
- StoMar uses `width='stretch'` (Streamlit 1.38+)

### App crashes on startup
1. Check Python version: `python --version`
2. Check all packages installed: `pip list`
3. Clear all caches: `rd /s /q __pycache__`
4. Check for syntax errors: `python -c "import ast; ast.parse(open('app.py').read())"`

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
2. Go to Predictions tab
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
- Reduce `@st.cache_data(ttl=3600)` TTL — more frequent data refresh
- Train fewer stocks — less data to scan
- Use `--server.headless true` — skips browser auto-open

### Reduce Memory
- Train one stock at a time — models stay in memory
- Close other tabs — Streamlit reruns entire script on each interaction
- Use `--server.maxUploadSize 10` — limit file uploads

---

## Git Workflow

### .gitignore Explained
```
__pycache__/     # Python bytecode cache
*.pyc            # Compiled Python files
.env             # Environment secrets
data/*.parquet   # Cached stock data (regenerable)
models/*.pkl     # Saved models (regenerable)
models/*.keras   # Legacy TF models (not used)
.venv/           # Virtual environment
venv/            # Virtual environment
*.egg-info/      # Python package metadata
.DS_Store        # macOS metadata
```

### What's Tracked vs Not Tracked
| Tracked | Not Tracked |
|---------|-------------|
| All `src/*.py` files | `data/*.parquet` (cache) |
| `app.py` | `models/*.pt, *.pkl` (trained models) |
| `requirements.txt` | `__pycache__/` |
| `docs/*.md` | `.env` |
| `run.bat` | `venv/` |

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
