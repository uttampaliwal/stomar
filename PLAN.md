# Cleanup & Reorganization Plan

Audit of current state, issues found, and exact steps to prepare for Stage 1 implementation.

---

## Current File Structure

```
stomar/
├── .git/
├── .gitignore              # INCOMPLETE - missing *.pt, *.pkl in data/, *.json
├── API.md                  # Documentation (keep)
├── ARCHITECTURE.md         # Documentation (keep)
├── DEPLOYMENT.md           # Documentation (keep)
├── DESIGN.md               # Documentation (keep)
├── IMPROVEMENTS.md         # Documentation (keep)
├── README.md               # Documentation (keep)
├── app.py                  # 1258 lines - main UI
├── requirements.txt        # INCOMPLETE - missing scipy, requests
├── run.bat                 # Launcher (keep)
├── data/                   # Cache files (should be gitignored)
│   ├── fii_dii.parquet
│   ├── HDFCBANK_NS.parquet
│   ├── mtf_HDFCBANK_NS.pkl
│   ├── mtf_RELIANCE_NS.pkl
│   ├── mtf_TCS_NS.pkl
│   ├── RELIANCE_NS.parquet
│   ├── sentiment_RELIANCE_NS.json
│   └── TCS_NS.parquet
├── models/                 # Trained models (should be gitignored)
│   ├── HDFCBANK_NS_*.pkl/pt  (8 files)
│   ├── RELIANCE_NS_*.pkl/pt  (8 files)
│   └── TCS_NS_*.pkl/pt       (8 files)
└── src/                    # Source code (14 modules)
    ├── __init__.py         # EMPTY
    ├── __pycache__/        # Should be gitignored
    ├── backtester.py
    ├── data_fetcher.py
    ├── ensemble.py
    ├── features.py
    ├── flow.py
    ├── holdings.py
    ├── model.py
    ├── multitimeframe.py
    ├── optimizer.py
    ├── portfolio.py
    ├── regime.py
    ├── risk.py
    ├── sentiment.py
    └── trainer.py
```

---

## Issues Found

### 1. .gitignore Gaps (Critical)

**Problem:** Model weight files (`.pt`) and data cache files (`.pkl`, `.json`) are NOT gitignored.

```
NOT IGNORED: models/HDFCBANK_NS_gru.pt     (663KB)
NOT IGNORED: models/HDFCBANK_NS_lstm.pt    (1.4MB)
NOT IGNORED: models/HDFCBANK_NS_transformer.pt (632KB)
NOT IGNORED: models/RELIANCE_NS_gru.pt     (664KB)
NOT IGNORED: models/RELIANCE_NS_lstm.pt    (1.4MB)
NOT IGNORED: models/RELIANCE_NS_transformer.pt (631KB)
NOT IGNORED: models/TCS_NS_gru.pt          (663KB)
NOT IGNORED: models/TCS_NS_lstm.pt         (1.4MB)
NOT IGNORED: models/TCS_NS_transformer.pt  (632KB)
```

**Impact:** 9 PyTorch model files (~8MB total) would be committed to git. These are regenerable and should NOT be tracked.

**Fix:** Update `.gitignore` to add `models/*.pt` and `data/*.pkl`, `data/*.json`.

### 2. Missing packages in requirements.txt

```
MISSING: scipy     (used in src/optimizer.py)
MISSING: requests  (used in src/flow.py, src/sentiment.py)
```

**Fix:** Add `scipy>=1.11` and `requests>=2.31` to `requirements.txt`.

### 3. Unused Imports (13 found)

| File | Unused Import |
|------|---------------|
| `src/data_fetcher.py` | `np` (numpy), `timedelta` |
| `src/model.py` | `MinMaxScaler` |
| `src/trainer.py` | `pd` (pandas) |
| `src/backtester.py` | `EPOCHS`, `LEARNING_RATE`, `add_technical_indicators` |
| `src/flow.py` | `np` (numpy), `timedelta` |
| `src/multitimeframe.py` | `np` (numpy) |
| `src/risk.py` | `Optional` |
| `src/holdings.py` | `datetime`, `json` |

### 4. Dead Code (Functions Never Called)

**Truly dead (0 external references):**

| File | Function | Status |
|------|----------|--------|
| `src/risk.py` | `check_portfolio_risk()` | Never called anywhere |
| `src/risk.py` | `fixed_fraction_sizing()` | Never called anywhere |
| `src/risk.py` | `volatility_position_size()` | Never called anywhere |
| `src/risk.py` | `max_position_value()` | Never called anywhere |
| `src/holdings.py` | `compute_xirr()` | Never called anywhere |
| `src/holdings.py` | `fetch_mf_performance()` | Never called anywhere |
| `src/holdings.py` | `save_holdings()` | Never called anywhere |
| `src/holdings.py` | `load_holdings()` | Never called anywhere |
| `src/data_fetcher.py` | `get_live_price()` | Never called anywhere |
| `src/trainer.py` | `train_multiple_stocks()` | Never called anywhere |
| `src/features.py` | `prepare_lstm_data()` | Never called anywhere |

**Note:** Some of these are utility functions that MIGHT be used later. `compute_xirr()` and `train_multiple_stocks()` are likely candidates for future use. Keep them but mark as unused.

### 5. Hardcoded Magic Numbers (Repeated Across Files)

| Value | Meaning | Files Using It |
|-------|---------|----------------|
| `0.0003` | Brokerage rate | `backtester.py`, `portfolio.py` |
| `0.001` | Slippage rate | `backtester.py` |
| `0.065` | Risk-free rate (6.5%) | `backtester.py`, `risk.py`, `optimizer.py` |
| `0.001` | Min denominator guard | `backtester.py`, `risk.py` |

**Fix:** Create `src/constants.py` with shared constants.

### 6. Missing Files for Stage 1

| File | Purpose | Priority |
|------|---------|----------|
| `tests/` | Test directory | CRITICAL |
| `tests/__init__.py` | Package init | CRITICAL |
| `tests/conftest.py` | Pytest fixtures | CRITICAL |
| `tests/test_backtester.py` | No-leakage tests | CRITICAL |
| `tests/test_features.py` | Feature tests | HIGH |
| `tests/test_model.py` | Model save/load tests | HIGH |
| `tests/test_risk.py` | Risk metric tests | HIGH |
| `tests/test_portfolio.py` | Portfolio tests | HIGH |
| `.github/workflows/ci.yml` | CI pipeline | HIGH |
| `Dockerfile` | Container | MEDIUM |
| `.env.example` | Env template | MEDIUM |
| `pyproject.toml` | Project config | MEDIUM |
| `ruff.toml` | Linting config | MEDIUM |
| `src/constants.py` | Shared constants | HIGH |

### 7. Code Inconsistencies

- **Error handling:** Some functions use try/except, others don't
- **Return types:** `load_models()` returns 7 values (inconsistent with old 6-value pattern)
- **Docstrings:** Zero docstrings in entire codebase
- **Type hints:** Only `src/risk.py` uses type hints, no other module does

---

## Execution Plan

### Step 1: Fix .gitignore
```
# Add these lines:
models/*.pt
data/*.pkl
data/*.json
tests/
*.egg-info/
.ruff_cache/
.pytest_cache/
```

### Step 2: Fix requirements.txt
```
# Add missing packages:
scipy>=1.11
requests>=2.31
pytest>=8.0
ruff>=0.4
```

### Step 3: Create src/constants.py
```python
# Shared constants across the project
BROKERAGE_RATE = 0.0003      # 0.03% per side (Zerodha delivery)
SLIPPAGE_RATE = 0.001        # 0.1% slippage estimate
RISK_FREE_RATE = 0.065       # 6.5% Indian 10Y G-Sec yield
MIN_DENOMINATOR = 0.001      # Guard against division by zero
NSE_TRANSACTION_COSTS = {
    "brokerage_pct": 0.0003,
    "stt_sell_pct": 0.001,
    "exchange_charge_pct": 0.0000345,
    "sebi_fees_pct": 0.000001,
    "stamp_duty_buy_pct": 0.00015,
    "gst_pct": 0.18,
}
```

### Step 4: Clean Unused Imports
Remove the 13 unused imports identified above.

### Step 5: Create Test Infrastructure
```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_backtester.py    # No-leakage tests (CRITICAL)
├── test_features.py      # Feature count, NaN handling
├── test_model.py         # Save/load roundtrip
├── test_risk.py          # VaR, Kelly, Sharpe
├── test_portfolio.py     # Buy/sell, equity curve
└── test_data_fetcher.py  # Data fetch, cache
```

### Step 6: Create CI Pipeline
`.github/workflows/ci.yml` with pytest + ruff.

### Step 7: Create Dockerfile
Simple Python 3.14 slim image.

### Step 8: Create pyproject.toml
Project metadata, ruff config, pytest config.

---

## Recommended Execution Order

Execute these in this exact order:

1. **Fix .gitignore** (1 min) — prevent model files from being committed
2. **Fix requirements.txt** (1 min) — add missing packages
3. **Create src/constants.py** (5 min) — extract magic numbers
4. **Update files to use constants** (15 min) — replace hardcoded values
5. **Clean unused imports** (10 min) — remove 13 unused imports
6. **Create tests/ directory + conftest.py** (5 min)
7. **Create test_backtester.py** (30 min) — most critical test
8. **Create test_features.py** (15 min)
9. **Create test_model.py** (15 min)
10. **Create test_risk.py** (15 min)
11. **Create test_portfolio.py** (15 min)
12. **Create .github/workflows/ci.yml** (10 min)
13. **Create Dockerfile** (5 min)
14. **Create pyproject.toml** (10 min)
15. **Run pytest to verify** (5 min)

**Total estimated time:** ~2.5 hours

---

## After Cleanup: Ready State for Stage 1

```
stomar/
├── .github/workflows/ci.yml    # NEW - CI pipeline
├── .gitignore                   # FIXED - comprehensive
├── Dockerfile                   # NEW - container
├── pyproject.toml               # NEW - project config
├── requirements.txt             # FIXED - all deps
├── app.py                       # CLEANED - no unused imports
├── tests/                       # NEW - test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_backtester.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_risk.py
│   └── test_portfolio.py
├── src/
│   ├── __init__.py
│   ├── constants.py             # NEW - shared constants
│   ├── backtester.py            # CLEANED - uses constants
│   ├── data_fetcher.py          # CLEANED
│   ├── ensemble.py
│   ├── features.py              # CLEANED
│   ├── flow.py                  # CLEANED
│   ├── holdings.py
│   ├── model.py                 # CLEANED
│   ├── multitimeframe.py        # CLEANED
│   ├── optimizer.py             # CLEANED - uses constants
│   ├── portfolio.py             # CLEANED - uses constants
│   ├── regime.py
│   ├── risk.py                  # CLEANED - uses constants
│   ├── sentiment.py
│   └── trainer.py               # CLEANED
├── data/                        # Gitignored
├── models/                      # Gitignored
└── docs/                        # All .md files
```

This state is ready for Stage 1 implementation (point-in-time fixes, realistic cost model, structured logging).
