# Improvement Roadmap

Honest assessment of current state, what matters most, and concrete steps to get there.

---

## Current State (Reality Check)

| Metric | Value | What It Means |
|--------|-------|---------------|
| Stocks trained | 3 | Not a universe — a sample |
| Directional accuracy | 45–50% | Statistically indistinguishable from coin flip for 2/3 stocks |
| RELIANCE accuracy | 49.4% / 50.8% (WF) | Borderline — could be noise |
| Data source | yfinance (delayed) | Fine for research, not live |
| Backtesting | Walk-forward, no leakage | Good practice — this is the strongest part |
| Ensemble | Equal-weighted 5-model | Baseline — not optimized |
| Tests | 0 | No automated verification |
| Live capability | None | Educational/research only |

### What's Actually Good
- Clean modular architecture (leaf-node design, minimal coupling)
- Walk-forward backtesting with brokerage/slippage (rare in hobby projects)
- Feature pipeline covers technical + alternative data
- The code is honest about its limitations (README disclaimers)

### What's Actually Missing
- No statistical significance testing
- No regime-conditional model routing
- Equal-weight ensemble (not learned)
- File-based caching (no point-in-time guarantees)
- Zero tests
- 3 stocks ≠ a trading system

---

## Priority-Ordered Improvements

### Tier 1: Highest Impact, Do First

#### 1.1 Replace Equal-Weight Ensemble with Stacked Meta-Learner
**Impact:** Medium (typically +2-5% accuracy)
**Effort:** Low (1-2 days)
**File:** `src/ensemble.py`

**What to do:**
```python
# Current: equal weights
# ensemble_prob = 0.2*lstm + 0.2*gru + 0.2*transformer + 0.2*xgb + 0.2*lgb

# Better: learned weights via logistic regression on base model outputs
from sklearn.linear_model import LogisticRegression

# Train meta-learner on out-of-fold predictions from base models
meta_X = np.column_stack([lstm_probs, gru_probs, transformer_probs, xgb_probs, lgb_probs])
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_train)
ensemble_prob = meta_model.predict_proba(meta_X_test)[:, 1]
```

**Specifics:**
- Use k-fold cross-validation to generate out-of-fold predictions for meta-learner training
- Store the fitted meta-learner alongside base models
- This is the single easiest win in the entire roadmap

#### 1.2 Add Statistical Significance Testing
**Impact:** Critical (knows if your signal is real)
**Effort:** Medium (3-5 days)
**Files:** New `src/significance.py`, update `src/backtester.py`

**What to do:**
- **Deflated Sharpe Ratio** (Marcos López de Prado): Adjusts Sharpe for multiple testing bias. If you tested 100 strategies and found one with Sharpe 2.0, the deflated Sharpe might be 0.8 — noise.
- **Combinatorial Purged Cross-Validation (CPCV):** Walk-forward is good, but CPCV gives you the distribution of performance, not just a point estimate. Shows probability of ruin.
- **Minimum Backtest Length:** How many years of data do you need to be 95% confident the strategy isn't luck? With daily data and ~50% accuracy, you need hundreds of independent trades.

**Concrete steps:**
1. Track every strategy variation you test (even mentally) — the more you try, the more you need to deflate
2. Compute CPCV on walk-forward results: probability of losing money over N trades
3. Report confidence intervals on all metrics, not just point estimates
4. Add a "required sample size" calculator — tells you how many more trades needed

#### 1.3 Reframe: Direction → Volatility/Ranking
**Impact:** High (historically more exploitable edge)
**Effort:** Medium (1-2 weeks)
**Files:** `src/model.py`, `src/trainer.py`, `src/ensemble.py`

**Why:** Predicting "will RELIANCE go up tomorrow" is the hardest problem in finance. Predicting "will RELIANCE be more volatile than TCS tomorrow" or "rank these 20 stocks by expected return" has historically had more retail-accessible edge.

**Concrete alternatives:**
| Target | Edge Source | Data Needed |
|--------|------------|-------------|
| **Volatility forecasting** | Mean-reversion in vol, GARCH effects | Historical vol, options IV |
| **Cross-sectional ranking** | Relative strength, momentum | Universe of stocks |
| **Sector rotation** | Business cycle, macro | Sector ETFs, economic data |
| **Pairs trading** | Cointegration breakdowns | Correlated stock pairs |

**For volatility specifically:**
- Target: next-day realized volatility (not direction)
- Features: GARCH inputs, options IV, ATR, historical vol surface
- Metric: QLIKE loss (quasi-likelihood, standard for vol forecasting)
- This is where retail traders can actually find edge because institutional focus is elsewhere

#### 1.4 Add Regime-Conditional Model Routing
**Impact:** Medium-High (your regime detection already exists but isn't used)
**Effort:** Low (1-2 days)
**Files:** `src/ensemble.py`, `src/regime.py`

**What to do:**
```python
# Current: detect_regime() shows the regime but doesn't change model behavior
# Better: different ensemble weights per regime

if regime == "Bull":
    weights = {"lstm": 0.15, "gru": 0.15, "transformer": 0.2, "xgb": 0.25, "lgb": 0.25}
elif regime == "Bear":
    weights = {"lstm": 0.25, "gru": 0.25, "transformer": 0.15, "xgb": 0.15, "lgb": 0.2}
elif regime == "Sideways":
    weights = {"lstm": 0.2, "gru": 0.2, "transformer": 0.2, "xgb": 0.2, "lgb": 0.2}
```

**Specifics:**
- Train regime-specific meta-learners (different stacking model for each regime)
- Backtest each regime separately — accuracy should differ meaningfully
- This is where `regime.py` becomes actually useful instead of just decorative

---

### Tier 2: Important for Scale

#### 2.1 Move to Proper Time-Series Store
**Impact:** Medium (prevents silent bugs, scales)
**Effort:** Medium (3-5 days)
**Files:** `src/data_fetcher.py`, new `src/db.py`

**Current problem:** `@st.cache_data` + flat JSON/parquet files have no point-in-time guarantees. If you fetch data on Monday and a Tuesday feature uses Monday's close, but the feature was computed on Wednesday using Tuesday's close — that's look-ahead bias, and you won't see it.

**Solution options (pick one):**
| Option | Pros | Cons |
|--------|------|------|
| **DuckDB** | Zero-config, SQL, fast, file-based | New dependency |
| **SQLite + partitioning** | Already in Python stdlib | Slower for large datasets |
| **DVC + parquet** | Version control for data | More complex setup |

**DuckDB approach:**
```python
import duckdb
conn = duckdb.connect("data/stomar.duckdb")
# Store with timestamp partitioning
conn.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE,
        low DOUBLE, close DOUBLE, volume BIGINT,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
# Query with point-in-time guarantee
conn.execute("SELECT * FROM prices WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1", [ticker, as_of_date])
```

#### 2.2 Feature Store with Versioning
**Impact:** Medium (reproducibility)
**Effort:** Low (1-2 days)
**Files:** `src/features.py`, new `src/feature_store.py`

**What to do:**
- Hash the feature computation code + input data to create a feature version ID
- Store feature version alongside model training metadata
- When loading a model, verify the feature version matches
- This prevents "model trained on v1 features, running on v3 features" silent failures

```python
import hashlib

def compute_feature_version(feature_cols, df_hash):
    content = json.dumps(sorted(feature_cols)) + df_hash
    return hashlib.sha256(content.encode()).hexdigest()[:12]
```

#### 2.3 Expand Trained Universe
**Impact:** Medium (more stocks = more opportunities, better regime detection)
**Effort:** Low (automated)

**Current:** 3 stocks trained out of 20 in `NSE_STOCKS`

**What to do:**
- Batch train all 20 stocks (script it, ~15 minutes total)
- Add sector classification to each stock
- Track accuracy per sector — some sectors are more predictable than others
- Consider expanding NSE_STOCKS to NIFTY 50 or NIFTY 100

#### 2.4 Add Tests (Critical for Credibility)
**Impact:** High (the walk-forward backtest claim is the project's foundation)
**Effort:** Medium (3-5 days)
**Files:** New `tests/` directory

**What to test:**
```
tests/
├── test_backtester.py      # Most critical
│   ├── test_walk_forward_no_leakage()    # Verify no future data in training
│   ├── test_computed_metrics()           # Known inputs → known outputs
│   └── test_portfolio_value()            # Buy/sell → correct cash/holdings
├── test_features.py
│   ├── test_feature_count()              # Always 41 features
│   ├── test_no_nans_in_features()        # NaN handling works
│   └── test_sentiment_range()            # Score in [-1, 1]
├── test_ensemble.py
│   ├── test_ensemble_direction()         # Output is 0 or 1
│   ├── test_confidence_range()           # Output in [0, 100]
│   └── test_meta_learner()              # Meta-learner outperforms equal weight
├── test_model.py
│   ├── test_save_load_roundtrip()        # Save → load → same predictions
│   └── test_models_exist()              # File detection works
└── test_risk.py
    ├── test_var_range()                  # VaR in reasonable range
    ├── test_kelly_positive()             # Kelly > 0 when win_rate > 0.5
    └── test_sharpe_known_input()         # Known return series → known Sharpe
```

**Priority tests:**
1. `test_walk_forward_no_leakage()` — THE most important test
2. `test_computed_metrics()` — sanity check backtester math
3. `test_save_load_roundtrip()` — models survive serialization

---

### Tier 3: Nice to Have

#### 3.1 Better Feature Selection
**Impact:** Low-Medium (some features might be noise)
**Effort:** Low (1 day)

- Use XGBoost/LightGBM feature importance to drop low-value features
- Try mutual information or permutation importance
- Current: 41 features. Might work better with 25-30
- Track which features are consistently important vs noise

#### 3.2 Concept Drift Detection
**Impact:** Medium (models degrade over time)
**Effort:** Medium (2-3 days)

- Monitor rolling accuracy: if ensemble accuracy drops below 48% for N days, flag retrain
- Track feature distribution shifts (KS test on feature distributions)
- Auto-retrain trigger when drift detected

#### 3.3 Walk-Forward with Expanding Window
**Impact:** Low (current rolling window is fine, expanding is just different)
**Effort:** Low (modify `walk_forward_split`)

- Currently: 3y train / 1y test / 6mo step (rolling)
- Alternative: expanding window (train on everything from start to current)
- Both are valid, expanding uses more data

#### 3.4 Paper Trading Mode
**Impact:** Medium (validates signals without real money)
**Effort:** Medium (2-3 days)

- Run signals in shadow mode for 3 months
- Track: what would have been bought/sold, actual returns
- Compare predicted vs actual before committing real capital

#### 3.5 Docker + CI
**Impact:** Low (nice for reproducibility, not urgent for solo project)
**Effort:** Low (1 day)

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "streamlit", "run", "app.py"]
```

#### 3.6 Structured Logging
**Impact:** Low (Streamlit state is hard to debug)
**Effort:** Low (1 day)

- Replace print statements with Python `logging` module
- Log: training start/end, prediction confidence, backtest results
- Structured format (JSON) for machine parsing

---

## What NOT to Do

| Don't | Why |
|-------|-----|
| Add more model types (CNN, VAE, GAN) | You already have 5 models. More ≠ better. Stacking is better. |
| Try to predict price (regression) | Direction is hard enough. Price prediction adds noise. |
| Add real-money trading now | 45-50% accuracy isn't enough. Paper trade first. |
| Build a mobile app | Premature. Fix the models first. |
| Add crypto/forex | Different market microstructure. Master NSE first. |
| Buy expensive data feeds | yfinance is fine for research. Paid data only matters for live trading. |
| Copy Jane Street/Goldman infrastructure | Different business entirely. They do market-making, not direction prediction. |

---

## Realistic Expectations

### What 51-53% Accuracy Actually Means

| Timeframe | Trades | Expected Profit | Std Dev | Probability of Loss |
|-----------|--------|-----------------|---------|---------------------|
| 1 month | 20 | +1.0% | ±11% | ~46% |
| 6 months | 120 | +6.0% | ±24% | ~40% |
| 1 year | 250 | +12.5% | ±35% | ~36% |
| 3 years | 750 | +37.5% | ±60% | ~26% |

**With 52% accuracy** (realistic best case after improvements):

| Timeframe | Trades | Expected Profit | Std Dev | Probability of Loss |
|-----------|--------|-----------------|---------|---------------------|
| 1 month | 20 | +2.0% | ±11% | ~42% |
| 6 months | 120 | +12.0% | ±24% | ~31% |
| 1 year | 250 | +25.0% | ±35% | ~23% |
| 3 years | 750 | +75.0% | ±60% | ~15% |

**Key insight:** Edge compounds, but variance is real. You WILL have losing months. The question is whether you have the discipline to keep trading when the edge is working but variance is against you.

### What "Beat Jane Street" Actually Requires
- $50M+ infrastructure budget
- Colocated servers at exchange
- Proprietary order-flow data (not public)
- FPGA/kernel-bypass networking
- Team of 50+ PhDs
- Leverage (borrowed money) at institutional rates

**This is not a gap you close by adding more features or better models.** It's a different business.

### What You Can Actually Achieve
- 51-53% directional accuracy with disciplined risk management
- 15-25% annualized returns (not guaranteed, with significant drawdowns)
- A system that's genuinely useful for personal research and learning
- A portfolio that outperforms naive buy-and-hold *sometimes* (not always)

---

## Recommended Execution Order

### Phase 1: Validate (1-2 weeks)
1. Add statistical significance testing (`src/significance.py`)
2. Replace equal-weight ensemble with stacked meta-learner (`src/ensemble.py`)
3. Add regime-conditional model routing (`src/ensemble.py`)
4. Train all 20 stocks (batch script)
5. Add 3 critical tests (backtester no-leakage, metrics, save/load)

**Exit criteria:** You know if your signal is statistically significant. If it is, proceed. If not, the problem is fundamental, not engineering.

### Phase 2: Harden (2-3 weeks)
1. Move to DuckDB for data storage
2. Add feature versioning
3. Add comprehensive test suite
4. Reframe toward volatility forecasting or cross-sectional ranking
5. Add concept drift detection

**Exit criteria:** The system is reproducible, testable, and you know its limitations.

### Phase 3: Scale (1-2 months)
1. Expand to NIFTY 50/100 universe
2. Add paper trading mode
3. Docker + CI
4. Structured logging
5. Better feature selection

**Exit criteria:** The system can handle a real universe and you have 3 months of paper trading results.

---

## Decision Point

After Phase 1, you'll know:
- **Is the signal real?** (statistical significance testing)
- **Is the edge meaningful?** (stacked ensemble accuracy vs baseline)
- **Does regime routing help?** (per-regime accuracy comparison)

If the answer to all three is "yes" — proceed to Phase 2.
If the answer to any is "no" — the problem is not engineering. It's the fundamental difficulty of the task. At that point, consider:
- Shifting to volatility forecasting (more exploitable)
- Shifting to cross-sectional ranking (relative strength)
- Accepting that 51% is actually fine with proper risk management
- Using the system for research/learning, not live trading

---

## Files to Modify (Quick Reference)

| Improvement | Files |
|-------------|-------|
| Stacked meta-learner | `src/ensemble.py`, `src/model.py` |
| Statistical significance | New `src/significance.py`, `src/backtester.py` |
| Volatility forecasting | `src/model.py`, `src/trainer.py`, `src/ensemble.py`, `src/features.py` |
| Regime routing | `src/ensemble.py`, `src/regime.py` |
| DuckDB storage | `src/data_fetcher.py`, New `src/db.py` |
| Feature versioning | `src/features.py`, New `src/feature_store.py` |
| Tests | New `tests/` directory |
| Concept drift | New `src/drift.py`, `src/trainer.py` |
| Paper trading | `src/portfolio.py`, `app.py` |
