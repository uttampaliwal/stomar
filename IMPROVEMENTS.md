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
- No point-in-time data discipline (silent look-ahead bias risk)
- No statistical significance testing
- No regime-conditional model routing
- Equal-weight ensemble (not learned)
- File-based caching (no guarantees)
- Zero tests
- 3 stocks ≠ a trading system

---

## Important: What This Project Is NOT

### Remove "18-35% Guaranteed" From Any Framing

No fund — public or private — "guarantees" 18-35% returns at any time. Even Renaissance Technologies' Medallion fund (~60%+ gross historically) is closed to outsiders, leverages enormously, and still has losing months. SEBI explicitly bars Indian advisors from promising fixed/guaranteed returns because no legitimate strategy can deliver this reliably. A tool that promised this would either be wrong or breaking the law.

### What "Beat Jane Street/Goldman" Actually Requires
- $50M+ infrastructure budget
- Colocated servers at exchange
- Proprietary order-flow data (not public)
- FPGA/kernel-bypass networking
- Team of 50+ PhDs
- Leverage at institutional rates

**This is not a gap you close by adding more features or better models.** Those firms don't run directional next-day stock prediction. Their PnL comes from market-making and statistical arbitrage at microsecond latency — a different business entirely.

### What You CAN Actually Achieve
- 51-53% directional accuracy with disciplined risk management
- 15-25% annualized returns (not guaranteed, with significant drawdowns)
- A statistically validated small edge (2-5 percentage points over coin flip), sized correctly with Kelly, compounding over years with controlled drawdowns
- A system that's genuinely useful for personal research and learning

That's the real version of "best in class open source," and it's an achievable one.

---

## Phase 1 — Foundations (Do This Before Touching Models)

This is unglamorous but it's what every credible project gets right first.

### 1.1 Point-in-Time Data Discipline
**Impact:** Critical (the single most common reason backtests lie)
**Effort:** Medium (3-5 days)
**Files:** `src/features.py`, `src/flow.py`, `src/data_fetcher.py`

**The problem:** Point-in-time data reflects only what was actually known at each historical timestamp, eliminating look-ahead bias in backtests. Without that discipline, strategies can appear profitable in backtests by using information that would not have been available at the time of the trade.

**Audit these specific leak points:**

| File | Risk | What to Check |
|------|------|---------------|
| `features.py:9` (`add_sentiment_features`) | Sentiment score from today's news joined to today's price row | Sentiment should be computed from headlines published BEFORE market close |
| `features.py:27` (`add_flow_features`) | FII/DII data published at end-of-day joined to same day's price | FII/DII data for day T should only be available on day T+1 |
| `features.py:49` (`add_pcr_features`) | Options PCR from market hours joined to intraday price | PCR computed during market hours, used with end-of-day close |
| `features.py:70` (`add_multitimeframe_features`) | MTF signal computed using data that includes today's close | MTF data fetch should lag by at least 1 day |
| `data_fetcher.py:17` (`fetch_stock_data`) | Parquet cache stale but still used | Cache invalidation strategy needed |

**Concrete fix:**
```python
# BAD: FII data for day T joined to day T's price
df["fii_net"] = fii_data["fii_net"]  # This is look-ahead bias

# GOOD: FII data for day T joined to day T+1's price
df["fii_net"] = fii_data["fii_net"].shift(1)  # Lag by 1 day
```

### 1.2 Move to Proper Time-Series Store
**Impact:** Medium (prevents silent bugs, scales)
**Effort:** Medium (3-5 days)
**Files:** `src/data_fetcher.py`, new `src/db.py`

**Current problem:** `@st.cache_data` + flat JSON/parquet files have no point-in-time guarantees. If you fetch data on Monday and a Tuesday feature uses Monday's close, but the feature was computed on Wednesday using Tuesday's close — that's look-ahead bias, and you won't see it.

**Recommendation: DuckDB**
| Option | Pros | Cons |
|--------|------|------|
| **DuckDB** | Zero-config, SQL, fast, file-based | New dependency |
| **SQLite + partitioning** | Already in Python stdlib | Slower for large datasets |
| **DVC + parquet** | Version control for data | More complex setup |

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

### 1.3 Tests Around Backtester No-Leakage Guarantee
**Impact:** Critical (the walk-forward claim is the project's foundation)
**Effort:** Medium (3-5 days)
**Files:** New `tests/` directory

The no-leakage guarantee is the ONE claim the whole project's credibility rests on. If this is wrong, everything else is wrong.

**Priority tests:**
```
tests/
├── test_backtester.py      # Most critical
│   ├── test_walk_forward_no_leakage()    # Verify no future data in training
│   ├── test_train_test_no_overlap()      # Train and test sets are disjoint
│   ├── test_feature_lag()               # Features use only past data
│   ├── test_computed_metrics()           # Known inputs → known outputs
│   └── test_portfolio_value()            # Buy/sell → correct cash/holdings
├── test_features.py
│   ├── test_feature_count()              # Always 41 features
│   ├── test_no_nans_in_features()        # NaN handling works
│   ├── test_sentiment_range()            # Score in [-1, 1]
│   └── test_point_in_time()              # No future data in features
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

### 1.4 Docker + CI
**Impact:** Low (nice for reproducibility, but do after the above)
**Effort:** Low (1 day)

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "streamlit", "run", "app.py"]
```

### 1.5 Reference: QuantConnect LEAN
Worth skimming (not copying) for design ideas:
- **Automated accounting** for splits, dividends, corporate actions (stomar doesn't handle these)
- **Algorithmically selected asset universes** to avoid selection bias
- **Point-in-time data** is baked into their architecture
- You don't need their infrastructure, just steal the checklist

### Phase 1 Exit Criteria
- [ ] All leakage points in `features.py` and `flow.py` identified and fixed
- [ ] DuckDB (or equivalent) stores data with point-in-time guarantees
- [ ] `test_walk_forward_no_leakage()` passes
- [ ] `test_point_in_time()` passes for all feature sources
- [ ] Docker builds and runs successfully

---

## Phase 2 — Statistical Rigor on the Backtest

Before adding any more models, find out if the 45-50% you already have is signal or noise.

### 2.1 Combinatorial Purged Cross-Validation (CPCV)
**Impact:** Critical (catches false discoveries in backtests)
**Effort:** Medium (3-5 days)
**Files:** New `src/significance.py`, update `src/backtester.py`

**Why:** Walk-forward is good, but CPCV gives you the distribution of performance, not just a point estimate. It shows probability of ruin.

**What to implement:**
- Generate all possible train/test combinations from the walk-forward windows
- For each combination, compute strategy performance
- Build a distribution of outcomes
- Calculate probability of losing money over N trades

**Reference:** Marcos López de Prado, *Advances in Financial Machine Learning*, Chapter 12

### 2.2 Deflated Sharpe Ratio
**Impact:** Critical (adjusts for multiple testing bias)
**Effort:** Low (1-2 days, once CPCV exists)
**Files:** `src/significance.py`

**The problem:** If you tested 100 strategies and found one with Sharpe 2.0, the deflated Sharpe might be 0.8 — noise. The more strategies you try, the more you need to deflate.

**What to implement:**
```python
def deflated_sharpe_ratio(sharpe_observed, n_trials, sharpe_max, T, skew, kurtosis):
    """
    Adjusts observed Sharpe for multiple testing bias.
    
    sharpe_observed: Your best strategy's Sharpe
    n_trials: How many strategy variations you tried
    sharpe_max: Maximum possible Sharpe (theoretical bound)
    T: Number of independent observations (trades)
    skew: Return distribution skewness
    kurtosis: Return distribution excess kurtosis
    """
    # Implementation from López de Prado
```

### 2.3 Permutation Test
**Impact:** High (simple, powerful null hypothesis test)
**Effort:** Low (1 day)
**Files:** `src/significance.py`

**What to do:**
1. Take your labeled data (UP/DOWN for each day)
2. Shuffle the labels randomly (destroying any signal)
3. Re-run your backtest on the shuffled labels
4. Repeat 1000 times → build null distribution
5. See where your real backtest falls in that distribution
6. If real performance isn't well outside the null distribution, the edge isn't there yet

**This is the single most honest test of whether your signal is real.**

### 2.4 Confidence Intervals on All Metrics
**Impact:** Medium (shows uncertainty)
**Effort:** Low (1 day)
**Files:** `src/backtester.py`

- Report confidence intervals on accuracy, Sharpe, annual return
- Use bootstrap resampling for CI estimation
- Never report point estimates without intervals

### Phase 2 Exit Criteria
- [ ] CPCV implemented and run on all 3 trained stocks
- [ ] Deflated Sharpe computed (accounting for how many variations were tried)
- [ ] Permutation test shows real performance is outside null distribution (p < 0.05)
- [ ] All metrics reported with confidence intervals
- [ ] **Decision: Is the signal real?** If yes → Phase 3. If no → pivot to volatility/ranking.

---

## Phase 3 — Modeling (Once Phase 2 Says There's Something Real)

### 3.1 Replace Equal-Weight Ensemble with Stacked Meta-Learner
**Impact:** Medium (typically +2-5% accuracy)
**Effort:** Low (1-2 days)
**File:** `src/ensemble.py`

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
- This is the single easiest win in the modeling phase

### 3.2 Regime-Conditional Model Routing
**Impact:** Medium-High (your regime detection already exists but isn't used)
**Effort:** Low (1-2 days)
**Files:** `src/ensemble.py`, `src/regime.py`

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

### 3.3 Reframe: Direction → Volatility/Ranking
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

### 3.4 Expand Trained Universe
**Impact:** Medium (more stocks = more opportunities, better regime detection)
**Effort:** Low (automated)

**Current:** 3 stocks trained out of 20 in `NSE_STOCKS`

**What to do:**
- Batch train all 20 stocks (script it, ~15 minutes total)
- Add sector classification to each stock
- Track accuracy per sector — some sectors are more predictable than others
- Consider expanding NSE_STOCKS to NIFTY 50 or NIFTY 100

### 3.5 Reference: Microsoft Qlib
Worth reading for ideas, particularly:
- **Alpha factor library** — how they define and combine factors
- **Model zoo** — what models work for equity prediction
- **Rolling retraining** — more rigorous than static equal-weight ensemble
- **Multi-factor combination** — how they handle many signals

GitHub: microsoft/qlib (39.3k stars, active as of 2026)

### 3.6 Research Direction: Multi-Agent LLM Debate Systems
**Impact:** Research only (not a return-generating layer yet)
**Effort:** N/A (exploration)

A 2025-26 trend worth knowing about: setups where multiple AI agents (analysts, debaters, risk manager) debate stock picks in real-time. Interesting for combining sentiment/flow/technical signals into a single reasoned view rather than just weighted-averaging probabilities.

**Treat as research direction, not production.** The value is in structured reasoning, not in generating trading signals.

### Phase 3 Exit Criteria
- [ ] Stacked meta-learner implemented and outperforms equal-weight
- [ ] Regime-conditional routing implemented and backtested per-regime
- [ ] Volatility forecasting or ranking prototype built (if Phase 2 suggested direction prediction isn't viable)
- [ ] All 20 stocks in NSE_STOCKS trained
- [ ] Qlib alpha factor library reviewed for ideas

---

## Phase 4 — Portfolio / Mutual Fund Framework (Factual)

Once the above is solid, this is purely informational (not advice).

### 4.1 True XIRR on Zerodha Holdings
**Files:** `src/holdings.py`

Your `holdings.py` already half-does this. What's missing:
- Proper cash flow extraction from Zerodha CSV (buy dates, sell dates, dividend dates)
- XIRR Newton-Raphson solver with convergence checks
- Comparison against benchmark (Nifty 50, category average)

### 4.2 Direct vs Regular Plan Expense Ratio Drag
- Regular plans: 0.5-1.5% higher expense ratio than direct plans
- Over 10-20 years, this compounds to 15-30% less wealth
- Your holdings parser could flag which funds are regular vs direct

### 4.3 Equity/Debt/Gold Allocation Framework
- Based on stated risk tolerance and time horizon
- Not "what's the best allocation" but "what allocation matches YOUR situation"
- Your regime detection could inform tactical allocation shifts

### 4.4 Concentration Risk (Herfindahl Index)
Your holdings tracker already computes Herfindahl. What to explain:
- HHI > 0.25 = concentrated (risky)
- HHI < 0.10 = diversified
- Top-3 weight > 60% = top-heavy
- Category breakdown (Gold/Silver/Debt/Equity) shows true diversification

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
| Promise guaranteed returns | SEBI violation. No legitimate strategy can guarantee returns. |

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

---

## Files to Modify (Quick Reference)

| Improvement | Files |
|-------------|-------|
| Point-in-time fixes | `src/features.py`, `src/flow.py`, `src/data_fetcher.py` |
| DuckDB storage | `src/data_fetcher.py`, New `src/db.py` |
| CPCV + Deflated Sharpe | New `src/significance.py`, `src/backtester.py` |
| Permutation test | New `src/significance.py` |
| Stacked meta-learner | `src/ensemble.py`, `src/model.py` |
| Regime routing | `src/ensemble.py`, `src/regime.py` |
| Volatility forecasting | `src/model.py`, `src/trainer.py`, `src/ensemble.py`, `src/features.py` |
| Feature versioning | `src/features.py`, New `src/feature_store.py` |
| Tests | New `tests/` directory |
| Concept drift | New `src/drift.py`, `src/trainer.py` |
| Paper trading | `src/portfolio.py`, `app.py` |
| XIRR / holdings | `src/holdings.py` |
| Docker + CI | New `Dockerfile`, `.github/workflows/` |

---

## Recommended Execution Order (Summary)

| Phase | Focus | Duration | Key Question |
|-------|-------|----------|--------------|
| **1** | Foundations | 2-3 weeks | Is the data clean? Is the backtest honest? |
| **2** | Statistical rigor | 1-2 weeks | Is the signal real or noise? |
| **3** | Modeling | 2-4 weeks | Can we make the edge bigger and more robust? |
| **4** | Portfolio framework | 1-2 weeks | How to apply this to real holdings? |

**After Phase 2, you'll know:** Is the 45-50% signal statistically significant? If yes, proceed. If no, the problem is fundamental, not engineering — pivot to volatility forecasting or cross-sectional ranking.

The realistic, still genuinely valuable target: a statistically validated small edge (2-5 percentage points over a coin flip), sized correctly with Kelly, compounding over years with controlled drawdowns. That's the real version of "best in class open source," and it's an achievable one.
