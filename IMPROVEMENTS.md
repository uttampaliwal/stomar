# Improvement Roadmap

Honest assessment of current state, what matters most, and concrete steps to get there.

---

## Current State (Reality Check)

### What We Have

| Metric | Value | What It Means |
|--------|-------|---------------|
| Codebase | ~2.3k lines src/, ~1.2k app.py | 14 modules, clean modular architecture |
| Stocks trained | 3 of 20 | Not a universe — a sample |
| Directional accuracy | 45–50% | Indistinguishable from coin flip for 2/3 stocks |
| RELIANCE accuracy | 49.4% / 50.8% (WF) | Borderline — could be noise |
| Data source | yfinance (delayed) | Fine for research, not live |
| Backtesting | Walk-forward, no leakage | Good practice — this is the strongest part |
| Ensemble | Equal-weighted 5-model | Baseline — not optimized |
| Tests | 0 | No automated verification |
| CI/CD | None | No GitHub Actions, no linting |
| Live capability | None | Educational/research only |
| Broker integration | None | Kite Connect mentioned but not implemented |
| Model versioning | None | No registry, no promotion logic |
| Automated retraining | None | Manual "Train" click in UI |

### What's Actually Good
- Clean modular architecture (leaf-node design, minimal coupling)
- Walk-forward backtesting with brokerage/slippage (rare in hobby projects)
- Feature pipeline covers technical + alternative data (41 features)
- Alternative data integration (sentiment, FII/DII, PCR, multi-timeframe)
- Portfolio tooling (MVO, Black-Litterman, Efficient Frontier, Ledoit-Wolf)
- Risk metrics (VaR/CVaR, Kelly, drawdown, regime detection)
- The code is honest about its limitations (README disclaimers)

### What's Actually Missing

**Data/Infra:**
- No point-in-time data discipline (silent look-ahead bias risk)
- No data validation for gaps/splits/dividends
- No multi-source reconciliation
- File-based caching (no point-in-time guarantees)
- No feature store or feature versioning

**Modeling:**
- No statistical significance testing
- No regime-conditional model routing
- Equal-weight ensemble (not learned)
- No alpha decay monitoring
- No hypothesis-driven alpha research process

**Engineering:**
- Zero tests
- No CI/CD
- No structured logging
- No Docker
- No model versioning or registry

**Execution:**
- No order routing
- No realistic cost modeling (STT, stamp duty, exchange charges)
- No slippage/queue models
- No intraday risk limits
- No event-driven architecture

---

## Important: What This Project Is NOT

### Remove "18-35% Guaranteed" From Any Framing

No fund — public or private — "guarantees" 18-35% returns at any time. Even Renaissance Technologies' Medallion fund (~60%+ gross historically) is closed to outsiders, leverages enormously, and still has losing months. SEBI explicitly bars Indian advisors from promising fixed/guaranteed returns because no legitimate strategy can deliver this reliably. A tool that promised this would either be wrong or breaking the law.

Even top quant funds experience significant drawdowns; backtests almost always look better than live, and many strategies decay quickly as capacity fills or regimes shift.

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

## The Journey: 5 Stages

```
Current: Prototype
    ↓
Stage 1: Hardening (Tests, CI, Cost Models, Logging)
    ↓
Stage 2: Alpha & Backtest Rigor (Purged CV, Benchmarks, Feature Store)
    ↓
Stage 3: Live Data + Scheduling (Ingestion, Retraining, Monitoring)
    ↓
Stage 4: Execution & Risk (Event-driven, Limits, Paper Mode)
    ↓
Stage 5: Open Platform (Multi-asset, MF, Plugins, Community)
```

---

## Stage 1 — Hardening (Do This First)

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

### 1.2 Add Tests
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
├── test_risk.py
│   ├── test_var_range()                  # VaR in reasonable range
│   ├── test_kelly_positive()             # Kelly > 0 when win_rate > 0.5
│   └── test_sharpe_known_input()         # Known return series → known Sharpe
└── test_data_fetcher.py
    ├── test_fetch_returns_dataframe()    # yfinance → DataFrame
    ├── test_cache_hit()                  # Second fetch uses cache
    └── test_market_status()              # Returns "Open" or "Closed"
```

### 1.3 Add CI (GitHub Actions)
**Impact:** Medium (automated quality gate)
**Effort:** Low (1 day)
**Files:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v --tb=short
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - run: pip install ruff
      - run: ruff check src/ app.py
```

### 1.4 Realistic Cost Model for NSE
**Impact:** High (backtests lie without realistic costs)
**Effort:** Low (1-2 days)
**Files:** `src/backtester.py`, `src/portfolio.py`

**Current:** Simple fixed costs (0.03% brokerage, 0.1% slippage)

**Reality for Indian markets:**

| Cost Component | Rate | Notes |
|----------------|------|-------|
| Brokerage | 0.03% (delivery) | Zerodha flat fee model |
| STT | 0.1% (sell side) | Securities Transaction Tax |
| Exchange txn charges | ~0.00345% | NSE specific |
| SEBI turnover fees | 0.0001% | Regulatory |
| Stamp duty | 0.015% (buy side) | State-specific, recently revised |
| GST | 18% on brokerage + exchange charges | |
| **Total round-trip** | **~0.3-0.4%** | For a buy-then-sell |

**What to implement:**
```python
NSE_COSTS = {
    "brokerage_pct": 0.0003,      # 0.03% per side
    "stt_sell_pct": 0.001,         # 0.1% on sell
    "exchange_charge_pct": 0.0000345,
    "sebi_fees_pct": 0.000001,
    "stamp_duty_buy_pct": 0.00015,
    "gst_pct": 0.18,              # On brokerage + exchange charges
}
```

### 1.5 Structured Logging + Config
**Impact:** Low (Streamlit state is hard to debug)
**Effort:** Low (1 day)

- Replace print statements with Python `logging` module
- Log: training start/end, prediction confidence, backtest results
- Structured format (JSON) for machine parsing
- Configuration via YAML/.env with no secret leaks

### 1.6 Docker
**Impact:** Low (nice for reproducibility)
**Effort:** Low (1 day)

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.headless", "true"]
```

### Stage 1 Exit Criteria
- [ ] All leakage points in `features.py` and `flow.py` identified and fixed
- [ ] Test suite passes with >80% coverage on critical modules
- [ ] GitHub Actions CI runs on every push
- [ ] Realistic NSE cost model implemented
- [ ] Structured logging in place
- [ ] Docker builds and runs successfully

---

## Stage 2 — Alpha & Backtest Rigor

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

**Key detail:** Purged CV requires "purging" (removing) samples from the training set that overlap with test set labels, and "embargoing" (adding a gap) between train and test to prevent leakage through overlapping labels. This is what makes it more robust than standard walk-forward.

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

### 2.4 Benchmarks & Comparison
**Impact:** High (context for your performance)
**Effort:** Low (1 day)
**Files:** `src/backtester.py`

Your reported accuracies are ~45-51% framed as "even 51% edge is profitable," but there are no comparisons to simple benchmarks. Add:
- **Buy-and-hold** (passive Nifty 50)
- **20-day momentum** (simple cross-sectional)
- **Always-long** (baseline)
- **Random strategy** (coin flip with same trade frequency)

If your ensemble doesn't beat buy-and-hold on a risk-adjusted basis (Sharpe, Sortino, max DD), the edge isn't actionable.

### 2.5 Confidence Intervals on All Metrics
**Impact:** Medium (shows uncertainty)
**Effort:** Low (1 day)
**Files:** `src/backtester.py`

- Report confidence intervals on accuracy, Sharpe, annual return
- Use bootstrap resampling for CI estimation
- Never report point estimates without intervals

### 2.6 Hypothesis-Driven Alpha Research
**Impact:** High (structured process vs ad-hoc)
**Effort:** Medium (2-3 days)
**Files:** New `src/alpha_research.py`

**Current state:** No clear hypothesis → feature → signal → portfolio construction pipeline. No alpha decay monitoring.

**What to implement:**
For each signal (sentiment, FII/DII, PCR, technical), define:
1. **Hypothesis:** "FII net buying predicts next-day Nifty direction"
2. **Standalone test:** Run backtest with ONLY this signal
3. **Decay monitoring:** How quickly does the signal's predictive power fade?
4. **Combination:** How does it add value when combined with other signals?

This is what firms like Jane Street stress — rigorous, repeatable experimentation over ad-hoc model ensembling.

### Stage 2 Exit Criteria
- [ ] CPCV implemented and run on all 3 trained stocks
- [ ] Deflated Sharpe computed (accounting for how many variations were tried)
- [ ] Permutation test shows real performance is outside null distribution (p < 0.05)
- [ ] All metrics reported with confidence intervals
- [ ] Performance compared to buy-and-hold and momentum benchmarks
- [ ] Alpha research pipeline defined for each signal type
- [ ] **Decision: Is the signal real?** If yes → Stage 3. If no → pivot to volatility/ranking.

---

## Stage 3 — Live Data + Scheduling (But No Auto-Execution Yet)

### 3.1 Reliable Data Feeds
**Impact:** High (garbage in, garbage out)
**Effort:** Medium (3-5 days)
**Files:** `src/data_fetcher.py`

**Current problem:** yfinance for NSE is convenient but not a trusted, SLA'd market data feed. No validation for gaps/splits/dividends. No multi-source reconciliation.

**Options:**

| Source | Cost | Latency | Reliability |
|--------|------|---------|-------------|
| yfinance (current) | Free | Delayed 15min+ | Rate-limited, gaps |
| NSE website scraping | Free | End-of-day | ToS risk, breaks often |
| Kite Connect (Zerodha) | ₹500/month | Real-time | Official API, reliable |
| Truedata | ₹500-2000/month | Real-time | Professional grade |
| Angel One / Upstox APIs | Free tier available | Near-real-time | Good for retail |

**What to implement:**
- Data validation: check for missing dates, corporate actions (splits/dividends)
- Multi-source reconciliation: fetch from 2 sources, flag discrepancies
- Graceful fallback: if primary source fails, use secondary

### 3.2 Feature Store (Lightweight)
**Impact:** Medium (prevents training/serving skew)
**Effort:** Medium (2-3 days)
**Files:** `src/features.py`, new `src/feature_store.py`

**Why:** Feature stores are industry standard to keep training/serving consistent. Without one, you risk training on v1 features and serving on v3 features — silent failure.

**Lightweight approach:**
```python
import hashlib, json

def compute_feature_version(feature_cols, df_hash):
    content = json.dumps(sorted(feature_cols)) + df_hash
    return hashlib.sha256(content.encode()).hexdigest()[:12]

# Store feature version with model
# When loading model, verify feature version matches current features
```

### 3.3 Automated Retraining Pipeline
**Impact:** High (models go stale without retraining)
**Effort:** Medium (3-5 days)
**Files:** New `src/pipeline.py`

**Current:** Manual "Train" click in UI; models go stale until you retrain.

**Target:** Nightly or event-driven retraining with validation gates.

```
Schedule (cron/Airflow):
  ├── 1. Fetch latest data
  ├── 2. Validate data quality (no gaps, no stale prices)
  ├── 3. Compute features (point-in-time safe)
  ├── 4. Train models
  ├── 5. Validate on out-of-sample window
  ├── 6. If OOS performance > threshold → promote to production
  ├── 7. If OOS performance < threshold → alert, keep old model
  └── 8. Log everything
```

### 3.4 Monitoring & Drift Detection
**Impact:** Medium (catch model degradation early)
**Effort:** Medium (2-3 days)
**Files:** New `src/monitoring.py`

- Track OOS performance (hit rate, PnL, drawdowns) with alerting
- Detect drift in features (KS test on feature distributions)
- Detect drift in predictions (distribution shift in model outputs)
- Auto-retrain trigger when drift detected

### Stage 3 Exit Criteria
- [ ] Primary data feed with SLA (Kite Connect or equivalent)
- [ ] Data validation (gaps, splits, dividends checked)
- [ ] Feature store with versioning
- [ ] Nightly retraining pipeline running
- [ ] Monitoring dashboard with drift alerts
- [ ] Model registry with promotion logic

---

## Stage 4 — Execution & Risk

### 4.1 Event-Driven Architecture
**Impact:** High (backtest-live parity)
**Effort:** High (1-2 weeks)
**Files:** Major refactor of `src/backtester.py`, new `src/engine.py`

**Current:** Walk-forward backtester is batch-oriented. No event-driven architecture.

**Target:** Event-driven engine where the same code path runs backtest and live.

**Reference engines (don't build from scratch):**
- **NautilusTrader** — Rust-based, Python control plane, event-driven, backtest-to-live parity
- **QuantConnect LEAN** — Most mature open-source engine, automated accounting for splits/dividends

You don't need to build another backtester. Consider wrapping NautilusTrader or LEAN for research→live parity.

### 4.2 Paper Trading Mode
**Impact:** High (validates signals without real money)
**Effort:** Medium (3-5 days)
**Files:** `src/portfolio.py`, `app.py`

- Live data ingestion + order simulation
- Track: what would have been bought/sold, actual returns
- Compare predicted vs actual before committing real capital
- Run for 3+ months before real money

### 4.3 Risk Controls
**Impact:** High (protects capital)
**Effort:** Medium (2-3 days)
**Files:** `src/risk.py`

- Position limits (max % per stock)
- Loss limits (daily/weekly max drawdown)
- Intraday margin checks
- Integrate broker risk APIs where possible
- Kelly-capped position sizing (never full Kelly, use half or quarter Kelly)

### 4.4 Execution Quality
**Impact:** Medium (slippage matters)
**Effort:** Low (1-2 days)
**Files:** `src/backtester.py`

- Slippage/queue models (not just fixed 0.1%)
- Market impact estimation
- Order type simulation (limit, market, stop-loss)
- Fill probability modeling

### Stage 4 Exit Criteria
- [ ] Event-driven engine with backtest-live parity
- [ ] Paper trading mode running for 3+ months
- [ ] Risk controls (position limits, loss limits, Kelly caps)
- [ ] Realistic execution simulation (slippage, impact, fill probability)

---

## Stage 5 — Open Platform (Multi-Asset + MF + Community)

### 5.1 Modeling Improvements (Once Stage 2 Says Signal Is Real)

**Stacked Meta-Learner:**
```python
# Replace equal-weight with learned weights
from sklearn.linear_model import LogisticRegression
meta_X = np.column_stack([lstm_probs, gru_probs, transformer_probs, xgb_probs, lgb_probs])
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_train)
```

**Regime-Conditional Routing:**
```python
# Different ensemble weights per regime
if regime == "Bull":
    weights = {"lstm": 0.15, "gru": 0.15, "transformer": 0.2, "xgb": 0.25, "lgb": 0.25}
elif regime == "Bear":
    weights = {"lstm": 0.25, "gru": 0.25, "transformer": 0.15, "xgb": 0.15, "lgb": 0.2}
```

**Reframe: Direction → Volatility/Ranking:**

| Target | Edge Source | Data Needed |
|--------|------------|-------------|
| **Volatility forecasting** | Mean-reversion in vol, GARCH effects | Historical vol, options IV |
| **Cross-sectional ranking** | Relative strength, momentum | Universe of stocks |
| **Sector rotation** | Business cycle, macro | Sector ETFs, economic data |
| **Pairs trading** | Cointegration breakdowns | Correlated stock pairs |

### 5.2 Mutual Fund Integration
**Impact:** Medium (your MF support is limited today)
**Effort:** Medium (3-5 days)
**Files:** `src/holdings.py`, new `src/mf_tracker.py`

**Current:** Imports Zerodha CSV, analyzes category allocation and concentration.

**What's missing:**
- MF NAV history + factor exposures (value, momentum, quality)
- AMFI API integration for daily NAV feeds
- Direct vs regular plan expense ratio tracking
- XIRR computation (your `holdings.py` half-does this)
- Combined portfolio backtests (MF core + stock satellite)

**Practical approach:**
- Equity slice: Use ML ensemble as one input among many, not the sole decision-maker
- MF slice: Use Holdings Tracker to monitor concentrations; rebalance periodically
- Mix: MF part = diversified core; stock signals = small satellite (5-15% of capital) with strict risk limits

### 5.3 Sentiment Upgrades
**Impact:** Medium (better NLP)
**Effort:** Medium (2-3 days)
**Files:** `src/sentiment.py`

**Current:** FinBERT on Google News headlines with 30-min cache. Good baseline.

**Improvements:**
- **FinGPT** and similar financial LLMs — competitive with GPT-4 on some finance NLP tasks, open-source
- **Multi-source NLP stack:** news, transcripts, filings, social media
- **Fine-tuned domain models** for Indian market context
- **Finance-specific evaluation metrics** and dataset shifts across sectors

### 5.4 Expand Universe
**Impact:** Medium (more stocks = more opportunities)
**Effort:** Low (automated)

- Batch train all 20 stocks in NSE_STOCKS
- Expand to NIFTY 50 or NIFTY 100
- Add sector classification
- Track accuracy per sector

### 5.5 References Worth Studying

| Project | Stars | What to Learn |
|---------|-------|---------------|
| **Microsoft Qlib** | 39.3k | Alpha factor library, model zoo, rolling retraining, multi-factor combination |
| **QuantConnect LEAN** | 9k+ | Event-driven architecture, automated accounting, backtest-live parity |
| **NautilusTrader** | 3k+ | Rust-based performance, event-driven design, Python control plane |
| **VectorBT** | 4k+ | Vectorized backtesting, realistic cost modeling |

### 5.6 Research Direction: Multi-Agent LLM Debate Systems
**Impact:** Research only (not a return-generating layer yet)

A 2025-26 trend: multiple AI analysts, debaters, and a risk manager debate stock picks in real-time. Interesting for combining sentiment/flow/technical signals into a single reasoned view rather than weighted-averaging probabilities.

**Treat as research direction, not production.**

### Stage 5 Exit Criteria
- [ ] Stacked meta-learner + regime routing implemented
- [ ] Volatility forecasting or ranking prototype built
- [ ] MF NAV integration + XIRR computation
- [ ] Sentiment upgraded with FinGPT or multi-source NLP
- [ ] All 20 stocks trained
- [ ] Open-source license + contribution guidelines

---

## What NOT to Do

| Don't | Why |
|-------|-----|
| Add more model types (CNN, VAE, GAN) | You already have 5 models. More ≠ better. Stacking is better. |
| Try to predict price (regression) | Direction is hard enough. Price prediction adds noise. |
| Add real-money trading now | 45-50% accuracy isn't enough. Paper trade first. |
| Build a mobile app | Premature. Fix the models first. |
| Add crypto/forex | Different market microstructure. Master NSE first. |
| Buy expensive data feeds (yet) | yfinance is fine for research. Paid data only matters for live trading. |
| Copy Jane Street/Goldman infrastructure | Different business entirely. They do market-making, not direction prediction. |
| Promise guaranteed returns | SEBI violation. No legitimate strategy can guarantee returns. |
| Build another backtester | Use NautilusTrader or LEAN for event-driven backtest-live parity. |

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

### Realistic "Best-in-Class Open Source" Architecture

It's a stack, not one app:

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  Parquet + DuckDB + DVC (versioned, point-in-time)      │
│  Equities + MFs + Alt data (sentiment, flow, options)   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   RESEARCH LAYER                         │
│  Streamlit/Panel dashboards                              │
│  VectorBT/NautilusTrader backtests                       │
│  Strict OOS validation (purged CV, benchmarks)          │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    OPS LAYER                              │
│  MLOps pipelines (MLflow/ClearML + scheduling)          │
│  Training, validation, promotion, monitoring            │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                 EXECUTION LAYER                          │
│  Event-driven engine (NautilusTrader or LEAN)           │
│  Paper and live modes, broker adapters, risk checks     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                 PORTFOLIO LAYER                          │
│  MVO/BL for allocation                                   │
│  MF tracker + rebalancing logic                          │
│  VaR/CVaR, regime-aware sizing, Kelly caps               │
└─────────────────────────────────────────────────────────┘
```

Stomar can evolve into the **research/dashboarding layer** of this stack. It's not there yet, but the architecture is modular enough to grow into it.

---

## Files to Modify (Quick Reference)

| Improvement | Files |
|-------------|-------|
| Point-in-time fixes | `src/features.py`, `src/flow.py`, `src/data_fetcher.py` |
| Tests | New `tests/` directory |
| CI | New `.github/workflows/ci.yml` |
| Cost model | `src/backtester.py`, `src/portfolio.py` |
| Logging | All `src/*.py`, `app.py` |
| Docker | New `Dockerfile` |
| CPCV + Deflated Sharpe | New `src/significance.py`, `src/backtester.py` |
| Permutation test | New `src/significance.py` |
| Benchmarks | `src/backtester.py` |
| Alpha research | New `src/alpha_research.py` |
| Data validation | `src/data_fetcher.py` |
| Feature store | `src/features.py`, New `src/feature_store.py` |
| Retraining pipeline | New `src/pipeline.py` |
| Monitoring | New `src/monitoring.py` |
| Event-driven engine | New `src/engine.py`, `src/backtester.py` |
| Paper trading | `src/portfolio.py`, `app.py` |
| Risk controls | `src/risk.py` |
| Stacked meta-learner | `src/ensemble.py`, `src/model.py` |
| Regime routing | `src/ensemble.py`, `src/regime.py` |
| Volatility forecasting | `src/model.py`, `src/trainer.py`, `src/ensemble.py`, `src/features.py` |
| MF integration | `src/holdings.py`, new `src/mf_tracker.py` |
| Sentiment upgrade | `src/sentiment.py` |

---

## Recommended Execution Order (Summary)

| Stage | Focus | Duration | Key Question |
|-------|-------|----------|--------------|
| **1** | Hardening | 2-3 weeks | Is the codebase solid enough to trust? |
| **2** | Alpha rigor | 1-2 weeks | Is the signal real or noise? |
| **3** | Live data | 2-3 weeks | Can we get reliable data and retrain automatically? |
| **4** | Execution | 2-4 weeks | Can we execute with proper risk controls? |
| **5** | Open platform | Ongoing | Can others use and extend this? |

**After Stage 2, you'll know:** Is the 45-50% signal statistically significant? If yes, proceed. If no, the problem is fundamental, not engineering — pivot to volatility forecasting or cross-sectional ranking.

The realistic, still genuinely valuable target: a statistically validated small edge (2-5 percentage points over a coin flip), sized correctly with Kelly, compounding over years with controlled drawdowns. That's the real version of "best in class open source," and it's an achievable one.
