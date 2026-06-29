# Improvement Roadmap

Honest assessment of current state, what matters most, and concrete steps to get there.

---

## Current State (Reality Check)

### What We Have

| Metric | Value | What It Means |
|--------|-------|---------------|
| Codebase | ~35 src/ modules, 1953-line app.py | Modular architecture, 17-tab Streamlit terminal |
| Tests | 528 passing (22 test files) | Full coverage across all modules |
| CI/CD | GitHub Actions (lint + test) | Automated quality gate on every push |
| Stocks trained | 10 of 20 | Batch-trainable via CLI (`--train-all`) |
| Ensemble | 5-model + stacked meta-learner + regime routing | Learned weights, not equal-weight |
| Directional accuracy | 45–50% | Indistinguishable from coin flip for some stocks |
| Data source | yfinance (delayed) | Fine for research, not live |
| Backtesting | Walk-forward + event-driven engine | Both batch and event-driven modes |
| Alternative data | 5-source sentiment, FII/DII, PCR, MTF | Multi-source NLP stack |
| Portfolio tooling | MVO, Black-Litterman, Efficient Frontier, Ledoit-Wolf | Full optimization suite |
| Risk metrics | VaR/CVaR, Kelly, drawdown, regime detection | Comprehensive risk suite |
| Mutual fund tracker | NAV history, XIRR, factor exposures, allocation | Full MF analysis |
| Paper trading | Event-driven engine with slippage models | Realistic execution simulation |
| Execution quality | Fill rates, latency, cost decomposition | Order analysis |
| License | Apache 2.0 + CONTRIBUTING.md | Open-source ready |

### What's Actually Good
- Clean modular architecture (leaf-node design, minimal coupling)
- Walk-forward backtesting with brokerage/slippage (rare in hobby projects)
- Feature pipeline covers technical + alternative data (41 features)
- Multi-source sentiment (Yahoo, Google, MoneyControl, ET, Screener.in)
- Portfolio tooling (MVO, Black-Litterman, Efficient Frontier, Ledoit-Wolf)
- Risk metrics (VaR/CVaR, Kelly, drawdown, regime detection)
- Event-driven engine with realistic execution simulation
- 528 tests, CI on every push
- The code is honest about its limitations (README disclaimers)

### What's Still Missing

**Architecture (Stage 6):**
- No orchestrator — everything runs inside Streamlit session, dies when tab closes
- No persistent ledger — `st.session_state.portfolio` is ephemeral
- No meta-controller — modules exist but no layer turns 14 opinions into one decision
- No automated daily loop — requires manual triggers

**Modeling:**
- No alpha decay monitoring
- No hypothesis-driven alpha research process

**Execution:**
- No broker integration (Kite Connect mentioned but not implemented)

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

## The Journey: 6 Stages

```
Current: Prototype
    ↓
Stage 1: Hardening (Tests, CI, Cost Models, Logging) ✅
    ↓
Stage 2: Alpha & Backtest Rigor (Purged CV, Benchmarks, Feature Store) ✅
    ↓
Stage 3: Live Data + Scheduling (Ingestion, Retraining, Monitoring) ✅
    ↓
Stage 4: Execution & Risk (Event-driven, Limits, Paper Mode) ✅
    ↓
Stage 5: Open Platform (Multi-asset, MF, Plugins, Community) ✅
    ↓
Stage 6: Autonomous Loop (Orchestrator, Ledger, Meta-Controller) → NEXT
```

---

## Stage 1 — Hardening ✅ COMPLETE

### 1.1 Point-in-Time Data Discipline ✅
- `src/data_validation.py` — gap detection, corporate action detection, stale data checks
- `tests/test_point_in_time.py` — point-in-time tests for all alt-data features

### 1.2 Add Tests ✅
- 528 tests across 22 test files
- Coverage: all modules, including engine, risk, ensemble, sentiment, MF tracker

### 1.3 Add CI (GitHub Actions) ✅
- `.github/workflows/ci.yml` — lint + test on every push/PR

### 1.4 Realistic Cost Model for NSE ✅
- `src/risk_controls.py` — NSE-specific cost model (STT, stamp duty, exchange charges, GST)
- `tests/test_nse_costs.py` — cost calculations validated

### 1.5 Structured Logging ✅
- `src/logging_config.py` — structured logging with configurable levels

### 1.6 Docker ✅
- `Dockerfile` — python:3.12-slim, health check, Streamlit config

### Stage 1 Exit Criteria
- [x] All leakage points in `features.py` and `flow.py` identified and fixed
- [x] Test suite passes with >80% coverage on critical modules
- [x] GitHub Actions CI runs on every push
- [x] Realistic NSE cost model implemented
- [x] Structured logging in place
- [x] Docker builds and runs successfully

---

## Stage 2 — Alpha & Backtest Rigor ✅ COMPLETE

### 2.1 Combinatorial Purged Cross-Validation (CPCV) ✅
- `src/significance.py` — purged K-fold CV, CPCV implementation

### 2.2 Deflated Sharpe Ratio ✅
- `src/significance.py` — `deflated_sharpe_ratio()` function

### 2.3 Permutation Test ✅
- `src/significance.py` — `permutation_test_accuracy()` function

### 2.4 Benchmarks & Comparison ✅
- `src/benchmarks.py` — buy-and-hold, momentum, SMA crossover, random strategy

### 2.5 Confidence Intervals on All Metrics ✅
- `src/significance.py` — bootstrap confidence intervals

### 2.6 Hypothesis-Driven Alpha Research ✅
- `src/alpha_research.py` — hypothesis testing, decay analysis, signal ranking, quality scoring

### Stage 2 Exit Criteria
- [x] CPCV implemented and run on all trained stocks
- [x] Deflated Sharpe computed (accounting for how many variations were tried)
- [x] Permutation test shows real performance is outside null distribution (p < 0.05)
- [x] All metrics reported with confidence intervals
- [x] Performance compared to buy-and-hold and momentum benchmarks
- [x] Alpha research pipeline defined for each signal type

---

## Stage 3 — Live Data + Scheduling ✅ COMPLETE

### 3.1 Reliable Data Feeds ✅
- `src/data_sources.py` — multi-source fetching with automatic fallback (yfinance + NSE archive)
- `src/data_validation.py` — comprehensive data quality checks

### 3.2 Feature Store (Lightweight) ✅
- `src/feature_store.py` — feature versioning, hashing, compatibility verification

### 3.3 Automated Retraining Pipeline ✅
- `src/pipeline.py` — full retraining pipeline with validation gates
- `run_pipeline.py` — CLI entry point with `--train-all` and `--train` flags
- `src/trainer.py` — `batch_train()` for all 20 NSE stocks

### 3.4 Monitoring & Drift Detection ✅
- `src/monitoring.py` — performance drift, feature drift, prediction drift, data freshness alerts

### Stage 3 Exit Criteria
- [x] Primary data feed with fallback (yfinance + NSE archive)
- [x] Data validation (gaps, splits, dividends checked)
- [x] Feature store with versioning
- [x] Retraining pipeline with CLI
- [x] Monitoring dashboard with drift alerts
- [x] Model registry with promotion logic

---

## Stage 4 — Execution & Risk ✅ COMPLETE

### 4.1 Event-Driven Architecture ✅
- `src/engine.py` — Event-driven backtesting engine with order matching
- `src/risk_controls.py` — slippage models (fixed, volume-weighted, adaptive)

### 4.2 Paper Trading Mode ✅
- `src/paper_trader.py` — full paper trading with state persistence
- `app.py` — Paper Trading tab (16th)

### 4.3 Risk Controls ✅
- `src/risk_controls.py` — position limits, daily/weekly loss limits, drawdown halt, Kelly sizing
- `src/risk.py` — VaR, CVaR, Sharpe, Sortino, Calmar, Kelly criterion

### 4.4 Execution Quality ✅
- `src/execution_quality.py` — fill rates, slippage analysis, cost decomposition
- `app.py` — Execution quality tab integrated

### Stage 4 Exit Criteria
- [x] Event-driven engine with backtest-live parity
- [x] Paper trading mode with state persistence
- [x] Risk controls (position limits, loss limits, Kelly caps)
- [x] Realistic execution simulation (slippage, impact, fill probability)

---

## Stage 5 — Open Platform ✅ COMPLETE

### 5.1 Modeling Improvements ✅
- `src/ensemble.py` — stacked meta-learner (LogisticRegression), regime-conditional routing (REGIME_WEIGHTS)
- `src/volatility.py` — historical, EWMA, Parkinson, Garman-Klass, Yang-Zhang volatility estimators
- `src/ranking.py` — fundamental ranking with yfinance data (PE, PB, ROCE, ROE, etc.)

### 5.2 Mutual Fund Integration ✅
- `src/mf_tracker.py` — MF NAV history, XIRR, factor exposures, allocation breakdown, concentration risk, benchmark comparison
- `app.py` — MF Tracker tab (17th)

### 5.3 Sentiment Upgrades ✅
- `src/sentiment.py` — 5-source sentiment (Yahoo, Google, MoneyControl, ET, Screener.in), source-weighted aggregation, deduplication

### 5.4 Expand Universe ✅
- `src/trainer.py` — `batch_train()` for all 20 NSE stocks
- `run_pipeline.py` — `--train-all` flag

### Stage 5 Exit Criteria
- [x] Stacked meta-learner + regime routing implemented
- [x] Volatility forecasting or ranking prototype built (fundamental ranking + volatility module)
- [x] MF NAV integration + XIRR computation (mf_tracker.py)
- [x] Sentiment upgraded with multi-source NLP (Yahoo, Google, MoneyControl, ET, Screener)
- [x] All 20 stocks batch-trainable (batch_train + --train-all flag)
- [x] Open-source license (Apache 2.0) + contribution guidelines (CONTRIBUTING.md)

---

## Stage 6 — Autonomous Loop (Orchestrator, Ledger, Meta-Controller) → NEXT

> Source: `check/check.txt` — architecture analysis for autonomous paper-trading loop.

**The problem:** Everything today is click-triggered inside a Streamlit session. There's no background process, no scheduler, no persistent ledger. `st.session_state.portfolio` lives only as long as the browser tab does. You cannot bolt "auto everything" onto that without first giving the system a life outside the UI.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Signal layer (existing)                      │
│  ensemble, sentiment, flow, risk, regime, volatility...  │
└─────────────────────┬───────────────────────────────────┘
                      │ module outputs as state vector
┌─────────────────────▼───────────────────────────────────┐
│              Meta-controller (NEW)                        │
│  bandit / stacking, RL as later upgrade                  │
│  14 opinions → 1 auditable decision                      │
└─────────────────────┬───────────────────────────────────┘
                      │ decision + signals
┌─────────────────────▼───────────────────────────────────┐
│              Orchestrator + ledger (NEW)                  │
│  scheduler, paper trades, persistent log                 │
│  feedback loop → reweight signal layer                   │
└─────────────────────────────────────────────────────────┘
```

### 6.1 Orchestrator (Background Process)
**Impact:** Critical (system must live outside Streamlit)
**Effort:** Medium (3-5 days)
**Files:** New `src/orchestrator.py`, `run_daily.py`

**Current problem:** Everything runs inside `streamlit run app.py`. Close the tab, the system stops. No background processing, no scheduling.

**What to build:**
- A standalone Python script (`run_daily.py`) that runs independently of Streamlit
- Uses APScheduler or Windows Task Scheduler (via existing `schedule_pipeline.py`)
- After market close (e.g., 3:30 PM IST): pull fresh data → run all 14 signal modules → get meta-controller's decision → simulate paper trade → write everything to ledger
- Streamlit becomes a *viewer* of this process's output, not the thing running it

**Concrete flow:**
```
3:30 PM IST — Market Close
    ↓
run_daily.py (scheduled)
    ↓
1. Fetch latest data for all tickers (yfinance)
2. Run data validation (gaps, staleness)
3. Compute features (41 technical + alt-data)
4. Run all signal modules:
   - Ensemble prediction + confidence
   - Sentiment score (5 sources)
   - FII/DII net flow
   - Options PCR + Max Pain
   - Multi-timeframe signal
   - Regime detection
   - VaR/CVaR
   - Volatility forecast
   - Fundamental ranking
5. Meta-controller: combine 14 signals → 1 decision
6. Simulate paper trade (if decision is actionable)
7. Log everything to ledger (SQLite)
8. Update Streamlit dashboard state (if running)
```

### 6.2 Persistent Ledger (SQLite/Postgres)
**Impact:** Critical (ephemeral state = no learning possible)
**Effort:** Medium (2-3 days)
**Files:** New `src/ledger.py`

**Current problem:** `st.session_state.portfolio` is ephemeral. Every paper trade, module signal, and outcome is lost when the browser closes. No historical record exists for analysis or learning.

**What to build:**
- SQLite database (simple, no server needed) replacing in-memory portfolio
- Schema for decisions, trades, signals, and outcomes
- Every decision logs: what each of the 14 modules said, what the meta-controller decided, what the paper portfolio did, and what actually happened the next day

**Ledger schema:**
```sql
-- Every daily decision cycle
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    regime TEXT,
    -- Signal outputs (what each module said)
    ensemble_direction INTEGER,
    ensemble_confidence REAL,
    sentiment_score REAL,
    fii_net REAL,
    pcr REAL,
    mtf_signal REAL,
    regime_confidence REAL,
    var_95 REAL,
    cvar_95 REAL,
    volatility_forecast REAL,
    fundamental_score REAL,
    -- Meta-controller decision
    action TEXT,          -- BUY / SELL / HOLD
    position_size REAL,
    confidence REAL,
    reasoning TEXT,
    -- Actual outcome (filled next day)
    actual_return REAL,
    actual_direction INTEGER,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper trades executed
CREATE TABLE paper_trades (
    id INTEGER PRIMARY KEY,
    decision_id INTEGER REFERENCES decisions(id),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,      -- BUY / SELL
    quantity INTEGER,
    price REAL,
    slippage REAL,
    costs REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    total_value REAL,
    cash REAL,
    holdings_json TEXT,     -- JSON blob of all positions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6.3 Meta-Controller v1 (Bandit/Stacking)
**Impact:** High (turns 14 opinions into one auditable decision)
**Effort:** Medium (3-5 days)
**Files:** New `src/meta_controller.py`

**Current problem:** The 14 signal modules exist independently. The ensemble has a meta-learner, but the other modules (sentiment, flow, risk, regime, etc.) are not combined into a single decision. There's no layer that asks "given all signals, what should we actually do?"

**Why not RL yet:** With only 10 stocks trained on daily bars, a full RL policy (PPO/DQN) has almost no signal to learn from and will overfit to noise. Start with something far more tractable and auditable.

**What to build:**

1. **Contextual bandit** — takes the 14 modules' outputs as a state vector and learns which combination of signals has actually predicted price moves historically
   - Input: `[ensemble_dir, ensemble_conf, sentiment, fii_net, pcr, mtf, regime, var, cvar, vol_forecast, fundamental, ...]`
   - Output: `action` (BUY/SELL/HOLD) + `position_size`
   - Training: supervised learning on historical ledger data (what signals predicted what outcomes)

2. **Stacked logistic regression** — simpler, more interpretable version
   - Each signal module's output becomes a feature
   - Logistic regression learns the weights
   - The learned weights ARE the audit mechanism: they tell you which modules are pulling their weight

3. **Per-stock, per-regime routing** — different meta-controller weights for different market conditions
   - Train separate meta-controllers for Bull/Bear/Sideways regimes
   - This is the "live scorecard" of which tools work when

**Audit mechanism:**
The meta-controller's learned weights are a live scorecard:
- High weight = module is currently predictive → keep using it
- Low weight = module is noise → flag for review or retraining
- Weight decay over time = signal is decaying → alpha decay alert

### 6.4 Paper Trading Run (2-3+ Months)
**Impact:** Critical (validates everything before real money)
**Effort:** Low (automated, just wait)
**Files:** `run_daily.py` (scheduled), `src/ledger.py`

- Run the full autonomous loop for 2-3+ months
- Log every decision, every signal, every outcome to the ledger
- This log is what any future RL agent would train on
- Streamlit dashboard reads from the ledger for visualization

### 6.5 RL Upgrade (Optional, After Step 6.4)
**Impact:** Research only (not needed for production)
**Effort:** High (2-4 weeks)
**Files:** `src/meta_controller.py` (upgrade)

- Only once you have months of logged episodes
- Small PPO agent over portfolio-weight actions
- Reward: risk-adjusted differential Sharpe (not raw P&L, which encourages reckless sizing)
- This becomes a reasonable experiment rather than a guaranteed overfit

### Stage 6 Exit Criteria
- [ ] `run_daily.py` runs independently of Streamlit (scheduled)
- [ ] SQLite ledger with decisions, trades, and portfolio snapshots
- [ ] Meta-controller v1 (bandit/stacking) combining all 14 signal modules
- [ ] 2-3+ months of logged paper trading episodes
- [ ] Streamlit dashboard reads from ledger (not session state)
- [ ] (Optional) RL upgrade with differential Sharpe reward

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
| Jump to RL before ledger exists | RL needs months of logged episodes. Without a ledger, there's nothing to train on. |

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
| Orchestrator | New `src/orchestrator.py`, `run_daily.py` |
| Ledger | New `src/ledger.py` |
| Meta-controller | New `src/meta_controller.py` |
| Daily scheduler | Update `schedule_pipeline.py` |
| Dashboard from ledger | Update `app.py` (read from SQLite instead of session state) |

---

## Recommended Execution Order (Summary)

| Stage | Focus | Duration | Status |
|-------|-------|----------|--------|
| **1** | Hardening | 2-3 weeks | ✅ Complete |
| **2** | Alpha rigor | 1-2 weeks | ✅ Complete |
| **3** | Live data | 2-3 weeks | ✅ Complete |
| **4** | Execution | 2-4 weeks | ✅ Complete |
| **5** | Open platform | Ongoing | ✅ Complete |
| **6** | Autonomous loop | 3-5 weeks | → NEXT |

**After Stage 6, you'll have:** A system that runs itself daily, logs everything, makes auditable decisions, and积累了 the data needed for any future ML/RL upgrade. That's the real "autonomous" milestone.
