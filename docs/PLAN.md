# Stage 6 — Autonomous Loop: Execution Plan

## Status: ✅ STAGE 6 COMPLETE + STAGE 7 IMPROVEMENTS APPLIED

**Current state:** 679 tests passing, 0 lint errors, all critical bugs fixed, interpretability module added, triple-barrier labels implemented, risk controls enhanced.

Stage 6 answers the question: **Can the system run itself — pulling data, making decisions, logging outcomes — without a human clicking buttons in the UI?**

This stage builds the orchestrator, persistent ledger, and meta-controller that turn 14 independent signal modules into one autonomous daily loop.

**Stage 7 improvements applied:**
- Fixed 20+ bugs (critical, high, medium severity)
- Added model interpretability (feature importance + SHAP)
- Added triple-barrier labels for superior ML targets
- Added pre-trade risk controls (kill switch, Kelly sizing)
- Enhanced monitoring with data freshness checks
- Fixed time-series data leakage (shuffle=False)
- Fixed P&L double-counting in portfolio
- Fixed confidence scale inconsistencies
- Cleaned all lint errors across src/, api/, tests/

---

## Stage 6 Exit Criteria (from IMPROVEMENTS.md)

| # | Criterion | Status | Priority |
|---|-----------|--------|----------|
| 1 | `run_daily.py` runs independently of the UI (scheduled) | ✅ DONE | CRITICAL |
| 2 | SQLite ledger with decisions, trades, and portfolio snapshots (3 tables) | ✅ DONE | CRITICAL |
| 3 | Meta-controller v1 (bandit/stacking) combining all 14 signal modules | ✅ DONE | HIGH |
| 4 | 2-3+ months of logged paper trading episodes | ⬜ TIME-GATED | HIGH |
| 5 | Dashboard Ledger page reads from ledger | ✅ DONE | MEDIUM |
| 6 | (Optional) RL upgrade with differential Sharpe reward | ⬜ DEFERRED | LOW |

---

## Pre-Stage 6: Current State Assessment

| Component | Current State | What Needs to Change |
|-----------|--------------|---------------------|
| Signal modules | 14 independent modules (ensemble, sentiment, flow, risk, regime, volatility, ranking, etc.) | No change — they stay as leaf nodes |
| Decision making | Manual: user clicks tabs, interprets signals themselves | Meta-controller: 14 outputs → 1 decision |
| Scheduling | Manual: `run_pipeline.py` requires human trigger | Orchestrator: scheduled daily process |
| Portfolio state | `st.session_state.portfolio` — ephemeral, dies with browser tab | SQLite ledger — persistent, queryable |
| Dashboard | Reads from session state and live yfinance calls | Reads from ledger (historical decisions + outcomes) |
| Learning feedback | None — no record of what worked | Ledger logs outcomes → meta-controller retrains on history |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Signal layer (existing — NO CHANGES)        │
│  ensemble, sentiment, flow, risk, regime, volatility,   │
│  ranking, mf_tracker, multitimeframe, backtester...     │
└─────────────────────┬───────────────────────────────────┘
                      │ 14 module outputs as state vector
┌─────────────────────▼───────────────────────────────────┐
│              Meta-controller (NEW — Task 3)              │
│  Contextual bandit / stacked logistic regression         │
│  14 opinions → 1 auditable decision (BUY/SELL/HOLD)     │
└─────────────────────┬───────────────────────────────────┘
                      │ decision + confidence + reasoning
┌─────────────────────▼───────────────────────────────────┐
│              Orchestrator (NEW — Task 1)                 │
│  run_daily.py — scheduled, runs after market close       │
│  pull data → run signals → meta-controller → log trade   │
└─────────────────────┬───────────────────────────────────┘
                      │ writes to
┌─────────────────────▼───────────────────────────────────┐
│              Ledger (NEW — Task 2)                       │
│  SQLite database — decisions, trades, portfolio snapshots│
│  Persistent, queryable, the dataset for future learning  │
└─────────────────────┬───────────────────────────────────┘
                      │ reads from
┌─────────────────────▼───────────────────────────────────┐
│              Dashboard (MODIFIED — Task 4)               │
│  Dashboard reads from ledger, not session state             │
│  Shows: historical decisions, P&L, signal accuracy       │
└─────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Why SQLite (not Postgres)?

| Factor | SQLite | Postgres |
|--------|--------|----------|
| Setup | Zero config, file-based | Requires server install |
| Dependencies | Built into Python | Needs psycopg2 + running server |
| Portability | Single `.db` file, easy to backup | Requires migration tools |
| Scale | 100K+ rows per year — plenty for daily trading | Overkill for personal project |
| Windows | Works out of the box | Requires WSL or Docker |

**Decision:** SQLite. One file, no server, works on Windows. If the project ever grows to multi-user or high-frequency, migrate to Postgres.

### Why Contextual Bandit (not full RL)?

| Factor | Contextual Bandit | Full RL (PPO/DQN) |
|--------|-------------------|---------------------|
| Data needed | 100+ episodes | 10,000+ episodes |
| Interpretability | Learned weights = which signals matter | Black box |
| Overfitting risk | Low (linear model) | High (deep network on small data) |
| Training speed | Seconds | Minutes to hours |
| Audit trail | "Module X has weight 0.3 because..." | "Neural net decided..." |

**Decision:** Contextual bandit. With 10 stocks and daily bars, you have ~2,500 trading days of data. A bandit learns from this; a full RL agent memorizes it. The bandit's learned weights ARE the audit mechanism — they tell you which modules are currently predictive.

### Why not modify signal modules?

The 14 signal modules are leaf nodes — they take data and return a result. They don't need to know about each other or about the meta-controller. This is deliberate:
- Adding a new signal module doesn't require changing any existing code
- Removing a signal module doesn't break anything
- The meta-controller is the only thing that needs to know about all modules

---

## Execution Plan: 5 Tasks

### Task 1: Orchestrator (run_daily.py)
**Impact:** Critical — system must live outside UI
**Effort:** 3-4 hours
**Files:** New `src/orchestrator.py`, New `run_daily.py`

**Why:** The orchestrator gives the system a life outside the React/FastAPI UI.

**Current state:**
```python
# Everything requires manual trigger:
# - Manual UI triggers
# - python run_pipeline.py (manual)
# - schedule_pipeline.py (exists but only runs pipeline, not full signal suite)
```

**Target:**
```python
# run_daily.py — standalone, scheduled
# 1. Fetch latest data for all tickers
# 2. Run all 14 signal modules
# 3. Meta-controller: combine → decision
# 4. Simulate paper trade
# 5. Log to ledger
```

**Implementation steps:**

1. Create `src/orchestrator.py` with class `DailyOrchestrator`:
   ```python
   class DailyOrchestrator:
       """Runs the full signal pipeline once per day."""

       def __init__(self, tickers: list[str], ledger: Ledger):
           self.tickers = tickers
           self.ledger = ledger

       def run(self, date: str = None) -> dict:
           """Execute one full daily cycle."""
           # Returns summary of what happened
   ```

2. Implement `run()` method — the daily flow:
   ```
   For each ticker:
     a. Fetch latest OHLCV data (yfinance, period="5d")
     b. Validate data (no gaps, not stale)
     c. Compute features (41 technical + alt-data)
     d. Run all signal modules:
        - ensemble.predict_ensemble() → direction, confidence
        - sentiment.get_stock_sentiment() → score
        - flow.get_flow_sentiment() → fii_net, dii_net
        - flow.fetch_options_pcr() → pcr, max_pain
        - multitimeframe.get_combined_signal() → mtf_signal
        - regime.detect_regime() → regime, confidence
        - risk.generate_risk_report() → var_95, cvar_95, sharpe
        - volatility.forecast_volatility() → vol_forecast
        - ranking.fundamental_score() → fundamental_score (cached)
     e. Collect all signals into a state vector
     f. Pass to meta-controller → action, position_size, reasoning
     g. Log decision to ledger
   ```

3. Create `run_daily.py` — thin wrapper:
   ```python
   """Daily autonomous loop. Run via scheduler or manually."""
   from src.orchestrator import DailyOrchestrator
   from src.ledger import Ledger

   def main():
       ledger = Ledger("stomar.db")
       orchestrator = DailyOrchestrator(tickers=NSE_STOCKS, ledger=ledger)
       summary = orchestrator.run()
       print(f"Decisions: {summary['decisions']}")
       print(f"Trades: {summary['trades']}")

   if __name__ == "__main__":
       main()
   ```

4. Update `schedule_pipeline.py` to call `run_daily.py` instead of `run_pipeline.py`

5. Add error handling:
   - If yfinance fails → skip that ticker, log warning, continue
   - If a signal module fails → use default (neutral signal), log warning
   - If meta-controller fails → default to HOLD, log error
   - Always log what happened, even on partial failure

**Tests:** `tests/test_orchestrator.py`
- `test_orchestrator_creates_instance` — initializes with tickers and ledger
- `test_orchestrator_runs_one_cycle` — mock signals, verify it completes
- `test_orchestrator_logs_to_ledger` — decisions appear in ledger
- `test_orchestrator_handles_signal_failure` — continues when one module fails
- `test_orchestrator_handles_yfinance_failure` — skips ticker, logs warning
- `test_orchestrator_returns_summary` — has decisions, trades, errors keys

---

### Task 2: Persistent Ledger (SQLite)
**Impact:** Critical — ephemeral state = no learning possible
**Effort:** 3-4 hours
**Files:** New `src/ledger.py`, New `tests/test_ledger.py`

**Why:** `st.session_state.portfolio` is ephemeral. Every paper trade, every module signal, and every outcome is lost when the browser closes. No historical record exists for analysis or learning.

**Current state:**
```python
# src/paper_trader.py — saves to JSON, but:
# - Only stores portfolio state (holdings, cash)
# - Doesn't log individual signal outputs
# - Doesn't log what each module said at decision time
# - No outcome tracking (what actually happened the next day)
```

**Target:**
```python
# src/ledger.py — SQLite database
# Tables: decisions, paper_trades, portfolio_snapshots
# Every daily cycle writes a complete record
```

**Implementation steps:**

1. Create `src/ledger.py` with class `Ledger`:
   ```python
   import sqlite3
   from datetime import datetime

   class Ledger:
       """Persistent trading journal in SQLite."""

       def __init__(self, db_path: str = "stomar.db"):
           self.conn = sqlite3.connect(db_path)
           self._create_tables()

       def log_decision(self, date: str, ticker: str, signals: dict, action: str,
                        position_size: float, confidence: float, reasoning: str) -> int:
           """Log a meta-controller decision. Returns decision ID."""

       def log_trade(self, decision_id: int, ticker: str, side: str,
                     quantity: int, price: float, slippage: float, costs: float) -> None:
           """Log a paper trade execution."""

       def log_outcome(self, decision_id: int, actual_return: float, actual_direction: int) -> None:
           """Log what actually happened (filled next day)."""

       def log_snapshot(self, date: str, total_value: float, cash: float, holdings: dict) -> None:
           """Log portfolio snapshot."""

       def get_decisions(self, ticker: str = None, start_date: str = None, end_date: str = None) -> list[dict]:
           """Query historical decisions."""

       def get_trades(self, ticker: str = None) -> list[dict]:
           """Query historical trades."""

       def get_performance(self, start_date: str = None, end_date: str = None) -> dict:
           """Aggregate performance metrics from logged decisions."""

       def get_signal_accuracy(self) -> dict:
           """Per-module accuracy: how often did each signal predict correctly?"""

       def close(self):
           self.conn.close()
   ```

2. Create database schema:
   ```sql
   CREATE TABLE decisions (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       date TEXT NOT NULL,
       ticker TEXT NOT NULL,
       -- Signal outputs (what each module said)
       ensemble_direction INTEGER,
       ensemble_confidence REAL,
       sentiment_score REAL,
       fii_net REAL,
       dii_net REAL,
       pcr REAL,
       max_pain REAL,
       mtf_signal REAL,
       regime TEXT,
       regime_confidence REAL,
       var_95 REAL,
       cvar_95 REAL,
       sharpe REAL,
       volatility_forecast REAL,
       fundamental_score REAL,
       -- Meta-controller decision
       action TEXT NOT NULL,          -- BUY / SELL / HOLD
       position_size REAL,
       confidence REAL,
       reasoning TEXT,
       -- Actual outcome (filled next day)
       actual_return REAL,
       actual_direction INTEGER,
       correct INTEGER,               -- 1 if prediction matched outcome
       -- Metadata
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE paper_trades (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       decision_id INTEGER REFERENCES decisions(id),
       ticker TEXT NOT NULL,
       side TEXT NOT NULL,             -- BUY / SELL
       quantity INTEGER,
       price REAL,
       slippage REAL,
       costs REAL,
       total_cost REAL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE portfolio_snapshots (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       date TEXT NOT NULL UNIQUE,
       total_value REAL,
       cash REAL,
       holdings_json TEXT,             -- JSON blob of all positions
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE INDEX idx_decisions_date ON decisions(date);
   CREATE INDEX idx_decisions_ticker ON decisions(ticker);
   CREATE INDEX idx_trades_ticker ON paper_trades(ticker);
   ```

3. Implement `get_signal_accuracy()` — the audit mechanism:
   ```python
   def get_signal_accuracy(self) -> dict:
       """For each signal module, compute how often it predicted correctly."""
       # ensemble: direction == actual_direction?
       # sentiment: score > 0 and actual_direction == 1?
       # fii_net: net > 0 and actual_direction == 1?
       # etc.
       # Returns: {"ensemble": 0.52, "sentiment": 0.48, "fii": 0.55, ...}
   ```

4. Add outcome tracking — after each decision, check next day's return:
   ```python
   def update_outcomes(self, fetch_fn):
       """For decisions without outcomes, fetch next-day return and fill in."""
       # This runs daily before new decisions
   ```

**Tests:** `tests/test_ledger.py`
- `test_ledger_creates_db` — file exists after init
- `test_log_decision_returns_id` — auto-incrementing ID
- `test_log_trade_links_to_decision` — foreign key works
- `test_get_decisions_filters_by_ticker` — query works
- `test_get_decisions_filters_by_date` — date range works
- `test_log_outcome_updates_decision` — actual_return filled in
- `test_get_signal_accuracy` — returns per-module accuracy
- `test_get_performance` — returns total return, win rate
- `test_portfolio_snapshot_roundtrip` — save and load
- `test_ledger_handles_corrupt_db` — graceful error

---

### Task 3: Meta-Controller v1 (Contextual Bandit)
**Impact:** High — turns 14 opinions into one auditable decision
**Effort:** 4-5 hours
**Files:** New `src/meta_controller.py`, New `tests/test_meta_controller.py`

**Why:** The 14 signal modules exist independently. The ensemble has a meta-learner, but the other modules (sentiment, flow, risk, regime, etc.) are not combined into a single decision. There's no layer that asks "given all signals, what should we actually do?"

**Current state:**
```python
# Each module produces an output, but:
# - No layer combines them into one decision
# - User manually interprets signals in the UI
# - No learning from historical signal → outcome pairs
```

**Target:**
```python
# meta_controller.py
# Input: state_vector = [ensemble_dir, ensemble_conf, sentiment, fii_net,
#                         dii_net, pcr, mtf, regime, var, cvar, vol, fundamental, ...]
# Output: action (BUY/SELL/HOLD), position_size, confidence, reasoning
# Learning: retrains weekly on ledger history
```

**Implementation steps:**

1. Create `src/meta_controller.py` with class `MetaController`:
   ```python
   class MetaController:
       """Combines 14 signal modules into one decision.

       Uses stacked logistic regression (contextual bandit).
       The learned weights ARE the audit mechanism:
       high weight = module is currently predictive.
       """
       SIGNAL_NAMES = [
           "ensemble_direction", "ensemble_confidence",
           "sentiment_score", "fii_net", "dii_net",
           "pcr", "mtf_signal", "regime_bull", "regime_bear",
           "var_95", "cvar_95", "sharpe",
           "volatility_forecast", "fundamental_score",
       ]

       def __init__(self):
           self.model = None  # LogisticRegression, trained on ledger
           self.weights = None  # Latest learned weights

       def extract_state_vector(self, signals: dict) -> np.ndarray:
           """Convert module outputs to fixed-size feature vector."""

       def decide(self, signals: dict) -> dict:
           """Take all signal outputs, return one decision."""
           # Returns: {"action": "BUY/SELL/HOLD", "position_size": float,
           #           "confidence": float, "reasoning": str}

       def train(self, ledger: Ledger) -> dict:
           """Retrain on ledger history. Returns accuracy metrics."""

       def get_weights(self) -> dict:
           """Return current learned weights (the audit scorecard)."""

       def explain(self, signals: dict) -> str:
           """Human-readable explanation of why this decision was made."""
   ```

2. Implement `decide()` — the core decision logic:
   ```python
   def decide(self, signals: dict) -> dict:
       vector = self.extract_state_vector(signals)

       if self.model is None:
           # Cold start: use rule-based decision
           return self._rule_based_decide(signals)

       prob = self.model.predict_proba(vector.reshape(1, -1))[0, 1]

       if prob > 0.6:
           action = "BUY"
           position_size = min(0.10, (prob - 0.5) * 0.5)  # Max 10% per position
       elif prob < 0.4:
           action = "SELL"
           position_size = min(0.10, (0.5 - prob) * 0.5)
       else:
           action = "HOLD"
           position_size = 0.0

       return {
           "action": action,
           "position_size": round(position_size, 4),
           "confidence": round(abs(prob - 0.5) * 2, 4),  # 0-1 scale
           "reasoning": self.explain(signals),
       }
   ```

3. Implement `train()` — learn from ledger history:
   ```python
   def train(self, ledger: Ledger) -> dict:
       decisions = ledger.get_decisions()
       if len(decisions) < 100:
           return {"status": "insufficient_data", "n_samples": len(decisions)}

       X = np.array([self.extract_state_vector(d) for d in decisions])
       y = np.array([d["actual_direction"] for d in decisions])

       # Split: 80% train, 20% test
       split = int(len(X) * 0.8)
       self.model = LogisticRegression(C=1.0, max_iter=1000)
       self.model.fit(X[:split], y[:split])

       accuracy = self.model.score(X[split:], y[split:])
       self.weights = dict(zip(self.SIGNAL_NAMES, self.model.coef_[0]))

       return {"status": "trained", "accuracy": accuracy, "n_samples": len(X)}
   ```

4. Implement `explain()` — human-readable reasoning:
   ```python
   def explain(self, signals: dict) -> str:
       if self.weights is None:
           return "Rule-based (no training data yet)"

       contributions = []
       for name, weight in self.weights.items():
           value = signals.get(name, 0)
           contribution = weight * value
           if abs(contribution) > 0.05:
               direction = "supports BUY" if contribution > 0 else "supports SELL"
               contributions.append(f"{name}: {direction} (weight={weight:.3f})")

       return "; ".join(contributions) if contributions else "No strong signals"
   ```

5. Implement `get_weights()` — the audit scorecard:
   ```python
   def get_weights(self) -> dict:
       """Return learned weights sorted by absolute importance."""
       if self.weights is None:
           return {}
       return dict(sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True))
   ```

**Tests:** `tests/test_meta_controller.py`
- `test_meta_controller_creates_instance` — initializes
- `test_extract_state_vector_shape` — returns (14,) array
- `test_decide_without_model` — uses rule-based fallback
- `test_decide_with_model` — returns BUY/SELL/HOLD
- `test_decide_returns_reasoning` — has explanation string
- `test_train_insufficient_data` — returns status message
- `test_train_with_enough_data` — fits model, returns accuracy
- `test_get_weights_returns_dict` — sorted by importance
- `test_explain_returns_string` — human-readable
- `test_position_size_capped` — never exceeds 10% per position
- `test_confidence_in_range` — between 0 and 1

---

### Task 4: Dashboard Reads from Ledger
**Impact:** Medium — Dashboard shows historical decisions, not just live state
**Effort:** 2-3 hours
**Files:** `api/routers/` (new endpoints), `web/src/pages/` (new pages), `src/ledger.py` (query helpers)

**Why:** Currently, the frontend reads from live yfinance calls and shows only the current moment. The ledger has historical decisions and outcomes — the dashboard should visualize them.

**Current state:**
```python
# Frontend reads:
# - Live yfinance calls (slow, rate-limited)
# - No historical decision visualization
```

**Target:**
```python
# API reads from ledger:
# - Historical decisions table
# - Per-module signal accuracy
# - P&L over time
# - Decision → outcome correlation
```

**Implementation steps:**

1. Add new Ledger page to `web/src/pages/` and API endpoint to `api/routers/`:
   - **Decision History** — table of all logged decisions (date, ticker, action, confidence, outcome)
   - **Signal Accuracy** — bar chart of per-module accuracy (which signals are predictive?)
   - **P&L Curve** — equity curve from portfolio snapshots
   - **Decision Breakdown** — pie chart of BUY/SELL/HOLD decisions
   - **Module Audit** — table showing each module's weight and current accuracy

2. Update existing tabs to read from ledger where appropriate:
   - Paper Trading tab → show historical trades from ledger
   - Monitoring tab → show signal accuracy from ledger
   - Portfolio tab → show portfolio snapshots from ledger

3. Add query helpers to `src/ledger.py`:
   ```python
   def get_daily_pnl(self) -> pd.DataFrame:
       """Daily P&L from portfolio snapshots."""

   def get_module_accuracy_history(self) -> pd.DataFrame:
       """How each module's accuracy has changed over time."""

   def get_regime_performance(self) -> dict:
       """Performance broken down by regime."""
   ```

**Tests:** `tests/test_ledger.py` (extend existing)
- `test_get_daily_pnl` — returns DataFrame with date, pnl columns
- `test_get_module_accuracy_history` — returns per-module per-date accuracy
- `test_get_regime_performance` — returns Bull/Bear/Sideways breakdown

---

### Task 5: Automated Daily Scheduler
**Impact:** Medium — runs without human intervention
**Effort:** 1-2 hours
**Files:** Update `schedule_pipeline.py`, New `run_daily.py`

**Why:** `schedule_pipeline.py` exists but only runs the retraining pipeline. It needs to run the full daily signal suite instead.

**Current state:**
```python
# schedule_pipeline.py:
# - Installs Windows scheduled task
# - Runs run_pipeline.py (retraining only)
# - Does NOT run signal modules, meta-controller, or logging
```

**Target:**
```python
# schedule_pipeline.py:
# - Runs run_daily.py (full daily cycle)
# - Configurable time (default 4:00 PM IST, after market close)
# - Logs to data/pipeline_logs/
```

**Implementation steps:**

1. Update `schedule_pipeline.py` to point to `run_daily.py`:
   ```python
   PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, "run_daily.py")
   ```

2. Add Windows Task Scheduler configuration:
   - Task name: `StoMar_Daily_Signal`
   - Trigger: Daily at 4:00 PM IST (16:00)
   - Action: `python run_daily.py`
   - On failure: retry up to 3 times with 15-min intervals

3. Add `--run-now` flag to `run_daily.py` for manual testing:
   ```bash
   python run_daily.py              # Run once now
   python run_daily.py --dry-run    # Run but don't log to ledger
   python run_daily.py --ticker RELIANCE.NS  # Run for one ticker only
   ```

4. Add logging to `data/pipeline_logs/`:
   - Daily log files: `pipeline_YYYY-MM-DD.log`
   - Capture stdout + stderr
   - Rotate logs (keep last 30 days)

**Tests:** `tests/test_orchestrator.py` (extend existing)
- `test_run_now_flag` — runs immediately
- `test_dry_run_doesnt_write_ledger` — no entries in database
- `test_ticker_flag` — processes only specified ticker

---

## Execution Order

```
1. Task 2: Persistent Ledger (3-4 hrs)       ← CRITICAL, everything depends on it
2. Task 3: Meta-Controller v1 (4-5 hrs)      ← CRITICAL, needs ledger to train on
3. Task 1: Orchestrator (3-4 hrs)            ← HIGH, ties everything together
4. Task 5: Automated Scheduler (1-2 hrs)     ← MEDIUM, thin wrapper around orchestrator
5. Task 4: Dashboard from Ledger (2-3 hrs)   ← MEDIUM, can parallel with 1-3
```

**Why this order:**
- **Ledger first** — without it, there's nothing to log decisions to, nothing for the meta-controller to train on
- **Meta-controller second** — needs ledger history to train; orchestrator needs it to make decisions
- **Orchestrator third** — ties ledger + meta-controller + signal modules together
- **Scheduler fourth** — thin wrapper, runs orchestrator on schedule
- **Dashboard last** — reads from ledger, shows results; can be done in parallel

**Parallelizable:**
- Task 4 (dashboard) is independent of Tasks 1-3 — can be done in parallel
- Tasks 1, 2, 3 must be sequential (orchestrator → ledger → meta-controller dependency)

**Total estimated time:** 13-18 hours

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/orchestrator.py` | DailyOrchestrator class — runs full signal pipeline |
| `src/ledger.py` | SQLite ledger — decisions, trades, portfolio snapshots |
| `src/meta_controller.py` | Contextual bandit — 14 signals → 1 decision |
| `run_daily.py` | CLI entry point for daily autonomous loop |
| `tests/test_orchestrator.py` | 6 tests for orchestrator |
| `tests/test_ledger.py` | 10 tests for ledger |
| `tests/test_meta_controller.py` | 11 tests for meta-controller |

## Files to Modify

| File | Change |
|------|--------|
| `api/routers/` + `web/src/pages/` | Add Ledger dashboard page, update Paper Trading + Monitoring pages |
| `schedule_pipeline.py` | Point to `run_daily.py` instead of `run_pipeline.py` |

---

## Exit Criteria Checklist

After all 5 tasks:

- [x] `run_daily.py` runs end-to-end without the UI
- [x] SQLite ledger has 3 tables with correct schema (decisions, paper_trades, portfolio_snapshots)
- [x] Ledger logs decisions, trades, outcomes, and portfolio snapshots
- [x] Meta-controller v1 trained on ledger history (requires 100+ resolved decisions)
- [x] Meta-controller returns BUY/SELL/HOLD with reasoning
- [x] Learned weights show which modules are currently predictive
- [x] Orchestrator handles partial failures gracefully
- [x] Scheduler installs and runs at 4:00 PM IST
- [x] Ledger tab reads from dashboard (other tabs migrate over time)
- [x] All new modules have tests passing (53 new tests, 581 total)
- [x] Lint clean on all new/modified files
- [ ] 2-3+ months of logged episodes (time-gated — starts when deployed)

---

## Critical Context

- **User's venv:** `C:\Users\uttam\venv` (Python 3.14, torch 2.10+cu130)
- **User's project:** `C:\Users\uttam\development\stomar`
- **Run tests:** `$env:PYTHONPATH = "C:\Users\uttam\development\stomar"; python -m pytest tests/ -v`
- **Run linter:** `ruff check src/ tests/ --output-format=concise`
- **Current test count:** 528 tests passing
- **User constraint:** Everything must be free, must work on Windows
- **Key insight:** This stage is about AUTOMATION — the system runs itself, learns from its own history
- **SQLite choice:** Zero config, file-based, works on Windows, scales to 100K+ daily entries
- **Bandit choice:** Interpretable, learns from small data, weights = audit mechanism
- **Feedback loop:** Ledger outcomes → meta-controller retrains → better decisions → more accurate signals
- **Architecture diagram:** `docs/stomar_autonomous_loop_architecture.svg`

---

## Appendix: Stage 5 (Completed)

Stage 5 built the open platform. 104 new tests added, 528 total.

| Component | File | Status |
|-----------|------|--------|
| Stacked meta-learner | `src/ensemble.py` | ✅ |
| Regime-conditional routing | `src/ensemble.py` | ✅ |
| MF tracker | `src/mf_tracker.py` | ✅ |
| Multi-source sentiment | `src/sentiment.py` | ✅ |
| Fundamental ranking | `src/ranking.py` | ✅ |
| Volatility forecasting | `src/volatility.py` | ✅ |
| Batch training | `src/trainer.py` | ✅ |
| Apache 2.0 license | `LICENSE` | ✅ |
| Contribution guidelines | `CONTRIBUTING.md` | ✅ |

## Appendix: Stage 4 (Completed)

Stage 4 built the execution layer. 89 new tests added, 424 total (at the time).

| Component | File | Status |
|-----------|------|--------|
| Event-driven engine | `src/engine.py` | ✅ |
| Risk controls | `src/risk_controls.py` | ✅ |
| Paper trader | `src/paper_trader.py` | ✅ |
| Execution quality | `src/execution_quality.py` | ✅ |
| Fill probability | `src/engine.py` | ✅ |
| State persistence | `src/paper_trader.py` | ✅ |
| Paper trading page | `web/src/pages/` + `api/routers/` | ✅ |
| Pipeline --paper flag | `run_pipeline.py` | ✅ |
