# StoMar — Improvement Roadmap to Real-Money Reliability

> Track each item: change `[ ]` → `[x]` when done. Work top-to-bottom.
> Do NOT skip phases. Each phase gates the next.

---

## How to Read This File

| Symbol | Meaning |
|--------|---------|
| 🔴 Blocker | Must be done before real money — safety or correctness issue |
| 🟠 High | Significantly improves reliability or trust |
| 🟡 Medium | Worthwhile but not a blocker |
| 🟢 Low / Research | Good to have, do after fundamentals are solid |

**Estimated effort** is for a focused solo developer who knows the codebase.

---

## Phase 0 — Operational Wiring (Do This Week)
*These are one-day fixes that close real safety gaps right now.*

- [x] 🔴 **P0.1** Wire drift detection to automatic retraining trigger
  - File: `src/signals/monitoring.py` → `ModelMonitor._log_alert()`
  - When `alert.severity == "critical"` on `oos_accuracy` or `feature_drift`, enqueue `RetrainingPipeline.run(ticker)`
  - Log the trigger event to `data/monitoring/retrain_triggers.json`
  - Effort: 2–3 hours

- [x] 🔴 **P0.2** Attach stop-loss orders to every paper trade entry
  - File: `src/signals/orchestrator.py` → `DailyOrchestrator._execute_paper_trade()`
  - After BUY fill, place a SELL STOP at `fill_price * (1 - 0.05)` (5% stop, configurable)
  - After SELL SHORT fill, place a BUY STOP at `fill_price * (1 + 0.05)`
  - The `ExecutionEngine` already supports `stop_price` on orders — just wire it
  - Effort: 2 hours

- [x] 🔴 **P0.3** Pipeline failure notification
  - File: `auto_pipeline.py` → `run()` except block
  - On any step failure, write to `data/pipeline_failures.json` with timestamp + stage + error
  - Add `/api/monitoring/failures` endpoint to surface this in the UI
  - Add a visible red banner on the Dashboard if failures exist from last 24h
  - Effort: 2–3 hours

- [x] 🔴 **P0.4** Daily P&L reset for `RiskController`
  - File: `src/trading/risk_controls.py` + `src/signals/orchestrator.py`
  - `RiskController.daily_pnl` is tracked but `reset_daily()` is never called automatically
  - Call `reset_daily()` at the start of each new trading day in the orchestrator
  - Without this, the 2% daily loss limit accumulates across days and becomes meaningless
  - Effort: 30 minutes

- [x] 🟠 **P0.5** Persist meta-controller per-ticker, not just global
  - File: `auto_pipeline.py` → `_ensure_meta_controller()`
  - Currently loads one global `meta_controller.pkl`; per-ticker meta models are trained in `trainer.py` but not used in the daily loop
  - Load `models/meta_{ticker}.pkl` per ticker in `_run_daily()` if it exists
  - Effort: 1 hour

---

## Phase 1 — Persistent State Machine for the Pipeline
*The pipeline currently has no memory of where it failed. A crash at step 3 restarts from step 0.*

- [x] 🔴 **P1.1** Add pipeline state machine with resume-from-checkpoint
  - New file: `src/core/pipeline_state.py`
  - States: `IDLE → FETCH → VALIDATE → FEATURES → TRAIN → EVALUATE → PROMOTE → DAILY → PAPER → ARCHIVE → DONE`
  - Store current state in `data/pipeline_checkpoint.json` with ticker + stage + timestamp
  - On startup, read checkpoint and resume from last incomplete stage
  - Each stage marks itself complete before starting the next
  - Per-ticker checkpoints for `RetrainingPipeline` (`pipeline_checkpoint_{TICKER}.json`)
  - Auto-pipeline uses shared checkpoint (`pipeline_checkpoint.json`)
  - GET `/api/monitoring/checkpoint` endpoint exposes checkpoint status
  - Atomic writes, corrupt-file handling, 54 tests
  - Effort: 4–6 hours

- [x] 🟠 **P1.2** Add per-stage retry logic
  - Stages `FETCH`, `VALIDATE`, and `DAILY` should retry up to 3 times with exponential backoff (1min, 5min, 15min) before marking as failed
  - Effort: 2 hours

^- [x] 🟠 **P1.3** Cross-platform scheduler support
  - File: `schedule_pipeline.py`
  - Add Linux/Mac path using `cron` via `python-crontab` library
  - Auto-detect OS and use the right scheduler
  - Effort: 2–3 hours

^- [x] 🟡 **P1.4** Add startup auto-run on system boot (not just daily 4PM)
  - Windows: add a second Task Scheduler entry with trigger `AtStartup` + delay 5 minutes
  - This ensures if PC was off during 4PM run, it catches up on next boot
  - Currently `auto_pipeline.py` does backfill but only if triggered; this makes it truly automatic
  - Effort: 1 hour

---

## Phase 2 — Data Quality & Validation Hardening
*Bad data is worse than no data. The model will confidently trade on garbage.*

- [x] 🔴 **P2.1** Auto-fill missing trading day data (not just detect gaps)
  - File: `src/data/data_validation.py` + `src/data/data_fetcher.py`
  - When gaps are detected, automatically re-fetch with extended period to fill them
  - Apply forward-fill only for non-price columns (volume, indicators) — never forward-fill OHLC
  - Log every filled gap to `data/monitoring/data_fixes.json`
  - Effort: 3 hours

- [x] 🟠 **P2.2** Corporate action adjustment pipeline
  - File: `src/data/data_fetcher.py`
  - yfinance returns adjusted prices by default (`auto_adjust=True`) — verify this is consistently set
  - Add a check: if overnight move >20%, flag as potential unadjusted split and force re-fetch with `auto_adjust=True`
  - Effort: 2 hours

- [x] 🟠 **P2.3** Indian market holiday calendar
  - File: `auto_pipeline.py` → `_get_trading_days_since()`
  - Currently uses Mon–Fri as business days; doesn't account for NSE holidays (Diwali, Holi, etc.)
  - Add a hardcoded NSE holiday list (or fetch from NSE website) and exclude those dates from backfill
  - False "missed day" triggers on holidays waste compute and create confusion
  - Effort: 2 hours

- [x] 🟡 **P2.4** Data source fallback chain with logging
  - File: `src/data/data_sources.py`
  - If yfinance returns empty or too-short data, log a warning and try an alternate source
  - Track per-ticker fetch success rate in `data/monitoring/fetch_stats.json`
  - Note: fetch health tracking done; no alternate data source exists in the codebase (yfinance is the only provider) — fallback is re-fetch with extended period / force refresh
  - Effort: 2 hours

---

## Phase 3 — Paper Trading Realism & Validation Period
*You must run paper trading for a minimum period before any real money. This phase is mostly time, not code.*

- [ ] 🔴 **P3.1** Start the paper trading clock — minimum 3 months uninterrupted
  - Command: `uv run run_daily.py --paper-trade --capital 200000`
  - Schedule via Task Scheduler to run every weekday at 3:45 PM IST (after market close)
  - Do NOT interrupt, reset, or modify the strategy during this period
  - This is a time gate, not a code task — it just has to run
  - ⏳ IN PROGRESS since 2026-08-04 (0/60 trading days) — scheduled via P1.3/P1.4

- [x] 🔴 **P3.2** Paper vs Nifty 50 benchmark comparison on Ledger page
  - File: `api/routers/ledger.py` + `web/src/pages/Ledger.tsx`
  - Fetch Nifty 50 (`^NSEI`) daily returns for the same period as paper trading
  - Show: paper cumulative return vs Nifty buy-and-hold on the same chart
  - Show: alpha (paper return − Nifty return), information ratio
  - This is the single most honest signal of whether the system adds value
  - Effort: 4–6 hours

- [x] 🔴 **P3.3** Paper trading outcome logging (fill actual_return next day)
  - File: `src/trading/ledger.py` + `run_daily.py`
  - At the start of each daily run, for yesterday's BUY/SELL decisions, fetch yesterday's close and today's open
  - Call `ledger.log_outcome(decision_id, actual_return, actual_direction)` for each resolved decision
  - This is what trains the meta-controller on real forward-looking data (not backfill)
  - Effort: 3 hours

^- [x] 🟠 **P3.4** Paper trading performance dashboard additions
  - File: `web/src/pages/PaperTrading.tsx`
  - Add: rolling 30-day accuracy (decisions that were correct)
  - Add: signal accuracy breakdown per module (ensemble vs sentiment vs MTF, etc.) from ledger
  - Add: current drawdown from peak equity with alert at 10%
  - Effort: 4 hours

- [x] 🟠 **P3.5** Minimum paper trading gate before live trading is enabled
  - File: new `src/core/readiness_check.py`
  - Function `is_ready_for_live_trading()` checks:
    - Paper trading running ≥ 60 business days
    - Rolling 30-day accuracy ≥ 51%
    - Max drawdown in paper period ≤ 15%
    - No critical drift alerts in last 7 days
    - All 20 models have been retrained at least once live
  - Returns a readiness report — surface it on the Dashboard
  - This is a guard, not a guarantee
  - Effort: 3 hours

---

## Phase 4 — Explainability & Trust Building
*Every recommendation should be self-explanatory. If you can't see WHY it said BUY, you can't trust it.*

^- [x] 🟠 **P4.1** Structured recommendation card with signal breakdown
  - File: `api/routers/consensus.py` + `web/src/pages/Consensus.tsx`
  - For each BUY/SELL decision, show a card like:
    ```
    BUY RELIANCE  Confidence: 68%
    ─────────────────────────────────────
    ✅ Ensemble ML      → UP (72% prob)
    ✅ Sentiment        → Positive (+0.31)
    ✅ FII Flow         → Net +₹842 Cr
    ⚠️  MTF Signal       → Neutral
    ✅ Regime           → Bull (ADX 28)
    ❌ Volatility       → High (σ=2.8%)
    ─────────────────────────────────────
    Stop Loss: ₹2,847  Take Profit: ₹3,102
    Risk: MEDIUM   Horizon: 3–5 days
    ```
  - Effort: 6 hours

^- [x] 🟠 **P4.2** Decision audit trail on Ledger page
  - Show all 14 signal values for any historical decision (already stored in SQLite — just expose it)
  - Allow filtering: "show only decisions where all 3 main signals agreed" → measure accuracy of those
  - Effort: 3 hours

^- [x] 🟡 **P4.3** SHAP explanations in production pipeline
  - File: `src/signals/interpretability.py` → called from `orchestrator.py`
  - Currently SHAP is implemented but not invoked in the live daily run
  - Add SHAP top-5 feature contributors to each decision's `reasoning` field in the ledger
  - Effort: 2 hours

- [x] 🟡 **P4.4** Confidence calibration check
  - File: new `src/models/calibration.py`
  - When model says "82% confidence BUY", does it actually win 82% of the time at that confidence level?
  - Compute reliability diagrams (calibration curves) from ledger data after 3 months
  - Apply Platt scaling or isotonic regression if badly calibrated
  - Effort: 4 hours

---

## Phase 5 — Monitoring, Alerting & Ops Hardening
*A system that fails silently is worse than a system that doesn't run.*

- [x] 🟠 **P5.1** Email / desktop notification on critical events
  - File: new `src/core/notifier.py`
  - Send notification on: pipeline failure, kill switch triggered, drift alert (critical), drawdown >10%
  - Use `smtplib` for email (free, no dependencies) or Windows toast notifications via `plyer`
  - Effort: 3 hours

^- [x] 🟠 **P5.2** System health metrics on Monitoring page
  - File: `api/routers/monitoring.py` + `web/src/pages/Monitoring.tsx`
  - Add: last successful pipeline run timestamp + status
  - Add: data freshness per ticker (days since last fetch)
  - Add: model age per ticker (days since last training)
  - Add: paper trading consecutive days running
  - Add: kill switch status (is trading halted?)
  - Effort: 4 hours

- [x] 🟠 **P5.3** Automated daily monitoring sweep
  - File: `auto_pipeline.py` → add step after paper trades
  - For each ticker, run `ModelMonitor.check_data_freshness()` and `check_performance_drift()`
  - Write summary to `data/monitoring/daily_health.json`
  - Effort: 2 hours

- [x] 🟡 **P5.4** SQLite backup strategy
  - File: new step in `auto_pipeline.py`
  - Once per week, copy `data/stomar.db` to `data/backups/stomar_YYYYMMDD.db`
  - Keep last 4 weekly backups (auto-delete older ones)
  - The ledger is the most valuable artifact in the entire system — protect it
  - Effort: 1 hour

^- [x] 🟡 **P5.5** Tax simulation (STCG/LTCG)
  - File: `src/core/constants.py` + `src/trading/backtester.py`
  - Indian STCG: 15% on gains from positions held <1 year
  - Indian LTCG: 10% on gains >₹1 lakh from positions held >1 year
  - Add this to backtester's cost model so simulated returns are post-tax
  - Effort: 3 hours

---

## Phase 6 — ML Improvements (After 3 Months of Live Data)
*Do not start this phase until you have real forward-looking data in the ledger. Backfill data is not enough.*

- [ ] 🟠 **P6.1** Retrain meta-controller on live ledger data (not backfill)
  - File: `src/models/meta_controller.py`
  - After 60+ live decisions per ticker, retrain using `ledger.get_decisions()` with `actual_direction IS NOT NULL`
  - Compare accuracy of live-trained vs backfill-trained meta-controller
  - Effort: 2 hours (code is ready, just needs live data)

- [ ] 🟠 **P6.2** Automatic weekly model retraining schedule
  - File: `auto_pipeline.py` + `schedule_pipeline.py`
  - Every Sunday, run `RetrainingPipeline.run_all()` for tickers where:
    - Model age > 30 days, OR
    - Feature drift alert was triggered in last 7 days
  - Gate on OOS accuracy ≥ 50% before promoting new model (already in `pipeline.py`)
  - Effort: 2 hours

- [ ] 🟠 **P6.3** Hyperparameter tuning with Optuna
  - File: new `src/models/tuner.py`
  - Use Optuna to optimize XGBoost/LightGBM hyperparameters (depth, lr, min_child)
  - Run on a schedule (monthly), not on every retrain (too slow)
  - Keep best hyperparams in `data/best_hyperparams.json` per ticker
  - Effort: 6 hours

- [ ] 🟡 **P6.4** Feature importance monitoring and pruning
  - File: `src/signals/monitoring.py`
  - After each retrain, log XGBoost feature importances to the monitoring system
  - If a feature's importance drops to near-zero for 3 consecutive retrains, flag it for removal
  - Features that don't contribute add noise and slow inference
  - Effort: 3 hours

- [ ] 🟡 **P6.5** Probability calibration for ensemble outputs
  - File: `src/models/calibration.py` (new, from Phase 4.4)
  - Apply calibration to the stacked meta-learner's probability outputs
  - A 70% confidence signal should be right ~70% of the time
  - Effort: 3 hours (after calibration data exists from 3+ months of paper trading)

- [ ] 🟡 **P6.6** Online/incremental learning for XGBoost and LightGBM
  - Both support warm-start / `continue_training` — use this to update models daily with new data
  - Instead of full retrain weekly, do micro-updates daily + full retrain monthly
  - This lets the model adapt faster to regime changes
  - Effort: 4 hours

---

## Phase 7 — Reinforcement Learning (After 6+ Months of Live Data)

> **Why this phase comes last:** RL needs a replay buffer of real forward-looking episodes.
> Using backfill data → RL overfits to historical patterns and then fails live.
> The meta-controller (contextual bandit) already does RL-lite. True RL is an upgrade, not a replacement.

- [ ] 🟢 **P7.1** Define the RL environment correctly
  - New file: `src/rl/trading_env.py`
  - State: the 14-signal vector from the meta-controller (already defined)
  - Action space: {BUY, SELL, HOLD} × position_size (0.01 to 0.25 in steps of 0.01)
  - Reward: **differential Sharpe ratio** (not raw P&L — raw P&L encourages reckless sizing)
    ```
    reward = Sharpe(t) - Sharpe(t-1)
    ```
  - Episode: one trading day (or one week for stability)
  - Effort: 6 hours

- [ ] 🟢 **P7.2** Implement PPO agent for position sizing
  - New file: `src/rl/ppo_agent.py`
  - Use `stable-baselines3` (well-tested, not rolling your own)
  - Train on the logged ledger episodes (shuffled to reduce autocorrelation)
  - The agent learns WHEN to be aggressive vs conservative based on signal strength
  - Do NOT let the RL agent override the kill switch or risk controls — it operates within them
  - Effort: 8–12 hours

- [ ] 🟢 **P7.3** RL agent evaluation gates before deployment
  - The RL agent must beat the meta-controller's Sharpe ratio on a held-out test period
  - Must not increase max drawdown vs the meta-controller baseline
  - Must complete 100+ paper trades before any live consideration
  - Effort: 4 hours (evaluation harness)

- [ ] 🟢 **P7.4** Regime-conditional RL routing
  - Train separate PPO agents for Bull, Bear, Sideways regimes
  - The regime detector (`src/signals/regime.py`) selects which agent runs
  - Different market conditions require different risk appetites
  - Effort: 6 hours (after P7.2 is proven to work)

- [ ] 🟢 **P7.5** Online RL — learn from every trade in real time
  - Update the PPO agent's replay buffer with every completed paper trade outcome
  - Re-run a mini training step (1 epoch, 64 samples) after each market close
  - This is the "learn from mistakes" loop you asked about
  - Effort: 4 hours (after P7.2 is stable)

---

## Phase 8 — Real Money (Gate Checklist)
*Every item below must be checked before committing any real capital.*

### Pre-requisites (Hard Gates)
- [ ] 🔴 All Phase 0 items complete
- [ ] 🔴 All Phase 1 items complete
- [ ] 🔴 All Phase 2 items complete
- [ ] 🔴 P3.1 complete (3+ months uninterrupted paper trading)
- [ ] 🔴 P3.2 complete (paper vs Nifty comparison shows positive alpha)
- [ ] 🔴 P3.3 complete (actual outcomes are being logged and used)
- [ ] 🔴 P3.5 complete (readiness check passes)
- [ ] 🔴 P5.1 complete (you will be notified if something goes wrong)
- [ ] 🔴 P5.4 complete (ledger is backed up)

### Evidence Required
- [ ] 🔴 Rolling 30-day accuracy ≥ 51% for at least 2 consecutive months
- [ ] 🔴 Paper trading max drawdown stayed below 15% across the entire period
- [ ] 🔴 Paper P&L beats Nifty 50 buy-and-hold for the same period (net of costs)
- [ ] 🔴 No critical drift alert was left unresolved for >7 days
- [ ] 🔴 Kill switch was tested manually and confirmed to halt all orders

### Capital Sizing Rules (if all gates pass)
- Start with ≤ ₹10,000 regardless of confidence
- Do not increase capital until 60 more live trading days at the new capital level
- Never deploy more than you can afford to lose entirely
- Increase in steps: ₹10k → ₹25k → ₹50k → ₹1L — each step requires 60 days at prior level
- Never use borrowed money or funds needed within 3 years

---

## Backlog (Nice to Have, No Fixed Priority)

- [ ] 🟢 **B1** Broker integration — Zerodha Kite Connect API for live execution
  - Only after all 8 phases above are complete and ≥6 months of live paper trading
  - Keep paper trading running in parallel for at least 3 months after going live

- [ ] 🟢 **B2** Sector exposure limits
  - Max 40% allocation to any single sector (Banking, IT, Energy, etc.)
  - Requires a sector mapping for the 20 NSE tickers

- [ ] 🟢 **B3** Multi-asset expansion (Nifty Midcap, sectoral ETFs)
  - Only after the system is proven on the current 20 large-caps

- [ ] 🟢 **B4** Real-time intraday data (WebSocket feed)
  - Only relevant if moving to intraday trading — current architecture is end-of-day

- [ ] 🟢 **B5** MLflow model registry
  - Replace the file-based model registry with MLflow for better experiment tracking

^- [x] 🟢 **B6** Docker Compose for one-command startup
  - `docker compose up` starts FastAPI + React + scheduler in one shot

^- [x] 🟢 **B7** Telegram/WhatsApp bot for daily summary
  - End-of-day message: top 3 signals, paper portfolio P&L, any alerts
  - Python `python-telegram-bot` library, free

- [ ] 🟢 **B8** Walk-forward optimization (not just validation)
  - Currently walk-forward validates. Add a loop that also optimizes hyperparams within each window.
  - Caution: easy to overfit — use deflated Sharpe ratio (already implemented) to penalize

---

## Progress Tracker

| Phase | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| Phase 0 — Operational Wiring | ✅ Complete | 2026-07-01 | 2026-07-01 | All 5 items done |
| Phase 1 — State Machine | ✅ Complete | 2026-07-01 | 2026-08-04 | P1.1–P1.4 done |
| Phase 2 — Data Quality | ✅ Complete | 2026-08-04 | 2026-08-04 | P2.1–P2.4 done |
| Phase 3 — Paper Trading | ⏳ Time-gated | 2026-08-04 | — | P3.2, P3.3, P3.4, P3.5 done; P3.1 clock running |
| Phase 4 — Explainability | ✅ Complete | 2026-08-04 | 2026-08-04 | P4.1–P4.4 done |
| Phase 5 — Ops Hardening | ✅ Complete | 2026-08-04 | 2026-08-04 | P5.1–P5.5 done |
| Phase 6 — ML Improvements | 🔲 Not started | — | — | Needs live data |
| Phase 7 — RL | 🔲 Not started | — | — | Needs 6+ months data |
| Phase 8 — Real Money | 🔲 Not started | — | — | All gates required |

---

## The Core Loop You Asked About (RL That Learns From Mistakes)

```
Every trading day:
  ┌─────────────────────────────────────────────────────┐
  │  1. Fetch data + run 14 signal modules              │
  │  2. Meta-controller → BUY/SELL/HOLD + size          │
  │  3. Execute paper trade with stop-loss              │
  │  4. Log: signals + decision + trade to ledger       │
  │  5. Next day: log actual outcome                    │   ← P3.3
  │  6. Was the decision correct?                       │
  │     YES → reinforce the signal weights that agreed  │   ← already in meta-controller
  │     NO  → penalize the weights that were wrong      │   ← already in meta-controller
  │  7. Weekly: retrain models if drift detected        │   ← P6.2
  │  8. Monthly: full hyperparameter optimization       │   ← P6.3
  │  9. After 6 months: PPO agent learns from           │
  │     the full episode history in the ledger          │   ← P7.2 / P7.5
  └─────────────────────────────────────────────────────┘
```

The meta-controller (already built) is already a learning system — it re-weights which of the 14 signals predicted correctly. Full PPO-based RL is the upgrade that learns *how much* to bet, not just which direction. Both are valuable. The meta-controller runs from day 1; RL starts after 6 months of data.

---

*Last updated: 2026-08-04*
*Version: v0.0.10 → targeting v0.1.0 (real-money ready)*
