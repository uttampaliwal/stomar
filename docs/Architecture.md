# Architecture

StoMar is a self-contained paper-trading research system for NSE India large-caps.
It fetches market data, trains an ensemble of ML models, produces daily
buy/hold/sell decisions, executes them on a simulated portfolio, and tracks
performance in a durable ledger — all without any broker or real money.

## High-level flow

```
yfinance / NSE archive ──► DuckDB store (point-in-time, raw bars)
        │
        ▼
Features (SOTA factor pipeline, ~60 factors, feature-hash versioned)
        │
        ▼
Models (LSTM, GRU, Transformer, XGBoost, LightGBM, CatBoost)
        │
        ▼
Ensemble (meta-learner → regime-conditional weights → equal weights)
        │
        ▼
DailyOrchestrator — 14 signal modules → meta-controller decision
        │
        ▼
ExecutionManager gate → PaperTrader (simulated fills, NSE costs, tax)
        │
        ▼
Ledger (SQLite stomar.db — decisions, trades, outcomes, snapshots)
        │
        ▼
Benchmarks / calibration / readiness gate / monitoring / Telegram summary
```

## Module map

| Directory | Responsibility |
|---|---|
| `src/core` | Settings, constants, NSE trading calendar, pipeline state/checkpoints, backfill, notifier (Telegram/email), secure I/O, readiness gate, trading mode |
| `src/data` | Data fetching (yfinance + NSE archive fallback), DuckDB point-in-time store, features, feature store/versioning, validation, retention, live quotes, universes, circuit breakers |
| `src/models` | LSTM/GRU/Transformer + tree models, walk-forward trainer, ensemble, meta-controller, calibration, manifest-verified artifacts, model registry |
| `src/signals` | DailyOrchestrator + 14 signal modules (sentiment, flow, PCR, MTF, regime, VaR, volatility, fundamentals, ensemble…), monitoring/drift, benchmarks, interpretability |
| `src/trading` | Paper trader, execution engine, risk controls, execution manager, ledger, portfolio/tax, backtester, optimizer |
| `src/brokers` | Broker adapter interface, dry-run broker, Kite (live, gated), order reconciler |
| `services/` | Risk guard (hardware circuit breakers), market data service |
| `api/` | FastAPI application — 24 routers under `/api` (73 endpoints) |
| `web/` | React + Vite dashboard (21 pages) talking to the API via `/api` |
| `scripts/` | One-off ops: dry-run supervision, kite sandbox validation, legacy model migration, GPU setup |

## Entry scripts

- **`run_pipeline.py`** — one-off retraining pipeline: FETCH → VALIDATE → FEATURES → TRAIN → EVALUATE → PROMOTE, gated by min OOS accuracy 50% / Sharpe 0.0 / max drawdown 30%.
- **`run_daily.py`** — single-day loop: resolve outcomes → signals → orchestrator → paper trades. Used for one-off runs and as the legacy scheduler target.
- **`auto_pipeline.py`** — **the recommended entry point.** Detects missed trading days → backfills (≤30 days, NSE holiday aware) → trains meta-controller if needed → daily orchestrator → paper trades → health sweep → weekly ledger backup → Telegram summary. Idempotent ("already ran today" detection); resumable via checkpoints; per-stage retries (1/5/15 min backoff); graceful shutdown.
- **`schedule_pipeline.py`** — installs the scheduled task: Windows Task Scheduler, Linux crontab, or a systemd user timer (`Persistent=true` = boot catch-up). All scheduled entries run `auto_pipeline.py` so any invocation — at the 15:45 slot or right after boot — covers backfill + today + paper trades.
- **`start_dev.sh` / `start_dev.bat`** — dev environment bootstrap (API + web).

## Scheduling and multi-device

Every scheduled invocation runs the full idempotent `auto_pipeline.py`:

- **Daily slot** (Mon–Fri 15:45 IST) — normal path.
- **Boot catch-up** — Windows Task Scheduler `onstart` (5 min delay), crontab `@reboot` (5 min delay), or systemd timer `Persistent=true` (fires after boot if a run was missed). Because backfill is gap-based, a device that boots at 10:00 runs everything due for that day immediately.
- **Multi-device** — the same repo on 2–3 machines; whichever is switched on that day covers it. Each device has its own local ledger/paper account; the Telegram summary header names the machine (`platform.node()`).

Non-trading days (weekends/holidays via `src/core/calendar.py`) skip the daily and paper stages — backfill still runs so nothing is ever lost.

## Key invariants

- **Point-in-time discipline** — only data `<= as_of` is used; corporate actions applied at query time; sentiment/PCR assigned to the most recent row only; FII/DII shifted +1 day; banned look-ahead features neutralized at inference.
- **Idempotence** — backfill skips ledger-existing dates; scheduler runs can safely overlap; paper trades keyed to decisions.
- **Fail-closed security** — API key enforced on protected routes (constant-time compare), artifact manifests verified before any deserialization, live trading gated by 4 independent flags.
- **Self-healing** — circuit breakers per data source, per-stage retries, checkpoint resume, scheduler auto-install on first run.

## Monitoring

- `data/monitoring/daily_health.json` — freshness per ticker, model age, paper days, kill-switch state (written by every auto-pipeline run).
- `data/monitoring/pipeline_failures.json` — last 24 h failures → red banner in the UI.
- `data/monitoring/retrain_triggers.json` — automatic retraining requests from drift detection.
- Telegram/email notifications for critical events (pipeline failure, kill switch, drawdown breach) and the daily summary.
