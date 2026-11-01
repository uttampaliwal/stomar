# ARCHITECTURE.md

Fast, factual entry point. Deep dives live in `docs/` — trust those for detail:

| Topic | Deep doc |
|---|---|
| Module map, layers, entry scripts | `docs/Architecture.md` |
| Data sources / DuckDB point-in-time store | `docs/Data-Pipeline.md` |
| 6-model ensemble, training gates | `docs/Model-Training.md` |
| Two-layer risk design | `docs/Risk-Engine.md` |
| API surface, auth, middleware order | `docs/API.md` |
| Paper trading flow | `docs/Paper-Trading.md` |
| Security model | `docs/Security-Model.md` |
| Go-lift gate / release discipline | `docs/Go-Live-Gate.md`, `docs/Master-Plan.md` |

## Stack snapshot (real, as of this writing)

- **Backend:** Python 3.13 (uv-managed), FastAPI + Pydantic v2 (`api/` — 26 routers), uvicorn/gunicorn, Single-process workers share file state (`data/`).
- **ML:** PyTorch (LSTM/GRU/Transformer) + XGBoost + LightGBM + CatBoost + scikit-learn meta-learner, FinBERT for sentiment, SHAP interpretability. 6 base models → stacked meta-controller over 14 signals.
- **Data:** yfinance (primary) → NSE archive fallback; DuckDB point-in-time store `data/market_data.db` (raw unadjusted OHLCV, `as_of` adjustments); parquet cache `data/*.parquet` (legacy path, still used by training); SQLite ledger `data/stomar.db` (SCHEMA_VERSION=2); JSON state files written atomically with filelock.
- **Trading core:** event-driven engine shared by backtest/paper/live; `ExecutionManager` is the single order choke point (mode gate → quote-staleness → risk → idempotency → reconciliation → JSONL audit).
- **Frontend:** React 19 + Vite 7 + Tailwind in `web/` (22 pages), talks to `/api` proxied in dev.
- **Ops:** Dockerfile (uv + Node 22 build → gunicorn), docker-compose (`api` + `scheduler` running weekday 15:45 IST), `auto_pipeline.py` automation with Telegram notify & stage checkpoints. Health `/api/health`, metrics `/api/metrics` (Prometheus).

## Layer map (dependency direction, top → bottom)

1. **Presentation** — `api/routers/*` (26 routers), `web/` SPA, auth/session + rate limit + audit middleware in `api/main.py`.
2. **Application** — `src/core/pipeline.RetrainingPipeline`, `src/signals/orchestrator.DailyOrchestrator`, `src/business automation` in `src/automation/daily_runner.py`, root scripts (`run_daily.py`, `auto_pipeline.py`).
3. **Domain** — `src/models/*` (model/trainer/ensemble/meta-controller/registry), `src/signals/*` (14 modules), `src/trading/*` (engine, execution_manager, paper_trader, portfolio, backtester, ledger, risk_controls, risk, optimizer).
4. **Infrastructure** — `src/data/*` (fetcher/sources/live/universes/resilience/store/feature_store), `src/brokers/*` (kite + dryrun + reconciler), `services/market_data.py`, `src/core/notifier.py` (Telegram).
5. **State** — DuckDB PIT store, SQLite ledger, JSON state, parquet cache, model bundles + manifests (`models/`).

Direction is one-way: routers/signals never reach brokers directly; every order goes through `src/trading/execution_manager.py`.

## Known architecture debts (load-bearing context for agents)

- Two data paths coexist: legacy fetch/parquet stack vs DuckDB PIT store. `services/market_data.py` glues them; PIT guarantee (`as_of`) only holds in the DuckDB path.
- `_FallbackSettings` in `src/core/settings.py` duplicates ~60 fields of the pydantic settings — update both or tests will catch drift late.
- Per-worker in-memory rate limiting and response cache in `api/main.py` (gunicorn `-w 4` → effective limits ×4).
- 16:10 vs 15:45, drawdown 8% vs 15%, order budget 5 vs 10 in older docs are resolved in `docs/Go-Live-Gate.md`; when a number is in dispute, count from code.
