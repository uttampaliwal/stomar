# DEVELOPMENT.md

## First-time setup (native)

```bash
# prerequisites: Python 3.13 via uv, Node >= 22 (web/.nvmrc)
./start_dev.sh      # or .\start_dev.bat on Windows
```

`start_dev.sh` will: install uv if missing → `uv sync` → `npm ci` in `web/` if Vite is missing → start the FastAPI backend on `127.0.0.1:8000` and the Vite dev server on `:5173` with `/api` proxied.

- API docs: `http://localhost:8000/docs` · health: `/api/health` · Prometheus: `/api/metrics`
- Backend logging config: `src/core/logging_config.py`

## Manual commands

```bash
uv sync                                  # install deps exactly as uv.lock pins them
uv run uvicorn api.main:app --reload --port 8000
cd web && npx vite --host                # frontend dev
uv run pytest tests/ -v
```

## Configuration model

Single source of truth: `src/core/settings.py` (**pydantic-settings**, `env_prefix="STOMAR_"`, reads root `.env`). Every knob is documented in the 111-line `.env.example`.

Facts that save you an hour:

- A **fallback settings class** duplicates all fields (`_FallbackSettings` in the same file) for when `pydantic-settings` import fails — keep both in sync.
- `STOMAR_DATA_DIR` is also read directly in `src/core/constants.py`.
- **Production is fail-closed**: `STOMAR_ENV=production` without `STOMAR_API_KEY` or `STOMAR_AUTH_PASSWORD` refuses to boot (`api/main.py`).
- Live trading additionally requires `STOMAR_LIVE_TRADING`, `STOMAR_LIVE_ACCOUNT_APPROVED`, `STOMAR_LIVE_CONFIRMATION` (phrase-exact) — see `src/core/trading_mode.py`. Paper mode is the default and requires none of these.
- Docker compose injects `STOMAR_ENV=production`, `TZ=Asia/Kolkata`, `PAPER_CAPITAL` from `.env`.

## Daily-use entry points

| Command | What it does |
|---|---|
| `uv run auto_pipeline.py` | The real automation: NSE-holiday-aware gap detection → backfill → train → run daily → telegram summary → stage retries/checkpoints |
| `uv run run_daily.py` | One research day: 14 signals → meta-controller → paper trades; `--backfill`, `--paper-trade`, `--dry-run` |
| `uv run run_pipeline.py` | Per-ticker retrain with OOS gates (`min_oos_accuracy`, `min_oos_sharpe`, drawdown cap) or `--train-all` |
| `uv run schedule_pipeline.py` | Installs cross-platform scheduler (Task Scheduler / cron / systemd) for weekday 15:45 IST |
| `uv run verify_system.py` | Smoke check on synthetic data (prints; no asserts) |
| `scripts/` | Edge gate, dry-run replay/supervision, meta refresh, state sync, watchdog, Kite sandbox validation |

Note: `run_pipeline.py --paper` currently only fetches a bar and saves state (vestigial); use `auto_pipeline.py` or `run_daily.py --paper-trade` for real paper activity.

## Runtime state & where it lives

- `data/stomar.db` — SQLite ledger (schema v2 + v1 migration)
- `data/market_data.db` — DuckDB point-in-time store
- `data/paper_state.json`, `data/kill_switch.json`, `data/risk_state.json`, `data/mf_state.json` — atomically written, filelock-guarded
- `data/monitoring/` — operational telemetry (data fixes, pipeline failures)
- `models/{TICKER}_NS_*` bundles + `_manifest.json` — verified on load (`src/models/artifacts.py`), lifecycle in `src/models/model_registry.py`

These state/artifact paths are **git-tracked on purpose** (multi-machine sync via `scripts/sync_state.py`). Do not commit half-trained experiments; quarantine leftovers belong in `models/_quarantine/`.

## Quality gates before you push

```bash
uv run ruff check . && uv run pytest tests/ -q
cd web && npm run lint && npm run build
```

Branching: work off `development`, PR back to `development`. CI (lint+test+web) must be green. See `CONTRIBUTING.md` for the full PR checklist.

## Recommended reading

`docs/Architecture.md` → `docs/Data-Pipeline.md` → `docs/Model-Training.md` → `docs/Risk-Engine.md` → `docs/API.md` → `docs/Paper-Trading.md` → `docs/Security-Model.md` → `docs/Go-Live-Gate.md`.
