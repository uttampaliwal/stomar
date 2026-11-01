# DEPLOYMENT.md

Operational runbook for deploying StoMar in Docker. A historical, machine-specific runbook is preserved at `docs/DEPLOYMENT.md`; deployment discipline is governed by `docs/Go-Live-Gate.md` (single source of truth: if docs disagree, that one wins).

## TL;DR

- **Mode:** paper trading is the safe default everywhere. Live trading requires explicit env gates and is covered by the separate go-live checklist.
- **Native and Docker deployments are mutually exclusive** — two schedulers double-run the pipeline and corrupt state. Pick one.

## Docker (recommended)

```bash
cp .env.example .env          # then fill required values
docker compose up -d --build
docker compose logs -f api    # watch boot + healthcheck
```

Services:

| Service | Image/command | Notes |
|---|---|---|
| `api` | gunicorn 4 × uvicorn workers, `0.0.0.0:8000`, healthcheck `/api/health` | `STOMAR_ENV=production`, `TZ=Asia/Kolkata`, named volumes `stomar-data:/app/data`, `stomar-models:/app/models`, env_file `.env` |
| `scheduler` | same image; installs cron, runs `auto_pipeline.py --paper --capital $PAPER_CAPITAL --boot` on weekdays 15:45 IST + boot catch-up | `depends_on: api (service_healthy)`; no healthcheck |

### Required config for a production container

- `.env` must set at minimum: `STOMAR_ENV=production` plus **`STOMAR_API_KEY` or `STOMAR_AUTH_PASSWORD`** (boot is fail-closed without one).
- Optional: `STOMAR_SCALE_MONEY`/risk thresholds, Telegram notifier (`STOMAR_TELEGRAM_*`) so the scheduler can summarize runs.
- Kite/live: `STOMAR_KITE_*` + the three live-gate variables — only when you actually intend live (see `src/core/trading_mode.py`).

### First-run model bootstrap (important)

`.dockerignore` excludes `data/` and `models/`, so a fresh container starts with **no trained bundles**. Choose one:

1. **Sync state from your keeper machine** into the volumes (intended path for multi-machine): run `scripts/sync_state.py` / copy `models/` + `data/stomar.db` + `data/paper_state.json` into the named volumes, or
2. **Cold-start train inside the container:** `docker compose exec scheduler uv run auto_pipeline.py --setup-run` (long; produces fresh bundles that the subsequent scheduled runs will keep warm).

Either way, verify: `curl localhost:8000/api/health` and check the dashboard loads, then confirm the ledger picked up state:

```bash
docker compose exec api uv run python -c "from src.trading.ledger import Ledger; l=Ledger(); print(l.stats())"
```

### Upgrades & rollback

```bash
git pull && docker compose build && docker compose up -d   # state lives in named volumes
docker compose down   # stop before failover to native scheduler
```

Data/model volumes persist across rebuilds. Never delete `stomar-data`/`stomar-models` without a backup — the ledger and paper state are the system's memory.

## Native (non-Docker)

Use `./start_dev.sh` for development; for an unattended box, install the OS scheduler directly:

```bash
uv sync
uv run schedule_pipeline.py --paper --capital 100000   # Windows: schtasks, Linux: cron/systemd timer
```

Keep one machine = one scheduler. `schedule_pipeline.py` is idempotent per trading day (ledger-aware "already ran" check + boot catch-up), but two installations will fight over the JSON state locks.

## Observability & failure recovery

- Pipeline failures: `data/monitoring/pipeline_failures.json` (surfaced by `auto_pipeline.py`, retried 60/300/900 s per stage).
- Telegram summary per run when `STOMAR_TELEGRAM_*` configured (`src/core/notifier.py`).
- Prometheus metrics at `/api/metrics`; health at `/api/health`.
- Ingest is circuit-broken (`src/data/resilience.py`: yf 5 fails/120s, NSE 3/90s) — flapping markets degrade gracefully to parquet/NSE archive fallbacks.
- Risk hardware stop: `data/kill_switch.json` freeze requires `override_token` (HMAC-compared in `services/risk_guard.py`) to resume.

## Scale & limits (be honest)

- Single-writer design: SQLite + JSON files with `FileLock` — **no multi-replica/HA**; do not run two `api` containers against one volume.
- Rate limiting & response caching are per-gunicorn-worker (4 workers → 4× limits); strict per-account limits live in the routes for this reason.
- LFS declared in `.gitattributes` for artifacts but blobs are currently stored raw; treat the git-tracked `models/` as a sync channel, not a backup (backup = volumes + private sync remote).
