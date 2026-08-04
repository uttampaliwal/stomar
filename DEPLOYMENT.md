# StoMar Deployment Runbook — Start the Paper-Trading Clock (P3.1)

Purpose: put StoMar on an always-on machine and keep it running uninterrupted
for 60+ trading days. Every command below was exercised during development;
the two deployment paths are **mutually exclusive — pick one** (running both
would double the daily run).

---

## Prerequisites (both paths)

- An always-on machine (Linux preferred; a $5 VPS or a Raspberry Pi 4/5 works —
  models run fine on CPU, the image/venv is now ~2.4GB with CPU-only torch).
- Git + network access to PyPI, download.pytorch.org, api.telegram.org.
- A `.env` file from `.env.example`:
  - `STOMAR_API_KEY` (required — write-protects the API)
  - `STOMAR_TELEGRAM_BOT_TOKEN` + `STOMAR_TELEGRAM_CHAT_ID` **or** SMTP vars
    (required for P5.1 alerts and the daily summary)
  - `PAPER_CAPITAL` (optional, default 200000)

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # → STOMAR_API_KEY
```

---

## Path A — Native install (recommended on Linux)

```bash
# 1. Dependencies (Debian/Ubuntu)
sudo apt install -y git curl cron

# 2. uv + repo
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone <your-repo-url> stomar && cd stomar
cp .env.example .env            # edit: API key, Telegram/SMTP, PAPER_CAPITAL

# 3. One-time build (CPU-only torch, ~2.4GB; first run ~5-10 min)
uv sync --frozen

# 4. First-time model training (only needed once; ~30-60 min CPU)
uv run run_pipeline.py --train-all
uv run run_daily.py --backfill --days 252 && uv run run_daily.py --train-meta

# 5. Install the scheduler — paper trading, ₹2L capital, boot catch-up
uv run schedule_pipeline.py --paper --capital 200000 --boot

# 6. Verify
uv run schedule_pipeline.py --status          # two tasks installed
uv run schedule_pipeline.py --run-now         # smoke test — watch logs:
tail -f data/pipeline_logs/daily_$(date +%Y%m%d).log

# 7. Serve the dashboard (optional — API + React SPA on :8000)
nohup .venv/bin/gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:8000 --timeout 120 >> data/pipeline_logs/api.log 2>&1 &
```

## Path B — Docker (one-command)

```bash
git clone <your-repo-url> stomar && cd stomar
cp .env.example .env            # edit: API key, Telegram/SMTP, PAPER_CAPITAL
docker compose up -d --build

docker compose ps                                       # api + scheduler healthy
curl -H "x-api-key: $STOMAR_API_KEY" http://localhost:8000/api/health
docker compose exec scheduler crontab -l                # daily + boot tasks
docker compose logs -f scheduler                        # first run output
```

First container start trains models lazily on the first daily run; to warm up
immediately: `docker compose exec scheduler python run_pipeline.py --train-all`.

---

## Ongoing operations (the 60-day clock)

| When | Check |
|---|---|
| Day 1-3 | Daily 15:45 IST run fires (log + Telegram summary arrives); kill switch reachable |
| Weekly | Models fresh (`Model age` on Monitoring page); ledger backup exists in `data/backups/` |
| Monthly | Paper P&L vs Nifty comparison (`/api/benchmark/compare`) still positive |
| Any time | Drawdown alert triggers at 10% (Paper Trading page + Telegram) |

Critical alert wiring (already automatic): pipeline failure, kill switch,
drift, drawdown breach → `notify()` → Telegram + email.

Pause everything: `touch data/kill_switch.json` with `{"halted": true}`;
remove scheduler tasks: `uv run schedule_pipeline.py --remove` (native) or
`docker compose stop scheduler` (Docker).

## Rollback

```bash
# Native
uv run schedule_pipeline.py --remove
# Docker
docker compose down            # volumes survive by default
docker compose down -v         # ...or destroy data + models too
```
