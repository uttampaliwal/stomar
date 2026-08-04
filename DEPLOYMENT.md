# StoMar Deployment Runbook — Start the Paper-Trading Clock (P3.1)

Purpose: put StoMar on a machine that is on daily and keep the paper-trading
clock running uninterrupted for 60+ trading days. Because the pipeline is
**idempotent** (gap detection → backfill → daily run → paper trades) and every
scheduled run carries a boot catch-up, the same repo can live on 2-3 devices
(laptop, desktop, GPU machine) — **whichever one you switch on that day picks
up where the others left off**. Each device runs its own local copy; the
Telegram summary identifies which machine ran.

Every command below was exercised during development; the two deployment
paths are **mutually exclusive — pick one** (running both would double the
daily run).

---

## Prerequisites (both paths)

- A machine that is on at least once daily (Linux preferred; models run fine
  on CPU — the venv is ~2.4GB with CPU-only torch).
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

### Get your Telegram chat id (one-time)

1. In Telegram, search your bot (e.g. `StomarDailyBot`), press **Start**,
   and send any message (`hi`).
2. Fetch your numeric id:

```bash
curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates?limit=5" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['result'][-1]['message']['chat']['id'])"
```

3. Set `STOMAR_TELEGRAM_CHAT_ID=<that number>` in `.env`.

---

## Path A — Native install (recommended on Linux)

```bash
# 1. Dependencies (Debian/Ubuntu)
sudo apt install -y git curl cron

# 2. uv + repo
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone <your-repo-url> stomar && cd stomar
cp .env.example .env            # edit: API key, Telegram chat id, PAPER_CAPITAL

# 3. One-time build (CPU-only torch, ~2.4GB; first run ~5-10 min)
uv sync --frozen

# 4. ONE-COMMAND DEVICE SETUP — verifies env, tests Telegram, checks GPU,
#    installs the daily task + boot catch-up, then runs the full pipeline.
uv run auto_pipeline.py --setup-run

# 5. Verify
uv run schedule_pipeline.py --status
tail -f data/pipeline_logs/daily_$(date +%Y%m%d).log
```

That is everything: the daily task runs Mon-Fri 16:00, and the boot catch-up
(crontab `@reboot`, or a systemd `Persistent=true` timer when cron is absent)
replays any missed days the moment the machine is switched on. Repeat steps
2-4 once on every device — whichever you open that day handles the run.

### Optional: GPU acceleration (RTX 50-series etc.)

The RTX 5060 (and any NVIDIA GPU) is detected by `--setup` but only used if
CUDA torch is installed. On a GPU machine only:

```bash
bash scripts/setup_gpu.sh          # or scripts\setup_gpu.bat on Windows
# swaps CPU torch → cu128 build; verify:
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Meta-controller training (the only heavy step) then runs on the GPU. Models
are small, so this is a convenience, not a requirement.

# 5. Serve the dashboard (optional — API + React SPA on :8000)
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
| Day 1-3 | Daily 16:00 IST run fires (log + Telegram summary arrives); kill switch reachable |
| Weekly | Models fresh (`Model age` on Monitoring page); ledger backup exists in `data/backups/` |
| Monthly | Paper P&L vs Nifty comparison (`/api/benchmark/compare`) still positive |
| Any time | Drawdown alert triggers at 10% (Paper Trading page + Telegram) |

Multi-device: because every device schedules the same idempotent pipeline +
boot catch-up, a day is covered as long as any one device is switched on.
The Telegram summary header includes the machine name (`NewNest`, …), so you
can see which device ran. Runs on different devices use their local ledger —
resolve at month end with `docker compose exec scheduler` / the benchmark API
from the primary device.

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
