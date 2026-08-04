# Security Model

StoMar treats the codebase, the paper-trading state, and (eventually) broker
credentials as attack surface. Design principle: **fail closed** — if any
verification cannot run, the sensitive operation is refused.

## API authentication (`api/main.py`)

- Every request to protected prefixes (`/api/paper-trading/`, `/api/automation/`, `/api/pipeline/`, `/api/ledger/`, `/api/risk-guard/`) requires the `X-API-Key` header.
- Constant-time comparison via `hmac.compare_digest`.
- Missing config on protected path → `503`; wrong key → `401`.
- In `production` the API **refuses to boot** without `STOMAR_API_KEY`.
- Frontend key injection: `web/src/lib/api-client.ts` sends `X-API-Key` from `window.__STOMAR_API_KEY__` (runtime-injected) or `VITE_STOMAR_API_KEY` (build-time) — the key never ships in the static JS bundle.

## Rate limiting & audit

- 60 requests / 60 s per IP on protected mutations → `429`.
- Every POST/PUT/PATCH/DELETE appended to `data/api_audit/mutations-YYYY-MM-DD.jsonl` (ts, method, path, status, client, request_id); audit failures never break the request.
- `X-Request-ID` correlation across middleware and logs.

## Artifact integrity (`src/models/artifacts.py`)

The most sensitive code path — deserializing pickles and torch checkpoints:

- Every artifact is saved with a sidecar manifest `{bundle}_manifest.json`
  (SHA-256 digest per file, plus `model_version`, `feature_schema_version`,
  `training_dataset_hash`).
- **Loading verifies the digest before any bytes reach `pickle`/`torch`**, and
  re-hashes immediately before deserialization; `torch.load` uses
  `weights_only=True`.
- Legacy unmanifested pickles are **refused** (migration via `scripts/migrate_legacy_models.py`).
- Path confinement: `ALLOWED_ROOTS = ("models", "models/production", "models/research")`; `safe_path_join` raises on any traversal outside the root.
- Atomic writes everywhere (`tmp` + `os.replace` + `fsync`).

## Risk override token

- Freeze state (`data/risk_guard_state.json`) can only be lifted with
  `STOMAR_RISK_OVERRIDE_TOKEN`, compared via `compare_digest` (`services/risk_guard.py`).
- If no token is configured, a freeze **cannot** be lifted — by design.
- Invalid attempts → `403` + warning log.
- The token is never logged (constant-time compare only).

## Kill switch

- `data/kill_switch.json` — manual/automatic halt; blocks all orders while
  active; state survives restarts.
- `{"active": true, "reason": "..."}` to halt; delete the file to resume.

## Secrets handling

- All configuration via `STOMAR_*` env vars in the git-ignored `.env`
  (`STOMAR_API_KEY`, `STOMAR_RISK_OVERRIDE_TOKEN`, Telegram/SMTP credentials,
  Kite keys, `GEMINI_API_KEY` for the wealth advisor).
- Broker adapter never logs credentials.
- `.env` is gitignored; `.env.example` documents every variable with defaults.

## Trading-mode separation

- Paper is the default mode, always.
- Live requires **four independent flags** (`STOMAR_LIVE_TRADING=true`,
  broker credentials, `STOMAR_LIVE_ACCOUNT_APPROVED=true`,
  `STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING`), re-verified on
  every execution — a single flipped flag is never enough.
- The paper-trading API refuses to run in live mode; `KiteLiveBroker`
  construction refuses without the gate.

## Input validation

- Pydantic models for request bodies (`ResumeRequest`, `EquityUpdate` with `gt=0`, …).
- Query params with regex/range constraints (ticker `^[A-Z0-9]{1,20}\.NS$`, history windows, intervals).
- Paper orders: ticker regex + quantity bounds (1–1,000,000), finite capital (1–100,000,000).
- Pipeline runs: ticker regex, max 50 tickers.
- Settings: typed `pydantic-settings` with `STOMAR_` prefix (unknown env vars are rejected — extra fields forbidden), with a non-pydantic fallback for constrained environments.

## Data integrity

- Ledger is SQLite with schema-version migration (`stomar.db`).
- Weekly automatic backup to `data/backups/` (4 retained), Sunday-only, from the auto-pipeline.
- 90-day retention archive to `data/archive/*.parquet` + VACUUM.
- Kill-switch and risk-guard state written atomically.

## Operational notes

- Telegram/email notifications carry no secrets (summaries only).
- Rate limiter is per-IP in-memory — sufficient for a self-hosted single-user system; replace with a shared store if multi-tenant hosting is ever attempted.
- Everything runs on your own hardware by default (see `DEPLOYMENT.md`); the Docker image is for self-hosting.
