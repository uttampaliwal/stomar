# AGENTS.md

## Mission

StoMar is a trustworthy algorithmic trading **research and signal platform** for the Indian NSE market, with clearly separated research, validation, and production workflows. It must stay honest about what is proven and what is experimental — the Go-Live-Gate docs make an explicit NO-GO/GO decision point, and agent work must never quietly erode it.

Golden rule: if any other doc disagrees with `docs/Go-Live-Gate.md`, **that doc wins**.

## Non-negotiable rules

1. Do not add secrets, tokens, API keys, or credentials to source code. Config lives in `.env` (see `.env.example`) and is loaded via `src/core/settings.py`.
2. Do not use hardcoded production URLs, endpoints, or environment values. New external endpoints go through `src/core/settings.py` (`STOMAR_` prefix).
3. Do not add broad "cleanup" changes unrelated to the task. Keep diffs minimal, structured, and reversible.
4. Do not introduce new frameworks or major dependencies without justification (the ML stack — torch, XGBoost, LightGBM, CatBoost — and FastAPI are already fixed by the training/API contract).
5. Do not mix research code, production code, and deployment code in the same layer (see boundaries below).
6. Write tests for bug fixes and new features. Tests live in `tests/` (pytest, ~1,430 cases today) and run with `uv run pytest tests/ -v`.
7. Document behavior changes in `README.md` or the relevant `docs/*.md` when they affect project understanding.
8. Preserve the existing architecture unless the task explicitly requires a refactor.

## Project boundaries (hard)

- **Research scripts/notebooks are not production runtime.** Research lives in `scripts/`, `src/signals/alpha_research.py`, `src/trading/simulate.py`; production runtime is `api/` + `services/` + the `src/core/pipeline` execution path.
- **All market data ingestion must be validated and observed** (`src/data/data_validation.py`, telemetry into `data/monitoring/`).
- **Point-in-time discipline is mandatory** for anything consumed by models: use the DuckDB store (`src/data/store.py`) with `as_of=` or the purge/embargo helpers (`src/models/validation.py`, `src/trading/backtester.py`). Do not reintroduce look-ahead.
- **Models/datasets/artifacts are versioned** — manifests live next to weights, integrity is verified in `src/models/artifacts.py`, lifecycle in `src/models/model_registry.py`. Never hand-edit a `.pkl`/`.pt`/manifest.
- **Risk rules are code, not docs.** Pre-trade checks: `src/trading/risk_controls.py`; hard circuit-breakers: `services/risk_guard.py`; order choke point & audit: `src/trading/execution_manager.py`. Do not bypass any of these from signals/agents code.
- **Live trading is gated**: `src/core/trading_mode.py` requires explicit env opt-ins (`STOMAR_LIVE_TRADING`, `STOMAR_LIVE_ACCOUNT_APPROVED`, `STOMAR_LIVE_CONFIRMATION`). Never relax or shortcut this gate.

## Repo state facts agents must know

- Python 3.13 managed by **uv** (`uv sync`, `uv run ...`); web frontend is a React/Vite app in `web/` (Node ≥ 22, `npm run lint`, `npm run build`).
- CI: `.github/workflows/ci.yml` (lint + pytest + web build), CodeQL weekly, Dependabot.
- Runtime state files (`data/stomar.db`, `data/paper_state.json`) and trained bundles under `models/` are **intentionally git-tracked** to sync state between machines (see `.gitignore` exceptions) — this contradicts an older README claim; treat them as state, not code, and never casually delete/regenerate them in a commit.
- `models/_quarantine/` holds expired experiment artifacts — leave them alone unless explicitly asked.
- Trained-model counts/routers/pages floats often in docs; when updating docs, count reality (`git ls-files`, files in `api/routers/`, `web/src/pages/`, `def test_` in `tests/`).
- Data-refresh automation entry point: `src/automation/auto_pipeline.py` (holiday-aware gap detection, Telegram notify, per-stage retries). Scheduler installers: `src/automation/schedule_pipeline.py`.
- External endpoints baked in code (NSE FII/DII + option chain, Yahoo, Google News RSS, Screener.in, Telegram) are **best-effort public endpoints** with circuit breakers in `src/data/resilience.py`; changes here are API-drift-prone, so isolate and test.

## Coding standards

- Prefer clarity over cleverness; name things by domain intent.
- New code is type-hinted (`from __future__ import annotations`, dataclasses, generics — matching `src/core/*`, `src/data/store.py`, `src/brokers/*`).
- Keep functions small and testable; validate inputs and error paths; log operational events (never secrets).
- AI/ML numerical code: no silent NaN handling, no unendowed randomness in runtime paths (seeds come from settings, not literals).

## Commit guidelines

- Small, atomic changes; one theme per commit; clear conventional-style messages (`docs:`, `fix:`, `feat:`, `refactor:`, `chore:`).
- Include tests whenever behavior changes. Branch off `development`; PRs target `development`.

## AI workflow expectations

- Before architecture changes: read `docs/Architecture.md`, `docs/Data-Pipeline.md`, `docs/Model-Training.md`, `docs/Risk-Engine.md`, `docs/Go-Live-Gate.md`.
- Prefer incremental changes over rewrites; never refactor a signal/pipeline module into a "new system" on your own.
- If a feature is uncertain, document the assumption before implementing it.

## Required outputs for significant work

1. Summary of the change. 2. Why it is needed. 3. Risks/tradeoffs (especially: data leakage, risk-gate impact, backtest/live parity). 4. Files affected. 5. Validation steps actually run.

## Validation before completing anything

- `uv run ruff check .`
- `uv run pytest tests/ -v` (plus `uv run pytest web`-side: `cd web && npm run lint && npm run build` when web changed)
- No secrets or debug artifacts in the diff
- Docs updated when behavior or setup changed

## Escalation

Stop and propose before proceeding if a task touches: risk controls, the live-trading gate, the point-in-time store, model artifact manifests, auth/middleware in `api/main.py`, or anything in `docs/Go-Live-Gate.md` / `docs/Master-Plan.md`.
