# TESTING.md

## How to run

```bash
# once: create .venv (Python 3.13 pinned via uv.lock)
uv sync

# full backend suite (97 files / ~1,430 test cases)
uv run pytest tests/ -v

# with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# lint
uv run ruff check .

# web (React/Vite app)
cd web && npm ci && npm run lint && npm run build
```

CI mirrors these: `.github/workflows/ci.yml` (lint job, test job, web job) on push/PR.

## What the suite actually protects

Priority suites — run these when touching anything related:

| Suite | Protects |
|---|---|
| `tests/test_safety_leakage.py`, `test_pit_discipline.py`, `test_point_in_time.py` | No look-ahead/leakage in features, labels, CV splits (purge/embargo), past-only neutralization |
| `tests/test_feature_parity.py` | Research features === runtime features (trainer↔runtime parity contract) |
| `tests/test_safety_execution.py` | ExecutionManager choke point: mode gates, staleness, idempotency, order budget, audit |
| `tests/test_safety_persistence.py` | Ledger schema v2, atomic JSON state, kill-switch / risk state durability |
| `tests/test_safety_api.py`, `test_api_auth.py` | API auth, rate limiting, fail-closed production guard |
| `tests/test_readiness_check.py` | System readiness gate used by automation |

`tests/conftest.py` sets `STOMAR_DATA_DIR` to a temp dir and isolates `AUDIT_DIR` per test — never bypass it; it exists to protect the tracked state files from test writes.

## Conventions

- Test names describe the risk, not the function (`test_live_requires_explicit_env_confirmation` style).
- Fixtures for synthetic OHLC/returns live in `conftest.py` (`sample_prices`, `sample_returns`, `sample_feature_df`).
- `scripts/dry_run_replay.py` & `scripts/dry_run_supervision.py` are supervised manual validations, not tests; expect them exercised against real bundles during go-live checks.
- `verify_system.py` is a smoke print-script using synthetic data — not part of CI.

## Coverage expectations

- Minimum target: keep the existing ~1,430-case suite green; add a red test first for any leakage/risk bug.
- New modules under `src/` need at least: one happy path, one failure path, and one anti-leakage/edge-case test when financial data is involved.
- No hard coverage gate is configured in CI yet (see ROADMAP.md).
