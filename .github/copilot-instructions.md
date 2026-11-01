# Copilot instructions

Read `AGENTS.md` at the repo root first — it is the binding contract for AI-assisted work here.

Fast rules (full details in AGENTS.md):

- Python 3.13 via **uv**: `uv sync`, then `uv run pytest tests/ -v`, `uv run ruff check .`. Web: `cd web && npm run lint && npm run build`.
- Config keys are `STOMAR_*` env vars declared in `src/core/settings.py` and documented in `.env.example`. Never hardcode endpoints/URLs/keys.
- Never place orders outside `src/trading/execution_manager.py`; never bypass `src/trading/risk_controls.py` / `services/risk_guard.py`; never relax the live-trading gate in `src/core/trading_mode.py`.
- Preserve point-in-time discipline (`src/data/store.py` `as_of`, purge/embargo in `src/models/validation.py`). Anti-leakage tests must stay green.
- `data/` state files and `models/` bundles are intentionally git-tracked — don't add/delete them in feature commits.
- Docs under `docs/` are authoritative; `docs/Go-Live-Gate.md` wins conflicts. Update the older `README.md` counters only to verified values (6 models, 26 routers, 22 web pages, ~1,430 tests).
- Conventional commits; small scoped diffs; tests accompany behavior changes.
