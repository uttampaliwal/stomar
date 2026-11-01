# Roadmap

This file is an index. The authoritative, long-form checklist lives in **`docs/ROADMAP.md`**; staging gates in **`docs/Live-Trading-Roadmap.md`**; the binding decision point in **`docs/Go-Live-Gate.md`**; and the recorded status in `docs/GATE-VERDICT.md` and `docs/REAL_MONEY_READINESS.md`.

## Current status snapshot (2026-11)

- **Verdict:** not live — edge-gate records accuracy 0.59–0.70 but post-cost returns below buy-and-hold; REAL-MONEY readiness audit is NO-GO.
- **Build contract:** frozen in `docs/MASTER-PLAN.md`; fix cadence tracked as one PR per gate item.

## Engineering debt queue (root-level, curated)

- [ ] Migrate `models/` artifacts + `data/` state to Git LFS or a private sync remote; reconcile docs that claim these paths are gitignored. *(Decision recorded 2026-11: `.gitattributes` no longer declares `filter=lfs` — LFS was deferred while state stays regular git-tracked blobs, and the unregistered filters would have failed add/checkout; file now carries plain `-text` binary markers with a re-enable recipe in the file itself.)*
- [x] Enforce `_FallbackSettings` ←→ pydantic `Settings` parity in CI (`tests/test_settings_parity.py`; also fixed the real drift it caught: 7 notification fields were missing from the fallback).
- [ ] Unify the parquet cache path and the DuckDB point-in-time store behind `services/market_data.py` (or retire the legacy path).
- [x] Fold root scripts (`auto_pipeline.py`, `run_daily.py`, `schedule_pipeline.py`, `run_pipeline.py`) into `src/automation/` — scheduler paths root-anchored, `docker-compose.yml`/cron commands/tests/web hints/doc references all re-pointed (see the package `__init__` docstring for the run-as-module vs run-as-file contract).
- [x] Replace/remove vestigial `run_pipeline.py --paper` dead mode (removed — the only paper-trading surface is the orchestrator + `src/automation/run_daily.py --paper-trade`); promote `verify_system.py` → real pytest smoke suite `tests/test_system_smoke.py` (the print-script's 200-row fixture never cleared the 500-sample training gate — the new one does, with assertions).
- [ ] Move in-memory rate limit / response cache to a shared store or document per-worker multiplication.
- [ ] Archive `models/_quarantine/` experiment bundles out of git.
- [ ] Re-check NSE bhavcopy archive fallback (`src/data/data_sources.py`) for live URL validity — source console discontinued.
- [ ] Coverage gate: add `pytest-cov` threshold in CI once baseline suite stabilizes.
- [ ] Pr-templates/issue-templates, release notes automation for tagged versions.
