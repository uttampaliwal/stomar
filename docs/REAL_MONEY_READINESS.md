# StoMar — Real-Money Trading Readiness Report

**Date:** 2026-08-01
**Suite:** 1045 tests passing (baseline 1019; +26 new safety regression tests)
**Interpreter:** `.venv-gcc/bin/python` (the original `.venv` is broken: Clang-built
CPython segfaults importing pandas/scipy/sklearn on this machine)

---

## 1. Verdict

**NOT ready for real money yet.** The critical safety layers (artifact
verification, mode gating, broker adapters, execution gate, kill switch,
leakage removal) are implemented and tested. What remains are operational
items: retraining models without the removed look-ahead feature, live
broker sandbox validation, API-key provisioning, and a supervised dry-run
period. With those closed, the gate described in §3 can be opened.

---

## 2. What was hardened (completed work)

### A. Unsafe deserialization — REMOVED
- Every runtime model load is now manifest-verified: `ArtifactBundle`
  (SHA-256 per file, `model_version`, `feature_schema_version`,
  `training_dataset_hash`, `created_at`), confined to `models/` via
  `safe_path_join`. Tampered/missing-manifest/traversal artifacts are
  refused with `ArtifactVerificationError`.
- `torch.load(..., weights_only=True)` everywhere; `joblib.load` only via
  verified bundles; legacy `pd.read_pickle` cache path removed from
  `multitimeframe.py`.
- All callers migrated (`model.py`, `meta_controller.py`, `ensemble.py`,
  `run_daily.py`, `auto_pipeline.py`, `daily_runner.py`, consensus API).
- `scripts/migrate_legacy_models.py` executed: 6 ticker bundles +
  1 meta-learner + 3 MTF pickles migrated to manifests/parquet; verified
  `load_models`, `promote_model` → production (registry entry created),
  `load_production_models`, `models_exist`.

### B. Paper/live separation — COMPLETE
- `src/core/trading_mode.py`: paper is the default and the only
  accidentally-reachable mode. Live requires **all** of:
  `STOMAR_LIVE_TRADING=true`, broker credentials, `STOMAR_LIVE_ACCOUNT_APPROVED=true`,
  `STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING`. The gate reads the
  live environment directly (not cached settings) and is re-checked on
  every execution.
- Mode banner/logging wired into `run_daily.py`, `auto_pipeline.py`,
  `daily_runner.py`, orchestrator, and the paper-trading API.

### C. Broker adapters — COMPLETE
- `src/brokers/`: `BrokerAdapter` ABC; `DryRunBroker` (deterministic
  fills, partial fills, rejections, cancellations, idempotent submits);
  `KiteLiveBroker` (idempotent client order IDs, timeouts, never logs
  credentials, refuses construction without the full gate);
  `OrderReconciler` (broker truth always wins; orders are never assumed
  filled because a request was sent).
- `src/trading/execution_manager.py`: single choke point for all order
  intents → mode → staleness → risk → submit → reconcile, with a JSONL
  audit log and daily order budget.
- **Live path is not yet validated against the broker sandbox** (§3).

### D. Risk controls — EXTENDED
- New hard limits wired in: liquidity (min daily traded value), expected
  slippage (bps), gap vs previous close, stale-quote freshness, daily
  order budget; plus the pre-existing position/exposure/drawdown/daily+weekly
  loss/consecutive-loss/correlation checks — all enforced in
  `ExecutionManager.execute()` before any broker contact.
- **Persistent kill switch** (`data/kill_switch.json`): survives restarts;
  only a manual `clear_kill_switch()` resumes trading.

### E. Look-ahead leakage — REMOVED
- `ichimoku_chikou` (future close shifted −26) removed from
  `trainer.FEATURE_COLS` and the factor registry; feature schema bumped
  to `"4"`.
- Inference now uses **the model's own stored feature schema**
  (`model_feature_cols`) instead of the current global one — this fixes
  both shape alignment with the trained scaler and prevents any future
  schema drift from feeding unseen columns.
- Belt-and-braces: `predict_ensemble` refuses/neutralizes any feature in
  `BANNED_LOOKAHEAD_FEATURES` and logs loudly.
- Walk-forward backtest: purge (60d) + embargo (20d) on the train/test
  boundary.

### F. API security — HARDENED
- Production startup refuses to run without `STOMAR_API_KEY`
  (fail-closed; previously no key = open endpoints).
- Protected prefixes are now fail-closed in every env (503 when
  unconfigured, 401 on wrong key, constant-time comparison).
- Per-IP rate limit (60/min) on protected mutating endpoints; JSONL audit
  log of every mutation in `data/api_audit/`.

---

## 3. Remaining before real money (blocking)

1. **Retrain all models without `ichimoku_chikou`** — IN PROGRESS.
   Deployed bundles were trained with a feature that leaked the future. They
   are neutralized at inference, but the honest baseline requires retraining
   with the v4 schema and re-promoting (bump `model_version`).
   - `trainer.py` now writes every bundle (including the tree-only fallback)
     through the manifest-aware `save_models` with `model_version="2"`;
     `save_models` tolerates missing DL models and only manifests artifacts
     that exist (regression-tested in `tests/test_safety_persistence.py`).
   - RELIANCE_NS retrained: 64 used features, model_version 2,
     feature_schema_version 4, no chikou, full DL bundle + meta-learner.
     Remaining 5 tickers (HDFCBANK_NS, HINDUNILVR_NS, ICICIBANK_NS,
     INFY_NS, TCS_NS) are being retrained; the first run hit yahoo
     throttling on the 5y fetch and was relaunched with primed caches.
   - After completion: verify each manifest, inference smoke test,
     `promote_model` to production, re-run the suite.
2. **Live broker sandbox validation** — IN PROGRESS.
   `KiteLiveBroker` instrument-token mapping is now implemented
   (`_instrument_token`, `resolve_symbol`; TTL-cached NSE snapshot,
   fail-closed on fetch failure/unknown symbol — 7 new regression tests in
   `tests/test_safety_execution.py`), and `validate_sandbox()` performs
   read-only checks (margins, mapping, positions; never places orders).
   `scripts/validate_kite_sandbox.py` runs it. Still required: install
   `kiteconnect` in `.venv-gcc`, run against the Zerodha sandbox with real
   credentials, then paper-run a full day through `ExecutionManager`.
3. **Operator credentials & checklist.** Set `STOMAR_API_KEY`, broker
   credentials, and the confirmation phrase deliberately; document who
   holds the kill-switch override.
4. **Supervised dry-run period** — TOOLING READY, PERIOD NOT RUN.
   `scripts/dry_run_supervision.py` runs the real execution path
   (ExecutionManager → DryRunBroker → reconciler) from today's
   orchestrator decisions, writes daily reports to `data/dry_run/`
   (order audit JSONL + `state.json` with positions, margin, risk status,
   gate state). `DryRunBroker` now honors `initial_cash`. Run it daily for
   ≥ 2 weeks; watch the audit log, reconciliations, and risk counters.

## 4. Recommended (non-blocking)

- Fold `PaperTrader` into the broker-backed path (remove the dual
  accounting layer).
- Add purge/embargo-aware backtest cost realism: commissions + slippage
  already modeled; add gap-to-open execution and partial-fill simulation
  (`BacktestExecutionSimulator` exists but is not wired into
  `walk_forward_split`).
- Wire `max_stale_quote_seconds`/liquidity fields from live quotes into
  the execution manager at runtime (gate supports it; data layer must
  supply them).
- DuckDB stale-lock recovery (`store.py::_recover_stale_lock`) is
  self-healing via `/proc` liveness checks; observed twice during suite
  runs — consider a dedicated unit test for the stale-lock path.

## 5. Operational knobs (all env-gated, paper by default)

```
STOMAR_LIVE_TRADING=true                 # master switch
STOMAR_KITE_API_KEY / STOMAR_KITE_ACCESS_TOKEN
STOMAR_LIVE_ACCOUNT_APPROVED=true
STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING
STOMAR_MAX_STALE_QUOTE_SECONDS=15        # freshness limit
STOMAR_MAX_DAILY_ORDERS=10               # daily order budget
STOMAR_MIN_DAILY_VOLUME_RS=1000000       # liquidity floor
STOMAR_MAX_EXPECTED_SLIPPAGE_BPS=30      # slippage ceiling
STOMAR_MAX_GAP_PCT=5                     # gap ceiling
STOMAR_API_KEY=...                       # required in production
```
