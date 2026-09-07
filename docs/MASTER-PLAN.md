# StoMar — Frozen Master Plan (Build-Mode Contract)

> Status: FROZEN. Every build-mode step is checked against this file.
> Goal: deterministic financial/portfolio OS with trading as one component —
> best return per buck, minimal risk, with reasoning — across Stocks/SIP/Bonds/
> ETFs/MFs. Free/open-source only. Paper/live segregation always.
> Verdict carried over: NOT ready for real money (NO-GO live, conditional advisory).

## Global invariants (never violate)

1. Paper is default and only accidentally-reachable mode. Live needs ALL of
   `STOMAR_LIVE_TRADING=true` + Kite creds + `STOMAR_LIVE_ACCOUNT_APPROVED=true`
   + `STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING` (re-checked every execution).
2. SOTA = continuous challenger-beating under realism, NOT model count.
   No candidate promotes without beating buy-hold + logistic/challenger + momentum
   OOS net of costs/slippage/turnover/tax with purge/embargo + walk-forward.
3. LLM explains/retrieves/proposes; deterministic engines decide
   (router → portfolio → risk → tax → suitability → decision → LLM + evidence).
   LLM never invents a price/holding/return/tax number.
4. Optimizer → constraints → execution. Never model → BUY.
5. Free-only software/data for development; provider-neutral `DataAdapter`
   (Yahoo/NSE-public/MF-NAV/G-Sec today; licensed/broker/fundamentals/news as plug-ins).
6. SEBI suitability/provenance is engineering (risk-profile → IPS → constraints),
   not docs. Every recommendation carries an evidence bundle
   (market/statements/filings/news/model/portfolio/risk/tax/timestamp).

## Coverage sequencing (locks nothing buggy)

- O0: tests ONLY for paths O1 does not touch (CI, sync, API shape, secure_io).
- O1: correctness tests written WITH each fix.
- 80% coverage lands as a byproduct of O1, not a gate ahead of it.

## O1b Edge Gate (HARD) + pre-committed FAIL branch

Gate: strategy > buy-hold AND logistic/challenger AND momentum on point-in-time
OOS data after realistic costs+slippage, statistically + economically significant.
- PASS → O2→O3→O4→O5/O6 while 60d clock runs.
- FAIL → F1/F2/F3 (time-boxed, chosen objectively now):
  - F1 Signal loop (2w): features → target variant (e.g. 3d forward vs next-day) → re-run O1b once. 1 retry max.
  - F2 Window extension (2w): larger/regime-split OOS; still fail → F3.
  - F3 Repurpose: freeze alpha; ship as execution/portfolio/research infra. No O5/O6/O7-rec.
- No second retry without a new hypothesis doc. No sunk-cost override.

## Objectives + timings

- O0 Freeze+contract (3-5d): clean tree, quarantine `models/T_*`, write
  `docs/Go-Live-Gate.md` (canonical: 60 biz days, 30d acc ≥51% x2mo, DD ≤15%
  with 8% freeze, ≤10 orders/day, single daily timer), non-O1 tests only.
- O1 Correctness (1-2w): flat-scoring exclusion for SELL (`ledger.py:233-239`),
  unified state via `secure_io.atomic_write_*` (fix `paper_trader.py:167,471,523,549`
  initial_capital/consecutive_losses/halted/save-bypass/fsync), hash linkage through
  EVALUATE (`pipeline.py:266-323`), active-only ensemble (meta + backtest paths,
  `ensemble.py:318,487`), Sharpe/slippage-sign/VaR-None/units, naive-ts→IST.
- O1b Edge Gate (3-5d): as above.
- O2 Repo/CI/repro (3-4d): LFS working + `stomar-state-private` split
  (ledger/paper/models/backups out of code repo), gitleaks, `ruff format`,
  CI concurrency, docs recount, `uv.lock --frozen` SBOM, fresh-clone restore <10min.
- O3 Light governance, solo-appropriate (3-5d): challenger-must-beat, validation note,
  model inventory + ownership, revalidation cadence. No role theater.
- O4 Data + Point-in-Time Provenance (1-2w): provider-neutral adapters, publication/
  effective timestamps, corp-action effective dates, survivorship/delisted/symbol-change
  control, tz normalization, revision tracking, evidence bundles, no-lookahead audit.
- O5 Model Lab, gated on O1b PASS (2-3w): baselines first, Optuna `tuner.py`, isotonic/
  temperature calibration, split-conformal (`mapie`), Chronos-2/TimesFM candidates,
  Boruta/mRMR prune, HRP already in O6, MLflow local, no auto-promote.
- O6 Portfolio Intelligence, gated on O1b PASS (2-3w): MVO/BL/HRP/min-var/max-div/
  CVaR/vol-target/regime/tax-aware; factor decomposition (beta/sector/size/value/
  momentum/quality/vol/liq/concentration/mVaR/cVaR/TE/active-share); historical
  (COVID/2008/2020/2022/INR-crude/election/VIX) + synthetic (NIFTY-20%, IT-30%,
  +150bps, USDINR+10%, crude+40%, VIX×2, liquidity) stress.
- O7 Wealth OS, capability-separated A-accounting/B-recommendation/C-execution/
  D-external (3-4w): equities/ETF/MF/G-Sec/bonds/REIT/InvIT/gold/FD/PPF-NPS,
  multi-account/folio/demat, dividends/splits/bonus/merger/demerger/rights/buyback,
  FIFO cost-basis, realized/unrealized, XIRR/TWRR.
- O8 Financial Planning (2w): FY-versioned STCG/LTCG/dividend/FIFO, tax-loss harvesting,
  SIP tax lots, corp-action basis adj, annual tax report; SIP/retirement Monte-Carlo
  (goal probability, required vs trajectory SIP); emergency-fund + liquidity +
  insurance/debt-gap; risk-profile → IPS → constraints.
- O9 Financial Copilot (2w): LLM router + RAG over ledger/provenance + tools,
  citations/evidence, suitability guard; LLM explains/proposes, engines decide.
- O10 Execution (1w): gap-to-open + partial-fill wiring, paper/live parity, pre-trade
  checks (liquidity/slippage/gap/staleness/budget), reconciliation = 100%.
- O11 Monitoring/DR (1w): KS drift + perf/pred/freshness, P0/P1 incidents + post-mortems,
  weekly `data/backups/` + restore test, Telegram+SMTP alerts.
- O12 Graduation: capital + evidence ladder (≥60 trading days ≈12 cal wks overlapping
  O5-O11; 12w = minimum observation, week-11 fail restarts window):
  Gates A-safety/B-data/C-recon/D-challenger/E-risk/F-60d/G-paper-live-divergence/
  H-no-P0-P1/I-human-approval → ₹10k→review→₹25k→review→₹50k→review→₹1L + auto-rollback.
  Never scale on profit alone. Start ≤₹10k regardless of confidence.

## Order

O0 → O1 → O1b → (PASS ? O2/O3/O4 → O5-O11 parallel with 60d clock → O12 : F-branch).
Total: ~14-18w engineering + 12w clock overlapped.
