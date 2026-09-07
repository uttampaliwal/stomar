# StoMar — Go-Live Gate (Canonical Thresholds)

> Single source of truth. Resolves prior doc contradictions (14d vs 60d, 15% vs 8%,
> 5 vs 10 orders, 15:45 vs 16:10, 58/60/64 feats). If any other doc disagrees, this wins.

## Canonical thresholds

- Paper: **≥60 business days** uninterrupted, one device runs/pushes per day.
- Accuracy: rolling **30d trade_accuracy ≥51% for 2 consecutive months**.
- Drawdown: paper max DD **≤15%**; hard **freeze at 8%** (freeze hits first, halt at 15%).
- Alpha: paper P&L **beats Nifty-50 buy-hold net of full NSE costs** same period.
- Models: all 20 tickers retrained on **schema v4 / 64 feats / model_version 2**, no `ichimoku_chikou`.
- Drift: no critical drift unresolved **>7 days**.
- Orders: **≤10/day** budget (atomic, IST day boundary); orchestrator target ≤5 entries/day.
- Kill-switch: manually tested halt + `clear_kill_switch()` resume, holder documented.
- Timers: **15:45 IST daily run** (single scheduler) + **16:10 watchdog read-only**;
  never two writers same day. Mutually exclusive deploy paths (native OR docker).
- Live gate env (all required, re-checked every execution):
  `STOMAR_LIVE_TRADING=true`, `STOMAR_KITE_API_KEY`, `STOMAR_KITE_ACCESS_TOKEN`,
  `STOMAR_LIVE_ACCOUNT_APPROVED=true`, `STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING`.

## Graduation ladders (A-I + capital)

A-safety, B-data, C-reconciliation 100%, D-beats-challenger, E-risk-respected,
F-60d, G-paper/live-divergence under threshold, H-no-P0/P1, I-human approval.
Then ₹10k → review → ₹25k → review → ₹50k → review → ₹1L + auto-rollback.
Never scale on profit alone. 60d+ clock = minimum observation; week-11 fail restarts window.

## Baselines (per MASTER-PLAN O1b/O3)

Every promotion and O1b must beat: buy-hold (`signals/scenarios.py:24`,
`benchmarks.py:15`), logistic/challenger, momentum — point-in-time OOS, after
costs+slippage, purge/embargo + walk-forward.
