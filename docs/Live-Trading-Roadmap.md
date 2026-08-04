# Live Trading Roadmap

## Phase status

| Phase | Status |
|---|---|
| 0 Operational Wiring | ✅ Complete (2026-07-01) |
| 1 State Machine | ✅ Complete (2026-08-04) |
| 2 Data Quality | ✅ Complete (2026-08-04) |
| 3 Paper Trading | ⏳ Time-gated — P3.1 clock running since 2026-08-04, 0/60 trading days |
| 4 Explainability | ✅ Complete |
| 5 Ops Hardening | ✅ Complete |
| 6 ML Improvements | 🔲 Not started (needs live data) |
| 7 RL | 🔲 Not started (needs 6+ months) |
| 8 Real Money | 🔲 Not started (all gates required) |

Full details in `ROADMAP.md`; the live-readiness audit is in
`docs/REAL_MONEY_READINESS.md`.

## Trading mode gate (`src/core/trading_mode.py`)

The system defaults to **paper** mode, always. Live mode requires **all four**
of these, re-read from the environment on every execution (never cached):

1. `STOMAR_LIVE_TRADING=true`
2. Broker credentials (`STOMAR_KITE_API_KEY`, `STOMAR_KITE_ACCESS_TOKEN`)
3. `STOMAR_LIVE_ACCOUNT_APPROVED=true`
4. `STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING`

`require_live_allowed()` raises `LiveTradingNotEnabledError` listing missing
requirements. `get_broker()` returns dry-run unless `force_live` **and** the
gate passes; `KiteLiveBroker` construction itself refuses without the gate.

## Broker layer (`src/brokers/`)

- `BrokerAdapter` ABC; `BrokerOrder` with `UNCONFIRMED` status; idempotent
  submits keyed by `client_order_id` (`ord-<uuid12>`).
- `KiteLiveBroker` — instrument-token mapping TTL-cached (3600 s) and
  fail-closed; 10 s network timeouts; **credentials never logged**.
- `OrderReconciler` — broker truth always wins; orders never assumed filled.
- Only `"kite"` is supported today.

## What must happen before real money (`REAL_MONEY_READINESS.md`)

Already done: manifest-verified artifact bundles, paper/live separation gate,
broker adapters + ExecutionManager choke point, extended risk controls +
persistent kill switch, look-ahead leakage removed (ichimoku chikou, feature
schema v4, purge/embargo), fail-closed API auth.

**Blocking items:**

1. Retrain all models without the banned `ichimoku_chikou` feature —
   RELIANCE_NS done; 5 tickers remain (HDFCBANK, HINDUNILVR, ICICIBANK, INFY, TCS).
2. Live broker sandbox validation — `scripts/validate_kite_sandbox.py`
   (read-only; needs `kiteconnect` installed).
3. Operator credentials + pre-trade checklist.
4. Supervised dry-run ≥ 2 weeks via `scripts/dry_run_supervision.py`.

**Phase 8 capital ladder:** start ≤ ₹10,000 → ₹25k → ₹50k → ₹1L, each step
requires 60 live trading days at the prior level.

## Execution guardrails (live)

- Stale-quote limit 15 s; max 10 orders/day; min daily volume ₹1,000,000;
  max expected slippage 30 bps; max gap 5%.
- Risk gate on every order (see `Risk-Engine.md`): 2% daily loss, 8% drawdown
  freeze, 15% stock cap, VIX 22 / ATR 2× volatility scalar.
- JSONL audit log of every order attempt.

## Current daily operation (paper)

Every scheduled invocation runs `auto_pipeline.py` (see `Architecture.md`):
backfill missed days → train if needed → daily decisions → paper trades →
health sweep → Telegram summary. Scheduled weekdays 15:45 IST plus boot
catch-up on every device, so a day is covered whenever any machine is on.
