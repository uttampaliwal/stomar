# Paper Trading

StoMar's simulated trading engine — realistic fills, NSE costs, taxes, and a
durable ledger. **No broker, no real money.**

## Daily decision flow

`DailyOrchestrator.run()` (`src/signals/orchestrator.py`):

1. Resolve yesterday's pending outcomes (next-day close-to-close return).
2. Pre-fetch FII/DII + PCR.
3. Per ticker: collect 14 signals (sentiment → flow → PCR → MTF → regime →
   VaR/CVaR/Sharpe → volatility → fundamentals → ensemble).
4. Meta-controller decision (or rule-based fallback: buy 0.6 / sell 0.4 /
   max position 0.10).
5. Caps: `max_daily_trades = 5`, `max_portfolio_exposure = 0.50`.
6. Execute paper trades at live prices with a 5% stop-loss, log to ledger.

## PaperTrader (`src/trading/paper_trader.py`)

- State in `data/paper_state.json`: `initial_capital` (default 200,000 in the
  pipeline; 100,000 library default), `cash`, `cumulative_pnl`, positions,
  closed positions, trade log, risk counters, full engine state. Atomic
  writes (mkstemp + replace); cross-process `FileLock` serializes API workers
  and scheduled scripts.
- `place_order` → risk check → engine submit; `execute_market_trade` fills
  immediately at the supplied price.

## Execution engine (`src/trading/engine.py`)

- Order types: MARKET, LIMIT, STOP_LOSS, STOP_MARKET; statuses PENDING / FILLED / PARTIALLY_FILLED / CANCELLED / REJECTED; IDs `ORD-000001` style.
- Fill logic: MARKET fills at bar open; LIMIT fills when `low <= price` at `min(price, open)`; STOP_LOSS at stop price; STOP_MARKET at bar open after trigger. Limit fills have distance-based probability (0.9 within 0.5%, 0.7 within 1%, 0.4 within 2%, else 0.1).
- Slippage models: Fixed (0.001), Volume (base 0.002, cap 3×), Adaptive, Backtest simulator. Default paper slippage 5 bps.
- **NSE cost model** (`src/core/constants.py`): brokerage 0.03%/side, STT 0.1% both sides, NSE exchange charge 0.00345%, SEBI 0.0001%, stamp duty 0.015% buy-side, GST 18% on brokerage + exchange.
- **Tax model (P5.5)**: STCG 15% (<365 days), LTCG 10% with ₹1L annual exemption; applied to realized gains with exemption tracking → `post_tax_return`.

## Ledger (`src/trading/ledger.py`)

SQLite at `data/stomar.db` (schema v2):

- `decisions` — the 14-signal vector + action/size/confidence/reasoning + `source` (`live` | `backfill`).
- `paper_trades` — executed trades.
- `portfolio_snapshots` — equity curve for drawdown/benchmark math.
- `resolve_outcomes` — each unresolved **live** decision gets its next-trading-day return; BUY correct iff direction +1, SELL iff 0, HOLD iff |return| ≤ 0.5%.
- Reads: performance, per-signal accuracy, 30-day rolling accuracy (live only), drawdown stats, daily P&L, agreement stats.

Backfill rows are excluded from trust metrics (in-sample by construction).

## Paper-trading API

All endpoints under `/api/paper-trading/` are protected by `X-API-Key`
(see `API.md`). The API **cannot** route live orders — it refuses to run
when the trading mode is `live`.

## Benchmarks (`src/signals/benchmarks.py`)

- `GET /api/ledger/benchmark` — paper equity curve vs **Nifty 50 buy-and-hold**
  over the same window: alpha, annualized information ratio, max drawdowns.
- Other baselines: buy & hold, 20D momentum, always long, random, SMA crossover.

## Supervision & dry runs

- `scripts/dry_run_supervision.py` — supervised dry-run harness → `data/dry_run/`.
- `src/brokers/dryrun.py` — `DryRunBroker` with deterministic fills/partials/rejections, same lifecycle as live.
- `verify_system.py` — environment sanity checks.

## Status

The P3.1 paper-trading clock has been running since **2026-08-04** (0/60
trading days). The readiness gate (`/api/monitoring/readiness`) requires
≥60 paper days, rolling 30-day accuracy ≥51%, drawdown ≤15%, no critical
drift in 7 days, models ≤30 days old before any live consideration.
