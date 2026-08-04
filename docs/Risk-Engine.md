# Risk Engine

Two complementary layers: a soft rule layer applied to every order
(`RiskController`), and a hard circuit-breaker layer (`RiskGuard`) that can
freeze trading entirely. Both are configurable via `STOMAR_*` env vars.

## Layer 1 — RiskController (`src/trading/risk_controls.py`)

Applied per order via `check_order()` (up to 12 checks):

| Limit | Default |
|---|---|
| Max position (single stock) | 25% |
| Max daily loss | 2% of current equity |
| Max weekly loss | 5% |
| Max drawdown | 15% → **permanent halt** |
| Max total exposure | 95% |
| Max open orders | 10 |
| Kelly fraction | 0.25 (quarter-Kelly sizing) |
| Max correlated exposure | 40% (correlation threshold 0.70) |
| Max consecutive losses | 5 → kill switch |
| Min daily volume | ₹1,000,000 |
| Max expected slippage | 30 bps |
| Max gap vs previous close | 5% |

- Closing/reducing orders skip entry-only checks.
- `kelly_sized_quantity` caps at both quarter-Kelly and max-position.
- **Kill switch** (`data/kill_switch.json`): manual or automatic halt; persists
  across restarts; while active, all orders are blocked. Clear by deleting
  the file (or `clear_kill_switch()`). Pause everything with
  `{"halted": true}` (or `{"active": true, "reason": ...}`).

## Layer 2 — RiskGuard hardware breakers (`services/risk_guard.py`)

Stricter, freeze-on-breach limits:

| Guard | Default |
|---|---|
| Daily loss | 2% |
| Drawdown | 8% → **freeze all trading** |
| Max single-stock allocation | 15% |
| VIX threshold | 22 |
| ATR expansion | 2.0× of 20-bar average |

- **Volatility scalar**: VIX factor tapers 1.0 → 0.25 between VIX 22 and 32;
  ATR factor likewise between 2× and 3×; combined product clamped at 0.25
  floor. VIX from `^INDIAVIX`; fetch failures degrade to 1.0 (fail-open for
  sizing).
- Freeze semantics: drawdown freeze halts everything **except closing sells**;
  daily-loss blocks buys only; sells are never blocked.
- Freeze persists in `data/risk_guard_state.json`; resuming requires
  `STOMAR_RISK_OVERRIDE_TOKEN` (constant-time compare, `403` on mismatch).
  If no token is configured, a freeze cannot be lifted by design.
- Sizing respects the scalar: lower leverage in high-volatility regimes.

## Execution gate (`src/trading/execution_manager.py`)

`gate_order()` evaluates the full gate **without submitting**:
`{approved, reason, checks}`. `execute()` enforces in order:

1. Quantity validation
2. Duplicate-intent suppression via `client_order_id` (idempotency)
3. Trading mode re-verified on **every** execution (`require_live_allowed()`)
4. Stale-quote limit (15 s)
5. Risk gate (RiskController + RiskGuard)
6. Daily order budget (10 orders)
7. Broker submit + immediate reconciliation (broker truth wins; orders never assumed filled)
8. JSONL audit log (`data/api_audit/orders-YYYY-MM-DD.jsonl`)

## Sizing & exposure caps (orchestrator level)

- `max_daily_trades = 5`
- `max_portfolio_exposure = 0.50`
- `stop_loss_pct = 0.05` — every paper trade is opened with a 5% stop
  (BUY → SELL STOP_MARKET at fill × 0.95; SELL → BUY STOP_MARKET at fill × 1.05).

## Daily P&L reset (P0.4)

`auto_pipeline._run_daily` resets the paper trader's daily P&L counter at the
start of each new trading day, so the 2% daily-loss limit applies to today's
activity only.

## Risk analytics

- `/api/risk/{ticker}`, `/api/risk/portfolio/all` — per-ticker and portfolio risk (VaR95/CVaR95/Sharpe).
- `/api/risk-guard/status`, `/api/risk-guard/scalar` — freeze state and live volatility scalar.
- `src/trading/risk.py` — Kelly and VaR helpers.
- `src/trading/optimizer.py` — portfolio construction with risk constraints.
