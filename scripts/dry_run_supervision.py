"""Supervised broker-backed dry-run.

Runs the REAL execution path (ExecutionManager -> broker -> reconciler)
against the DryRunBroker, using today's orchestrator decisions. This is
the supervised dry-run period required before live trading: the same
idempotency, risk-gate, audit, and reconciliation code paths are exercised
with zero market exposure.

This script NEVER constructs a live broker. It is paper-only by design.

Usage:
    python scripts/dry_run_supervision.py                      # all tickers
    python scripts/dry_run_supervision.py --ticker RELIANCE.NS TCS.NS
    python scripts/dry_run_supervision.py --capital 500000

Report artifacts:
    data/dry_run/orders-YYYY-MM-DD.jsonl   (audit log of every intent)
    data/dry_run/state.json                (positions, equity, risk state)
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.brokers import reset_broker
from src.core.secure_io import atomic_write_json
from src.core.trading_mode import get_trading_mode, mode_banner
from src.trading.execution_manager import ExecutionManager, ExecutionError
from src.trading.risk_controls import MarketContext, RiskController, RiskLimits
from src.data.data_fetcher import NSE_STOCKS
from src.trading.ledger import Ledger
from src.signals.orchestrator import DailyOrchestrator
from src.models.meta_controller import MetaController
from src.core.constants import META_CONTROLLER_PATH, DATA_DIR

_REPORT_DIR = os.path.join(DATA_DIR, "dry_run")


def _quote_for(ticker: str, price: float) -> dict:
    """Build a fresh-quote payload from a known price (timestamped now)."""
    return {
        "close": price,
        "last_price": price,
        "prev_close": price,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_supervision(tickers: list[str], capital: float = 500_000,
                    db_path: str | None = None) -> dict:
    reset_broker()
    from src.brokers.dryrun import DryRunBroker
    broker = DryRunBroker(initial_cash=capital)  # paper-only by construction
    mode = get_trading_mode()
    assert not broker.is_live, "supervision script must never use a live broker"

    os.makedirs(_REPORT_DIR, exist_ok=True)
    audit_dir = _REPORT_DIR
    risk = RiskController(RiskLimits(), initial_capital=capital)
    manager = ExecutionManager(broker, risk, audit_dir=audit_dir)

    ledger = Ledger(db_path)
    meta_controller = MetaController()
    if os.path.exists(META_CONTROLLER_PATH):
        meta_controller.load()  # manifest-verified

    orchestrator = DailyOrchestrator(
        tickers=tickers, ledger=ledger, meta_controller=meta_controller,
    )

    print("=== StoMar Supervised Dry-Run ===")
    print(mode_banner())
    print(f"Broker:   {broker.name} (live={broker.is_live})")
    print(f"Capital:  Rs {capital:,.0f}")
    print(f"Audit:    {audit_dir}")
    print()

    summary = orchestrator.run(dry_run=True)
    ledger.close()

    decisions = summary.get("decisions", [])
    print(f"Decisions: {len(decisions)} | Errors: {len(summary.get('errors', []))}")

    orders_placed = 0
    blocked = []
    for d in decisions:
        ticker = d["ticker"]
        action = d["action"]
        size_pct = d.get("position_size", 0)
        price = d.get("current_price", 0)
        if action == "HOLD" or size_pct <= 0 or price <= 0:
            continue
        qty = max(1, int(capital * size_pct / price))
        try:
            order = manager.execute(
                ticker, action, qty,
                quotes={ticker: _quote_for(ticker, price)},
                order_value=price * qty,
                market=MarketContext(price=price, prev_close=price),
            )
            orders_placed += 1
            print(f"  SUBMIT {action:4s} {qty:5d} {ticker:15s} @ {price:.2f} "
                  f"(id={order.client_order_id}) status={order.status.value}")
        except ExecutionError as exc:
            blocked.append({"ticker": ticker, "action": action, "reason": str(exc)})
            print(f"  BLOCK  {action:4s} {ticker:15s} -> {exc}")

    # Advance the simulated market: fills at last known prices, then reconcile.
    price_map = {
        d["ticker"]: d.get("current_price", 0) for d in decisions if d.get("current_price", 0) > 0
    }
    if price_map:
        broker.process_pending(price_map)
    reconciled = manager.reconcile()

    positions = broker.get_positions()
    margin = broker.get_margin()
    equity = margin.get("available_cash", capital)

    report = {
        "date": date.today().isoformat(),
        "mode": mode.value,
        "broker": broker.name,
        "orders_placed": orders_placed,
        "orders_blocked": len(blocked),
        "blocked": blocked,
        "filled": [
            o.to_dict() for o in reconciled
            if o.status.value in ("FILLED", "PARTIALLY_FILLED")
        ],
        "pending": [o.to_dict() for o in reconciled
                    if o.status.value not in ("FILLED", "CANCELLED", "REJECTED")],
        "positions": [p.to_dict() for p in positions],
        "margin": margin,
        "risk": risk.get_status(),
        "gate": manager.status()["reconciler"],
    }
    atomic_write_json(os.path.join(_REPORT_DIR, "state.json"), report)

    print("\n=== Supervision Report ===")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervised broker-backed dry-run")
    parser.add_argument("--ticker", nargs="+", default=None,
                        help="Tickers (default: all NSE stocks)")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Paper capital (default 500000)")
    parser.add_argument("--db", default=None, help="Ledger DB path")
    args = parser.parse_args()

    tickers = args.ticker if args.ticker else list(NSE_STOCKS)
    report = run_supervision(tickers, capital=args.capital, db_path=args.db)
    return 0 if report["orders_blocked"] == 0 else 0  # block ≠ failure


if __name__ == "__main__":
    sys.exit(main())
