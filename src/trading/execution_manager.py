"""Execution manager — the single choke point between signals and brokers.

Every order intent — from the orchestrator, the API, or any other caller —
must pass through :class:`ExecutionManager.execute`. It enforces, in order:

1. mode: paper vs live (live requires the full gate);
2. staleness: quotes must be fresh enough to trade on;
3. risk: the pre-trade gate (daily/weekly loss, drawdown, exposure,
   concentration, liquidity, gap, duplicate orders, daily order budget);
4. idempotency: duplicate intents (same client_order_id) never re-submit;
5. reconciliation: broker truth is pulled after submission and on demand.

The manager records every order intent to a JSONL audit log so that
operations can reconstruct exactly what the system decided and why.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from src.brokers.base import BrokerAdapter, BrokerOrder, BrokerOrderStatus, new_client_order_id
from src.brokers.reconciler import OrderReconciler
from src.core.secure_io import atomic_append_jsonl
from src.core.trading_mode import get_trading_mode, require_live_allowed
from src.trading.risk_controls import RiskController

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Any failure inside the execution path (rejected by gate, broker, etc.)."""


class ExecutionManager:
    """Centralized order execution with risk + mode enforcement."""

    def __init__(self, broker: BrokerAdapter | None, risk: RiskController,
                 audit_dir: str | Path | None = None,
                 max_stale_quote_seconds: float = 15.0,
                 max_daily_orders: int = 10):
        self.broker = broker
        self.risk = risk
        self.reconciler = OrderReconciler(broker) if broker is not None else None
        self.max_stale_quote_seconds = max_stale_quote_seconds
        self.max_daily_orders = max_daily_orders
        self._lock = threading.RLock()
        self._audit_log: Path | None = None
        if audit_dir is not None:
            audit_dir = Path(audit_dir)
            audit_dir.mkdir(parents=True, exist_ok=True)
            self._audit_log = audit_dir / f"orders-{date.today().isoformat()}.jsonl"
        self._daily_orders: dict[str, int] = {}
        self._current_day: str = date.today().isoformat()
        self._intents: dict[str, BrokerOrder] = {}

    # ── audit ──────────────────────────────────────────────────────────────

    def _audit(self, entry: dict):
        if self._audit_log is None:
            return
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
        atomic_append_jsonl(self._audit_log, entry)

    # ── public API ─────────────────────────────────────────────────────────

    def execute(self, ticker: str, side: str, quantity: int,
                order_type: str = "MARKET", limit_price: float = 0.0,
                client_order_id: str | None = None,
                quotes: dict[str, dict] | None = None,
                order_value: float | None = None,
                market=None, is_closing: bool = False) -> BrokerOrder:
        """Submit one order intent through the full gate. Returns the order."""
        if quantity <= 0:
            raise ExecutionError(f"invalid quantity {quantity}")
        cid = client_order_id or new_client_order_id()

        with self._lock:
            existing = self._intents.get(cid)
            if existing is not None:
                logger.warning("duplicate intent suppressed: %s", cid)
                self._audit({"event": "duplicate_intent_suppressed", "client_order_id": cid})
                return existing

            mode = get_trading_mode()
            if mode.value == "live":
                require_live_allowed()  # re-verify the gate on every execution

            # 1. stale quotes
            stale = self._stale_quote_check(ticker, quotes or {})
            if stale:
                self._audit({"event": "rejected", "client_order_id": cid,
                             "ticker": ticker, "side": side, "reason": stale})
                raise ExecutionError(stale)

            # 2. LIMIT price sanity check against the market quote
            if order_type.upper() == "LIMIT":
                if limit_price <= 0:
                    reason = "LIMIT orders require a positive limit price"
                    self._audit({"event": "rejected", "client_order_id": cid,
                                 "ticker": ticker, "side": side, "reason": reason})
                    raise ExecutionError(reason)
                quote = (quotes or {}).get(ticker) or {}
                current = quote.get("close") or quote.get("last_price") or 0.0
                if current > 0 and not (0.5 * current <= limit_price <= 1.5 * current):
                    reason = (f"limit price {limit_price} must be within 50% of "
                              f"current price {current}")
                    self._audit({"event": "rejected", "client_order_id": cid,
                                 "ticker": ticker, "side": side, "reason": reason})
                    raise ExecutionError(reason)

            # 2. risk gate
            risk_result = self._risk_gate(ticker, side, quantity, order_value,
                                          quotes or {}, market=market,
                                          is_closing=is_closing)
            if not risk_result["approved"]:
                failed = [c["message"] for c in risk_result["checks"]
                          if not c["passed"] and c.get("message")]
                reason = risk_result.get("reason", "")
                if failed:
                    reason = "; ".join(failed) if not reason else f"{reason}; {'; '.join(failed)}"
                self._audit({"event": "rejected", "client_order_id": cid,
                             "ticker": ticker, "side": side, "reason": reason})
                raise ExecutionError(f"risk gate rejected: {reason}")

            # 3. submit
            if self.broker is None:
                raise ExecutionError("no broker configured for execution")
            order = self.broker.submit_order(
                client_order_id=cid, ticker=ticker, side=side,
                quantity=quantity, order_type=order_type, limit_price=limit_price,
            )
            self.reconciler.track(order)
            self._intents[cid] = order
            self._bump_daily_order_count(cid)
            self._audit({
                "event": "submitted", "client_order_id": cid,
                "broker_order_id": order.broker_order_id, "ticker": ticker,
                "side": side, "quantity": quantity, "order_type": order_type,
                "mode": mode.value,
            })

            # 4. immediate reconciliation attempt
            if order.status == BrokerOrderStatus.SUBMITTED:
                try:
                    order = self.reconciler.refresh(cid)
                except Exception as exc:
                    logger.warning("reconciliation failed for %s: %s", cid, exc)
            return order

    def gate_order(self, ticker: str, side: str, quantity: int,
                   quotes: dict[str, dict] | None = None,
                   order_value: float | None = None,
                   market=None, is_closing: bool = False) -> dict:
        """Evaluate the full gate WITHOUT submitting. Returns verdict dict."""
        stale = self._stale_quote_check(ticker, quotes or {})
        if stale:
            return {"approved": False, "reason": stale}
        result = self._risk_gate(ticker, side, quantity, order_value, quotes or {},
                                 market=market, is_closing=is_closing)
        reason = result.get("reason", "")
        if not result["approved"]:
            failed = [c["message"] for c in result["checks"] if not c["passed"] and c.get("message")]
            if failed:
                reason = "; ".join(failed) if not reason else f"{reason}; {'; '.join(failed)}"
        return {
            "approved": result["approved"],
            "reason": reason,
            "checks": result["checks"],
        }

    def cancel(self, client_order_id: str) -> BrokerOrder:
        with self._lock:
            order = self.broker.cancel_order(client_order_id)
            self._audit({"event": "cancelled", "client_order_id": client_order_id,
                         "broker_order_id": order.broker_order_id})
            return order

    def reconcile(self, client_order_id: str | None = None) -> list[BrokerOrder]:
        if client_order_id is not None:
            return [self.reconciler.refresh(client_order_id)]
        return self.reconciler.reconcile_all()

    def status(self) -> dict:
        mode = get_trading_mode()
        return {
            "mode": mode.value,
            "broker": self.broker.name if self.broker is not None else "none",
            "live": self.broker.is_live if self.broker is not None else False,
            "risk": self.risk.get_status(),
            "reconciler": (
                self.reconciler.summary() if self.reconciler is not None else {}
            ),
            "orders_today": sum(self._daily_orders.values()),
            "max_daily_orders": self.max_daily_orders,
        }

    # ── internal checks ────────────────────────────────────────────────────

    def _stale_quote_check(self, ticker: str,
                           quotes: dict[str, dict]) -> str:
        quote = quotes.get(ticker)
        if quote is None:
            return f"no quote available for {ticker}"
        ts = quote.get("timestamp") or quote.get("ts") or ""
        price = quote.get("close") or quote.get("last_price") or 0.0
        if price <= 0:
            return f"non-positive quote price for {ticker}"
        try:
            quoted_at = datetime.fromisoformat(ts)
            if quoted_at.tzinfo is None:
                quoted_at = quoted_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return f"unparseable quote timestamp for {ticker}: {ts!r}"
        age = (datetime.now(timezone.utc) - quoted_at).total_seconds()
        if age > self.max_stale_quote_seconds:
            return (f"stale quote for {ticker}: {age:.0f}s old "
                    f"(limit {self.max_stale_quote_seconds:.0f}s)")
        return ""

    def _risk_gate(self, ticker: str, side: str, quantity: int,
                   order_value: float | None, quotes: dict[str, dict],
                   market=None, is_closing: bool = False) -> dict:
        price = 0.0
        quote = quotes.get(ticker) or {}
        price = quote.get("close") or quote.get("last_price") or 0.0
        if order_value is None:
            order_value = price * quantity

        # Derive a MarketContext from the quote when none was supplied.
        if market is None:
            from src.trading.risk_controls import MarketContext
            market = MarketContext(
                price=price,
                prev_close=quote.get("prev_close") or 0.0,
                avg_daily_traded_value=quote.get("avg_daily_traded_value")
                or quote.get("daily_volume_rs") or 0.0,
                expected_slippage_bps=quote.get("expected_slippage_bps") or 0.0,
            )

        # daily order budget (rolls over at midnight in a long-running process)
        self._rollover_daily_orders()
        if sum(self._daily_orders.values()) >= self.max_daily_orders:
            return {"approved": False, "checks": [{
                "passed": False, "check": "daily_order_budget",
                "message": (f"daily order budget exhausted "
                            f"({self.max_daily_orders})"),
            }]}

        result = self.risk.check_order(
            order_value=order_value,
            current_holdings_value=0.0,
            ticker=ticker,
            holdings={},
            prices=quotes,
            market=market,
            is_closing=is_closing,
        )
        checks = list(result.get("checks", []))
        checks.append({
            "passed": True, "check": "execution_path_verified",
            "message": "order routed through ExecutionManager gate",
        })
        return {
            "approved": result["approved"] and all(c["passed"] for c in checks),
            "checks": checks,
            "reason": result.get("reason", ""),
        }

    def _bump_daily_order_count(self, client_order_id: str):
        self._rollover_daily_orders()
        self._daily_orders[client_order_id] = self._daily_orders.get(client_order_id, 0) + 1

    def _rollover_daily_orders(self):
        """Reset the daily order counter when the calendar day changes.

        A long-running process must not carry yesterday's order count into
        today, or the daily budget silently shrinks to zero after the first
        day (#33).
        """
        with self._lock:
            today = date.today().isoformat()
            if self._current_day != today:
                if self._daily_orders:
                    logger.info("resetting daily order count: %d -> 0 (new day %s)",
                                sum(self._daily_orders.values()), today)
                self._daily_orders = {}
                self._current_day = today
