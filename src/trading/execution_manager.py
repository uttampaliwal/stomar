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
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from src.brokers.base import BrokerAdapter, BrokerOrder, BrokerOrderStatus, new_client_order_id
from src.brokers.reconciler import OrderReconciler
from src.core.secure_io import atomic_append_jsonl
from src.core.trading_mode import get_trading_mode, require_live_allowed
from src.trading.risk_controls import RiskController

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Any failure inside the execution path (rejected by gate, broker, etc.)."""


@dataclass
class PortfolioSnapshot:
    """Broker-truth portfolio state fed into the pre-trade risk gate.

    holdings maps ticker -> (signed quantity, average price). Quantity is
    signed: positive = long, negative = short. This is the single source of
    truth for exposure/concentration checks, so the gate evaluates the whole
    portfolio after the order rather than the order in isolation.
    """

    holdings: dict[str, tuple[float, float]] | None = None
    equity: float | None = None

    def market_value(self) -> float:
        """Gross exposure: sum of |qty| * price across all positions."""
        if not self.holdings:
            return 0.0
        return sum(abs(q) * p for (q, p) in self.holdings.values())

    def ticker_value(self, ticker: str) -> float:
        holding = (self.holdings or {}).get(ticker)
        if holding is None:
            return 0.0
        qty, price = holding
        return abs(qty) * price

    def ticker_quantity(self, ticker: str) -> float:
        holding = (self.holdings or {}).get(ticker)
        return holding[0] if holding is not None else 0.0


# Order intent classification: how this order interacts with the existing
# position on the same ticker (used instead of a binary is_closing flag).
OPENING = "opening"      # new long
INCREASING = "increasing"  # add to an existing long
REDUCING = "reducing"    # shrink an existing long/short
CLOSING = "closing"      # fully exit an existing long/short
REVERSING = "reversing"  # exceed the existing position (flip direction)
SHORTING = "shorting"    # new short

_REDUCING_INTENTS = {REDUCING, CLOSING}


class ExecutionManager:
    """Centralized order execution with risk + mode enforcement."""

    def __init__(self, broker: BrokerAdapter | None, risk: RiskController,
                 audit_dir: str | Path | None = None,
                 max_stale_quote_seconds: float = 15.0,
                 max_daily_orders: int = 10,
                 portfolio_provider: Callable[[], PortfolioSnapshot] | None = None):
        self.broker = broker
        self.risk = risk
        self.reconciler = OrderReconciler(broker) if broker is not None else None
        self.max_stale_quote_seconds = max_stale_quote_seconds
        self.max_daily_orders = max_daily_orders
        self.portfolio_provider = portfolio_provider
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

    def _portfolio_snapshot(self) -> PortfolioSnapshot:
        """Broker-truth portfolio, from the provider or the broker adapter.

        Never raises: a failure to read positions must not silently turn
        into an unlimited portfolio — it degrades to an empty snapshot,
        which the risk engine treats as zero existing exposure.
        """
        if self.portfolio_provider is not None:
            try:
                snap = self.portfolio_provider()
                if snap is not None:
                    return snap
            except Exception as exc:
                logger.warning("portfolio provider failed for risk gate: %s", exc)
        if self.broker is not None:
            try:
                positions = self.broker.get_positions()
                holdings = {
                    p.ticker: (float(p.quantity), float(p.average_price))
                    for p in positions
                    if p.quantity != 0
                }
                equity = None
                try:
                    margin = self.broker.get_margin()
                    available = margin.get("available_cash")
                    exposure = margin.get("total_exposure") or margin.get("used_margin")
                    if available is not None and exposure is not None:
                        equity = float(available) + abs(float(exposure))
                except Exception as exc:
                    logger.debug("margin unavailable for risk gate equity: %s", exc)
                return PortfolioSnapshot(holdings=holdings, equity=equity)
            except Exception as exc:
                logger.warning("broker positions unavailable for risk gate: %s", exc)
        return PortfolioSnapshot(holdings={})

    @staticmethod
    def _classify_intent(ticker: str, side: str, quantity: int,
                         snapshot: PortfolioSnapshot) -> str:
        """Classify how this order interacts with the existing position."""
        cur_qty = snapshot.ticker_quantity(ticker)
        if cur_qty == 0:
            return OPENING if side == "BUY" else SHORTING
        if side == "BUY":
            if cur_qty > 0:
                return INCREASING
            # buying back a short
            if quantity < abs(cur_qty):
                return REDUCING
            return CLOSING if quantity == abs(cur_qty) else REVERSING
        # SELL against an existing long
        if quantity < cur_qty:
            return REDUCING
        return CLOSING if quantity == cur_qty else REVERSING

    @staticmethod
    def _post_order_exposure(intent: str, snapshot: PortfolioSnapshot,
                             ticker: str, order_value: float) -> float:
        """Direction-aware total exposure AFTER this order fills."""
        cur_value = snapshot.market_value()
        if intent in _REDUCING_INTENTS:
            return max(0.0, cur_value - order_value)
        if intent == REVERSING:
            # existing position is exited and replaced by a larger one:
            # exposure = current - old_ticker_value + new_order_value
            return max(0.0, cur_value - snapshot.ticker_value(ticker) + order_value)
        return cur_value + order_value

    @staticmethod
    def _post_position_value(intent: str, snapshot: PortfolioSnapshot,
                             ticker: str, order_value: float) -> float | None:
        """Value of THIS ticker's position after the order (None = skip)."""
        cur = snapshot.ticker_value(ticker)
        if intent == INCREASING:
            return cur + order_value
        if intent == REVERSING:
            return max(0.0, order_value - cur)
        if intent in _REDUCING_INTENTS:
            return max(0.0, cur - order_value)
        return None  # opening/shorting: falls back to order_value

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

        # Broker-truth portfolio: existing positions drive every exposure and
        # concentration check, so the gate evaluates the portfolio AFTER
        # this order, never the order in isolation.
        snapshot = self._portfolio_snapshot()
        intent = self._classify_intent(ticker, side, quantity, snapshot)
        current_value = snapshot.market_value()
        post_exposure = self._post_order_exposure(intent, snapshot, ticker, order_value)
        post_position = self._post_position_value(intent, snapshot, ticker, order_value)
        effective_closing = is_closing or intent in _REDUCING_INTENTS

        # Only an explicit provider (e.g. PaperTrader.get_equity) may update the
        # risk controller's equity. Equity derived from raw broker margin is
        # a static, low-quality proxy that would clobber the controller's own
        # peak/drawdown tracking (e.g. a deliberate test drawdown) on every
        # gate evaluation.
        if (self.portfolio_provider is not None
                and snapshot.equity is not None and snapshot.equity > 0):
            self.risk.update_equity(snapshot.equity)

        result = self.risk.check_order(
            order_value=order_value,
            current_holdings_value=current_value,
            post_order_exposure=post_exposure,
            post_position_value=post_position,
            ticker=ticker,
            holdings=snapshot.holdings,
            prices=quotes,
            market=market,
            is_closing=effective_closing,
        )
        checks = list(result.get("checks", []))
        checks.append({
            "passed": True, "check": "execution_path_verified",
            "message": "order routed through ExecutionManager gate",
        })
        checks.append({
            "passed": True, "check": "order_intent",
            "message": (f"intent={intent} current_exposure={current_value:,.0f} "
                        f"post_exposure={post_exposure:,.0f}"),
        })
        return {
            "approved": result["approved"] and all(c["passed"] for c in checks),
            "checks": checks,
            "reason": result.get("reason", ""),
            "intent": intent,
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
