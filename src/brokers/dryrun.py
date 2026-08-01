"""Dry-run broker: deterministic simulated execution with reconciliation.

Models the same life cycle as a live broker — submit, partial fills,
rejections, cancellations, fills — so that paper trading exercises the
identical idempotency and reconciliation code paths as live trading.

This broker NEVER touches a real market. It is the default broker.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from src.brokers.base import (
    BrokerAdapter,
    BrokerFill,
    BrokerOrder,
    BrokerOrderStatus,
    BrokerPosition,
)

logger = logging.getLogger(__name__)


class DryRunBroker(BrokerAdapter):
    """In-process simulated broker.

    Parameters:
        fill_delay_seconds: simulated latency before a market order fills.
        partial_fill_pct: fraction of orders filled partially first
            (0.0 = always full fill, 1.0 = always partial first).
        reject_rate: probability that a random order is rejected
            (used by tests to exercise rejection handling).
    """

    name = "dry-run"
    is_live = False

    def __init__(self, fill_delay_seconds: float = 0.0,
                 partial_fill_pct: float = 0.0,
                 reject_rate: float = 0.0,
                 slippage_bps: float = 5.0,
                 initial_cash: float = 1_000_000.0):
        self.fill_delay_seconds = max(0.0, fill_delay_seconds)
        self.partial_fill_pct = min(1.0, max(0.0, partial_fill_pct))
        self.reject_rate = min(1.0, max(0.0, reject_rate))
        self.slippage_bps = slippage_bps
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[BrokerFill] = []
        self._positions: dict[str, BrokerPosition] = {}
        self._cash: float = initial_cash
        self._lock = threading.RLock()
        self._next_fill_time: dict[str, float] = {}
        self._order_seq = 0

    # ── lifecycle helpers (dry-run bookkeeping) ───────────────────────────

    def _touch(self, order: BrokerOrder):
        order.updated_at = datetime.now(timezone.utc).isoformat()

    def _apply_fill(self, order: BrokerOrder, qty: int, price: float):
        prev_filled = order.filled_quantity
        order.filled_quantity += qty
        order.avg_fill_price = (
            (prev_filled * order.avg_fill_price + qty * price)
            / max(order.filled_quantity, 1)
        )
        if order.filled_quantity >= order.quantity:
            order.status = BrokerOrderStatus.FILLED
        else:
            order.status = BrokerOrderStatus.PARTIALLY_FILLED
        self._fills.append(BrokerFill(
            broker_order_id=order.broker_order_id,
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=qty,
            price=price,
        ))
        pos = self._positions.get(order.ticker)
        if pos is None:
            pos = BrokerPosition(ticker=order.ticker, quantity=0, average_price=0.0)
            self._positions[order.ticker] = pos
        signed = qty if order.side == "BUY" else -qty
        prev = pos.quantity
        if prev == 0:
            pos.average_price = price
        elif (prev > 0) == (signed > 0):
            pos.average_price = (
                (abs(prev) * pos.average_price + qty * price) / (abs(prev) + qty)
            )
        pos.quantity = prev + signed
        self._touch(order)

    def _do_fill(self, client_order_id: str, price: float, qty: int | None = None):
        with self._lock:
            order = self._orders.get(client_order_id)
            if order is None or order.status in (
                BrokerOrderStatus.FILLED, BrokerOrderStatus.CANCELLED,
                BrokerOrderStatus.REJECTED,
            ):
                return
            fill_qty = qty if qty is not None else order.quantity - order.filled_quantity
            fill_qty = min(fill_qty, order.quantity - order.filled_quantity)
            if fill_qty <= 0:
                return
            self._apply_fill(order, fill_qty, price)

    # ── BrokerAdapter implementation ──────────────────────────────────────

    def submit_order(self, client_order_id: str, ticker: str, side: str,
                     quantity: int, order_type: str = "MARKET",
                     limit_price: float = 0.0) -> BrokerOrder:
        with self._lock:
            existing = self._orders.get(client_order_id)
            if existing is not None:
                # Idempotent: duplicate submit returns the original order.
                logger.info("dry-run: duplicate submit suppressed for %s", client_order_id)
                return existing

            self._order_seq += 1
            order = BrokerOrder(
                client_order_id=client_order_id,
                broker_order_id=f"DB-{self._order_seq:08d}",
                ticker=ticker, side=side.upper(), quantity=quantity,
                order_type=order_type.upper(), limit_price=limit_price,
                status=BrokerOrderStatus.SUBMITTED,
            )
            if limit_price <= 0:
                limit_price = 1.0  # marker: no limit provided
            self._orders[client_order_id] = order

            # Simulated rejection
            if quantity <= 0:
                order.status = BrokerOrderStatus.REJECTED
                order.reject_reason = "invalid quantity"
            elif order_type not in ("MARKET", "LIMIT"):
                order.status = BrokerOrderStatus.REJECTED
                order.reject_reason = f"unsupported order type {order_type}"
            elif self.reject_rate > 0 and hash(client_order_id) % 1000 / 1000 < self.reject_rate:
                order.status = BrokerOrderStatus.REJECTED
                order.reject_reason = "simulated rejection"
            else:
                self._next_fill_time[client_order_id] = (
                    time.monotonic() + self.fill_delay_seconds
                )
            self._touch(order)
            logger.info("dry-run: submitted %s %s %s @ %s (id=%s)",
                        side, quantity, ticker, order_type, client_order_id)
            return order

    def process_pending(self, price_map: dict[str, float]):
        """Advance the simulated market: fill due market orders at given prices.

        Called by the execution manager on each bar/quote tick.
        """
        now = time.monotonic()
        with self._lock:
            for cid, due in list(self._next_fill_time.items()):
                if now < due:
                    continue
                order = self._orders.get(cid)
                if order is None:
                    continue
                price = price_map.get(order.ticker, 0.0)
                if price <= 0:
                    continue
                slip = price * self.slippage_bps / 10_000
                fill_price = price + slip if order.side == "BUY" else price - slip
                del self._next_fill_time[cid]
                if self.partial_fill_pct > 0 and order.status == BrokerOrderStatus.SUBMITTED:
                    first = max(1, int(order.quantity * (1 - self.partial_fill_pct)))
                    self._apply_fill(order, min(first, order.quantity), fill_price)
                    # remainder fills on the next tick
                    self._next_fill_time[cid] = now + 0.05
                else:
                    self._apply_fill(order, order.quantity - order.filled_quantity, fill_price)

    def cancel_order(self, client_order_id: str) -> BrokerOrder:
        with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                raise KeyError(f"unknown order {client_order_id}")
            if order.status in (BrokerOrderStatus.FILLED, BrokerOrderStatus.REJECTED):
                raise RuntimeError(
                    f"cannot cancel {client_order_id}: already {order.status.value}"
                )
            order.status = BrokerOrderStatus.CANCELLED
            self._next_fill_time.pop(client_order_id, None)
            self._touch(order)
            return order

    def modify_order(self, client_order_id: str, quantity: int = 0,
                     limit_price: float = 0.0) -> BrokerOrder:
        with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                raise KeyError(f"unknown order {client_order_id}")
            if order.status not in (BrokerOrderStatus.SUBMITTED, BrokerOrderStatus.PARTIALLY_FILLED):
                raise RuntimeError(
                    f"cannot modify {client_order_id}: already {order.status.value}"
                )
            if quantity > 0:
                order.quantity = quantity
            if limit_price > 0:
                order.limit_price = limit_price
            self._touch(order)
            return order

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        with self._lock:
            order = self._orders.get(client_order_id)
            return order

    def get_fills(self, since: str | None = None) -> list[BrokerFill]:
        with self._lock:
            if since is None:
                return list(self._fills)
            return [f for f in self._fills if f.filled_at >= since]

    def get_positions(self) -> list[BrokerPosition]:
        with self._lock:
            return list(self._positions.values())

    def get_margin(self) -> dict:
        with self._lock:
            exposure = sum(
                abs(p.quantity) * p.average_price for p in self._positions.values()
            )
            return {
                "available_cash": self._cash,
                "used_margin": exposure,
                "total_exposure": exposure,
            }

    def health(self) -> dict:
        return {"name": self.name, "live": False, "ok": True}
