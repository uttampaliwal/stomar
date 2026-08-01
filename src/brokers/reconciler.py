"""Order-fill reconciliation: broker truth vs local expectation.

Never assume an order filled because a request was sent. Reconciliation
compares what the broker reports with what the local execution manager
believes and returns a normalized view.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.brokers.base import (
    BrokerAdapter,
    BrokerFill,
    BrokerOrder,
    BrokerOrderStatus,
)

logger = logging.getLogger(__name__)


class OrderReconciler:
    """Reconciles broker-reported state against locally tracked orders."""

    def __init__(self, broker: BrokerAdapter):
        self.broker = broker
        self._known: dict[str, BrokerOrder] = {}

    def track(self, order: BrokerOrder):
        self._known[order.client_order_id] = order

    def refresh(self, client_order_id: str) -> BrokerOrder:
        """Pull the latest broker state for one order and merge it."""
        order = self._known.get(client_order_id)
        if order is None:
            raise KeyError(f"untracked order {client_order_id}")
        broker_state = self.broker.get_order(client_order_id)
        if broker_state is None:
            order.status = BrokerOrderStatus.UNCONFIRMED
            return order
        order.status = broker_state.status
        order.filled_quantity = broker_state.filled_quantity
        order.avg_fill_price = broker_state.avg_fill_price
        order.broker_order_id = broker_state.broker_order_id
        order.reject_reason = broker_state.reject_reason
        order.updated_at = broker_state.updated_at
        return order

    def reconcile_all(self) -> list[BrokerOrder]:
        """Reconcile every tracked order; returns the merged view."""
        return [self.refresh(cid) for cid in list(self._known)]

    def fills_since(self, since: str | None = None) -> list[BrokerFill]:
        return self.broker.get_fills(since=since)

    def is_filled(self, client_order_id: str) -> bool:
        order = self._known.get(client_order_id)
        if order is None:
            return False
        if order.status == BrokerOrderStatus.FILLED:
            return True
        if order.status == BrokerOrderStatus.PARTIALLY_FILLED and (
            order.filled_quantity >= order.quantity
        ):
            return True
        return False

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for order in self._known.values():
            counts[order.status.value] = counts.get(order.status.value, 0) + 1
        return {
            "tracked": len(self._known),
            "by_status": counts,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
        }
