"""Broker adapter layer.

Strict separation between signal generation and order execution:

* signals are produced by ``src/signals/*`` and the meta-controller;
* orders are produced here, in the broker layer, and ONLY here.

Every broker (dry-run or live) must implement :class:`BrokerAdapter`.
The live path is idempotent via client order IDs so that retries can never
double-submit.

Usage:
    from src.brokers import get_broker

    broker = get_broker()          # dry-run by default; live only if gate passes
    order = broker.submit_order(client_order_id="ord-20260801-001",
                                ticker="RELIANCE.NS", side="BUY", quantity=10)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class BrokerOrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNCONFIRMED = "UNCONFIRMED"  # sent but broker state unknown


@dataclass
class BrokerOrder:
    """Canonical order record used between the execution manager and brokers."""

    client_order_id: str
    ticker: str
    side: str  # "BUY" | "SELL"
    quantity: int
    order_type: str = "MARKET"
    limit_price: float = 0.0
    status: BrokerOrderStatus = BrokerOrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    broker_order_id: str = ""
    reject_reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "reject_reason": self.reject_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BrokerOrder":
        return cls(
            client_order_id=d["client_order_id"],
            ticker=d["ticker"],
            side=d["side"],
            quantity=d["quantity"],
            order_type=d.get("order_type", "MARKET"),
            limit_price=d.get("limit_price", 0.0),
            status=BrokerOrderStatus(d.get("status", "PENDING")),
            filled_quantity=d.get("filled_quantity", 0),
            avg_fill_price=d.get("avg_fill_price", 0.0),
            broker_order_id=d.get("broker_order_id", ""),
            reject_reason=d.get("reject_reason", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class BrokerPosition:
    ticker: str
    quantity: int  # signed: +long / -short
    average_price: float
    pnl: float = 0.0
    unrealised_pnl: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "pnl": self.pnl,
            "unrealised_pnl": self.unrealised_pnl,
        }


@dataclass
class BrokerFill:
    broker_order_id: str
    client_order_id: str
    ticker: str
    side: str
    quantity: int
    price: float
    filled_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "filled_at": self.filled_at,
        }


def new_client_order_id(prefix: str = "ord") -> str:
    """Generate an idempotency key for a new order intent."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class BrokerAdapter(ABC):
    """Strict broker interface. All methods are order-execution only."""

    name: str = "base"
    is_live: bool = False

    @abstractmethod
    def submit_order(self, client_order_id: str, ticker: str, side: str,
                     quantity: int, order_type: str = "MARKET",
                     limit_price: float = 0.0) -> BrokerOrder:
        """Submit an order. MUST be idempotent on client_order_id: calling
        twice with the same ID returns the original order without re-sending."""

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> BrokerOrder:
        """Cancel a pending/partially-filled order. Returns updated order."""

    @abstractmethod
    def modify_order(self, client_order_id: str, quantity: int = 0,
                     limit_price: float = 0.0) -> BrokerOrder:
        """Replace quantity/limit price of an open order."""

    @abstractmethod
    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        """Current broker-side status of an order, or None if unknown."""

    @abstractmethod
    def get_fills(self, since: str | None = None) -> list[BrokerFill]:
        """All fills, optionally since an ISO timestamp."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Current positions as reported by the broker."""

    @abstractmethod
    def get_margin(self) -> dict:
        """Margin/leverage summary: available cash, used margin, exposure."""

    def health(self) -> dict:
        return {"name": self.name, "live": self.is_live, "ok": True}
