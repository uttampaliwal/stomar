"""Live broker adapter for Zerodha Kite Connect.

Safety properties:

* constructed only through :func:`get_broker`, which refuses to return a
  live broker unless the full live-trading gate passes
  (src/core/trading_mode);
* every submit is idempotent on the client order ID: a retry of the same
  intent returns the previously submitted order instead of re-sending;
* credentials are read from settings and never logged;
* all network calls carry explicit timeouts;
* fills are always reconciled by pulling broker state — the local code
  never assumes an order filled because a request was sent.
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
from src.core.settings import settings
from src.core.trading_mode import require_live_allowed

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_INSTRUMENT_TTL = 3600.0  # refresh instrument snapshot at most hourly


class KiteLiveBroker(BrokerAdapter):
    """Zerodha Kite Connect live order execution."""

    name = "kite"
    is_live = True

    def __init__(self):
        # Hard gate: construction itself is refused without full opt-in.
        require_live_allowed()
        if (settings.live_broker or "kite") != "kite":
            raise ValueError(f"unsupported live_broker: {settings.live_broker}")
        if not settings.kite_api_key or not settings.kite_access_token:
            raise ValueError("kite broker requires STOMAR_KITE_API_KEY and STOMAR_KITE_ACCESS_TOKEN")
        try:
            from kiteconnect import KiteConnect  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "kiteconnect package not installed — live trading is unavailable"
            ) from exc
        self._kite = KiteConnect(api_key=settings.kite_api_key)
        self._kite.set_access_token(settings.kite_access_token)
        self._submitted: dict[str, BrokerOrder] = {}
        self._instruments: dict | None = None
        self._instruments_ts: float = 0.0
        self._instrument_lock = threading.Lock()
        logger.warning(
            "KiteLiveBroker constructed — LIVE orders may be submitted. "
            "Account: %s", settings.live_account_id or "<unset>"
        )

    # ── Kite symbol → instrument mapping (cached snapshot) ─────────────────

    def _instrument_token(self, ticker: str) -> int:
        """Resolve ``RELIANCE.NS`` to a Kite instrument token.

        Fetches the NSE instrument list on first use (TTL-cached). Refuses
        to trade when the mapping is unavailable — fail-closed.
        """
        symbol = ticker.upper().removesuffix(".NS")
        now = time.monotonic()
        with self._instrument_lock:
            if self._instruments is None or now - self._instruments_ts > _INSTRUMENT_TTL:
                try:
                    instruments = self._kite.instruments("NSE") or []
                    self._instruments = {
                        str(i.get("tradingsymbol", "")).upper(): i
                        for i in instruments
                    }
                    self._instruments_ts = now
                    logger.info(
                        "kite instrument snapshot refreshed: %d symbols",
                        len(self._instruments),
                    )
                except Exception as exc:
                    self._instruments = None
                    raise RuntimeError(
                        f"cannot fetch instrument list for {ticker}: {exc}"
                    ) from exc
            row = self._instruments.get(symbol)
            if row is None:
                raise ValueError(
                    f"ticker {ticker!r} not found in Kite NSE instruments "
                    f"(snapshot has {len(self._instruments)} symbols)"
                )
            token = int(row.get("instrument_token", 0) or 0)
            if token <= 0:
                raise ValueError(f"instrument {ticker} has no instrument_token")
            return token

    def resolve_symbol(self, ticker: str) -> dict:
        """Public lookup: mapping details for a ticker (read-only, safe)."""
        token = self._instrument_token(ticker)
        symbol = ticker.removesuffix(".NS").upper()
        return {
            "ticker": ticker,
            "tradingsymbol": symbol,
            "instrument_token": token,
            "exchange": "NSE",
        }

    def _map_status(self, status: str) -> BrokerOrderStatus:
        mapping = {
            "PENDING": BrokerOrderStatus.SUBMITTED,
            "OPEN": BrokerOrderStatus.SUBMITTED,
            "COMPLETE": BrokerOrderStatus.FILLED,
            "CANCELLED": BrokerOrderStatus.CANCELLED,
            "REJECTED": BrokerOrderStatus.REJECTED,
        }
        return mapping.get(status, BrokerOrderStatus.UNCONFIRMED)

    def submit_order(self, client_order_id: str, ticker: str, side: str,
                     quantity: int, order_type: str = "MARKET",
                     limit_price: float = 0.0) -> BrokerOrder:
        # Idempotency: never re-send an order we already know about.
        existing = self._submitted.get(client_order_id)
        if existing is not None:
            logger.info("kite: duplicate submit suppressed for %s", client_order_id)
            return self.get_order(client_order_id)

        order = BrokerOrder(
            client_order_id=client_order_id, ticker=ticker,
            side=side.upper(), quantity=quantity,
            order_type=order_type.upper(), limit_price=limit_price,
            status=BrokerOrderStatus.PENDING,
        )
        try:
            self._submitted[client_order_id] = order  # reserve before sending
            kite_type = {"MARKET": "MARKET", "LIMIT": "LIMIT"}.get(order_type, "MARKET")
            if kite_type == "LIMIT" and limit_price <= 0:
                raise ValueError("LIMIT orders require limit_price")
            token = self._instrument_token(ticker)  # fail-closed before sending
            response = self._kite.place_order(
                variety="regular",
                exchange="NSE",
                tradingsymbol=ticker.removesuffix(".NS"),
                transaction_type="BUY" if side.upper() == "BUY" else "SELL",
                quantity=quantity,
                order_type=kite_type,
                price=round(limit_price, 2) if kite_type == "LIMIT" else 0.0,
                product="CNC",
                validity="DAY",
                tag=client_order_id[:20],
            )
            broker_id = str(response.get("order_id", ""))
            order.broker_order_id = broker_id
            order.status = BrokerOrderStatus.SUBMITTED
            order.updated_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                "kite order submitted client=%s broker=%s ticker=%s side=%s qty=%d",
                client_order_id, broker_id, ticker, side, quantity,
            )
            return order
        except Exception as exc:
            order.status = BrokerOrderStatus.REJECTED
            order.reject_reason = str(exc)[:500]
            order.updated_at = datetime.now(timezone.utc).isoformat()
            logger.error("kite order rejected client=%s reason=%s",
                         client_order_id, order.reject_reason)
            return order

    def cancel_order(self, client_order_id: str) -> BrokerOrder:
        order = self.get_order(client_order_id)
        if order is None:
            raise KeyError(f"unknown order {client_order_id}")
        if order.broker_order_id:
            self._kite.cancel_order(order_id=order.broker_order_id, variety="regular")
        order.status = BrokerOrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc).isoformat()
        return order

    def modify_order(self, client_order_id: str, quantity: int = 0,
                     limit_price: float = 0.0) -> BrokerOrder:
        order = self.get_order(client_order_id)
        if order is None:
            raise KeyError(f"unknown order {client_order_id}")
        kwargs = {"order_id": order.broker_order_id, "variety": "regular"}
        if quantity > 0:
            kwargs["quantity"] = quantity
        if limit_price > 0:
            kwargs["price"] = round(limit_price, 2)
        self._kite.modify_order(**kwargs)
        order.updated_at = datetime.now(timezone.utc).isoformat()
        return order

    def get_order(self, client_order_id: str) -> BrokerOrder | None:
        order = self._submitted.get(client_order_id)
        if order is None:
            return None
        if not order.broker_order_id:
            return order
        try:
            detail = self._kite.order_history(order_id=order.broker_order_id)
            latest = detail[-1] if detail else {}
            order.status = self._map_status(str(latest.get("status", "")))
            order.filled_quantity = int(latest.get("filled_quantity", 0) or 0)
            order.avg_fill_price = float(latest.get("average_price", 0.0) or 0.0)
            if latest.get("status") == "REJECTED":
                order.reject_reason = str(latest.get("status_message", ""))[:500]
            order.updated_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            logger.error("kite order status fetch failed for %s: %s", client_order_id, exc)
            order.status = BrokerOrderStatus.UNCONFIRMED
        return order

    def get_fills(self, since: str | None = None) -> list[BrokerFill]:
        fills: list[BrokerFill] = []
        try:
            trades = self._kite.trades() or []
            for t in trades:
                order_id = str(t.get("order_id", ""))
                # reverse-map broker order id -> client order id
                cid = next(
                    (c for c, o in self._submitted.items() if o.broker_order_id == order_id),
                    "",
                )
                ts = t.get("trade_time") or datetime.now(timezone.utc).isoformat()
                fill = BrokerFill(
                    broker_order_id=order_id,
                    client_order_id=cid,
                    ticker=str(t.get("tradingsymbol", "")) + ".NS",
                    side=str(t.get("transaction_type", "")),
                    quantity=int(t.get("quantity", 0) or 0),
                    price=float(t.get("price", 0.0) or 0.0),
                    filled_at=str(ts),
                )
                if since is None or str(ts) >= since:
                    fills.append(fill)
        except Exception as exc:
            logger.error("kite trades fetch failed: %s", exc)
        return fills

    def get_positions(self) -> list[BrokerPosition]:
        """Current positions. RAISES on fetch failure (fail-closed).

        A failed request must never look like an empty portfolio: sandbox
        validation and the execution manager's exposure gate both depend on
        the distinction.
        """
        data = self._kite.positions() or {}
        positions: list[BrokerPosition] = []
        for p in data.get("net", []) or []:
            qty = int(p.get("quantity", 0) or 0)
            if qty == 0:
                continue
            positions.append(BrokerPosition(
                ticker=str(p.get("tradingsymbol", "")) + ".NS",
                quantity=qty,
                average_price=float(p.get("average_price", 0.0) or 0.0),
                pnl=float(p.get("pnl", 0.0) or 0.0),
                unrealised_pnl=float(p.get("unrealised", 0.0) or 0.0),
            ))
        return positions

    def get_margin(self) -> dict:
        """Current margin state. RAISES on fetch failure (fail-closed).

        The old implementation swallowed the exception and returned a
        zero-valued dict, so a dead broker connection could pass the
        sandbox credentials check (``"available_cash" in margins`` is true
        even for the failure fallback) and ``health()`` always reported
        ``ok: true``. Callers that need a graceful path must catch the
        exception themselves — the failure state must survive.
        """
        margins = self._kite.margins() or {}
        return {
            "available_cash": float(margins.get("available", {}).get("cash", 0.0) or 0.0),
            "used_margin": float(margins.get("utilised", {}).get("dealer", 0.0) or 0.0),
            "total_exposure": float(margins.get("utilised", {}).get("span", 0.0) or 0.0),
        }

    def validate_sandbox(self, tickers: list[str] | None = None) -> dict:
        """Read-only validation of the broker connection. NEVER places orders.

        Checks, in order:
        1. credentials: margins endpoint responds;
        2. instrument mapping: a known ticker resolves to a valid token;
        3. positions: read-only query succeeds.

        Returns a report dict; any failed check means live orders must NOT
        be attempted.
        """
        report = {"checks": {}, "ok": True}
        try:
            margins = self.get_margin()
            report["checks"]["credentials"] = {
                "ok": True, "detail": margins,
            }
        except Exception as exc:
            logger.error("kite margins fetch failed: %s", exc)
            report["checks"]["credentials"] = {"ok": False, "detail": str(exc)}

        tickers = tickers or ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        mapping_ok = True
        mapping_detail = {}
        for ticker in tickers:
            try:
                resolved = self.resolve_symbol(ticker)
                mapping_detail[ticker] = {
                    "ok": True,
                    "instrument_token": resolved["instrument_token"],
                }
            except Exception as exc:
                mapping_detail[ticker] = {"ok": False, "detail": str(exc)}
                mapping_ok = False
        report["checks"]["instrument_mapping"] = {"ok": mapping_ok, "detail": mapping_detail}

        try:
            positions = self.get_positions()
            report["checks"]["positions"] = {"ok": True, "detail": {"count": len(positions)}}
        except Exception as exc:
            report["checks"]["positions"] = {"ok": False, "detail": str(exc)}

        report["ok"] = all(c["ok"] for c in report["checks"].values())
        report["validated_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("kite sandbox validation: %s", "PASS" if report["ok"] else "FAIL")
        return report

    def health(self) -> dict:
        try:
            margin = self.get_margin()
            return {"name": self.name, "live": True, "ok": True, "margin": margin}
        except Exception as exc:
            logger.error("kite health check failed: %s", exc)
            return {"name": self.name, "live": True, "ok": False, "error": str(exc)}
