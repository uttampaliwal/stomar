"""Broker factory — the ONLY place brokers are constructed.

``get_broker()`` returns a dry-run broker unless the full live-trading gate
(``src.core.trading_mode.require_live_allowed``) passes AND the configured
broker is available. Live mode can never be activated accidentally.
"""

from __future__ import annotations

import logging

from src.brokers.base import BrokerAdapter
from src.core import trading_mode

logger = logging.getLogger(__name__)

_singleton: BrokerAdapter | None = None
_live_refused_reason: str = ""


def get_broker(force_live: bool = False) -> BrokerAdapter:
    """Return the configured broker.

    Default: dry-run broker (paper trading), which is always safe.
    Live: only when ``force_live`` is True and the live gate passes.
    """
    global _singleton, _live_refused_reason

    if _singleton is not None and (not force_live or _singleton.is_live):
        return _singleton

    if not force_live:
        from src.brokers.dryrun import DryRunBroker
        _singleton = DryRunBroker()
        return _singleton

    # ── live path ──────────────────────────────────────────────────────────
    _live_refused_reason = ""
    check = trading_mode.live_gate_status()
    if not check["live_allowed"]:
        missing_parts = []
        if not check.get("live_trading_enabled"):
            missing_parts.append("STOMAR_LIVE_TRADING=true")
        if not check.get("broker_configured"):
            missing_parts.append("broker credentials configured")
        if not check.get("account_approved"):
            missing_parts.append("STOMAR_LIVE_ACCOUNT_APPROVED=true")
        if not check.get("confirmation_acknowledged"):
            missing_parts.append("STOMAR_LIVE_CONFIRMATION phrase")
        _live_refused_reason = "; ".join(missing_parts)
        logger.error("live broker refused — missing requirements: %s", _live_refused_reason)
        raise trading_mode.LiveTradingNotEnabledError(
            f"live trading not enabled: {_live_refused_reason or 'gate unmet'}"
        )

    from src.core.settings import settings
    broker_name = settings.live_broker or "kite"
    if broker_name != "kite":
        raise ValueError(f"unsupported live_broker: {broker_name}")
    from src.brokers.kite import KiteLiveBroker
    _singleton = KiteLiveBroker()
    logger.warning("LIVE broker active: %s", _singleton.name)
    return _singleton


def reset_broker():
    """Drop the cached broker (tests / config reloads)."""
    global _singleton, _live_refused_reason
    _singleton = None
    _live_refused_reason = ""


def live_refused_reason() -> str:
    return _live_refused_reason
