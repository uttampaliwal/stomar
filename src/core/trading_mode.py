"""Trading mode gating: paper by default, live only behind explicit opt-in.

Live trading requires ALL of the following to be true, so that merely
having environment variables present can never accidentally enable it:

* ``STOMAR_LIVE_TRADING=true`` explicitly set;
* a broker is configured (api key + access token present);
* ``STOMAR_LIVE_ACCOUNT_APPROVED=true`` explicitly set (broker sandbox or
  approved live account);
* ``STOMAR_LIVE_CONFIRMATION`` equals the required confirmation phrase
  (a deliberate, human-readable acknowledgement);
* the pre-trade risk gate approves the specific order at execution time.

Every path that can touch a broker consults :func:`get_trading_mode` /
:func:`live_gate_status` before doing anything.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

from src.core.settings import settings

logger = logging.getLogger(__name__)

# The exact string a human must set as STOMAR_LIVE_CONFIRMATION.
LIVE_CONFIRMATION_PHRASE = "I_CONFIRM_REAL_MONEY_TRADING"

_LIVE_ENV = "STOMAR_LIVE_TRADING"
_APPROVED_ENV = "STOMAR_LIVE_ACCOUNT_APPROVED"
_CONFIRM_ENV = "STOMAR_LIVE_CONFIRMATION"
_KEY_ENV = "STOMAR_KITE_API_KEY"
_TOKEN_ENV = "STOMAR_KITE_ACCESS_TOKEN"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"

    @property
    def is_live(self) -> bool:
        return self is TradingMode.LIVE


class LiveTradingNotEnabledError(RuntimeError):
    """Raised when live trading is attempted without the full opt-in gate."""


def _env_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _env_or_setting(name: str, setting_name: str, default: str = "") -> str:
    """Read the live gate directly from the environment, so that gate
    decisions always reflect the CURRENT process environment and are not
    frozen at settings-import time."""
    return os.environ.get(name, str(getattr(settings, setting_name, default) or ""))


def live_gate_status() -> dict:
    """Return a full breakdown of the live-trading gate (safe to expose)."""
    explicit = _env_bool(_env_or_setting(_LIVE_ENV, "live_trading", "false"))
    broker_configured = bool(
        _env_or_setting(_KEY_ENV, "kite_api_key")
        and _env_or_setting(_TOKEN_ENV, "kite_access_token")
    )
    account_approved = _env_bool(_env_or_setting(_APPROVED_ENV, "live_account_approved", "false"))
    confirmation_ok = (
        _env_or_setting(_CONFIRM_ENV, "live_confirmation").strip()
        == LIVE_CONFIRMATION_PHRASE
    )
    allowed = (
        explicit and broker_configured and account_approved and confirmation_ok
    )
    return {
        "mode": TradingMode.LIVE.value if allowed else TradingMode.PAPER.value,
        "live_trading_enabled": explicit,
        "broker_configured": broker_configured,
        "account_approved": account_approved,
        "confirmation_acknowledged": confirmation_ok,
        "live_allowed": allowed,
    }


def is_live_enabled() -> bool:
    """True only when every gate requirement passes."""
    return live_gate_status()["live_allowed"]


def get_trading_mode() -> TradingMode:
    """The effective mode of the process. Defaults to PAPER — always."""
    return TradingMode.LIVE if is_live_enabled() else TradingMode.PAPER


def require_live_allowed() -> None:
    """Raise LiveTradingNotEnabledError unless the full gate passes."""
    if not is_live_enabled():
        status = live_gate_status()
        missing = [
            name for name, ok in {
                "STOMAR_LIVE_TRADING=true": status["live_trading_enabled"],
                "broker credentials configured": status["broker_configured"],
                "STOMAR_LIVE_ACCOUNT_APPROVED=true": status["account_approved"],
                f"STOMAR_LIVE_CONFIRMATION={LIVE_CONFIRMATION_PHRASE}": status["confirmation_acknowledged"],
            }.items()
            if not ok
        ]
        raise LiveTradingNotEnabledError(
            "Live trading is not enabled. Paper mode is the default. "
            f"Missing requirements: {', '.join(missing)}"
        )


def mode_banner() -> str:
    """Human-readable banner used in logs and API payloads."""
    if is_live_enabled():
        return "LIVE MODE — REAL MONEY TRADING ENABLED. Risk controls active."
    return "PAPER MODE — simulated trading only. No real orders can be placed."


def log_mode_change(reason: str) -> None:
    mode = get_trading_mode()
    logger.warning(
        "TRADING MODE: %s — %s", "LIVE" if mode.is_live else "PAPER", reason
    )
