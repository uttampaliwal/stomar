"""Hardware circuit breakers & risk control engine.

Strict, fail-safe risk layer that gates every order before submission and
freezes the trading engine on material adverse conditions:

    1. Max Daily Loss Limit — block new buys once intraday (mark-to-market)
       losses exceed ``max_daily_loss_pct`` (2%) of total portfolio equity.
    2. Max Portfolio Drawdown — freeze the engine when cumulative drawdown
       exceeds ``max_drawdown_pct`` (8%). Only a manual override token can
       resume trading; the freeze state survives process restarts.
    3. Volatility Scalar — dynamically scale down position sizing when
       VIX > ``vix_threshold`` (22) or ATR expands beyond
       ``atr_expansion_threshold`` (2x) its historical average.
    4. Max Capital Allocation Per Stock — cap single-ticker exposure at
       ``max_stock_allocation_pct`` (15%) of total capital.

Sells / position reductions are never blocked: they reduce risk.

Usage:
    from services.risk_guard import risk_guard
    verdict = risk_guard.check_order(order_value=25000, ticker="RELIANCE.NS",
                                     current_ticker_value=120000)
    if verdict["approved"]:
        engine.submit_order(...)
    risk_guard.update_equity(new_equity=980000)   # called each mark
    risk_guard.resume_trading(override_token="...")  # un-freeze drawdown
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.core.constants import DATA_DIR
from src.core.settings import settings

logger = logging.getLogger(__name__)

STATE_PATH = os.path.join(DATA_DIR, "risk_guard_state.json")

VIX_TICKER = "^INDIAVIX"


@dataclass
class RiskGuardLimits:
    """Tunable limits for the risk guard (defaults match the hard requirements)."""

    max_daily_loss_pct: float = 0.02          # 1: block buys beyond 2% intraday loss
    max_drawdown_pct: float = 0.08            # 2: freeze engine beyond 8% drawdown
    max_stock_allocation_pct: float = 0.15    # 4: cap per-ticker allocation
    vix_threshold: float = 22.0               # 3: VIX level that reduces sizing
    atr_expansion_threshold: float = 2.0      # 3: ATR multiple that reduces sizing
    atr_lookback: int = 20                    # ATR average window (bars)
    volatility_floor: float = 0.25            # never size below 25% of base


def _ist_date() -> str:
    """NSE market date (Asia/Kolkata), fallback to UTC."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def compute_volatility_scalar(
    vix: float | None = None,
    atr_current: float | None = None,
    atr_average: float | None = None,
    limits: RiskGuardLimits | None = None,
) -> dict:
    """Position-size multiplier under high-volatility regimes.

    - VIX factor: 1.0 up to ``vix_threshold``, tapering linearly to the
      volatility floor 10 points above it.
    - ATR factor: 1.0 up to ``atr_expansion_threshold`` x average, tapering
      linearly to the floor one multiple beyond it.
    - Combined scalar is the product of both factors, clamped to the floor.

    Returns the scalar plus per-leg flags so callers can log the reason.
    """
    limits = limits or RiskGuardLimits()

    vix_factor = 1.0
    vix_breach = False
    if vix is not None:
        vix_breach = vix > limits.vix_threshold
        if vix_breach:
            overshoot = vix - limits.vix_threshold
            vix_factor = max(1.0 - overshoot / 10.0 * (1.0 - limits.volatility_floor),
                             limits.volatility_floor)

    atr_factor = 1.0
    atr_expansion = False
    atr_ratio: float | None = None
    if atr_current is not None and atr_average and atr_average > 0:
        atr_ratio = atr_current / atr_average
        atr_expansion = atr_ratio > limits.atr_expansion_threshold
        if atr_expansion:
            overshoot = atr_ratio - limits.atr_expansion_threshold
            atr_factor = max(1.0 - overshoot * (1.0 - limits.volatility_floor),
                             limits.volatility_floor)

    scalar = round(max(vix_factor * atr_factor, limits.volatility_floor), 4)
    return {
        "scalar": scalar,
        "vix_factor": round(vix_factor, 4),
        "atr_factor": round(atr_factor, 4),
        "vix": vix,
        "vix_breach": vix_breach,
        "atr_current": atr_current,
        "atr_average": atr_average,
        "atr_ratio": round(atr_ratio, 4) if atr_ratio is not None else None,
        "atr_expansion": atr_expansion,
        "high_volatility": vix_breach or atr_expansion,
        "volatility_floor": limits.volatility_floor,
    }


class RiskGuard:
    """Strict order-gating risk layer with persistent circuit breakers."""

    def __init__(
        self,
        limits: RiskGuardLimits | None = None,
        initial_equity: float = 1_000_000.0,
        state_path: str | None = None,
        override_token: str | None = None,
    ):
        self.limits = limits or RiskGuardLimits()
        self.initial_equity = initial_equity
        self.state_path = state_path or STATE_PATH
        self.override_token = override_token
        if self.override_token is None:
            self.override_token = settings.risk_override_token

        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.session_start_equity = initial_equity
        self.session_date = _ist_date()
        self.daily_pnl = 0.0
        self.realized_pnl = 0.0
        self.halted = False
        self.halt_reason = ""
        self.halt_timestamp: str | None = None
        self.last_updated = time.time()
        self._lock = threading.RLock()
        self._load_state()

    # ── state / equity tracking ──────────────────────────────────────────

    def update_equity(self, equity: float):
        """Mark-to-market update. Tracks peak, intraday P&L and drawdown.

        Intraday loss is measured against the session's starting equity;
        when the NSE trading date rolls over the session resets.
        """
        with self._lock:
            equity = max(float(equity), 0.0)
            self._roll_session_if_needed(equity)
            self.equity = equity
            self.peak_equity = max(self.peak_equity, equity)
            self.daily_pnl = equity - self.session_start_equity
            self.last_updated = time.time()

            drawdown = self.drawdown_pct()
            if drawdown > self.limits.max_drawdown_pct and not self.halted:
                self._freeze(f"Max drawdown breached ({drawdown:.2%})")
            self._save_state()

    def record_realized_pnl(self, pnl: float):
        """Add realized P&L to the daily tracker (for loss-limit checks)."""
        with self._lock:
            self.realized_pnl += float(pnl)
            self.daily_pnl += float(pnl)
            self._save_state()

    def _roll_session_if_needed(self, equity: float):
        today = _ist_date()
        if today != self.session_date:
            self.session_date = today
            self.session_start_equity = equity
            self.daily_pnl = 0.0
            self.realized_pnl = 0.0
            logger.info("Risk guard: new trading session %s — daily P&L reset", today)

    # ── circuit breakers ─────────────────────────────────────────────────

    def _freeze(self, reason: str):
        self.halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.now().isoformat(timespec="seconds")
        logger.critical("RISK GUARD FREEZE: %s", reason)

    def resume_trading(self, override_token: str) -> dict:
        """Manual override to resume after a freeze.

        The token must match the configured override token
        (constant-time comparison). Without a configured token the
        freeze cannot be lifted — by design.
        """
        with self._lock:
            if not self.halted:
                return {"resumed": False, "reason": "Trading is not halted"}
            if not self.override_token:
                return {
                    "resumed": False,
                    "reason": "No override token configured (STOMAR_RISK_OVERRIDE_TOKEN)",
                }
            if not override_token or not hmac.compare_digest(
                override_token, self.override_token
            ):
                logger.warning("Risk guard: invalid override token attempt")
                return {"resumed": False, "reason": "Invalid override token"}
            self.halted = False
            self.halt_reason = ""
            self.halt_timestamp = None
            logger.critical("RISK GUARD: manual override accepted — trading resumed")
            self._save_state()
            return {"resumed": True, "reason": "Manual override accepted"}

    def drawdown_pct(self) -> float:
        """Cumulative drawdown from peak equity (0.0 .. 1.0)."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def daily_loss_pct(self) -> float:
        """Intraday loss as a fraction of session-start equity (negative = loss)."""
        if self.session_start_equity <= 0:
            return 0.0
        return self.daily_pnl / self.session_start_equity

    def daily_loss_limit_value(self) -> float:
        return self.equity * self.limits.max_daily_loss_pct

    def max_allocation_value(self) -> float:
        """Maximum capital allocatable to a single ticker (15% by default)."""
        return self.equity * self.limits.max_stock_allocation_pct

    # ── order gating ─────────────────────────────────────────────────────

    def check_order(
        self,
        order_value: float,
        ticker: str = "",
        current_ticker_value: float = 0.0,
        is_buy: bool = True,
        is_closing: bool = False,
    ) -> dict:
        """Gate an order against all active circuit breakers.

        Args:
            order_value: Notional value of the proposed order (₹).
            ticker: Symbol being traded (for allocation checks).
            current_ticker_value: Value of the existing position in *ticker*.
            is_buy: False for sells (never blocked — they reduce risk).
            is_closing: True when the order fully/partially closes a position.

        Returns:
            Dict with approved (bool), reason, checks, plus guard state.
        """
        with self._lock:
            checks: list[dict] = []
            equity = self.equity

            # 1. Drawdown freeze — hard stop on new exposure; position
            # closing orders are always allowed (they reduce risk)
            if self.halted:
                if not is_buy and is_closing:
                    checks.append({"passed": True, "check": "drawdown_freeze"})
                else:
                    return {
                        "approved": False,
                        "reason": f"Trading frozen: {self.halt_reason}",
                        "checks": [{"passed": False, "check": "drawdown_freeze"}],
                        "drawdown_pct": self.drawdown_pct(),
                        "daily_loss_pct": self.daily_loss_pct(),
                        "halted": True,
                    }

            # 2. Max daily loss — block new buys, never sells
            daily_loss_breach = self.daily_pnl < -self.daily_loss_limit_value()
            if is_buy and not is_closing and daily_loss_breach:
                checks.append({
                    "passed": False, "check": "daily_loss",
                    "message": (
                        f"Intraday loss {self.daily_pnl:,.2f} exceeds "
                        f"{-self.daily_loss_limit_value():,.2f} ("
                        f"{self.limits.max_daily_loss_pct:.1%} of equity) — buys blocked"
                    ),
                })
            else:
                checks.append({"passed": True, "check": "daily_loss"})

            # 3. Max allocation per stock — cap single-ticker exposure at 15%
            total_ticker = current_ticker_value + (order_value if is_buy else 0.0)
            max_allocation = self.max_allocation_value()
            if is_buy and not is_closing and total_ticker > max_allocation:
                checks.append({
                    "passed": False, "check": "max_allocation",
                    "message": (
                        f"{ticker} exposure {total_ticker:,.2f} exceeds "
                        f"cap {max_allocation:,.2f} "
                        f"({self.limits.max_stock_allocation_pct:.1%} of equity)"
                    ),
                })
            else:
                checks.append({"passed": True, "check": "max_allocation"})

            approved = all(c["passed"] for c in checks)
            return {
                "approved": approved,
                "reason": "" if approved else next(
                    c["message"] for c in checks if not c["passed"]
                ),
                "checks": checks,
                "drawdown_pct": self.drawdown_pct(),
                "daily_loss_pct": self.daily_loss_pct(),
                "halted": self.halted,
            }

    # ── volatility scalar ────────────────────────────────────────────────

    def get_volatility_scalar(
        self,
        symbol: str | None = None,
        vix: float | None = None,
        atr_current: float | None = None,
        atr_average: float | None = None,
    ) -> dict:
        """Volatility scalar for the current regime.

        When values are omitted they are fetched live: India VIX from
        ``^INDIAVIX`` and the symbol's ATR(14) vs its 20-bar average via the
        market data engine. Any fetch failure degrades gracefully to a 1.0
        factor for that leg (fail-open, sizing is also capped elsewhere).
        """
        if vix is None:
            vix = self._fetch_vix()
        if (atr_current is None or atr_average is None) and symbol:
            atr_current, atr_average = self._fetch_atr(symbol)
        return compute_volatility_scalar(
            vix=vix, atr_current=atr_current, atr_average=atr_average,
            limits=self.limits,
        )

    def scaled_position_size(self, base_value: float, **vol_kwargs) -> dict:
        """Apply the volatility scalar to a proposed position size."""
        vol = self.get_volatility_scalar(**vol_kwargs)
        return {
            "base_value": base_value,
            "scaled_value": round(base_value * vol["scalar"], 2),
            **vol,
        }

    @staticmethod
    def _fetch_vix() -> float | None:
        try:
            from services.market_data import market_data_service

            payload = market_data_service.get_candles(
                VIX_TICKER, interval="1d", limit=1, indicators=False
            )
            bars = payload.get("bars") or []
            return float(bars[-1]["close"]) if bars else None
        except Exception as exc:
            logger.debug("VIX fetch failed: %s", exc)
            return None

    @staticmethod
    def _fetch_atr(symbol: str) -> tuple[float | None, float | None]:
        """Return (current ATR, average ATR over the lookback window)."""
        try:
            from services.market_data import market_data_service

            payload = market_data_service.get_candles(
                symbol, interval="1d", limit=60, indicators=True
            )
            atr_series = pd.Series(
                [b.get("atr") for b in payload.get("bars") or []]
            ).dropna()
            if atr_series.empty:
                return None, None
            current = float(atr_series.iloc[-1])
            avg = float(atr_series.iloc[:-1].tail(20).mean())
            return current, avg
        except Exception as exc:
            logger.debug("ATR fetch failed for %s: %s", symbol, exc)
            return None, None

    # ── persistence ──────────────────────────────────────────────────────

    def _save_state(self):
        """Persist breaker state so a process restart cannot silently clear
        a drawdown freeze."""
        try:
            state = {
                "equity": self.equity,
                "peak_equity": self.peak_equity,
                "session_start_equity": self.session_start_equity,
                "session_date": self.session_date,
                "daily_pnl": self.daily_pnl,
                "realized_pnl": self.realized_pnl,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "halt_timestamp": self.halt_timestamp,
            }
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            logger.debug("risk guard state save failed: %s", exc)

    def _load_state(self):
        """Restore persistent state (halt survives restarts)."""
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, encoding="utf-8") as fh:
                state = json.load(fh)
            self.equity = float(state.get("equity", self.initial_equity))
            self.peak_equity = float(state.get("peak_equity", self.equity))
            self.session_start_equity = float(
                state.get("session_start_equity", self.equity)
            )
            self.session_date = state.get("session_date", _ist_date())
            self.daily_pnl = float(state.get("daily_pnl", 0.0))
            self.realized_pnl = float(state.get("realized_pnl", 0.0))
            self.halted = bool(state.get("halted", False))
            self.halt_reason = state.get("halt_reason", "")
            self.halt_timestamp = state.get("halt_timestamp")
            if self.halted:
                logger.critical(
                    "RISK GUARD restored in FROZEN state: %s", self.halt_reason
                )
            # reset the daily session if the persisted date is stale
            self._roll_session_if_needed(self.equity)
        except Exception as exc:
            logger.warning("risk guard state load failed: %s", exc)

    # ── status ───────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            return {
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "halt_timestamp": self.halt_timestamp,
                "equity": self.equity,
                "peak_equity": self.peak_equity,
                "drawdown_pct": round(self.drawdown_pct(), 6),
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "daily_pnl": self.daily_pnl,
                "daily_loss_pct": round(self.daily_loss_pct(), 6),
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "daily_loss_limit_value": round(self.daily_loss_limit_value(), 2),
                "realized_pnl": self.realized_pnl,
                "session_date": self.session_date,
                "max_allocation_per_stock": self.max_allocation_value(),
                "max_stock_allocation_pct": self.limits.max_stock_allocation_pct,
                "vix_threshold": self.limits.vix_threshold,
                "atr_expansion_threshold": self.limits.atr_expansion_threshold,
                "override_token_configured": bool(self.override_token),
                "state_path": self.state_path,
            }


_risk_guard: RiskGuard | None = None
_guard_lock = threading.Lock()


def get_risk_guard() -> RiskGuard:
    """Process-wide risk guard singleton."""
    global _risk_guard
    with _guard_lock:
        if _risk_guard is None:
            _risk_guard = RiskGuard()
        return _risk_guard


risk_guard = get_risk_guard()
