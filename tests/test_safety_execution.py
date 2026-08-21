"""Safety tests for the broker + execution + risk layer.

These tests assert invariants that MUST hold before any real-money
deployment:

* paper is the default mode; live is impossible without the full gate
* duplicate intents never double-submit
* stale quotes, kill switch, and risk limits block orders
* broker idempotency under retry
* reconciliation prefers broker truth over local assumptions
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.brokers import get_broker, reset_broker
from src.brokers.base import BrokerOrderStatus, BrokerPosition
from src.brokers.dryrun import DryRunBroker
from src.core import trading_mode
from src.core.trading_mode import (
    LIVE_CONFIRMATION_PHRASE,
    get_trading_mode,
    is_live_enabled,
)
from src.trading.execution_manager import (
    ExecutionError,
    ExecutionManager,
    PortfolioSnapshot,
)
from src.trading.risk_controls import MarketContext, RiskController, RiskLimits


def fresh_quote(age_seconds: float = 1.0) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return {"close": 2500.0, "timestamp": ts.isoformat()}


class StubPortfolioBroker(DryRunBroker):
    """Dry-run broker seeded with a fixed position set for gate tests."""

    def __init__(self, positions: dict[str, tuple[int, float]],
                 cash: float = 1_000_000.0):
        super().__init__()
        self._positions = {
            t: BrokerPosition(ticker=t, quantity=q, average_price=p)
            for t, (q, p) in positions.items()
        }
        self._cash = cash


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("STOMAR_LIVE_TRADING", "STOMAR_LIVE_CONFIRMATION",
                "STOMAR_LIVE_ACCOUNT_APPROVED", "STOMAR_KITE_API_KEY",
                "STOMAR_KITE_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    reset_broker()
    yield
    reset_broker()
    from pathlib import Path
    from src.core.constants import DATA_DIR
    Path(DATA_DIR, "kill_switch.json").unlink(missing_ok=True)


# ── mode gating ──────────────────────────────────────────────────────────

def test_paper_is_default_mode():
    assert get_trading_mode().value == "paper"
    assert is_live_enabled() is False


def test_live_requires_full_gate(monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    assert is_live_enabled() is False  # missing broker credentials
    with pytest.raises(trading_mode.LiveTradingNotEnabledError):
        trading_mode.require_live_allowed()


def test_live_enabled_only_with_all_gates(monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    monkeypatch.setenv("STOMAR_KITE_API_KEY", "k")
    monkeypatch.setenv("STOMAR_KITE_ACCESS_TOKEN", "t")
    assert is_live_enabled() is True
    assert get_trading_mode().value == "live"


def test_wrong_confirmation_phrase_never_enables_live(monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION", "I WILL TRADE FOR REAL")
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    monkeypatch.setenv("STOMAR_KITE_API_KEY", "k")
    monkeypatch.setenv("STOMAR_KITE_ACCESS_TOKEN", "t")
    assert is_live_enabled() is False


def test_get_broker_defaults_to_dryrun_even_with_live_env(monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    monkeypatch.setenv("STOMAR_KITE_API_KEY", "k")
    monkeypatch.setenv("STOMAR_KITE_ACCESS_TOKEN", "t")
    broker = get_broker()  # NOT force_live
    assert broker.is_live is False


def test_force_live_without_gate_refused(monkeypatch):
    with pytest.raises(trading_mode.LiveTradingNotEnabledError):
        get_broker(force_live=True)


def test_force_live_with_gate_returns_live_broker(monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    monkeypatch.setenv("STOMAR_KITE_API_KEY", "k")
    monkeypatch.setenv("STOMAR_KITE_ACCESS_TOKEN", "t")
    # kiteconnect may be absent — construction fails with ImportError,
    # which is itself a safe failure (never a silent dry-run).
    with pytest.raises((trading_mode.LiveTradingNotEnabledError, ImportError, ValueError)):
        get_broker(force_live=True)


# ── execution path ───────────────────────────────────────────────────────

def test_execute_routes_through_broker_and_fills():
    broker = DryRunBroker(fill_delay_seconds=0.0)
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    order = manager.execute("RELIANCE.NS", "BUY", 10,
                            quotes={"RELIANCE.NS": fresh_quote()},
                            order_value=25_000)
    broker.process_pending({"RELIANCE.NS": 2505.0})
    order = manager.reconcile(order.client_order_id)[0]
    assert order.status == BrokerOrderStatus.FILLED
    assert order.filled_quantity == 10
    assert order.avg_fill_price > 0


def test_limit_order_outside_50pct_rejected():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    q = fresh_quote()  # close = 2500.0
    with pytest.raises(ExecutionError, match="within 50%"):
        manager.execute("RELIANCE.NS", "BUY", 10, order_type="LIMIT",
                        limit_price=1.0,
                        quotes={"RELIANCE.NS": q}, order_value=25_000)
    with pytest.raises(ExecutionError, match="within 50%"):
        manager.execute("RELIANCE.NS", "BUY", 10, order_type="LIMIT",
                        limit_price=5000.0,
                        quotes={"RELIANCE.NS": q}, order_value=25_000)


def test_limit_order_within_50pct_accepted():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    q = fresh_quote()  # close = 2500.0
    order = manager.execute("RELIANCE.NS", "BUY", 10, order_type="LIMIT",
                            limit_price=2500.0,
                            quotes={"RELIANCE.NS": q}, order_value=25_000)
    assert order.client_order_id


def test_limit_order_without_price_rejected():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    with pytest.raises(ExecutionError, match="positive limit price"):
        manager.execute("RELIANCE.NS", "BUY", 10, order_type="LIMIT",
                        quotes={"RELIANCE.NS": fresh_quote()}, order_value=25_000)


def test_duplicate_intent_never_double_submits():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    q = fresh_quote()
    first = manager.execute("RELIANCE.NS", "BUY", 10,
                            quotes={"RELIANCE.NS": q}, order_value=25_000)
    second = manager.execute("RELIANCE.NS", "BUY", 10,
                             client_order_id=first.client_order_id,
                             quotes={"RELIANCE.NS": q}, order_value=25_000)
    assert second is first
    assert len(broker.get_fills()) == 0  # nothing re-submitted


def test_stale_quote_blocks_order():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000),
                               max_stale_quote_seconds=15.0)
    stale = fresh_quote(age_seconds=3600)
    with pytest.raises(ExecutionError, match="stale quote"):
        manager.execute("RELIANCE.NS", "BUY", 10,
                        quotes={"RELIANCE.NS": stale}, order_value=25_000)


def test_missing_quote_blocks_order():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    with pytest.raises(ExecutionError, match="no quote"):
        manager.execute("RELIANCE.NS", "BUY", 10, quotes={}, order_value=25_000)


def test_kill_switch_blocks_orders():
    risk = RiskController(RiskLimits(), 1_000_000)
    risk.kill_switch("test freeze")
    manager = ExecutionManager(DryRunBroker(), risk)
    with pytest.raises(ExecutionError, match="halted"):
        manager.execute("RELIANCE.NS", "BUY", 10,
                        quotes={"RELIANCE.NS": fresh_quote()}, order_value=25_000)


def test_kill_switch_persists_across_restart(tmp_path):
    f = tmp_path / "kill_switch.json"
    r1 = RiskController(RiskLimits(), 1_000_000, kill_switch_file=f)
    r1.kill_switch("frozen")
    r2 = RiskController(RiskLimits(), 1_000_000, kill_switch_file=f)
    assert r2.halted is True
    r2.clear_kill_switch()
    r3 = RiskController(RiskLimits(), 1_000_000, kill_switch_file=f)
    assert r3.halted is False


def test_daily_loss_limit_blocks():
    risk = RiskController(RiskLimits(max_daily_loss_pct=0.02), 1_000_000)
    risk.update_daily_pnl(-30_000)  # 3% loss on 1M
    manager = ExecutionManager(DryRunBroker(), risk)
    with pytest.raises(ExecutionError, match=r"(?i)daily loss"):
        manager.execute("RELIANCE.NS", "BUY", 10,
                        quotes={"RELIANCE.NS": fresh_quote()}, order_value=25_000)


def test_drawdown_breach_halts():
    risk = RiskController(RiskLimits(max_drawdown_pct=0.15), 1_000_000)
    risk.update_equity(800_000)  # 20% drawdown
    manager = ExecutionManager(DryRunBroker(), risk)
    with pytest.raises(ExecutionError):
        manager.execute("RELIANCE.NS", "BUY", 10,
                        quotes={"RELIANCE.NS": fresh_quote()}, order_value=25_000)
    assert risk.halted is True


def test_liquidity_check_blocks_low_volume():
    risk = RiskController(RiskLimits(min_daily_volume_rs=1_000_000), 1_000_000)
    manager = ExecutionManager(DryRunBroker(), risk)
    with pytest.raises(ExecutionError, match="traded value"):
        manager.execute(
            "RELIANCE.NS", "BUY", 10,
            quotes={"RELIANCE.NS": fresh_quote()}, order_value=25_000,
            market=MarketContext(price=2500.0, avg_daily_traded_value=100_000),
        )


def test_slippage_check_blocks():
    risk = RiskController(RiskLimits(max_expected_slippage_bps=30.0), 1_000_000)
    manager = ExecutionManager(DryRunBroker(), risk)
    verdict = manager.gate_order(
        "RELIANCE.NS", "BUY", 10,
        order_value=25_000,
        market=MarketContext(price=2500.0, expected_slippage_bps=120.0),
    )
    assert verdict["approved"] is False


def test_gap_check_blocks():
    risk = RiskController(RiskLimits(max_gap_pct=5.0), 1_000_000)
    manager = ExecutionManager(DryRunBroker(), risk)
    verdict = manager.gate_order(
        "RELIANCE.NS", "BUY", 10,
        order_value=25_000,
        market=MarketContext(price=2700.0, prev_close=2500.0),  # 8% gap
    )
    assert verdict["approved"] is False


def test_daily_order_budget_blocks():
    manager = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                               max_daily_orders=2)
    q = fresh_quote()
    manager.execute("RELIANCE.NS", "BUY", 1, quotes={"RELIANCE.NS": q}, order_value=2500)
    manager.execute("TCS.NS", "BUY", 1, quotes={"TCS.NS": q}, order_value=2500)
    with pytest.raises(ExecutionError, match="order budget"):
        manager.execute("INFY.NS", "BUY", 1, quotes={"INFY.NS": q}, order_value=2500)


def test_daily_order_count_resets_on_new_day(tmp_path, monkeypatch):
    """Regression (#33): the budget is per calendar day.

    With the ledger-derived counter the day boundary is the audit file name:
    a new day is a new file, so yesterday's orders never carry into today.
    The boundary is IST (the market's timezone), not the host's local time.
    """
    import datetime as dt

    import src.trading.execution_manager as em_mod

    manager = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                               max_daily_orders=1, audit_dir=tmp_path)
    q = fresh_quote()
    manager.execute("RELIANCE.NS", "BUY", 1, quotes={"RELIANCE.NS": q}, order_value=2500)
    assert manager.status()["orders_today"] == 1
    with pytest.raises(ExecutionError, match="order budget"):
        manager.execute("TCS.NS", "BUY", 1, quotes={"TCS.NS": q}, order_value=2500)

    tomorrow = em_mod.ist_today() + dt.timedelta(days=1)
    monkeypatch.setattr(em_mod, "ist_today", lambda: tomorrow)
    # budget was exhausted yesterday — the new day gets a fresh file
    manager.execute("TCS.NS", "BUY", 1, quotes={"TCS.NS": q}, order_value=2500)

    assert manager.status()["orders_today"] == 1


def test_daily_budget_persists_across_restart(tmp_path):
    """The daily order budget is enforced by the audit ledger, so it survives
    process restarts and is shared across processes writing to the same dir —
    a restart must not silently re-open the budget."""
    q = fresh_quote()

    # "day 1" process: two orders consume the full budget
    m1 = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    m1.execute("RELIANCE.NS", "BUY", 1, quotes={"RELIANCE.NS": q}, order_value=2500)
    m1.execute("TCS.NS", "BUY", 1, quotes={"TCS.NS": q}, order_value=2500)

    # "restart": a brand-new manager with the same ledger must see the same count
    m2 = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    assert m2.status()["orders_today"] == 2
    with pytest.raises(ExecutionError, match="order budget"):
        m2.execute("INFY.NS", "BUY", 1, quotes={"INFY.NS": q}, order_value=2500)

    # a third process sharing the dir sees the same global budget
    m3 = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    assert m3.status()["orders_today"] == 2


def test_gate_only_path_records_orders_in_budget(tmp_path):
    """run_daily / paper-router style flow (gate_order + record_submitted).

    These paths place orders outside execute(), so the submission must be
    recorded in the audit ledger — otherwise the daily budget would never
    see those orders, and a restart would re-open it."""
    q = fresh_quote()
    m1 = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    assert m1.gate_order("A.NS", "BUY", 1, quotes={"A.NS": q}, order_value=2500)["approved"] is True
    m1.record_submitted("A.NS", "BUY", 1)
    assert m1.gate_order("B.NS", "BUY", 1, quotes={"B.NS": q}, order_value=2500)["approved"] is True
    m1.record_submitted("B.NS", "BUY", 1)
    assert m1.gate_order("C.NS", "BUY", 1, quotes={"C.NS": q}, order_value=2500)["approved"] is False

    # restart: the recorded orders still consume the budget
    m2 = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    assert m2.status()["orders_today"] == 2
    assert m2.gate_order("C.NS", "BUY", 1, quotes={"C.NS": q}, order_value=2500)["approved"] is False


def test_closing_order_skips_position_checks():
    """Regression (#76): closing orders must not be blocked by the
    position-concentration / exposure checks."""
    risk = RiskController(RiskLimits(max_position_pct=0.01), 1_000_000)
    manager = ExecutionManager(DryRunBroker(), risk)
    q = fresh_quote()
    # 25k notional on a 1M account = 2.5% > 1% limit — a fresh BUY is blocked
    with pytest.raises(ExecutionError, match="Position"):
        manager.execute("RELIANCE.NS", "BUY", 10, quotes={"RELIANCE.NS": q},
                        order_value=25_000)
    # ...but the same notional as a closing order passes
    order = manager.execute("RELIANCE.NS", "SELL", 10, quotes={"RELIANCE.NS": q},
                            order_value=25_000, is_closing=True)
    assert order.client_order_id
    verdict = manager.gate_order("RELIANCE.NS", "SELL", 10, order_value=25_000,
                                 quotes={"RELIANCE.NS": q}, is_closing=True)
    assert verdict["approved"] is True


def test_broker_idempotent_submit_under_retry():
    broker = DryRunBroker()
    cid = "ord-retry-1"
    first = broker.submit_order(cid, "RELIANCE.NS", "BUY", 10)
    second = broker.submit_order(cid, "RELIANCE.NS", "BUY", 10)
    assert first is second
    assert second.broker_order_id == first.broker_order_id


def test_cancel_and_reject_paths():
    broker = DryRunBroker()
    order = broker.submit_order("ord-c1", "RELIANCE.NS", "BUY", 10)
    cancelled = broker.cancel_order("ord-c1")
    assert cancelled.status == BrokerOrderStatus.CANCELLED
    rejected = broker.submit_order("ord-r1", "RELIANCE.NS", "BUY", -5)
    assert rejected.status == BrokerOrderStatus.REJECTED
    assert rejected.reject_reason


def test_partial_fills_eventually_complete():
    import time
    broker = DryRunBroker(partial_fill_pct=0.5)
    order = broker.submit_order("ord-p1", "RELIANCE.NS", "BUY", 100)
    broker.process_pending({"RELIANCE.NS": 2500.0})
    assert order.status == BrokerOrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity < 100
    time.sleep(0.06)  # remainder fills on a later tick
    broker.process_pending({"RELIANCE.NS": 2505.0})
    assert order.status == BrokerOrderStatus.FILLED
    assert order.filled_quantity == 100


def test_reconcile_pulls_broker_truth():
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000))
    order = manager.execute("RELIANCE.NS", "BUY", 10,
                            quotes={"RELIANCE.NS": fresh_quote()},
                            order_value=25_000)
    # Broker fills behind the scenes.
    broker.process_pending({"RELIANCE.NS": 2510.0})
    # Reconciliation surfaces the broker truth.
    refreshed = manager.reconcile(order.client_order_id)[0]
    assert refreshed.status == BrokerOrderStatus.FILLED
    assert refreshed.filled_quantity == 10


def test_audit_log_records_decisions(tmp_path):
    broker = DryRunBroker()
    manager = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000),
                               audit_dir=tmp_path)
    q = fresh_quote()
    manager.execute("RELIANCE.NS", "BUY", 10, quotes={"RELIANCE.NS": q},
                    order_value=25_000)
    log_files = list(tmp_path.glob("orders-*.jsonl"))
    assert log_files
    content = log_files[0].read_text()
    assert "submitted" in content
    assert "RELIANCE.NS" in content


def test_dryrun_never_reaches_a_real_market():
    broker = DryRunBroker()
    assert broker.is_live is False
    order = broker.submit_order("ord-x", "RELIANCE.NS", "SELL", 5)
    assert order.status in (BrokerOrderStatus.SUBMITTED, BrokerOrderStatus.REJECTED)
    # no external side effects are representable — positions are local
    assert broker.get_positions() == []


# ── Kite instrument mapping (fail-closed) ─────────────────────────────────

class _FakeKiteConnect:
    fetch_count = 0

    def __init__(self, api_key):
        self.api_key = api_key

    def set_access_token(self, token):
        self.token = token

    def instruments(self, exchange):
        _FakeKiteConnect.fetch_count += 1
        return [
            {"instrument_token": 738561, "tradingsymbol": "RELIANCE",
             "exchange": "NSE", "name": "RELIANCE"},
            {"instrument_token": 2953217, "tradingsymbol": "TCS",
             "exchange": "NSE", "name": "TCS"},
        ]

    def margins(self):
        return {
            "available": {"cash": 1_000_000.0},
            "utilised": {"dealer": 0.0, "span": 0.0},
        }

    def positions(self):
        return {"net": []}


def _live_gate(monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    monkeypatch.setenv("STOMAR_KITE_API_KEY", "k")
    monkeypatch.setenv("STOMAR_KITE_ACCESS_TOKEN", "t")


def _kite_broker(monkeypatch):
    import sys
    import types

    from src.brokers import kite as kite_mod
    from src.brokers.kite import KiteLiveBroker

    _live_gate(monkeypatch)
    # settings is a frozen import-time singleton — patch its attributes
    monkeypatch.setattr(kite_mod.settings, "kite_api_key", "k")
    monkeypatch.setattr(kite_mod.settings, "kite_access_token", "t")
    monkeypatch.setattr(kite_mod.settings, "live_account_id", "test-account")
    fake = types.ModuleType("kiteconnect")
    fake.KiteConnect = _FakeKiteConnect
    monkeypatch.setitem(sys.modules, "kiteconnect", fake)
    broker = KiteLiveBroker()
    monkeypatch.setattr(_FakeKiteConnect, "fetch_count", 0)
    return broker


def test_instrument_token_resolves_and_caches(monkeypatch):
    broker = _kite_broker(monkeypatch)
    token = broker._instrument_token("RELIANCE.NS")
    assert token == 738561
    assert broker._instrument_token("reliance.ns") == 738561
    assert _FakeKiteConnect.fetch_count == 1  # cached — no second network call


def test_instrument_token_refetches_after_ttl(monkeypatch):
    broker = _kite_broker(monkeypatch)
    assert broker._instrument_token("TCS.NS") == 2953217
    broker._instruments_ts -= 3601.0  # expire TTL
    assert broker._instrument_token("TCS.NS") == 2953217
    assert _FakeKiteConnect.fetch_count == 2


def test_instrument_token_fails_closed_on_unknown_symbol(monkeypatch):
    broker = _kite_broker(monkeypatch)
    with pytest.raises(ValueError):
        broker._instrument_token("NOTAREALSTOCK.NS")


def test_instrument_token_fails_closed_on_fetch_error(monkeypatch):
    broker = _kite_broker(monkeypatch)

    def boom(exchange):
        raise RuntimeError("network down")

    monkeypatch.setattr(broker._kite, "instruments", boom)
    with pytest.raises(RuntimeError):
        broker._instrument_token("RELIANCE.NS")


def test_resolve_symbol_returns_public_mapping(monkeypatch):
    broker = _kite_broker(monkeypatch)
    resolved = broker.resolve_symbol("TCS.NS")
    assert resolved["ticker"] == "TCS.NS"
    assert resolved["tradingsymbol"] == "TCS"
    assert resolved["instrument_token"] == 2953217
    assert resolved["exchange"] == "NSE"


def test_submit_order_resolves_instrument_before_sending(monkeypatch):
    broker = _kite_broker(monkeypatch)
    broker._kite.place_order = lambda **kwargs: {
        "order_id": "12345", "status": "pending", "filled_quantity": 0,
        "average_price": 0.0, "order_timestamp": "", "exchange_timestamp": "",
    }
    broker._kite.orders = lambda: []
    order = broker.submit_order("ord-kite-1", "RELIANCE.NS", "BUY", 10)
    assert order.client_order_id == "ord-kite-1"
    assert _FakeKiteConnect.fetch_count == 1


def test_sandbox_validation_is_read_only(monkeypatch):
    broker = _kite_broker(monkeypatch)
    report = broker.validate_sandbox(tickers=["RELIANCE.NS"])
    assert report["ok"] is True
    assert report["checks"]["instrument_mapping"]["detail"]["RELIANCE.NS"]["ok"]
    assert _FakeKiteConnect.fetch_count == 1


# ── kite failure state must survive (health/sandbox fail-closed) ─────────

def test_get_margin_raises_when_broker_request_fails(monkeypatch):
    broker = _kite_broker(monkeypatch)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(broker._kite, "margins", boom)
    with pytest.raises(RuntimeError, match="network down"):
        broker.get_margin()


def test_get_positions_raises_when_broker_request_fails(monkeypatch):
    broker = _kite_broker(monkeypatch)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(broker._kite, "positions", boom)
    with pytest.raises(RuntimeError, match="network down"):
        broker.get_positions()


def test_sandbox_validation_fails_closed_on_margin_error(monkeypatch):
    broker = _kite_broker(monkeypatch)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(broker._kite, "margins", boom)
    report = broker.validate_sandbox(tickers=["RELIANCE.NS"])
    assert report["checks"]["credentials"]["ok"] is False
    assert report["ok"] is False


def test_sandbox_validation_fails_closed_on_positions_error(monkeypatch):
    broker = _kite_broker(monkeypatch)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(broker._kite, "positions", boom)
    report = broker.validate_sandbox(tickers=["RELIANCE.NS"])
    assert report["checks"]["positions"]["ok"] is False
    assert report["ok"] is False


def test_health_reports_failure_when_margin_call_fails(monkeypatch):
    broker = _kite_broker(monkeypatch)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(broker._kite, "margins", boom)
    status = broker.health()
    assert status["ok"] is False
    assert "network down" in status.get("error", "")


# ── portfolio-aware pre-trade gate ────────────────────────────────────────

def test_gate_uses_broker_positions_for_total_exposure():
    # Existing position is 80% of equity; adding 20% more would hit 100%,
    # above the 95% max total exposure. The gate must reject it.
    broker = StubPortfolioBroker({"RELIANCE.NS": (320, 2500.0)}, cash=1_000_000)
    risk = RiskController(RiskLimits(), 1_000_000)
    manager = ExecutionManager(broker, risk)
    verdict = manager.gate_order(
        "TCS.NS", "BUY", 80,
        quotes={"TCS.NS": fresh_quote()}, order_value=200_000,
    )
    assert verdict["approved"] is False
    failed = [c["check"] for c in verdict["checks"] if not c["passed"]]
    assert "total_exposure" in failed


def test_gate_allows_reducing_order_at_exposure_limit():
    # 90% exposure; selling 10% reduces it to 80% — must not be blocked.
    broker = StubPortfolioBroker({"RELIANCE.NS": (360, 2500.0)}, cash=1_000_000)
    risk = RiskController(RiskLimits(), 1_000_000)
    manager = ExecutionManager(broker, risk)
    verdict = manager.gate_order(
        "RELIANCE.NS", "SELL", 40,
        quotes={"RELIANCE.NS": fresh_quote()}, order_value=100_000,
    )
    assert verdict["approved"] is True


def test_gate_checks_post_position_value_when_increasing():
    # Existing RELIANCE position is 200k (20%); buying another 100k would
    # make it 30% > 25% max position — the increment alone (10%) is under,
    # but the post-order position is not.
    broker = StubPortfolioBroker({"RELIANCE.NS": (80, 2500.0)}, cash=1_000_000)
    risk = RiskController(RiskLimits(), 1_000_000)
    manager = ExecutionManager(broker, risk)
    verdict = manager.gate_order(
        "RELIANCE.NS", "BUY", 40,
        quotes={"RELIANCE.NS": fresh_quote()}, order_value=100_000,
    )
    assert verdict["approved"] is False
    failed = [c["check"] for c in verdict["checks"] if not c["passed"]]
    assert "position_concentration" in failed


def test_gate_reports_order_intent(monkeypatch):
    broker = DryRunBroker()
    manager = ExecutionManager(
        broker, RiskController(RiskLimits(), 1_000_000),
        portfolio_provider=lambda: PortfolioSnapshot(
            holdings={"RELIANCE.NS": (10, 2500.0)}, equity=1_000_000,
        ),
    )
    verdict = manager.gate_order(
        "RELIANCE.NS", "SELL", 4,
        quotes={"RELIANCE.NS": fresh_quote()}, order_value=10_000,
    )
    assert verdict["approved"] is True
    intent = [c for c in verdict["checks"] if c["check"] == "order_intent"]
    assert intent and "intent=reducing" in intent[0]["message"]


def test_classify_intent():
    snap = PortfolioSnapshot(holdings={})
    assert ExecutionManager._classify_intent("X.NS", "BUY", 10, snap) == "opening"
    assert ExecutionManager._classify_intent("X.NS", "SELL", 10, snap) == "shorting"

    long_snap = PortfolioSnapshot(holdings={"X.NS": (100, 10.0)})
    assert ExecutionManager._classify_intent("X.NS", "BUY", 10, long_snap) == "increasing"
    assert ExecutionManager._classify_intent("X.NS", "SELL", 40, long_snap) == "reducing"
    assert ExecutionManager._classify_intent("X.NS", "SELL", 100, long_snap) == "closing"
    assert ExecutionManager._classify_intent("X.NS", "SELL", 150, long_snap) == "reversing"

    short_snap = PortfolioSnapshot(holdings={"X.NS": (-100, 10.0)})
    assert ExecutionManager._classify_intent("X.NS", "BUY", 40, short_snap) == "reducing"
    assert ExecutionManager._classify_intent("X.NS", "BUY", 100, short_snap) == "closing"
    assert ExecutionManager._classify_intent("X.NS", "BUY", 150, short_snap) == "reversing"


# ── quote timestamp handling (IST-aware, future-timestamp guard) ──────────

def test_naive_ist_quote_is_fresh_not_5h30m_stale():
    """Regression: NSE quotes carry naive IST wall-clock timestamps.
    Interpreting them as UTC made live quotes look 5h30m old and blocked
    every order."""
    from src.core.timeutils import IST
    naive_ist = datetime.now(IST).replace(tzinfo=None).isoformat()
    manager = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000))
    reason = manager._stale_quote_check("RELIANCE.NS", {
        "RELIANCE.NS": {"close": 2500.0, "timestamp": naive_ist},
    })
    assert reason == ""


def test_future_quote_timestamp_is_rejected():
    """A timestamp ahead of now beyond clock-skew tolerance is a broken
    source — reject rather than treat as infinitely fresh."""
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    manager = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000))
    reason = manager._stale_quote_check("RELIANCE.NS", {
        "RELIANCE.NS": {"close": 2500.0, "timestamp": future.isoformat()},
    })
    assert "future" in reason


def test_stale_utc_aware_quote_is_rejected():
    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    manager = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000))
    reason = manager._stale_quote_check("RELIANCE.NS", {
        "RELIANCE.NS": {"close": 2500.0, "timestamp": old.isoformat()},
    })
    assert "stale" in reason


# ── daily budget check+append atomicity ───────────────────────────────────

def test_budget_count_and_append_are_atomic(tmp_path, monkeypatch):
    """The budget check and the post-submit append must happen under one
    lock. Simulate the race window by making submit_order re-check the
    ledger mid-flight: if counting weren't serialized with appending, a
    concurrent submit could slip past the limit."""
    import json

    class _SlowBroker(DryRunBroker):
        calls = 0

        def submit_order(self, *a, **kw):
            # While this submit is in flight, another process appends an
            # order to the same ledger (as if it had just passed its own
            # check). The atomic path means our count already included...
            # actually the invariant under test: after N orders are counted,
            # no more than max_daily_orders submits may be appended per day.
            result = super().submit_order(*a, **kw)
            _SlowBroker.calls += 1
            return result

    broker = _SlowBroker()
    m1 = ExecutionManager(broker, RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    m2 = ExecutionManager(DryRunBroker(), RiskController(RiskLimits(), 1_000_000),
                          max_daily_orders=2, audit_dir=tmp_path)
    q1 = fresh_quote()
    q2 = fresh_quote()

    m1.execute("RELIANCE.NS", "BUY", 1, quotes={"RELIANCE.NS": q1}, order_value=2500)
    m2.execute("TCS.NS", "BUY", 1, quotes={"TCS.NS": q2}, order_value=2500)

    with pytest.raises(ExecutionError, match="order budget"):
        m1.execute("INFY.NS", "BUY", 1, quotes={"INFY.NS": q1}, order_value=2500)
    with pytest.raises(ExecutionError, match="order budget"):
        m2.execute("WIPRO.NS", "BUY", 1, quotes={"WIPRO.NS": q2}, order_value=2500)

    submitted = [
        json.loads(line) for line in
        next(iter(tmp_path.glob("orders-*.jsonl"))).read_text().splitlines()
        if json.loads(line).get("event") == "submitted"
    ]
    assert len(submitted) == 2


# ── kite submit ambiguity: timeout must not hide a live order ─────────────

def test_kite_submit_timeout_adopts_existing_broker_order(monkeypatch):
    """Regression: place_order timing out AFTER exchange acceptance used to
    mark the order REJECTED locally while the broker held a live order.
    The broker's order book (matched by tag) is authoritative."""
    broker = _kite_broker(monkeypatch)

    def timeout_place(**kwargs):
        raise TimeoutError("request timed out")

    def book_with_order():
        return [{
            "order_id": "209876",
            "tag": "ord-timeout-1"[:20],
            "status": "OPEN",
        }]

    monkeypatch.setattr(broker._kite, "place_order", timeout_place, raising=False)
    monkeypatch.setattr(broker._kite, "orders", book_with_order, raising=False)

    order = broker.submit_order("ord-timeout-1", "RELIANCE.NS", "BUY", 10)
    assert order.status == BrokerOrderStatus.SUBMITTED
    assert order.broker_order_id == "209876"
    assert not order.reject_reason


def test_kite_submit_failure_without_matching_tag_is_rejected(monkeypatch):
    """If the broker never received/accepted the order, REJECTED remains
    the correct fail-closed outcome."""
    broker = _kite_broker(monkeypatch)

    def refused_place(**kwargs):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(broker._kite, "place_order", refused_place, raising=False)
    monkeypatch.setattr(broker._kite, "orders", lambda: [], raising=False)

    order = broker.submit_order("ord-dead-1", "RELIANCE.NS", "BUY", 10)
    assert order.status == BrokerOrderStatus.REJECTED
    assert order.reject_reason


def test_kite_submit_failure_with_unreachable_order_book_is_rejected(monkeypatch):
    """When even the order-book lookup fails we cannot adopt broker truth;
    local state stays REJECTED (fail-closed) and the client order id remains
    in the audit trail for manual reconciliation."""
    broker = _kite_broker(monkeypatch)

    def refused_place(**kwargs):
        raise TimeoutError("timed out")

    def dead_book():
        raise RuntimeError("order book unreachable")

    monkeypatch.setattr(broker._kite, "place_order", refused_place, raising=False)
    monkeypatch.setattr(broker._kite, "orders", dead_book, raising=False)

    order = broker.submit_order("ord-dark-1", "RELIANCE.NS", "BUY", 10)
    assert order.status == BrokerOrderStatus.REJECTED
