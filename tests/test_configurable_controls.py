"""Tests: orchestrator/meta-controller controls are driven by settings, not hardcoded."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.core import settings as settings_module


@pytest.fixture
def patch_setting(monkeypatch):
    def _patch(name, value):
        monkeypatch.setattr(settings_module.settings, name, value)

    return _patch


# ── DailyOrchestrator ───────────────────────────────────────────────────────


def _orchestrator_with_buy_results(ledger, monkeypatch):
    from src.signals.orchestrator import DailyOrchestrator

    orch = DailyOrchestrator(tickers=["A.NS", "B.NS", "C.NS"], ledger=ledger)

    def fake_process(ticker, date, dry_run):
        return {
            "ticker": ticker,
            "action": "BUY",
            "position_size": 0.05,
            "signals": {},
            "reasoning": "test",
            "confidence": 0.8,
            "price": 100.0,
        }

    monkeypatch.setattr(orch, "_process_ticker", fake_process)
    return orch


@pytest.fixture
def ledger():
    import tempfile

    from src.trading.ledger import Ledger

    db_path = os.path.join(tempfile.mkdtemp(), "test_4x_controls.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


def test_daily_trade_limit_reads_settings(ledger, monkeypatch, patch_setting):
    patch_setting("max_daily_trades", 2)
    patch_setting("max_portfolio_exposure", 1.0)
    orch = _orchestrator_with_buy_results(ledger, monkeypatch)

    summary = orch.run(dry_run=True)

    executed = [d for d in summary["decisions"] if d["action"] == "BUY"]
    blocked = [d for d in summary["decisions"] if d["action"] == "HOLD"]
    assert len(executed) == 2
    assert len(blocked) == 1
    assert "Daily trade limit (2)" in blocked[0]["reasoning"]


def test_portfolio_exposure_reads_settings(ledger, monkeypatch, patch_setting):
    patch_setting("max_daily_trades", 10)
    patch_setting("max_portfolio_exposure", 0.06)  # first 0.05 position fits, second doesn't
    orch = _orchestrator_with_buy_results(ledger, monkeypatch)

    summary = orch.run(dry_run=True)

    blocked = [d for d in summary["decisions"] if d["action"] == "HOLD"]
    assert len(blocked) == 2
    assert "exposure cap" in blocked[0]["reasoning"].lower()


def test_stop_loss_reads_settings(monkeypatch, patch_setting):
    from src.signals.orchestrator import DailyOrchestrator

    patch_setting("stop_loss_pct", 0.03)
    orch = DailyOrchestrator(tickers=[], ledger=None)
    trader = monkeypatch_fake_trader(monkeypatch)
    orch.paper_trader = trader
    monkeypatch.setattr(orch, "ledger", FakeLedger())

    result = {
        "ticker": "SL.NS", "action": "BUY", "position_size": 0.05,
        "signals": {}, "reasoning": "test", "confidence": 0.8,
        "decision_id": "d1",
    }
    orch._execute_paper_trade("SL.NS", result)

    assert trader.stop_prices == [round(100.0 * 0.97, 2)]


class FakeLedger:
    def log_trade(self, **kwargs):
        pass


def monkeypatch_fake_trader(monkeypatch):
    class FakeTrader:
        def __init__(self):
            self.stop_prices = []

        def get_equity(self):
            return 200000.0

        def execute_market_trade(self, ticker, side, quantity, price=None):
            return FakeOrder(price)

        def place_order(self, ticker, side, order_type, quantity, stop_price=None):
            self.stop_prices.append(stop_price)
            return FakeOrder(stop_price)

        @property
        def positions(self):
            return {}

        @property
        def risk_controller(self):
            return FakeRisk()

    class FakeRisk:
        class limits:
            max_position_pct = 0.5

        def check_order(self, **kwargs):
            return {"approved": True}

    class FakeOrder:
        def __init__(self, price):
            self.filled_price = price
            self.status = FakeStatus()

    class FakeStatus:
        name = "FILLED"

    monkeypatch.setattr(
        "src.data.data_fetcher.get_live_price", lambda *a, **k: 100.0
    )
    return FakeTrader()


# ── MetaController ──────────────────────────────────────────────────────────


def test_meta_controller_discount_reads_settings(patch_setting):
    from src.models.meta_controller import MetaController

    patch_setting("uncalibrated_confidence_discount", 0.5)
    mc = MetaController()
    assert mc.uncalibrated_confidence_discount == 0.5


def test_meta_controller_discount_default_when_missing(monkeypatch):
    from src.models import meta_controller as mc_mod
    from src.models.meta_controller import MetaController

    monkeypatch.delattr(settings_module.settings, "uncalibrated_confidence_discount", raising=False)
    monkeypatch.setattr(mc_mod, "UNCALIBRATED_CONFIDENCE_DISCOUNT", 0.7)
    mc = MetaController()
    assert mc.uncalibrated_confidence_discount == 0.7


# ── Settings env plumbing ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("env_name", "env_value", "attr", "expected"),
    [
        ("STOMAR_MAX_DAILY_TRADES", "3", "max_daily_trades", 3),
        ("STOMAR_STOP_LOSS_PCT", "0.04", "stop_loss_pct", 0.04),
        ("STOMAR_UNCALIBRATED_CONFIDENCE_DISCOUNT", "0.55", "uncalibrated_confidence_discount", 0.55),
        ("STOMAR_MAX_PORTFOLIO_EXPOSURE", "0.40", "max_portfolio_exposure", 0.4),
    ],
)
def test_settings_env_overrides(monkeypatch, env_name, env_value, attr, expected):
    import importlib

    monkeypatch.setenv(env_name, env_value)
    mod = importlib.reload(settings_module)
    if mod._HAS_PYDANTIC_SETTINGS:
        instance = mod.Settings()
    else:
        instance = mod._FallbackSettings()
    assert getattr(instance, attr) == expected
