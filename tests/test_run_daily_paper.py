"""Regression tests for run_daily paper trading.

* #15: the paper-state FileLock must be released even when execution
  raises mid-run (a bare acquire()/release() would leak it and deadlock
  every later run).
* #20: decisions already executed (and logged) by the daily orchestrator
  are never traded a second time here.
"""

import pytest

from run_daily import run_paper_trades


class _FakeLedger:
    def __init__(self, trades):
        self._trades = trades

    def get_trades(self):
        return list(self._trades)


class _FakeManager:
    def __init__(self, broker, risk):
        pass

    def gate_order(self, ticker, side, quantity, order_value=None,
                   market=None, is_closing=False, **kwargs):
        return {"approved": True, "reason": "", "checks": []}


def _decision(did, ticker="A.NS", action="BUY"):
    return {
        "decision_id": did,
        "ticker": ticker,
        "action": action,
        "confidence": 0.8,
        "position_size": 0.1,
        "current_price": 100.0,
    }


class _RecordingTrader:
    """Minimal PaperTrader stand-in that records order placement."""

    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.place_calls = 0

    def load_state(self, path):
        return True

    def get_equity(self):
        return self.initial_capital

    def place_order(self, ticker, side, order_type, quantity, price=0):
        self.place_calls += 1
        from src.trading.engine import Order
        return Order(order_id="", ticker=ticker, side=side,
                     order_type=order_type, quantity=quantity, price=price)

    def on_bar(self, *args, **kwargs):
        return []

    def update_prices(self, prices):
        pass

    def get_summary(self):
        return {
            "current_equity": self.initial_capital,
            "total_return_pct": 0.0,
            "total_trades": self.place_calls,
            "open_positions": {},
        }

    def save_state(self, path):
        pass


@pytest.fixture
def recording_trader(monkeypatch):
    monkeypatch.setattr("src.brokers.get_broker", lambda: object())
    monkeypatch.setattr(
        "src.trading.execution_manager.ExecutionManager", _FakeManager
    )
    monkeypatch.setattr(
        "src.trading.risk_controls.RiskController", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "src.data.data_fetcher.get_live_price", lambda ticker: 100.0
    )
    trader = _RecordingTrader(200_000)
    monkeypatch.setattr("src.trading.paper_trader.PaperTrader", lambda *a, **k: trader)
    return trader


class TestNoDoubleExecution:
    """#20: a decision with a logged trade is never executed again."""

    def test_already_traded_decisions_are_skipped(self, recording_trader, tmp_path):
        ledger = _FakeLedger([{"decision_id": 11}])
        decisions = [_decision(11, ticker="A.NS"), _decision(12, ticker="B.NS")]

        summary = run_paper_trades(
            decisions, ledger, capital=200_000,
            state_path=str(tmp_path / "paper_state.json"),
        )

        # Only decision 12 is new; decision 11 was already executed+logged.
        assert recording_trader.place_calls == 1
        assert summary["total_trades"] == 1

    def test_without_ledger_all_decisions_run(self, recording_trader, tmp_path):
        decisions = [_decision(11), _decision(12)]
        run_paper_trades(
            decisions, None, capital=200_000,
            state_path=str(tmp_path / "paper_state.json"),
        )
        assert recording_trader.place_calls == 2


class TestLockReleased:
    """#15: the state lock is released via the context manager on failure."""

    def test_lock_free_after_exception(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "paper_state.json")

        class ExplodingTrader:
            def __init__(self, initial_capital):
                self.initial_capital = initial_capital
                self.cash = initial_capital
                self.positions = {}

            def load_state(self, path):
                return True

            def get_equity(self):
                raise RuntimeError("boom mid-run")

            def get_summary(self):
                raise RuntimeError("should not get here")

        monkeypatch.setattr("src.brokers.get_broker", lambda: object())
        monkeypatch.setattr(
            "src.trading.execution_manager.ExecutionManager", _FakeManager
        )
        monkeypatch.setattr(
            "src.trading.risk_controls.RiskController", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            "src.trading.paper_trader.PaperTrader",
            lambda *a, **k: ExplodingTrader(200_000),
        )

        with pytest.raises(RuntimeError):
            run_paper_trades(
                [_decision(1)], None, capital=200_000, state_path=state_path
            )

        # The lock must be free — acquiring it again must not time out.
        # With the old bare acquire()/release() this raised filelock.Timeout.
        from filelock import FileLock
        probe = FileLock(f"{state_path}.lock", timeout=5)
        probe.acquire()
        probe.release()
