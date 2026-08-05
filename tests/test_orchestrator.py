"""Tests for src/orchestrator.py."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.signals.orchestrator import DailyOrchestrator
from src.trading.ledger import Ledger


@pytest.fixture
def ledger():
    db_path = os.path.join(tempfile.mkdtemp(), "test_orch.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


def _mock_meta_controller():
    mc = MagicMock()
    mc.decide.return_value = {
        "action": "HOLD",
        "position_size": 0.0,
        "confidence": 0.5,
        "reasoning": "Mock decision",
    }
    return mc


# --- Basic creation ---

def test_creates_instance(ledger):
    orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
    assert orch.tickers == ["TEST.NS"]
    assert orch.ledger is ledger


def test_creates_with_meta_controller(ledger):
    mc = _mock_meta_controller()
    orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger, meta_controller=mc)
    assert orch.meta_controller is mc


# --- run() ---

def test_run_returns_summary(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    summary = orch.run()
    assert "date" in summary
    assert "decisions" in summary
    assert "errors" in summary


def test_run_logs_to_ledger(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    orch.run(dry_run=True)
    decisions = ledger.get_decisions()
    assert len(decisions) == 0  # dry run shouldn't log


def test_run_dry_run_doesnt_write(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    orch.run(dry_run=True)
    assert len(ledger.get_decisions()) == 0


def test_run_handles_empty_tickers(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    summary = orch.run()
    assert len(summary["decisions"]) == 0
    assert len(summary["errors"]) == 0


# --- _run_ensemble ---

def test_run_ensemble_keeps_latest_bar_for_inference(ledger, sample_df):
    """The NaN `target` on the latest bar must not drop it from inference."""
    feature_cols = ["close", "volume", "rsi_14", "macd"]

    def fake_features(df_in):
        out = df_in.copy()
        out["rsi_14"] = 50.0
        out["macd"] = 0.25
        out["target"] = out["close"].shift(-1) / out["close"] - 1.0
        return out

    tup = tuple(MagicMock() for _ in range(7))
    captured = {}

    def fake_predict(*args, **kwargs):
        captured["df_feat"] = args[6]
        return (1, 80.0, {})

    orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
    with patch("src.models.model.models_exist", return_value=True), \
         patch("src.models.model.load_models", return_value=tup), \
         patch("src.models.model.model_feature_cols", return_value=feature_cols), \
         patch("src.data.features.add_technical_indicators", side_effect=fake_features), \
         patch("src.models.ensemble.predict_ensemble", side_effect=fake_predict), \
         patch("src.signals.interpretability.explain_prediction", return_value={"top": []}):
        result = orch._run_ensemble("TEST.NS", sample_df)

    assert result["ensemble_direction"] == 1
    feat = captured["df_feat"]
    assert feat.index[-1] == sample_df.index[-1]  # today's bar survived
    assert pd.isna(feat["target"].iloc[-1])        # target still NaN on last row
    assert not pd.isna(feat["target"].iloc[-2])    # but defined for earlier rows


# --- _process_ticker ---

def test_process_ticker_collects_signals(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    with patch.object(orch, "_fetch_data") as mock_fetch, \
         patch.object(orch, "_run_ensemble", return_value={"ensemble_direction": 1, "ensemble_confidence": 0.7}), \
         patch.object(orch, "_run_sentiment", return_value={"sentiment_score": 0.3}), \
         patch.object(orch, "_run_flow", return_value={"fii_net": 500}), \
         patch.object(orch, "_run_pcr", return_value={"pcr": 1.1}), \
         patch.object(orch, "_run_mtf", return_value={"mtf_signal": 0.5}), \
         patch.object(orch, "_run_regime", return_value={"regime": "Bull", "regime_confidence": 0.8}), \
         patch.object(orch, "_run_risk", return_value={"var_95": -0.02}), \
         patch.object(orch, "_run_volatility", return_value={"volatility_forecast": 0.18}), \
         patch.object(orch, "_run_fundamentals", return_value={"fundamental_score": 70}):
        mock_fetch.return_value = pd.DataFrame({
            "close": np.random.randn(100) + 100,
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 101,
            "low": np.random.randn(100) + 99,
            "volume": np.random.randint(1000, 10000, 100).astype(float),
        }, index=pd.bdate_range("2024-01-01", periods=100))
        result = orch._process_ticker("TEST.NS", "2025-01-15", dry_run=True)
    assert result["ticker"] == "TEST.NS"
    assert "action" in result
    assert "signals" in result


def test_process_ticker_handles_fetch_failure(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    with patch.object(orch, "_fetch_data", return_value=None):
        result = orch._process_ticker("BAD.NS", "2025-01-15", dry_run=True)
    assert result["action"] == "HOLD"  # default when no data


def test_process_ticker_logs_to_ledger_via_run(ledger, monkeypatch):
    """Decisions are logged with the FINAL action after block checks (#50)."""
    from src.signals import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.settings, "max_daily_trades", 0)
    orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
    with patch.object(orch, "_process_ticker",
                      return_value={"ticker": "TEST.NS", "action": "BUY",
                                    "position_size": 0.05, "confidence": 0.8,
                                    "reasoning": "test",
                                    "signals": {"sentiment_score": 1.0}}):
        orch.run("2025-01-15", dry_run=False)
    decisions = ledger.get_decisions()
    assert len(decisions) == 1
    assert decisions[0]["ticker"] == "TEST.NS"
    # The BUY was the 2nd trade — blocked, so logged as HOLD
    assert decisions[0]["action"] == "HOLD"
    assert "limit" in decisions[0]["reasoning"]


# --- _make_decision ---

def test_make_decision_uses_meta_controller(ledger):
    mc = _mock_meta_controller()
    orch = DailyOrchestrator(tickers=[], ledger=ledger, meta_controller=mc)
    result = orch._make_decision({"ensemble_direction": 1})
    mc.decide.assert_called_once_with({"ensemble_direction": 1})
    assert result["action"] == "HOLD"


def test_make_decision_default_fallback(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    result = orch._make_decision({"ensemble_direction": 1, "ensemble_confidence": 0.7})
    assert result["action"] == "BUY"


def test_make_decision_default_hold(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    result = orch._make_decision({"ensemble_direction": None})
    assert result["action"] == "HOLD"


# --- run() with multiple tickers ---

def test_run_multiple_tickers(ledger):
    orch = DailyOrchestrator(tickers=["A.NS", "B.NS"], ledger=ledger)
    with patch.object(orch, "_process_ticker") as mock_process:
        mock_process.return_value = {"ticker": "X.NS", "action": "HOLD", "position_size": 0, "confidence": 0.5, "reasoning": "test"}
        summary = orch.run()
    assert len(summary["decisions"]) == 2


def test_run_catches_errors(ledger):
    orch = DailyOrchestrator(tickers=["GOOD.NS", "BAD.NS"], ledger=ledger)
    with patch.object(orch, "_process_ticker") as mock_process:
        def side_effect(ticker, date, dry_run):
            if ticker == "BAD.NS":
                raise ValueError("Something went wrong")
            return {"ticker": ticker, "action": "HOLD", "position_size": 0, "confidence": 0.5, "reasoning": "ok"}
        mock_process.side_effect = side_effect
        summary = orch.run()
    assert len(summary["decisions"]) == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["ticker"] == "BAD.NS"


# --- stop-loss attachment ---

class TestExecutePaperTradeStopLoss:
    """Stop-loss order is placed on the engine after every filled entry."""

    def _make_orch(self, ledger):
        from src.trading.paper_trader import PaperTrader
        trader = PaperTrader(initial_capital=200_000)
        return DailyOrchestrator(tickers=[], ledger=ledger, paper_trader=trader)

    def _fill_pending(self, orch, ticker: str, price: float):
        """Feed a bar so pending market orders fill."""
        from src.trading.engine import Bar
        bar = Bar(ticker, "2025-01-01", price, price * 1.01, price * 0.99, price, 1_000_000)
        orch.paper_trader.on_bar(ticker, bar.open, bar.high, bar.low, bar.close)

    def test_buy_attaches_sell_stop(self, ledger):
        orch = self._make_orch(ledger)
        result = {
            "action": "BUY",
            "position_size": 0.05,
            "confidence": 0.70,
            "reasoning": "test",
            "decision_id": None,
        }
        with patch("src.data.data_fetcher.get_live_price", return_value=1000.0):
            # Feed bar so market order fills immediately inside place_order → on_bar flow
            # The orchestrator calls place_order then checks status.value == "FILLED"
            # We need to simulate the fill by patching place_order to auto-fill
            original_place = orch.paper_trader.place_order

            def auto_fill_place(ticker, side, order_type, quantity, price=0, stop_price=0):
                order = original_place(ticker, side, order_type, quantity,
                                       price=price, stop_price=stop_price)
                if order_type.value == "MARKET" and order.status.value == "PENDING":
                    from src.trading.engine import Bar
                    bar = Bar(ticker, "t", price, price * 1.01,
                              price * 0.99, price, 1_000_000)
                    orch.paper_trader.engine.on_bar(bar)
                    # re-check status from filled_orders
                    for filled in orch.paper_trader.engine.filled_orders:
                        if filled.order_id == order.order_id:
                            order.status = filled.status
                            order.filled_price = filled.filled_price
                            order.filled_quantity = filled.filled_quantity
                            order.fill_cost = filled.fill_cost
                            break
                return order

            orch.paper_trader.place_order = auto_fill_place
            trade = orch._execute_paper_trade("TEST.NS", result)

        assert trade is not None, "Trade should have been executed"
        assert trade["side"] == "BUY"
        # Stop price should be ~5% below fill price
        assert trade["stop_price"] == pytest.approx(1000.0 * 0.95, rel=0.02)

        # Engine should have one pending STOP_MARKET order (the stop-loss)
        from src.trading.engine import OrderType
        pending = orch.paper_trader.engine.get_pending("TEST.NS")
        stop_orders = [o for o in pending if o.order_type == OrderType.STOP_MARKET]
        assert len(stop_orders) == 1
        assert stop_orders[0].stop_price == pytest.approx(1000.0 * 0.95, rel=0.02)

    def test_stop_info_in_trade_result(self, ledger):
        """trade dict returned by _execute_paper_trade includes stop_price key."""
        orch = self._make_orch(ledger)
        result = {
            "action": "BUY",
            "position_size": 0.05,
            "confidence": 0.70,
            "reasoning": "test",
            "decision_id": None,
        }
        original_place = orch.paper_trader.place_order

        def auto_fill_place(ticker, side, order_type, quantity, price=0, stop_price=0):
            order = original_place(ticker, side, order_type, quantity,
                                   price=price, stop_price=stop_price)
            if order_type.value == "MARKET" and order.status.value == "PENDING":
                from src.trading.engine import Bar
                bar = Bar(ticker, "t", price, price * 1.01,
                          price * 0.99, price, 1_000_000)
                orch.paper_trader.engine.on_bar(bar)
                for filled in orch.paper_trader.engine.filled_orders:
                    if filled.order_id == order.order_id:
                        order.status = filled.status
                        order.filled_price = filled.filled_price
                        order.filled_quantity = filled.filled_quantity
                        order.fill_cost = filled.fill_cost
                        break
            return order

        orch.paper_trader.place_order = auto_fill_place
        with patch("src.data.data_fetcher.get_live_price", return_value=500.0):
            trade = orch._execute_paper_trade("TEST.NS", result)

        assert trade is not None
        assert "stop_price" in trade
        assert trade["stop_price"] > 0
