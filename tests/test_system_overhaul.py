"""Integration tests verifying StoMar system overhaul."""
import os
import tempfile
import numpy as np

from src.trading.paper_trader import PaperTrader
from src.trading.engine import OrderSide, OrderStatus
from src.signals.significance import calculate_quant_stats


def test_paper_trader_instant_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = os.path.join(tmpdir, "paper_state.json")
        trader = PaperTrader(initial_capital=100000)

        # Place BUY market trade for RELIANCE.NS at 2500
        order = trader.execute_market_trade("RELIANCE.NS", OrderSide.BUY, 10, price=2500.0)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10
        assert order.filled_price > 0

        # Position should exist
        assert "RELIANCE.NS" in trader.positions
        pos = trader.positions["RELIANCE.NS"]
        assert pos.quantity == 10

        # Save and reload state
        trader.save_state(state_file)
        trader2 = PaperTrader(initial_capital=100000)
        loaded = trader2.load_state(state_file)
        assert loaded is True
        assert "RELIANCE.NS" in trader2.positions
        assert trader2.positions["RELIANCE.NS"].quantity == 10

        # Execute SELL trade to close position
        order2 = trader2.execute_market_trade("RELIANCE.NS", OrderSide.SELL, 10, price=2600.0)
        assert order2.status == OrderStatus.FILLED
        assert "RELIANCE.NS" not in trader2.positions
        assert len(trader2.closed_positions) == 1
        assert trader2.closed_positions[0]["pnl"] > 0  # Profit recorded after costs


def test_quant_stats_calculation():
    # Synthetic return series with known properties
    returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01, 0.008, 0.012, -0.003, 0.005, 0.02])
    stats = calculate_quant_stats(returns, risk_free_rate=0.07)

    assert "sharpe" in stats
    assert "sortino" in stats
    assert "calmar" in stats
    assert "cagr" in stats
    assert "max_drawdown" in stats
    assert "win_rate" in stats
    assert "profit_factor" in stats
    assert stats["win_rate"] == 0.7  # 7 out of 10 returns positive


def test_paper_trader_reset():
    trader = PaperTrader(initial_capital=100000)
    trader.execute_market_trade("TCS.NS", OrderSide.BUY, 5, price=3500.0)
    assert len(trader.positions) > 0

    trader.reset()
    assert len(trader.positions) == 0
    assert trader.cash == 100000
    assert len(trader.trade_log) == 0
