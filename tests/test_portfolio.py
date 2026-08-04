"""Tests for Portfolio class."""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPortfolioBasic:
    def test_initial_state(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        assert p.cash == 100000
        assert p.initial_capital == 100000
        assert len(p.holdings) == 0
        assert len(p.trades) == 0

    def test_buy_reduces_cash(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        result = p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        assert result is True
        assert p.cash < 100000

    def test_buy_creates_holding(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        assert "TEST.NS" in p.holdings
        qty, avg_price, first_buy_date = p.holdings["TEST.NS"]
        assert qty == 10
        assert first_buy_date == datetime(2024, 1, 1)
        assert avg_price > 100.0  # includes buy-side costs
        assert avg_price < 101.0

    def test_sell_removes_holding(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 120.0, 10, datetime(2024, 1, 2))
        assert "TEST.NS" not in p.holdings

    def test_sell_records_pnl(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 120.0, 10, datetime(2024, 1, 2))
        sell_trade = [t for t in p.trades if t["action"] == "SELL"][0]
        assert sell_trade["pnl"] > 0

    def test_buy_exceeds_cash(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(10000)
        result = p.buy("TEST.NS", 100.0, 200, datetime(2024, 1, 1))
        # Should adjust quantity to what we can afford
        assert result is True
        assert p.cash >= 0

    def test_sell_nonexistent_ticker(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        result = p.sell("NONEXISTENT.NS", 100.0, 10, datetime(2024, 1, 1))
        assert result is False

    def test_buy_updates_equity_curve(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        initial_len = len(p.equity_curve)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        assert len(p.equity_curve) == initial_len + 1


class TestPortfolioValue:
    def test_portfolio_value_initial(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        assert p.portfolio_value() == 100000

    def test_portfolio_value_with_holdings(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        # Value = cash + holdings at avg price
        assert p.portfolio_value() > 0

    def test_portfolio_value_with_prices(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        val = p.portfolio_value(prices={"TEST.NS": 150.0})
        assert val > p.portfolio_value()


class TestPortfolioStats:
    def test_empty_stats(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        stats = p.get_stats()
        assert stats == {}

    def test_stats_after_trades(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 110.0, 10, datetime(2024, 1, 2))
        stats = p.get_stats()
        assert stats["total_trades"] == 2


class TestCapitalGainsTax:
    """P5.5: STCG 15% (<1yr), LTCG 10% (>=1yr, ₹1L exemption)."""

    def test_stcg_applied_on_short_term_profit(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 120.0, 10, datetime(2024, 3, 1))  # 60 days
        trade = p.trades[-1]
        assert trade["tax_category"] == "STCG"
        # tax applies to net P&L (after brokerage/STT costs), not gross
        assert trade["tax"] == round(trade["pnl"] * 0.15, 2)
        assert p.taxes_paid == trade["tax"]

    def test_ltcg_applied_after_one_year(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 130.0, 10, datetime(2025, 6, 1))  # > 365 days
        trade = p.trades[-1]
        assert trade["tax_category"] == "LTCG"
        assert trade["tax"] == round(max(trade["pnl"] - 100000, 0.0) * 0.10, 2)
        assert trade["tax"] == 0.0  # ₹1L exemption covers small gains

    def test_ltcg_exemption_exhausted_then_taxed(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(500000)
        assert p.buy("TEST.NS", 100.0, 2000, datetime(2024, 1, 1))
        assert p.buy("TEST2.NS", 100.0, 2000, datetime(2024, 1, 1))
        p.sell("TEST.NS", 160.0, 2000, datetime(2025, 6, 1))  # pnl ~120k
        first = p.trades[-1]
        assert first["tax_category"] == "LTCG"
        # pnl above 100k exemption taxed at 10%
        assert first["tax"] == round((first["pnl"] - 100000) * 0.10, 2)
        p.sell("TEST2.NS", 160.0, 2000, datetime(2025, 6, 2))  # pnl ~120k again
        second = p.trades[-1]
        # exemption exhausted: full gain taxed at 10%
        assert second["tax"] == round(second["pnl"] * 0.10, 2)

    def test_no_tax_on_loss(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 80.0, 10, datetime(2024, 3, 1))
        trade = p.trades[-1]
        assert trade["tax"] == 0.0
        assert p.taxes_paid == 0.0

    def test_stats_report_taxes(self):
        from src.trading.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 100.0, 10, datetime(2024, 1, 1))
        p.sell("TEST.NS", 120.0, 10, datetime(2024, 3, 1))
        stats = p.get_stats()
        assert stats["taxes_paid"] > 0
        assert "post_tax_return" in stats
