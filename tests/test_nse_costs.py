"""Tests for realistic NSE transaction cost model."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestNSECosts:
    def test_buy_side_costs(self):
        from src.constants import calculate_nse_costs
        costs = calculate_nse_costs(1000.0, 10, "buy")
        assert costs["brokerage"] > 0
        assert costs["stt"] > 0  # STT on buy (delivery: both sides)
        assert costs["stamp_duty"] > 0  # Stamp duty on buy
        assert costs["gst"] > 0
        assert costs["total"] > 0
        assert costs["effective_rate"] > 0

    def test_sell_side_costs(self):
        from src.constants import calculate_nse_costs
        costs = calculate_nse_costs(1000.0, 10, "sell")
        assert costs["brokerage"] > 0
        assert costs["stt"] > 0  # STT on sell
        assert costs["stamp_duty"] == 0.0  # No stamp duty on sell
        assert costs["gst"] > 0
        assert costs["total"] > 0

    def test_round_trip_cost_rate(self):
        from src.constants import calculate_nse_costs
        buy = calculate_nse_costs(1000.0, 10, "buy")
        sell = calculate_nse_costs(1000.0, 10, "sell")
        total_round_trip = buy["total"] + sell["total"]
        trade_value = 1000.0 * 10
        rate = total_round_trip / trade_value
        # Round-trip without slippage should be ~0.3-0.5% (STT on both sides)
        assert 0.002 < rate < 0.006, f"Round-trip cost rate {rate:.4f} outside expected range"

    def test_costs_scale_with_trade_value(self):
        from src.constants import calculate_nse_costs
        c1 = calculate_nse_costs(100.0, 10, "buy")
        c2 = calculate_nse_costs(200.0, 10, "buy")
        # Double price should roughly double costs
        assert c2["total"] > c1["total"]
        ratio = c2["total"] / c1["total"]
        assert 1.8 < ratio < 2.2

    def test_zero_quantity(self):
        from src.constants import calculate_nse_costs
        costs = calculate_nse_costs(1000.0, 0, "buy")
        assert costs["total"] == 0.0
        assert costs["effective_rate"] == 0.0

    def test_gst_applied_to_brokerage_and_exchange(self):
        from src.constants import calculate_nse_costs, BROKERAGE_RATE, EXCHANGE_CHARGE_RATE, GST_RATE
        price, qty = 500.0, 20
        costs = calculate_nse_costs(price, qty, "buy")
        expected_gst = (price * qty * BROKERAGE_RATE + price * qty * EXCHANGE_CHARGE_RATE) * GST_RATE
        assert abs(costs["gst"] - expected_gst) < 0.01


class TestPortfolioWithNSECosts:
    def test_buy_uses_full_costs(self):
        from src.portfolio import Portfolio
        p = Portfolio(100000)
        result = p.buy("TEST.NS", 1000.0, 10, "2024-01-01")
        assert result is True
        # Cash should be reduced by trade value + all NSE costs
        assert p.cash < 100000 - 10000  # Less than just trade value

    def test_sell_uses_full_costs(self):
        from src.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 1000.0, 10, "2024-01-01")
        cash_after_buy = p.cash
        p.sell("TEST.NS", 1100.0, 10, "2024-01-02")
        # Sell proceeds should be less than trade value due to costs
        assert p.cash < cash_after_buy + 11000

    def test_trade_records_costs(self):
        from src.portfolio import Portfolio
        p = Portfolio(100000)
        p.buy("TEST.NS", 1000.0, 10, "2024-01-01")
        trade = p.trades[0]
        assert "costs" in trade
        assert "stt" in trade["costs"]
        assert "stamp_duty" in trade["costs"]
        assert "gst" in trade["costs"]
        assert "total" in trade["costs"]
