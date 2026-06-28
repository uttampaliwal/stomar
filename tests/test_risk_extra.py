"""Additional tests for risk module to increase coverage."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFixedFractionSizing:
    def test_basic_sizing(self):
        from src.risk import fixed_fraction_sizing
        shares = fixed_fraction_sizing(100000, 0.02, 1000, 950)
        assert shares > 0

    def test_zero_stop_price(self):
        from src.risk import fixed_fraction_sizing
        shares = fixed_fraction_sizing(100000, 0.02, 1000, 1000)
        assert shares == 0

    def test_cannot_afford_any(self):
        from src.risk import fixed_fraction_sizing
        shares = fixed_fraction_sizing(100, 0.02, 10000, 9000)
        assert shares == 0


class TestVolatilityPositionSize:
    def test_basic_sizing(self):
        from src.risk import volatility_position_size
        shares = volatility_position_size(100000, 0.5, 1.0, 100)
        assert shares > 0

    def test_zero_atr(self):
        from src.risk import volatility_position_size
        shares = volatility_position_size(100000, 0.02, 0, 1000)
        assert shares == 0

    def test_zero_price(self):
        from src.risk import volatility_position_size
        shares = volatility_position_size(100000, 0.02, 5.0, 0)
        assert shares == 0


class TestCalculateMaxDrawdown:
    def test_monotonic_increase(self):
        from src.risk import calculate_max_drawdown
        equity = np.array([1, 2, 3, 4, 5])
        assert calculate_max_drawdown(equity) == 0.0

    def test_drawdown_present(self):
        from src.risk import calculate_max_drawdown
        equity = np.array([100, 110, 90, 95, 80])
        mdd = calculate_max_drawdown(equity)
        assert mdd < 0
        assert abs(mdd - (80 - 110) / 110) < 0.01

    def test_short_array(self):
        from src.risk import calculate_max_drawdown
        assert calculate_max_drawdown(np.array([100])) == 0.0

    def test_empty_array(self):
        from src.risk import calculate_max_drawdown
        assert calculate_max_drawdown(np.array([])) == 0.0


class TestCalculateCalmar:
    def test_basic_calmar(self):
        from src.risk import calculate_calmar
        returns = np.random.randn(252) * 0.01
        equity = np.cumprod(1 + returns) * 100
        calmar = calculate_calmar(returns, equity)
        assert isinstance(calmar, float)

    def test_zero_drawdown(self):
        from src.risk import calculate_calmar
        returns = np.array([0.01] * 100)
        equity = np.cumprod(1 + returns) * 100
        calmar = calculate_calmar(returns, equity)
        assert calmar == 0.0

    def test_short_data(self):
        from src.risk import calculate_calmar
        assert calculate_calmar(np.array([0.01]), np.array([100])) == 0.0


class TestPortfolioVar:
    def test_basic_portfolio_var(self):
        from src.risk import portfolio_var
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        var = portfolio_var(weights, cov)
        assert var > 0


class TestMaxPositionValue:
    def test_basic_max_position(self):
        from src.risk import max_position_value
        assert max_position_value(100000) == 25000
        assert max_position_value(100000, 0.1) == 10000


class TestCheckPortfolioRisk:
    def test_basic_check(self):
        from src.risk import check_portfolio_risk
        holdings = {"A": (10, 100.0), "B": (5, 200.0)}
        prices = {"A": 110.0, "B": 190.0}
        result = check_portfolio_risk(holdings, prices, 50000)
        assert "total_value" in result
        assert "concentration_risk" in result
        assert result["n_positions"] == 2

    def test_empty_holdings(self):
        from src.risk import check_portfolio_risk
        result = check_portfolio_risk({}, {}, 100000)
        assert result["total_value"] == 100000
        assert result["n_positions"] == 0


class TestGenerateRiskReport:
    def test_report_has_all_keys(self, sample_returns):
        from src.risk import generate_risk_report
        equity = np.cumprod(1 + sample_returns) * 100
        report = generate_risk_report(sample_returns, equity)
        expected = ["sharpe", "sortino", "calmar", "max_drawdown",
                    "var_95", "cvar_95", "var_99", "annual_volatility",
                    "annual_return", "positive_days_pct", "best_day",
                    "worst_day", "skewness", "kurtosis"]
        for k in expected:
            assert k in report

    def test_insufficient_data(self):
        from src.risk import generate_risk_report
        report = generate_risk_report(np.array([0.01, 0.02]), np.array([100, 101]))
        assert "error" in report

    def test_single_element_returns(self):
        from src.risk import generate_risk_report
        report = generate_risk_report(np.array([0.01]), np.array([100]))
        assert "error" in report
