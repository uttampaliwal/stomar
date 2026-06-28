"""Tests for risk management module."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestKellyCriterion:
    def test_positive_edge_positive_kelly(self):
        from src.risk import kelly_criterion
        k = kelly_criterion(0.6, 2.0, 1.0)
        assert k > 0

    def test_no_edge_zero_kelly(self):
        from src.risk import kelly_criterion
        k = kelly_criterion(0.5, 1.0, 1.0)
        assert k == 0.0

    def test_kelly_capped_at_25(self):
        from src.risk import kelly_criterion
        k = kelly_criterion(0.9, 5.0, 1.0)
        assert k <= 0.25

    def test_zero_loss_returns_zero(self):
        from src.risk import kelly_criterion
        k = kelly_criterion(0.6, 2.0, 0.0)
        assert k == 0.0


class TestVaR:
    def test_var_95_is_negative(self, sample_returns):
        from src.risk import calculate_var
        var = calculate_var(sample_returns, 0.95)
        assert var <= 0

    def test_var_with_insufficient_data(self):
        from src.risk import calculate_var
        returns = np.array([0.01, -0.01])
        var = calculate_var(returns, 0.95)
        assert var == 0.0


class TestCVaR:
    def test_cvar_worse_than_var(self, sample_returns):
        from src.risk import calculate_var, calculate_cvar
        var = calculate_var(sample_returns, 0.95)
        cvar = calculate_cvar(sample_returns, 0.95)
        assert cvar <= var

    def test_cvar_with_insufficient_data(self):
        from src.risk import calculate_cvar
        returns = np.array([0.01, -0.01])
        cvar = calculate_cvar(returns, 0.95)
        assert cvar == 0.0


class TestSharpe:
    def test_sharpe_uses_risk_free_rate(self, sample_returns):
        from src.risk import calculate_sharpe
        sharpe = calculate_sharpe(sample_returns, risk_free_rate=0.0)
        assert isinstance(sharpe, float)

    def test_zero_volatility_returns_zero(self):
        from src.risk import calculate_sharpe
        returns = np.zeros(100)
        sharpe = calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_sharpe_with_insufficient_data(self):
        from src.risk import calculate_sharpe
        sharpe = calculate_sharpe(np.array([0.01]))
        assert sharpe == 0.0


class TestSortino:
    def test_sortino_with_insufficient_data(self):
        from src.risk import calculate_sortino
        sortino = calculate_sortino(np.array([0.01]))
        assert sortino == 0.0


class TestMaxDrawdown:
    def test_max_drawdown_is_negative(self, sample_returns):
        from src.risk import calculate_max_drawdown
        equity = np.cumprod(1 + sample_returns) * 100
        mdd = calculate_max_drawdown(equity)
        assert mdd <= 0

    def test_no_drawdown_zero(self):
        from src.risk import calculate_max_drawdown
        equity = np.array([1, 2, 3, 4, 5])
        mdd = calculate_max_drawdown(equity)
        assert mdd == 0.0


class TestRiskReport:
    def test_risk_report_keys(self, sample_returns):
        from src.risk import generate_risk_report
        equity = np.cumprod(1 + sample_returns) * 100
        report = generate_risk_report(sample_returns, equity)
        expected_keys = ["sharpe", "sortino", "calmar", "max_drawdown", "var_95", "cvar_95"]
        for key in expected_keys:
            assert key in report

    def test_insufficient_data_report(self):
        from src.risk import generate_risk_report
        report = generate_risk_report(np.array([0.01]), np.array([100]))
        assert "error" in report
