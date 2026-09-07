import logging

import numpy as np
import pandas as pd
from src.core.constants import RISK_FREE_RATE

logger = logging.getLogger(__name__)


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    f = (win_rate * b - (1 - win_rate)) / b
    return max(0.0, min(f, 0.25))


def fixed_fraction_sizing(capital: float, risk_per_trade: float, entry_price: float, stop_price: float) -> int:
    risk_amount = capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    shares = int(risk_amount / risk_per_share)
    max_affordable = int(capital * 0.95 / entry_price)
    return min(shares, max_affordable)


def volatility_position_size(capital: float, target_risk: float, atr: float, price: float) -> int:
    if atr <= 0 or price <= 0:
        return 0
    vol_pct = atr / price
    shares = int((capital * target_risk) / (atr))
    max_affordable = int(capital * 0.95 / price)
    return min(shares, max_affordable)


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 20:
        logger.warning(
            "VaR on %d samples (<20): returning 0.0 — treat as UNKNOWN, not no-risk",
            len(returns),
        )
        return 0.0
    sorted_returns = np.sort(returns)
    idx = int((1 - confidence) * len(sorted_returns))
    return float(sorted_returns[max(idx, 0)])


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    if len(returns) < 20:
        logger.warning(
            "CVaR on %d samples (<20): returning 0.0 — treat as UNKNOWN, not no-risk",
            len(returns),
        )
        return 0.0
    var = calculate_var(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    if len(returns) < 10:
        return 0.0
    ann_return = returns.mean() * 252
    ann_vol = returns.std(ddof=1) * np.sqrt(252)
    if ann_vol == 0:
        return 0.0
    return float((ann_return - risk_free_rate) / ann_vol)


def calculate_sortino(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    if len(returns) < 10:
        return 0.0
    ann_return = returns.mean() * 252
    mar = risk_free_rate / 252
    downside_diff = np.minimum(returns - mar, 0)
    downside_vol = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
    if downside_vol == 0:
        return 0.0
    return float((ann_return - risk_free_rate) / downside_vol)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    return float(drawdown.min())


def calculate_calmar(returns: np.ndarray, equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    ann_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    return float(ann_return / max_dd)


def portfolio_var(weights: np.ndarray, cov_matrix: np.ndarray, confidence: float = 0.95) -> float:
    port_return_var = weights @ cov_matrix @ weights
    port_std = np.sqrt(port_return_var)
    # Map confidence to z-score (95% -> 1.645, 99% -> 2.326)
    from scipy import stats as sp_stats
    z_score = float(sp_stats.norm.ppf(confidence))
    return float(z_score * port_std)


def max_position_value(capital: float, max_single_pct: float = 0.25) -> float:
    return capital * max_single_pct


def check_portfolio_risk(holdings: dict, prices: dict, capital: float, max_drawdown_limit: float = 0.15) -> dict:
    total_value = capital
    for ticker, (qty, _) in holdings.items():
        price = prices.get(ticker, _)
        total_value += qty * price

    position_values = {}
    for ticker, (qty, avg_price) in holdings.items():
        price = prices.get(ticker, avg_price)
        value = qty * price
        position_values[ticker] = {
            "value": value,
            "weight": value / max(total_value, 1),
            "pnl": (price - avg_price) * qty,
            "pnl_pct": ((price / avg_price) - 1) * 100 if avg_price > 0 else 0,
        }

    max_weight = max([p["weight"] for p in position_values.values()] + [0])
    concentration_risk = max_weight > 0.30

    return {
        "total_value": total_value,
        "n_positions": len(holdings),
        "positions": position_values,
        "max_position_weight": max_weight,
        "concentration_risk": concentration_risk,
    }


def generate_risk_report(returns: np.ndarray, equity_curve: np.ndarray) -> dict:
    if len(returns) < 10:
        return {"error": "Insufficient data"}

    return {
        "sharpe": calculate_sharpe(returns),
        "sortino": calculate_sortino(returns),
        "calmar": calculate_calmar(returns, equity_curve),
        "max_drawdown": calculate_max_drawdown(equity_curve) * 100,
        "var_95": calculate_var(returns, 0.95) * 100,
        "cvar_95": calculate_cvar(returns, 0.95) * 100,
        "var_99": calculate_var(returns, 0.99) * 100,
        "annual_volatility": float(returns.std() * np.sqrt(252) * 100),
        "annual_return": float(returns.mean() * 252 * 100),
        "positive_days_pct": float((returns > 0).mean() * 100),
        "best_day": float(returns.max() * 100),
        "worst_day": float(returns.min() * 100),
        "skewness": float(pd.Series(returns).skew()),
        "kurtosis": float(pd.Series(returns).kurtosis()),
    }
