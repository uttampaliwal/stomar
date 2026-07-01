import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.core.constants import RISK_FREE_RATE


def compute_returns(prices: pd.DataFrame, period: str = "daily") -> pd.DataFrame:
    if period == "daily":
        return prices.pct_change().dropna()
    elif period == "weekly":
        return prices.resample("W").last().pct_change().dropna()
    elif period == "monthly":
        return prices.resample("M").last().pct_change().dropna()
    return prices.pct_change().dropna()


def compute_covariance(returns: pd.DataFrame, method: str = "standard") -> np.ndarray:
    if method == "shrinkage":
        sample_cov = returns.cov().values
        n_assets = sample_cov.shape[0]
        trace_val = np.trace(sample_cov) / n_assets
        shrink = np.eye(n_assets) * trace_val * 0.3
        return sample_cov * 0.7 + shrink
    return returns.cov().values


def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> dict:
    port_return = weights @ mean_returns
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return {"return": port_return, "volatility": port_vol, "sharpe": sharpe}


def max_sharpe_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = RISK_FREE_RATE / 252) -> dict:
    n = len(mean_returns)
    result = {
        "fun": None,
        "success": False,
    }

    def neg_sharpe(w):
        stats = portfolio_stats(w, mean_returns, cov_matrix)
        excess = stats["return"] - risk_free_rate
        return -excess / max(stats["volatility"], 1e-10)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.40)] * n

    best = None
    for _ in range(10):
        w0 = np.random.dirichlet(np.ones(n))
        try:
            res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            continue

    if best is not None and best.success:
        return {
            "weights": best.x,
            "return": portfolio_stats(best.x, mean_returns, cov_matrix)["return"] * 252,
            "volatility": portfolio_stats(best.x, mean_returns, cov_matrix)["volatility"] * np.sqrt(252),
            "sharpe": -best.fun,
        }
    return {"weights": np.ones(n) / n, "return": 0, "volatility": 0, "sharpe": 0}


def min_variance_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> dict:
    n = len(mean_returns)

    def portfolio_variance(w):
        return w @ cov_matrix @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.40)] * n
    w0 = np.ones(n) / n

    try:
        res = minimize(portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            stats = portfolio_stats(res.x, mean_returns, cov_matrix)
            return {
                "weights": res.x,
                "return": stats["return"] * 252,
                "volatility": stats["volatility"] * np.sqrt(252),
                "sharpe": stats["sharpe"],
            }
    except Exception:
        pass
    return {"weights": np.ones(n) / n, "return": 0, "volatility": 0, "sharpe": 0}


def black_litterman(market_weights: np.ndarray, cov_matrix: np.ndarray,
                     views: list, confidences: list,
                     risk_aversion: float = 2.5, tau: float = 0.05) -> dict:
    n = len(market_weights)
    pi = risk_aversion * cov_matrix @ market_weights

    if not views:
        return {
            "weights": market_weights,
            "expected_return": pi,
            "posterior_return": pi,
        }

    n_views = len(views)
    P = np.zeros((n_views, n))
    Q = np.zeros(n_views)
    for i, (view_idx, view_return) in enumerate(views):
        if isinstance(view_idx, int) and 0 <= view_idx < n:
            P[i, view_idx] = 1
        Q[i] = view_return

    # Omega: uncertainty of views. Use diagonal with tau * sigma^2 * (1/c - 1)
    # confidences should have n_views elements (one per view)
    if len(confidences) != n_views:
        confidences = [0.5] * n_views

    omega = np.diag([tau * cov_matrix[min(i, n-1), min(i, n-1)] * (1 - c) / max(c, 0.01)
                     for i, c in enumerate(confidences)])

    try:
        tau_cov = tau * cov_matrix
        M1 = np.linalg.inv(tau_cov)
        M2 = P.T @ np.linalg.inv(omega) @ P
        posterior_cov = np.linalg.inv(M1 + M2)

        M3 = M1 @ pi + P.T @ np.linalg.inv(omega) @ Q
        posterior_return = posterior_cov @ M3

        try:
            inv_cov = np.linalg.inv(cov_matrix)
            weights = (1.0 / risk_aversion) * inv_cov @ posterior_return
        except np.linalg.LinAlgError:
            weights = posterior_return / (risk_aversion * np.diag(cov_matrix))
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = market_weights

        return {
            "weights": weights,
            "expected_return": pi,
            "posterior_return": posterior_return,
            "posterior_cov": posterior_cov,
        }
    except Exception:
        return {
            "weights": market_weights,
            "expected_return": pi,
            "posterior_return": pi,
        }


def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, n_points: int = 50) -> list:
    n = len(mean_returns)
    min_ret = min_variance_portfolio(mean_returns, cov_matrix)["return"]
    max_ret = float(np.max(mean_returns * 252))
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:
        def ret_deviation(w):
            return (w @ mean_returns * 252 - target) ** 2

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, 0.40)] * n
        w0 = np.ones(n) / n

        try:
            res = minimize(ret_deviation, w0, method="SLSQP", bounds=bounds, constraints=constraints)
            if res.success:
                stats = portfolio_stats(res.x, mean_returns, cov_matrix)
                frontier.append({
                    "return": stats["return"] * 252,
                    "volatility": stats["volatility"] * np.sqrt(252),
                    "weights": res.x,
                })
        except Exception:
            continue

    return frontier


def optimize_portfolio(prices: pd.DataFrame, views: list = None, confidences: list = None,
                       risk_free_rate: float = RISK_FREE_RATE) -> dict:
    returns = compute_returns(prices)
    mean_returns = returns.mean().values
    cov_matrix = compute_covariance(returns, method="shrinkage")
    tickers = list(prices.columns)
    n = len(tickers)

    market_w = np.ones(n) / n

    ms = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate / 252)
    mv = min_variance_portfolio(mean_returns, cov_matrix)

    bl = black_litterman(market_w, cov_matrix, views or [], confidences or [])

    frontier = efficient_frontier(mean_returns, cov_matrix, n_points=30)

    return {
        "tickers": tickers,
        "max_sharpe": {**ms, "tickers": tickers},
        "min_variance": {**mv, "tickers": tickers},
        "black_litterman": {**bl, "tickers": tickers},
        "efficient_frontier": frontier,
        "returns_stats": {
            "mean_daily": {t: float(mean_returns[i]) for i, t in enumerate(tickers)},
            "annualized": {t: float(mean_returns[i] * 252) for i, t in enumerate(tickers)},
        },
        "correlation": returns.corr().to_dict(),
    }
