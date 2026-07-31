"""Wealth goal planning endpoints — Monte Carlo simulation and strategies."""

import numpy as np
from fastapi import APIRouter, Body

router = APIRouter()

# Static wealth strategies
WEALTH_STRATEGIES = [
    {
        "name": "Dual Momentum",
        "description": "Combines absolute and relative momentum to switch between equity and safe assets. Captures upside while limiting drawdowns.",
        "cagr": "14.2%",
        "max_drawdown": "12.8%",
        "sharpe": "1.35",
        "complexity": "Medium",
    },
    {
        "name": "Magic Formula (Greenblatt)",
        "description": "Ranks stocks by earnings yield (EV/EBIT) and return on capital. Buys top-ranked, rebalances annually. Exploits value + quality cheaply.",
        "cagr": "16.8%",
        "max_drawdown": "18.5%",
        "sharpe": "1.12",
        "complexity": "Low",
    },
    {
        "name": "Core-Satellite Black-Litterman",
        "description": "Market-cap weighted core (Nifty 50 ETF) + alpha-generating satellite tilted by model views. Bayesian optimization keeps turnover low.",
        "cagr": "13.5%",
        "max_drawdown": "14.2%",
        "sharpe": "1.22",
        "complexity": "High",
    },
    {
        "name": "Risk Parity",
        "description": "Equalizes risk contribution across asset classes (equity, gold, debt). Rebalanced monthly. Resilient in regime shifts.",
        "cagr": "11.4%",
        "max_drawdown": "9.6%",
        "sharpe": "1.41",
        "complexity": "Medium",
    },
]


@router.get("/strategies")
def get_strategies():
    """Return curated quant wealth strategies."""
    return {"strategies": WEALTH_STRATEGIES}


@router.post("/monte-carlo")
def monte_carlo_simulation(data: dict = Body(...)):
    """Run goal-based Monte Carlo SIP simulation.

    Body params:
        target_corpus: float — target amount in INR
        current_capital: float — lumpsum already invested
        monthly_sip: float — monthly SIP contribution
        horizon_years: int — investment horizon in years
        expected_return: float — annualized expected return (default 0.12)
        volatility: float — annualized volatility (default 0.18)
        inflation_rate: float — expected inflation (default 0.06)
        num_simulations: int — number of paths (default 500)
    """
    try:
        target = float(data.get("target_corpus", 10_000_000))
        current = float(data.get("current_capital", 0))
        sip = float(data.get("monthly_sip", 10000))
        horizon = int(data.get("horizon_years", 10))
        exp_return = float(data.get("expected_return", 0.12))
        vol = float(data.get("volatility", 0.18))
        inflation = float(data.get("inflation_rate", 0.06))
        n_sims = int(data.get("num_simulations", 500))

        n_months = horizon * 12
        monthly_return_mean = exp_return / 12
        monthly_vol = vol / np.sqrt(12)

        np.random.seed(42)
        all_paths = np.zeros((n_sims, n_months + 1))
        all_paths[:, 0] = current

        for t in range(1, n_months + 1):
            z = np.random.standard_normal(n_sims)
            monthly_return = monthly_return_mean + z * monthly_vol
            all_paths[:, t] = all_paths[:, t - 1] * (1 + monthly_return) + sip

        # Percentile timelines (yearly snapshots)
        year_indices = list(range(0, n_months + 1, 12))
        p10 = [float(np.percentile(all_paths[:, i], 10)) for i in year_indices]
        p50 = [float(np.percentile(all_paths[:, i], 50)) for i in year_indices]
        p90 = [float(np.percentile(all_paths[:, i], 90)) for i in year_indices]
        labels = [f"Year {i}" for i in range(len(year_indices))]

        final_values = all_paths[:, -1]
        prob_success = float(np.mean(final_values >= target))
        median_corpus = float(np.median(final_values))
        real_corpus = median_corpus / ((1 + inflation) ** horizon)

        return {
            "probability": round(prob_success * 100, 1),
            "median_corpus": round(median_corpus, 0),
            "real_corpus": round(real_corpus, 0),
            "target": target,
            "percentiles": {"p10": p10, "p50": p50, "p90": p90, "labels": labels},
            "params": {
                "current_capital": current,
                "monthly_sip": sip,
                "horizon_years": horizon,
                "expected_return": exp_return,
                "volatility": vol,
                "inflation_rate": inflation,
                "simulations": n_sims,
            },
        }
    except Exception as e:
        return {"error": str(e)}
