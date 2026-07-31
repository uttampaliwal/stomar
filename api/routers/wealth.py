"""Wealth goal planning endpoints — Monte Carlo simulation and strategies."""

import os
import logging
import numpy as np
from fastapi import APIRouter, Body

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to load Gemini AI SDK
_genai = None
try:
    from google import genai as _genai_sdk
    _api_key = os.environ.get("GEMINI_API_KEY")
    if _api_key:
        _genai = _genai_sdk.Client(api_key=_api_key)
except ImportError:
    pass

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


_ADVISOR_PROMPT = """You are an institutional-grade quant portfolio advisor for the Indian market.
Analyze the investor profile and provide a structured portfolio audit.

Investor Profile:
- Target Corpus: ₹{target:,.0f}
- Current Capital: ₹{current:,.0f}
- Monthly SIP: ₹{sip:,.0f}
- Horizon: {horizon} years
- Expected Return: {exp_return:.1%}
- Risk Tolerance: {risk_tolerance}

Provide a JSON response with:
{{
  "executive_summary": "2-3 sentence overview",
  "asset_allocation": {{"equity_pct": number, "debt_pct": number, "gold_pct": number, "cash_pct": number}},
  "recommended_funds": ["fund1", "fund2", "fund3"],
  "action_items": ["action1", "action2", "action3"],
  "risk_warnings": ["warning1", "warning2"],
  "expected_cagr": "X%",
  "expected_timeline": "X years"
}}"""


def _fallback_advisor(params: dict) -> dict:
    """Mathematical fallback when Gemini is unavailable."""
    current = params.get("current", 0)
    sip = params.get("sip", 10000)
    horizon = params.get("horizon", 10)
    target = params.get("target", 10_000_000)
    exp_return = params.get("exp_return", 0.12)

    # Simple rule-based allocation
    if horizon <= 3:
        alloc = {"equity_pct": 40, "debt_pct": 40, "gold_pct": 10, "cash_pct": 10}
    elif horizon <= 7:
        alloc = {"equity_pct": 60, "debt_pct": 25, "gold_pct": 10, "cash_pct": 5}
    else:
        alloc = {"equity_pct": 75, "debt_pct": 15, "gold_pct": 7, "cash_pct": 3}

    required_monthly = (target - current * (1 + exp_return) ** horizon) / (
        sum((1 + exp_return / 12) ** i for i in range(horizon * 12))
    ) if horizon > 0 else 0

    return {
        "source": "Mathematical Quant Engine",
        "advisor": {
            "executive_summary": (
                f"With ₹{current:,.0f} invested and ₹{sip:,.0f}/month SIP over {horizon} years, "
                f"you are on track for approximately ₹{(current * (1 + exp_return) ** horizon + sip * sum((1 + exp_return / 12) ** i for i in range(horizon * 12))):,.0f}. "
                f"{'Target achievable.' if required_monthly <= sip else f'Consider increasing SIP by ₹{(required_monthly - sip):,.0f}/month.'}"
            ),
            "asset_allocation": alloc,
            "recommended_funds": [
                "Nifty 50 Index Fund (core)",
                "Nifty Next 50 Index Fund (satellite)",
                "Short Duration Debt Fund (stability)",
                "Gold ETF (hedge)",
            ],
            "action_items": [
                f"Maintain monthly SIP of ₹{sip:,.0f}",
                "Review allocation annually and rebalance",
                "Increase SIP by 10% each year (step-up SIP)",
            ],
            "risk_warnings": [
                "Past performance does not guarantee future returns",
                "Equity investments are subject to market risk",
                f"At {exp_return:.0%} expected return, worst-case 1-year loss could be ~{exp_return + 2 * 0.18:.0%}",
            ],
            "expected_cagr": f"{exp_return:.1%}",
            "expected_timeline": f"{horizon} years",
        },
    }


@router.post("/wealth-advisor")
def wealth_advisor(data: dict = Body(...)):
    """AI-powered portfolio audit using Gemini, with mathematical fallback."""
    try:
        params = {
            "target": float(data.get("target_corpus", 10_000_000)),
            "current": float(data.get("current_capital", 0)),
            "sip": float(data.get("monthly_sip", 10000)),
            "horizon": int(data.get("horizon_years", 10)),
            "exp_return": float(data.get("expected_return", 0.12)),
            "risk_tolerance": data.get("risk_tolerance", "Moderate"),
        }

        if _genai is None:
            result = _fallback_advisor(params)
            result["note"] = "Gemini API not configured — using mathematical engine"
            return result

        prompt = _ADVISOR_PROMPT.format(**params)
        response = _genai.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        import json
        advisor = json.loads(response.text)
        return {"source": "Gemini AI Quant Advisor", "advisor": advisor}

    except Exception as e:
        logger.warning("Gemini advisor failed, using fallback: %s", e)
        result = _fallback_advisor(params if 'params' in dir() else {})
        result["note"] = f"Gemini error: {e}"
        return result
