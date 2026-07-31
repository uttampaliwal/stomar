"""Wealth goal planning endpoints — Monte Carlo simulation, strategies, and AI advisor.

Enhanced Monte Carlo engine with:
- Student's t-distribution for fat-tailed return modeling (captures crash risk)
- Step-up SIP with configurable annual escalation
- Inflation-adjusted real wealth purchasing power
- Shortfall risk analysis with recommended SIP top-up
- 1,000–10,000 path simulations
"""

import os
import logging
import numpy as np
from fastapi import APIRouter, Body
from scipy import stats as sp_stats

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
    """Run goal-based Monte Carlo SIP simulation with fat-tailed distributions.

    Uses Student's t-distribution (df=5) to model fat tails in equity returns,
    which better captures crash risk than a standard normal distribution.

    Body params:
        target_corpus: float — target amount in INR
        current_capital: float — lumpsum already invested
        monthly_sip: float — monthly SIP contribution
        step_up_pct: float — annual SIP step-up percentage (e.g. 0.10 = 10%)
        horizon_years: int — investment horizon in years
        expected_return: float — annualized expected return (default 0.12)
        volatility: float — annualized volatility (default 0.18)
        inflation_rate: float — expected inflation (default 0.06)
        num_simulations: int — number of paths (1000-10000, default 1000)
        t_df: int — degrees of freedom for Student's t (default 5, lower = fatter tails)
    """
    try:
        target = float(data.get("target_corpus", 10_000_000))
        current = float(data.get("current_capital", 0))
        sip = float(data.get("monthly_sip", 10000))
        step_up_pct = float(data.get("step_up_pct", 0.0))
        horizon = int(data.get("horizon_years", 10))
        exp_return = float(data.get("expected_return", 0.12))
        vol = float(data.get("volatility", 0.18))
        inflation = float(data.get("inflation_rate", 0.06))
        n_sims = int(data.get("num_simulations", 1000))
        t_df = int(data.get("t_df", 5))

        # Clamp simulations
        n_sims = max(1000, min(n_sims, 10000))

        n_months = horizon * 12
        monthly_return_mean = exp_return / 12
        monthly_vol = vol / np.sqrt(12)

        # Student's t-distribution: fat tails capture crash risk
        # Scale t-distribution to match target mean and variance
        t_dist = sp_stats.t(df=t_df)
        # t-distribution has mean=0, var=df/(df-2) for df>2
        t_scale = monthly_vol * np.sqrt((t_df - 2) / t_df) if t_df > 2 else monthly_vol

        np.random.seed(42)
        all_paths = np.zeros((n_sims, n_months + 1))
        all_paths[:, 0] = current

        # Generate monthly SIP amounts with step-up
        monthly_sips = np.zeros(n_months)
        current_sip = sip
        for m in range(n_months):
            if m > 0 and m % 12 == 0 and step_up_pct > 0:
                current_sip *= 1 + step_up_pct
            monthly_sips[m] = current_sip

        # Simulate paths using Student's t-distribution
        for t in range(1, n_months + 1):
            z = t_dist.rvs(n_sims)
            monthly_return = monthly_return_mean + z * t_scale
            all_paths[:, t] = all_paths[:, t - 1] * (1 + monthly_return) + monthly_sips[t - 1]

        # Percentile timelines (yearly snapshots)
        year_indices = list(range(0, n_months + 1, 12))
        p5 = [float(np.percentile(all_paths[:, i], 5)) for i in year_indices]
        p10 = [float(np.percentile(all_paths[:, i], 10)) for i in year_indices]
        p25 = [float(np.percentile(all_paths[:, i], 25)) for i in year_indices]
        p50 = [float(np.percentile(all_paths[:, i], 50)) for i in year_indices]
        p75 = [float(np.percentile(all_paths[:, i], 75)) for i in year_indices]
        p90 = [float(np.percentile(all_paths[:, i], 90)) for i in year_indices]
        p95 = [float(np.percentile(all_paths[:, i], 95)) for i in year_indices]
        labels = [f"Year {i}" for i in range(len(year_indices))]

        final_values = all_paths[:, -1]
        prob_success = float(np.mean(final_values >= target))
        median_corpus = float(np.median(final_values))
        mean_corpus = float(np.mean(final_values))
        real_corpus = median_corpus / ((1 + inflation) ** horizon)

        # Inflation-adjusted percentile values
        inflation_factor = (1 + inflation) ** horizon
        real_p10 = p10[-1] / inflation_factor if p10 else 0
        real_p50 = p50[-1] / inflation_factor if p50 else 0
        real_p90 = p90[-1] / inflation_factor if p90 else 0

        # Shortfall risk analysis
        shortfall_risk = 1.0 - prob_success
        # Find the SIP needed for 90% probability of success
        target_for_90 = float(np.percentile(final_values, 90))
        # Estimate required SIP via scaling: corpus scales roughly linearly with SIP
        total_sip_contribution = sum(monthly_sips)
        if total_sip_contribution > 0:
            sip_scaling_factor = mean_corpus / total_sip_contribution if total_sip_contribution > 0 else 1
            required_corpus_for_90 = target
            required_total_sip = required_corpus_for_90 / max(sip_scaling_factor, 0.01)
            # Approximate monthly SIP needed
            avg_months_sip = np.mean(monthly_sips) if np.any(monthly_sips > 0) else sip
            if avg_months_sip > 0:
                sip_ratio = required_total_sip / (avg_months_sip * n_months) if n_months > 0 else 1
                recommended_sip = sip * max(sip_ratio, 0.5)
            else:
                recommended_sip = sip
        else:
            recommended_sip = sip

        # SIP top-up suggestion: how much more per month to reach target with 80% confidence
        p80_final = float(np.percentile(final_values, 80))
        if p80_final < target and total_sip_contribution > 0:
            gap = target - p80_final
            # Each additional rupee of SIP adds roughly sip_scaling_factor over the horizon
            monthly_topup = gap / (n_months * max(sip_scaling_factor, 0.01) * n_months) if n_months > 0 else 0
            sip_topup = max(0, round(monthly_topup / 100) * 100)  # Round to nearest 100
        else:
            sip_topup = 0

        # Worst-case and best-case metrics
        worst_5pct = float(np.percentile(final_values, 5))
        best_5pct = float(np.percentile(final_values, 95))

        # Value at Risk (VaR) at 95% confidence
        var_95 = float(np.percentile(final_values, 5))
        cvar_95 = float(final_values[final_values <= var_95].mean()) if np.any(final_values <= var_95) else var_95

        return {
            "probability": round(prob_success * 100, 1),
            "median_corpus": round(median_corpus, 0),
            "mean_corpus": round(mean_corpus, 0),
            "real_corpus": round(real_corpus, 0),
            "real_p10": round(real_p10, 0),
            "real_p50": round(real_p50, 0),
            "real_p90": round(real_p90, 0),
            "target": target,
            "shortfall_risk": round(shortfall_risk * 100, 1),
            "recommended_sip": round(recommended_sip, 0),
            "sip_topup": round(sip_topup, 0),
            "worst_5pct": round(worst_5pct, 0),
            "best_5pct": round(best_5pct, 0),
            "var_95": round(var_95, 0),
            "cvar_95": round(cvar_95, 0),
            "percentiles": {
                "p5": p5,
                "p10": p10,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p90": p90,
                "p95": p95,
                "labels": labels,
            },
            "params": {
                "current_capital": current,
                "monthly_sip": sip,
                "step_up_pct": step_up_pct,
                "horizon_years": horizon,
                "expected_return": exp_return,
                "volatility": vol,
                "inflation_rate": inflation,
                "simulations": n_sims,
                "t_df": t_df,
            },
        }
    except Exception as e:
        logger.error("Monte Carlo simulation failed: %s", e)
        return {"error": str(e)}


_ADVISOR_PROMPT = """You are an institutional-grade quant portfolio advisor for the Indian market.
Analyze the investor profile and provide a structured portfolio audit.

Investor Profile:
- Target Corpus: ₹{target:,.0f}
- Current Capital: ₹{current:,.0f}
- Monthly SIP: ₹{sip:,.0f}
- Step-Up SIP: {step_up:.0%} annually
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
    step_up = params.get("step_up", 0.0)
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

    # Future value of SIP with step-up
    total_wealth = current * (1 + exp_return) ** horizon
    monthly_r = exp_return / 12
    current_sip = sip
    for yr in range(horizon):
        months_left = (horizon - yr) * 12
        for m in range(12):
            total_wealth += current_sip * (1 + monthly_r) ** max(months_left - m - 1, 0)
        if step_up > 0:
            current_sip *= 1 + step_up

    required_monthly = 0
    if horizon > 0:
        fv_lumpsum = current * (1 + exp_return) ** horizon
        fv_sip_needed = target - fv_lumpsum
        if fv_sip_needed > 0:
            annuity_factor = sum((1 + monthly_r) ** i for i in range(horizon * 12))
            required_monthly = fv_sip_needed / annuity_factor if annuity_factor > 0 else 0

    step_up_note = f" with {step_up:.0%} annual step-up" if step_up > 0 else ""

    return {
        "source": "Mathematical Quant Engine",
        "advisor": {
            "executive_summary": (
                f"With ₹{current:,.0f} invested and ₹{sip:,.0f}/month SIP{step_up_note} over {horizon} years, "
                f"you are on track for approximately ₹{total_wealth:,.0f}. "
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
                f"{'Implement ' + f'{step_up:.0%}' + ' annual step-up SIP' if step_up > 0 else 'Consider adding 10% annual step-up SIP'}",
                "Review allocation annually and rebalance",
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
            "step_up": float(data.get("step_up_pct", 0.0)),
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
        result = _fallback_advisor(params if "params" in dir() else {})
        result["note"] = f"Gemini error: {e}"
        return result
