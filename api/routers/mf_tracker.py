"""Mutual fund tracker endpoint."""

import os
from fastapi import APIRouter
from src.mf_tracker import MFTracker

router = APIRouter()


def get_tracker():
    tracker = MFTracker()
    state_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mf_state.json")
    if os.path.exists(state_path):
        try:
            tracker.load_state(state_path)
        except Exception:
            pass
    return tracker


@router.get("/portfolio")
def mf_portfolio():
    try:
        tracker = get_tracker()
        return {
            "invested": round(tracker.get_invested_value(), 2),
            "current_value": round(tracker.get_portfolio_value(), 2),
            "pnl": round(tracker.get_total_pnl(), 2),
            "return_pct": round(tracker.get_total_return_pct() * 100, 2),
            "holdings": tracker.get_top_holdings(20),
            "allocation": tracker.get_allocation_breakdown(),
            "concentration_risk": tracker.detect_concentration_risk(),
        }
    except Exception as e:
        return {"error": str(e)}
