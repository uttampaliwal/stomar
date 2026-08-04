"""Shared constants across the StoMar project."""

import os
from pathlib import Path

from src.core.settings import settings

# ─── Paths ───
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LEDGER_DB = os.path.join(DATA_DIR, "stomar.db")
META_CONTROLLER_PATH = os.path.join(MODELS_DIR, "meta_controller.pkl")
PAPER_STATE_PATH = os.path.join(DATA_DIR, "paper_state.json")
MF_STATE_PATH = os.path.join(DATA_DIR, "mf_state.json")
MONITORING_DIR = os.path.join(DATA_DIR, "monitoring")
FEATURE_VERSIONS_DIR = os.path.join(DATA_DIR, "feature_versions")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")

# ─── Trading Costs (NSE India) — from settings ───
BROKERAGE_RATE = settings.brokerage_rate
SLIPPAGE_RATE = settings.slippage_rate
STT_RATE = settings.stt_rate
EXCHANGE_CHARGE_RATE = settings.exchange_charge_rate
SEBI_FEES_RATE = settings.sebi_fees_rate
STAMP_DUTY_BUY_RATE = settings.stamp_duty_buy_rate
GST_RATE = settings.gst_rate

# ─── Risk Parameters — from settings ───
RISK_FREE_RATE = settings.risk_free_rate
MIN_DENOMINATOR = 0.001          # Guard against division by zero

# ─── Capital Gains Tax (India, P5.5) ───
STCG_TAX_RATE = 0.15             # Short-term: held < 1 year
LTCG_TAX_RATE = 0.10             # Long-term: held >= 1 year
LTCG_EXEMPTION = 100_000.0       # ₹1 lakh LTCG exemption per financial year
LONG_TERM_HOLDING_DAYS = 365     # Holding period that qualifies as long-term

# ─── Trading Days ───
TRADING_DAYS_PER_YEAR = 252      # Approximate NSE trading days per year

# ─── Model Training — from settings ───
DEFAULT_SEQ_LENGTH = settings.default_seq_length
DEFAULT_EPOCHS = settings.default_epochs
DEFAULT_BATCH_SIZE = settings.default_batch_size
DEFAULT_LEARNING_RATE = settings.default_learning_rate

# ─── Ensemble Weights (Equal by default) ───
DEFAULT_ENSEMBLE_WEIGHTS = {
    "lstm": 0.20,
    "gru": 0.20,
    "transformer": 0.20,
    "xgb": 0.20,
    "lgb": 0.20,
}

# ─── NSE Transaction Cost Dictionary ───
NSE_TRANSACTION_COSTS = {
    "brokerage_pct": BROKERAGE_RATE,
    "stt_pct": STT_RATE,
    "exchange_charge_pct": EXCHANGE_CHARGE_RATE,
    "sebi_fees_pct": SEBI_FEES_RATE,
    "stamp_duty_buy_pct": STAMP_DUTY_BUY_RATE,
    "gst_pct": GST_RATE,
}


def calculate_nse_costs(price: float, quantity: int, side: str) -> dict:
    """Calculate full NSE transaction costs for Indian equity delivery trades.

    Args:
        price: Execution price per share
        quantity: Number of shares
        side: "buy" or "sell" (case-insensitive)

    Returns:
        Dict with cost breakdown and total
    """
    trade_value = price * quantity
    side = side.lower().strip()

    brokerage = trade_value * BROKERAGE_RATE
    exchange_charge = trade_value * EXCHANGE_CHARGE_RATE
    sebi_fees = trade_value * SEBI_FEES_RATE
    gst = (brokerage + exchange_charge) * GST_RATE

    # STT is charged on both buy and sell sides for delivery at 0.1%
    stt = trade_value * STT_RATE

    if side == "buy":
        stamp_duty = trade_value * STAMP_DUTY_BUY_RATE
    else:
        stamp_duty = 0.0

    total = brokerage + stt + exchange_charge + sebi_fees + stamp_duty + gst

    return {
        "brokerage": round(brokerage, 4),
        "stt": round(stt, 4),
        "exchange_charge": round(exchange_charge, 4),
        "sebi_fees": round(sebi_fees, 4),
        "stamp_duty": round(stamp_duty, 4),
        "gst": round(gst, 4),
        "total": round(total, 4),
        "effective_rate": round(total / trade_value, 6) if trade_value > 0 else 0.0,
    }
