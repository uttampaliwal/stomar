"""Shared constants across the StoMar project."""

import os

# ─── Paths ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LEDGER_DB = os.path.join(DATA_DIR, "stomar.db")
META_CONTROLLER_PATH = os.path.join(MODELS_DIR, "meta_controller.pkl")
PAPER_STATE_PATH = os.path.join(DATA_DIR, "paper_state.json")
MF_STATE_PATH = os.path.join(DATA_DIR, "mf_state.json")

# ─── Trading Costs (NSE India) ───
BROKERAGE_RATE = 0.0003          # 0.03% per side (Zerodha delivery)
SLIPPAGE_RATE = 0.001            # 0.1% slippage estimate
STT_SELL_RATE = 0.001            # 0.1% Securities Transaction Tax (sell side)
EXCHANGE_CHARGE_RATE = 0.0000345 # NSE exchange transaction charges
SEBI_FEES_RATE = 0.000001        # SEBI turnover fees
STAMP_DUTY_BUY_RATE = 0.00015   # Stamp duty (buy side)
GST_RATE = 0.18                  # 18% GST on brokerage + exchange charges

# ─── Risk Parameters ───
RISK_FREE_RATE = 0.065           # 6.5% Indian 10-Year G-Sec yield
MIN_DENOMINATOR = 0.001          # Guard against division by zero

# ─── Trading Days ───
TRADING_DAYS_PER_YEAR = 252      # Approximate NSE trading days per year

# ─── Model Training ───
DEFAULT_SEQ_LENGTH = 60          # LSTM/GRU/Transformer sequence length
DEFAULT_EPOCHS = 40              # Training epochs
DEFAULT_BATCH_SIZE = 32          # Training batch size
DEFAULT_LEARNING_RATE = 0.001    # Adam learning rate

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
    "stt_sell_pct": STT_SELL_RATE,
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
        side: "buy" or "sell"

    Returns:
        Dict with cost breakdown and total
    """
    trade_value = price * quantity

    brokerage = trade_value * BROKERAGE_RATE
    exchange_charge = trade_value * EXCHANGE_CHARGE_RATE
    sebi_fees = trade_value * SEBI_FEES_RATE
    gst = (brokerage + exchange_charge) * GST_RATE

    if side == "sell":
        stt = trade_value * STT_SELL_RATE
        stamp_duty = 0.0
    else:
        stt = 0.0
        stamp_duty = trade_value * STAMP_DUTY_BUY_RATE

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
