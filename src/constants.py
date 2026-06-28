"""Shared constants across the StoMar project."""

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
