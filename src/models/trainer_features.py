"""Trainer feature contract: required vs optional feature columns.

Defined separately from ``src.models.trainer`` so the contract is importable
without pulling in torch (pure pandas module).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "close", "volume", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "bb_width", "atr", "obv", "volume_ratio",
    "high_low_pct", "close_open_pct", "close_position",
    "returns_1d", "returns_2d", "returns_3d", "returns_5d", "returns_10d", "returns_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "day_of_week", "month", "quarter", "day_of_month",
    "sentiment_score", "fii_net", "dii_net", "flow_signal",
    "pcr", "mtf_signal", "mtf_confidence",
    "stoch_k", "stoch_d", "williams_r", "cci", "mfi",
    "adx", "vwap",
    # SOTA factor pipeline (appended to preserve alignment with legacy models)
    "supertrend", "supertrend_dir",
    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
    "ichimoku_senkou_b",
    "cmf", "ofi", "vol_zscore", "pcr_slope", "fii_momentum",
    "mom_1m_voladj", "mom_3m_voladj", "mom_6m_voladj", "mom_12m_voladj",
    "rel_strength_1m", "rel_strength_3m", "high_low_spread",
]

# Anchors produced unconditionally by add_technical_indicators(); their loss
# indicates a broken pipeline, not a short history.
REQUIRED_FEATURES = ["close", "volume", "returns_1d"]

# Everything else is optional: missing values are logged, not fatal
# (e.g. factor-pipeline columns absent for very short histories).
OPTIONAL_FEATURES = [c for c in FEATURE_COLS if c not in REQUIRED_FEATURES]


def select_training_features(df_feat: pd.DataFrame, ticker: str) -> list[str]:
    """Select FEATURE_COLS present in the frame, validating required ones.

    Required features missing → ValueError: silently training without them
    would yield different models per ticker. Optional features missing →
    warning only (short histories legitimately lack e.g. sma_50 or
    factor-pipeline columns).
    """
    missing_required = sorted(set(REQUIRED_FEATURES) - set(df_feat.columns))
    if missing_required:
        raise ValueError(
            f"required features missing for {ticker}: {missing_required} "
            "(broken feature pipeline)"
        )
    missing_optional = sorted(set(OPTIONAL_FEATURES) - set(df_feat.columns))
    if missing_optional:
        logger.warning(
            "optional_features_missing ticker=%s missing=%s "
            "(training on %d/%d FEATURE_COLS)",
            ticker, missing_optional,
            len(FEATURE_COLS) - len(missing_optional), len(FEATURE_COLS),
        )
    return [c for c in FEATURE_COLS if c in df_feat.columns]


def build_feature_frame(df: pd.DataFrame, ticker: str | None = None,
                        index_df: pd.DataFrame | None = None,
                        load_index: bool = True) -> pd.DataFrame:
    """Build the trainer-identical feature frame for inference (O2 parity).

    Every consumer (trainer, daily orchestrator, pipeline hash, backfill,
    edge gate) must build features through this single function. Previously
    inference paths used bare ``add_technical_indicators`` and silently
    zero-filled the 16 SOTA factors at serve time.

    Args:
        df: OHLCV data indexed by date with open/high/low/close/volume.
        ticker: Optional ticker for external PIT features (sentiment, flows,
            PCR, MTF). None → neutral defaults (historical convention).
        index_df: NIFTY history for relative strength, or None to skip.
            Defaults to best-effort local load (never raises).

    Returns:
        Feature DataFrame (all-NaN columns dropped, like training).
        Never raises for missing optionals; raises ValueError on empty input
        (from compute_features) or missing REQUIRED_FEATURES.
    """
    from src.signals.feature_pipeline import compute_features

    if index_df is None and load_index:
        try:
            from src.signals.feature_pipeline import load_index_history

            index_df = load_index_history(period="5y")
        except Exception as exc:
            logger.debug("index history unavailable, continuing without it: %s", exc)

    df_feat = compute_features(df, ticker=ticker, index_df=index_df)
    df_feat = df_feat.dropna(axis=1, how="all")

    missing_required = sorted(set(REQUIRED_FEATURES) - set(df_feat.columns))
    if missing_required:
        raise ValueError(f"required features missing: {missing_required}")
    return df_feat
