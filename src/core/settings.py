"""Centralized configuration via environment variables.

All tunable parameters are loaded from environment variables with sensible
defaults matching the original hardcoded values. Override via .env file or
actual environment variables.

Usage:
    from src.core.settings import settings
    print(settings.risk_free_rate)
    print(settings.default_epochs)
"""

import os
from pathlib import Path

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    _HAS_PYDANTIC_SETTINGS = False

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


if _HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        """Application settings — all values configurable via env vars."""

        model_config = {"env_prefix": "STOMAR_", "env_file": ".env", "env_file_encoding": "utf-8"}

        # ── Environment ────────────────────────────────────────────────────
        env: str = Field(default="dev", description="dev or production")
        api_key: str = Field(default="", description="API key for protected endpoints")
        cors_origins: str = Field(default="", description="Comma-separated CORS origins (production)")

        # ── Trading Costs (NSE India) ──────────────────────────────────────
        brokerage_rate: float = Field(default=0.0003, description="Brokerage per side (0.03% Zerodha)")
        slippage_rate: float = Field(default=0.001, description="Slippage estimate (0.1%)")
        stt_rate: float = Field(default=0.001, description="Securities Transaction Tax (0.1%)")
        exchange_charge_rate: float = Field(default=0.0000345, description="NSE exchange charges")
        sebi_fees_rate: float = Field(default=0.000001, description="SEBI turnover fees")
        stamp_duty_buy_rate: float = Field(default=0.00015, description="Stamp duty (buy side)")
        gst_rate: float = Field(default=0.18, description="GST on brokerage + exchange charges")

        # ── Risk Parameters ────────────────────────────────────────────────
        risk_free_rate: float = Field(default=0.065, description="Risk-free rate (Indian G-Sec yield)")

        # ── Model Training ─────────────────────────────────────────────────
        default_seq_length: int = Field(default=60, description="LSTM/GRU/Transformer sequence length")
        default_epochs: int = Field(default=40, description="Training epochs")
        default_batch_size: int = Field(default=32, description="Training batch size")
        default_learning_rate: float = Field(default=0.001, description="Adam learning rate")

        # ── API Cache ──────────────────────────────────────────────────────
        cache_ttl: int = Field(default=30, description="HTTP response cache TTL (seconds)")
        cache_max_entries: int = Field(default=50, description="Max cache entries before eviction")

        # ── Data Cache ─────────────────────────────────────────────────────
        fetch_cache_ttl: int = Field(default=600, description="In-memory data cache TTL (seconds)")
        model_cache_ttl: int = Field(default=3600, description="Model cache TTL (seconds)")

        # ── Meta-Controller ────────────────────────────────────────────────
        max_position_pct: float = Field(default=0.10, description="Max position size as fraction of equity")
        buy_threshold: float = Field(default=0.6, description="Buy signal threshold")
        sell_threshold: float = Field(default=0.4, description="Sell signal threshold")
        min_confidence_to_trade: float = Field(default=0.3, description="Minimum confidence to trade")
        min_samples_to_train: int = Field(default=500, description="Min resolved samples to train meta-controller")
        meta_controller_c: float = Field(default=0.1, description="Regularization strength (inverse) for meta-controller LogisticRegression")

        # ── Risk Controls ──────────────────────────────────────────────────
        max_daily_loss_pct: float = Field(default=0.02, description="Daily loss limit")
        max_weekly_loss_pct: float = Field(default=0.05, description="Weekly loss limit")
        max_drawdown_pct: float = Field(default=0.15, description="Max drawdown limit")

        # ── Orchestrator ───────────────────────────────────────────────────
        max_daily_trades: int = Field(default=5, description="Max trades per day")
        max_portfolio_exposure: float = Field(default=0.50, description="Max portfolio exposure")
        stop_loss_pct: float = Field(default=0.05, description="Stop-loss percentage")

        # ── Circuit Breaker ────────────────────────────────────────────────
        yf_failure_threshold: int = Field(default=5, description="yfinance breaker failure threshold")
        yf_recovery_timeout: int = Field(default=120, description="yfinance breaker recovery (seconds)")
        nse_failure_threshold: int = Field(default=3, description="NSE breaker failure threshold")
        nse_recovery_timeout: int = Field(default=90, description="NSE breaker recovery (seconds)")

        # ── Pipeline ───────────────────────────────────────────────────────
        min_oos_accuracy: float = Field(default=0.50, description="Min out-of-sample accuracy gate")
        min_oos_sharpe: float = Field(default=0.0, description="Min out-of-sample Sharpe gate")
        max_drawdown_threshold: float = Field(default=0.30, description="Max drawdown gate for pipeline")

        # ── Data Retention ─────────────────────────────────────────────────
        cache_retention_days: int = Field(default=7, description="Days to keep stale cache files (parquet, pkl, json)")
        ledger_retention_days: int = Field(default=90, description="Days to keep ledger rows before archival")
        monitoring_retention_days: int = Field(default=30, description="Days to keep monitoring JSON files")
        feature_versions_keep: int = Field(default=10, description="Number of recent feature version files to keep")
        pipeline_logs_retention_days: int = Field(default=14, description="Days to keep pipeline log files")


    settings = Settings()
else:
    # Fallback when pydantic-settings is not installed — read env vars directly
    class _FallbackSettings:
        """Reads STOMAR_* env vars with hardcoded defaults."""

        def __init__(self):
            self.env = os.environ.get("STOMAR_ENV", "dev")
            self.api_key = os.environ.get("STOMAR_API_KEY", "")
            self.cors_origins = os.environ.get("STOMAR_CORS_ORIGINS", "")
            self.brokerage_rate = float(os.environ.get("STOMAR_BROKERAGE_RATE", "0.0003"))
            self.slippage_rate = float(os.environ.get("STOMAR_SLIPPAGE_RATE", "0.001"))
            self.stt_rate = float(os.environ.get("STOMAR_STT_RATE", "0.001"))
            self.exchange_charge_rate = float(os.environ.get("STOMAR_EXCHANGE_CHARGE_RATE", "0.0000345"))
            self.sebi_fees_rate = float(os.environ.get("STOMAR_SEBI_FEES_RATE", "0.000001"))
            self.stamp_duty_buy_rate = float(os.environ.get("STOMAR_STAMP_DUTY_BUY_RATE", "0.00015"))
            self.gst_rate = float(os.environ.get("STOMAR_GST_RATE", "0.18"))
            self.risk_free_rate = float(os.environ.get("STOMAR_RISK_FREE_RATE", "0.065"))
            self.default_seq_length = int(os.environ.get("STOMAR_DEFAULT_SEQ_LENGTH", "60"))
            self.default_epochs = int(os.environ.get("STOMAR_DEFAULT_EPOCHS", "40"))
            self.default_batch_size = int(os.environ.get("STOMAR_DEFAULT_BATCH_SIZE", "32"))
            self.default_learning_rate = float(os.environ.get("STOMAR_DEFAULT_LEARNING_RATE", "0.001"))
            self.cache_ttl = int(os.environ.get("STOMAR_CACHE_TTL", "30"))
            self.cache_max_entries = int(os.environ.get("STOMAR_CACHE_MAX_ENTRIES", "50"))
            self.fetch_cache_ttl = int(os.environ.get("STOMAR_FETCH_CACHE_TTL", "600"))
            self.model_cache_ttl = int(os.environ.get("STOMAR_MODEL_CACHE_TTL", "3600"))
            self.max_position_pct = float(os.environ.get("STOMAR_MAX_POSITION_PCT", "0.10"))
            self.buy_threshold = float(os.environ.get("STOMAR_BUY_THRESHOLD", "0.6"))
            self.sell_threshold = float(os.environ.get("STOMAR_SELL_THRESHOLD", "0.4"))
            self.min_confidence_to_trade = float(os.environ.get("STOMAR_MIN_CONFIDENCE_TO_TRADE", "0.3"))
            self.min_samples_to_train = int(os.environ.get("STOMAR_MIN_SAMPLES_TO_TRAIN", "500"))
            self.meta_controller_c = float(os.environ.get("STOMAR_META_CONTROLLER_C", "0.1"))
            self.max_daily_loss_pct = float(os.environ.get("STOMAR_MAX_DAILY_LOSS_PCT", "0.02"))
            self.max_weekly_loss_pct = float(os.environ.get("STOMAR_MAX_WEEKLY_LOSS_PCT", "0.05"))
            self.max_drawdown_pct = float(os.environ.get("STOMAR_MAX_DRAWDOWN_PCT", "0.15"))
            self.max_daily_trades = int(os.environ.get("STOMAR_MAX_DAILY_TRADES", "5"))
            self.max_portfolio_exposure = float(os.environ.get("STOMAR_MAX_PORTFOLIO_EXPOSURE", "0.50"))
            self.stop_loss_pct = float(os.environ.get("STOMAR_STOP_LOSS_PCT", "0.05"))
            self.yf_failure_threshold = int(os.environ.get("STOMAR_YF_FAILURE_THRESHOLD", "5"))
            self.yf_recovery_timeout = int(os.environ.get("STOMAR_YF_RECOVERY_TIMEOUT", "120"))
            self.nse_failure_threshold = int(os.environ.get("STOMAR_NSE_FAILURE_THRESHOLD", "3"))
            self.nse_recovery_timeout = int(os.environ.get("STOMAR_NSE_RECOVERY_TIMEOUT", "90"))
            self.min_oos_accuracy = float(os.environ.get("STOMAR_MIN_OOS_ACCURACY", "0.50"))
            self.min_oos_sharpe = float(os.environ.get("STOMAR_MIN_OOS_SHARPE", "0.0"))
            self.max_drawdown_threshold = float(os.environ.get("STOMAR_MAX_DRAWDOWN_THRESHOLD", "0.30"))
            # Data Retention
            self.cache_retention_days = int(os.environ.get("STOMAR_CACHE_RETENTION_DAYS", "7"))
            self.ledger_retention_days = int(os.environ.get("STOMAR_LEDGER_RETENTION_DAYS", "90"))
            self.monitoring_retention_days = int(os.environ.get("STOMAR_MONITORING_RETENTION_DAYS", "30"))
            self.feature_versions_keep = int(os.environ.get("STOMAR_FEATURE_VERSIONS_KEEP", "10"))
            self.pipeline_logs_retention_days = int(os.environ.get("STOMAR_PIPELINE_LOGS_RETENTION_DAYS", "14"))

    settings = _FallbackSettings()
