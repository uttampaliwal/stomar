"""Automated retraining pipeline with validation gates.

Pipeline stages:
1. FETCH  — get latest price data
2. VALIDATE — check data quality
3. FEATURES — compute features with versioning
4. TRAIN — train all 5 models
5. EVALUATE — OOS performance checks
6. PROMOTE — deploy if passes gates

Usage:
    from src.core.pipeline import RetrainingPipeline, PipelineConfig
    pipeline = RetrainingPipeline()
    result = pipeline.run("RELIANCE.NS")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the retraining pipeline."""
    tickers: list = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    ])
    lookback_period: str = "3y"
    min_oos_accuracy: float = 0.50
    min_oos_sharpe: float = -1.0
    max_drawdown_threshold: float = 0.30
    epochs: int = 40


@dataclass
class PipelineResult:
    """Result of a single pipeline run."""
    ticker: str
    stage: str
    status: str  # "success", "failed", "rejected"
    message: str
    metrics: dict = field(default_factory=dict)
    model_path: Optional[str] = None
    feature_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RetrainingPipeline:
    """Automated retraining pipeline with validation gates."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.results: list[PipelineResult] = []

    def run(self, ticker: str) -> PipelineResult:
        """Run the full pipeline for a single ticker."""
        logger.info(f"Starting retraining pipeline for {ticker}")

        result = self._stage_fetch(ticker)
        if result.status != "success":
            self.results.append(result)
            return result

        result = self._stage_validate(ticker)
        if result.status != "success":
            self.results.append(result)
            return result

        result = self._stage_features(ticker)
        if result.status != "success":
            self.results.append(result)
            return result

        result = self._stage_train(ticker, result)
        if result.status != "success":
            self.results.append(result)
            return result

        result = self._stage_evaluate(ticker)
        if result.status != "success":
            self.results.append(result)
            return result

        result = self._stage_promote(ticker, result)
        self.results.append(result)
        return result

    def _stage_fetch(self, ticker: str) -> PipelineResult:
        """Stage 1: Fetch latest price data."""
        try:
            from src.data.data_fetcher import fetch_stock_data
            df = fetch_stock_data(
                ticker,
                period=self.config.lookback_period,
                force_refresh=True,
            )
            return PipelineResult(
                ticker=ticker, stage="fetch", status="success",
                message=f"Fetched {len(df)} rows",
                metrics={"rows": len(df)},
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="fetch", status="failed",
                message=f"Fetch failed: {e}",
            )

    def _stage_validate(self, ticker: str) -> PipelineResult:
        """Stage 2: Validate data quality."""
        try:
            from src.data.data_fetcher import fetch_stock_data
            from src.data.data_validation import validate_data

            df = fetch_stock_data(ticker, period=self.config.lookback_period)
            validation = validate_data(df, ticker)

            if not validation["passed"]:
                return PipelineResult(
                    ticker=ticker, stage="validate", status="failed",
                    message=f"Validation errors: {validation['errors']}",
                    metrics=validation,
                )

            msg = f"Validation passed ({validation['data_points']} points)"
            if validation["warnings"]:
                msg += f", {len(validation['warnings'])} warnings"

            return PipelineResult(
                ticker=ticker, stage="validate", status="success",
                message=msg, metrics=validation,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="validate", status="failed",
                message=f"Validation error: {e}",
            )

    def _stage_features(self, ticker: str) -> PipelineResult:
        """Stage 3: Compute features with versioning."""
        try:
            from src.data.data_fetcher import fetch_stock_data
            from src.data.features import add_technical_indicators
            from src.data.feature_store import compute_feature_hash, register_feature_version

            df = fetch_stock_data(ticker, period=self.config.lookback_period)
            df = add_technical_indicators(df, ticker)

            skip = {"open", "high", "low", "close", "volume", "target", "target_direction"}
            feature_cols = sorted([c for c in df.columns if c not in skip])
            fhash = compute_feature_hash(feature_cols)

            register_feature_version(
                fhash, feature_cols,
                description=f"Pipeline run for {ticker}",
            )

            return PipelineResult(
                ticker=ticker, stage="features", status="success",
                message=f"{len(feature_cols)} features (hash: {fhash})",
                metrics={"n_features": len(feature_cols), "feature_hash": fhash},
                feature_hash=fhash,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="features", status="failed",
                message=f"Feature computation failed: {e}",
            )

    def _stage_train(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 4: Train all 5 models."""
        try:
            from src.models.trainer import train_for_ticker

            metrics = train_for_ticker(ticker, epochs=self.config.epochs, save=True)

            return PipelineResult(
                ticker=ticker, stage="train", status="success",
                message="Training complete",
                metrics=metrics.get("metrics", {}) if isinstance(metrics, dict) else {},
                feature_hash=prev.feature_hash,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="train", status="failed",
                message=f"Training failed: {e}",
            )

    def _stage_evaluate(self, ticker: str) -> PipelineResult:
        """Stage 5: Evaluate on OOS window with validation gates."""
        try:
            from src.trading.backtester import run_walk_forward_backtest
            from src.data.data_fetcher import fetch_stock_data
            from src.data.features import add_technical_indicators

            df = fetch_stock_data(ticker, period="5y", force_refresh=False)
            df_feat = add_technical_indicators(df, ticker=ticker)
            skip = {"open", "high", "low", "close", "volume", "target", "target_direction"}
            feature_cols = sorted([c for c in df_feat.columns if c not in skip])

            metrics, portfolio, test_results = run_walk_forward_backtest(
                ticker, df_feat, feature_cols,
            )

            oos_accuracy = metrics.get("ensemble_accuracy", 0)
            sharpe = metrics.get("simulated_sharpe", -999)
            max_dd = metrics.get("max_drawdown", 1.0)

            reasons = []
            if oos_accuracy < self.config.min_oos_accuracy:
                reasons.append(
                    f"OOS accuracy {oos_accuracy:.1%} < {self.config.min_oos_accuracy:.1%}"
                )
            if sharpe < self.config.min_oos_sharpe:
                reasons.append(
                    f"Sharpe {sharpe:.2f} < {self.config.min_oos_sharpe}"
                )
            if max_dd > self.config.max_drawdown_threshold:
                reasons.append(
                    f"Max DD {max_dd:.1%} > {self.config.max_drawdown_threshold:.1%}"
                )

            if reasons:
                return PipelineResult(
                    ticker=ticker, stage="evaluate", status="rejected",
                    message=f"Rejected: {'; '.join(reasons)}",
                    metrics=metrics,
                )

            return PipelineResult(
                ticker=ticker, stage="evaluate", status="success",
                message=f"Passed: accuracy={oos_accuracy:.1%}, sharpe={sharpe:.2f}",
                metrics=metrics,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="evaluate", status="failed",
                message=f"Evaluation failed: {e}",
            )

    def _stage_promote(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 6: Promote model to production."""
        try:
            from src.models.model import promote_model

            promote_model(ticker, prev.feature_hash)

            return PipelineResult(
                ticker=ticker, stage="promote", status="success",
                message="Model promoted to production",
                metrics=prev.metrics,
                feature_hash=prev.feature_hash,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="promote", status="failed",
                message=f"Promotion failed: {e}",
                metrics=prev.metrics,
            )

    def run_all(self) -> list[PipelineResult]:
        """Run pipeline for all configured tickers."""
        results = []
        for ticker in self.config.tickers:
            result = self.run(ticker)
            results.append(result)
            logger.info(f"{ticker}: {result.status} — {result.message}")
        return results

    def get_summary(self) -> dict:
        """Get summary of all pipeline runs."""
        return {
            "total": len(self.results),
            "success": sum(1 for r in self.results if r.status == "success"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "rejected": sum(1 for r in self.results if r.status == "rejected"),
            "results": [
                {
                    "ticker": r.ticker,
                    "stage": r.stage,
                    "status": r.status,
                    "message": r.message,
                }
                for r in self.results
            ],
        }
