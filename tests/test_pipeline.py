"""Tests for src/pipeline.py."""

import numpy as np
import pandas as pd
from unittest.mock import patch

from src.core.pipeline import RetrainingPipeline, PipelineConfig, PipelineResult
from src.data.data_fetcher import NSE_STOCKS


def _mock_df(n=100):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.ones(n) * 5000,
    }, index=dates)


# ── PipelineConfig ──

class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert len(config.tickers) == len(NSE_STOCKS) > 0
        assert config.lookback_period == "3y"
        assert config.min_oos_accuracy == 0.50

    def test_custom(self):
        config = PipelineConfig(tickers=["TEST.NS"], min_oos_accuracy=0.55)
        assert config.tickers == ["TEST.NS"]
        assert config.min_oos_accuracy == 0.55


# ── PipelineResult ──

class TestPipelineResult:
    def test_success_result(self):
        r = PipelineResult(ticker="TEST.NS", stage="fetch", status="success", message="ok")
        assert r.status == "success"
        assert r.ticker == "TEST.NS"
        assert r.timestamp

    def test_default_metrics(self):
        r = PipelineResult(ticker="T", stage="s", status="ok", message="m")
        assert r.metrics == {}
        assert r.feature_hash is None


# ── Pipeline stages (mocked at source) ──

class TestPipelineStages:
    def test_fetch_success(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock:
            mock.return_value = _mock_df(50)
            pipeline = RetrainingPipeline()
            result = pipeline._stage_fetch("TEST.NS")
            assert result.status == "success"
            assert "50" in result.message

    def test_fetch_failure(self):
        with patch("src.data.data_fetcher.fetch_stock_data", side_effect=Exception("network")):
            pipeline = RetrainingPipeline()
            result = pipeline._stage_fetch("TEST.NS")
            assert result.status == "failed"
            assert "network" in result.message

    def test_validate_pass(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.data_validation.validate_data") as mock_val:
            mock_fetch.return_value = _mock_df()
            mock_val.return_value = {
                "passed": True, "errors": [], "warnings": [],
                "data_points": 100,
            }
            pipeline = RetrainingPipeline()
            result = pipeline._stage_validate("TEST.NS")
            assert result.status == "success"

    def test_validate_failure(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.data_validation.validate_data") as mock_val:
            mock_fetch.return_value = _mock_df()
            mock_val.return_value = {
                "passed": False, "errors": ["bad prices"], "warnings": [],
                "data_points": 100,
            }
            pipeline = RetrainingPipeline()
            result = pipeline._stage_validate("TEST.NS")
            assert result.status == "failed"

    def test_features_computes_hash(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.data.feature_store.compute_feature_hash") as mock_hash, \
             patch("src.data.feature_store.register_feature_version") as mock_reg:
            mock_fetch.return_value = _mock_df()
            mock_feat.return_value = _mock_df()
            mock_hash.return_value = "abc123def456"
            mock_reg.return_value = {}

            pipeline = RetrainingPipeline()
            result = pipeline._stage_features("TEST.NS")
            assert result.status == "success"
            assert result.feature_hash == "abc123def456"

    def test_evaluate_rejects_low_accuracy(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.trading.backtester.run_walk_forward_backtest") as mock_bt:
            mock_fetch.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_feat.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_bt.return_value = (
                {"ensemble_accuracy": 0.48, "simulated_sharpe": -0.5, "simulated_max_drawdown": 0.15},
                None, [],
            )
            pipeline = RetrainingPipeline(PipelineConfig(min_oos_accuracy=0.55))
            result = pipeline._stage_evaluate("TEST.NS")
            assert result.status == "rejected"
            assert "accuracy" in result.message.lower()

    def test_evaluate_rejects_low_sharpe(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.trading.backtester.run_walk_forward_backtest") as mock_bt:
            mock_fetch.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_feat.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_bt.return_value = (
                {"ensemble_accuracy": 0.53, "simulated_sharpe": -1.0, "simulated_max_drawdown": 0.15},
                None, [],
            )
            pipeline = RetrainingPipeline(PipelineConfig(min_oos_sharpe=-0.5))
            result = pipeline._stage_evaluate("TEST.NS")
            assert result.status == "rejected"
            assert "sharpe" in result.message.lower()

    def test_evaluate_rejects_high_drawdown(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.trading.backtester.run_walk_forward_backtest") as mock_bt:
            mock_fetch.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_feat.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_bt.return_value = (
                {"ensemble_accuracy": 0.53, "simulated_sharpe": 0.5, "simulated_max_drawdown": 0.35},
                None, [],
            )
            pipeline = RetrainingPipeline(PipelineConfig(max_drawdown_threshold=0.20))
            result = pipeline._stage_evaluate("TEST.NS")
            assert result.status == "rejected"
            assert "dd" in result.message.lower()

    def test_evaluate_passes_good_model(self):
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.trading.backtester.run_walk_forward_backtest") as mock_bt:
            mock_fetch.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_feat.return_value = pd.DataFrame({"close": [100]*100, "open": [100]*100, "high": [101]*100, "low": [99]*100, "volume": [1000]*100})
            mock_bt.return_value = (
                {"ensemble_accuracy": 0.53, "simulated_sharpe": 0.5, "simulated_max_drawdown": 0.15},
                None, [],
            )
            pipeline = RetrainingPipeline()
            result = pipeline._stage_evaluate("TEST.NS")
            assert result.status == "success"

    def test_promote_success(self):
        with patch("src.models.model.promote_model") as mock_promote:
            pipeline = RetrainingPipeline()
            prev = PipelineResult(
                ticker="T", stage="eval", status="success", message="ok",
                feature_hash="abc123",
            )
            result = pipeline._stage_promote("TEST.NS", prev)
            assert result.status == "success"

    def test_promote_failure(self):
        with patch("src.models.model.promote_model", side_effect=Exception("disk full")):
            pipeline = RetrainingPipeline()
            prev = PipelineResult(
                ticker="T", stage="eval", status="success", message="ok",
            )
            result = pipeline._stage_promote("TEST.NS", prev)
            assert result.status == "failed"


# ── Full run (mocked) ──

class TestPipelineRun:
    def test_run_stops_on_fetch_failure(self):
        pipeline = RetrainingPipeline()
        with patch("src.data.data_fetcher.fetch_stock_data", side_effect=Exception("down")):
            result = pipeline.run("TEST.NS")
            assert result.status == "failed"
            assert result.stage == "fetch"
            assert len(pipeline.results) == 1

    def test_run_stops_on_validate_failure(self):
        pipeline = RetrainingPipeline()
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.data_validation.validate_data") as mock_val:
            mock_fetch.return_value = _mock_df()
            mock_val.return_value = {
                "passed": False, "errors": ["bad"], "warnings": [], "data_points": 0,
            }
            result = pipeline.run("TEST.NS")
            assert result.status == "failed"
            assert result.stage == "validate"

    def test_run_stops_on_evaluate_rejection(self):
        pipeline = RetrainingPipeline(PipelineConfig(min_oos_accuracy=0.99))
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.data_validation.validate_data") as mock_val, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.data.feature_store.compute_feature_hash") as mock_hash, \
             patch("src.data.feature_store.register_feature_version"), \
             patch("src.models.trainer.train_for_ticker") as mock_train, \
             patch("src.data.data_fetcher.fetch_stock_data") as mock_pipe_fetch, \
             patch("src.data.features.add_technical_indicators") as mock_pipe_feat, \
             patch("src.trading.backtester.run_walk_forward_backtest") as mock_bt:
            mock_fetch.return_value = _mock_df()
            mock_val.return_value = {"passed": True, "errors": [], "warnings": [], "data_points": 100}
            mock_feat.return_value = _mock_df()
            mock_hash.return_value = "h1"
            mock_train.return_value = {"metrics": {}}
            mock_pipe_fetch.return_value = _mock_df()
            mock_pipe_feat.return_value = _mock_df()
            mock_bt.return_value = (
                {"ensemble_accuracy": 0.53, "simulated_sharpe": 0.5, "simulated_max_drawdown": 0.15},
                None, [],
            )
            result = pipeline.run("TEST.NS")
            assert result.status == "rejected"

    def test_get_summary(self):
        pipeline = RetrainingPipeline()
        pipeline.results = [
            PipelineResult("A", "p", "success", "ok"),
            PipelineResult("B", "f", "failed", "err"),
            PipelineResult("C", "p", "rejected", "bad"),
        ]
        s = pipeline.get_summary()
        assert s["total"] == 3
        assert s["success"] == 1
        assert s["failed"] == 1
        assert s["rejected"] == 1
