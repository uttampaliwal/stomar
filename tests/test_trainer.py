"""Tests for src/trainer.py — batch_train function."""

from unittest.mock import patch


from src.models.trainer import batch_train


class TestBatchTrain:
    def test_returns_summary_keys(self):
        with patch("src.models.trainer.train_multiple_stocks") as mock_train:
            mock_train.return_value = {
                "RELIANCE.NS": {"oos_accuracy": 0.55},
                "TCS.NS": {"error": "timeout"},
            }
            result = batch_train(["RELIANCE.NS", "TCS.NS"])
            assert "success" in result
            assert "failed" in result
            assert "skipped" in result
            assert "duration_seconds" in result
            assert "results" in result

    def test_separates_success_and_failure(self):
        with patch("src.models.trainer.train_multiple_stocks") as mock_train:
            mock_train.return_value = {
                "RELIANCE.NS": {"oos_accuracy": 0.55},
                "TCS.NS": {"error": "timeout"},
                "INFY.NS": {"skipped": True},
            }
            result = batch_train(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
            assert "RELIANCE.NS" in result["success"]
            assert "TCS.NS" in result["failed"]
            assert "INFY.NS" in result["skipped"]

    def test_default_uses_nse_stocks(self):
        with patch("src.models.trainer.train_multiple_stocks") as mock_train:
            mock_train.return_value = {}
            batch_train()
            call_args = mock_train.call_args
            assert call_args[0][1] is False  # force_retrain positional

    def test_force_retrain_passed(self):
        with patch("src.models.trainer.train_multiple_stocks") as mock_train:
            mock_train.return_value = {}
            batch_train(["RELIANCE.NS"], force_retrain=True)
            call_args = mock_train.call_args
            assert call_args[0][1] is True

    def test_duration_is_positive(self):
        with patch("src.models.trainer.train_multiple_stocks") as mock_train:
            mock_train.return_value = {}
            result = batch_train(["RELIANCE.NS"])
            assert result["duration_seconds"] >= 0

    def test_empty_results(self):
        with patch("src.models.trainer.train_multiple_stocks") as mock_train:
            mock_train.return_value = {}
            result = batch_train([])
            assert result["success"] == []
            assert result["failed"] == {}
            assert result["skipped"] == []
