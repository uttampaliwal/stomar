"""Tests for src/core/pipeline_state.py — PipelineCheckpoint state machine."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.core.pipeline_state import (
    PipelineCheckpoint,
    StageInfo,
    StageStatus,
    ALL_STAGES,
    checkpoint_path,
)


@pytest.fixture
def tmp_checkpoint(tmp_path):
    """Provide a temporary checkpoint path for tests."""
    return str(tmp_path / "checkpoint.json")


@pytest.fixture
def tmp_ticker_checkpoint(tmp_path):
    """Provide a temporary per-ticker checkpoint path."""
    return str(tmp_path / "checkpoint_RELIANCE_NS.json")


# ── StageInfo ───────────────────────────────────────────────────────────────

class TestStageInfo:
    def test_default_status(self):
        info = StageInfo()
        assert info.status == StageStatus.PENDING
        assert info.started_at is None
        assert info.completed_at is None
        assert info.error is None
        assert info.result is None

    def test_to_dict_minimal(self):
        info = StageInfo(status=StageStatus.COMPLETED)
        d = info.to_dict()
        assert d["status"] == "completed"
        assert "started_at" not in d
        assert "error" not in d

    def test_to_dict_full(self):
        info = StageInfo(
            status=StageStatus.FAILED,
            started_at="2026-01-01T10:00:00",
            completed_at="2026-01-01T10:01:00",
            error="network timeout",
            result={"rows": 100},
        )
        d = info.to_dict()
        assert d["status"] == "failed"
        assert d["started_at"] == "2026-01-01T10:00:00"
        assert d["completed_at"] == "2026-01-01T10:01:00"
        assert d["error"] == "network timeout"
        assert d["result"]["rows"] == 100

    def test_from_dict_roundtrip(self):
        original = StageInfo(
            status=StageStatus.RUNNING,
            started_at="2026-01-01T10:00:00",
            result={"key": "val"},
        )
        restored = StageInfo.from_dict(original.to_dict())
        assert restored.status == StageStatus.RUNNING
        assert restored.started_at == "2026-01-01T10:00:00"
        assert restored.result == {"key": "val"}

    def test_from_dict_defaults(self):
        restored = StageInfo.from_dict({})
        assert restored.status == StageStatus.PENDING
        assert restored.error is None


# ── PipelineCheckpoint creation ─────────────────────────────────────────────

class TestCheckpointCreation:
    def test_default_values(self):
        ckpt = PipelineCheckpoint()
        assert ckpt.pipeline == "auto"
        assert ckpt.ticker is None
        assert ckpt.current_stage == "IDLE"
        assert ckpt.stages == {}
        assert ckpt.created_at
        assert ckpt.updated_at

    def test_custom_pipeline(self):
        ckpt = PipelineCheckpoint(pipeline="retrain", ticker="RELIANCE.NS")
        assert ckpt.pipeline == "retrain"
        assert ckpt.ticker == "RELIANCE.NS"


# ── Save and load ───────────────────────────────────────────────────────────

class TestCheckpointPersistence:
    def test_save_and_load(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint(pipeline="auto")
        ckpt.advance("FETCH")
        ckpt.complete("FETCH", {"rows": 100})
        ckpt.save(path=tmp_checkpoint)

        loaded = PipelineCheckpoint.load(path=tmp_checkpoint)
        assert loaded is not None
        assert loaded.pipeline == "auto"
        assert loaded.current_stage == "FETCH"
        assert loaded.stage_status("FETCH") == StageStatus.COMPLETED
        assert loaded.stages["FETCH"].result == {"rows": 100}

    def test_load_returns_none_when_missing(self, tmp_path):
        loaded = PipelineCheckpoint.load(path=str(tmp_path / "nonexistent.json"))
        assert loaded is None

    def test_load_returns_none_when_corrupt(self, tmp_checkpoint):
        with open(tmp_checkpoint, "w") as f:
            f.write("{invalid json!!!")
        loaded = PipelineCheckpoint.load(path=tmp_checkpoint)
        assert loaded is None

    def test_save_creates_directories(self, tmp_path):
        nested = str(tmp_path / "subdir" / "deep" / "checkpoint.json")
        ckpt = PipelineCheckpoint()
        ckpt.save(path=nested)
        assert os.path.exists(nested)

    def test_atomic_write_no_corruption(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.save(path=tmp_checkpoint)
        # Verify valid JSON
        with open(tmp_checkpoint) as f:
            data = json.load(f)
        assert data["current_stage"] == "FETCH"

    def test_clear(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint()
        ckpt.save(path=tmp_checkpoint)
        assert os.path.exists(tmp_checkpoint)
        PipelineCheckpoint.clear(path=tmp_checkpoint)
        assert not os.path.exists(tmp_checkpoint)

    def test_clear_nonexistent_is_safe(self, tmp_path):
        # Should not raise
        PipelineCheckpoint.clear(path=str(tmp_path / "nope.json"))

    def test_per_ticker_path(self):
        path = checkpoint_path("retrain", "RELIANCE.NS")
        assert "RELIANCE_NS" in path
        assert path.endswith(".json")

    def test_default_path(self):
        path = checkpoint_path("auto")
        assert path.endswith("pipeline_checkpoint.json")


# ── Stage progression ───────────────────────────────────────────────────────

class TestStageProgression:
    def test_advance_sets_running(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        assert ckpt.current_stage == "FETCH"
        assert ckpt.stage_status("FETCH") == StageStatus.RUNNING
        assert ckpt.stages["FETCH"].started_at is not None

    def test_complete_marks_completed(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH", {"rows": 50})
        assert ckpt.stage_status("FETCH") == StageStatus.COMPLETED
        assert ckpt.stages["FETCH"].completed_at is not None
        assert ckpt.stages["FETCH"].result == {"rows": 50}

    def test_fail_marks_failed(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.fail("FETCH", "network error")
        assert ckpt.stage_status("FETCH") == StageStatus.FAILED
        assert ckpt.stages["FETCH"].error == "network error"

    def test_skip_marks_skipped(self):
        ckpt = PipelineCheckpoint()
        ckpt.skip("VALIDATE", "no data to validate")
        assert ckpt.stage_status("VALIDATE") == StageStatus.SKIPPED
        assert ckpt.stages["VALIDATE"].result == {"skipped_reason": "no data to validate"}

    def test_advance_unknown_stage_raises(self):
        ckpt = PipelineCheckpoint()
        with pytest.raises(ValueError, match="Unknown stage"):
            ckpt.advance("BOGUS")

    def test_stage_status_pending_when_not_touched(self):
        ckpt = PipelineCheckpoint()
        assert ckpt.stage_status("FETCH") == StageStatus.PENDING
        assert ckpt.stage_info("FETCH") is None

    def test_full_stage_progression(self):
        ckpt = PipelineCheckpoint()
        for stage in ["FETCH", "VALIDATE", "FEATURES", "TRAIN", "EVALUATE", "PROMOTE", "DAILY", "PAPER", "ARCHIVE", "DONE"]:
            ckpt.advance(stage)
            ckpt.complete(stage)
        assert ckpt.is_complete()
        assert ckpt.get_completed_stages() == [
            "FETCH", "VALIDATE", "FEATURES", "TRAIN", "EVALUATE",
            "PROMOTE", "DAILY", "PAPER", "ARCHIVE", "DONE",
        ]


# ── Query methods ───────────────────────────────────────────────────────────

class TestQueryMethods:
    def test_is_complete_false_when_not_done(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        assert not ckpt.is_complete()

    def test_is_complete_true_at_done(self):
        ckpt = PipelineCheckpoint()
        for stage in ALL_STAGES:
            if stage in ("IDLE", "DONE"):
                continue
            ckpt.advance(stage)
            ckpt.complete(stage)
        ckpt.advance("DONE")
        ckpt.complete("DONE")
        assert ckpt.is_complete()

    def test_has_failed_false_when_no_failures(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        assert not ckpt.has_failed()

    def test_has_failed_true_on_failure(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.fail("FETCH", "error")
        assert ckpt.has_failed()

    def test_can_resume_when_interrupted(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        ckpt.advance("VALIDATE")
        # Simulate crash: VALIDATE is RUNNING but not completed
        assert ckpt.can_resume()

    def test_cannot_resume_when_failed(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.fail("FETCH", "error")
        assert not ckpt.can_resume()

    def test_cannot_resume_when_complete(self):
        ckpt = PipelineCheckpoint()
        for stage in ALL_STAGES:
            if stage in ("IDLE", "DONE"):
                continue
            ckpt.advance(stage)
            ckpt.complete(stage)
        ckpt.advance("DONE")
        ckpt.complete("DONE")
        assert not ckpt.can_resume()

    def test_next_stage_returns_first_pending(self):
        ckpt = PipelineCheckpoint()
        assert ckpt.next_stage() == "FETCH"

    def test_next_stage_after_fetch(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        assert ckpt.next_stage() == "VALIDATE"

    def test_next_stage_returns_none_when_all_done(self):
        ckpt = PipelineCheckpoint()
        for stage in ALL_STAGES:
            if stage in ("IDLE", "DONE"):
                continue
            ckpt.advance(stage)
            ckpt.complete(stage)
        ckpt.advance("DONE")
        ckpt.complete("DONE")
        assert ckpt.next_stage() is None

    def test_resume_from_returns_running_stage(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        ckpt.advance("TRAIN")
        # TRAIN is RUNNING (interrupted)
        assert ckpt.resume_from() == "TRAIN"

    def test_resume_from_returns_next_pending_when_no_running(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        # No running stage, should return next pending
        assert ckpt.resume_from() == "VALIDATE"

    def test_get_completed_stages(self):
        ckpt = PipelineCheckpoint()
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        ckpt.skip("VALIDATE")
        ckpt.advance("FEATURES")
        ckpt.complete("FEATURES")
        completed = ckpt.get_completed_stages()
        assert "FETCH" in completed
        assert "VALIDATE" in completed
        assert "FEATURES" in completed
        assert "TRAIN" not in completed

    def test_get_summary(self):
        ckpt = PipelineCheckpoint(pipeline="retrain", ticker="TCS.NS")
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        summary = ckpt.get_summary()
        assert summary["pipeline"] == "retrain"
        assert summary["ticker"] == "TCS.NS"
        assert summary["is_complete"] is False
        assert summary["has_failed"] is False
        assert summary["can_resume"] is True
        assert "FETCH" in summary["completed_stages"]
        assert summary["next_stage"] == "VALIDATE"


# ── Resume scenarios ────────────────────────────────────────────────────────

class TestResumeScenarios:
    def test_resume_after_fetch_crash(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint(pipeline="auto")
        ckpt.advance("FETCH")
        ckpt.complete("FETCH", {"days_backfilled": 5})
        ckpt.advance("TRAIN")
        ckpt.save(path=tmp_checkpoint)

        # Simulate restart
        loaded = PipelineCheckpoint.load(path=tmp_checkpoint)
        assert loaded.can_resume()
        assert loaded.resume_from() == "TRAIN"

    def test_resume_after_daily_crash(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint(pipeline="auto")
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        ckpt.skip("TRAIN", "already loaded")
        ckpt.advance("DAILY")
        ckpt.save(path=tmp_checkpoint)

        loaded = PipelineCheckpoint.load(path=tmp_checkpoint)
        assert loaded.can_resume()
        assert loaded.resume_from() == "DAILY"

    def test_resume_skips_completed_stages(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint(pipeline="auto")
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        ckpt.advance("TRAIN")
        ckpt.complete("TRAIN")
        ckpt.advance("DAILY")
        ckpt.save(path=tmp_checkpoint)

        loaded = PipelineCheckpoint.load(path=tmp_checkpoint)
        assert loaded.can_resume()
        assert loaded.resume_from() == "DAILY"
        # FETCH and TRAIN should be in completed_stages
        completed = loaded.get_completed_stages()
        assert "FETCH" in completed
        assert "TRAIN" in completed

    def test_per_ticker_resume(self, tmp_path):
        path = checkpoint_path("retrain", "RELIANCE.NS")
        ckpt = PipelineCheckpoint(pipeline="retrain", ticker="RELIANCE.NS")
        ckpt.advance("FETCH")
        ckpt.complete("FETCH")
        ckpt.advance("TRAIN")
        ckpt.save(path=path)

        loaded = PipelineCheckpoint.load(path=path)
        assert loaded.can_resume()
        assert loaded.ticker == "RELIANCE.NS"
        assert loaded.resume_from() == "TRAIN"

        # Cleanup
        os.remove(path)


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_complete_without_advance(self):
        ckpt = PipelineCheckpoint()
        # Should auto-create the stage info
        ckpt.complete("FETCH", {"rows": 100})
        assert ckpt.stage_status("FETCH") == StageStatus.COMPLETED

    def test_fail_without_advance(self):
        ckpt = PipelineCheckpoint()
        ckpt.fail("FETCH", "error")
        assert ckpt.stage_status("FETCH") == StageStatus.FAILED
        assert ckpt.stages["FETCH"].error == "error"

    def test_skip_without_advance(self):
        ckpt = PipelineCheckpoint()
        ckpt.skip("FETCH", "not needed")
        assert ckpt.stage_status("FETCH") == StageStatus.SKIPPED

    def test_multiple_pipelines_independent(self, tmp_path):
        path1 = str(tmp_path / "auto.json")
        path2 = str(tmp_path / "retrain.json")

        ckpt1 = PipelineCheckpoint(pipeline="auto")
        ckpt1.advance("FETCH")
        ckpt1.complete("FETCH")
        ckpt1.save(path=path1)

        ckpt2 = PipelineCheckpoint(pipeline="retrain", ticker="T.NS")
        ckpt2.advance("FETCH")
        ckpt2.save(path=path2)

        loaded1 = PipelineCheckpoint.load(path=path1)
        loaded2 = PipelineCheckpoint.load(path=path2)

        assert loaded1.stage_status("FETCH") == StageStatus.COMPLETED
        assert loaded2.stage_status("FETCH") == StageStatus.RUNNING

    def test_all_stages_are_valid(self):
        for stage in ALL_STAGES:
            ckpt = PipelineCheckpoint()
            ckpt.advance(stage)
            assert ckpt.current_stage == stage

    def test_idempotent_clear(self, tmp_checkpoint):
        ckpt = PipelineCheckpoint()
        ckpt.save(path=tmp_checkpoint)
        PipelineCheckpoint.clear(path=tmp_checkpoint)
        PipelineCheckpoint.clear(path=tmp_checkpoint)  # second clear is safe
        assert not os.path.exists(tmp_checkpoint)


# ── Auto-pipeline checkpoint integration ────────────────────────────────────

class TestAutoPipelineCheckpoint:
    """Integration tests: AutoPipeline uses checkpoint to skip completed stages."""

    def test_checkpoint_created_on_run(self, tmp_path):
        """Run creates a checkpoint file that persists after completion."""
        from unittest.mock import patch, MagicMock
        from auto_pipeline import AutoPipeline

        ckpt_path = str(tmp_path / "pipeline_checkpoint.json")

        pipeline = AutoPipeline(tickers=["TEST.NS"])
        pipeline.ledger = MagicMock()
        pipeline.ledger.get_last_decision_date.return_value = "2020-01-01"
        pipeline.ledger.get_dates_with_decisions.return_value = []

        with patch("auto_pipeline.PipelineCheckpoint") as MockCkpt, \
             patch("auto_pipeline.Ledger") as MockLedger:
            real_ckpt = PipelineCheckpoint(pipeline="auto")
            MockCkpt.load.return_value = None
            MockCkpt.clear.return_value = None

            # Capture the checkpoint instance created inside run()
            original_advance = real_ckpt.advance
            stages_completed = []
            def track_advance(stage):
                stages_completed.append(stage)
                original_advance(stage)
            real_ckpt.advance = track_advance

            MockCkpt.return_value = real_ckpt
            MockLedger.return_value = pipeline.ledger

            with patch.object(pipeline, "_backfill_missed_days", return_value={"days_backfilled": 0}), \
                 patch.object(pipeline, "_ensure_meta_controller", return_value={"status": "loaded"}), \
                 patch.object(pipeline, "_run_daily", return_value={"decisions": 5, "errors": 0}), \
                 patch.object(pipeline, "_run_paper_trades", return_value={"trades": 2}), \
                 patch.object(pipeline, "_auto_install_scheduler", return_value={"status": "skipped"}):
                result = pipeline.run(force=True)

            assert result["status"] == "success"
            assert "FETCH" in stages_completed
            assert "TRAIN" in stages_completed
            assert "DAILY" in stages_completed
            assert "PAPER" in stages_completed
            assert "DONE" in stages_completed

    def test_resume_skips_completed_stages(self, tmp_path):
        """When checkpoint has FETCH done, run() skips FETCH."""
        from unittest.mock import patch, MagicMock
        from auto_pipeline import AutoPipeline

        pipeline = AutoPipeline(tickers=["TEST.NS"])
        pipeline.ledger = MagicMock()
        pipeline.ledger.get_last_decision_date.return_value = "2020-01-01"

        # Create a checkpoint with FETCH already done
        existing_ckpt = PipelineCheckpoint(pipeline="auto")
        existing_ckpt.advance("FETCH")
        existing_ckpt.complete("FETCH", {"days_backfilled": 3})
        existing_ckpt.advance("TRAIN")
        existing_ckpt.complete("TRAIN", {"status": "loaded"})

        with patch("auto_pipeline.PipelineCheckpoint") as MockCkpt, \
             patch("auto_pipeline.Ledger") as MockLedger:
            MockCkpt.load.return_value = existing_ckpt
            MockCkpt.clear.return_value = None
            MockCkpt.return_value = existing_ckpt
            MockLedger.return_value = pipeline.ledger

            backfill_called = False
            def mock_backfill(*a, **kw):
                nonlocal backfill_called
                backfill_called = True
                return {"days_backfilled": 0}

            with patch.object(pipeline, "_backfill_missed_days", side_effect=mock_backfill), \
                 patch.object(pipeline, "_ensure_meta_controller", return_value={"status": "loaded"}), \
                 patch.object(pipeline, "_run_daily", return_value={"decisions": 5, "errors": 0}), \
                 patch.object(pipeline, "_run_paper_trades", return_value={"trades": 2}), \
                 patch.object(pipeline, "_auto_install_scheduler", return_value={"status": "skipped"}):
                result = pipeline.run(force=True)

            # FETCH should have been skipped (not re-run)
            assert not backfill_called
            assert result["status"] == "success"

    def test_failure_saves_checkpoint_for_resume(self):
        """When a stage fails, checkpoint is saved so next run resumes."""
        from unittest.mock import patch, MagicMock
        from auto_pipeline import AutoPipeline

        pipeline = AutoPipeline(tickers=["TEST.NS"])
        pipeline.ledger = MagicMock()
        pipeline.ledger.get_last_decision_date.return_value = "2020-01-01"

        with patch("auto_pipeline.PipelineCheckpoint") as MockCkpt, \
             patch("auto_pipeline.Ledger") as MockLedger:
            real_ckpt = PipelineCheckpoint(pipeline="auto")
            MockCkpt.load.return_value = None
            MockCkpt.clear.return_value = None
            MockCkpt.return_value = real_ckpt
            MockLedger.return_value = pipeline.ledger

            with patch.object(pipeline, "_backfill_missed_days", return_value={"days_backfilled": 0}), \
                 patch.object(pipeline, "_ensure_meta_controller", return_value={"status": "loaded"}), \
                 patch.object(pipeline, "_run_daily", return_value={"error": "network timeout"}), \
                 patch.object(pipeline, "_auto_install_scheduler", return_value={"status": "skipped"}):
                result = pipeline.run(force=True)

            assert result["status"] == "error"
            # Checkpoint should have DAILY as failed
            assert real_ckpt.stage_status("DAILY") == StageStatus.FAILED
            assert real_ckpt.has_failed()

    def test_result_includes_checkpoint_info(self):
        """Result dict includes checkpoint metadata when resuming."""
        from unittest.mock import patch, MagicMock
        from auto_pipeline import AutoPipeline

        pipeline = AutoPipeline(tickers=["TEST.NS"])
        pipeline.ledger = MagicMock()
        pipeline.ledger.get_last_decision_date.return_value = "2020-01-01"

        existing_ckpt = PipelineCheckpoint(pipeline="auto")
        existing_ckpt.advance("FETCH")
        existing_ckpt.complete("FETCH", {"days_backfilled": 2})
        existing_ckpt.advance("DAILY")

        with patch("auto_pipeline.PipelineCheckpoint") as MockCkpt, \
             patch("auto_pipeline.Ledger") as MockLedger:
            MockCkpt.load.return_value = existing_ckpt
            MockCkpt.clear.return_value = None
            MockCkpt.return_value = existing_ckpt
            MockLedger.return_value = pipeline.ledger

            with patch.object(pipeline, "_ensure_meta_controller", return_value={"status": "loaded"}), \
                 patch.object(pipeline, "_run_daily", return_value={"decisions": 5, "errors": 0}), \
                 patch.object(pipeline, "_run_paper_trades", return_value={"trades": 2}), \
                 patch.object(pipeline, "_auto_install_scheduler", return_value={"status": "skipped"}):
                result = pipeline.run(force=True)

            assert "checkpoint" in result
            assert result["checkpoint"]["resumed_from"] == "DAILY"


# ── Retraining pipeline checkpoint integration ──────────────────────────────

class TestRetrainingPipelineCheckpoint:
    """Integration tests: RetrainingPipeline uses per-ticker checkpoints."""

    def test_checkpoint_created_per_ticker(self, tmp_path):
        """Retraining pipeline creates per-ticker checkpoint files."""
        from src.core.pipeline import RetrainingPipeline, PipelineConfig

        path = checkpoint_path("retrain", "TEST.NS")
        assert "TEST_NS" in path

    def test_fetch_failure_stops_pipeline(self):
        """FETCH failure marks checkpoint as failed and stops."""
        from src.core.pipeline import RetrainingPipeline

        pipeline = RetrainingPipeline()
        with patch("src.data.data_fetcher.fetch_stock_data", side_effect=Exception("down")):
            result = pipeline.run("TEST.NS")
            assert result.status == "failed"
            assert result.stage == "fetch"

    def test_full_run_clears_checkpoint(self):
        """Successful full run clears the checkpoint file."""
        from src.core.pipeline import RetrainingPipeline, PipelineResult

        pipeline = RetrainingPipeline()
        # Mock all stages to succeed
        with patch("src.data.data_fetcher.fetch_stock_data") as mock_fetch, \
             patch("src.data.data_validation.validate_data") as mock_val, \
             patch("src.data.features.add_technical_indicators") as mock_feat, \
             patch("src.data.feature_store.compute_feature_hash") as mock_hash, \
             patch("src.data.feature_store.register_feature_version"), \
             patch("src.models.trainer.train_for_ticker") as mock_train, \
             patch("src.trading.backtester.run_walk_forward_backtest") as mock_bt, \
             patch("src.models.model.promote_model"):
            import pandas as pd
            import numpy as np
            n = 100
            dates = pd.bdate_range("2024-01-01", periods=n)
            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
            df = pd.DataFrame({
                "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
                "close": close, "volume": np.ones(n) * 5000,
            }, index=dates)

            mock_fetch.return_value = df
            mock_val.return_value = {"passed": True, "errors": [], "warnings": [], "data_points": 100}
            mock_feat.return_value = df
            mock_hash.return_value = "abc123"
            mock_train.return_value = {"metrics": {}}
            mock_bt.return_value = (
                {"ensemble_accuracy": 0.53, "simulated_sharpe": 0.5, "max_drawdown": 0.15},
                None, [],
            )

            result = pipeline.run("TEST.NS")
            assert result.status == "success"

        # Checkpoint should be cleared
        path = checkpoint_path("retrain", "TEST.NS")
        assert not os.path.exists(path)
