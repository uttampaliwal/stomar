"""Tests for auto-pipeline retry logic (P1.2) and ops hygiene (P5.3/P5.4)."""

import json
import os
import time

import pytest

from auto_pipeline import AutoPipeline


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No real waits in tests."""
    monkeypatch.setattr("auto_pipeline.RETRY_DELAYS", (0, 0))


class TestRetryLogic:
    def test_succeeds_on_first_attempt(self):
        p = AutoPipeline(tickers=["TEST.NS"])
        calls = []
        result = p._run_stage_with_retries("TEST", lambda: calls.append(1) or {"ok": True})
        assert result == {"ok": True}
        assert len(calls) == 1

    def test_retries_until_success(self, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        calls = []
        def fn():
            calls.append(1)
            if len(calls) < 3:
                return {"error": "transient"}
            return {"ok": True}
        result = p._run_stage_with_retries("TEST", fn)
        assert result == {"ok": True}
        assert len(calls) == 3

    def test_exhausts_attempts_and_records_failure(self, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        recorded = []
        monkeypatch.setattr(p, "_record_failure",
                            lambda stage, error: recorded.append((stage, error)))
        calls = []
        result = p._run_stage_with_retries("TEST", lambda: calls.append(1) or {"error": "always"})
        assert result == {"error": "always"}
        assert len(calls) == 3  # RETRY_DELAYS=(0,0) → 3 attempts
        assert recorded and recorded[0][0] == "TEST"

    def test_retries_on_exception(self, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        calls = []
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("boom")
            return {"ok": True}
        result = p._run_stage_with_retries("TEST", fn)
        assert result == {"ok": True}
        assert len(calls) == 2

    def test_shutdown_aborts_retries(self, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        p._shutdown_requested = True
        result = p._run_stage_with_retries("TEST", lambda: {"error": "x"})
        assert result == {"error": "interrupted by shutdown"}


class TestBackupLedger:
    def test_backup_only_on_sunday(self, tmp_path, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        from datetime import datetime
        monkeypatch.setattr("auto_pipeline.datetime", type(
            "FakeDT", (), {
                "now": staticmethod(lambda: datetime(2026, 8, 3, 12, 0)),  # Monday
                "strftime": staticmethod(lambda *a, **k: datetime(2026, 8, 3, 12, 0).strftime(*a, **k)),
            }
        ))
        ledger_path = tmp_path / "data" / "stomar.db"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text("fake db")
        monkeypatch.setattr("src.core.constants.DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr("src.core.constants.LEDGER_DB", str(ledger_path))
        result = p._backup_ledger()
        assert result["backed_up"] is False  # not Sunday

    def test_backup_creates_copy_on_sunday(self, tmp_path, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        from datetime import datetime
        monkeypatch.setattr("auto_pipeline.datetime", type(
            "FakeDT", (), {
                "now": staticmethod(lambda: datetime(2026, 8, 2, 12, 0)),  # Sunday
                "strftime": staticmethod(lambda *a, **k: datetime(2026, 8, 2, 12, 0).strftime(*a, **k)),
            }
        ))
        ledger_path = tmp_path / "data" / "stomar.db"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text("fake db")
        monkeypatch.setattr("src.core.constants.DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr("src.core.constants.LEDGER_DB", str(ledger_path))
        result = p._backup_ledger()
        assert result["backed_up"] is True
        assert os.path.exists(result["path"])
        assert os.path.basename(result["path"]) == "stomar_20260802.db"


class TestHealthSweep:
    def test_writes_daily_health_json(self, tmp_path, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        p._status = "completed"
        p._last_run_date = "2026-08-03"
        monkeypatch.setattr("src.core.constants.DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr("auto_pipeline.MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setattr("src.core.constants.PAPER_STATE_PATH", str(tmp_path / "paper_state.json"))
        monkeypatch.setattr("auto_pipeline.MONITORING_DIR", str(tmp_path / "monitoring"))

        health = p._run_health_sweep()

        path = tmp_path / "monitoring" / "daily_health.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["status"] == "completed"
        assert data["last_run_date"] == "2026-08-03"
        assert data["kill_switch_active"] is False
        assert "TEST.NS" in data["data_freshness_days"]

    def test_model_age_read_from_manifest(self, tmp_path, monkeypatch):
        # Models are per-artifact files ({ticker}_lstm.pt, {ticker}_xgb.pkl)
        # gated by {ticker}_manifest.json — the old {ticker}_models.pkl bundle
        # no longer exists and must not gate freshness.
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        manifest = models_dir / "TEST_NS_manifest.json"
        manifest.write_text('{"manifest_version": 1, "files": {}}')
        old = time.time() - 60 * 60 * 24 * 5  # 5 days old
        os.utime(manifest, (old, old))

        p = AutoPipeline(tickers=["TEST.NS"])
        monkeypatch.setattr("src.core.constants.DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr("auto_pipeline.MODELS_DIR", str(models_dir))
        monkeypatch.setattr("src.core.constants.PAPER_STATE_PATH", str(tmp_path / "paper_state.json"))
        monkeypatch.setattr("auto_pipeline.MONITORING_DIR", str(tmp_path / "monitoring"))

        health = p._run_health_sweep()
        assert health["model_age_days"]["TEST.NS"] == 5

    def test_model_missing_reports_none(self, tmp_path, monkeypatch):
        p = AutoPipeline(tickers=["TEST.NS"])
        monkeypatch.setattr("src.core.constants.DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr("auto_pipeline.MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setattr("src.core.constants.PAPER_STATE_PATH", str(tmp_path / "paper_state.json"))
        monkeypatch.setattr("auto_pipeline.MONITORING_DIR", str(tmp_path / "monitoring"))

        health = p._run_health_sweep()
        assert health["model_age_days"]["TEST.NS"] is None


class _FakeRiskController:
    def reset_daily(self):
        pass


class _FakePaperTrader:
    def __init__(self, load_ok: bool = True):
        self._load_ok = load_ok
        self.save_calls = 0
        self.risk_controller = _FakeRiskController()

    def load_state(self, path=None):
        return self._load_ok

    def save_state(self, path=None):
        self.save_calls += 1


class TestDailyRunStatePersistence:
    """#17: _run_daily must never overwrite good paper state on load failure."""

    def _stub_daily(self, monkeypatch, tmp_path, trader):
        state_path = tmp_path / "paper_state.json"
        monkeypatch.setattr("src.core.constants.PAPER_STATE_PATH", str(state_path))
        monkeypatch.setattr(
            "auto_pipeline.META_CONTROLLER_PATH",
            str(tmp_path / "no_meta_controller.json"),
        )
        monkeypatch.setattr(
            "src.trading.paper_trader.PaperTrader", lambda *a, **k: trader
        )

        class FakeOrchestrator:
            def __init__(self, *a, **k):
                pass

            def run(self, date=None, dry_run=False, resolve_outcomes=False):
                return {"decisions": [], "trades": [], "errors": []}

        monkeypatch.setattr(
            "src.signals.orchestrator.DailyOrchestrator", FakeOrchestrator
        )
        monkeypatch.setattr(AutoPipeline, "_check_paper_stops", lambda self, pt: None)
        return state_path

    def test_load_failure_does_not_overwrite_existing_state(self, monkeypatch, tmp_path):
        trader = _FakePaperTrader(load_ok=False)
        state_path = self._stub_daily(monkeypatch, tmp_path, trader)
        original = '{"cash": 12345, "positions": {}}'
        state_path.write_text(original)

        p = AutoPipeline(tickers=["TEST.NS"])
        result = p._run_daily("2026-08-05")

        assert trader.save_calls == 0
        assert state_path.read_text() == original
        assert result["decisions"] == 0

    def test_successful_load_persists_state(self, monkeypatch, tmp_path):
        trader = _FakePaperTrader(load_ok=True)
        state_path = self._stub_daily(monkeypatch, tmp_path, trader)
        state_path.write_text('{"cash": 12345, "positions": {}}')

        p = AutoPipeline(tickers=["TEST.NS"])
        p._run_daily("2026-08-05")

        assert trader.save_calls == 1

    def test_first_run_without_state_file_persists(self, monkeypatch, tmp_path):
        trader = _FakePaperTrader(load_ok=False)
        self._stub_daily(monkeypatch, tmp_path, trader)

        p = AutoPipeline(tickers=["TEST.NS"])
        p._run_daily("2026-08-05")

        assert trader.save_calls == 1
