"""Tests for the readiness automation: watchdog cycle + GO/NO-GO report."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def watchdog(tmp_path, monkeypatch):
    mod = _load("readiness_watchdog")
    monkeypatch.setattr(mod, "READINESS_DIR", str(tmp_path / "readiness"))
    monkeypatch.setattr(mod, "PROGRESS_PATH",
                        str(tmp_path / "readiness" / "progress.json"))
    monkeypatch.setattr(mod, "REPLAY_DIR", str(tmp_path / "replay"))
    monkeypatch.setattr(mod, "SUPERVISION_DIR", str(tmp_path / "dry_run"))
    monkeypatch.setattr(mod, "notify_result", lambda *a, **k: None)
    return mod


@pytest.fixture()
def report_mod(tmp_path, monkeypatch):
    mod = _load("readiness_report")
    monkeypatch.setattr(mod, "READINESS_DIR", str(tmp_path / "readiness"))
    monkeypatch.setattr(mod, "PROGRESS_PATH",
                        str(tmp_path / "readiness" / "progress.json"))
    monkeypatch.setattr(mod, "SANDBOX_PATH",
                        str(tmp_path / "readiness" / "kite_sandbox.json"))
    monkeypatch.setattr(mod, "MODELS_DIR", str(tmp_path / "models"))
    return mod


# ── watchdog ───────────────────────────────────────────────────────────────

def test_green_streak_counts_consecutive_passes(watchdog):
    hist = [{"passed": True}, {"passed": True}, {"passed": False},
            {"passed": True}, {"passed": True}, {"passed": True}]
    assert watchdog.green_streak(hist) == 3
    assert watchdog.green_streak([]) == 0


def test_cycle_is_idempotent_per_day(watchdog, monkeypatch):
    def boom():
        raise AssertionError("stages must not rerun for a recorded day")

    monkeypatch.setattr(watchdog, "stage_safety_battery", boom)
    monkeypatch.setattr(watchdog, "stage_supervision", boom)

    entry = {"date": __import__("datetime").date.today().isoformat(),
             "passed": True, "failures": [], "battery": {},
             "supervision": None, "deep_replay": None}
    watchdog.save_progress({"started_at": "x", "requirement_days": 14,
                            "history": [entry]})

    result = watchdog.run_cycle(None, 500_000)
    assert result["date"] == entry["date"]


def test_cycle_records_pass_and_updates_streak(watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "deep_scheduled", lambda today: False)
    monkeypatch.setattr(watchdog, "stage_safety_battery",
                        lambda: {"passed": 23, "total": 23, "failures": []})
    monkeypatch.setattr(watchdog, "stage_supervision",
                        lambda t, c: {"orders_placed": 2, "orders_blocked": 0,
                                      "filled": 2, "positions": 2})
    monkeypatch.setattr(watchdog, "prune_old_artifacts", lambda: None)

    entry = watchdog.run_cycle(None, 500_000, deep=False)
    assert entry["passed"] is True
    progress = watchdog.load_progress()
    assert progress["days_completed"] == 1
    assert progress["last_result"] == "pass"


def test_failure_breaks_streak(watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "deep_scheduled", lambda today: False)
    monkeypatch.setattr(watchdog, "stage_supervision",
                        lambda t, c: {"orders_placed": 0, "orders_blocked": 0,
                                      "filled": 0, "positions": 0})
    monkeypatch.setattr(watchdog, "prune_old_artifacts", lambda: None)
    monkeypatch.setattr(watchdog, "stage_safety_battery",
                        lambda: {"passed": 22, "total": 23,
                                 "failures": ["kill switch scenario"]})

    entry = watchdog.run_cycle(None, 500_000, deep=False)
    assert entry["passed"] is False
    assert watchdog.load_progress()["days_completed"] == 0


def test_watchdog_refuses_live_mode(watchdog, monkeypatch):
    monkeypatch.setenv("STOMAR_LIVE_TRADING", "true")
    monkeypatch.setenv("STOMAR_LIVE_ACCOUNT_APPROVED", "true")
    monkeypatch.setenv("STOMAR_LIVE_CONFIRMATION",
                       "I_CONFIRM_REAL_MONEY_TRADING")
    monkeypatch.setenv("STOMAR_KITE_API_KEY", "k")
    monkeypatch.setenv("STOMAR_KITE_ACCESS_TOKEN", "t")
    with pytest.raises(RuntimeError, match="paper-only"):
        watchdog.assert_paper_only()


# ── report ────────────────────────────────────────────────────────────────

def _write_manifest(path, schema="4", version="2"):
    path.write_text(json.dumps({
        "model_version": version, "feature_schema_version": schema,
        "files": {}}))


def test_models_check_passes_on_v4_bundles(report_mod, tmp_path):
    (tmp_path / "models").mkdir()
    _write_manifest(tmp_path / "models" / "A_manifest.json")
    _write_manifest(tmp_path / "models" / "B_manifest.json")
    assert report_mod.check_models()["ok"] is True


def test_models_check_flags_stale_schema(report_mod, tmp_path):
    (tmp_path / "models").mkdir()
    _write_manifest(tmp_path / "models" / "A_manifest.json", schema="3")
    assert report_mod.check_models()["ok"] is False


def test_sandbox_check_missing_fresh_and_stale(report_mod, tmp_path):
    assert report_mod.check_sandbox()["ok"] is False  # not run

    from datetime import datetime, timezone
    sandbox = tmp_path / "readiness" / "kite_sandbox.json"
    sandbox.parent.mkdir()
    sandbox.write_text(json.dumps(
        {"ok": True, "validated_at": datetime.now(timezone.utc).isoformat()}))
    assert report_mod.check_sandbox()["ok"] is True

    stale = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    sandbox.write_text(json.dumps({"ok": True, "validated_at": stale}))
    assert report_mod.check_sandbox()["ok"] is False


def test_dry_run_check_requires_full_streak(report_mod):
    assert report_mod.check_dry_run()["ok"] is False  # no ledger

    Path(report_mod.PROGRESS_PATH).parent.mkdir(exist_ok=True)
    Path(report_mod.PROGRESS_PATH).write_text(
        json.dumps({"history": [{"passed": True}] * 14}))
    assert report_mod.check_dry_run()["ok"] is True

    Path(report_mod.PROGRESS_PATH).write_text(
        json.dumps({"history": [{"passed": True}] * 13}))
    assert report_mod.check_dry_run()["ok"] is False

    # A failure resets the streak even with many total greens.
    Path(report_mod.PROGRESS_PATH).write_text(json.dumps(
        {"history": [{"passed": True}] * 10 + [{"passed": False}]
         + [{"passed": True}] * 9}))
    assert report_mod.check_dry_run()["ok"] is False


def test_verdict_is_no_go_while_items_open(report_mod, tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    _write_manifest(tmp_path / "models" / "A_manifest.json")
    report = report_mod.build_report()
    assert report["ready_for_real_money"] is False
    assert report["verdict"].startswith("NO-GO")

    # Everything green except the deliberate human opt-in.
    monkeypatch.delenv("STOMAR_LIVE_TRADING", raising=False)
    Path(report_mod.PROGRESS_PATH).parent.mkdir(exist_ok=True)
    Path(report_mod.PROGRESS_PATH).write_text(
        json.dumps({"history": [{"passed": True}] * 14}))
    from datetime import datetime, timezone
    sandbox = Path(report_mod.READINESS_DIR) / "kite_sandbox.json"
    sandbox.write_text(json.dumps(
        {"ok": True, "validated_at": datetime.now(timezone.utc).isoformat()}))
    report = report_mod.build_report()
    assert report["items"]["4_supervised_dry_run_window"]["ok"] is True
    # Credentials remain deliberately unset -> still NO-GO.
    assert report["ready_for_real_money"] is False
