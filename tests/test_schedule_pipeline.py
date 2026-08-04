"""Tests for the cross-platform scheduler (P1.3/P1.4)."""

import subprocess
from unittest.mock import patch

import pytest

import schedule_pipeline as sp


@pytest.fixture(autouse=True)
def mock_system(monkeypatch):
    """Force Linux + crontab path for deterministic tests regardless of host OS."""
    monkeypatch.setattr(sp, "IS_WINDOWS", False)
    monkeypatch.setattr(sp, "_cron_available", lambda: True)
    monkeypatch.setattr(sp, "_systemd_available", lambda: False)


class TestCronEntry:
    def test_daily_entry_defaults(self):
        entry = sp._cron_entry("15:45")
        assert entry.startswith("45 15 * * 1-5 ")
        assert "run_daily.py" in entry
        assert "--paper-trade" not in entry
        assert entry.endswith(sp._CRON_COMMENT)

    def test_daily_entry_with_paper_trading(self):
        entry = sp._cron_entry("15:45", paper=True, capital=200_000)
        assert "--paper-trade" in entry
        assert "--capital 200000" in entry

    def test_boot_entry_has_delay_and_comment(self):
        entry = sp._cron_boot_entry(paper=True, capital=100_000)
        assert entry.startswith("@reboot sleep 300 && ")
        assert "--paper-trade" in entry
        assert "--capital 100000" in entry
        assert entry.endswith(sp._BOOT_CRON_COMMENT)

    def test_custom_time(self):
        entry = sp._cron_entry("09:30")
        assert entry.startswith("30 09 * * 1-5 ")


class TestCronInstallRemove:
    def test_install_appends_and_preserves_other_entries(self):
        existing = ["0 2 * * * /usr/bin/other_job\n", "# another comment\n"]
        written_stdin = []

        def fake_run(cmd, input=None, capture_output=True, text=True):
            if cmd == ["crontab", "-l"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="".join(existing), stderr="")
            if cmd == ["crontab", "-"]:
                written_stdin.append(input or "")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(sp, "subprocess", autospec=True) as mock_sp:
            mock_sp.run.side_effect = fake_run
            ret = sp._linux_install("15:45", paper=True, capital=200_000)

        assert ret == 0
        written_lines = written_stdin[0].splitlines()
        assert len(written_lines) == 3
        assert "other_job" in written_lines[0]
        assert written_lines[1] == "# another comment"
        assert "--paper-trade" in written_lines[2]

    def test_install_is_idempotent(self):
        written_stdin = []

        def fake_run(cmd, input=None, capture_output=True, text=True):
            if cmd == ["crontab", "-l"]:
                current = written_stdin[-1] if written_stdin else ""
                return subprocess.CompletedProcess(cmd, 0, stdout=current, stderr="")
            if cmd == ["crontab", "-"]:
                written_stdin.append(input or "")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(sp, "subprocess", autospec=True) as mock_sp:
            mock_sp.run.side_effect = fake_run
            sp._linux_install("15:45")
            sp._linux_install("15:45")

        # End state: exactly one StoMar entry, no duplicates
        final = written_stdin[-1]
        assert final.count(sp._CRON_COMMENT) == 1

    def test_remove_clears_all_stomar_entries(self):
        lines = [
            "0 2 * * * /usr/bin/other_job",
            sp._cron_entry("15:45"),
            sp._cron_boot_entry(),
        ]
        written_stdin = []

        def fake_run(cmd, input=None, capture_output=True, text=True):
            if cmd == ["crontab", "-l"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(lines) + "\n", stderr="")
            if cmd == ["crontab", "-"]:
                written_stdin.append(input or "")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(sp, "subprocess", autospec=True) as mock_sp:
            mock_sp.run.side_effect = fake_run
            ret = sp.remove_task()

        assert ret == 0
        assert "other_job" in written_stdin[0]
        assert sp._CRON_COMMENT not in written_stdin[0]
        assert sp._BOOT_CRON_COMMENT not in written_stdin[0]


class TestPublicAPI:
    def test_install_task_routes_to_linux(self):
        with patch.object(sp, "_linux_install", return_value=0) as install:
            ret = sp.install_task(run_time="10:00", paper=True, capital=50_000)
        assert ret == 0
        install.assert_called_once_with("10:00", True, 50_000)

    def test_install_with_boot_adds_boot_entry(self):
        with patch.object(sp, "_linux_install", return_value=0) as install, \
             patch.object(sp, "_linux_install_boot", return_value=0) as boot:
            sp.install_task(boot=True)
        install.assert_called_once()
        boot.assert_called_once()

    def test_boot_not_installed_without_flag(self):
        with patch.object(sp, "_linux_install", return_value=0) as install, \
             patch.object(sp, "_linux_install_boot", return_value=0) as boot:
            sp.install_task()
        boot.assert_not_called()


class TestSystemdFallback:
    def test_units_have_persistent_timer_and_daily_schedule(self):
        service, timer = sp._systemd_units("16:00", paper=True, capital=100_000)
        assert "OnCalendar=Mon..Fri 16:00:00" in timer
        assert "Persistent=true" in timer
        assert f"Unit={sp.TASK_NAME}.service" in timer
        assert "WorkingDirectory=" in service
        assert "--paper-trade" in service
        assert "--capital 100000" in service

    def test_linux_install_falls_back_to_systemd_when_no_cron(self):
        with patch.object(sp, "_cron_available", return_value=False), \
             patch.object(sp, "_systemd_available", return_value=True), \
             patch.object(sp, "_systemd_install", return_value=0) as s_install:
            ret = sp._linux_install("16:00")
        assert ret == 0
        s_install.assert_called_once_with("16:00", False, sp.DEFAULT_CAPITAL)

    def test_linux_boot_is_noop_on_systemd(self):
        with patch.object(sp, "_cron_available", return_value=False), \
             patch.object(sp, "_systemd_available", return_value=True):
            ret = sp._linux_install_boot()
        assert ret == 0

    def test_linux_install_errors_when_no_scheduler(self):
        with patch.object(sp, "_cron_available", return_value=False), \
             patch.object(sp, "_systemd_available", return_value=False):
            ret = sp._linux_install("16:00")
        assert ret == 1

    def test_systemd_install_writes_units_and_enables_timer(self, tmp_path, monkeypatch):
        import os
        monkeypatch.setattr(sp, "os", os)  # keep real os
        unit_dir = str(tmp_path)
        monkeypatch.setattr(os.path, "expanduser", lambda p: unit_dir)

        calls = []

        def fake_run(cmd, capture_output=True, text=True):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(sp, "subprocess", autospec=True) as mock_sp:
            mock_sp.run.side_effect = fake_run
            ret = sp._systemd_install("16:00", paper=True, capital=50_000)

        assert ret == 0
        assert os.path.exists(f"{unit_dir}/{sp.TASK_NAME}.service")
        assert os.path.exists(f"{unit_dir}/{sp.TASK_NAME}.timer")
        enabled = [c for c in calls if "enable" in c]
        assert len(enabled) == 1
        assert f"{sp.TASK_NAME}.timer" in enabled[0]
