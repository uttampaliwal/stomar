"""Readiness watchdog: automated daily evidence collection for live-money gate.

Chains the supervised dry-run and the safety-scenario battery every
trading day, accumulates a tamper-evident progress ledger, notifies on
failures, and prunes stale artifacts. After the required observation
window (default 14 green sessions), `scripts/readiness_report.py` turns
this ledger into the GO/NO-GO verdict.

Paper-only by construction: every stage asserts DryRunBroker / paper mode
before doing anything.

Usage:
    python scripts/readiness_watchdog.py               # one daily cycle
    python scripts/readiness_watchdog.py --install     # cron: weekdays 16:10
    python scripts/readiness_watchdog.py --remove
    python scripts/readiness_watchdog.py --status
    python scripts/readiness_watchdog.py --deep        # force weekly replay
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# All artifact dirs resolve against the PROJECT ROOT's data/ — the same
# location every other script writes to (they must never diverge).
READINESS_DIR = os.path.join(PROJECT_ROOT, "data", "readiness")
PROGRESS_PATH = os.path.join(READINESS_DIR, "progress.json")
REPLAY_DIR = os.path.join(PROJECT_ROOT, "data", "dry_run_replay")
SUPERVISION_DIR = os.path.join(PROJECT_ROOT, "data", "dry_run")
CRON_MARKER = "STOMAR_READINESS_WATCHDOG"
REQUIREMENT_DAYS = 14
RETENTION_DAYS = 35


# ── progress ledger ────────────────────────────────────────────────────────

def load_progress() -> dict:
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requirement_days": REQUIREMENT_DAYS,
        "history": [],
    }


def save_progress(p: dict):
    os.makedirs(READINESS_DIR, exist_ok=True)
    from src.core.secure_io import atomic_write_json
    atomic_write_json(PROGRESS_PATH, p)


def green_streak(history: list[dict]) -> int:
    """Consecutive passing days counted from the most recent entry."""
    streak = 0
    for entry in reversed(history):
        if entry.get("passed"):
            streak += 1
        else:
            break
    return streak


# ── stages ─────────────────────────────────────────────────────────────────

def assert_paper_only() -> None:
    """Hard invariant: the watchdog must never run with a live broker."""
    from src.core.trading_mode import get_trading_mode
    mode = get_trading_mode()
    if mode.value != "paper":
        raise RuntimeError(
            f"watchdog refuses to run in '{mode}' mode — it is a paper-only "
            "tool. Unset the live-trading environment to collect readiness "
            "evidence.")


def stage_safety_battery() -> dict:
    """The 24-check safety battery from dry_run_replay (isolated dirs)."""
    sys.path.insert(0, SCRIPT_DIR)
    from scripts.dry_run_replay import run_safety_scenarios

    checks, failures = run_safety_scenarios()
    return {
        "passed": sum(1 for c in checks if c["ok"]),
        "total": len(checks),
        "failures": failures,
    }


def stage_supervision(tickers: list[str] | None, capital: float) -> dict:
    """Today's decisions through ExecutionManager -> DryRunBroker."""
    from scripts.dry_run_supervision import run_supervision

    report = run_supervision(tickers, capital=capital)
    errors = report.get("orders_blocked", 0)
    return {
        "orders_placed": report.get("orders_placed", 0),
        "orders_blocked": errors,
        "filled": len(report.get("filled", [])),
        "positions": len(report.get("positions", [])),
        "report": os.path.join(SUPERVISION_DIR, "state.json"),
    }


def stage_deep_replay(days: int = 15) -> dict:
    """Full historical replay regression (weekly / --deep)."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "dry_run_replay.py"),
           "--days", str(days), "--skip-scenarios"]
    proc = subprocess.run(cmd, cwd=os.path.dirname(SCRIPT_DIR),
                          capture_output=True, text=True, timeout=3600)
    summary_path = os.path.join(REPLAY_DIR, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, ValueError):
            pass
    checks = summary.get("checks_total", 0)
    passed = summary.get("checks_passed", 0)
    if proc.returncode != 0 or checks == 0:
        tail = (proc.stderr or proc.stdout or "")[-300:].strip()
        failures = summary.get("failures") or [
            f"replay rc={proc.returncode}" + (f"; {tail}" if tail
                                              else "; no summary written")]
    else:
        failures = summary.get("failures", [])
    return {
        "sessions": len(summary.get("replay", {}).get("sessions", [])),
        "passed": passed,
        "total": checks,
        "failures": failures,
        "returncode": proc.returncode,
    }


def prune_old_artifacts():
    """Keep artifact dirs bounded — delete session files older than N days."""
    cutoff = datetime.now().timestamp() - RETENTION_DAYS * 86400
    for directory, prefixes in (
        (REPLAY_DIR, ("state-", "orders-")),
        (SUPERVISION_DIR, ("orders-",)),
    ):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            try:
                if name.startswith(prefixes) and os.path.getmtime(path) < cutoff:
                    os.unlink(path)
            except OSError:
                continue


def notify_result(day_result: dict, streak: int):
    try:
        from src.core.notifier import notify
        ok = day_result["passed"]
        severity = "info" if ok else "error"
        detail = (
            f"day={'PASS' if ok else 'FAIL'} "
            f"streak={streak}/{REQUIREMENT_DAYS} "
            f"battery={day_result['battery']['passed']}/"
            f"{day_result['battery']['total']}"
        )
        if not ok:
            detail += f" failures={day_result['failures'][:4]}"
        notify("readiness_watchdog", severity=severity, details=detail)
    except Exception:
        pass  # notification is best-effort; never fail the watchdog


# ── daily cycle ────────────────────────────────────────────────────────────

def deep_scheduled(today: date) -> bool:
    """Weekly regression day: Saturday (also forced via --deep)."""
    return today.weekday() == 5


def run_cycle(tickers: list[str] | None, capital: float,
              deep: bool = False) -> dict:
    today = date.today().isoformat()
    progress = load_progress()

    for entry in progress["history"]:
        if entry["date"] == today:
            print(f"[watchdog] {today} already recorded — idempotent skip")
            return entry

    assert_paper_only()

    print(f"=== StoMar Readiness Watchdog — {today} ===")
    battery = stage_safety_battery()
    print(f"battery: {battery['passed']}/{battery['total']}")
    if battery["failures"]:
        print(f"  failures: {battery['failures']}")

    supervision = None
    supervision_error = ""
    try:
        supervision = stage_supervision(tickers, capital)
        print(f"supervision: orders={supervision['orders_placed']} "
              f"blocked={supervision['orders_blocked']}")
    except Exception as exc:  # noqa: BLE001 — record, don't crash the ledger
        supervision_error = f"{type(exc).__name__}: {exc}"[:300]
        print(f"supervision ERROR: {supervision_error}")

    deep_replay = None
    if deep or deep_scheduled(date.today()):
        print("deep replay (weekly regression)...")
        try:
            deep_replay = stage_deep_replay()
            print(f"deep replay: {deep_replay['passed']}/{deep_replay['total']} "
                  f"over {deep_replay['sessions']} sessions")
        except Exception as exc:  # noqa: BLE001
            deep_replay = {"failures": [f"{type(exc).__name__}: {exc}"[:300]],
                           "passed": 0, "total": 0, "sessions": 0}

    failures = list(battery["failures"])
    if supervision_error:
        failures.append(f"supervision: {supervision_error}")
    if deep_replay and deep_replay.get("failures"):
        failures.extend(f"deep_replay: {f}" for f in deep_replay["failures"])

    entry = {
        "date": today,
        "passed": not failures,
        "failures": failures,
        "battery": battery,
        "supervision": supervision,
        "deep_replay": deep_replay,
    }
    progress["history"].append(entry)
    progress["history"] = progress["history"][-90:]  # bound the ledger
    progress["days_completed"] = green_streak(progress["history"])
    progress["last_result"] = "pass" if entry["passed"] else "fail"
    save_progress(progress)
    prune_old_artifacts()

    streak = progress["days_completed"]
    print(f"streak: {streak}/{REQUIREMENT_DAYS} green sessions")
    notify_result(entry, streak)

    if not entry["passed"]:
        print("FAILED:", "; ".join(failures))
    return entry


# ── self-scheduling (cron if present, else systemd user timers) ────────────

DAILY_TASK = "StoMar_Readiness_Watchdog"
DEEP_TASK = "Stomar_Readiness_Deep".replace("mar", "Mar")  # StoMar_Readiness_Deep


def _scheduler_kind() -> str:
    import platform
    import shutil
    if platform.system() == "Windows":
        return "windows"
    if shutil.which("crontab"):
        return "cron"
    if shutil.which("systemctl") and os.path.isdir("/run/systemd/system"):
        return "systemd"
    return "none"


def _cron_lines() -> list[str]:
    python = sys.executable
    script = os.path.join(SCRIPT_DIR, "readiness_watchdog.py")
    log = os.path.join(PROJECT_ROOT, "data", "pipeline_logs", "readiness_%a.log")
    daily = (f"10 16 * * 1-5 {python} {script} >> {log} 2>&1 "
             f"# {CRON_MARKER}")
    weekly = (f"0 10 * * 6 {python} {script} --deep >> {log} 2>&1 "
              f"# {CRON_MARKER}")
    return [daily, weekly]


def _systemd_units(name: str, on_calendar: str,
                   extra_args: str = "") -> tuple[str, str]:
    python = sys.executable
    script = os.path.join(SCRIPT_DIR, "readiness_watchdog.py")
    service = f"""[Unit]
Description=StoMar readiness watchdog ({name})

[Service]
Type=oneshot
WorkingDirectory={PROJECT_ROOT}
ExecStart={python} {script} {extra_args}

[Install]
WantedBy=default.target
"""
    timer = f"""[Unit]
Description=StoMar readiness watchdog timer ({name}, missed-run catch-up)

[Timer]
OnCalendar={on_calendar}
Persistent=true
Unit={name}.service

[Install]
WantedBy=timers.target
"""
    return service, timer


def _install_systemd() -> int:
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    units = {
        DAILY_TASK: ("Mon..Fri 16:10:00", ""),
        DEEP_TASK: ("Sat 10:00:00", "--deep"),
    }
    for name, (cal, args) in units.items():
        service, timer = _systemd_units(name, cal, args)
        with open(os.path.join(unit_dir, f"{name}.service"), "w") as f:
            f.write(service)
        with open(os.path.join(unit_dir, f"{name}.timer"), "w") as f:
            f.write(timer)
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", f"{DAILY_TASK}.timer"],
        ["systemctl", "--user", "enable", "--now", f"{DEEP_TASK}.timer"],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"systemd error ({' '.join(cmd)}): {result.stderr}")
            return result.returncode
    subprocess.run(["loginctl", "enable-linger", os.environ.get("USER", "")],
                   capture_output=True, text=True)
    print(f"Installed systemd user timers: {DAILY_TASK} (Mon-Fri 16:10), "
          f"{DEEP_TASK} (Sat 10:00, full replay regression)")
    print("Persistent=true -> missed runs fire after next boot/login.")
    return 0


def _remove_systemd() -> int:
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    for name in (DAILY_TASK, DEEP_TASK):
        subprocess.run(["systemctl", "--user", "disable", "--now",
                        f"{name}.timer"], capture_output=True, text=True)
        for ext in (".service", ".timer"):
            path = os.path.join(unit_dir, f"{name}{ext}")
            if os.path.exists(path):
                os.remove(path)
    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   capture_output=True, text=True)
    print("watchdog timers removed")
    return 0


def install_cron() -> int:
    kind = _scheduler_kind()
    if kind == "windows":
        print("On Windows use Task Scheduler:")
        print(f'  schtasks /create /tn StoMar_Readiness_Watchdog /tr '
              f'"{sys.executable} '
              f'{os.path.join(SCRIPT_DIR, "readiness_watchdog.py")}" '
              f'/sc weekly /d MON,TUE,WED,THU,FRI /st 16:10 /f')
        return 1
    if kind == "cron":
        existing = subprocess.run(["crontab", "-l"], capture_output=True,
                                  text=True)
        lines = (existing.stdout if existing.returncode == 0 else "")
        kept = [ln for ln in lines.splitlines() if CRON_MARKER not in ln]
        kept.extend(_cron_lines())
        payload = "\n".join(kept) + "\n"
        proc = subprocess.run(["crontab", "-"], input=payload,
                              text=True, capture_output=True)
        if proc.returncode != 0:
            print("crontab install failed:", proc.stderr)
            return 1
        for line in _cron_lines():
            print("installed:", line)
        return 0
    if kind == "systemd":
        return _install_systemd()
    print("No scheduler available (no crontab, no systemd). "
          "Run the watchdog manually or from auto_pipeline.")
    return 1


def remove_cron() -> int:
    kind = _scheduler_kind()
    if kind == "cron":
        existing = subprocess.run(["crontab", "-l"], capture_output=True,
                                  text=True)
        lines = (existing.stdout if existing.returncode == 0 else "")
        kept = [ln for ln in lines.splitlines() if CRON_MARKER not in ln]
        payload = "\n".join(kept) + ("\n" if kept else "")
        subprocess.run(["crontab", "-"], input=payload, text=True,
                       capture_output=True)
        removed = len(kept) != len(lines.splitlines())
        print("watchdog cron removed" if removed else "not installed")
        return 0
    if kind == "systemd":
        return _remove_systemd()
    print("nothing to remove")
    return 0


def status() -> int:
    progress = load_progress()
    streak = progress.get("days_completed", green_streak(progress["history"]))
    print(f"green streak : {streak}/{REQUIREMENT_DAYS}")
    print(f"last result  : {progress.get('last_result', 'n/a')}")
    print(f"days recorded: {len(progress['history'])}")
    for entry in progress["history"][-5:]:
        mark = "PASS" if entry["passed"] else "FAIL"
        extra = "" if entry["passed"] else f" ({entry['failures'][0][:60]})"
        print(f"  {entry['date']}  {mark}{extra}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Readiness watchdog")
    parser.add_argument("--ticker", nargs="+", default=None)
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--deep", action="store_true",
                        help="force the weekly historical replay regression")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.install:
        return install_cron()
    if args.remove:
        return remove_cron()
    if args.status:
        return status()

    entry = run_cycle(args.ticker, args.capital, deep=args.deep)
    return 0 if entry["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
