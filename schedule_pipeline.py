"""Cross-platform scheduler for the daily signal loop.

Supports:
  - Windows: Task Scheduler (schtasks.exe)
  - Linux/macOS: crontab

Usage:
    python schedule_pipeline.py                # Install (daily at 4 PM)
    python schedule_pipeline.py --remove       # Remove scheduled task
    python schedule_pipeline.py --run-now      # Run daily loop immediately
    python schedule_pipeline.py --status       # Check task status
    python schedule_pipeline.py --time 16:00   # Custom run time
"""

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

TASK_NAME = "StoMar_Daily_Signal"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_SCRIPT = os.path.join(SCRIPT_DIR, "run_daily.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "data", "pipeline_logs")

IS_WINDOWS = platform.system() == "Windows"


def _python_path() -> str:
    """Get the current Python executable path."""
    return sys.executable


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Windows (schtasks)
# ---------------------------------------------------------------------------

def _win_install(run_time: str = "16:00") -> int:
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, f"daily_{datetime.now().strftime('%Y%m%d')}.log")
    python_path = _python_path()

    cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", f'"{python_path}" "{DAILY_SCRIPT}" >> "{log_file}" 2>&1',
        "/sc", "daily",
        "/st", run_time,
        "/f",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' installed successfully (Windows Task Scheduler).")
        print(f"  Schedule: Daily at {run_time}")
        print(f"  Log: {log_file}")
        print(f"  Python: {python_path}")
        print(f"  Script: {DAILY_SCRIPT}")
    else:
        print(f"Failed to install task: {result.stderr}")
    return result.returncode


def _win_remove() -> int:
    cmd = ["schtasks", "/delete", "/tn", TASK_NAME, "/f"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' removed.")
    else:
        print(f"Failed to remove task: {result.stderr}")
    return result.returncode


def _win_status() -> int:
    cmd = ["schtasks", "/query", "/tn", TASK_NAME, "/fo", "LIST"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' is installed:")
        print(result.stdout)
    else:
        print(f"Task '{TASK_NAME}' is not installed.")
        print("Run: python schedule_pipeline.py  (to install)")
    return result.returncode


# ---------------------------------------------------------------------------
# Linux / macOS (crontab)
# ---------------------------------------------------------------------------

_CRON_COMMENT = f"# {TASK_NAME}"


def _cron_available() -> bool:
    """Check if the crontab command is available on this system."""
    return shutil.which("crontab") is not None


def _cron_entry(run_time: str) -> str:
    """Build a cron line like '0 16 * * 1-5 ...'."""
    hour, minute = run_time.split(":")[:2]
    python_path = _python_path()
    log_file = os.path.join(LOG_DIR, f"daily_$(date +\\%Y\\%m\\%d).log")
    # Run Mon-Fri (1-5)
    return f"{minute} {hour} * * 1-5 {python_path} {DAILY_SCRIPT} >> {log_file} 2>&1 {_CRON_COMMENT}"


def _cron_get_lines() -> list[str]:
    """Read current crontab lines (excluding our entry)."""
    if not _cron_available():
        return []
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if _CRON_COMMENT not in line]


def _cron_write_lines(lines: list[str]) -> int:
    """Write lines back to crontab. Returns 0 on success."""
    if not _cron_available():
        print("Error: crontab is not installed on this system.")
        print("Install it with: sudo apt install cron  (Debian/Ubuntu)")
        print("                 sudo yum install cronie  (CentOS/RHEL)")
        return 1
    content = "\n".join(lines) + "\n" if lines else ""
    result = subprocess.run(
        ["crontab", "-"],
        input=content, capture_output=True, text=True,
    )
    return result.returncode


def _linux_install(run_time: str = "16:00") -> int:
    ensure_log_dir()
    if not _cron_available():
        print("Error: crontab is not installed on this system.")
        print("Install it with: sudo apt install cron  (Debian/Ubuntu)")
        print("                 sudo yum install cronie  (CentOS/RHEL)")
        return 1
    lines = _cron_get_lines()
    entry = _cron_entry(run_time)
    lines.append(entry)
    ret = _cron_write_lines(lines)
    if ret == 0:
        print(f"Task '{TASK_NAME}' installed successfully (crontab).")
        print(f"  Schedule: Mon-Fri at {run_time}")
        print(f"  Log: {LOG_DIR}/daily_<date>.log")
        print(f"  Python: {_python_path()}")
        print(f"  Script: {DAILY_SCRIPT}")
    else:
        print("Failed to install crontab entry.")
    return ret


def _linux_remove() -> int:
    lines = _cron_get_lines()
    ret = _cron_write_lines(lines)
    if ret == 0:
        print(f"Task '{TASK_NAME}' removed from crontab.")
    else:
        print("Failed to remove crontab entry.")
    return ret


def _linux_status() -> int:
    if not _cron_available():
        print("Error: crontab is not installed on this system.")
        print("Install it with: sudo apt install cron  (Debian/Ubuntu)")
        print("                 sudo yum install cronie  (CentOS/RHEL)")
        return 1
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode == 0 and _CRON_COMMENT in result.stdout:
        print(f"Task '{TASK_NAME}' is installed:")
        for line in result.stdout.splitlines():
            if _CRON_COMMENT in line or (line.strip() and not line.startswith("#")):
                print(f"  {line}")
        return 0
    else:
        print(f"Task '{TASK_NAME}' is not installed.")
        print("Run: python schedule_pipeline.py  (to install)")
        return 1


# ---------------------------------------------------------------------------
# Public API (cross-platform)
# ---------------------------------------------------------------------------

def install_task(run_time: str = "16:00") -> int:
    """Install a daily scheduled task."""
    if IS_WINDOWS:
        return _win_install(run_time)
    else:
        return _linux_install(run_time)


def remove_task() -> int:
    """Remove the scheduled task."""
    if IS_WINDOWS:
        return _win_remove()
    else:
        return _linux_remove()


def check_status() -> int:
    """Check if the scheduled task exists."""
    if IS_WINDOWS:
        return _win_status()
    else:
        return _linux_status()


def run_now() -> int:
    """Run the daily loop immediately."""
    print(f"Running daily signal loop now...")
    result = subprocess.run(
        [_python_path(), DAILY_SCRIPT],
        capture_output=False,
        cwd=SCRIPT_DIR,
    )
    return result.returncode


def main():
    args = sys.argv[1:]

    if "--remove" in args:
        return remove_task()
    elif "--run-now" in args:
        return run_now()
    elif "--status" in args:
        return check_status()
    else:
        run_time = "16:00"
        for i, arg in enumerate(args):
            if arg == "--time" and i + 1 < len(args):
                run_time = args[i + 1]
        return install_task(run_time)


if __name__ == "__main__":
    sys.exit(main())
