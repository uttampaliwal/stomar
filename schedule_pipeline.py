"""Cross-platform scheduler for the daily signal loop.

Supports:
  - Windows: Task Scheduler (schtasks.exe)
  - Linux/macOS: crontab (system crontab binary, no python-crontab dependency)

Usage:
    python schedule_pipeline.py                     # Install (weekdays 3:45 PM)
    python schedule_pipeline.py --paper             # Install with paper trading
    python schedule_pipeline.py --paper --boot      # + boot-time catch-up run
    python schedule_pipeline.py --remove            # Remove scheduled task
    python schedule_pipeline.py --run-now           # Run daily loop immediately
    python schedule_pipeline.py --status            # Check task status
    python schedule_pipeline.py --time 16:00        # Custom run time
    python schedule_pipeline.py --capital 200000    # Paper capital
"""

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

TASK_NAME = "StoMar_Daily_Signal"
BOOT_TASK_NAME = "StoMar_Boot_Catchup"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_SCRIPT = os.path.join(SCRIPT_DIR, "run_daily.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "data", "pipeline_logs")
DEFAULT_CAPITAL = 200_000

IS_WINDOWS = platform.system() == "Windows"


def _python_path() -> str:
    """Get the current Python executable path."""
    return sys.executable


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Windows (schtasks)
# ---------------------------------------------------------------------------

def _base_cmd(paper: bool, capital: float) -> list[str]:
    """Build the run_daily.py command line for scheduled entries."""
    cmd = [_python_path(), DAILY_SCRIPT]
    if paper:
        cmd += ["--paper-trade", "--capital", str(int(capital))]
    return cmd


def _win_install(run_time: str = "15:45", paper: bool = False,
                 capital: float = DEFAULT_CAPITAL) -> int:
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, f"daily_{datetime.now().strftime('%Y%m%d')}.log")
    cmd_line = " ".join(_base_cmd(paper, capital))

    cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", f'"{cmd_line}" >> "{log_file}" 2>&1',
        "/sc", "daily",
        "/st", run_time,
        "/f",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' installed successfully (Windows Task Scheduler).")
        print(f"  Schedule: Daily at {run_time}")
        print(f"  Paper trading: {paper}")
        print(f"  Log: {log_file}")
        print(f"  Python: {_python_path()}")
        print(f"  Script: {DAILY_SCRIPT}")
    else:
        print(f"Failed to install task: {result.stderr}")
    return result.returncode


def _win_install_boot(paper: bool = False, capital: float = DEFAULT_CAPITAL) -> int:
    """Boot-time task: catches up missed days when the PC was off at 4PM."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, f"boot_{datetime.now().strftime('%Y%m%d')}.log")
    cmd_line = " ".join(_base_cmd(paper, capital))

    cmd = [
        "schtasks", "/create",
        "/tn", BOOT_TASK_NAME,
        "/tr", f'"{cmd_line}" >> "{log_file}" 2>&1',
        "/sc", "onstart",
        "/delay", "0005:00",
        "/f",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{BOOT_TASK_NAME}' installed (boot-time catch-up).")
    else:
        print(f"Failed to install boot task: {result.stderr}")
    return result.returncode


def _win_remove() -> int:
    for name in (TASK_NAME, BOOT_TASK_NAME):
        cmd = ["schtasks", "/delete", "/tn", name, "/f"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Task '{name}' removed.")
    return 0


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
_BOOT_CRON_COMMENT = f"# {BOOT_TASK_NAME}"


def _cron_available() -> bool:
    """Check if the crontab command is available on this system."""
    return shutil.which("crontab") is not None


# ---------------------------------------------------------------------------
# Linux (systemd user timer) — fallback when cron is not installed
# ---------------------------------------------------------------------------

def _systemd_available() -> bool:
    """Check if systemd user sessions are usable (no cron needed)."""
    return shutil.which("systemctl") is not None and os.path.isdir("/run/systemd/system")


def _systemd_units(run_time: str, paper: bool = False,
                   capital: float = DEFAULT_CAPITAL) -> tuple[str, str]:
    """Build the systemd service + timer unit contents (user level).

    The timer uses ``Persistent=true``: if the machine was off at the
    scheduled time, the service fires right after the next boot/login —
    this is the systemd equivalent of the cron @reboot catch-up.
    """
    hour, minute = run_time.split(":")[:2]
    cmd_line = " ".join(_base_cmd(paper, capital))

    service = f"""[Unit]
Description=StoMar daily signal pipeline

[Service]
Type=oneshot
WorkingDirectory={SCRIPT_DIR}
ExecStart={cmd_line}

[Install]
WantedBy=default.target
"""
    timer = f"""[Unit]
Description=StoMar daily signal timer (with missed-run catch-up)

[Timer]
OnCalendar=Mon..Fri {hour}:{minute}:00
Persistent=true
Unit={TASK_NAME}.service

[Install]
WantedBy=timers.target
"""
    return service, timer


def _systemd_install(run_time: str = "15:45", paper: bool = False,
                     capital: float = DEFAULT_CAPITAL) -> int:
    """Install a systemd user timer for the daily run."""
    ensure_log_dir()
    service, timer = _systemd_units(run_time, paper, capital)
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    with open(os.path.join(unit_dir, f"{TASK_NAME}.service"), "w") as f:
        f.write(service)
    with open(os.path.join(unit_dir, f"{TASK_NAME}.timer"), "w") as f:
        f.write(timer)

    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", f"{TASK_NAME}.timer"],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"systemd error ({' '.join(cmd)}): {result.stderr}")
            return result.returncode

    # Linger keeps the timer running even when no session is open
    subprocess.run(["loginctl", "enable-linger", os.environ.get("USER", "")],
                   capture_output=True, text=True)

    print(f"Task '{TASK_NAME}' installed successfully (systemd user timer).")
    print(f"  Schedule: Mon-Fri at {run_time} (+ catch-up after boot if missed)")
    print(f"  Paper trading: {paper}")
    print(f"  Units: {unit_dir}/{TASK_NAME}.{{service,timer}}")
    print(f"  Python: {_python_path()}")
    print(f"  Script: {DAILY_SCRIPT}")
    return 0


def _systemd_remove() -> int:
    subprocess.run(["systemctl", "--user", "disable", "--now", f"{TASK_NAME}.timer"],
                   capture_output=True, text=True)
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    for name in (f"{TASK_NAME}.service", f"{TASK_NAME}.timer"):
        path = os.path.join(unit_dir, name)
        if os.path.exists(path):
            os.remove(path)
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True)
    print(f"Task '{TASK_NAME}' removed (systemd user timer).")
    return 0


def _systemd_status() -> int:
    result = subprocess.run(
        ["systemctl", "--user", "list-timers", f"{TASK_NAME}.timer", "--no-pager"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and TASK_NAME in result.stdout:
        print(f"Task '{TASK_NAME}' is installed (systemd user timer):")
        print(result.stdout)
        return 0
    print(f"Task '{TASK_NAME}' is not installed.")
    print("Run: python schedule_pipeline.py  (to install)")
    return 1


def _linux_available() -> bool:
    """Return the scheduler available on this Linux machine."""
    return _cron_available() or _systemd_available()


def _cron_entry(run_time: str, paper: bool = False,
                capital: float = DEFAULT_CAPITAL) -> str:
    """Build a cron line like '45 15 * * 1-5 ...'."""
    hour, minute = run_time.split(":")[:2]
    cmd_line = " ".join(_base_cmd(paper, capital))
    log_file = os.path.join(LOG_DIR, "daily_$(date +\\%Y\\%m\\%d).log")
    # Run Mon-Fri (1-5)
    return f"{minute} {hour} * * 1-5 {cmd_line} >> {log_file} 2>&1 {_CRON_COMMENT}"


def _cron_boot_entry(paper: bool = False, capital: float = DEFAULT_CAPITAL) -> str:
    """Boot-time entry: catch up missed days when the machine was off."""
    cmd_line = " ".join(_base_cmd(paper, capital))
    log_file = os.path.join(LOG_DIR, "boot_$(date +\\%Y\\%m\\%d).log")
    # 5-minute delay lets the network come up after boot
    return f"@reboot sleep 300 && {cmd_line} >> {log_file} 2>&1 {_BOOT_CRON_COMMENT}"


def _cron_get_lines(comment: str = _CRON_COMMENT) -> list[str]:
    """Read current crontab lines (excluding our entries)."""
    if not _cron_available():
        return []
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines()
            if _CRON_COMMENT not in line and _BOOT_CRON_COMMENT not in line]


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


def _linux_install(run_time: str = "15:45", paper: bool = False,
                   capital: float = DEFAULT_CAPITAL) -> int:
    """Install the daily task. Prefers crontab, falls back to systemd."""
    if not _cron_available():
        if _systemd_available():
            return _systemd_install(run_time, paper, capital)
        print("Error: neither crontab nor systemd user timers are available.")
        print("Install cron with: sudo apt install cron  (Debian/Ubuntu)")
        print("                  sudo yum install cronie  (CentOS/RHEL)")
        return 1
    ensure_log_dir()
    lines = _cron_get_lines()
    lines.append(_cron_entry(run_time, paper, capital))
    ret = _cron_write_lines(lines)
    if ret == 0:
        print(f"Task '{TASK_NAME}' installed successfully (crontab).")
        print(f"  Schedule: Mon-Fri at {run_time}")
        print(f"  Paper trading: {paper}")
        print(f"  Log: {LOG_DIR}/daily_<date>.log")
        print(f"  Python: {_python_path()}")
        print(f"  Script: {DAILY_SCRIPT}")
    else:
        print("Failed to install crontab entry.")
    return ret


def _linux_install_boot(paper: bool = False, capital: float = DEFAULT_CAPITAL) -> int:
    """Boot-time crontab entry (@reboot, after a 5-minute network delay).

    With systemd the boot catch-up is already covered by Persistent=true
    on the daily timer, so nothing extra is installed there.
    """
    if not _cron_available():
        if _systemd_available():
            print(f"Boot catch-up not needed — systemd timer {TASK_NAME} has Persistent=true.")
            return 0
        print("Error: crontab is not installed on this system.")
        print("Install it with: sudo apt install cron  (Debian/Ubuntu)")
        print("                 sudo yum install cronie  (CentOS/RHEL)")
        return 1
    lines = _cron_get_lines()
    lines.append(_cron_boot_entry(paper, capital))
    ret = _cron_write_lines(lines)
    if ret == 0:
        print(f"Task '{BOOT_TASK_NAME}' installed (boot-time catch-up).")
        print(f"  Entry: @reboot (sleep 300) -> {DAILY_SCRIPT}")
    else:
        print("Failed to install boot crontab entry.")
    return ret


def _linux_remove() -> int:
    if _systemd_available() and os.path.exists(
        os.path.join(os.path.expanduser("~/.config/systemd/user"), f"{TASK_NAME}.service")
    ):
        return _systemd_remove()
    if not _cron_available():
        print("Error: crontab is not installed on this system.")
        print("Install it with: sudo apt install cron  (Debian/Ubuntu)")
        print("                 sudo yum install cronie  (CentOS/RHEL)")
        return 1
    lines = _cron_get_lines()
    ret = _cron_write_lines(lines)
    if ret == 0:
        print(f"Task '{TASK_NAME}' removed from crontab.")
    else:
        print("Failed to remove crontab entry.")
    return ret


def _linux_status() -> int:
    if _systemd_available() and os.path.exists(
        os.path.join(os.path.expanduser("~/.config/systemd/user"), f"{TASK_NAME}.service")
    ):
        return _systemd_status()
    if not _cron_available():
        print("Error: crontab is not installed on this system.")
        print("Install it with: sudo apt install cron  (Debian/Ubuntu)")
        print("                 sudo yum install cronie  (CentOS/RHEL)")
        return 1
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    installed = [name for name in (TASK_NAME, BOOT_TASK_NAME)
                 if name in result.stdout]
    if result.returncode == 0 and installed:
        print(f"Tasks installed: {', '.join(installed)}")
        for line in result.stdout.splitlines():
            if _CRON_COMMENT in line or _BOOT_CRON_COMMENT in line:
                print(f"  {line}")
        return 0
    else:
        print(f"Task '{TASK_NAME}' is not installed.")
        print("Run: python schedule_pipeline.py  (to install)")
        return 1


# ---------------------------------------------------------------------------
# Public API (cross-platform)
# ---------------------------------------------------------------------------

def install_task(run_time: str = "15:45", paper: bool = False,
                 capital: float = DEFAULT_CAPITAL, boot: bool = False) -> int:
    """Install the daily scheduled task (and optionally the boot catch-up)."""
    if IS_WINDOWS:
        ret = _win_install(run_time, paper, capital)
        if boot and ret == 0:
            ret = _win_install_boot(paper, capital)
    else:
        ret = _linux_install(run_time, paper, capital)
        if boot and ret == 0:
            ret = _linux_install_boot(paper, capital)
    return ret


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
    print("Running daily signal loop now...")
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
        run_time = "15:45"
        capital = DEFAULT_CAPITAL
        for i, arg in enumerate(args):
            if arg == "--time" and i + 1 < len(args):
                run_time = args[i + 1]
            elif arg == "--capital" and i + 1 < len(args):
                capital = float(args[i + 1])
        return install_task(run_time=run_time,
                            paper="--paper" in args,
                            capital=capital,
                            boot="--boot" in args)


if __name__ == "__main__":
    sys.exit(main())
