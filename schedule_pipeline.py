"""Windows Task Scheduler helper for nightly pipeline runs.

Usage:
    python schedule_pipeline.py                # Install scheduled task (runs daily at 6 PM)
    python schedule_pipeline.py --remove       # Remove scheduled task
    python schedule_pipeline.py --run-now      # Run pipeline immediately
    python schedule_pipeline.py --status       # Check task status

Requires: Windows Task Scheduler (schtasks.exe, pre-installed on Windows)
"""

import os
import subprocess
import sys
from datetime import datetime

TASK_NAME = "StoMar_Retraining_Pipeline"
PYTHON_PATH = os.path.join(os.path.dirname(sys.executable), "python.exe")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, "run_pipeline.py")
LOG_DIR = os.path.join(SCRIPT_DIR, "data", "pipeline_logs")


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def install_task(run_time: str = "18:00"):
    """Install a daily scheduled task via schtasks."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")

    cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", f'"{PYTHON_PATH}" "{PIPELINE_SCRIPT}" >> "{log_file}" 2>&1',
        "/sc", "daily",
        "/st", run_time,
        "/f",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' installed successfully.")
        print(f"  Schedule: Daily at {run_time}")
        print(f"  Log: {log_file}")
        print(f"  Python: {PYTHON_PATH}")
        print(f"  Script: {PIPELINE_SCRIPT}")
    else:
        print(f"Failed to install task: {result.stderr}")
    return result.returncode


def remove_task():
    """Remove the scheduled task."""
    cmd = ["schtasks", "/delete", "/tn", TASK_NAME, "/f"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' removed.")
    else:
        print(f"Failed to remove task: {result.stderr}")
    return result.returncode


def run_now():
    """Run the pipeline immediately."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    print(f"Running pipeline now...")
    print(f"Log: {log_file}")

    result = subprocess.run(
        [PYTHON_PATH, PIPELINE_SCRIPT],
        capture_output=False,
        cwd=SCRIPT_DIR,
    )

    return result.returncode


def check_status():
    """Check if the scheduled task exists."""
    cmd = ["schtasks", "/query", "/tn", TASK_NAME, "/fo", "LIST"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' is installed:")
        print(result.stdout)
    else:
        print(f"Task '{TASK_NAME}' is not installed.")
        print("Run: python schedule_pipeline.py  (to install)")
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
        run_time = "18:00"
        for i, arg in enumerate(args):
            if arg == "--time" and i + 1 < len(args):
                run_time = args[i + 1]
        return install_task(run_time)


if __name__ == "__main__":
    sys.exit(main())
