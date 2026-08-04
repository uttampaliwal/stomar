"""Cross-device state sync via the git remote (private repo).

StoMar's authoritative state is tiny (~150 KB): the ledger, the paper
account, and the trained model bundles. Syncing it through the private git
remote gives you ONE paper-trading clock shared across all your devices —
no repeated 252-day backfills, no split accounts, no missed days.

Usage:
    python scripts/sync_state.py pull     # BEFORE the pipeline on this device
    python scripts/sync_state.py push     # AFTER the pipeline run finishes
    python scripts/sync_state.py status   # local vs remote state

What is shared (force-tracked in git):
    data/stomar.db          — ledger (decisions, outcomes, snapshots)
    data/paper_state.json   — paper account (cash, positions, equity)
    data/paper_session.json — session export
    models/                 — trained bundles + manifests + registry

What stays local (regenerable):
    parquet caches, market_data.db, monitoring files, pipeline logs, .env

Rules:
    - pull BEFORE the run, push AFTER it finishes (never mid-run).
    - Only one device runs/pushes per day (your travel pattern).
    - The repo MUST be private — .env and secrets are never committed.
"""

import os
import platform
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_PATHS = [
    "data/stomar.db",
    "data/paper_state.json",
    "data/paper_session.json",
    "models/",
]


def _run(args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd or ROOT, capture_output=True, text=True, check=check,
    )


def _device() -> str:
    return platform.node()


def pull() -> int:
    """Fetch the latest shared state (code + ledger + paper + models)."""
    print(f"[{_device()}] Pulling shared state from remote...")
    result = _run(["git", "fetch", "origin"], check=False)
    if result.returncode != 0:
        print(f"ERROR: git fetch failed:\n{result.stderr}")
        return 1

    result = _run(["git", "pull", "--ff-only"], check=False)
    if result.returncode != 0:
        print(
            "ERROR: git pull --ff-only failed.\n"
            "This usually means this device has uncommitted changes to shared "
            "files (a previous run was not pushed).\n"
            f"{result.stdout}\n{result.stderr}\n"
            "If the pipeline is NOT running, resolve first:\n"
            "  git stash && python scripts/sync_state.py pull && git stash pop\n"
            "or force-sync the shared files with:\n"
            "  git checkout FETCH_HEAD -- data/stomar.db data/paper_state.json "
            "data/paper_session.json models/"
        )
        return 1

    print("Pull OK. Current state:")
    print(_run(["git", "log", "-1", "--oneline"]).stdout.strip())
    return 0


def push(dry_run: bool = False) -> int:
    """Checkpoint the ledger WAL, commit shared state, push to remote."""
    print(f"[{_device()}] Preparing shared state for push...")

    # 1. Make stomar.db self-contained (merge WAL into the main file)
    sys.path.insert(0, ROOT)
    try:
        from src.trading.ledger import Ledger
        ledger = Ledger()
        ledger.checkpoint_wal()
        ledger.close()
        print("  ledger WAL checkpointed")
    except Exception as e:
        print(f"  WARNING: ledger checkpoint failed: {e}")

    # 2. Stage only the shared paths
    add = _run(["git", "add", "--"] + SHARED_PATHS, check=False)
    if add.returncode != 0:
        print(f"ERROR: git add failed:\n{add.stderr}")
        return 1

    staged = _run(["git", "diff", "--cached", "--quiet"], check=False)
    if staged.returncode == 0:
        print("  no state changes to push")
        return 0

    print("  staged:")
    print(_run(["git", "diff", "--cached", "--stat"]).stdout.strip())

    if dry_run:
        print("DRY-RUN: not committing/pushing.")
        return 0

    # 3. Commit + push
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = _run(
        ["git", "commit", "-m", f"state: {stamp} [{_device()}]"],
        check=False,
    )
    if commit.returncode != 0:
        print(f"ERROR: git commit failed:\n{commit.stderr}")
        return 1

    push_result = _run(["git", "push", "origin"], check=False)
    if push_result.returncode != 0:
        print(
            "ERROR: git push failed (remote may have newer state).\n"
            f"{push_result.stderr}\n"
            "If another device pushed meanwhile, resolve:\n"
            "  python scripts/sync_state.py pull\n"
            "  git push origin\n"
        )
        return 1

    print("Push OK:", _run(["git", "log", "-1", "--oneline"]).stdout.strip())
    return 0


def status() -> int:
    """Show shared state: local vs remote."""
    _run(["git", "fetch", "origin"], check=False)
    print(f"Device:        {_device()}")
    print(_run(["git", "status", "-sb"]).stdout.strip())
    print(_run(["git", "log", "-1", "--oneline", "HEAD"]).stdout.strip())
    print(_run(["git", "log", "-1", "--oneline", "origin/HEAD"]).stdout.strip())
    print("\nShared files (local mtime):")
    for path in SHARED_PATHS:
        full = os.path.join(ROOT, path)
        if os.path.exists(full):
            print(f"  {path:32s} {datetime.fromtimestamp(os.path.getmtime(full)):%Y-%m-%d %H:%M}")
        else:
            print(f"  {path:32s} MISSING (run pull)")
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("pull", "push", "status"):
        print(__doc__)
        return 2
    if sys.argv[1] == "pull":
        return pull()
    if sys.argv[1] == "status":
        return status()
    return push(dry_run="--dry-run" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
