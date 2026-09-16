"""Cross-device state sync via a PRIVATE companion git repo.

StoMar's authoritative state is small: the ledger (~100 KB), the paper
account, the 14-day watchdog evidence (a few hundred bytes of JSON), and
the trained model bundles (~60 MB one-time, deltas after that). Syncing
it through a private companion repo gives you ONE paper-trading clock
shared across all your devices — no repeated backfills, no split
accounts, no missed days.

Design (see docs/REAL_MONEY_READINESS.md section 7/8):
  - Code lives in the PUBLIC stomar repo. State NEVER goes there.
  - State lives in the PRIVATE companion repo (default: ../stomar-state),
    mirroring the same relative paths.
  - The watchdog is idempotent per day ("already recorded" skips), so the
    first machine that runs on a given day wins; the streak counts
    trailing green sessions, so missed days are harmless.

Usage:
    python scripts/sync_state.py pull [--repo PATH]    # BEFORE the pipeline
    python scripts/sync_state.py push [--repo PATH]    # AFTER the run
    python scripts/sync_state.py status [--repo PATH]  # local vs companion

What is shared (copied worktree <-> companion):
    data/stomar.db              — ledger (decisions, outcomes; WAL-checkpointed)
    data/paper_state.json       — paper account (cash, positions, equity)
    data/paper_session.json     — session export
    data/readiness/progress.json — 14-day watchdog evidence (the streak itself)
    data/readiness/report.json   — latest GO/NO-GO report
    models/                     — trained bundles + manifests + registry +
                                 production + quarantine state (full tree on
                                 purpose: quarantine/registry must travel too)

What stays local (regenerable, never synced):
    parquet caches, market_data.db, monitoring files, pipeline logs,
    sentiment/FII-DII/PCR/MTF snapshots (short TTL — refetch), .env/secrets.

Rules:
    - pull BEFORE the run, push AFTER it finishes (never mid-run).
    - Only one device runs/pushes per day.
    - The companion repo MUST be private — paper positions and the
      readiness trajectory never touch the public repo.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_PATHS = [
    "data/stomar.db",
    "data/paper_state.json",
    "data/paper_session.json",
    "data/readiness/progress.json",
    "data/readiness/report.json",
    "models/",
]
DEFAULT_STATE_REPO = os.path.normpath(os.path.join(ROOT, "..", "stomar-state"))


def _run(args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd or ROOT, capture_output=True, text=True, check=check,
    )


def _device() -> str:
    return platform.node()


def _resolve_repo(explicit: str | None) -> str | None:
    """Return the companion checkout path, or None with setup instructions."""
    repo = os.path.abspath(explicit) if explicit else DEFAULT_STATE_REPO
    if os.path.isdir(os.path.join(repo, ".git")):
        return repo
    print(f"Companion state repo not found at: {repo}")
    print("One-time setup (private repo, e.g. github.com/<you>/stomar-state):")
    print(f"  git clone <private-remote> {repo}")
    print("  # then seed it once from the machine holding the newest state:")
    print("  python scripts/sync_state.py push --repo " + repo)
    return None


def _mirror_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)


def _mirror_tree(src: str, dst: str) -> None:
    """Copy a directory via temp + atomic swap, so an interrupted copy can
    never leave a half-written models/ tree behind."""
    tmp = dst + ".__sync_new__"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, symlinks=False)
    backup = dst + ".__sync_old__"
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)
    if os.path.isdir(dst):
        os.replace(dst, backup)
    os.replace(tmp, dst)
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)


def _mirror(src_root: str, dst_root: str) -> list[str]:
    """Copy every shared path that exists in src_root. Returns copied list."""
    copied = []
    for rel in SHARED_PATHS:
        rel = rel.rstrip("/\\")
        src = os.path.join(src_root, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_root, rel)
        if os.path.isdir(src):
            _mirror_tree(src, dst)
        else:
            _mirror_file(src, dst)
        copied.append(rel)
    return copied


def pull(repo: str | None) -> int:
    """Fetch the latest shared state, then copy companion -> worktree."""
    state = _resolve_repo(repo)
    if state is None:
        return 1
    print(f"[{_device()}] Pulling shared state...")
    fetch = _run(["git", "-C", state, "fetch", "origin"], check=False)
    if fetch.returncode != 0:
        print(f"ERROR: git fetch failed:\n{fetch.stderr}")
        return 1
    pr = _run(["git", "-C", state, "pull", "--ff-only"], check=False)
    if pr.returncode != 0:
        _explain_divergence(state)
        return 1
    copied = _mirror(state, ROOT)
    if copied:
        print("  updated from companion:")
        for rel in copied:
            print(f"    {rel}")
    else:
        print("  companion has no shared files yet (nothing to copy)")
    print("Pull OK:", _run(
        ["git", "-C", state, "log", "-1", "--oneline"]).stdout.strip())
    return 0


def _explain_divergence(state: str) -> None:
    """Solo-debugger-friendly output for a non-fast-forward pull."""
    print("ERROR: companion pull --ff-only failed: this checkout and the")
    print("companion remote have diverged (usually an unpushed run).")
    local = _run(["git", "-C", state, "log", "--oneline", "-3",
                  "HEAD"], check=False).stdout.strip()
    remote = _run(["git", "-C", state, "log", "--oneline", "-3",
                   "FETCH_HEAD"], check=False).stdout.strip()
    print(f"  companion HEAD:\n    {local or '(empty)'}")
    print(f"  remote FETCH_HEAD:\n    {remote or '(empty)'}")
    stat = _run(["git", "-C", state, "diff", "--stat", "HEAD..FETCH_HEAD",
                 "--", ".", ], check=False).stdout.strip()
    if stat:
        print(f"  remote-only changes:\n{stat}")
    print("If the pipeline is NOT running anywhere, resolve with:")
    print(f"  git -C {state} pull --rebase && python scripts/sync_state.py pull")


def push(repo: str | None, dry_run: bool = False) -> int:
    """Checkpoint the ledger WAL, copy worktree -> companion, commit, push."""
    state = _resolve_repo(repo)
    if state is None:
        return 1
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

    # 2. Copy shared paths into the companion checkout
    copied = _mirror(ROOT, state)
    if not copied:
        print("  no shared files present yet (run pull first)")
        return 0

    add = _run(["git", "-C", state, "add", "--"] + copied, check=False)
    if add.returncode != 0:
        print(f"ERROR: git add failed:\n{add.stderr}")
        return 1

    staged = _run(["git", "-C", state, "diff", "--cached", "--quiet"],
                  check=False)
    if staged.returncode == 0:
        print("  no state changes to push")
        return 0

    print("  staged:")
    print(_run(["git", "-C", state, "diff", "--cached", "--stat"]
               ).stdout.strip())

    if dry_run:
        print("DRY-RUN: not committing/pushing; companion worktree restored "
              "below.")
        _run(["git", "-C", state, "reset", "-q"], check=False)
        return 0

    # 3. Commit + push (private remote only — never the public repo)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = _run(
        ["git", "-C", state, "commit", "-m", f"state: {stamp} [{_device()}]"],
        check=False,
    )
    if commit.returncode != 0:
        print(f"ERROR: git commit failed:\n{commit.stderr}")
        return 1

    push_result = _run(["git", "-C", state, "push", "origin", "HEAD"],
                       check=False)
    if push_result.returncode != 0:
        print("ERROR: git push failed (remote may have newer state).\n"
              f"{push_result.stderr}\n"
              "If another device pushed meanwhile, resolve:\n"
              "  python scripts/sync_state.py pull\n"
              "  python scripts/sync_state.py push")
        return 1

    print("Push OK:", _run(
        ["git", "-C", state, "log", "-1", "--oneline"]).stdout.strip())
    return 0


def status(repo: str | None) -> int:
    """Show shared state: worktree, companion local, companion remote."""
    print(f"Device: {_device()}")
    print(_run(["git", "status", "-sb"]).stdout.strip())
    state = _resolve_repo(repo)
    if state is None:
        print("\nShared files in this worktree (local mtime):")
        for rel in SHARED_PATHS:
            full = os.path.join(ROOT, rel)
            if os.path.exists(full):
                if os.path.isdir(full):
                    n = sum(len(f) for _, _, f in os.walk(full))
                    print(f"  {rel:32s} dir, {n} files")
                else:
                    print(f"  {rel:32s} "
                          f"{datetime.fromtimestamp(os.path.getmtime(full)):%Y-%m-%d %H:%M}")
            else:
                print(f"  {rel:32s} MISSING")
        return 0
    _run(["git", "-C", state, "fetch", "origin"], check=False)
    print(f"Companion: {state}")
    print("  local :", _run(["git", "-C", state, "log", "-1", "--oneline",
                             "HEAD"], check=False).stdout.strip() or "(empty)")
    print("  remote:", _run(["git", "-C", state, "log", "-1", "--oneline",
                             "origin/HEAD"], check=False).stdout.strip()
          or "(no remote state yet)")
    print("\nShared files (worktree vs companion mtime):")
    for rel in SHARED_PATHS:
        for label, base in (("work", ROOT), ("comp", state)):
            full = os.path.join(base, rel)
            mark = (f"{datetime.fromtimestamp(os.path.getmtime(full)):%Y-%m-%d %H:%M}"
                    if os.path.exists(full) else "MISSING")
            print(f"  {label} {rel:32s} {mark}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-device state sync")
    parser.add_argument("command", nargs="?", choices=("pull", "push", "status"),
                        help="sync direction (default: usage)")
    parser.add_argument("--repo", default=None,
                        help="companion checkout path "
                             f"(default: {DEFAULT_STATE_REPO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="push: stage and report without committing")
    args = parser.parse_args()
    if args.command == "pull":
        return pull(args.repo)
    if args.command == "push":
        return push(args.repo, dry_run=args.dry_run)
    if args.command == "status":
        return status(args.repo)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
