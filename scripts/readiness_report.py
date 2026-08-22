"""Readiness report: turn the watchdog ledger into a GO/NO-GO verdict.

Evaluates every blocking requirement from docs/REAL_MONEY_READINESS.md §3:

1. models retrained on the v4 schema (no look-ahead feature)
2. live broker sandbox validated (kite_sandbox.json present + ok)
3. operator credentials & checklist acknowledged (env deliberately set —
   reported, never auto-passed)
4. supervised dry-run: >= N consecutive green sessions from the watchdog
   ledger

Usage:
    python scripts/readiness_report.py            # print + write report.json
    python scripts/readiness_report.py --json     # machine-readable only

Exit code 0 only when every blocking item passes.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
READINESS_DIR = os.path.join(PROJECT_ROOT, "data", "readiness")
PROGRESS_PATH = os.path.join(READINESS_DIR, "progress.json")
SANDBOX_PATH = os.path.join(READINESS_DIR, "kite_sandbox.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REQUIRED_SCHEMA = "4"
REQUIRED_MODEL_VERSION = "2"
REQUIRED_DAYS = 14


def check_models() -> dict:
    """Every ticker bundle must be v2/schema-4 (retrained without chikou)."""
    manifests = [f for f in os.listdir(MODELS_DIR)
                 if f.endswith("_manifest.json") and not f.startswith("meta_")]
    bad = []
    for name in sorted(manifests):
        try:
            with open(os.path.join(MODELS_DIR, name)) as f:
                m = json.load(f)
            if (str(m.get("feature_schema_version")) != REQUIRED_SCHEMA
                    or str(m.get("model_version")) != REQUIRED_MODEL_VERSION):
                bad.append(name)
        except (OSError, ValueError):
            bad.append(name)
    return {
        "ok": bool(manifests) and not bad,
        "detail": f"{len(manifests) - len(bad)}/{len(manifests)} bundles at "
                  f"v{REQUIRED_MODEL_VERSION}/schema{REQUIRED_SCHEMA}"
                  + (f"; stale: {bad[:4]}" if bad else ""),
    }


def check_sandbox() -> dict:
    """Kite sandbox validation artifact (operator runs validate script once)."""
    if not os.path.exists(SANDBOX_PATH):
        return {"ok": False,
                "detail": "not run — execute scripts/validate_kite_sandbox.py "
                          "with sandbox credentials"}
    try:
        with open(SANDBOX_PATH) as f:
            report = json.load(f)
        ok = bool(report.get("ok"))
        when = report.get("validated_at", "?")
        # Stale validation (>30 days) does not count.
        try:
            age_days = (datetime.now(timezone.utc)
                        - datetime.fromisoformat(str(when))).days
            fresh = age_days <= 30
        except ValueError:
            fresh = False
        return {"ok": ok and fresh,
                "detail": f"ok={ok} validated_at={when}"
                          + ("" if fresh else " (stale >30d)")}
    except (OSError, ValueError) as exc:
        return {"ok": False, "detail": f"unreadable: {exc}"}


def check_dry_run() -> dict:
    """Watchdog streak >= REQUIRED_DAYS with zero failures."""
    try:
        with open(PROGRESS_PATH) as f:
            progress = json.load(f)
    except (OSError, ValueError):
        return {"ok": False,
                "detail": "no watchdog ledger — start it with "
                          "'python scripts/readiness_watchdog.py'"}
    history = progress.get("history", [])
    streak = 0
    for entry in reversed(history):
        if entry.get("passed"):
            streak += 1
        else:
            break
    total = len(history)
    failed_days = sum(1 for e in history if not e.get("passed"))
    return {
        "ok": streak >= REQUIRED_DAYS,
        "detail": f"{streak}/{REQUIRED_DAYS} green sessions "
                  f"({total} recorded, {failed_days} failed)",
    }


def check_credentials_acknowledged() -> dict:
    """Reported, never auto-passed: deliberate human opt-in is §3 item 3."""
    required = ("STOMAR_LIVE_TRADING", "STOMAR_KITE_API_KEY",
                "STOMAR_KITE_ACCESS_TOKEN", "STOMAR_LIVE_ACCOUNT_APPROVED",
                "STOMAR_LIVE_CONFIRMATION")
    missing = [k for k in required if not os.environ.get(k)]
    return {
        "ok": not missing,
        "detail": ("all 5 live-gate env vars set deliberately"
                   if not missing else f"unset: {missing}"),
    }


def build_report() -> dict:
    items = {
        "1_models_v4_schema": check_models(),
        "2_kite_sandbox_validated": check_sandbox(),
        "3_operator_credentials_set": check_credentials_acknowledged(),
        "4_supervised_dry_run_window": check_dry_run(),
    }
    go = all(v["ok"] for v in items.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "GO — all blocking requirements satisfied" if go
        else "NO-GO — blocking requirements open",
        "ready_for_real_money": go,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Readiness GO/NO-GO report")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = build_report()
    os.makedirs(READINESS_DIR, exist_ok=True)
    from src.core.secure_io import atomic_write_json
    atomic_write_json(os.path.join(READINESS_DIR, "report.json"), report)

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 64)
        print("STOMAR REAL-MONEY READINESS REPORT")
        print("=" * 64)
        for name, item in report["items"].items():
            mark = "[PASS]" if item["ok"] else "[OPEN]"
            print(f"{mark} {name}: {item['detail']}")
        print("-" * 64)
        print(report["verdict"])
        print(f"full report: {os.path.join(READINESS_DIR, 'report.json')}")
        print("=" * 64)
    return 0 if report["ready_for_real_money"] else 1


if __name__ == "__main__":
    sys.exit(main())
