"""Live-trading readiness gate (P3.5).

No amount of code makes the system trustworthy — only forward evidence
does. This module encodes the hard gates that must pass before any real
money is involved:

    1. paper_period   — >= 60 business days of paper trading snapshots
    2. accuracy       — rolling 30-day BUY/SELL accuracy >= 51%
                        (HOLD decisions excluded; they put no capital at risk)
    3. drawdown       — max drawdown across the paper period <= 15%
    4. drift          — no critical drift alert unresolved in the last 7 days
    5. models_fresh   — all tickers have a model trained within 30 days

Backfill decisions are never evidence: they are in-sample by construction
(see src/core/backfill.py) and are excluded from every metric here.

Usage:
    from src.core.readiness_check import check_readiness, is_ready_for_live_trading
    report = check_readiness()
    if report["ready"]:
        print("All gates passed")
"""

import glob
import json
import logging
import os
from datetime import datetime, timedelta

from src.core.constants import MODELS_DIR, MONITORING_DIR
from src.trading.ledger import Ledger

logger = logging.getLogger(__name__)

MIN_PAPER_BUSINESS_DAYS = 60
MIN_ROLLING_ACCURACY = 0.51
MAX_DRAWDOWN = 0.15
MAX_MODEL_AGE_DAYS = 30
DRIFT_LOOKBACK_DAYS = 7
TRADE_ACTIONS = ("BUY", "SELL")


def _load_json(path: str) -> list | dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _recent_critical_alerts(days: int = DRIFT_LOOKBACK_DAYS) -> list[dict]:
    """Return critical alerts from *_alerts.json files within the window."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    alerts: list[dict] = []
    if not os.path.isdir(MONITORING_DIR):
        return alerts
    for path in glob.glob(os.path.join(MONITORING_DIR, "*_alerts.json")):
        data = _load_json(path)
        if not isinstance(data, list):
            continue
        for a in data:
            if a.get("severity") == "critical" and a.get("timestamp", "") > cutoff:
                alerts.append({"ticker": os.path.basename(path), **a})
    return alerts


def _paper_days_from_snapshots(ledger: Ledger) -> list[str]:
    """Distinct business-day dates with portfolio snapshots."""
    snapshots = ledger.get_snapshots()
    if not snapshots:
        return []
    days = []
    seen = set()
    for s in snapshots:
        d = str(s.get("date", ""))[:10]
        if d and d not in seen:
            seen.add(d)
            days.append(d)
    # Snapshots come back DESC — chronological for calculations
    return list(reversed(sorted(days)))


def _max_drawdown_from_equity(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def check_readiness(ledger: Ledger = None, tickers: list[str] = None) -> dict:
    """Evaluate all live-trading readiness gates.

    Args:
        ledger: Open ledger. If None, opens and closes one internally.
        tickers: Universe to check model freshness for. If None, uses
            the full NSE universe from the data fetcher.

    Returns:
        Report dict: {"ready": bool, "gates": {name: {passed, detail}},
                      "evidence": {...}, "summary": str}
    """
    close_after = ledger is None
    ledger = ledger or Ledger()

    if tickers is None:
        try:
            from src.data.data_fetcher import NSE_STOCKS
            tickers = NSE_STOCKS
        except Exception:
            tickers = []

    report: dict = {"ready": False, "gates": {}, "evidence": {}}

    try:
        # ── Gate 1: paper trading period ──────────────────────────────────
        paper_days = _paper_days_from_snapshots(ledger)
        # Also count live decisions — the ledger is the primary source of truth
        live_decisions = ledger.get_decisions(source="live")
        live_days = sorted({str(d["date"])[:10] for d in live_decisions})
        all_days = sorted(set(paper_days) | set(live_days))
        period_ok = len(all_days) >= MIN_PAPER_BUSINESS_DAYS
        report["gates"]["paper_period"] = {
            "passed": period_ok,
            "detail": f"{len(all_days)}/{MIN_PAPER_BUSINESS_DAYS} trading days "
                      f"(snapshots: {len(paper_days)}, live decisions: {len(live_days)})",
        }
        report["evidence"]["paper_days"] = len(all_days)

        # ── Gate 2: rolling 30-day trade accuracy (BUY/SELL only) ─────────
        resolved_live = [d for d in live_decisions
                         if d.get("actual_return") is not None]
        resolved_live.sort(key=lambda d: str(d["date"]))
        recent = resolved_live[-30:]
        trades = [d for d in recent if d.get("action") in TRADE_ACTIONS]
        n_trades = len(trades)
        n_correct = sum(1 for d in trades if d.get("correct"))
        acc = (n_correct / n_trades) if n_trades else None
        acc_ok = acc is not None and acc >= MIN_ROLLING_ACCURACY
        report["gates"]["accuracy"] = {
            "passed": acc_ok,
            "detail": (f"rolling BUY/SELL accuracy {acc:.1%} ({n_correct}/{n_trades})"
                       if acc is not None
                       else f"no resolved live trades yet ({len(resolved_live)} resolved)"),
        }
        report["evidence"]["rolling_accuracy"] = acc

        # ── Gate 3: max drawdown from paper snapshots ─────────────────────
        snapshots = ledger.get_snapshots()
        snapshots.sort(key=lambda s: str(s["date"]))
        equity_curve = [float(s.get("total_value") or 0) for s in snapshots
                        if s.get("total_value") is not None]
        max_dd = _max_drawdown_from_equity(equity_curve) if equity_curve else 0.0
        dd_ok = max_dd <= MAX_DRAWDOWN
        report["gates"]["drawdown"] = {
            "passed": dd_ok,
            "detail": (f"max drawdown {max_dd:.1%} (limit {MAX_DRAWDOWN:.0%})"
                       if equity_curve else "no snapshots yet"),
        }
        report["evidence"]["max_drawdown"] = max_dd

        # ── Gate 4: unresolved critical drift alerts ──────────────────────
        alerts = _recent_critical_alerts()
        drift_ok = len(alerts) == 0
        report["gates"]["drift"] = {
            "passed": drift_ok,
            "detail": f"{len(alerts)} critical alert(s) in last {DRIFT_LOOKBACK_DAYS} days"
                      if alerts else "no critical drift alerts",
        }
        report["evidence"]["critical_alerts"] = len(alerts)

        # ── Gate 5: model freshness ────────────────────────────────────────
        stale_tickers = []
        for ticker in tickers:
            safe = ticker.replace(".", "_")
            # Models are individual artifacts ({safe}_lstm.pt, {safe}_xgb.pkl,
            # ...) gated by a SHA-256 manifest; the manifest mtime is the
            # authoritative "last written" time for the bundle.
            path = os.path.join(MODELS_DIR, f"{safe}_manifest.json")
            if not os.path.exists(path):
                stale_tickers.append(ticker)
                continue
            age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
            if age > MAX_MODEL_AGE_DAYS:
                stale_tickers.append(f"{ticker} ({age}d)")
        models_ok = not stale_tickers and len(tickers) > 0
        report["gates"]["models_fresh"] = {
            "passed": models_ok,
            "detail": (f"all {len(tickers)} models fresh (<= {MAX_MODEL_AGE_DAYS} days)"
                       if models_ok
                       else f"stale/missing models: {', '.join(stale_tickers[:5])}"
                       + (f" (+{len(stale_tickers) - 5} more)" if len(stale_tickers) > 5 else "")),
        }
        report["evidence"]["stale_tickers"] = stale_tickers

        report["ready"] = all(g["passed"] for g in report["gates"].values())
        passed = sum(1 for g in report["gates"].values() if g["passed"])
        report["summary"] = (f"{passed}/{len(report['gates'])} readiness gates passed"
                             if report["ready"]
                             else f"{passed}/{len(report['gates'])} readiness gates passed — not ready for live trading")
        report["checked_at"] = datetime.now().isoformat()
        return report
    finally:
        if close_after:
            ledger.close()


def is_ready_for_live_trading() -> bool:
    """Convenience wrapper: True only if every gate passes."""
    return check_readiness()["ready"]


def format_report(report: dict) -> str:
    """Render the readiness report as human-readable text."""
    lines = [report.get("summary", ""), ""]
    for name, gate in report.get("gates", {}).items():
        status = "PASS" if gate.get("passed") else "FAIL"
        lines.append(f"[{status}] {name}: {gate.get('detail', '')}")
    return "\n".join(lines)
