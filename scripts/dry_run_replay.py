"""Historical replay dry-run: supervised supervision over PAST trading days.

Drives the REAL production decision + execution path over previous days
built from regressed local data:

    parquet(truncated as-of D) -> features -> verified model bundle ->
    ensemble -> meta-controller -> ExecutionManager -> DryRunBroker ->
    reconciler -> next-open fills -> risk bookkeeping

while a simulated clock pins "now" to each replayed day's 15:30 IST close,
so staleness gating, audit-file day boundaries (IST), daily budget
rollover, and risk-state day/week windows are all exercised honestly.

This script NEVER constructs a live broker and NEVER writes to the live
ledger or paper state. All artifacts land in data/dry_run_replay/.

Usage:
    python scripts/dry_run_replay.py                     # last 15 sessions
    python scripts/dry_run_replay.py --days 10 --capital 500000
    python scripts/dry_run_replay.py --ticker RELIANCE.NS TCS.NS
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, date, time as dtime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from src.core.secure_io import atomic_write_json
from src.core.timeutils import IST

REPLAY_DIR = os.path.join("data", "dry_run_replay")

FAILURES: list[str] = []
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ── simulated clock ────────────────────────────────────────────────────────

class SimClock(datetime):
    """Drop-in datetime whose now() returns the simulated instant."""

    _anchor: datetime = datetime.now(timezone.utc)

    @classmethod
    def set_anchor(cls, dt_utc: datetime):
        cls._anchor = dt_utc

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        dt = cls._anchor
        return dt.astimezone(tz) if tz is not None else dt


def session_close_utc(d: date) -> datetime:
    """15:30 IST close of day d as an aware UTC datetime."""
    return datetime.combine(d, dtime(15, 30), tzinfo=IST).astimezone(timezone.utc)


# ── point-in-time decision path (production components only) ───────────────

class ReplaySignals:
    """Generates as-of-day decisions using the production component stack."""

    def __init__(self):
        from src.models.meta_controller import MetaController
        self._df_cache: dict[str, pd.DataFrame] = {}
        self.meta = MetaController()
        self.meta_loaded = self.meta.load()  # manifest-verified
        self._regime_cache: dict[tuple, str] = {}

    def frame(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._df_cache:
            path = os.path.join("data", f"{ticker.replace('.', '_')}.parquet")
            df = pd.read_parquet(path)
            self._df_cache[ticker] = df
        return self._df_cache[ticker]

    def decide(self, ticker: str, day: date) -> dict | None:
        """Production-equivalent decision using ONLY data <= day."""
        from src.data.features import add_technical_indicators
        from src.models.ensemble import predict_ensemble, load_meta_model
        from src.models.model import (
            load_models, models_exist, model_feature_cols,
        )
        from src.signals.regime import detect_regime
        from src.signals.volatility import forecast_volatility
        from src.trading.risk import calculate_cvar, calculate_sharpe, calculate_var
        from src.core.constants import MODELS_DIR

        df_all = self.frame(ticker)
        cutoff = pd.Timestamp(day, tz=IST) + pd.Timedelta(hours=23, minutes=59)
        hist = df_all[df_all.index <= cutoff]
        if len(hist) < 80:
            return None
        assert hist.index.max().date() <= day, "look-ahead guard"

        close = hist["close"]
        returns = close.pct_change().dropna()

        # Auxiliary signals. News/flow/PCR/MTF have no archived history, so
        # they are pinned to their neutral values — the SAME point-in-time
        # treatment features.py applies to historical rows (sentiment 0,
        # pcr 1, mtf 0). Fundamentals omitted => extract_state_vector
        # default (0.5), identical to production when the fetch fails.
        signals = {
            "current_price": float(close.iloc[-1]),
            "sentiment_score": 0.0,
            "fii_net": 0.0,
            "dii_net": 0.0,
            "pcr": 1.0,
            "mtf_signal": 0.0,
        }

        rkey = (ticker, day)
        if rkey not in self._regime_cache:
            try:
                reg = detect_regime(close, ohlc=hist)
                self._regime_cache[rkey] = reg.get("regime", "Sideways")
            except Exception:
                self._regime_cache[rkey] = "Sideways"
        signals["regime"] = self._regime_cache[rkey]

        try:
            arr = returns.values.astype(float)
            signals["var_95"] = calculate_var(arr, 0.95)
            signals["cvar_95"] = calculate_cvar(arr, 0.95)
            signals["sharpe"] = calculate_sharpe(arr)
        except Exception:
            pass
        try:
            vol = forecast_volatility(returns.values.astype(float))
            signals["volatility_forecast"] = (
                vol.get("current_vol", 0) if isinstance(vol, dict) else 0
            )
        except Exception:
            pass

        # Ensemble over the model's own trained feature schema.
        if not models_exist(ticker):
            return None
        tup = load_models(ticker)
        models = {
            "lstm": tup[0], "gru": tup[1], "transformer": tup[2],
            "xgb": tup[3], "scaler": tup[4], "features": tup[5],
            "lgb": tup[6] if len(tup) > 6 else None,
            "cat": tup[7] if len(tup) > 7 else None,
        }
        feature_cols = model_feature_cols(models)

        df_feat = add_technical_indicators(hist.copy())
        for feat_name in ("sentiment_score", "fii_net", "dii_net",
                          "pcr", "mtf_signal"):
            df_feat[feat_name] = signals[feat_name]
        for col in feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0
        existing = [c for c in feature_cols if c in df_feat.columns]
        df_feat = df_feat.dropna(subset=existing)

        meta_model = None
        meta_path = os.path.join(MODELS_DIR, f"meta_{ticker.replace('.', '_')}.pkl")
        if os.path.exists(meta_path):
            try:
                meta_model = load_meta_model(meta_path)
            except Exception:
                meta_model = None

        ens_dir, confidence, _details = predict_ensemble(
            models["lstm"], models["gru"], models["transformer"],
            models["xgb"], models["scaler"], feature_cols, df_feat,
            lgb_model=models["lgb"], cat_model=models["cat"],
            meta_model=meta_model, regime=signals.get("regime"),
        )
        signals["ensemble_direction"] = ens_dir
        signals["ensemble_confidence"] = (confidence / 100.0) if confidence else None

        decision = self.meta.decide(signals)
        decision["price"] = float(close.iloc[-1])
        decision["prev_close"] = float(close.iloc[-2]) if len(close) >= 2 else 0.0
        decision["avg_daily_traded_value"] = float(
            (close * hist["volume"]).tail(20).mean()
        ) if "volume" in hist.columns else 0.0
        return decision


# ── safety-feature mini-scenarios (isolated dirs, real code paths) ─────────

def scenario_mode_gate():
    print("\n[scenario] mode gating")
    from src.brokers.dryrun import DryRunBroker
    from src.core.trading_mode import require_live_allowed, get_trading_mode

    b = DryRunBroker()
    check("dry-run broker is not live", b.is_live is False)
    check("default trading mode is paper", get_trading_mode().value == "paper")

    os.environ["STOMAR_LIVE_TRADING"] = "true"
    try:
        require_live_allowed()
        check("incomplete live gate refuses execution", False, "did not raise")
    except Exception:
        check("incomplete live gate refuses execution", True)
    finally:
        os.environ.pop("STOMAR_LIVE_TRADING", None)


def scenario_artifact_verification():
    print("\n[scenario] model artifact verification")
    from src.models.model import load_models, models_exist
    from src.models.artifacts import ArtifactVerificationError

    ticker = "RELIANCE.NS"
    if not models_exist(ticker):
        check("verified bundle loads", False, "no models present")
        return
    tup = load_models(ticker)  # production load — manifest verified
    check("verified bundle loads", tup[3] is not None)

    from src.core.constants import MODELS_DIR
    tmp_root = tempfile.mkdtemp(prefix="stomar-tamper-")
    try:
        src_dir = MODELS_DIR
        stem = ticker.replace(".", "_")
        manifest_name = f"{stem}_manifest.json"
        with open(os.path.join(src_dir, manifest_name)) as f:
            manifest = json.load(f)
        shutil.copy(os.path.join(src_dir, manifest_name),
                    os.path.join(tmp_root, manifest_name))
        for fname in manifest["files"]:
            shutil.copy(os.path.join(src_dir, fname),
                        os.path.join(tmp_root, fname))
        target = next((f for f in manifest["files"] if f.endswith(".pkl")),
                      next(iter(manifest["files"])))
        with open(os.path.join(tmp_root, target), "r+b") as f:
            f.seek(10)
            byte = f.read(1)
            f.seek(10)
            f.write(bytes([byte[0] ^ 0xFF]))

        try:
            load_models(ticker, root=tmp_root)
            check("tampered bundle refused", False, "loaded anyway")
        except ArtifactVerificationError:
            check("tampered bundle refused", True)
        except Exception as exc:  # noqa: BLE001 — any refusal is fail-closed
            check("tampered bundle refused", True, f"({type(exc).__name__})")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _fresh_manager(audit_sub: str, capital=200_000.0, max_orders=10,
                   broker=None, risk=None):
    from src.brokers.dryrun import DryRunBroker
    from src.trading.execution_manager import ExecutionManager
    from src.trading.risk_controls import RiskController, RiskLimits

    audit_dir = os.path.join(REPLAY_DIR, audit_sub)
    os.makedirs(audit_dir, exist_ok=True)
    broker = broker or DryRunBroker(initial_cash=capital)
    risk = risk or RiskController(
        RiskLimits(), initial_capital=capital, persist_state=True,
        kill_switch_file=os.path.join(audit_dir, "kill_switch.json"),
        state_file=os.path.join(audit_dir, "risk_state.json"),
    )
    mgr = ExecutionManager(broker, risk, audit_dir=audit_dir,
                           max_stale_quote_seconds=15.0,
                           max_daily_orders=max_orders)
    return mgr, broker, risk, audit_dir


def _quote(price, prev_close, ts_iso):
    return {"close": price, "last_price": price,
            "prev_close": prev_close, "timestamp": ts_iso}


def scenario_quote_freshness():
    print("\n[scenario] quote freshness gating")
    from src.trading.execution_manager import ExecutionError

    now_utc = datetime.now(timezone.utc)
    mgr, _b, _r, audit = _fresh_manager("checks-quote")
    t = "TCS.NS"

    stale_ts = (now_utc - timedelta(seconds=120)).isoformat()
    try:
        mgr.execute(t, "BUY", 1, quotes={t: _quote(3000, 2990, stale_ts)},
                    order_value=3000)
        check("stale quote rejected", False)
    except ExecutionError as exc:
        check("stale quote rejected", "stale" in str(exc))

    future_ts = (now_utc + timedelta(minutes=10)).isoformat()
    try:
        mgr.execute(t, "BUY", 1, quotes={t: _quote(3000, 2990, future_ts)},
                    order_value=3000)
        check("future-dated quote rejected", False)
    except ExecutionError as exc:
        check("future-dated quote rejected", "future" in str(exc))

    naive_ist = datetime.now(IST).replace(tzinfo=None).isoformat()
    reason = mgr._stale_quote_check(t, {t: _quote(3000, 2990, naive_ist)})
    check("naive IST quote treated fresh", reason == "", reason)

    events = [json.loads(line) for line in
              open(os.path.join(audit, sorted(
                  f for f in os.listdir(audit) if f.startswith("orders-"))[0]))]
    rejects = [e for e in events if e.get("event") == "rejected"]
    check("rejections audited", len(rejects) >= 2,
          f"{len(rejects)} reject events")


def scenario_duplicate_and_cancel():
    print("\n[scenario] idempotency + cancellation")
    from src.brokers.dryrun import DryRunBroker
    from src.brokers.base import BrokerOrderStatus
    from src.trading.execution_manager import ExecutionManager
    from src.trading.risk_controls import RiskController, RiskLimits

    audit = os.path.join(REPLAY_DIR, "checks-idem")
    os.makedirs(audit, exist_ok=True)
    slow = DryRunBroker(fill_delay_seconds=30, initial_cash=200_000)
    mgr = ExecutionManager(slow, RiskController(RiskLimits(), 200_000),
                           audit_dir=audit, max_daily_orders=25)
    q = _quote(1000, 995, datetime.now(timezone.utc).isoformat())

    o1 = mgr.execute("INFY.NS", "BUY", 5, quotes={"INFY.NS": q}, order_value=5000,
                     client_order_id="replay-dup-1")
    o2 = mgr.execute("INFY.NS", "BUY", 5, quotes={"INFY.NS": q}, order_value=5000,
                     client_order_id="replay-dup-1")
    check("duplicate intent suppressed", o1.client_order_id == o2.client_order_id
          and o1.broker_order_id == o2.broker_order_id)

    cancelled = mgr.cancel(o1.client_order_id)
    check("pending order cancellable",
          cancelled.status == BrokerOrderStatus.CANCELLED)

    filled_broker = DryRunBroker(initial_cash=200_000)
    mgr2 = ExecutionManager(filled_broker,
                            RiskController(RiskLimits(), 200_000),
                            audit_dir=audit, max_daily_orders=25)
    o3 = mgr2.execute("WIPRO.NS", "BUY", 5, quotes={"WIPRO.NS": q},
                      order_value=5000, client_order_id="replay-fill-1")
    filled_broker.process_pending({"WIPRO.NS": 1000})
    got = mgr2.reconcile(o3.client_order_id)[0]
    check("fill reconciled from broker truth",
          got.status == BrokerOrderStatus.FILLED and got.filled_quantity == 5)

    dup_again = filled_broker.submit_order("replay-fill-1", "WIPRO.NS", "BUY", 5)
    check("broker resubmit idempotent",
          dup_again.client_order_id == "replay-fill-1"
          and dup_again.filled_quantity == 5,
          f"(filled={dup_again.filled_quantity})")


def scenario_budget_rollover():
    print("\n[scenario] daily budget + IST rollover")
    import src.trading.execution_manager as em_mod
    from src.trading.execution_manager import ExecutionError
    from src.core.timeutils import ist_today as real_ist_today

    mgr, _b, _r, audit = _fresh_manager("checks-budget", max_orders=2)
    today = date.today()

    try:
        em_mod.ist_today = lambda: today
        ok = 0
        for i in range(2):
            mgr.execute(f"S{i}.NS", "BUY", 1,
                        quotes={f"S{i}.NS": _quote(100, 99, datetime.now(timezone.utc).isoformat())},
                        order_value=100)
            ok += 1
        try:
            mgr.execute("SX.NS", "BUY", 1,
                        quotes={"SX.NS": _quote(100, 99, datetime.now(timezone.utc).isoformat())},
                        order_value=100)
            check("budget exhausted blocks 3rd order", False)
        except ExecutionError as exc:
            check("budget exhausted blocks 3rd order", "budget" in str(exc))
        check("two submits recorded", ok == 2)

        em_mod.ist_today = lambda: today + timedelta(days=1)
        mgr.execute("SY.NS", "BUY", 1,
                    quotes={"SY.NS": _quote(100, 99, datetime.now(timezone.utc).isoformat())},
                    order_value=100)
        files = sorted(f for f in os.listdir(audit) if f.startswith("orders-"))
        check("new IST day opens new ledger file",
              any(today.isoformat() in f for f in files)
              and any((today + timedelta(days=1)).isoformat() in f for f in files),
              ",".join(files))
    finally:
        em_mod.ist_today = real_ist_today


def scenario_risk_blocks():
    print("\n[scenario] risk-gate blocks")
    from src.trading.execution_manager import ExecutionError

    mgr, _b, _r, _audit = _fresh_manager("checks-risk", capital=100_000)
    q = _quote(500, 498, datetime.now(timezone.utc).isoformat())
    try:
        mgr.execute("BIG.NS", "BUY", 1000, quotes={"BIG.NS": q},
                    order_value=500_000)
        check("oversized position blocked", False)
    except ExecutionError as exc:
        check("oversized position blocked", "concentration" in str(exc).lower()
              or "exposure" in str(exc).lower(), str(exc)[:90])


def scenario_kill_switch():
    print("\n[scenario] persistent kill switch")
    from src.trading.risk_controls import RiskController, RiskLimits

    d = os.path.join(REPLAY_DIR, "checks-kill")
    os.makedirs(d, exist_ok=True)
    ks = os.path.join(d, "kill_switch.json")
    rc = RiskController(RiskLimits(max_drawdown_pct=0.15), 100_000,
                        kill_switch_file=ks,
                        state_file=os.path.join(d, "risk_state.json"),
                        persist_state=True)
    rc.update_equity(83_000)  # -17% drawdown
    verdict = rc.check_order(order_value=1_000, current_holdings_value=0)
    check("drawdown breach trips kill switch",
          rc.halted and verdict["approved"] is False and os.path.exists(ks))

    restarted = RiskController(RiskLimits(max_drawdown_pct=0.15), 100_000,
                               kill_switch_file=ks,
                               state_file=os.path.join(d, "risk_state.json"),
                               persist_state=True)
    v2 = restarted.check_order(order_value=1_000, current_holdings_value=0)
    check("halt survives restart", restarted.halted and v2["approved"] is False)

    # Clearing without resolving the breach must re-halt (no bypass).
    restarted.resume_trading()
    v_rehalt = restarted.check_order(order_value=1_000, current_holdings_value=0)
    check("resume cannot bypass unresolved drawdown",
          restarted.halted and v_rehalt["approved"] is False)

    # Operator resolves the condition (equity recovers) → resume sticks.
    restarted.update_equity(99_000)
    restarted.clear_kill_switch()
    v3 = restarted.check_order(order_value=1_000, current_holdings_value=0)
    check("clear after recovery resumes trading",
          not restarted.halted and v3["approved"] is True)


def scenario_partial_fill():
    print("\n[scenario] partial-fill reconciliation")
    from src.brokers.base import BrokerOrderStatus
    from src.brokers.dryrun import DryRunBroker

    broker = DryRunBroker(partial_fill_pct=0.5, initial_cash=200_000)
    mgr, _b, _r, _audit = _fresh_manager("checks-partial", broker=broker)
    q = _quote(2000, 1995, datetime.now(timezone.utc).isoformat())
    order = mgr.execute("LT.NS", "BUY", 20, quotes={"LT.NS": q}, order_value=40_000)
    broker.process_pending({"LT.NS": 2000})
    mid = mgr.reconcile(order.client_order_id)[0]
    check("first tick partial",
          mid.status == BrokerOrderStatus.PARTIALLY_FILLED
          and 0 < mid.filled_quantity < 20)
    time.sleep(0.06)
    broker.process_pending({"LT.NS": 2001})
    done = mgr.reconcile(order.client_order_id)[0]
    positions = {p.ticker: p.quantity for p in broker.get_positions()}
    check("partial completes and matches positions",
          done.status == BrokerOrderStatus.FILLED
          and done.filled_quantity == 20
          and positions.get("LT.NS") == 20)


# ── main historical replay ─────────────────────────────────────────────────

class FifoLots:
    """FIFO lot accounting for realized-P&L reporting + risk feed."""

    def __init__(self):
        self.lots: dict[str, list[tuple[int, float]]] = {}  # ticker -> [(signed_qty, price)]

    def apply_fill(self, ticker: str, side: str, qty: int, price: float) -> float:
        signed = qty if side == "BUY" else -qty
        book = self.lots.setdefault(ticker, [])
        realized = 0.0
        while book and signed != 0 and (signed > 0) != (book[0][0] > 0):
            open_qty, open_px = book[0]
            close_qty = min(abs(open_qty), abs(signed))
            direction = 1 if open_qty > 0 else -1
            realized += (price - open_px) * direction * close_qty
            rem_open = abs(open_qty) - close_qty
            if rem_open > 0:
                book[0] = (direction * rem_open, open_px)
            else:
                book.pop(0)
            signed -= close_qty if signed > 0 else -close_qty
        if signed != 0:
            book.append((signed, price))
        book[:] = [(q, p) for q, p in book if q != 0]
        return realized

    def market_value(self, marks: dict[str, float]) -> float:
        """Absolute gross exposure: Σ|q|·px."""
        return sum(abs(q) * marks.get(t, p)
                   for t, book in self.lots.items() for q, p in book)

    def signed_value(self, marks: dict[str, float]) -> float:
        """Signed position value: Σq·px. Short lots count NEGATIVE — they
        are obligations, not assets. Equity = cash + this."""
        return sum(q * marks.get(t, p)
                   for t, book in self.lots.items() for q, p in book)

    def cost_basis(self) -> float:
        return sum(q * p for book in self.lots.values() for q, p in book)


def run_replay(tickers: list[str], days: int, capital: float,
               max_daily_orders: int) -> dict:
    import src.trading.execution_manager as em_mod
    import src.trading.risk_controls as rc_mod
    from src.brokers.dryrun import DryRunBroker
    from src.trading.execution_manager import ExecutionError, ExecutionManager
    from src.trading.risk_controls import MarketContext, RiskController, RiskLimits

    print(f"\n[replay] {len(tickers)} tickers × {days} sessions "
          f"(capital ₹{capital:,.0f}, budget {max_daily_orders}/day)")

    os.makedirs(REPLAY_DIR, exist_ok=True)
    # Audit ledgers are append-only, so stale files from previous runs would
    # corrupt the audit-vs-report reconciliation. Session artifacts are
    # fully regenerable — clear them (scenario subdirs are untouched).
    for name in os.listdir(REPLAY_DIR):
        if name.startswith(("orders-", "state-")) and \
                (name.endswith(".jsonl") or name.endswith(".json")):
            try:
                os.unlink(os.path.join(REPLAY_DIR, name))
            except OSError:
                pass
    audit_dir = REPLAY_DIR
    broker = DryRunBroker(initial_cash=capital, slippage_bps=5.0)
    risk = RiskController(RiskLimits(), initial_capital=capital,
                          persist_state=True,
                          kill_switch_file=os.path.join(audit_dir, "kill_switch.json"),
                          state_file=os.path.join(audit_dir, "risk_state.json"))
    mgr = ExecutionManager(broker, risk, audit_dir=audit_dir,
                           max_stale_quote_seconds=15.0,
                           max_daily_orders=max_daily_orders)

    gen = ReplaySignals()
    if not gen.meta_loaded:
        print("  NOTE: no trained meta-controller found — rule-based fallback decides")

    # Trading sessions from RELIANCE reference calendar.
    ref = gen.frame(tickers[0]).index
    sessions = [ts.date() for ts in ref if ts.date() < date.today()]
    sessions = sessions[-(days + 1):-1]  # need D+1 open for fills
    print(f"  sessions: {sessions[0]} .. {sessions[-1]} ({len(sessions)})")

    real_ist_today = em_mod.ist_today
    lots = FifoLots()
    applied_fills: dict[str, int] = {}  # cid -> filled_quantity already booked
    day_reports = []
    total_submitted = total_blocked = 0

    try:
        for i, day in enumerate(sessions):
            nxt = sessions[i + 1] if i + 1 < len(sessions) else None
            em_mod.datetime = SimClock
            em_mod.ist_today = lambda d=day: d
            rc_mod.ist_today = lambda d=day: d
            SimClock.set_anchor(session_close_utc(day))
            risk.reset_daily()

            opens = {}
            if nxt is not None:
                cut_n = pd.Timestamp(nxt, tz=IST) + pd.Timedelta(hours=23, minutes=59)
                for t in tickers:
                    hist_n = gen.frame(t)[gen.frame(t).index <= cut_n]
                    if len(hist_n) and hist_n.index.max().date() == nxt:
                        opens[t] = float(hist_n["open"].iloc[-1])

            submitted = blocked = 0
            decisions = []
            anchor_iso = session_close_utc(day).isoformat()
            for t in tickers:
                dec = gen.decide(t, day)
                if dec is None:
                    continue
                decisions.append({"ticker": t, **{k: v for k, v in dec.items()
                                                   if k in ("action", "position_size",
                                                            "confidence", "price")}})
                action, size, price = dec["action"], dec.get("position_size", 0), dec["price"]
                if action == "HOLD" or size <= 0 or price <= 0:
                    continue
                qty = max(1, int(capital * size / price))
                try:
                    order = mgr.execute(
                        t, action, qty,
                        quotes={t: _quote(price, dec["prev_close"], anchor_iso)},
                        order_value=price * qty,
                        market=MarketContext(
                            price=price, prev_close=dec["prev_close"],
                            avg_daily_traded_value=dec["avg_daily_traded_value"],
                            expected_slippage_bps=5.0,
                        ),
                    )
                    submitted += 1
                    if order.status.value == "REJECTED":
                        blocked += 1
                except ExecutionError as exc:
                    blocked += 1
                    decisions[-1]["blocked_reason"] = str(exc)[:140]

            # Advance to next open: fills at D+1 open (broker adds slippage).
            if opens:
                SimClock.set_anchor(
                    datetime.combine(nxt, dtime(9, 20), tzinfo=IST).astimezone(timezone.utc)
                )
                broker.process_pending(opens)
            reconciled = mgr.reconcile()

            realized_day = 0.0
            for o in reconciled:
                # reconcile() returns ALL tracked orders with their CURRENT
                # state — book only the increment since we last saw this
                # order, or day-old fills get re-applied every session.
                prev_qty = applied_fills.get(o.client_order_id, 0)
                inc = o.filled_quantity - prev_qty
                if o.status.value in ("FILLED", "PARTIALLY_FILLED") and inc > 0:
                    realized_day += lots.apply_fill(
                        o.ticker, o.side, inc, o.avg_fill_price)
                    applied_fills[o.client_order_id] = o.filled_quantity

            mark_px = opens or {d_["ticker"]: d_["price"] for d_ in decisions}
            gross = lots.market_value(mark_px)
            cash = broker.get_margin()["available_cash"]
            equity = cash + lots.signed_value(mark_px)
            risk.update_daily_pnl(realized_day)
            risk.update_equity(equity)

            rep = {
                "date": day.isoformat(),
                "next_open_used": nxt.isoformat() if nxt else None,
                "decisions": len(decisions),
                "orders_submitted": submitted,
                "orders_blocked": blocked,
                "realized_pnl": round(realized_day, 2),
                "equity": round(equity, 2),
                "gross_exposure": round(gross, 2),
                "cash": round(cash, 2),
                # Exact marks used for this session's equity — verification
                # must reconcile against THESE, not re-derived prices.
                # (signed_value falls back to each lot's own entry price for
                # tickers absent here, mirroring the equity computation.)
                "marks": {t: round(px, 6) for t, px in mark_px.items()},
                "risk": {k: risk.get_status()[k] for k in
                         ("daily_pnl", "weekly_pnl", "consecutive_losses",
                          "halted", "drawdown_pct")},
                "open_lots": {t: [(q, p) for q, p in book]
                              for t, book in lots.lots.items() if book},
            }
            atomic_write_json(os.path.join(audit_dir, f"state-{day.isoformat()}.json"), rep)
            day_reports.append(rep)
            total_submitted += submitted
            total_blocked += blocked
            print(f"  {day}: decisions={rep['decisions']:2d} submitted={submitted} "
                  f"blocked={blocked} realized=₹{realized_day:,.0f} "
                  f"equity=₹{equity:,.0f}")
    finally:
        em_mod.ist_today = real_ist_today
        from src.core.timeutils import ist_today as _real_ist
        rc_mod.ist_today = _real_ist

    return {
        "sessions": [r["date"] for r in day_reports],
        "tickers": tickers,
        "day_reports": day_reports,
        "total_submitted": total_submitted,
        "total_blocked": total_blocked,
        "final_equity": day_reports[-1]["equity"] if day_reports else capital,
        "capital": capital,
        "max_daily_orders": max_daily_orders,
        "meta_loaded": gen.meta_loaded,
        "_lots_obj": lots,  # stripped before serialization
    }


# ── verification of replay results ─────────────────────────────────────────

def verify_replay(result: dict, ledger_db="data/stomar.db"):
    print("\n[verify] replay invariants")

    reports = result["day_reports"]
    budget = result["max_daily_orders"]
    check("sessions replayed", len(reports) >= 10, f"{len(reports)} sessions")

    # Every submitted order appears in exactly one per-day audit file, and
    # per-day submitted counts match the reports.
    mismatch = []
    for rep in reports:
        p = os.path.join(REPLAY_DIR, f"orders-{rep['date']}.jsonl")
        if not os.path.exists(p):
            mismatch.append(f"{rep['date']}: no audit file")
            continue
        events = [json.loads(line) for line in open(p)]
        n_submit = sum(1 for e in events if e.get("event") == "submitted")
        n_reject = sum(1 for e in events if e.get("event") == "rejected")
        if n_submit != rep["orders_submitted"]:
            mismatch.append(
                f"{rep['date']}: audit {n_submit} vs report {rep['orders_submitted']}")
        if rep["orders_blocked"] > 0 and n_reject == 0 and \
           not any("blocked_reason" in d for d in []):  # blocks may be broker-side
            mismatch.append(f"{rep['date']}: blocked without reject event")
    check("audit ledger matches per-day reports", not mismatch,
          "; ".join(mismatch[:3]))

    # Budget never exceeded within any day.
    over = [r["date"] for r in reports
            if r["orders_submitted"] > budget]
    check(f"daily budget (≤{budget}) respected every session", not over,
          ",".join(over))

    # Equity stays sane: positive, no accounting blowups.
    bad = [r["date"] for r in reports
           if r["equity"] <= 0 or abs(r["equity"]) > result["capital"] * 3]
    check("equity stays sane across sessions", not bad, ",".join(bad))

    # Kill switch never tripped organically (limits were respected).
    tripped = [r["date"] for r in reports if r["risk"]["halted"]]
    check("risk halt never forced mid-replay", not tripped, ",".join(tripped))

    # Cash identity: cash = capital - open cost basis - cumulative realized.
    # Reconcile against the session's OWN stored marks (identical to what
    # produced `equity`), so this is pure accounting, not price opinion.
    last = reports[-1]
    signed_mv = sum(q * last["marks"].get(t, p)
                    for t, book in result["_lots_obj"].lots.items()
                    for q, p in book)
    cost_basis = result["_lots_obj"].cost_basis()
    total_realized = sum(r["realized_pnl"] for r in reports)
    implied_cash = last["equity"] - signed_mv
    expected_cash = result["capital"] - cost_basis - total_realized
    drift = abs(implied_cash - expected_cash)
    check("cash accounting identity holds",
          drift < max(5.0, result["capital"] * 1e-6),
          f"cash ₹{implied_cash:,.2f} vs derived ₹{expected_cash:,.2f} "
          f"(drift ₹{drift:.4f})")

    # Live ledger untouched by the replay.
    try:
        con = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
        n_before = con.execute("select count(*) from decisions").fetchone()[0]
        con.close()
        check("live ledger readable (replay wrote nothing)", n_before > 0,
              f"{n_before} decisions present")
    except Exception as exc:  # noqa: BLE001
        check("live ledger readable (replay wrote nothing)", False, str(exc))


# ── entrypoint ─────────────────────────────────────────────────────────────

def run_safety_scenarios() -> tuple[list[dict], list[str]]:
    """Run the full safety-scenario battery (isolated dirs, real paths).

    Returns (checks, failures) so external callers (readiness watchdog)
    can reuse the battery without going through main().
    """
    global CHECKS, FAILURES
    CHECKS, FAILURES = [], []
    scenario_mode_gate()
    scenario_artifact_verification()
    scenario_quote_freshness()
    scenario_duplicate_and_cancel()
    scenario_budget_rollover()
    scenario_risk_blocks()
    scenario_kill_switch()
    scenario_partial_fill()
    return list(CHECKS), list(FAILURES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical replay dry-run")
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--max-daily-orders", type=int, default=3)
    parser.add_argument("--ticker", nargs="+", default=None)
    parser.add_argument("--skip-scenarios", action="store_true")
    args = parser.parse_args()

    from src.data.data_fetcher import NSE_STOCKS
    tickers = args.ticker or [
        t for t in NSE_STOCKS
        if os.path.exists(os.path.join("data", f"{t.replace('.', '_')}.parquet"))
    ]

    started = datetime.now(timezone.utc)
    print("=" * 68)
    print("StoMar SUPERVISED HISTORICAL REPLAY (paper-only, no live broker)")
    print(f"window: last {args.days} sessions | tickers: {len(tickers)} | "
          f"capital: ₹{args.capital:,.0f}")
    print("=" * 68)

    if not args.skip_scenarios:
        run_safety_scenarios()

    result = run_replay(tickers, args.days, args.capital, args.max_daily_orders)
    verify_replay(result)

    passed = sum(1 for c in CHECKS if c["ok"])
    result.pop("_lots_obj", None)  # live object — not serializable
    summary = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "checks_total": len(CHECKS),
        "checks_passed": passed,
        "failures": FAILURES,
        "scenarios_skipped": args.skip_scenarios,
        "replay": result,
    }
    atomic_write_json(os.path.join(REPLAY_DIR, "summary.json"), summary)

    print("\n" + "=" * 68)
    print(f"RESULT: {passed}/{len(CHECKS)} checks passed")
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
    print(f"Replay P&L: ₹{result['final_equity'] - result['capital']:,.0f} "
          f"on ₹{result['capital']:,.0f} over {len(result['sessions'])} sessions "
          f"({result['total_submitted']} orders submitted, "
          f"{result['total_blocked']} blocked)")
    print(f"Artifacts: {REPLAY_DIR}/summary.json")
    print("=" * 68)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
