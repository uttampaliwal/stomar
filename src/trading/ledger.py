"""Persistent trading journal in SQLite.

Stores every decision, trade, portfolio snapshot, and outcome.
This is the dataset for meta-controller training and performance analysis.

Usage:
    ledger = Ledger()  # uses data/stomar.db by default
    decision_id = ledger.log_decision(date="2025-01-15", ticker="RELIANCE.NS", ...)
    ledger.log_trade(decision_id, ticker="RELIANCE.NS", side="BUY", ...)
    ledger.log_outcome(decision_id, actual_return=0.012, actual_direction=1)
    ledger.close()
"""

import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    ensemble_direction INTEGER,
    ensemble_confidence REAL,
    sentiment_score REAL,
    fii_net REAL,
    dii_net REAL,
    pcr REAL,
    max_pain REAL,
    mtf_signal REAL,
    regime TEXT,
    regime_confidence REAL,
    var_95 REAL,
    cvar_95 REAL,
    sharpe REAL,
    volatility_forecast REAL,
    fundamental_score REAL,
    action TEXT NOT NULL,
    position_size REAL,
    confidence REAL,
    reasoning TEXT,
    actual_return REAL,
    actual_direction INTEGER,
    correct INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER REFERENCES decisions(id),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER,
    price REAL,
    slippage REAL,
    costs REAL,
    total_cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_value REAL,
    cash REAL,
    holdings_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

INSERT INTO schema_version (version) VALUES (1);

CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(date);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON paper_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_decision ON paper_trades(decision_id);
"""

_MIGRATIONS: dict[int, str] = {
    # Add new migrations here. Each key is the target version.
    # v2: track where a decision came from ('live' vs 'backfill') so
    #     trust metrics can be computed on live data only.
    2: "ALTER TABLE decisions ADD COLUMN source TEXT NOT NULL DEFAULT 'live';",
}

SIGNAL_COLUMNS = [
    "ensemble_direction", "ensemble_confidence",
    "sentiment_score", "fii_net", "dii_net",
    "pcr", "max_pain", "mtf_signal",
    "regime", "regime_confidence",
    "var_95", "cvar_95", "sharpe",
    "volatility_forecast", "fundamental_score",
]


class Ledger:
    """Persistent trading journal in SQLite."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            from src.core.constants import LEDGER_DB
            db_path = LEDGER_DB
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout = 10000")
        self._create_tables()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _create_tables(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            self.conn.executescript(SCHEMA_V1)
        self._migrate()
        self.conn.commit()

    def _get_version(self) -> int:
        row = self.conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0

    def _migrate(self):
        current = self._get_version()
        for target in sorted(_MIGRATIONS):
            if target <= current:
                continue
            logger.info(f"Migrating ledger schema: {current} -> {target}")
            self.conn.execute("BEGIN")
            try:
                self.conn.executescript(_MIGRATIONS[target])
                self.conn.execute("UPDATE schema_version SET version = ?", (target,))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # --- Write operations ---

    def log_decision(self, date: str, ticker: str, signals: dict,
                     action: str, position_size: float, confidence: float,
                     reasoning: str = "", source: str = "live") -> int:
        """Log a meta-controller decision. Returns decision ID.

        Args:
            source: "live" for forward daily decisions, "backfill" for
                decisions reconstructed from historical data. Backfill
                decisions are in-sample by construction and must never be
                used as evidence of live performance.
        """
        if source not in ("live", "backfill"):
            raise ValueError(f"source must be 'live' or 'backfill', got {source!r}")
        cursor = self.conn.execute("""
            INSERT INTO decisions (
                date, ticker,
                ensemble_direction, ensemble_confidence,
                sentiment_score, fii_net, dii_net,
                pcr, max_pain, mtf_signal,
                regime, regime_confidence,
                var_95, cvar_95, sharpe,
                volatility_forecast, fundamental_score,
                action, position_size, confidence, reasoning, source
            ) VALUES (?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?)
        """, (
            date, ticker,
            signals.get("ensemble_direction"),
            signals.get("ensemble_confidence"),
            signals.get("sentiment_score"),
            signals.get("fii_net"),
            signals.get("dii_net"),
            signals.get("pcr"),
            signals.get("max_pain"),
            signals.get("mtf_signal"),
            signals.get("regime"),
            signals.get("regime_confidence"),
            signals.get("var_95"),
            signals.get("cvar_95"),
            signals.get("sharpe"),
            signals.get("volatility_forecast"),
            signals.get("fundamental_score"),
            action, position_size, confidence, reasoning, source,
        ))
        self.conn.commit()
        decision_id = cursor.lastrowid
        logger.info(f"Logged decision: {ticker} {action} (id={decision_id})")
        return decision_id

    def log_trade(self, decision_id: int, ticker: str, side: str,
                  quantity: int, price: float, slippage: float = 0.0,
                  costs: float = 0.0) -> None:
        """Log a paper trade execution."""
        total_cost = slippage + costs
        self.conn.execute("""
            INSERT INTO paper_trades (decision_id, ticker, side, quantity, price, slippage, costs, total_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (decision_id, ticker, side, quantity, price, slippage, costs, total_cost))
        self.conn.commit()
        logger.info(f"Logged trade: {side} {quantity} {ticker} @ {price}")

    def log_outcome(self, decision_id: int, actual_return: float,
                    actual_direction: int) -> None:
        """Log what actually happened (typically filled next day)."""
        decision = self.conn.execute(
            "SELECT action, ensemble_direction FROM decisions WHERE id = ?",
            (decision_id,)
        ).fetchone()
        if decision is None:
            logger.warning(f"Decision {decision_id} not found, skipping outcome")
            return

        action = decision["action"]
        ensemble_dir = decision["ensemble_direction"]

        # Determine if prediction was correct based on action taken
        if action == "BUY":
            correct = 1 if actual_direction == 1 else 0
        elif action == "SELL":
            correct = 1 if actual_direction == 0 else 0
        elif action == "HOLD":
            # HOLD is correct if market went sideways (return between -0.5% and +0.5%)
            correct = 1 if abs(actual_return) <= 0.005 else 0
        else:
            correct = None

        self.conn.execute("""
            UPDATE decisions SET actual_return = ?, actual_direction = ?, correct = ?
            WHERE id = ?
        """, (actual_return, actual_direction, correct, decision_id))
        self.conn.commit()

    def log_snapshot(self, date: str, total_value: float, cash: float,
                     holdings: dict) -> None:
        """Log portfolio snapshot."""
        holdings_json = json.dumps(holdings)
        self.conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots (date, total_value, cash, holdings_json)
            VALUES (?, ?, ?, ?)
        """, (date, total_value, cash, holdings_json))
        self.conn.commit()

    # --- Read operations ---

    def get_decisions(self, ticker: str = None, start_date: str = None,
                      end_date: str = None, limit: int = None,
                      source: str = None) -> list[dict]:
        """Query historical decisions.

        Args:
            source: If set, only return decisions from this source
                ("live" or "backfill"). None returns all sources.
        """
        conditions = ["1=1"]
        params: list = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        if source is not None:
            if source not in ("live", "backfill"):
                raise ValueError(f"source must be 'live' or 'backfill', got {source!r}")
            conditions.append("source = ?")
            params.append(source)
        query = "SELECT * FROM decisions WHERE " + " AND ".join(conditions)
        query += " ORDER BY date DESC, id DESC"
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise ValueError(f"limit must be >= 0, got {limit}")
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_unresolved_decisions(self, ticker: str = None) -> list[dict]:
        """Return decisions that have no logged outcome yet.

        These are typically yesterday's live decisions that need their
        next-day return resolved by the daily loop.
        """
        conditions = ["actual_return IS NULL"]
        params: list = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        query = ("SELECT * FROM decisions WHERE " + " AND ".join(conditions)
                 + " ORDER BY date ASC, id ASC")
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_trades(self, ticker: str = None) -> list[dict]:
        """Query historical trades."""
        conditions = ["1=1"]
        params: list = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        query = "SELECT * FROM paper_trades WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_snapshots(self) -> list[dict]:
        """Query portfolio snapshots."""
        rows = self.conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY date DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_performance(self, start_date: str = None,
                        end_date: str = None) -> dict:
        """Aggregate performance metrics from logged decisions."""
        conditions = ["1=1"]
        params: list = []
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        query = """
            SELECT COUNT(*) as total_decisions,
                   SUM(CASE WHEN actual_return IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                   AVG(CASE WHEN actual_return IS NOT NULL THEN actual_return ELSE NULL END) as avg_return,
                   SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buys,
                   SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sells,
                   SUM(CASE WHEN action = 'HOLD' THEN 1 ELSE 0 END) as holds,
                   SUM(CASE WHEN action IN ('BUY','SELL') AND actual_return IS NOT NULL THEN 1 ELSE 0 END) as resolved_trades,
                   SUM(CASE WHEN action IN ('BUY','SELL') AND correct = 1 THEN 1 ELSE 0 END) as correct_trades,
                   SUM(CASE WHEN source = 'live' THEN 1 ELSE 0 END) as live_decisions,
                   SUM(CASE WHEN source = 'backfill' THEN 1 ELSE 0 END) as backfill_decisions
            FROM decisions WHERE """ + " AND ".join(conditions)
        row = self.conn.execute(query, params).fetchone()
        d = dict(row)
        d["accuracy"] = (d["correct_predictions"] / d["resolved"]) if d["resolved"] and d["resolved"] > 0 else None
        # Accuracy on decisions that actually put capital at risk (BUY/SELL).
        # HOLD decisions can inflate/deflate the headline accuracy depending
        # on how flat the market was, so track them separately.
        d["trade_accuracy"] = (d["correct_trades"] / d["resolved_trades"]) if d["resolved_trades"] and d["resolved_trades"] > 0 else None
        d["hold_ratio"] = (d["holds"] / d["total_decisions"]) if d["total_decisions"] and d["total_decisions"] > 0 else None
        return d

    def get_signal_accuracy(self, source: str = None) -> dict:
        """For each signal module, compute how often it predicted correctly.

        Args:
            source: Restrict to "live" or "backfill" decisions if set.
        """
        query = ("SELECT * FROM decisions WHERE actual_direction IS NOT NULL "
                 "AND ensemble_direction IS NOT NULL")
        params: list = []
        if source is not None:
            if source not in ("live", "backfill"):
                raise ValueError(f"source must be 'live' or 'backfill', got {source!r}")
            query += " AND source = ?"
            params.append(source)
        resolved = self.conn.execute(query, params).fetchall()
        if not resolved:
            return {}

        accuracy = {}
        binary_signals = {
            "ensemble_direction": lambda r: r["ensemble_direction"] == r["actual_direction"] if r["ensemble_direction"] is not None else None,
            "sentiment": lambda r: ((r["sentiment_score"] or 0) > 0) == (r["actual_direction"] == 1),
            "fii": lambda r: ((r["fii_net"] or 0) > 0) == (r["actual_direction"] == 1),
            "dii": lambda r: ((r["dii_net"] or 0) > 0) == (r["actual_direction"] == 1),
            "pcr": lambda r: ((r["pcr"] or 0) > 1.0) == (r["actual_direction"] == 1),
            "mtf": lambda r: ((r["mtf_signal"] or 0) > 0) == (r["actual_direction"] == 1),
        }

        for name, check_fn in binary_signals.items():
            correct = 0
            total = 0
            for row in resolved:
                result = check_fn(row)
                if result is not None:
                    total += 1
                    if result:
                        correct += 1
            accuracy[name] = correct / total if total > 0 else None

        return accuracy

    def get_daily_pnl(self) -> list[dict]:
        """Daily P&L from portfolio snapshots."""
        rows = self.conn.execute("""
            SELECT date, total_value, cash
            FROM portfolio_snapshots ORDER BY date ASC
        """).fetchall()
        result = []
        prev_value = None
        for row in rows:
            d = dict(row)
            d["daily_pnl"] = (d["total_value"] - prev_value) if prev_value is not None else 0.0
            d["daily_return"] = (d["daily_pnl"] / prev_value) if prev_value and prev_value > 0 else 0.0
            prev_value = d["total_value"]
            result.append(d)
        return result

    def get_last_decision_date(self) -> str | None:
        """Return the most recent decision date, or None if ledger is empty."""
        row = self.conn.execute(
            "SELECT MAX(date) as last_date FROM decisions"
        ).fetchone()
        if row and row["last_date"]:
            return row["last_date"]
        return None

    def get_dates_with_decisions(self) -> set[str]:
        """Return set of all dates that have decisions logged."""
        rows = self.conn.execute(
            "SELECT DISTINCT date FROM decisions"
        ).fetchall()
        return {row["date"] for row in rows}

    def has_decision(self, date: str, ticker: str) -> bool:
        """Check if a decision already exists for a date+ticker pair."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM decisions WHERE date = ? AND ticker = ?",
            (date, ticker)
        ).fetchone()
        return row["cnt"] > 0

    def close(self):
        self.conn.close()
