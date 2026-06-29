"""Persistent trading journal in SQLite.

Stores every decision, trade, portfolio snapshot, and outcome.
This is the dataset for meta-controller training and performance analysis.

Usage:
    ledger = Ledger("stomar.db")
    decision_id = ledger.log_decision(date="2025-01-15", ticker="RELIANCE.NS", ...)
    ledger.log_trade(decision_id, ticker="RELIANCE.NS", side="BUY", ...)
    ledger.log_outcome(decision_id, actual_return=0.012, actual_direction=1)
    ledger.close()
"""

import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA = """
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

CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(date);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON paper_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_decision ON paper_trades(decision_id);
"""

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

    def __init__(self, db_path: str = "stomar.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- Write operations ---

    def log_decision(self, date: str, ticker: str, signals: dict,
                     action: str, position_size: float, confidence: float,
                     reasoning: str = "") -> int:
        """Log a meta-controller decision. Returns decision ID."""
        cursor = self.conn.execute("""
            INSERT INTO decisions (
                date, ticker,
                ensemble_direction, ensemble_confidence,
                sentiment_score, fii_net, dii_net,
                pcr, max_pain, mtf_signal,
                regime, regime_confidence,
                var_95, cvar_95, sharpe,
                volatility_forecast, fundamental_score,
                action, position_size, confidence, reasoning
            ) VALUES (?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?)
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
            action, position_size, confidence, reasoning,
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
        elif ensemble_dir is not None:
            # HOLD: correct if ensemble matched reality
            correct = 1 if ensemble_dir == actual_direction else 0
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
                      end_date: str = None) -> list[dict]:
        """Query historical decisions."""
        query = "SELECT * FROM decisions WHERE 1=1"
        params = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date DESC, id DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_trades(self, ticker: str = None) -> list[dict]:
        """Query historical trades."""
        query = "SELECT * FROM paper_trades WHERE 1=1"
        params = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
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
        query = """
            SELECT COUNT(*) as total_decisions,
                   SUM(CASE WHEN actual_return IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                   AVG(CASE WHEN actual_return IS NOT NULL THEN actual_return ELSE NULL END) as avg_return,
                   SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buys,
                   SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sells,
                   SUM(CASE WHEN action = 'HOLD' THEN 1 ELSE 0 END) as holds
            FROM decisions WHERE 1=1
        """
        params = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        row = self.conn.execute(query, params).fetchone()
        d = dict(row)
        d["accuracy"] = (d["correct_predictions"] / d["resolved"]) if d["resolved"] and d["resolved"] > 0 else None
        return d

    def get_signal_accuracy(self) -> dict:
        """For each signal module, compute how often it predicted correctly."""
        resolved = self.conn.execute(
            "SELECT * FROM decisions WHERE actual_direction IS NOT NULL AND ensemble_direction IS NOT NULL"
        ).fetchall()
        if not resolved:
            return {}

        accuracy = {}
        binary_signals = {
            "ensemble_direction": lambda r, s: r[s] == r["actual_direction"] if r[s] is not None else None,
            "sentiment": lambda r, s: (r["sentiment_score"] or 0) > 0 == (r["actual_direction"] == 1),
            "fii": lambda r, s: (r["fii_net"] or 0) > 0 == (r["actual_direction"] == 1),
            "dii": lambda r, s: (r["dii_net"] or 0) > 0 == (r["actual_direction"] == 1),
            "pcr": lambda r, s: (r["pcr"] or 0) > 1.0 == (r["actual_direction"] == 1),
            "mtf": lambda r, s: (r["mtf_signal"] or 0) > 0 == (r["actual_direction"] == 1),
        }

        for name, check_fn in binary_signals.items():
            correct = 0
            total = 0
            for row in resolved:
                result = check_fn(row, name)
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

    def close(self):
        self.conn.close()
