"""Automatic data retention and archival policy.

Prevents unbounded disk growth by cleaning stale cache files,
archiving old SQLite ledger rows, and pruning monitoring/feature files.

Usage:
    from src.data.data_retention import DataRetentionPolicy
    policy = DataRetentionPolicy()
    report = policy.run_all(active_tickers=["RELIANCE.NS", "TCS.NS"])
    print(report)

    # CLI
    python -m src.data.data_retention --dry-run
    python -m src.data.data_retention
"""

import glob
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Files that should never be deleted by cache cleanup
_PROTECTED_FILES = {
    "stomar.db",
    "paper_state.json",
    "mf_state.json",
    "paper_session.json",
    "options_pcr.json",
    "fii_dii.parquet",
}

# Cache file extensions to clean
_CACHE_EXTENSIONS = {".parquet", ".pkl", ".json", ".csv"}


class DataRetentionPolicy:
    """Enforces data retention rules across all data files."""

    def __init__(self, data_dir: str = None, ledger_db: str = None):
        from src.core.settings import settings
        from src.core.constants import DATA_DIR, LEDGER_DB, MONITORING_DIR, FEATURE_VERSIONS_DIR, ARCHIVE_DIR

        self.data_dir = data_dir or DATA_DIR
        self.ledger_db = ledger_db or LEDGER_DB
        self.monitoring_dir = MONITORING_DIR
        self.feature_versions_dir = FEATURE_VERSIONS_DIR
        self.archive_dir = ARCHIVE_DIR

        self.cache_retention_days = settings.cache_retention_days
        self.ledger_retention_days = settings.ledger_retention_days
        self.monitoring_retention_days = settings.monitoring_retention_days
        self.feature_versions_keep = settings.feature_versions_keep
        self.pipeline_logs_retention_days = settings.pipeline_logs_retention_days

    def _file_age_days(self, filepath: str) -> float:
        """Return age of file in days based on mtime."""
        mtime = os.path.getmtime(filepath)
        return (time.time() - mtime) / 86400

    def clean_stale_cache_files(self, dry_run: bool = False) -> dict:
        """Delete cache files older than cache_retention_days.

        Skips protected files (stomar.db, paper_state.json, etc.) and
        files outside the data directory root (not in subdirectories).
        """
        deleted = []
        skipped = []
        cutoff = time.time() - (self.cache_retention_days * 86400)

        if not os.path.isdir(self.data_dir):
            return {"deleted": 0, "skipped": 0, "bytes_freed": 0}

        for entry in os.scandir(self.data_dir):
            if entry.is_dir():
                continue
            if entry.name in _PROTECTED_FILES:
                skipped.append(entry.name)
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in _CACHE_EXTENSIONS:
                skipped.append(entry.name)
                continue
            if entry.stat().st_mtime < cutoff:
                if dry_run:
                    deleted.append(entry.name)
                else:
                    try:
                        size = entry.stat().st_size
                        os.unlink(entry.path)
                        deleted.append(entry.name)
                        logger.info("Deleted stale cache: %s (%d days old)", entry.name, self._file_age_days(entry.path))
                    except OSError as e:
                        logger.warning("Failed to delete %s: %s", entry.name, e)

        bytes_freed = 0
        if not dry_run and deleted:
            bytes_freed = sum(
                entry.stat().st_size
                for entry in os.scandir(self.data_dir)
                if entry.name in deleted
            )

        return {"deleted": len(deleted), "skipped": len(skipped), "bytes_freed": bytes_freed}

    def clean_stale_ticker_files(self, active_tickers: list[str], dry_run: bool = False) -> dict:
        """Delete cache files for tickers not in the active watchlist.

        Matches files like: {TICKER}.parquet, mtf_{TICKER}.pkl,
        sentiment_{TICKER}.json, {TICKER}_baseline.json, etc.
        """
        deleted = []
        if not active_tickers:
            return {"deleted": 0}

        # Normalize tickers to set of base names (e.g., "RELIANCE", "HDFCBANK")
        active_bases = set()
        for t in active_tickers:
            base = t.replace(".NS", "").replace("_NS", "").upper()
            active_bases.add(base)

        patterns_to_check = [
            (os.path.join(self.data_dir, "*.parquet"), re_fn := lambda f: self._extract_ticker_from_filename(f, active_bases)),
            (os.path.join(self.data_dir, "mtf_*.pkl"), re_fn),
            (os.path.join(self.data_dir, "sentiment_*.json"), re_fn),
            (os.path.join(self.monitoring_dir, "*.json"), re_fn),
        ]

        for pattern, _ in patterns_to_check:
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                if self._is_ticker_file_for_inactive_ticker(filename, active_bases):
                    if dry_run:
                        deleted.append(filepath)
                    else:
                        try:
                            os.unlink(filepath)
                            deleted.append(filepath)
                            logger.info("Deleted stale ticker file: %s", filename)
                        except OSError as e:
                            logger.warning("Failed to delete %s: %s", filename, e)

        return {"deleted": len(deleted)}

    def _is_ticker_file_for_inactive_ticker(self, filename: str, active_bases: set) -> bool:
        """Check if a filename belongs to a ticker not in active_bases."""
        name_upper = filename.upper()
        for base in active_bases:
            if base in name_upper:
                return False
        return True

    def _extract_ticker_from_filename(self, filepath: str, active_bases: set) -> str | None:
        """Extract ticker base from filename, return None if active."""
        return None

    def archive_and_trim_ledger(self, dry_run: bool = False) -> dict:
        """Export old ledger rows to Parquet, then delete from SQLite.

        Rows older than ledger_retention_days are archived to:
            data/archive/decisions_YYYY-MM.parquet
            data/archive/trades_YYYY-MM.parquet
            data/archive/snapshots_YYYY-MM.parquet
        """
        if not os.path.exists(self.ledger_db):
            return {"archived_decisions": 0, "archived_trades": 0, "archived_snapshots": 0, "deleted": 0}

        cutoff_date = (datetime.now() - timedelta(days=self.ledger_retention_days)).strftime("%Y-%m-%d")
        result = {"archived_decisions": 0, "archived_trades": 0, "archived_snapshots": 0, "deleted": 0}

        conn = sqlite3.connect(self.ledger_db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        try:
            # Count rows to archive
            decisions_count = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE date < ?", (cutoff_date,)
            ).fetchone()[0]
            trades_count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE decision_id IN (SELECT id FROM decisions WHERE date < ?)",
                (cutoff_date,)
            ).fetchone()[0]
            snapshots_count = conn.execute(
                "SELECT COUNT(*) FROM portfolio_snapshots WHERE date < ?", (cutoff_date,)
            ).fetchone()[0]

            if decisions_count == 0 and trades_count == 0 and snapshots_count == 0:
                return result

            if dry_run:
                result["archived_decisions"] = decisions_count
                result["archived_trades"] = trades_count
                result["archived_snapshots"] = snapshots_count
                return result

            os.makedirs(self.archive_dir, exist_ok=True)

            # Archive decisions by month
            if decisions_count > 0:
                months = conn.execute(
                    "SELECT DISTINCT substr(date, 1, 7) as month FROM decisions WHERE date < ? ORDER BY month",
                    (cutoff_date,)
                ).fetchall()

                for row in months:
                    month = row[0]
                    rows = conn.execute(
                        "SELECT * FROM decisions WHERE date < ? AND substr(date, 1, 7) = ?",
                        (cutoff_date, month)
                    ).fetchall()

                    if rows:
                        archive_path = os.path.join(self.archive_dir, f"decisions_{month}.parquet")
                        self._append_rows_to_parquet(rows, archive_path)
                        result["archived_decisions"] += len(rows)

            # Archive trades by month
            if trades_count > 0:
                months = conn.execute(
                    """SELECT DISTINCT substr(d.date, 1, 7) as month
                       FROM paper_trades t
                       JOIN decisions d ON t.decision_id = d.id
                       WHERE d.date < ? ORDER BY month""",
                    (cutoff_date,)
                ).fetchall()

                for row in months:
                    month = row[0]
                    rows = conn.execute(
                        """SELECT t.* FROM paper_trades t
                           JOIN decisions d ON t.decision_id = d.id
                           WHERE d.date < ? AND substr(d.date, 1, 7) = ?""",
                        (cutoff_date, month)
                    ).fetchall()

                    if rows:
                        archive_path = os.path.join(self.archive_dir, f"trades_{month}.parquet")
                        self._append_rows_to_parquet(rows, archive_path)
                        result["archived_trades"] += len(rows)

            # Archive snapshots
            if snapshots_count > 0:
                months = conn.execute(
                    "SELECT DISTINCT substr(date, 1, 7) as month FROM portfolio_snapshots WHERE date < ? ORDER BY month",
                    (cutoff_date,)
                ).fetchall()

                for row in months:
                    month = row[0]
                    rows = conn.execute(
                        "SELECT * FROM portfolio_snapshots WHERE date < ? AND substr(date, 1, 7) = ?",
                        (cutoff_date, month)
                    ).fetchall()

                    if rows:
                        archive_path = os.path.join(self.archive_dir, f"snapshots_{month}.parquet")
                        self._append_rows_to_parquet(rows, archive_path)
                        result["archived_snapshots"] += len(rows)

            # Delete archived rows from SQLite
            conn.execute("DELETE FROM paper_trades WHERE decision_id IN (SELECT id FROM decisions WHERE date < ?)", (cutoff_date,))
            conn.execute("DELETE FROM decisions WHERE date < ?", (cutoff_date,))
            conn.execute("DELETE FROM portfolio_snapshots WHERE date < ?", (cutoff_date,))
            conn.commit()
            result["deleted"] = decisions_count + trades_count + snapshots_count

            # VACUUM to reclaim space
            conn.execute("VACUUM")
            logger.info("Archived %d decisions, %d trades, %d snapshots; VACUUM complete",
                        result["archived_decisions"], result["archived_trades"], result["archived_snapshots"])

        except Exception as e:
            logger.error("Ledger archival failed: %s", e)
            conn.rollback()
        finally:
            conn.close()

        return result

    def _append_rows_to_parquet(self, rows: list, archive_path: str) -> None:
        """Append rows to a Parquet file (create or merge)."""
        import pandas as pd

        data = [dict(r) for r in rows]
        new_df = pd.DataFrame(data)

        if os.path.exists(archive_path):
            existing_df = pd.read_parquet(archive_path)
            new_df = pd.concat([existing_df, new_df], ignore_index=True)

        new_df.to_parquet(archive_path, index=False)

    def clean_feature_versions(self, dry_run: bool = False) -> dict:
        """Keep only the N most recent feature version files, delete older ones."""
        if not os.path.isdir(self.feature_versions_dir):
            return {"deleted": 0}

        files = []
        for f in os.scandir(self.feature_versions_dir):
            if f.is_file() and f.name != "latest.json":
                files.append(f)

        if len(files) <= self.feature_versions_keep:
            return {"deleted": 0}

        # Sort by mtime descending (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        to_delete = files[self.feature_versions_keep:]

        deleted = 0
        for f in to_delete:
            if dry_run:
                deleted += 1
            else:
                try:
                    os.unlink(f.path)
                    deleted += 1
                    logger.info("Deleted old feature version: %s", f.name)
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", f.name, e)

        return {"deleted": deleted}

    def clean_monitoring_files(self, dry_run: bool = False) -> dict:
        """Delete monitoring files older than monitoring_retention_days."""
        if not os.path.isdir(self.monitoring_dir):
            return {"deleted": 0}

        cutoff = time.time() - (self.monitoring_retention_days * 86400)
        deleted = 0

        for entry in os.scandir(self.monitoring_dir):
            if entry.is_file() and entry.name.endswith(".json"):
                if entry.stat().st_mtime < cutoff:
                    if dry_run:
                        deleted += 1
                    else:
                        try:
                            os.unlink(entry.path)
                            deleted += 1
                            logger.info("Deleted old monitoring file: %s", entry.name)
                        except OSError as e:
                            logger.warning("Failed to delete %s: %s", entry.name, e)

        return {"deleted": deleted}

    def clean_pipeline_checkpoints(self, dry_run: bool = False) -> dict:
        """Delete orphaned per-ticker pipeline checkpoint files."""
        pattern = os.path.join(self.data_dir, "pipeline_checkpoint_*.json")
        deleted = 0

        for filepath in glob.glob(pattern):
            if dry_run:
                deleted += 1
            else:
                try:
                    os.unlink(filepath)
                    deleted += 1
                    logger.info("Deleted orphaned checkpoint: %s", os.path.basename(filepath))
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", filepath, e)

        return {"deleted": deleted}

    def clean_old_pipeline_logs(self, dry_run: bool = False) -> dict:
        """Delete pipeline log files older than pipeline_logs_retention_days."""
        log_dir = os.path.join(self.data_dir, "pipeline_logs")
        if not os.path.isdir(log_dir):
            return {"deleted": 0}

        cutoff = time.time() - (self.pipeline_logs_retention_days * 86400)
        deleted = 0

        for entry in os.scandir(log_dir):
            if entry.is_file() and entry.name.endswith(".log"):
                if entry.stat().st_mtime < cutoff:
                    if dry_run:
                        deleted += 1
                    else:
                        try:
                            os.unlink(entry.path)
                            deleted += 1
                            logger.info("Deleted old log: %s", entry.name)
                        except OSError as e:
                            logger.warning("Failed to delete %s: %s", entry.name, e)

        return {"deleted": deleted}

    def get_disk_usage(self) -> dict:
        """Return disk usage summary for the data directory."""
        total_size = 0
        file_count = 0
        by_type = {}

        if os.path.isdir(self.data_dir):
            for entry in os.scandir(self.data_dir):
                if entry.is_file():
                    size = entry.stat().st_size
                    ext = os.path.splitext(entry.name)[1].lower() or "(none)"
                    total_size += size
                    file_count += 1
                    by_type.setdefault(ext, {"count": 0, "bytes": 0})
                    by_type[ext]["count"] += 1
                    by_type[ext]["bytes"] += size

        # Include subdirectories
        for subdir in [self.monitoring_dir, self.feature_versions_dir, self.archive_dir]:
            if os.path.isdir(subdir):
                for entry in os.scandir(subdir):
                    if entry.is_file():
                        size = entry.stat().st_size
                        ext = os.path.splitext(entry.name)[1].lower() or "(none)"
                        total_size += size
                        file_count += 1
                        by_type.setdefault(ext, {"count": 0, "bytes": 0})
                        by_type[ext]["count"] += 1
                        by_type[ext]["bytes"] += size

        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "by_type": by_type,
        }

    def run_all(self, active_tickers: list[str] = None, dry_run: bool = False) -> dict:
        """Execute all retention steps and return a summary report."""
        logger.info("Running data retention policy (dry_run=%s)", dry_run)

        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "cache_files_deleted": 0,
            "stale_ticker_files_deleted": 0,
            "ledger_archived": {},
            "feature_versions_deleted": 0,
            "monitoring_files_deleted": 0,
            "checkpoints_deleted": 0,
            "pipeline_logs_deleted": 0,
            "disk_usage_before": self.get_disk_usage(),
        }

        # 1. Clean stale cache files
        result = self.clean_stale_cache_files(dry_run=dry_run)
        report["cache_files_deleted"] = result["deleted"]
        report["cache_bytes_freed"] = result.get("bytes_freed", 0)

        # 2. Clean stale ticker files
        if active_tickers:
            result = self.clean_stale_ticker_files(active_tickers, dry_run=dry_run)
            report["stale_ticker_files_deleted"] = result["deleted"]

        # 3. Archive and trim ledger
        result = self.archive_and_trim_ledger(dry_run=dry_run)
        report["ledger_archived"] = result

        # 4. Clean feature versions
        result = self.clean_feature_versions(dry_run=dry_run)
        report["feature_versions_deleted"] = result["deleted"]

        # 5. Clean monitoring files
        result = self.clean_monitoring_files(dry_run=dry_run)
        report["monitoring_files_deleted"] = result["deleted"]

        # 6. Clean pipeline checkpoints
        result = self.clean_pipeline_checkpoints(dry_run=dry_run)
        report["checkpoints_deleted"] = result["deleted"]

        # 7. Clean pipeline logs
        result = self.clean_old_pipeline_logs(dry_run=dry_run)
        report["pipeline_logs_deleted"] = result["deleted"]

        # Disk usage after (only meaningful if not dry_run)
        if not dry_run:
            report["disk_usage_after"] = self.get_disk_usage()

        logger.info(
            "Retention complete: cache=%d, ticker=%d, ledger_archived=%d, "
            "features=%d, monitoring=%d, checkpoints=%d, logs=%d",
            report["cache_files_deleted"],
            report["stale_ticker_files_deleted"],
            sum(v for v in report["ledger_archived"].values() if isinstance(v, int)),
            report["feature_versions_deleted"],
            report["monitoring_files_deleted"],
            report["checkpoints_deleted"],
            report["pipeline_logs_deleted"],
        )

        return report


def _format_bytes(size: int) -> str:
    """Human-readable byte size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def print_report(report: dict) -> None:
    """Print a human-readable retention report."""
    print(f"\n{'=' * 60}")
    print(f"  Data Retention Report — {report['timestamp']}")
    print(f"{'=' * 60}")
    print(f"  Dry run: {'Yes' if report['dry_run'] else 'No'}")
    print()

    before = report["disk_usage_before"]
    print(f"  Disk usage before: {before['total_mb']:.1f} MB ({before['file_count']} files)")

    print("\n  Actions taken:")
    print(f"    Cache files deleted:       {report['cache_files_deleted']}")
    if "cache_bytes_freed" in report:
        print(f"    Cache bytes freed:         {_format_bytes(report['cache_bytes_freed'])}")
    print(f"    Stale ticker files:        {report['stale_ticker_files_deleted']}")

    ledger = report["ledger_archived"]
    total_ledger = sum(v for v in ledger.values() if isinstance(v, int))
    if total_ledger > 0:
        print(f"    Ledger rows archived:      {ledger.get('archived_decisions', 0)} decisions, "
              f"{ledger.get('archived_trades', 0)} trades, {ledger.get('archived_snapshots', 0)} snapshots")
        print(f"    Ledger rows deleted:       {ledger.get('deleted', 0)}")
    else:
        print("    Ledger: no rows to archive")

    print(f"    Feature versions deleted:  {report['feature_versions_deleted']}")
    print(f"    Monitoring files deleted:  {report['monitoring_files_deleted']}")
    print(f"    Checkpoints deleted:       {report['checkpoints_deleted']}")
    print(f"    Pipeline logs deleted:     {report['pipeline_logs_deleted']}")

    if "disk_usage_after" in report:
        after = report["disk_usage_after"]
        print(f"\n  Disk usage after:  {after['total_mb']:.1f} MB ({after['file_count']} files)")
        diff = before["total_bytes"] - after["total_bytes"]
        if diff > 0:
            print(f"  Space freed:       {_format_bytes(diff)}")

    print(f"{'=' * 60}\n")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="StoMar Data Retention Policy")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be deleted without making changes")
    parser.add_argument("--ticker", nargs="*", metavar="TICKER",
                        help="Active tickers for stale ticker file cleanup")
    args = parser.parse_args()

    policy = DataRetentionPolicy()
    report = policy.run_all(active_tickers=args.ticker, dry_run=args.dry_run)
    print_report(report)

    if report.get("cache_files_deleted", 0) == 0 and \
       report.get("stale_ticker_files_deleted", 0) == 0 and \
       sum(v for v in report.get("ledger_archived", {}).values() if isinstance(v, int)) == 0 and \
       report.get("feature_versions_deleted", 0) == 0 and \
       report.get("monitoring_files_deleted", 0) == 0 and \
       report.get("checkpoints_deleted", 0) == 0 and \
       report.get("pipeline_logs_deleted", 0) == 0:
        print("No cleanup needed — all data is within retention limits.")
