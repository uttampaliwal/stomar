"""CLI entry point for the retraining pipeline.

Usage:
    python src/automation/run_pipeline.py                       # All configured tickers
    python src/automation/run_pipeline.py RELIANCE.NS TCS.NS    # Specific tickers
    python src/automation/run_pipeline.py --train-all           # Batch train all 20 NSE stocks
    python src/automation/run_pipeline.py --train RELIANCE.NS   # Train specific ticker(s)
"""

import argparse
import logging
import sys

from src.core.pipeline import RetrainingPipeline, PipelineConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="StoMar Retraining Pipeline")
    parser.add_argument("tickers", nargs="*", help="Tickers to process")
    parser.add_argument("--train-all", action="store_true",
                        help="Batch train all 20 NSE stocks")
    parser.add_argument("--train", nargs="+", metavar="TICKER",
                        help="Train specific ticker(s) without running full pipeline")
    args = parser.parse_args()

    # --- Batch train mode (standalone) ---
    if args.train_all or args.train:
        from src.models.trainer import batch_train

        tickers = None if args.train_all else args.train
        print("=== StoMar Batch Training ===")
        if tickers:
            print(f"Tickers: {tickers}")
        else:
            print("Tickers: ALL (20 NSE stocks)")
        print()

        summary = batch_train(tickers, force_retrain=False)

        print("\n=== Training Summary ===")
        print(f"Success: {len(summary['success'])}")
        print(f"Failed:  {len(summary['failed'])}")
        print(f"Skipped: {len(summary['skipped'])}")
        print(f"Time:    {summary['duration_seconds']}s")

        for t in summary["success"]:
            print(f"  [OK]   {t}")
        for t, err in summary["failed"].items():
            print(f"  [FAIL] {t}: {err}")
        for t in summary["skipped"]:
            print(f"  [SKIP] {t} (already trained)")

        sys.exit(0 if not summary["failed"] else 1)

    # --- Normal pipeline mode ---
    config = PipelineConfig()
    if args.tickers:
        config.tickers = args.tickers

    print("=== StoMar Retraining Pipeline ===")
    print(f"Tickers: {config.tickers}")
    print(f"Lookback: {config.lookback_period}")
    print(f"Min OOS accuracy: {config.min_oos_accuracy:.1%}")
    print()

    pipeline = RetrainingPipeline(config)
    results = pipeline.run_all()

    summary = pipeline.get_summary()
    print("\n=== Pipeline Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    print(f"Rejected: {summary['rejected']}")

    for r in results:
        icon = {"success": "OK", "failed": "FAIL", "rejected": "REJECT"}.get(r.status, "?")
        print(f"  [{icon}] {r.ticker}: [{r.stage}] {r.message}")

    exit_code = 0 if summary["failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
