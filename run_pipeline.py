"""CLI entry point for the retraining pipeline.

Usage:
    python run_pipeline.py                       # All configured tickers
    python run_pipeline.py RELIANCE.NS TCS.NS    # Specific tickers
"""

import logging
import sys

from src.pipeline import RetrainingPipeline, PipelineConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tickers = sys.argv[1:] if len(sys.argv) > 1 else None

    config = PipelineConfig()
    if tickers:
        config.tickers = tickers

    print(f"=== StoMar Retraining Pipeline ===")
    print(f"Tickers: {config.tickers}")
    print(f"Lookback: {config.lookback_period}")
    print(f"Min OOS accuracy: {config.min_oos_accuracy:.1%}")
    print()

    pipeline = RetrainingPipeline(config)
    results = pipeline.run_all()

    summary = pipeline.get_summary()
    print(f"\n=== Pipeline Summary ===")
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
