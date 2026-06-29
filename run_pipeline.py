"""CLI entry point for the retraining pipeline.

Usage:
    python run_pipeline.py                       # All configured tickers
    python run_pipeline.py RELIANCE.NS TCS.NS    # Specific tickers
    python run_pipeline.py --paper               # Run paper trading after training
"""

import argparse
import logging
import sys

from src.pipeline import RetrainingPipeline, PipelineConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="StoMar Retraining Pipeline")
    parser.add_argument("tickers", nargs="*", help="Tickers to process")
    parser.add_argument("--paper", action="store_true",
                        help="Run paper trading after training")
    args = parser.parse_args()

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

    if args.paper:
        print("\n=== Paper Trading Mode ===")
        from src.paper_trader import PaperTrader

        trader = PaperTrader(initial_capital=100_000)
        trader.load_state()

        for r in results:
            if r.status == "success":
                ticker = r.ticker
                print(f"  Loading latest data for {ticker}...")
                try:
                    from src.data_fetcher import fetch_stock_data
                    df = fetch_stock_data(ticker, period="5d")
                    if len(df) >= 2:
                        last = df.iloc[-1]
                        bar_data = {
                            "o": float(last["open"]), "h": float(last["high"]),
                            "low": float(last["low"]), "c": float(last["close"]),
                        }
                        print(f"  {ticker}: OHLC = {bar_data}")
                except Exception as e:
                    print(f"  {ticker}: Could not fetch data: {e}")

        trader.save_state()
        print("  Paper trading state saved.")

    exit_code = 0 if summary["failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
