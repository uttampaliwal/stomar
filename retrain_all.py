"""Retrain all 20 stocks with fixed walk-forward split."""
from src.trainer import train_for_ticker
from src.data_fetcher import NSE_STOCKS
import time

results = {"success": [], "failed": []}
for i, ticker in enumerate(NSE_STOCKS):
    print(f"[{i+1}/{len(NSE_STOCKS)}] Training {ticker}...")
    t0 = time.time()
    try:
        metrics = train_for_ticker(ticker)
        elapsed = time.time() - t0
        ens = metrics.get("ensemble_accuracy", 0)
        xgb_acc = metrics.get("xgb_accuracy", 0)
        lgb_acc = metrics.get("lgb_accuracy", 0)
        print(f"  OK ({elapsed:.1f}s) ensemble={ens:.4f} xgb={xgb_acc:.4f} lgb={lgb_acc:.4f}")
        results["success"].append(ticker)
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(ticker)

print(f"\nDone: {len(results['success'])} success, {len(results['failed'])} failed")
if results["failed"]:
    print("Failed:", results["failed"])
