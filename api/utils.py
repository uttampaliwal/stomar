"""Shared utilities for parallel data fetching across API routers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any


def parallel_fetch(func: Callable, items: list, max_workers: int = 8) -> dict[str, Any]:
    """Run func(item) in parallel for each item. Returns {item: result} dict.
    Items that error are silently skipped.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = {pool.submit(func, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results[item] = future.result()
            except Exception:
                pass
    return results


def parallel_fetch_with_args(func: Callable, items: list, **kwargs) -> dict[str, Any]:
    """Run func(item, **kwargs) in parallel. Returns {item: result} dict."""
    results = {}
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
        futures = {pool.submit(func, item, **kwargs): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results[item] = future.result()
            except Exception:
                pass
    return results
