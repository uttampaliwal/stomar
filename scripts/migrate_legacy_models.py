#!/usr/bin/env python3
"""One-time migration of legacy (unmanifested) artifacts.

This script is the ONLY place where legacy pickle artifacts may be touched.
It is not part of the runtime path and should be run once during an upgrade,
then retired.

What it does:
1. For every per-ticker model bundle in models/ (and models/production/)
   that lacks a manifest, writes a SHA-256 manifest from the CURRENT file
   contents. After migration, the runtime loader will refuse any file that
   deviates from these hashes — treat the hashes as the trust anchor.
2. For the meta-controller pickle at models/meta_controller.pkl, writes its
   manifest when missing.
3. For legacy mtf_*.pkl multi-timeframe caches in data/, converts them to
   the parquet format and deletes the pickle (only if conversion succeeds).

Usage:
    python scripts/migrate_legacy_models.py [--no-delete-pkl]
"""

import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.constants import MODELS_DIR, DATA_DIR  # noqa: E402
from src.models.artifacts import ArtifactBundle, ArtifactVerificationError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate_legacy_models")

_TICKER_ARTIFACTS = [
    "lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl", "scaler.pkl",
    "features.pkl", "lstm_dim.pkl", "lgb.pkl", "cat.pkl",
]


def migrate_meta_controller():
    mc_path = os.path.join(MODELS_DIR, "meta_controller.pkl")
    if not os.path.exists(mc_path):
        return 0
    bundle_name = os.path.splitext(os.path.basename(mc_path))[0]
    manifest = os.path.join(MODELS_DIR, f"{bundle_name}_manifest.json")
    if os.path.exists(manifest):
        logger.info("meta_controller manifest already exists")
        return 0
    ArtifactBundle.create(MODELS_DIR, bundle_name, {"meta_controller.pkl": ""})
    logger.info("migrated meta_controller.pkl -> manifest")
    return 1


def migrate_ticker_bundles(root: str) -> int:
    count = 0
    if not os.path.isdir(root):
        return 0
    for ext in ("xgb.pkl", "lstm.pt"):
        for path in glob.glob(os.path.join(root, f"*_{ext}")):
            ticker_clean = os.path.basename(path).rsplit(f"_{ext}", 1)[0]
            manifest = os.path.join(root, f"{ticker_clean}_manifest.json")
            if os.path.exists(manifest):
                continue
            files = {
                f"{ticker_clean}_{a}": ""
                for a in _TICKER_ARTIFACTS
                if os.path.exists(os.path.join(root, f"{ticker_clean}_{a}"))
            }
            if not files:
                continue
            try:
                ArtifactBundle.create(root, ticker_clean, files)
                count += 1
                logger.info("migrated %s in %s", ticker_clean, root)
            except ArtifactVerificationError as e:
                logger.error("failed to migrate %s: %s", ticker_clean, e)
    return count


def migrate_meta_learners(root: str) -> int:
    """meta_{TICKER}.pkl per-ticker meta-learners -> individual manifests."""
    count = 0
    if not os.path.isdir(root):
        return 0
    for path in glob.glob(os.path.join(root, "meta_*.pkl")):
        name = os.path.basename(path)
        bundle_name = os.path.splitext(name)[0]
        manifest = os.path.join(root, f"{bundle_name}_manifest.json")
        if os.path.exists(manifest):
            continue
        ArtifactBundle.create(root, bundle_name, {name: ""})
        count += 1
        logger.info("migrated meta-learner %s", name)
    return count


def migrate_mtf_pickles(delete: bool) -> int:
    count = 0
    for path in glob.glob(os.path.join(DATA_DIR, "mtf_*.pkl")):
        ticker_clean = os.path.basename(path).split(".pkl")[0]
        # legacy name pattern: mtf_{TICKER}.pkl
        prefix = os.path.join(DATA_DIR, ticker_clean)
        try:
            import pandas as pd

            data = pd.read_pickle(path)
        except Exception as e:
            logger.error("cannot read legacy pickle %s: %s — leaving in place", path, e)
            continue
        if not isinstance(data, dict):
            logger.error("%s is not a dict — leaving in place", path)
            continue
        ok = True
        for tf, df in data.items():
            try:
                df.to_parquet(f"{prefix}_{tf}.parquet")
            except Exception as e:
                logger.error("parquet write failed for %s: %s", tf, e)
                ok = False
        if ok:
            if delete:
                os.unlink(path)
                logger.info("migrated and removed %s", path)
            else:
                logger.info("migrated %s (kept pickle)", path)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-delete-pkl", action="store_true",
                        help="keep legacy mtf pickles after conversion")
    args = parser.parse_args()

    n = migrate_meta_controller()
    n += migrate_ticker_bundles(MODELS_DIR)
    n += migrate_ticker_bundles(os.path.join(MODELS_DIR, "production"))
    n += migrate_meta_learners(MODELS_DIR)
    n += migrate_meta_learners(os.path.join(MODELS_DIR, "production"))
    n += migrate_mtf_pickles(delete=not args.no_delete_pkl)
    logger.info("migration complete: %d items processed", n)


if __name__ == "__main__":
    main()
