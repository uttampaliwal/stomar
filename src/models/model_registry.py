"""Model registry with versioning and promotion lifecycle.

Workflow:
1. Train → register as "staging"
2. Evaluate → if passes gates, promote to "production"
3. Old "production" → archived
4. Load always reads "production" version
5. Rollback available to any archived version

Usage:
    from src.models.model_registry import ModelRegistry
    registry = ModelRegistry()
    registry.register("RELIANCE.NS", "/path/to/model.pt", "abc123")
    registry.promote("RELIANCE.NS", version=1)
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.core.constants import MODELS_DIR

logger = logging.getLogger(__name__)

REGISTRY_DIR = os.path.join(MODELS_DIR, "registry")


@dataclass
class ModelVersion:
    """A single model version record."""
    ticker: str
    version: int
    model_path: str
    feature_hash: str
    metrics: dict = field(default_factory=dict)
    status: str = "staging"  # "staging", "production", "archived"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    promoted_at: Optional[str] = None
    notes: str = ""


class ModelRegistry:
    """Version-controlled model registry with staging/production/archived lifecycle."""

    def __init__(self, registry_dir: str = None):
        self.registry_dir = registry_dir or REGISTRY_DIR
        os.makedirs(self.registry_dir, exist_ok=True)

    def _registry_path(self, ticker: str) -> str:
        return os.path.join(
            self.registry_dir, f"{ticker.replace('.', '_')}_registry.json"
        )

    def _load_registry(self, ticker: str) -> list:
        path = self._registry_path(ticker)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return []

    def _save_registry(self, ticker: str, versions: list):
        path = self._registry_path(ticker)
        with open(path, "w") as f:
            json.dump(versions, f, indent=2)

    def register(
        self, ticker: str, model_path: str, feature_hash: str,
        metrics: dict = None, notes: str = "",
    ) -> ModelVersion:
        """Register a new model version as staging."""
        versions = self._load_registry(ticker)

        max_version = max((v["version"] for v in versions), default=0)
        new_version = max_version + 1

        record = ModelVersion(
            ticker=ticker,
            version=new_version,
            model_path=model_path,
            feature_hash=feature_hash,
            metrics=metrics or {},
            status="staging",
            notes=notes,
        )

        versions.append(record.__dict__)
        self._save_registry(ticker, versions)

        logger.info(f"Registered {ticker} v{new_version} (staging)")
        return record

    def promote(self, ticker: str, version: int) -> ModelVersion:
        """Promote a staging model to production."""
        versions = self._load_registry(ticker)

        target = None
        for v in versions:
            if v["version"] == version:
                target = v
                break

        if target is None:
            raise ValueError(f"Version {version} not found for {ticker}")

        if target["status"] != "staging":
            raise ValueError(
                f"Version {version} is '{target['status']}', not 'staging'"
            )

        for v in versions:
            if v["status"] == "production":
                v["status"] = "archived"

        target["status"] = "production"
        target["promoted_at"] = datetime.now().isoformat()

        self._save_registry(ticker, versions)

        prod_dir = os.path.join(MODELS_DIR, "production")
        os.makedirs(prod_dir, exist_ok=True)
        if os.path.exists(target["model_path"]):
            dst = os.path.join(
                prod_dir, f"{ticker.replace('.', '_')}_v{version}.pt"
            )
            shutil.copy2(target["model_path"], dst)

        logger.info(f"Promoted {ticker} v{version} to production")
        return ModelVersion(**target)

    def get_production(self, ticker: str) -> Optional[ModelVersion]:
        """Get the current production model version."""
        versions = self._load_registry(ticker)
        for v in versions:
            if v["status"] == "production":
                return ModelVersion(**v)
        return None

    def get_version(self, ticker: str, version: int) -> Optional[ModelVersion]:
        """Get a specific version."""
        versions = self._load_registry(ticker)
        for v in versions:
            if v["version"] == version:
                return ModelVersion(**v)
        return None

    def list_versions(self, ticker: str) -> list:
        """List all versions for a ticker."""
        versions = self._load_registry(ticker)
        return [ModelVersion(**v) for v in versions]

    def rollback(self, ticker: str, to_version: int) -> ModelVersion:
        """Rollback to a previous production version.

        Unlike promote(), this accepts archived versions.
        """
        versions = self._load_registry(ticker)

        target = None
        for v in versions:
            if v["version"] == to_version and v["status"] == "archived":
                target = v
                break

        if target is None:
            raise ValueError(
                f"Archived version {to_version} not found for {ticker}"
            )

        for v in versions:
            if v["status"] == "production":
                v["status"] = "archived"

        target["status"] = "production"
        target["promoted_at"] = datetime.now().isoformat()

        self._save_registry(ticker, versions)

        prod_dir = os.path.join(MODELS_DIR, "production")
        os.makedirs(prod_dir, exist_ok=True)
        if os.path.exists(target["model_path"]):
            dst = os.path.join(
                prod_dir, f"{ticker.replace('.', '_')}_v{to_version}.pt"
            )
            shutil.copy2(target["model_path"], dst)
        else:
            logger.warning(
                f"Model file not found for {ticker} v{to_version}: {target['model_path']}"
            )

        logger.info(f"Rolled back {ticker} to v{to_version}")
        return ModelVersion(**target)

    def get_summary(self) -> dict:
        """Get summary of all tickers in the registry."""
        tickers = {}
        for fname in os.listdir(self.registry_dir):
            if not fname.endswith("_registry.json"):
                continue
            encoded = fname.replace("_registry.json", "")
            path = os.path.join(self.registry_dir, fname)
            try:
                with open(path) as f:
                    versions = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if versions and "ticker" in versions[0]:
                ticker = versions[0]["ticker"]
            else:
                ticker = encoded.replace("_", ".")
            tickers[ticker] = {
                "total_versions": len(versions),
                "production": next(
                    (v["version"] for v in versions if v["status"] == "production"),
                    None,
                ),
                "staging": [
                    v["version"] for v in versions if v["status"] == "staging"
                ],
                "archived": [
                    v["version"] for v in versions if v["status"] == "archived"
                ],
            }
        return tickers
