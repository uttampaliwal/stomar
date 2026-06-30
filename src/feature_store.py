"""Feature store with versioning.

Pins feature definitions to hashes so that models can be verified
against the features they were trained on.

Usage:
    from src.feature_store import compute_feature_hash, register_feature_version

    feature_cols = sorted([c for c in df.columns if c not in SKIP])
    fhash = compute_feature_hash(feature_cols)
    register_feature_version(fhash, feature_cols, description="v1 baseline")
"""

import hashlib
import json
import logging
import os
from datetime import datetime

from src.constants import FEATURE_VERSIONS_DIR

logger = logging.getLogger(__name__)


def compute_feature_hash(feature_cols: list, transformations: dict = None) -> str:
    """Deterministic hash of feature definition.

    Args:
        feature_cols: Sorted list of feature column names
        transformations: Optional dict of column -> transformation applied

    Returns:
        12-char hex hash
    """
    content = json.dumps({
        "columns": sorted(feature_cols),
        "transformations": transformations or {},
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def register_feature_version(
    version_hash: str,
    feature_cols: list,
    transformations: dict = None,
    description: str = "",
) -> dict:
    """Register a feature version with metadata.

    Args:
        version_hash: Hash from compute_feature_hash
        feature_cols: List of feature column names
        transformations: Optional transformation metadata
        description: Human-readable description

    Returns:
        Version record dict
    """
    os.makedirs(FEATURE_VERSIONS_DIR, exist_ok=True)

    record = {
        "hash": version_hash,
        "columns": sorted(feature_cols),
        "n_features": len(feature_cols),
        "transformations": transformations or {},
        "description": description,
        "created_at": datetime.now().isoformat(),
    }

    path = os.path.join(FEATURE_VERSIONS_DIR, f"{version_hash}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    latest_path = os.path.join(FEATURE_VERSIONS_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Registered feature version {version_hash} ({len(feature_cols)} features)")
    return record


def load_feature_version(version_hash: str) -> dict:
    """Load a feature version record.

    Args:
        version_hash: Hash to load

    Returns:
        Version record dict

    Raises:
        FileNotFoundError: If version not found
    """
    path = os.path.join(FEATURE_VERSIONS_DIR, f"{version_hash}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature version {version_hash} not found")
    with open(path) as f:
        return json.load(f)


def load_latest_version() -> dict:
    """Load the latest feature version record.

    Returns:
        Version record dict

    Raises:
        FileNotFoundError: If no versions registered
    """
    path = os.path.join(FEATURE_VERSIONS_DIR, "latest.json")
    if not os.path.exists(path):
        raise FileNotFoundError("No feature versions registered yet")
    with open(path) as f:
        return json.load(f)


def verify_feature_compatibility(
    model_feature_hash: str, current_feature_hash: str
) -> dict:
    """Check if current features match what the model was trained on.

    Args:
        model_feature_hash: Hash of features the model was trained with
        current_feature_hash: Hash of current features

    Returns:
        Dict with compatible, model_hash, current_hash, differences
    """
    if model_feature_hash == current_feature_hash:
        return {
            "compatible": True,
            "model_hash": model_feature_hash,
            "current_hash": current_feature_hash,
        }

    try:
        model_record = load_feature_version(model_feature_hash)
    except FileNotFoundError:
        return {
            "compatible": False,
            "error": f"Model feature version {model_feature_hash} not found in registry",
        }

    try:
        current_record = load_feature_version(current_feature_hash)
    except FileNotFoundError:
        return {
            "compatible": False,
            "error": f"Current feature version {current_feature_hash} not found in registry",
        }

    model_cols = set(model_record["columns"])
    current_cols = set(current_record["columns"])

    return {
        "compatible": False,
        "model_hash": model_feature_hash,
        "current_hash": current_feature_hash,
        "missing_in_current": sorted(model_cols - current_cols),
        "extra_in_current": sorted(current_cols - model_cols),
        "model_n_features": model_record["n_features"],
        "current_n_features": current_record["n_features"],
    }


def list_versions() -> list:
    """List all registered feature versions.

    Returns:
        List of version record dicts
    """
    if not os.path.exists(FEATURE_VERSIONS_DIR):
        return []

    versions = []
    for fname in os.listdir(FEATURE_VERSIONS_DIR):
        if fname == "latest.json" or not fname.endswith(".json"):
            continue
        path = os.path.join(FEATURE_VERSIONS_DIR, fname)
        try:
            with open(path) as f:
                versions.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping corrupted feature version file %s: %s", fname, e)

    return sorted(versions, key=lambda v: v.get("created_at", ""))
