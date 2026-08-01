"""Verified model artifact loading.

Every serialized artifact (joblib pickle, torch checkpoint) is loaded only
after its SHA-256 digest matches a manifest that was written when the bundle
was saved. No bytes are ever handed to a deserializer until the hash check
passes, so a tampered or replaced file can never reach pickle/torch.

A manifest is a JSON file named ``{bundle}_manifest.json`` next to the
artifacts:

    {
      "bundle": "RELIANCE_NS",
      "model_version": "1",
      "feature_schema_version": "1",
      "training_dataset_hash": "sha256:...",
      "created_at": "2026-08-01T00:00:00+00:00",
      "files": {
        "RELIANCE_NS_xgb.pkl": "sha256:hex...",
        "RELIANCE_NS_lstm.pt":  "sha256:hex..."
      }
    }

Rules enforced here:

* artifacts must live inside a configured root (default ``MODELS_DIR``);
* the manifest itself must exist — legacy unmanifested pickles are refused
  by the runtime loader (migrate them with ``scripts/migrate_legacy_models.py``);
* every file listed in the manifest must match its recorded digest;
* the manifest must not list files outside the bundle directory.

Usage:
    from src.models.artifacts import ArtifactBundle

    bundle = ArtifactBundle.for_ticker("RELIANCE.NS")
    xgb = bundle.load_joblib("RELIANCE_NS_xgb.pkl", expected_type_check=hasattr_predict)
    lstm_state = bundle.load_torch("RELIANCE_NS_lstm.pt")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable

from src.core.secure_io import (
    atomic_write_json,
    safe_path_join,
    sha256_file,
    verify_sha256,
)

logger = logging.getLogger(__name__)

MANIFEST_SUFFIX = "_manifest.json"
MANIFEST_VERSION = 1

# Hard "model registry" confinement: runtime loading is only allowed from
# these directories. Anything else is an untrusted path.
ALLOWED_ROOTS = ("models", "models/production", "models/research")


class ArtifactVerificationError(Exception):
    """Raised when an artifact bundle fails verification."""


class ArtifactBundle:
    """A verified set of model files for one ticker (or one model object)."""

    def __init__(self, root: str, manifest_path: str):
        self.root = os.path.abspath(root)
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest: dict[str, Any] = {}
        self._digests: dict[str, str] = {}

    # ── creation ─────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        root: str,
        bundle_name: str,
        files: dict[str, str],
        *,
        model_version: str = "1",
        feature_schema_version: str = "1",
        training_dataset_hash: str = "",
    ) -> "ArtifactBundle":
        """Record the current digests of *files* and persist the manifest.

        Args:
            root: Directory containing the artifacts.
            bundle_name: e.g. "RELIANCE_NS" or "meta_controller".
            files: {file_name: optional note} — digests are computed now.
        """
        root_abs = os.path.abspath(root)
        digests: dict[str, str] = {}
        for name in files:
            path = safe_path_join(root_abs, name)
            if not os.path.exists(path):
                raise ArtifactVerificationError(
                    f"cannot create manifest: {path} does not exist"
                )
            digests[name] = sha256_file(path)

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "bundle": bundle_name,
            "model_version": str(model_version),
            "feature_schema_version": str(feature_schema_version),
            "training_dataset_hash": str(training_dataset_hash),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": digests,
        }
        bundle = cls(root_abs, os.path.join(root_abs, f"{bundle_name}{MANIFEST_SUFFIX}"))
        bundle._digests = digests
        bundle.manifest = manifest
        atomic_write_json(bundle.manifest_path, manifest)
        logger.info("manifest written: %s (%d files)", bundle.manifest_path, len(digests))
        return bundle

    @classmethod
    def for_ticker(cls, root: str, ticker: str) -> "ArtifactBundle":
        """Locate and load the manifest for a ticker bundle."""
        clean = ticker.replace(".", "_").replace(os.sep, "_")
        return cls.load(root, clean)

    @classmethod
    def load(cls, root: str, bundle_name: str) -> "ArtifactBundle":
        """Load a manifest and verify it against the files on disk.

        Raises ArtifactVerificationError for missing manifests, missing
        files, or any digest mismatch. Loading never deserializes anything.
        """
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            raise ArtifactVerificationError(f"artifact root does not exist: {root_abs}")
        manifest_path = safe_path_join(
            root_abs, f"{bundle_name}{MANIFEST_SUFFIX}"
        )
        if not os.path.exists(manifest_path):
            raise ArtifactVerificationError(
                f"no manifest for bundle {bundle_name!r} at {manifest_path} — "
                f"refusing to load unverified artifacts (run "
                f"scripts/migrate_legacy_models.py to create manifests for "
                f"legacy artifacts)"
            )
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ArtifactVerificationError(
                f"manifest {manifest_path} is unreadable/corrupt: {e}"
            ) from e

        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ArtifactVerificationError(
                f"manifest {manifest_path} is malformed (missing 'files')"
            )
        if str(manifest.get("manifest_version")) != str(MANIFEST_VERSION):
            raise ArtifactVerificationError(
                f"manifest {manifest_path} has unsupported version "
                f"{manifest.get('manifest_version')}"
            )

        bundle = cls(root_abs, manifest_path)
        bundle.manifest = manifest
        for name, digest in manifest["files"].items():
            # Confine: listed files must resolve inside the root.
            path = safe_path_join(root_abs, name)
            if not os.path.exists(path):
                raise ArtifactVerificationError(
                    f"manifest {manifest_path} lists {name} but the file is missing"
                )
            if not verify_sha256(path, str(digest)):
                raise ArtifactVerificationError(
                    f"SHA-256 mismatch for {name} in bundle {bundle_name} "
                    f"(expected {digest}) — refusing to load"
                )
            bundle._digests[name] = str(digest)
        return bundle

    # ── accessors ────────────────────────────────────────────────────────

    @property
    def bundle_name(self) -> str:
        return str(self.manifest.get("bundle", ""))

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", ""))

    @property
    def feature_schema_version(self) -> str:
        return str(self.manifest.get("feature_schema_version", ""))

    @property
    def training_dataset_hash(self) -> str:
        return str(self.manifest.get("training_dataset_hash", ""))

    def file_path(self, name: str) -> str:
        if name not in self._digests:
            raise ArtifactVerificationError(
                f"{name} is not listed in manifest {self.manifest_path}"
            )
        return safe_path_join(self.root, name)

    def listed_files(self) -> list[str]:
        return list(self._digests)

    # ── deserialization (only after verification) ────────────────────────

    def load_joblib(self, name: str, type_check: Callable[[Any], bool] | None = None) -> Any:
        """Load a joblib/pickle artifact whose digest already matched.

        The digest check happened in ``load()``; this method re-verifies
        immediately before deserializing so a file swapped after load()
        cannot be deserialized.
        """
        path = self.file_path(name)
        if not verify_sha256(path, self._digests[name]):
            raise ArtifactVerificationError(
                f"file {path} changed since manifest verification — refusing to load"
            )
        import joblib

        with open(path, "rb") as f:
            raw = joblib.load(f)
        if type_check is not None and not type_check(raw):
            raise ArtifactVerificationError(
                f"object loaded from {name} failed type validation "
                f"({type(raw).__name__})"
            )
        return raw

    def load_torch(self, name: str) -> dict[str, Any]:
        """Load a PyTorch state dict with weights_only=True (never pickle)."""
        path = self.file_path(name)
        if not verify_sha256(path, self._digests[name]):
            raise ArtifactVerificationError(
                f"file {path} changed since manifest verification — refusing to load"
            )
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)

    def load_json(self, name: str) -> dict[str, Any]:
        path = self.file_path(name)
        if not verify_sha256(path, self._digests[name]):
            raise ArtifactVerificationError(
                f"file {path} changed since manifest verification — refusing to load"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.manifest)


def has_predict(obj: Any) -> bool:
    return hasattr(obj, "predict") and hasattr(obj, "predict_proba")


def has_transform(obj: Any) -> bool:
    return hasattr(obj, "transform")


def has_fit_predict(obj: Any) -> bool:
    return hasattr(obj, "predict") and hasattr(obj, "fit")


def is_meta_controller_state(obj: Any) -> bool:
    return isinstance(obj, dict) and "model" in obj and "weights" in obj
