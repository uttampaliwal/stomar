"""Regression tests for manifest-aware model persistence.

Lock in the fix: the trainer's tree-only fallback used to write raw,
unmanifested joblib pickles — unverifiable, unpinnable artifacts that
bypass the SHA-256 manifest gate. Every bundle must now go through
``save_models`` and ship a manifest listing exactly the artifacts that
exist (DL artifacts omitted for tree-only bundles).
"""

import pytest

from src.models.artifacts import ArtifactBundle, ArtifactVerificationError
from src.models.model import save_models


class _DummyModel:
    def __init__(self, tag: str = "x"):
        self.tag = tag

    def predict(self, x):
        return x


class _DummyScaler:
    def transform(self, x):
        return x


class _FakeLSTM:
    class _Inner:
        def __init__(self):
            self.input_size = 3

    def __init__(self):
        self.lstm = self._Inner()

    def state_dict(self):
        return {"fake": 1}


def test_tree_only_save_manifests_only_existing_artifacts(tmp_path):
    ticker = "DUMMY_NS"
    save_models(None, None, None, _DummyModel("xgb"), _DummyScaler(),
                ["close", "rsi"], ticker,
                lgb_model=_DummyModel("lgb"), cat_model=_DummyModel("cat"),
                model_version="2", root=str(tmp_path))

    bundle = ArtifactBundle.for_ticker(str(tmp_path), ticker)
    files = bundle.listed_files()
    for ext in ("xgb.pkl", "lgb.pkl", "cat.pkl", "scaler.pkl", "features.pkl"):
        assert f"{ticker}_{ext}" in files
    # DL artifacts must never be written or manifested for a tree-only bundle
    for ext in ("lstm.pt", "gru.pt", "transformer.pt", "lstm_dim.pkl"):
        assert f"{ticker}_{ext}" not in files
        assert not (tmp_path / f"{ticker}_{ext}").exists()
    assert bundle.manifest.get("model_version") == "2"
    # nothing exists on disk outside the manifest
    on_disk = {p.name for p in tmp_path.iterdir() if p.suffix in (".pkl", ".pt")}
    assert on_disk == {f"{ticker}_{ext}" for ext in
                       ("xgb.pkl", "lgb.pkl", "cat.pkl", "scaler.pkl", "features.pkl")}
    # and the tree-only bundle reloads hash-verified
    assert bundle.load_joblib(f"{ticker}_xgb.pkl").tag == "xgb"


def test_full_bundle_save_manifests_dl_artifacts_too(tmp_path):
    ticker = "DUMMY2_NS"
    save_models(_FakeLSTM(), _FakeLSTM(), _FakeLSTM(), _DummyModel("xgb"),
                _DummyScaler(), ["close"], ticker,
                model_version="2", root=str(tmp_path))

    bundle = ArtifactBundle.for_ticker(str(tmp_path), ticker)
    files = bundle.listed_files()
    for ext in ("lstm.pt", "gru.pt", "transformer.pt", "lstm_dim.pkl",
                "xgb.pkl", "scaler.pkl", "features.pkl"):
        assert f"{ticker}_{ext}" in files


def test_manifest_hash_verifies_each_artifact(tmp_path):
    ticker = "DUMMY3_NS"
    save_models(None, None, None, _DummyModel("xgb"), _DummyScaler(),
                ["close"], ticker, root=str(tmp_path))
    path = tmp_path / f"{ticker}_xgb.pkl"
    assert path.exists()
    original = path.read_bytes()
    path.write_bytes(original + b"tampered")
    with pytest.raises(ArtifactVerificationError):
        # for_ticker() itself verifies every listed artifact; a tampered
        # bundle must be refused at ANY load attempt
        bundle = ArtifactBundle.for_ticker(str(tmp_path), ticker)
        bundle.load_joblib(f"{ticker}_xgb.pkl")
