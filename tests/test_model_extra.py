"""Additional tests for model module to increase coverage."""
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBuildFunctions:
    def test_build_lstm_returns_module(self):
        from src.models.model import build_lstm
        model = build_lstm(14)
        assert isinstance(model, torch.nn.Module)

    def test_build_gru_returns_module(self):
        from src.models.model import build_gru
        model = build_gru(14)
        assert isinstance(model, torch.nn.Module)

    def test_build_transformer_returns_module(self):
        from src.models.model import build_transformer
        model = build_transformer(14)
        assert isinstance(model, torch.nn.Module)

    def test_build_xgb_has_fit(self):
        from src.models.model import build_xgb_model
        model = build_xgb_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_lgb_has_fit(self):
        from src.models.model import build_lgb_model
        model = build_lgb_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


class TestModelDimensions:
    def test_lstm_output_shape(self):
        from src.models.model import StockLSTM
        model = StockLSTM(input_dim=14)
        x = torch.randn(2, 60, 14)
        out = model(x)
        assert out.shape == (2, 1)

    def test_gru_output_shape(self):
        from src.models.model import StockGRU
        model = StockGRU(input_dim=14)
        x = torch.randn(2, 60, 14)
        out = model(x)
        assert out.shape == (2, 1)

    def test_transformer_output_shape(self):
        from src.models.model import StockTransformer
        model = StockTransformer(input_dim=14)
        x = torch.randn(2, 60, 14)
        out = model(x)
        assert out.shape == (2, 1)

    def test_lstm_different_hidden_dims(self):
        from src.models.model import StockLSTM
        for hidden in [32, 64, 128]:
            model = StockLSTM(input_dim=14, hidden_dim=hidden)
            x = torch.randn(1, 60, 14)
            out = model(x)
            assert out.shape == (1, 1)


class TestSaveLoadRoundtrip:
    def test_lstm_state_dict_roundtrip(self, tmp_path):
        from src.models.model import StockLSTM
        model = StockLSTM(input_dim=14)
        x = torch.randn(1, 60, 14)
        model.eval()
        with torch.no_grad():
            pred_before = model(x).item()

        path = tmp_path / "test.pt"
        torch.save(model.state_dict(), path)

        loaded = StockLSTM(input_dim=14)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        loaded.eval()
        with torch.no_grad():
            pred_after = loaded(x).item()

        assert abs(pred_before - pred_after) < 1e-6

    def test_gru_state_dict_roundtrip(self, tmp_path):
        from src.models.model import StockGRU
        model = StockGRU(input_dim=14)
        x = torch.randn(1, 60, 14)
        model.eval()
        with torch.no_grad():
            pred_before = model(x).item()

        path = tmp_path / "test.pt"
        torch.save(model.state_dict(), path)

        loaded = StockGRU(input_dim=14)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        loaded.eval()
        with torch.no_grad():
            pred_after = loaded(x).item()

        assert abs(pred_before - pred_after) < 1e-6

    def test_transformer_state_dict_roundtrip(self, tmp_path):
        from src.models.model import StockTransformer
        model = StockTransformer(input_dim=14)
        x = torch.randn(1, 60, 14)
        model.eval()
        with torch.no_grad():
            pred_before = model(x).item()

        path = tmp_path / "test.pt"
        torch.save(model.state_dict(), path)

        loaded = StockTransformer(input_dim=14)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        loaded.eval()
        with torch.no_grad():
            pred_after = loaded(x).item()

        assert abs(pred_before - pred_after) < 1e-6


class TestModelsExist:
    def test_nonexistent_ticker(self):
        from src.models.model import models_exist
        assert models_exist("NONEXISTENT_XYZ_999.NS") is False

    def test_zero_byte_required_file_fails(self, tmp_path):
        """A zero-byte file must not pass models_exist (would crash on load)."""

        from src.models.artifacts import ArtifactBundle
        from src.models.model import models_exist, FEATURE_SCHEMA_VERSION

        ticker = "ZEROBYTE.NS"
        root = str(tmp_path)
        ticker_clean = ticker.replace(".", "_")
        for name, payload in [
            ("lstm.pt", b"\x80\x02empty-state"),      # non-empty placeholder
            ("gru.pt", b"\x80\x02empty-state"),
            ("transformer.pt", b"\x80\x02empty-state"),
            ("lstm_dim.pkl", b"\x80\x02dim8"),
            ("xgb.pkl", b"\x80\x02empty-xgb"),
            ("scaler.pkl", b"\x80\x02empty-scaler"),
            ("features.pkl", b"\x80\x02feat"),
        ]:
            with open(os.path.join(root, f"{ticker_clean}_{name}"), "wb") as f:
                f.write(payload)
        ArtifactBundle.create(root, ticker_clean, {
            f"{ticker_clean}_{ext}": "" for ext in
            ("lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl",
             "scaler.pkl", "features.pkl", "lstm_dim.pkl")
        }, feature_schema_version=FEATURE_SCHEMA_VERSION)
        assert models_exist(ticker, root=root) is True

        # Zero out one required file — digest still matches (manifest was
        # created AFTER truncation), but the file is unloadable garbage.
        with open(os.path.join(root, f"{ticker_clean}_xgb.pkl"), "wb"):
            pass
        assert models_exist(ticker, root=root) is False


class TestPromoteModelCleanup:
    def test_promote_removes_stale_artifacts(self, tmp_path, monkeypatch):
        """Old production artifacts not in the new manifest are removed."""

        from unittest.mock import patch
        from src.models.artifacts import ArtifactBundle
        from src.models.model import models_exist

        # Build a source bundle under a temp "models" dir.
        root = str(tmp_path / "models")
        prod_dir = os.path.join(root, "production")
        os.makedirs(prod_dir, exist_ok=True)
        ticker = "CLEANUP.NS"
        ticker_clean = ticker.replace(".", "_")
        for name, payload in [
            ("lstm.pt", b"\x80\x02empty-state"),
            ("gru.pt", b"\x80\x02empty-state"),
            ("transformer.pt", b"\x80\x02empty-state"),
            ("lstm_dim.pkl", b"\x80\x02dim8"),
            ("xgb.pkl", b"\x80\x02empty-xgb"),
            ("scaler.pkl", b"\x80\x02empty-scaler"),
            ("features.pkl", b"\x80\x02feat"),
        ]:
            with open(os.path.join(root, f"{ticker_clean}_{name}"), "wb") as f:
                f.write(payload)
        ArtifactBundle.create(root, ticker_clean, {
            f"{ticker_clean}_{ext}": "" for ext in
            ("lstm.pt", "gru.pt", "transformer.pt", "xgb.pkl",
             "scaler.pkl", "features.pkl", "lstm_dim.pkl")
        })

        # Seed production with a stale legacy artifact from an older version.
        stale = os.path.join(prod_dir, f"{ticker_clean}_oldmodel.pt")
        with open(stale, "wb") as f:
            f.write(b"stale")

        from src.models import model as model_mod
        monkeypatch.setattr(model_mod, "MODELS_DIR", root)
        with patch("src.models.model_registry.ModelRegistry") as registry_cls:
            registry_cls.return_value.register.return_value.version = 1
            result = model_mod.promote_model(ticker)

        assert result["status"] == "promoted"
        assert os.path.exists(os.path.join(prod_dir, f"{ticker_clean}_xgb.pkl"))
        assert not os.path.exists(stale), "stale production artifact must be removed"
        assert models_exist(ticker, root=prod_dir) is True
