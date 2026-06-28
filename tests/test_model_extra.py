"""Additional tests for model module to increase coverage."""
import numpy as np
import pytest
import torch
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBuildFunctions:
    def test_build_lstm_returns_module(self):
        from src.model import build_lstm
        model = build_lstm(14)
        assert isinstance(model, torch.nn.Module)

    def test_build_gru_returns_module(self):
        from src.model import build_gru
        model = build_gru(14)
        assert isinstance(model, torch.nn.Module)

    def test_build_transformer_returns_module(self):
        from src.model import build_transformer
        model = build_transformer(14)
        assert isinstance(model, torch.nn.Module)

    def test_build_xgb_has_fit(self):
        from src.model import build_xgb_model
        model = build_xgb_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_lgb_has_fit(self):
        from src.model import build_lgb_model
        model = build_lgb_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


class TestModelDimensions:
    def test_lstm_output_shape(self):
        from src.model import StockLSTM
        model = StockLSTM(input_dim=14)
        x = torch.randn(2, 60, 14)
        out = model(x)
        assert out.shape == (2, 1)

    def test_gru_output_shape(self):
        from src.model import StockGRU
        model = StockGRU(input_dim=14)
        x = torch.randn(2, 60, 14)
        out = model(x)
        assert out.shape == (2, 1)

    def test_transformer_output_shape(self):
        from src.model import StockTransformer
        model = StockTransformer(input_dim=14)
        x = torch.randn(2, 60, 14)
        out = model(x)
        assert out.shape == (2, 1)

    def test_lstm_different_hidden_dims(self):
        from src.model import StockLSTM
        for hidden in [32, 64, 128]:
            model = StockLSTM(input_dim=14, hidden_dim=hidden)
            x = torch.randn(1, 60, 14)
            out = model(x)
            assert out.shape == (1, 1)


class TestSaveLoadRoundtrip:
    def test_lstm_state_dict_roundtrip(self, tmp_path):
        from src.model import StockLSTM
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
        from src.model import StockGRU
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
        from src.model import StockTransformer
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
        from src.model import models_exist
        assert models_exist("NONEXISTENT_XYZ_999.NS") is False
