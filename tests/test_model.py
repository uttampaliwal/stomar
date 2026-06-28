"""Tests for model save/load roundtrip and architectures."""
import numpy as np
import pytest
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestModelArchitectures:
    """Test neural network model construction and forward pass."""

    def test_lstm_forward_pass(self):
        from src.model import StockLSTM
        model = StockLSTM(input_dim=14)
        x = torch.randn(1, 60, 14)
        out = model(x)
        assert out.shape == (1, 1)

    def test_gru_forward_pass(self):
        from src.model import StockGRU
        model = StockGRU(input_dim=14)
        x = torch.randn(1, 60, 14)
        out = model(x)
        assert out.shape == (1, 1)

    def test_transformer_forward_pass(self):
        from src.model import StockTransformer
        model = StockTransformer(input_dim=14)
        x = torch.randn(1, 60, 14)
        out = model(x)
        assert out.shape == (1, 1)

    def test_build_xgb_model(self):
        from src.model import build_xgb_model
        model = build_xgb_model()
        assert hasattr(model, "fit")

    def test_build_lgb_model(self):
        from src.model import build_lgb_model
        model = build_lgb_model()
        assert hasattr(model, "fit")


class TestModelSaveLoad:
    """Test save/load roundtrip preserves model state."""

    def test_lstm_save_load_roundtrip(self, tmp_path):
        from src.model import StockLSTM, save_models, load_models
        model = StockLSTM(input_dim=14)
        x = torch.randn(1, 60, 14)
        model.eval()
        with torch.no_grad():
            pred_before = model(x).item()

        # We can't easily test full save_models since it needs all model types
        # but we can test state dict roundtrip
        state = model.state_dict()
        path = tmp_path / "test_lstm.pt"
        torch.save(state, path)

        loaded = StockLSTM(input_dim=14)
        loaded.load_state_dict(torch.load(path, weights_only=True))
        loaded.eval()
        with torch.no_grad():
            pred_after = loaded(x).item()

        assert abs(pred_before - pred_after) < 1e-6

    def test_build_lstm(self):
        from src.model import build_lstm
        model = build_lstm(input_dim=14)
        assert isinstance(model, torch.nn.Module)
