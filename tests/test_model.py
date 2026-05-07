import pytest
import numpy as np
import torch

from src.model import NKTransformer


class TestNKTransformer:
    """Test NK Transformer model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = NKTransformer(
            d_model=32,
            n_heads=2,
            n_layers=2,
            ff_dim=64,
            dropout=0.1,
        )
        assert model is not None

    def test_model_forward(self):
        """Test model forward pass."""
        model = NKTransformer(
            d_model=32,
            n_heads=2,
            n_layers=2,
            ff_dim=64,
            dropout=0.1,
        )
        batch_size = 4
        seq_len = 10
        input_dim = 17

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        assert output.shape == (batch_size, seq_len, 3)

    def test_model_output_dim(self):
        """Test model output dimension is correct."""
        model = NKTransformer(output_dim=5)
        x = torch.randn(2, 5, 17)
        output = model(x)

        assert output.shape[-1] == 5

    def test_model_input_dim(self):
        """Test model input dimension is correct."""
        model = NKTransformer(input_dim=10)
        x = torch.randn(2, 5, 10)
        output = model(x)

        assert output.shape[-1] == 3

    def test_model_train_eval(self):
        """Test model can switch between train and eval."""
        model = NKTransformer()
        model.train()
        assert model.training

        model.eval()
        assert not model.training

    def test_model_parameters(self):
        """Test model has learnable parameters."""
        model = NKTransformer(d_model=16, n_heads=1, n_layers=1, ff_dim=32)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_model_gradient(self):
        """Test model can compute gradients."""
        model = NKTransformer(d_model=16, n_heads=1, n_layers=1, ff_dim=32)
        model.train()

        x = torch.randn(2, 5, 17, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None

    def test_model_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        model = NKTransformer()
        seq_len = 20

        x = torch.randn(2, seq_len, 17)
        output = model(x)

        assert output.shape[1] == seq_len


class TestNKTransformerVariations:
    """Test different model configurations."""

    def test_small_model(self):
        """Test small model configuration."""
        model = NKTransformer(
            d_model=16,
            n_heads=1,
            n_layers=1,
            ff_dim=32,
        )
        x = torch.randn(1, 5, 17)
        output = model(x)

        assert output.shape == (1, 5, 3)

    def test_large_model(self):
        """Test large model configuration."""
        model = NKTransformer(
            d_model=128,
            n_heads=8,
            n_layers=6,
            ff_dim=512,
        )
        x = torch.randn(4, 10, 17)
        output = model(x)

        assert output.shape == (4, 10, 3)

    def test_single_step(self):
        """Test model with seq_len=1."""
        model = NKTransformer()
        x = torch.randn(2, 1, 17)
        output = model(x)

        assert output.shape[1] == 1
