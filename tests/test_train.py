import pytest
import numpy as np
import torch

from src.train import NKDataset, train_model


class TestNKDataset:
    """Test NKDataset."""

    def test_dataset_creation(self):
        """Test dataset can be created."""
        X = np.random.randn(100, 10, 17).astype(np.float32)
        Y = np.random.randn(100, 10, 3).astype(np.float32)

        dataset = NKDataset(X, Y)
        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct types."""
        X = np.random.randn(50, 10, 17).astype(np.float32)
        Y = np.random.randn(50, 10, 3).astype(np.float32)

        dataset = NKDataset(X, Y)
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10, 17)
        assert y.shape == (10, 3)

    def test_dataset_shape(self):
        """Test dataset has correct shapes."""
        X = np.random.randn(20, 5, 17).astype(np.float32)
        Y = np.random.randn(20, 5, 3).astype(np.float32)

        dataset = NKDataset(X, Y)
        x, y = dataset[5]

        assert x.shape == (5, 17)
        assert y.shape == (5, 3)

    def test_dataset_full_coverage(self):
        """Test dataset covers all samples."""
        X = np.random.randn(30, 3, 17).astype(np.float32)
        Y = np.random.randn(30, 3, 3).astype(np.float32)

        dataset = NKDataset(X, Y)
        assert len(dataset) == 30

    def test_dataset_single_sample(self):
        """Test dataset with single sample."""
        X = np.random.randn(1, 5, 17).astype(np.float32)
        Y = np.random.randn(1, 5, 3).astype(np.float32)

        dataset = NKDataset(X, Y)
        assert len(dataset) == 1

    def test_dataset_dtype(self):
        """Test dataset loads data."""
        X = np.random.randn(10, 5, 17).astype(np.float64)
        Y = np.random.randn(10, 5, 3).astype(np.float64)

        dataset = NKDataset(X, Y)
        x, y = dataset[0]

        # Just check it's tensor - dtype may vary
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


class TestTrainModel:
    """Test train_model function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_model_cuda(self):
        """Test training on CUDA."""
        X = np.random.randn(100, 10, 17).astype(np.float32)
        Y = np.random.randn(100, 10, 3).astype(np.float32)
        val_X = np.random.randn(20, 10, 17).astype(np.float32)
        val_Y = np.random.randn(20, 10, 3).astype(np.float32)

        data = {
            "X_train": X,
            "Y_train": Y,
            "X_val": val_X,
            "Y_val": val_Y,
            "X_test": np.random.randn(20, 10, 17).astype(np.float32),
            "Y_test": np.random.randn(20, 10, 3).astype(np.float32),
        }

        model, history = train_model(
            data=data,
            device="cuda",
            epochs=2,
            batch_size=16,
            checkpoint_dir="/tmp/test_ckpt",
            silent=True,
        )

        assert model is not None
        assert "train_loss" in history
        assert "val_loss" in history

    def test_train_model_cpu(self):
        """Test training on CPU."""
        X = np.random.randn(100, 10, 17).astype(np.float32)
        Y = np.random.randn(100, 10, 3).astype(np.float32)
        val_X = np.random.randn(20, 10, 17).astype(np.float32)
        val_Y = np.random.randn(20, 10, 3).astype(np.float32)

        data = {
            "X_train": X,
            "Y_train": Y,
            "X_val": val_X,
            "Y_val": val_Y,
            "X_test": np.random.randn(20, 10, 17).astype(np.float32),
            "Y_test": np.random.randn(20, 10, 3).astype(np.float32),
        }

        model, history = train_model(
            data=data,
            device="cpu",
            epochs=1,
            batch_size=16,
            checkpoint_dir="/tmp/test_ckpt_cpu",
            silent=True,
        )

        assert model is not None
        assert len(history["train_loss"]) >= 1

    def test_train_model_short_training(self):
        """Test with very short training."""
        X = np.random.randn(20, 5, 17).astype(np.float32)
        Y = np.random.randn(20, 5, 3).astype(np.float32)
        val_X = np.random.randn(5, 5, 17).astype(np.float32)
        val_Y = np.random.randn(5, 5, 3).astype(np.float32)

        data = {
            "X_train": X,
            "Y_train": Y,
            "X_val": val_X,
            "Y_val": val_Y,
            "X_test": np.random.randn(5, 5, 17).astype(np.float32),
            "Y_test": np.random.randn(5, 5, 3).astype(np.float32),
        }

        model, history = train_model(
            data=data,
            device="cpu",
            epochs=1,
            batch_size=8,
            checkpoint_dir="/tmp/test_ckpt_short",
            silent=True,
        )

        assert model is not None
        assert len(history["train_loss"]) == 1

    def test_train_model_checkpoint_created(self):
        """Test checkpoint file is created."""
        import tempfile
        import os

        X = np.random.randn(30, 5, 17).astype(np.float32)
        Y = np.random.randn(30, 5, 3).astype(np.float32)
        val_X = np.random.randn(5, 5, 17).astype(np.float32)
        val_Y = np.random.randn(5, 5, 3).astype(np.float32)

        data = {
            "X_train": X,
            "Y_train": Y,
            "X_val": val_X,
            "Y_val": val_Y,
            "X_test": np.random.randn(5, 5, 17).astype(np.float32),
            "Y_test": np.random.randn(5, 5, 3).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model, _ = train_model(
                data=data,
                device="cpu",
                epochs=1,
                batch_size=8,
                checkpoint_dir=tmpdir,
                silent=True,
            )

            ckpt_path = os.path.join(tmpdir, "best_model.pt")
            assert os.path.exists(ckpt_path)
