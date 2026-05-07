"""Tests for NK Transformers."""

import pytest
import numpy as np
from pathlib import Path

from src.config import Config, TrainConfig, ExperimentConfig, Paths


class TestConfig:
    """Test Config dataclass."""

    def test_config_from_toml(self, tmp_path):
        config_content = """device = "cuda"

[paths]
cache = "./cache"
checkpoints = "./ckpt"
figures = "./figs"

[training]
epochs = 10
batch_size = 32
learning_rate = 1.0e-3
compile = false

[experiment]
policy_holdout = "high-phi-pi"
n_irf = 5
sample_sizes = [100, 500]
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)

        import tomllib

        with open(config_file, "rb") as f:
            cfg_dict = tomllib.load(f)
        cfg = Config.from_toml(cfg_dict)

        assert cfg.device == "cuda"
        assert cfg.training.epochs == 10
        assert cfg.experiment.policy_holdout == "high-phi-pi"

    def test_config_paths_become_path_objects(self, tmp_path):
        config_content = """device = "cuda"

[paths]
cache = "./cache"
checkpoints = "./ckpt"
figures = "./figs"

[training]
epochs = 10
batch_size = 32
learning_rate = 1.0e-3
compile = false

[experiment]
policy_holdout = "high-phi-pi"
n_irf = 5
sample_sizes = [100]
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)

        import tomllib

        with open(config_file, "rb") as f:
            cfg_dict = tomllib.load(f)
        cfg = Config.from_toml(cfg_dict)

        # Verify paths are Path objects
        assert isinstance(cfg.paths.cache, Path)
        assert isinstance(cfg.paths.checkpoints, Path)
        assert isinstance(cfg.paths.figures, Path)

    def test_config_path_access(self):
        """Test paths can be accessed via dot notation."""
        cfg = Config(
            device="cuda",
            training=TrainConfig(
                epochs=10, batch_size=32, learning_rate=0.001, compile=False
            ),
            experiment=ExperimentConfig(
                policy_holdout="high-phi-pi", n_irf=5, sample_sizes=[100]
            ),
            paths=Paths(
                cache=Path("./cache"),
                checkpoints=Path("./ckpt"),
                figures=Path("./figs"),
            ),
        )

        assert cfg.paths.cache == Path("./cache")
        assert cfg.training.epochs == 10
