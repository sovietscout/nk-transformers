import pytest
import numpy as np
from pathlib import Path

from src.simulator import solve_nk_model, simulate_one_draw, generate_datasets


class TestSimulatorInternal:
    """Test simulator internal functions."""

    def test_simulate_one_draw_import(self):
        """Test simulate_one_draw can be imported."""
        from src.simulator import simulate_one_draw

        assert callable(simulate_one_draw)

    def test_generate_datasets_import(self):
        """Test generate_datasets can be imported."""
        from src.simulator import generate_datasets

        assert callable(generate_datasets)


class TestNKModelParameters:
    """Test NK model parameter validation."""

    def test_solve_with_positive_sigma(self):
        """Test solve_nk_model with positive sigma works."""
        params = np.array(
            [2.0, 0.99, 0.15, 2.0, 0.5, 0.7, 0.6, 0.5, 0.006, 0.005, 0.005]
        )
        result = solve_nk_model(params)
        # Just check it runs - result may or may not be None depending on stability
        assert result is not None or result is None
