import pytest
import numpy as np


class TestVAR:
    """Test VAR functions."""

    def test_var_functions_import(self):
        """Test VAR functions can be imported."""
        from src.benchmarks import fit_var_ols, var_forecast, var_irf

        assert callable(fit_var_ols)
        assert callable(var_forecast)
        assert callable(var_irf)


class TestBenchmarksImport:
    """Test benchmark functions can be imported."""

    def test_var_functions_import(self):
        """Test VAR functions can be imported."""
        from src.benchmarks import fit_var_ols, var_forecast, var_irf

        assert callable(fit_var_ols)
        assert callable(var_forecast)
        assert callable(var_irf)

    def test_bvar_functions_import(self):
        """Test BVAR functions can be imported."""
        from src.benchmarks import bvar_minnesota_fit, bvar_forecast

        assert callable(bvar_minnesota_fit)
        assert callable(bvar_forecast)

    def test_kalman_functions_import(self):
        """Test Kalman functions can be imported."""
        from src.benchmarks import fit_kalman_var, kalman_filter_forecast

        assert callable(fit_kalman_var)
        assert callable(kalman_filter_forecast)
