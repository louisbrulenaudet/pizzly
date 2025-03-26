"""Test suite for pizzly package initialization."""

import pizzly


def test_version():
    """Test version is accessible."""
    assert hasattr(pizzly, "__version__")
    assert isinstance(pizzly.__version__, str)


def test_main_components():
    """Test main components are accessible."""
    assert hasattr(pizzly, "AlpacaStock")
    assert hasattr(pizzly, "FinancialTool")
