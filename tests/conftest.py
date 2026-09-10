"""Shared fixtures for the paracellular_transport test suite."""
import pytest

from paracellular_transport.config import PHYSICAL_CONSTANTS


@pytest.fixture
def constants():
    """Physical constants (R, F, Avogadro) used throughout the model."""
    return PHYSICAL_CONSTANTS


@pytest.fixture
def temperature():
    """A representative physiological temperature (K), independent of any scenario."""
    return 310.0
