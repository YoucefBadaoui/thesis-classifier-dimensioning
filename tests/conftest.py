"""Shared fixtures for the analytical test suite."""
import numpy as np
import pytest


@pytest.fixture
def identity_cm_3x3():
    return np.eye(3)


@pytest.fixture
def sample_loads_2():
    return np.array([5.0, 3.0])


@pytest.fixture
def sample_loads_3():
    return np.array([10.0, 6.0, 4.0])


@pytest.fixture
def sample_demands_2():
    return np.array([1, 2])


@pytest.fixture
def sample_demands_3():
    return np.array([1, 2, 4])


@pytest.fixture
def small_V():
    return 20


@pytest.fixture
def medium_V():
    return 100


@pytest.fixture
def row_stochastic_cm():
    """A valid 3x3 row-stochastic CM with off-diagonal entries."""
    return np.array([
        [0.90, 0.05, 0.05],
        [0.10, 0.85, 0.05],
        [0.05, 0.10, 0.85],
    ])
