"""The grid and bisection forms of r*_k agree, and the isolated-class matrix is well formed."""

import numpy as np

from src.analytical.constants import A_OTT, T_OTT, V_NOMINAL_OTT, B_TARGET_DEFAULT
from src.analytical.recall_thresholds import (
    isolated_class_cm,
    per_class_recall_search,
    rstar_per_class,
)


def test_isolated_class_cm_is_row_stochastic():
    C = isolated_class_cm(5, 2, 0.8)
    assert np.allclose(C.sum(axis=1), 1.0)
    assert C[2, 2] == 0.8 and np.allclose(np.delete(C[2], 2), 0.05)
    assert np.allclose(np.delete(C, 2, axis=0), np.delete(np.eye(5), 2, axis=0))


def test_grid_and_bisection_agree_on_the_ott_scenario():
    eps = 0.05
    grid = per_class_recall_search(A_OTT, T_OTT, V_NOMINAL_OTT, B_TARGET_DEFAULT, eps)
    bisect = rstar_per_class(A_OTT, T_OTT, V_NOMINAL_OTT, eps)
    assert np.all(np.abs(grid - bisect) <= 0.006), (grid, bisect)
