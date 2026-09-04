"""Tests for the bridge equation."""
import numpy as np
import pytest

from src.analytical.kaufman_roberts import bridge_equation


class TestBridgeEquation:

    def test_bridge_identity_cm_preserves_load(self, identity_cm_3x3, sample_loads_3):
        a_hat = bridge_equation(identity_cm_3x3, sample_loads_3)
        np.testing.assert_allclose(a_hat, sample_loads_3, atol=1e-12)

    def test_bridge_total_load_preserved(self, row_stochastic_cm, sample_loads_3):
        a_hat = bridge_equation(row_stochastic_cm, sample_loads_3)
        assert abs(a_hat.sum() - sample_loads_3.sum()) < 1e-12, (
            f"Load not conserved: {a_hat.sum():.15f} != {sample_loads_3.sum():.15f}"
        )

    def test_bridge_rejects_negative_entries(self, sample_loads_2):
        C_neg = np.array([[1.1, -0.1], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            bridge_equation(C_neg, sample_loads_2)

    def test_bridge_uniform_cm_collapses_to_weighted_mean(self):
        """Uniform off-diagonal CM: each a_hat_k = r*a_k + (1-r)/(K-1)*sum(a_j for j!=k)."""
        K = 3
        r = 0.7
        a = np.array([10.0, 6.0, 4.0])
        C = np.full((K, K), (1 - r) / (K - 1))
        np.fill_diagonal(C, r)
        a_hat = bridge_equation(C, a)
        expected = np.array([
            r * a[k] + (1 - r) / (K - 1) * (a.sum() - a[k])
            for k in range(K)
        ])
        np.testing.assert_allclose(a_hat, expected, atol=1e-12)
