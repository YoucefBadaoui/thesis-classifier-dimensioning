"""Tests for the sensitivity tensor."""
import numpy as np

from src.analytical.kaufman_roberts import kaufman_roberts, sensitivity_analysis


class TestSensitivity:

    def test_matches_finite_differences(
        self, medium_V, sample_loads_3, sample_demands_3, row_stochastic_cm
    ):
        """S[k, i, j] = a_i (dB_k/da_j - dB_k/da_i) at the apparent load."""
        a, t, C = sample_loads_3, sample_demands_3, row_stochastic_cm
        S = sensitivity_analysis(medium_V, a, C, t)
        a_hat = C.T @ a
        K = len(a)
        h = 1e-4
        grad = np.zeros((K, K))            # grad[k, j] = dB_k / da_hat_j
        for j in range(K):
            up = a_hat.copy(); up[j] += h
            dn = a_hat.copy(); dn[j] -= h
            grad[:, j] = (kaufman_roberts(medium_V, up, t)[1]
                          - kaufman_roberts(medium_V, dn, t)[1]) / (2 * h)
        for k in range(K):
            for i in range(K):
                for j in range(K):
                    expected = a[i] * (grad[k, j] - grad[k, i])
                    assert abs(S[k, i, j] - expected) < 1e-4, (k, i, j, S[k, i, j], expected)

    def test_sensitivity_at_identity_cm_is_finite(
        self, medium_V, sample_loads_3, sample_demands_3, identity_cm_3x3
    ):
        """An identity C makes a_hat == a, so the gradient is taken at the undistorted point."""
        S = sensitivity_analysis(medium_V, sample_loads_3, identity_cm_3x3,
                                 sample_demands_3)
        assert np.all(np.isfinite(S))
