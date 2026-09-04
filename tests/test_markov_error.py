"""Acceptance tests for the two-state Markov classifier-error process.

The correlated-error sweep of Chapter 6 needs three properties of ``monte_carlo.markov_error``: the marginal error probability holds at every correlation level, so the marginal confusion matrix is unchanged; the lag-1 autocorrelation of the error indicator equals the requested rho; and error clusters have the geometric mean length the construction implies.
"""


import numpy as np
import pytest

from src.monte_carlo.markov_error import (
    markov_transitions,
    mean_cluster_length,
    simulate_fag_markov,
)

C_3CLASS = np.array([
    [0.80, 0.12, 0.08],
    [0.10, 0.85, 0.05],
    [0.07, 0.13, 0.80],
])
A_TRUE = np.array([20.0, 15.0, 10.0])
DEMANDS = np.array([1, 4, 10], dtype=np.int64)
V = 60


def test_transition_probabilities_are_stationary():
    """P(1|1) and P(1|0) must give stationary marginal p for any rho."""
    p = np.array([0.2, 0.15, 0.2])
    for rho in (0.0, 0.3, 0.6, 0.9):
        p11, p01 = markov_transitions(p, rho)
        # Stationary distribution of a two-state chain: pi_1 = p01 / (1 - p11 + p01)
        stationary = p01 / (1.0 - p11 + p01)
        assert np.allclose(stationary, p), f"rho={rho} breaks stationarity"


def test_rho_zero_gives_memoryless_transitions():
    p = np.array([0.2, 0.15, 0.2])
    p11, p01 = markov_transitions(p, 0.0)
    assert np.allclose(p11, p)
    assert np.allclose(p01, p)


def test_rho_out_of_range_rejected():
    with pytest.raises(ValueError):
        markov_transitions(np.array([0.2]), 1.0)
    with pytest.raises(ValueError):
        markov_transitions(np.array([0.2]), -0.1)


def test_analytical_cluster_length():
    """Mean error run is geometric with mean 1 / ((1 - p)(1 - rho))."""
    p = np.array([0.2, 0.5])
    assert np.allclose(mean_cluster_length(p, 0.0), 1.0 / (1.0 - p))
    assert np.allclose(mean_cluster_length(p, 0.6), 1.0 / ((1.0 - p) * 0.4))


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6])
def test_simulated_process_matches_its_targets(rho):
    """The realised marginal, autocorrelation and cluster length hit their targets."""
    _, _, _, diag = simulate_fag_markov(
        V, A_TRUE, C_3CLASS, rho, DEMANDS, n_arrivals=400_000, seed=11
    )
    assert np.allclose(diag["p_measured"], diag["p_target"], atol=5e-3), \
        "marginal error rate drifted, so the marginal matrix is not preserved"
    assert np.allclose(diag["rho_measured"], rho, atol=1e-2), \
        "realised lag-1 autocorrelation does not match the requested rho"
    assert np.allclose(
        diag["cluster_measured"], diag["cluster_target"], rtol=5e-2
    ), "measured error-cluster length does not match the geometric prediction"


def test_marginal_matrix_is_preserved_across_rho():
    """Per-class error rates agree across rho, which is the invariance claimed.

    Looser than the 5e-3 of the per-rho test above: this compares two sampled estimates against each other, so both carry Monte Carlo error.
    """
    measured = []
    for rho in (0.0, 0.3, 0.6):
        _, _, _, diag = simulate_fag_markov(
            V, A_TRUE, C_3CLASS, rho, DEMANDS, n_arrivals=400_000, seed=5
        )
        measured.append(diag["p_measured"])
    for later in measured[1:]:
        assert np.allclose(measured[0], later, atol=6e-3)
