"""Tests for the Kaufman-Roberts recursion."""
import numpy as np

from src.analytical.kaufman_roberts import kaufman_roberts


def erlang_b(a: float, V: int) -> float:
    """Erlang-B formula via normalised recursion (Jagerman 1974)."""
    inv = 1.0
    for n in range(1, V + 1):
        inv = 1.0 + inv * n / a
    return 1.0 / inv


class TestKaufmanRoberts:

    def test_kr_single_class_matches_erlang_b(self):
        V = 10
        a = 5.0
        loads = np.array([a])
        demands = np.array([1])
        _, blocking = kaufman_roberts(V, loads, demands)
        expected = erlang_b(a, V)
        assert abs(blocking[0] - expected) < 1e-9, (
            f"KR blocking {blocking[0]:.12f} != Erlang-B {expected:.12f}"
        )

    def test_kr_state_probabilities_normalise(self, small_V, sample_loads_2, sample_demands_2):
        P, _ = kaufman_roberts(small_V, sample_loads_2, sample_demands_2)
        assert abs(P.sum() - 1.0) < 1e-9, f"P sums to {P.sum():.12f}, expected 1"

    def test_kr_zero_load_returns_zero_blocking(self, small_V, sample_demands_2):
        loads_zero = np.array([0.0, 0.0])
        _, blocking = kaufman_roberts(small_V, loads_zero, sample_demands_2)
        assert np.all(blocking == 0.0), f"Expected zero blocking, got {blocking}"

    def test_kr_blocking_monotone_in_V(self, sample_loads_2, sample_demands_2):
        """Blocking B_k is monotone non-increasing in V at fixed loads."""
        prev_blocking = None
        for V in range(5, 30, 5):
            _, blocking = kaufman_roberts(V, sample_loads_2, sample_demands_2)
            if prev_blocking is not None:
                assert np.all(blocking <= prev_blocking + 1e-12), (
                    f"Non-monotone blocking at V={V}: {blocking} > {prev_blocking}"
                )
            prev_blocking = blocking.copy()
