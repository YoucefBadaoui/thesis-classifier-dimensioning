"""Kaufman-Roberts recursion, bridge equation, and the capacity searches built on them.

Notation follows Kaufman (1981) and Stasiak et al. (2011): V capacity in allocation units (AU), a_k offered load (Erlangs), t_k AU demand, B_k per-class blocking probability, y_k(n) proportional approximation of class-k calls at state n, sigma_k state-dependent load in the BPP extension. Intermediate q(n) values overflow float64 once max(a_k * t_k) passes roughly 1000, and the result then comes back as NaN rather than raising; a log-space recursion would be needed there. The thesis scenarios run at V = 277 to 1809 with max(a_k * t_k) at most 322, well inside the safe range.
"""

import numpy as np

from .constants import (
    R_STEPS_SYSTEM,
    V_SEARCH_MAX,
)


def kaufman_roberts(V: int, loads: np.ndarray,
                    demands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaufman-Roberts recursion for a Full-Availability Group, Poisson traffic.

    Recursion: n * q(n) = sum_k a_k * t_k * q(n - t_k),  n = 1, ..., V
    Blocking:  B_k = sum_{n=V-t_k+1}^{V} P(n)

    Returns (P, blocking): the normalised state distribution over 0..V and the per-class B_k.
    """
    K = len(loads)
    assert np.all(demands >= 1), "All demands must be >= 1"

    q = np.zeros(V + 1)
    q[0] = 1.0

    for n in range(1, V + 1):
        s = 0.0
        for k in range(K):
            t_k = int(demands[k])
            if n - t_k >= 0:
                s += loads[k] * t_k * q[n - t_k]
        q[n] = s / n

    total = np.sum(q)
    P = q / total

    return P, _tail_blocking(P, V, demands)


def _tail_blocking(P: np.ndarray, V: int, demands: np.ndarray) -> np.ndarray:
    """B_k = sum_{n=V-t_k+1}^{V} P(n) for every class."""
    K = len(demands)
    blocking = np.zeros(K)
    for k in range(K):
        t_k = int(demands[k])
        start = max(V - t_k + 1, 0)
        blocking[k] = np.sum(P[start: V + 1])
    return blocking


def _blocking_gradient(V: int, a_hat: np.ndarray, demands: np.ndarray,
                       delta: float) -> np.ndarray:
    """Central-difference dB_k/da_hat_j, shape (K, K), step max(|a_hat_j| delta, delta)."""
    K = len(a_hat)
    dB_da = np.zeros((K, K))
    for j in range(K):
        a_hat_plus = a_hat.copy()
        a_hat_minus = a_hat.copy()
        step = max(abs(a_hat[j]) * delta, delta)
        a_hat_plus[j] += step
        a_hat_minus[j] -= step

        _, B_plus = kaufman_roberts(V, a_hat_plus, demands)
        _, B_minus = kaufman_roberts(V, a_hat_minus, demands)

        dB_da[:, j] = (B_plus - B_minus) / (2 * step)
    return dB_da


def uniform_spillover_cm(K: int, r: float) -> np.ndarray:
    """Confusion matrix with recall r on every class and the error mass (1 - r) spread uniformly over the other K - 1 classes."""
    C = np.full((K, K), (1.0 - r) / (K - 1))
    np.fill_diagonal(C, r)
    return C


def row_normalise(counts: np.ndarray) -> np.ndarray:
    """Row-stochastic form of a count matrix; all-zero rows stay zero."""
    counts = np.asarray(counts, dtype=np.float64)
    sums = counts.sum(axis=1, keepdims=True)
    return counts / np.where(sums == 0, 1.0, sums)


def fix_zero_rows(C: np.ndarray) -> np.ndarray:
    """Copy of C with every all-zero row replaced by its identity row, so a class absent from an evaluation counts as perfectly classified."""
    C_fixed = np.array(C, dtype=np.float64, copy=True)
    for i in range(C_fixed.shape[0]):
        if np.sum(C_fixed[i, :]) < 1e-10:
            C_fixed[i, i] = 1.0
    return C_fixed


def population_covariance(a: np.ndarray, t: np.ndarray) -> float:
    """cov(a, t) with the population normalisation (ddof = 0)."""
    a = np.asarray(a, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return float(np.mean(a * t) - np.mean(a) * np.mean(t))


def bridge_equation(C: np.ndarray, a: np.ndarray, normalise: bool = False) -> np.ndarray:
    """Bridge equation a_hat = C^T @ a, with C_ij = P(predict j | true i).

    Row sums are checked at atol=1e-9. CMs read off published heatmaps at three-decimal precision deviate by about 1e-3, so they must be renormalised by the caller (normalise=True) or by published_cms.validate_cm before reaching this entry point. Set normalise=True to row-normalise C instead of raising when a row misses 1.0.

    Returns a_hat, the distorted offered load vector of shape (K,).
    """
    if (C < 0).any():
        raise ValueError(
            "confusion matrix entries must be non-negative; got min "
            f"{C.min():.6g}"
        )
    row_sums = C.sum(axis=1)
    bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-9, rtol=0.0))[0]
    if bad.size > 0:
        if normalise:
            C = C / row_sums[:, None]
        else:
            offending = ", ".join(
                f"row {int(i)} sum={row_sums[i]:.6f}" for i in bad
            )
            raise ValueError(f"C is not row-stochastic: {offending}")

    return C.T @ a


def blocking_deviation(V: int, a: np.ndarray, C: np.ndarray,
                       demands: np.ndarray) -> dict:
    """Blocking deviation Delta_B_k = B_k(a_hat, V) - B_k(a, V).

    Returns a dict with keys a_true, a_hat, B_true, B_distorted, delta_B.
    """
    a_hat = bridge_equation(C, a)

    _, B_true = kaufman_roberts(V, a, demands)
    _, B_dist = kaufman_roberts(V, a_hat, demands)

    delta_B = B_dist - B_true

    return {
        'a_true': a,
        'a_hat': a_hat,
        'B_true': B_true,
        'B_distorted': B_dist,
        'delta_B': delta_B,
    }


def capacity_overhead(a_hat: np.ndarray, demands: np.ndarray,
                      B_target: float, V_start: int,
                      V_max: int = V_SEARCH_MAX) -> int:
    """Smallest capacity V' with B_k(a_hat, V') <= B_target for every class.

    Binary search over V, valid because B_k is non-increasing in V at fixed loads. Relative overhead is Delta_V/V = (V' - V_start) / V_start.

    Returns V_prime, the minimum capacity in AUs meeting the target for every class. The search runs over V_start..V_max.
    """
    _, blocking = kaufman_roberts(V_start, a_hat, demands)
    if np.all(blocking <= B_target):
        return V_start

    _, blocking = kaufman_roberts(V_max, a_hat, demands)
    if not np.all(blocking <= B_target):
        raise ValueError(f"Could not find V' <= {V_max} meeting B_target={B_target}")

    lo, hi = V_start, V_max
    while lo < hi:
        mid = (lo + hi) // 2
        _, blocking = kaufman_roberts(mid, a_hat, demands)
        if np.all(blocking <= B_target):
            hi = mid
        else:
            lo = mid + 1

    return lo


def sensitivity_analysis(V: int, a: np.ndarray, C: np.ndarray,
                         demands: np.ndarray,
                         delta: float = 1e-6) -> np.ndarray:
    """Constrained sensitivity of blocking probability to confusion-matrix entries.

    Row-stochasticity ties C_ii to C_ij, so the diagonal partial is subtracted (thesis Ch.3 Eq. 3.17):
        dB_k/dC_ij |constr = (dB_k/da_hat_j - dB_k/da_hat_i) * a_i
    dB_k/da_hat_j comes from central finite differences at relative step delta.

    Returns the sensitivity tensor S of shape (K, K, K).
    """
    K = len(a)
    a_hat = bridge_equation(C, a)

    dB_da = _blocking_gradient(V, a_hat, demands, delta)

    S = np.zeros((K, K, K))
    for k in range(K):
        for i in range(K):
            for j in range(K):
                S[k, i, j] = (dB_da[k, j] - dB_da[k, i]) * a[i]

    return S


def sensitivity_analysis_projected(V: int, a: np.ndarray, C: np.ndarray,
                                   demands: np.ndarray,
                                   delta: float = 1e-6) -> dict:
    """Projected-gradient sensitivity of blocking probability to CM entries.

    Row-stochasticity is honoured by centring each gradient row, the Riemannian gradient on the product of simplices. Since a_hat_l = sum_m C_ml a_m, the partial dB_k/dC_ij equals a_i * dB_k/da_hat_j, and centring gives

        g_proj[k, i, j] = a_i * [dB_k/da_hat_j - (1/K) sum_l dB_k/da_hat_l]

    System blocking follows the load-weighted K-R convention B_sys = (sum_k a_k t_k B_k) / (sum_k a_k t_k). dB_k/da_hat_j comes from central finite differences at relative step delta.

    Returns:
        Dict with S_proj, the row-centred gradient tensor of shape (K, K, K); S_sys_proj, its load-weighted (K, K) contraction; the scalars max_row_l2, mean_row_l2 and frobenius taken over the rows of S_sys_proj; and dB_da of shape (K, K), the partials dB_k/da_hat_j.
    """
    K = len(a)
    a_hat = bridge_equation(C, a)

    dB_da = _blocking_gradient(V, a_hat, demands, delta)

    S_proj = np.zeros((K, K, K))
    for k in range(K):
        for i in range(K):
            row_grad = a[i] * dB_da[k, :]
            S_proj[k, i, :] = row_grad - row_grad.mean()

    total_load = float(np.sum(a * demands))
    weights = (a * demands) / total_load
    S_sys_proj = np.einsum('k,kij->ij', weights, S_proj)

    row_norms = np.linalg.norm(S_sys_proj, axis=1)
    max_row_l2 = float(row_norms.max())
    mean_row_l2 = float(row_norms.mean())
    frobenius = float(np.linalg.norm(S_sys_proj))

    return {
        'S_proj': S_proj,
        'S_sys_proj': S_sys_proj,
        'max_row_l2': max_row_l2,
        'mean_row_l2': mean_row_l2,
        'frobenius': frobenius,
        'dB_da': dB_da,
    }


def perturbation_variance(a: np.ndarray, C_var: np.ndarray) -> np.ndarray:
    """Variance of the distorted load, Var(a_hat_k) = sum_i a_i^2 * Var(C_ik).

    Assumes estimation errors are independent across rows of C. C_var[i, k] is Var(C_ik). Returns the variance of each distorted load component.
    """
    return (a ** 2) @ C_var


def minimum_recall_search(a: np.ndarray, demands: np.ndarray,
                          V: int, B_target: float,
                          epsilon: float = 0.05,
                          r_steps: int = R_STEPS_SYSTEM) -> float:
    """Minimum per-class recall r* keeping the capacity overhead at or below epsilon.

    Uniform spillover: C_kk = r and C_kj = (1 - r) / (K - 1) for j != k. The sweep runs from r_max down to r_min and returns the last r before Delta_V/V exceeds epsilon, where epsilon is the maximum relative capacity overhead. Returns r_star, the minimum recall satisfying Delta_V/V <= epsilon.
    """
    K = len(a)
    r_min, r_max = 0.5, 1.0

    step = (r_max - r_min) / (r_steps - 1) if r_steps > 1 else 0.0
    for r in np.linspace(r_max, r_min, r_steps):
        C = uniform_spillover_cm(K, r)

        a_hat = bridge_equation(C, a)

        try:
            V_prime = capacity_overhead(a_hat, demands, B_target, V)
        except ValueError:
            continue

        relative_overhead = (V_prime - V) / V
        if relative_overhead > epsilon:
            r_prev = r + step
            return min(r_prev, r_max)

    return r_min


def kaufman_roberts_bpp(V: int, loads: np.ndarray, demands: np.ndarray,
                        traffic_types: list[str],
                        source_counts: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Binomial-Poisson-Pascal extension of the Kaufman-Roberts recursion.

    Recursion: n * q(n) = sum_k sigma_k(n - t_k) * t_k * q(n - t_k), with the state-dependent load sigma_k(n) = a_k for 'poisson' (infinite sources), a_k * (N_k - y_k(n)) / N_k for 'binomial' (Engset, finite sources) and a_k * (N_k + y_k(n)) / N_k for 'pascal' (peaked). y_k(n) uses the proportional approximation, first order for the binomial and Pascal cases (Stasiak et al. 2011, Sections 5.3 to 5.5).

    traffic_types is 'poisson', 'binomial' or 'pascal' per class; the source counts N_k are required unless every class is poisson.

    Returns (P, blocking): the normalised state distribution over 0..V and the per-class B_k.
    """
    K = len(loads)
    assert len(demands) == K
    assert len(traffic_types) == K
    assert V > 0
    assert np.all(demands >= 1), "All demands must be >= 1"

    if any(t != 'poisson' for t in traffic_types) and source_counts is None:
        raise ValueError(
            "source_counts is required when traffic_types contains "
            "binomial or pascal classes"
        )

    # proportional approximation in AU units, thesis Eq. (3.5)
    total_load_demand = np.sum(loads * demands)
    proportions = loads / total_load_demand

    q = np.zeros(V + 1)
    q[0] = 1.0

    for n in range(1, V + 1):
        s = 0.0
        for k in range(K):
            t_k = int(demands[k])
            if n - t_k < 0:
                continue

            y_k = proportions[k] * (n - t_k)

            if traffic_types[k] == 'binomial':
                N_k = source_counts[k]
                sigma = max(loads[k] * (N_k - y_k) / N_k, 0.0)
            elif traffic_types[k] == 'pascal':
                N_k = source_counts[k]
                sigma = loads[k] * (N_k + y_k) / N_k
            elif traffic_types[k] == 'poisson':
                sigma = loads[k]
            else:
                raise ValueError(
                    f"unknown traffic type {traffic_types[k]!r} for class {k}; "
                    "expected 'poisson', 'binomial', or 'pascal'")

            s += sigma * t_k * q[n - t_k]
        q[n] = s / n

    total = np.sum(q)
    P = q / total

    return P, _tail_blocking(P, V, demands)
