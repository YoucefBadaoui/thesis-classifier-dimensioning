"""Isolated-class-k minimum recall r*_k and the rank tests applied to it.

Class k has recall r and spills its error mass uniformly over the other K - 1 classes while every other class is perfect; r*_k is the smallest recall that keeps the relative capacity overhead within epsilon. Two search forms exist: the 200-step grid behind the thesis tables (per_class_recall_search) and the bisection of the high-K power analysis (rstar_per_class). Both rank tests are two-sided permutation tests; perm_spearman_p is the loop form that produced the K = 23 CESNET archive and reproduces cesnet_highk_real.npz bit for bit.
"""

from itertools import permutations

import numpy as np
from scipy import stats

from .constants import B_TARGET_DEFAULT, R_STEPS_PER_CLASS
from .kaufman_roberts import bridge_equation, capacity_overhead, kaufman_roberts

SEED = 20260529            # default seed of the loop-form permutation test
R_FLOOR, R_CEIL = 0.50, 1.0
R_TOL = 0.0025             # bisection resolution, matches the 200-step grid over [0.5, 1.0]


def isolated_class_cm(K: int, k: int, r: float) -> np.ndarray:
    """Identity except row k: class k has recall r and spills (1 - r) uniformly over the other K - 1 classes; every other class is perfect."""
    C = np.eye(K)
    C[k, :] = (1.0 - r) / (K - 1)
    C[k, k] = r
    return C


def per_class_recall_search(
    a: np.ndarray, t: np.ndarray, V: int, B_target: float,
    epsilon: float, r_min: float = 0.50, r_max: float = 1.0,
    r_steps: int = R_STEPS_PER_CLASS,
) -> np.ndarray:
    """Per-class isolated-class-k minimum recall r*_k.

    Isolated-class-k spillover, see isolated_class_cm. Sweeps r downward and returns the smallest r keeping the capacity overhead at most epsilon.
    """
    K = len(a)
    r_stars = np.full(K, r_min)
    step = (r_max - r_min) / (r_steps - 1)   # spacing of the linspace grid below
    for k in range(K):
        found = r_max
        stopped = False
        for r in np.linspace(r_max, r_min, r_steps):
            a_hat = bridge_equation(isolated_class_cm(K, k, r), a)
            try:
                V_prime = capacity_overhead(a_hat, t, B_target,
                                            V_start=V, V_max=2 * V + 500)
            except ValueError:
                continue
            rel_overhead = (V_prime - V) / V
            if rel_overhead > epsilon:
                found = min(r + step, r_max)
                stopped = True
                break
        if not stopped:
            found = r_min
        r_stars[k] = found
    return r_stars


def _row_k_within(a, t, V_nom, epsilon, k, r):
    """True iff isolated-class-k spillover at recall r keeps (V' - V_nom)/V_nom <= epsilon.

    Equivalent to all per-class blocking <= B_target at V = floor(V_nom(1+eps)), so one Kaufman-Roberts evaluation answers it.
    """
    a_hat = isolated_class_cm(len(a), k, r).T @ a
    V_target = int(np.floor(V_nom * (1.0 + epsilon)))
    _, B = kaufman_roberts(V_target, a_hat, t)
    return bool(np.all(B <= B_TARGET_DEFAULT))


def rstar_per_class(a, t, V_nom, epsilon, tol=R_TOL):
    """Bisection form of per_class_recall_search: r*_k is the smallest recall keeping the relative capacity overhead within epsilon.

    Isolated-class-k spillover, see isolated_class_cm. Overhead is monotone decreasing in r, so bisection on [R_FLOOR, R_CEIL] finds it; r*_k floors at R_FLOOR when r = R_FLOOR already stays within eps.
    """
    K = len(a)
    rstar = np.full(K, R_FLOOR)
    for k in range(K):
        if _row_k_within(a, t, V_nom, epsilon, k, R_FLOOR):
            rstar[k] = R_FLOOR
            continue
        lo, hi = R_FLOOR, R_CEIL
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if _row_k_within(a, t, V_nom, epsilon, k, mid):
                hi = mid
            else:
                lo = mid
        rstar[k] = hi
    return rstar


def predictors(a, t):
    """Return the two r*_k predictors: H3 load-demand product a_k*t_k and F1 upward bandwidth gap max_j(t_j) - t_k."""
    at = a * t
    gap = t.max() - t
    return at, gap


def spearman_perm(x, y, rng, n_perm=4999):
    """Spearman rho and its two-sided permutation p-value.

    Average ranks, so ties at the r* floor are handled. Exact over all n! orderings for n <= 7, Monte Carlo with n_perm draws otherwise. NaN if x or y is constant.
    """
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    n = len(x)
    rho0 = float(np.mean(rx * ry))
    if n <= 7:
        P = np.array(list(permutations(range(n))))
        rhos = (rx[P] * ry).mean(axis=1)
        p = float(np.mean(np.abs(rhos) >= abs(rho0) - 1e-12))
    else:
        P = rng.permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)
        rhos = (rx[P] * ry).mean(axis=1)
        p = (int(np.sum(np.abs(rhos) >= abs(rho0) - 1e-12)) + 1) / (n_perm + 1)
    return rho0, p


def perm_spearman_p(x, y, n_perm=20000, rng=None):
    """Loop form of the permutation test, kept so cesnet_highk_real.npz reproduces bit for bit. Returns (rho, p)."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    rho0, _ = stats.spearmanr(x, y)
    if rng is None:
        rng = np.random.default_rng(SEED)
    count = 0
    yc = np.array(y)
    for _ in range(n_perm):
        rho_p, _ = stats.spearmanr(x, rng.permutation(yc))
        if abs(rho_p) >= abs(rho0) - 1e-12:
            count += 1
    return float(rho0), (count + 1) / (n_perm + 1)
