"""Common-shock correlated-error Monte Carlo for the Assumption A3 sweep of thesis Ch.6 Sec. 6.3.

Per-flow error indicator for true class i, with p_i = 1 - C_ii:

    X^(l)_i = Z_i * I^(l) + (1 - I^(l)) * U^(l)_i
    Z_i     ~ Bernoulli(p_i), class-conditional shock, one per (class, rep)
    U^(l)_i ~ Bernoulli(p_i), flow-specific draw, one per arrival
    I^(l)   ~ Bernoulli(w),   shock activator, one per arrival

For two flows of the same class the only shared randomness is Z_i with both activators firing, so Cov(X^(l), X^(l')) = w^2 * p_i(1 - p_i). Each indicator has variance p_i(1 - p_i), hence Corr = w^2 and w = sqrt(rho).

Only the binary error bit is correlated. Conditional on X = 1 the wrong class is drawn i.i.d. from the row off-diagonal, P(j | i, error) = C_ij / (1 - C_ii). Drawing Z_i once per replication is what produces the across-replication variance pattern 1 + (M-1)*rho_eff.

Per-class blocking is reported by apparent (predicted) class, so it compares directly against the bridge-equation prediction B_k(a_hat, V) with a_hat = C^T a.
"""

import multiprocessing as mp

import numpy as np
from numba import njit

from src.monte_carlo import WORKER_CAP


@njit(cache=True)
def _classify_arrival(
    true_class: int,
    Z_class: int,
    p_err: float,
    w: float,
    cum_offdiag: np.ndarray,
    rand_I: float,
    rand_U: float,
    rand_J: float,
    K: int,
) -> int:
    """Common-shock classifier for one arrival; returns the apparent class index.

    Args:
        true_class: index of the true class i
        Z_class: common-shock draw Z_i in {0, 1}, pre-drawn per class and replication
        p_err: marginal error probability p_i = 1 - C_ii
        w: shock-activator probability, sqrt(rho)
        cum_offdiag: row-i cumulative off-diagonal distribution, cum_offdiag[i, j] = sum_{k != i, k <= j} C_ik / (1 - C_ii)
        rand_I, rand_U, rand_J: three independent U[0,1) draws for the flow
    """
    I_act = 1 if rand_I < w else 0
    U_flow = 1 if rand_U < p_err else 0
    if I_act == 1:
        X = Z_class
    else:
        X = U_flow

    if X == 0:
        return true_class

    for j in range(K):
        if j == true_class:
            continue
        if rand_J <= cum_offdiag[true_class, j]:
            return j
    # the cumulative row can fall short of 1.0 by a few ulp
    return (true_class + 1) % K if K > 1 else true_class


@njit(cache=True)
def _simulate_fag_correlated_numba(
    V: int,
    loads_true: np.ndarray,
    demands: np.ndarray,
    p_err: np.ndarray,
    cum_offdiag: np.ndarray,
    w: float,
    Z_draw: np.ndarray,
    n_arrivals: int,
    seed: int,
):
    """Discrete-event FAG simulator with common-shock per-flow classification.

    Arrivals come from the true class distribution at rates loads_true, are then classified into an apparent class, and are admitted against that apparent class's demand t_j. Per-class counters are indexed by apparent class.
    """
    K = len(loads_true)
    np.random.seed(seed)

    arrivals_count = np.zeros(K, dtype=np.int64)
    blocked_count = np.zeros(K, dtype=np.int64)
    occupancy = np.int64(0)

    max_active = V + 100
    dep_times = np.full(max_active, 1e30)
    dep_bw = np.zeros(max_active, dtype=np.int64)
    n_active = np.int64(0)

    total_rate = np.float64(0.0)
    for k in range(K):
        total_rate += loads_true[k]

    # cumulative class-selection probabilities over true classes
    cum_true = np.zeros(K)
    cum_true[0] = loads_true[0] / total_rate
    for k in range(1, K):
        cum_true[k] = cum_true[k - 1] + loads_true[k] / total_rate

    current_time = np.float64(0.0)

    for _ in range(n_arrivals):
        u1 = np.random.random()
        inter_arrival = -np.log(u1) / total_rate
        next_arrival_time = current_time + inter_arrival

        # process departures before this arrival
        changed = True
        while changed and n_active > 0:
            changed = False
            min_idx = np.int64(0)
            min_time = dep_times[0]
            for idx in range(1, n_active):
                if dep_times[idx] < min_time:
                    min_time = dep_times[idx]
                    min_idx = idx
            if min_time <= next_arrival_time:
                occupancy -= dep_bw[min_idx]
                n_active -= 1
                dep_times[min_idx] = dep_times[n_active]
                dep_bw[min_idx] = dep_bw[n_active]
                dep_times[n_active] = 1e30
                changed = True

        current_time = next_arrival_time

        u2 = np.random.random()
        i_true = np.int64(0)
        for c in range(K):
            if u2 <= cum_true[c]:
                i_true = np.int64(c)
                break

        rand_I = np.random.random()
        rand_U = np.random.random()
        rand_J = np.random.random()
        j_app = _classify_arrival(
            i_true,
            int(Z_draw[i_true]),
            p_err[i_true],
            w,
            cum_offdiag,
            rand_I,
            rand_U,
            rand_J,
            K,
        )

        t_j = demands[j_app]
        arrivals_count[j_app] += 1

        if occupancy + t_j <= V:
            u3 = np.random.random()
            holding = -np.log(u3)
            dep_times[n_active] = current_time + holding
            dep_bw[n_active] = t_j
            n_active += 1
            occupancy += t_j
        else:
            blocked_count[j_app] += 1

    blocking = np.zeros(K)
    for k in range(K):
        if arrivals_count[k] > 0:
            blocking[k] = float(blocked_count[k]) / float(arrivals_count[k])

    return blocking, arrivals_count, blocked_count


def build_cum_offdiag(C: np.ndarray) -> np.ndarray:
    """Build the row-wise cumulative off-diagonal distribution for error draws.

    Returns a (K, K) array where cum_offdiag[i, j] is the probability that a class-i error lands at index <= j among the classes other than i. The diagonal entry contributes nothing and repeats cum_offdiag[i, j-1], which keeps indexing simple in the numba kernel.
    """
    K = C.shape[0]
    cum = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        denom = 1.0 - C[i, i]
        if denom <= 0:
            # perfect classifier row
            cum[i, :] = 0.0
            continue
        s = 0.0
        for j in range(K):
            if j == i:
                # carry running sum forward without contribution
                cum[i, j] = s
            else:
                s += C[i, j] / denom
                cum[i, j] = s
        # raise the final entry to at least 1.0 against floating-point shortfall
        cum[i, -1] = max(cum[i, -1], 1.0)
    return cum


def draw_common_shocks(
    p_err: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw Z_i ~ Bernoulli(p_err[i]) once per class (one replication)."""
    return (rng.random(p_err.shape[0]) < p_err).astype(np.int64)


def simulate_fag_correlated(
    V: int,
    a_true: np.ndarray,
    C: np.ndarray,
    rho: float,
    demands: np.ndarray,
    n_arrivals: int = 5_000_000,
    seed: int = 42,
):
    """Single-replication wrapper around the numba kernel.

    Returns (blocking_apparent, arrivals_apparent, blocked_apparent).
    """
    K = len(a_true)
    assert C.shape == (K, K), "C shape mismatch"
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho out of [0,1]: {rho}")

    p_err = 1.0 - np.diag(C).astype(np.float64)
    w = float(np.sqrt(rho))
    cum_offdiag = build_cum_offdiag(C)

    # separate RNG so the replication-level shock stays independent of the kernel's per-arrival stream
    rng_shock = np.random.default_rng(seed * 7919 + 31)
    Z = draw_common_shocks(p_err, rng_shock)

    blocking, arrivals, blocked = _simulate_fag_correlated_numba(
        np.int64(V),
        a_true.astype(np.float64),
        demands.astype(np.int64),
        p_err,
        cum_offdiag,
        np.float64(w),
        Z,
        np.int64(n_arrivals),
        np.int64(seed),
    )
    return blocking, arrivals, blocked


def _worker(args):
    V, a_true, C, rho, demands, n_arrivals, seed = args
    return simulate_fag_correlated(V, a_true, C, rho, demands, n_arrivals, seed)


def run_replications_with_rho(
    V: int,
    a_true: np.ndarray,
    C: np.ndarray,
    rho: float,
    demands: np.ndarray,
    M: int = 30,
    n_arrivals: int = 5_000_000,
    n_workers: int | None = None,
    base_seed: int = 1,
):
    """Run M independent replications at fixed (V, a_true, C, rho).

    Each replication takes a distinct base seed and draws its common shock Z_i from a separate RNG derived from that seed, so the shocks are independent across replications.

    Returns:
        all_B, all_arr, all_blk: apparent-class blocking, arrival counts and blocked counts, each of shape (M, K).
    """
    if n_workers is None:
        n_workers = min(WORKER_CAP, mp.cpu_count())
    args = [
        (int(V), a_true.copy(), C.copy(), float(rho), demands.copy(),
         int(n_arrivals), int(s))
        for s in range(base_seed, base_seed + M)
    ]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker, args)
    all_B = np.array([r[0] for r in results])
    all_arr = np.array([r[1] for r in results])
    all_blk = np.array([r[2] for r in results])
    return all_B, all_arr, all_blk
