"""Two-state Markov-modulated classifier errors for the A3 correlation sweep.

Every true class ``i`` carries its own two-state chain, advanced on each arrival of that class. With marginal error probability ``p_i = 1 - C_ii`` and target lag-1 autocorrelation ``rho``:

    P(S=1 | S=1) = p_i + (1 - p_i) * rho
    P(S=1 | S=0) = p_i * (1 - rho)

The chain is stationary at ``P(S=1) = p_i`` for every ``rho``, so the marginal confusion matrix is preserved exactly, ``Corr(X_t, X_{t+k}) = rho^k``, and error runs are geometric with mean length ``1 / ((1 - p_i) * (1 - rho))``. At ``rho = 0`` it reduces to independent Bernoulli(p_i) draws.

Error destinations are drawn from the row off-diagonal, as in the common-shock kernel of :mod:`monte_carlo.rho_sweep`; the two differ only in the temporal law of the error indicator. That kernel draws one shock per class and replication, so its error process is exchangeable within a replication and has no finite correlation time.
"""

import multiprocessing as mp

import numpy as np
from numba import njit

from src.monte_carlo import WORKER_CAP
from src.monte_carlo.rho_sweep import build_cum_offdiag


def markov_transitions(p_err: np.ndarray, rho: float):
    """Return (p11, p01) for the stationary two-state chains, one per class."""
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho out of [0,1): {rho}")
    p = np.asarray(p_err, dtype=np.float64)
    p11 = p + (1.0 - p) * rho
    p01 = p * (1.0 - rho)
    return p11, p01


def mean_cluster_length(p_err: np.ndarray, rho: float) -> np.ndarray:
    """Analytical mean error-run length per class, in class arrivals."""
    p = np.asarray(p_err, dtype=np.float64)
    denom = (1.0 - p) * (1.0 - rho)
    out = np.full_like(denom, np.inf)
    np.divide(1.0, denom, out=out, where=denom > 0)
    return out


@njit(cache=True)
def _simulate_fag_markov_numba(
    V: int,
    loads_true: np.ndarray,
    demands: np.ndarray,
    p11: np.ndarray,
    p01: np.ndarray,
    cum_offdiag: np.ndarray,
    S0: np.ndarray,
    n_arrivals: int,
    seed: int,
):
    """Per-class counters are indexed by apparent (predicted) class, as in the common-shock kernel. The error-process diagnostics (arrival, error, 1->1 transition and run-start counts) are indexed by true class, where the chain lives."""
    K = len(loads_true)
    np.random.seed(seed)

    arrivals_count = np.zeros(K, dtype=np.int64)
    blocked_count = np.zeros(K, dtype=np.int64)
    true_count = np.zeros(K, dtype=np.int64)
    err_count = np.zeros(K, dtype=np.int64)
    pair11_count = np.zeros(K, dtype=np.int64)
    pair_count = np.zeros(K, dtype=np.int64)
    runstart_count = np.zeros(K, dtype=np.int64)

    S = np.zeros(K, dtype=np.int64)
    seen = np.zeros(K, dtype=np.int64)
    for k in range(K):
        S[k] = S0[k]

    occupancy = np.int64(0)
    max_active = V + 100
    dep_times = np.full(max_active, 1e30)
    dep_bw = np.zeros(max_active, dtype=np.int64)
    n_active = np.int64(0)

    total_rate = np.float64(0.0)
    for k in range(K):
        total_rate += loads_true[k]

    cum_true = np.zeros(K)
    cum_true[0] = loads_true[0] / total_rate
    for k in range(1, K):
        cum_true[k] = cum_true[k - 1] + loads_true[k] / total_rate

    current_time = np.float64(0.0)

    for _ in range(n_arrivals):
        u1 = np.random.random()
        inter_arrival = -np.log(u1) / total_rate
        next_arrival_time = current_time + inter_arrival

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

        # Advance the class-i chain one step, then read the error indicator.
        prev = S[i_true]
        r = np.random.random()
        if prev == 1:
            S[i_true] = 1 if r < p11[i_true] else 0
        else:
            S[i_true] = 1 if r < p01[i_true] else 0
        X = S[i_true]

        true_count[i_true] += 1
        if X == 1:
            err_count[i_true] += 1
        if seen[i_true] == 1:
            pair_count[i_true] += 1
            if prev == 1 and X == 1:
                pair11_count[i_true] += 1
        if X == 1 and prev == 0:
            runstart_count[i_true] += 1
        seen[i_true] = 1

        j_app = i_true
        if X == 1:
            rand_J = np.random.random()
            found = -1
            for j in range(K):
                if j == i_true:
                    continue
                if rand_J <= cum_offdiag[i_true, j]:
                    found = j
                    break
            if found >= 0:
                j_app = found
            elif K > 1:
                j_app = (i_true + 1) % K

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

    return (blocking, arrivals_count, blocked_count,
            true_count, err_count, pair11_count, pair_count, runstart_count)


def simulate_fag_markov(
    V: int,
    a_true: np.ndarray,
    C: np.ndarray,
    rho: float,
    demands: np.ndarray,
    n_arrivals: int = 5_000_000,
    seed: int = 42,
):
    """Single-replication wrapper. Returns (blocking, arrivals, blocked, diag).

    ``diag`` carries the measured error-process statistics per true class: realised marginal error rate, realised lag-1 autocorrelation, and mean error-cluster length, alongside their analytical targets.
    """
    K = len(a_true)
    assert C.shape == (K, K), "C shape mismatch"
    p_err = 1.0 - np.diag(C).astype(np.float64)
    p11, p01 = markov_transitions(p_err, rho)
    cum_offdiag = build_cum_offdiag(C)

    rng = np.random.default_rng(seed * 7919 + 31)
    S0 = (rng.random(K) < p_err).astype(np.int64)

    (blocking, arrivals, blocked, true_count, err_count,
     pair11, pairs, runstarts) = _simulate_fag_markov_numba(
        np.int64(V),
        a_true.astype(np.float64),
        demands.astype(np.int64),
        p11,
        p01,
        cum_offdiag,
        S0,
        np.int64(n_arrivals),
        np.int64(seed),
    )

    p_hat = np.where(true_count > 0, err_count / np.maximum(true_count, 1), np.nan)
    # Corr(X_t, X_{t+1}) = (P(1|1) - p) / (1 - p); estimate P(1|1) from the joint 1->1 rate divided by the marginal error rate.
    joint11 = np.where(pairs > 0, pair11 / np.maximum(pairs, 1), np.nan)
    rho_hat = np.where(
        (p_hat > 0) & (p_hat < 1),
        (joint11 - p_hat ** 2) / np.maximum(p_hat * (1.0 - p_hat), 1e-300),
        np.nan,
    )
    cluster_hat = np.where(runstarts > 0, err_count / np.maximum(runstarts, 1), np.nan)

    diag = {
        "p_target": p_err,
        "p_measured": p_hat,
        "rho_measured": rho_hat,
        "cluster_target": mean_cluster_length(p_err, rho),
        "cluster_measured": cluster_hat,
        "true_count": true_count,
        "err_count": err_count,
        "run_count": runstarts,
    }
    return blocking, arrivals, blocked, diag


def _worker(args):
    V, a_true, C, rho, demands, n_arrivals, seed = args
    b, a, bl, d = simulate_fag_markov(V, a_true, C, rho, demands, n_arrivals, seed)
    return b, a, bl, d


def run_replications_markov(
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
    """Run M independent replications of the Markov-modulated simulator.

    Replications differ only in their random stream, so the ensemble samples one ergodic error process rather than a mixture over shock regimes.
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
    diags = [r[3] for r in results]
    return all_B, all_arr, all_blk, diags
