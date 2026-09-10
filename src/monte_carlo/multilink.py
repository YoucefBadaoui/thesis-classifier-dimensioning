"""Path-based event-driven Monte Carlo for a multi-link loss network.

Measures the accuracy of the Erlang fixed-point approximation in ``analytical.efpa``, which assumes link-blocking independence. A call of stream s arrives Poisson at unit mean holding time, so its arrival rate equals its offered load; it holds ``demands[s]`` AUs on every link of its route at once and is lost if any of those links is full. Per-stream end-to-end blocking is the lost fraction. Occupancy is a per-link vector and admission a conjunction over the route, unlike the single-FAG simulator in ``rho_sweep``. M independent replications run on the fixed seed sequence, with Student-t 95 percent intervals across replications.
"""

import multiprocessing as mp
from collections.abc import Sequence

import numpy as np
from numba import njit

from src.analytical.efpa import route_incidence
from src.monte_carlo import WORKER_CAP


@njit(cache=True)
def _simulate_multilink_numba(V_links: np.ndarray, offered: np.ndarray,
                              demands: np.ndarray, route: np.ndarray,
                              n_arrivals: int, seed: int):
    """Event-driven path-based simulation. Returns (blocking, arrivals, blocked)."""
    np.random.seed(seed)
    L = V_links.shape[0]
    S = offered.shape[0]

    total_rate = 0.0
    for s in range(S):
        total_rate += offered[s]

    cum = np.zeros(S)
    acc = 0.0
    for s in range(S):
        acc += offered[s] / total_rate
        cum[s] = acc

    occupancy = np.zeros(L, dtype=np.int64)
    max_active = 0
    for l in range(L):
        max_active += int(V_links[l])
    max_active += 100
    dep_time = np.full(max_active, 1e30)
    dep_stream = np.zeros(max_active, dtype=np.int64)
    n_active = 0

    arrivals = np.zeros(S, dtype=np.int64)
    blocked = np.zeros(S, dtype=np.int64)
    current_time = 0.0

    for _ in range(n_arrivals):
        u1 = 1.0 - np.random.random()   # in (0, 1], so -log is finite
        current_time += -np.log(u1) / total_rate

        # Process every departure that occurs at or before the next arrival.
        while True:
            min_idx = -1
            min_t = current_time
            for i in range(n_active):
                if dep_time[i] <= min_t:
                    min_t = dep_time[i]
                    min_idx = i
            if min_idx == -1:
                break
            s_dep = dep_stream[min_idx]
            t_dep = demands[s_dep]
            for l in range(L):
                if route[s_dep, l]:
                    occupancy[l] -= t_dep
            n_active -= 1
            dep_time[min_idx] = dep_time[n_active]
            dep_stream[min_idx] = dep_stream[n_active]
            dep_time[n_active] = 1e30

        u2 = np.random.random()
        s = S - 1
        for c in range(S):
            if u2 <= cum[c]:
                s = c
                break
        arrivals[s] += 1
        t_s = demands[s]

        # Admit only if every link on the route has room.
        ok = True
        for l in range(L):
            if route[s, l] and occupancy[l] + t_s > V_links[l]:
                ok = False
                break

        if ok:
            u3 = 1.0 - np.random.random()   # in (0, 1], so -log is finite
            holding = -np.log(u3)
            dep_time[n_active] = current_time + holding
            dep_stream[n_active] = s
            n_active += 1
            for l in range(L):
                if route[s, l]:
                    occupancy[l] += t_s
        else:
            blocked[s] += 1

    blocking = np.zeros(S)
    for s in range(S):
        if arrivals[s] > 0:
            blocking[s] = blocked[s] / arrivals[s]
    return blocking, arrivals, blocked


def _validate(demands: np.ndarray, V_links: np.ndarray, R: np.ndarray) -> None:
    """Guard the Numba kernel, which does not bounds-check (a zero demand or an empty route would corrupt memory rather than raise)."""
    assert np.all(np.asarray(demands) >= 1), "all demands must be >= 1"
    assert np.all(np.asarray(V_links) >= 1), "all link capacities must be >= 1"
    assert R.sum(axis=1).min() >= 1, "every stream must traverse at least one link"


def _worker(args):
    V_links, offered, demands, R, n_arrivals, seed = args
    return _simulate_multilink_numba(V_links, offered, demands, R,
                                     int(n_arrivals), int(seed))


def run_replications_multilink(V_links: np.ndarray, offered: np.ndarray,
                               demands: np.ndarray,
                               routes: Sequence[Sequence[int]] | np.ndarray,
                               M: int = 30, n_arrivals: int = 5_000_000,
                               base_seed: int = 1, n_workers: int | None = None
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M independent replications on seeds base_seed..base_seed+M-1.

    Returns all_B, all_arr, all_blk, each shape (M, S).
    """
    V_links = np.asarray(V_links, dtype=np.int64)
    offered = np.asarray(offered, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.int64)
    S = len(offered)
    R = route_incidence(routes, len(V_links), S, dtype=np.int64)
    _validate(demands, V_links, R)

    tasks = [(V_links, offered, demands, R, n_arrivals, base_seed + m)
             for m in range(M)]
    if n_workers is None:
        n_workers = min(WORKER_CAP, mp.cpu_count())

    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker, tasks)

    all_B = np.array([r[0] for r in results])
    all_arr = np.array([r[1] for r in results])
    all_blk = np.array([r[2] for r in results])
    return all_B, all_arr, all_blk
