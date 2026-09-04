"""Erlang fixed-point approximation for multi-link multirate loss networks.

Extends the single-FAG Kaufman-Roberts solver to a network where each stream traverses a route, takes the same number of allocation units on every link of that route, and is lost if any of those links is full at arrival. Link blocking is taken as independent, so a link sees the ingress load thinned by acceptance on the rest of the route, and the coupled values are solved by repeated substitution:

  rho_{s,l} = nu_s * prod_{m in r(s), m != l} (1 - B_{m,s})
  B_{l,.}   = KaufmanRoberts(V_l, rho_{.,l}, t)
  B^e2e_s   = 1 - prod_{l in r(s)} (1 - B_{l,s})

Kelly (1986) reduced-load fixed point; Stasiak et al. (2011) EFPA.
"""

from collections.abc import Sequence

import numpy as np

from .kaufman_roberts import kaufman_roberts

_MAX_ITER = 2000


def route_incidence(routes: Sequence[Sequence[int]] | np.ndarray, n_links: int,
                    n_streams: int, dtype=bool) -> np.ndarray:
    """Return an (S, L) route-incidence matrix in the requested dtype."""
    if isinstance(routes, np.ndarray) and routes.ndim == 2:
        return routes.astype(dtype)
    R = np.zeros((n_streams, n_links), dtype=dtype)
    for s, links in enumerate(routes):
        for l in links:
            R[s, int(l)] = 1
    return R


def _prepare(V_links: np.ndarray, offered: np.ndarray, demands: np.ndarray,
             routes: Sequence[Sequence[int]] | np.ndarray) -> tuple:
    """Cast the inputs and build the per-link and per-stream index lists."""
    V_links = np.asarray(V_links)
    offered = np.asarray(offered, dtype=float)
    demands = np.asarray(demands)
    L = len(V_links)
    S = len(offered)
    R = route_incidence(routes, L, S)
    streams_on = [np.where(R[:, l])[0] for l in range(L)]   # stream indices per link
    links_of = [np.where(R[s, :])[0] for s in range(S)]     # link indices per stream
    return V_links, offered, demands, streams_on, links_of


def _link_blocking(V_links: np.ndarray, streams_on: list, demands: np.ndarray,
                   loads, B: np.ndarray) -> np.ndarray:
    """Per-link Kaufman-Roberts at ``loads(l, s_idx)``, written into B in place."""
    for l in range(len(V_links)):
        s_idx = streams_on[l]
        if s_idx.size == 0:
            continue
        _, B[l, s_idx] = kaufman_roberts(int(V_links[l]), loads(l, s_idx),
                                         demands[s_idx])
    return B


def _end_to_end(B: np.ndarray, links_of: list) -> np.ndarray:
    """End-to-end blocking per stream, one minus the product of route acceptances."""
    B_e2e = np.zeros(len(links_of))
    for s, r in enumerate(links_of):
        B_e2e[s] = 1.0 - np.prod(1.0 - B[r, s]) if r.size else 0.0
    return B_e2e


def efpa_fixed_point(V_links: np.ndarray, offered: np.ndarray,
                     demands: np.ndarray,
                     routes: Sequence[Sequence[int]] | np.ndarray,
                     tol: float = 1e-12) -> dict:
    """Solve the reduced-load (Erlang fixed-point) approximation.

    Args:
        V_links: per-link capacities in AUs, shape (L,).
        offered: per-stream ingress offered loads in Erlang, shape (S,). Pass the distorted loads (a_hat) for the classifier case.
        demands: per-stream AU demand t_s, shape (S,); same on every link of the stream's route. Integer-valued, >= 1.
        routes: (S, L) incidence or a list of link-index lists per stream.
        tol: convergence tolerance on the max change in reduced loads.

    Returns:
        dict with keys:
          B_link       (L, S) per-link blocking; np.nan where stream not on link
          B_e2e        (S,)   end-to-end blocking per stream
          reduced_loads(L, S) converged reduced loads; np.nan off-route
          iterations   int
          converged    bool
          residual     float  final max change in reduced loads
    """
    V_links, offered, demands, streams_on, links_of = _prepare(
        V_links, offered, demands, routes)
    L = len(V_links)
    S = len(offered)
    assert demands.shape == (S,), "demands must have one entry per stream"
    assert np.all(demands >= 1), "all demands must be >= 1"

    def _solve(damp):
        rho = np.full((L, S), np.nan)
        for s in range(S):
            for l in links_of[s]:
                rho[l, s] = offered[s]
        B = np.full((L, S), np.nan)
        residual = np.inf
        it = 0
        for it in range(1, _MAX_ITER + 1):
            _link_blocking(V_links, streams_on, demands,
                           lambda l, s_idx: rho[l, s_idx], B)
            rho_new = np.full((L, S), np.nan)
            for s in range(S):
                r = links_of[s]
                for l in r:
                    others = r[r != l]
                    accept = np.prod(1.0 - B[others, s]) if others.size else 1.0
                    rho_new[l, s] = offered[s] * accept
            diff = rho_new[~np.isnan(rho_new)] - rho[~np.isnan(rho)]
            residual = float(np.max(np.abs(diff))) if diff.size else 0.0
            rho = np.where(np.isnan(rho_new), rho,
                           (1.0 - damp) * rho + damp * rho_new)
            if residual < tol:
                return rho, B, it, True, residual
        return rho, B, it, False, residual

    # repeated substitution can oscillate as utilisation rises; under-relax until it converges, which reaches the same fixed point more slowly
    damp = 1.0
    rho, B, it, converged, residual = _solve(damp)
    while not converged and damp > 0.05:
        damp *= 0.5
        rho, B, it, converged, residual = _solve(damp)

    _link_blocking(V_links, streams_on, demands,
                   lambda l, s_idx: rho[l, s_idx], B)

    return {
        'B_link': B,
        'B_e2e': _end_to_end(B, links_of),
        'reduced_loads': rho,
        'iterations': it,
        'converged': converged,
        'residual': residual,
    }


def efpa_independent(V_links: np.ndarray, offered: np.ndarray,
                     demands: np.ndarray,
                     routes: Sequence[Sequence[int]] | np.ndarray) -> dict:
    """Independent-links baseline: every link sees the full ingress load.

    No reduced-load thinning, but end-to-end blocking still combines the per-link values multiplicatively, so the gap against efpa_fixed_point isolates the reduced-load coupling.
    """
    V_links, offered, demands, streams_on, links_of = _prepare(
        V_links, offered, demands, routes)

    B = np.full((len(V_links), len(offered)), np.nan)
    _link_blocking(V_links, streams_on, demands,
                   lambda l, s_idx: offered[s_idx], B)

    return {'B_link': B, 'B_e2e': _end_to_end(B, links_of)}
