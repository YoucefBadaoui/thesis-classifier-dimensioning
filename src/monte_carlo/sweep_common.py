"""Shared pieces of the two correlation-sweep drivers under scripts/experiments/.

Both sweeps run the same two scenarios over the same rho grid with the same replication budget and write the same base archive keys; only the error process and its extra diagnostics differ.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from src.analytical.constants import (
    A_5G,
    A_OTT,
    CLASS_ORDER_5G,
    CLASS_ORDER_OTT,
    T_5G,
    T_OTT,
    V_NOMINAL_5G,
    V_NOMINAL_OTT,
)
from src.analytical.kaufman_roberts import bridge_equation, kaufman_roberts
from src.analytical.published_cms import MALKOC_CM_BNB
from src.monte_carlo import WORKER_CAP

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
RHOS = (0.0, 0.3, 0.6)

SCENARIOS = {
    "ott": dict(V=V_NOMINAL_OTT, a=A_OTT, t=T_OTT, cm_key="xgb_clean",
                order=list(CLASS_ORDER_OTT)),
    "5g": dict(V=V_NOMINAL_5G, a=A_5G, t=T_5G, cm_key="malkoc_bnb",
               order=list(CLASS_ORDER_5G)),
}


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--M", type=int, default=300, help="replications per rho")
    ap.add_argument("--arrivals", type=int, default=5_000_000, help="arrivals per replication")
    ap.add_argument("--scenarios", default="ott,5g",
                    help="comma-separated scenario names")
    ap.add_argument("--workers", type=int, default=WORKER_CAP,
                    help="worker processes in the replication pool")
    ap.add_argument("--out", type=Path, default=PROCESSED, help="archive directory")
    return ap.parse_args()


def load_scenario(name: str):
    """Scenario spec plus its true loads, demands, confusion matrix, analytical rho = 0 blocking and per-class error rates."""
    spec = SCENARIOS[name]
    a_true = np.asarray(spec["a"], float)
    t = np.asarray(spec["t"], dtype=np.int64)
    if name == "ott":
        z = np.load(PROCESSED / "confusion_matrices.npz", allow_pickle=True)
        C = np.asarray(z[spec["cm_key"]], float)
    else:
        C = np.asarray(MALKOC_CM_BNB, float)
    a_hat = bridge_equation(C, a_true)
    _, B_an = kaufman_roberts(spec["V"], a_hat, t)
    return spec, a_true, t, C, a_hat, B_an, 1.0 - np.diag(C)


def summarise(allB: np.ndarray):
    """Replication mean, sample deviation and Student-t 95 percent half-width over axis 1 of an (n_rho, M, K) array. Returns (M, mean, deviation, half-width)."""
    M = allB.shape[1]
    tcrit = float(stats.t.ppf(0.975, M - 1))
    B_mean = allB.mean(1)
    bp_std = allB.std(1, ddof=1)
    return M, B_mean, bp_std, tcrit * bp_std / np.sqrt(M)


def print_drift_table(order, B_mean: np.ndarray) -> None:
    base = B_mean[0]
    print("\n  drift versus rho=0 (percent):")
    for ci, cname in enumerate(order):
        d = [(B_mean[ri, ci] - base[ci]) / base[ci] * 100 for ri in range(len(RHOS))]
        print(f"    {cname:14s} " + "  ".join(
            f"rho={RHOS[ri]}: {d[ri]:+7.2f}%" for ri in range(len(RHOS))))


def base_archive(name, spec, args, allB, allA, allBl, B_mean, bp_std, ci_half,
                 a_true, t, C, a_hat, B_an) -> dict:
    """The archive keys both sweeps write."""
    return dict(
        rho_values=np.array(RHOS), all_blocking=allB, all_arrivals=allA,
        all_blocked=allBl, B_mean=B_mean, ci_half=ci_half, bp_std=bp_std,
        class_order=np.array(spec["order"]), M=np.int64(allB.shape[1]),
        n_arrivals_per_rep=np.int64(args.arrivals), scenario=np.array(name),
        cm_name=np.array(spec["cm_key"]), V=np.int64(spec["V"]), a_true=a_true,
        t_k=t, C=C, a_hat=a_hat, B_analytical_rho0=B_an,
    )
