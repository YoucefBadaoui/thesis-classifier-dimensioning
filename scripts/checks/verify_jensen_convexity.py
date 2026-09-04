"""Exact check of the Jensen mechanism behind the ch6 rho-sweep drift.

For the OTT/IPTV scenario under the XGB-clean matrix at V = 499, the correlated error model draws per replication a regime vector Z in {0,1}^K with Z_i ~ Bernoulli(p_i) for class error rate p_i, so the per-replication error probability is q_i = w Z_i + (1 - w) p_i with w = sqrt(rho). Exit code 1 if convexity fails or, on the binding classes, the comparison against monte_carlo_rho_sweep_M300_ott.npz disagrees in sign or grossly in size.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.kaufman_roberts import kaufman_roberts
from src.analytical.constants import A_OTT, T_OTT, CLASS_ORDER_OTT

PROCESSED = ROOT / "data" / "processed"
V_OTT = 499
# the two classes that bind the OTT dimensioning; only these gate the exit code
BINDING = ("Streaming", "FileTransfer")
# drift agreement band: 10 percentage points absolute, or 20 percent relative, whichever is looser; M = 300 replications leave a few percent of noise
DRIFT_ABS_PP, DRIFT_REL = 10.0, 0.2


def main() -> int:
    z = np.load(PROCESSED / "confusion_matrices.npz", allow_pickle=True)
    C = np.asarray(z["xgb_clean"], float)
    a_true = np.asarray(A_OTT, float)
    t = np.asarray(T_OTT)
    K = len(a_true)

    p_vec = 1.0 - np.diag(C)
    pos = p_vec[:, None] > 0
    offrow = np.where(pos, C / np.where(pos, p_vec[:, None], 1.0), 0.0)
    np.fill_diagonal(offrow, 0.0)

    def a_hat_given(q: np.ndarray) -> np.ndarray:
        return a_true * (1.0 - q) + (a_true * q) @ offrow

    sweep = np.load(PROCESSED / "monte_carlo_rho_sweep_M300_ott.npz",
                    allow_pickle=True)
    B_mean = sweep["B_mean"]
    rho_values = list(sweep["rho_values"])

    failures = 0
    _, B_at_mean = kaufman_roberts(V_OTT, a_hat_given(p_vec), t)

    for rho in (0.3, 0.6):
        w = np.sqrt(rho)
        EB = np.zeros(K)
        for m in range(2 ** K):
            Z = np.array([(m >> i) & 1 for i in range(K)], dtype=float)
            prob = float(np.prod(np.where(Z == 1, p_vec, 1.0 - p_vec)))
            q = w * Z + (1.0 - w) * p_vec
            _, B = kaufman_roberts(V_OTT, a_hat_given(q), t)
            EB += prob * B
        i_rho = rho_values.index(rho)
        print(f"rho = {rho}:")
        for k, name in enumerate(CLASS_ORDER_OTT):
            pred = (EB[k] / B_at_mean[k] - 1.0) * 100.0
            sim = (B_mean[i_rho][k] / B_mean[0][k] - 1.0) * 100.0
            ok = (pred > 0 and sim > 0
                  and abs(pred - sim) < max(DRIFT_ABS_PP, DRIFT_REL * sim))
            if name in BINDING and not ok:
                failures += 1
            print(f"  {name:13s} predicted drift {pred:+7.1f}%  "
                  f"simulated (M=300) {sim:+7.1f}%")

    print("\nLocal convexity (second difference along single-class "
          "Z-directions at the operating point):")
    w = np.sqrt(0.3)
    for k, name in enumerate(CLASS_ORDER_OTT):
        qa = p_vec.copy()
        qb = p_vec.copy()
        qb[k] = w * 1.0 + (1.0 - w) * p_vec[k]
        Bs = []
        for lam in (0.0, 0.5, 1.0):
            q = qa + lam * (qb - qa)
            _, B = kaufman_roberts(V_OTT, a_hat_given(q), t)
            Bs.append(B[k])
        d2 = Bs[0] - 2.0 * Bs[1] + Bs[2]
        ok = d2 > 0
        failures += not ok
        print(f"  {name:13s} second difference {d2:+.3e} "
              f"({'convex' if ok else 'NOT convex'})")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} "
          f"({failures} failing checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
