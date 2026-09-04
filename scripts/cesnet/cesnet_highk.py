"""Finding F1 anchor on the native 23 CESNET service categories.

Runs the isolated-class-k r*_k protocol of src/analytical/recall_thresholds.py (rstar_per_class, predictors, perm_spearman_p) on the real K=23 scenario with measured category supports, and tests by permutation the rank correlation of r*_k against the upward bandwidth gap and against the load-demand product a_k t_k. Both offered-load priors are computed, flow-count and Erlang-corrected, because they disagree on this scenario.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.analytical.kaufman_roberts import capacity_overhead, population_covariance
from src.cesnet.tiers import A_TOTAL, B_TARGET, GROUP_REC, TIER_AU
from src.analytical.recall_thresholds import perm_spearman_p, predictors, rstar_per_class

EPS = 0.01

# native 23-category AU demands: retained categories at their tier AU, the eight control and telemetry categories at the 1 AU floor of their measured rates
AU_23 = {name: (TIER_AU[tier] if tier >= 0 else 1) for name, tier in GROUP_REC.items()}


def anchor(a, t):
    """r*, predictors and both rank tests at one offered-load prior."""
    V = capacity_overhead(a, t, B_TARGET, V_start=1, V_max=8000)
    rstar = rstar_per_class(a, t, V, EPS)
    at, gap = predictors(a, t)
    rho_h3, p_h3 = perm_spearman_p(at, rstar)
    rho_f1, p_f1 = perm_spearman_p(gap, rstar)
    return {"V": V, "rstar": rstar, "at": at, "gap": gap,
            "rho_h3": rho_h3, "p_h3": p_h3, "rho_f1": rho_f1, "p_f1": p_f1,
            "cov": population_covariance(a, t)}


def main():
    z = np.load(ROOT / "data" / "processed" / "cesnet_definitive.npz", allow_pickle=True)
    names = [str(x) for x in z["category_names"]]
    sup = z["train_support"].astype(float)
    t = np.array([AU_23[n] for n in names], dtype=float)
    a = sup / sup.sum() * A_TOTAL
    a_erl = sup * z["hold_mean"]
    a_erl = a_erl / a_erl.sum() * A_TOTAL

    A = anchor(a, t)
    E = anchor(a_erl, t)

    print(f"=== Real CESNET K={len(a)} high-K anchor (eps={EPS*100:.0f}%, A_total={A_TOTAL}) ===")
    for tag, R in (("flow-count", A), ("Erlang-corrected", E)):
        print(f"[{tag:17s}] V={R['V']} cov(a,t)={R['cov']:+.4f}  "
              f"H3 rho={R['rho_h3']:+.4f} p={R['p_h3']:.4f}  "
              f"F1 rho={R['rho_f1']:+.4f} p={R['p_f1']:.4f}")
    print(f"non-degenerate r*: min {A['rstar'].min():.3f} max {A['rstar'].max():.3f}")
    order = np.argsort(t)
    print("\n per-category (sorted by AU):")
    print(f"  {'category':24s} {'AU':>3s} {'a':>6s} {'a*t':>7s} {'gap':>4s} {'r*':>6s}")
    for i in order:
        print(f"  {names[i]:24s} {int(t[i]):3d} {a[i]:6.2f} {A['at'][i]:7.1f} "
              f"{int(A['gap'][i]):4d} {A['rstar'][i]:6.3f}")
    np.savez(ROOT / "data" / "processed" / "cesnet_highk_real.npz",
             category_names=np.array(names), a=a, t=t,
             rstar=A["rstar"], at=A["at"], gap=A["gap"],
             V_nominal=np.int64(A["V"]), rho_h3=np.float64(A["rho_h3"]),
             p_h3=np.float64(A["p_h3"]), rho_f1=np.float64(A["rho_f1"]),
             p_f1=np.float64(A["p_f1"]), eps=np.float64(EPS),
             cov_at=np.float64(A["cov"]),
             a_erl=a_erl, rstar_erl=E["rstar"], at_erl=E["at"], gap_erl=E["gap"],
             V_nominal_erl=np.int64(E["V"]), cov_at_erl=np.float64(E["cov"]),
             rho_h3_erl=np.float64(E["rho_h3"]), p_h3_erl=np.float64(E["p_h3"]),
             rho_f1_erl=np.float64(E["rho_f1"]), p_f1_erl=np.float64(E["p_f1"]))
    print("\nsaved data/processed/cesnet_highk_real.npz")


if __name__ == "__main__":
    main()
