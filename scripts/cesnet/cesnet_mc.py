"""Monte Carlo validation of the CESNET headline dimensioning scenario.

At rho = 0 an event-driven single-FAG simulator (src/monte_carlo/rho_sweep.py) draws true-class arrivals, misclassifies them through the confusion matrix C, admits each under its predicted class and demand, and measures per-class apparent blocking. That is compared against the Kaufman-Roberts blocking under the apparent load a_hat = C^T a. Grouping helpers come from src/cesnet/tiers.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.cesnet.tiers import A_TOTAL, B_TARGET, GROUP_REC, agg, grouping_arrays, pick_matrix, tier_load
from src.analytical.kaufman_roberts import bridge_equation, capacity_overhead, kaufman_roberts
from src.monte_carlo.rho_sweep import run_replications_with_rho

MC_CONDS = ["xgb_clean", "mlp_clean", "xgb_drift"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--arrivals", type=int, default=3_000_000)
    a_args = ap.parse_args()

    z = np.load(ROOT / "data" / "processed" / "cesnet_definitive.npz", allow_pickle=True)
    names = [str(x) for x in z["category_names"]]
    ct, tiers, t = grouping_arrays(names, GROUP_REC)
    t = t.astype(np.int64)
    sup = z["train_support"].astype(float)
    a = tier_load(sup, ct, tiers)

    tier_lbl = [f"t{i}(AU{int(x)})" for i, x in enumerate(t)]
    print(f"A_total={A_TOTAL}  M={a_args.M}  arrivals/rep={a_args.arrivals:,}  tiers AU={list(t)}")
    print(f"offered load a = {np.round(a,2)}")

    out = {}
    for cond in MC_CONDS:
        # the median-balanced-accuracy seed, the same matrix the dimensioning chain uses
        cm, _ = pick_matrix(z, cond)
        if cm is None:
            print(f"  [skip] {cond} absent")
            continue
        C = agg(cm, ct, tiers)
        V = capacity_overhead(a, t, B_TARGET, V_start=1)
        a_hat = bridge_equation(C, a, normalise=True)
        _, B_an = kaufman_roberts(V, a_hat, t)
        allB, _, _ = run_replications_with_rho(
            V, a, C, rho=0.0, demands=t, M=a_args.M, n_arrivals=a_args.arrivals, base_seed=1)
        Bsim = allB.mean(axis=0)
        Bsd = allB.std(axis=0)
        half = 1.96 * Bsd / np.sqrt(a_args.M)
        print(f"\n=== {cond}  bacc={float(np.mean(np.diag(C))):.3f}  V_nominal={V} ===")
        print(f"  {'tier':9s} {'B_analytic':>11s} {'B_sim':>11s} {'95%half':>10s} {'rel_err%':>9s}")
        pos = B_an > 0
        rel = np.where(pos, (Bsim - B_an) / np.where(pos, B_an, 1) * 100, np.nan)
        for k in range(len(t)):
            print(f"  {tier_lbl[k]:9s} {B_an[k]:11.3e} {Bsim[k]:11.3e} {half[k]:10.2e} {rel[k]:9.2f}")
        # the 1e-9 floor drops tiers whose analytical blocking is numerical noise
        mask = B_an > 1e-9
        maxrel = np.max(np.abs(rel[mask])) if mask.any() else float("nan")
        meanrel = np.mean(np.abs(rel[mask])) if mask.any() else float("nan")
        print(f"  max |rel err| over non-trivial tiers = {maxrel:.2f}%")
        out[cond] = {
            "C": C, "V": V, "bacc": float(np.mean(np.diag(C))),
            "B_analytical": B_an, "B_sim_mean": Bsim, "ci_half": half,
            "rel_err_pct": rel, "max_rel_err_pct": maxrel, "mean_rel_err_pct": meanrel,
            "M": a_args.M, "n_arrivals": a_args.arrivals,
        }
    if out:
        np.savez(ROOT / "data" / "processed" / "cesnet_mc_results.npz", **out)
        print(f"\nsaved data/processed/cesnet_mc_results.npz ({len(out)} conditions)")


if __name__ == "__main__":
    main()
