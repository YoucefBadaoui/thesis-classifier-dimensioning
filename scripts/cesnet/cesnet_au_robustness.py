"""Sensitivity of the CESNET dimensioning conclusions to the class-to-AU ladder.

Four ladders: baseline {1, 2, 4, 6, 10, 15} AU from the Chapter 3 rate arguments, conservative from the 99th-percentile downstream rate, throughput from the mean downstream rate, burst from the 99th-percentile packet-sequence rate. The three derived ladders aggregate over a tier with flow-count weights, normalised to the lightest tier and rounded onto the integer AU lattice. Monotonicity is not imposed, so a ladder may reorder the tiers.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.kaufman_roberts import capacity_overhead, population_covariance
from src.cesnet.tiers import (
    A_TOTAL,
    B_TARGET,
    CONDS,
    GROUP_REC,
    TIER_AU,
    agg,
    dimension,
    grouping_arrays,
    pick_matrix,
    tier_vec,
)
from src.analytical.recall_thresholds import per_class_recall_search

NPZ = ROOT / "data" / "processed" / "cesnet_definitive.npz"
OUT = ROOT / "data" / "processed" / "cesnet_au_robustness.npz"
EPSILON = 0.05


def derive_ladder(stat, sup, ct, tiers):
    """Tier rate aggregate normalised to the lightest tier, rounded onto the integer AU lattice."""
    num = tier_vec(stat * sup, ct, tiers)
    den = tier_vec(sup, ct, tiers)
    per_tier = num / np.where(den == 0, 1, den)
    base = per_tier[per_tier > 0].min()
    raw = per_tier / base
    lad = np.maximum(1, np.rint(raw)).astype(int)
    return lad, per_tier


def main():
    z = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in z["category_names"]]
    ct, tiers, _ = grouping_arrays(names, GROUP_REC)
    sup = z["train_support"].astype(float)
    holdm = z["hold_mean"]

    a_count = tier_vec(sup, ct, tiers)
    a_count = a_count / a_count.sum() * A_TOTAL
    a_erl = tier_vec(sup * holdm, ct, tiers)
    a_erl = a_erl / a_erl.sum() * A_TOTAL

    lad_cons, raw_cons = derive_ladder(z["down_p99"], sup, ct, tiers)
    lad_thr, raw_thr = derive_ladder(z["down_mean"], sup, ct, tiers)
    lad_brs, raw_brs = derive_ladder(z["ppi_rate_p99"], sup, ct, tiers)

    variants = {
        "baseline": np.array(TIER_AU, dtype=int),
        "conservative": lad_cons,
        "throughput": lad_thr,
        "burst": lad_brs,
    }

    covs = {nm: (population_covariance(a_count, t), population_covariance(a_erl, t))
            for nm, t in variants.items()}
    print(f"offered load: count a = {np.round(a_count, 2)}  Erlang a = {np.round(a_erl, 2)}")
    for nm, t in variants.items():
        print(f"  {nm:13s} t = {list(map(int, t))}   cov(a_erl,t) = {covs[nm][1]:+.3f}")

    save = {
        "variant_names": np.array(list(variants)),
        "tier_index": np.array(tiers),
        "a_count": a_count,
        "a_erlang": a_erl,
        "conds": np.array(CONDS),
        "epsilon": np.array(EPSILON),
        "raw_conservative_mbps": raw_cons,
        "raw_throughput_mbps": raw_thr,
        "raw_burst_mbps": raw_brs,
    }

    for nm, t in variants.items():
        save[f"{nm}_t"] = t
        save[f"{nm}_cov_count"] = np.array(covs[nm][0])
        save[f"{nm}_cov_erlang"] = np.array(covs[nm][1])
        rows = {k: [] for k in ["Vc", "dVc", "Wc", "dc", "Ve", "dVe", "We", "de", "bacc"]}
        print(f"\n{nm}  t = {list(map(int, t))}")
        print(f"  {'cond':13s} {'bacc':>6s} | {'V':>5s} {'count dV/V':>10s} {'WBD':>7s} "
              f"| {'V':>5s} {'erl dV/V':>9s} {'WBD':>7s}")
        for cond in CONDS:
            cm, _ = pick_matrix(z, cond)
            if cm is None:
                continue
            C = agg(cm, ct, tiers)
            bacc = float(np.mean(np.diag(C)))
            Vc, dVc, wc, dc = dimension(C, a_count, t)
            Ve, dVe, we, de = dimension(C, a_erl, t)
            print(f"  {cond:13s} {bacc:6.3f} | {Vc:5d} {dVc*100:+9.2f}% {wc:7.1f} "
                  f"| {Ve:5d} {dVe*100:+8.2f}% {we:7.1f}")
            for k, v in zip(rows, [Vc, dVc, wc, dc, Ve, dVe, we, de, bacc]):
                rows[k].append(v)
        for k, v in rows.items():
            save[f"{nm}_{k}"] = np.array(v)

        # r* on the Erlang-corrected prior at this ladder's nominal capacity
        V_nom = capacity_overhead(a_erl, t, B_TARGET, V_start=1)
        rstar = per_class_recall_search(a_erl, t, V_nom, B_TARGET, EPSILON)
        order = np.argsort(-rstar)
        save[f"{nm}_V_nominal"] = np.array(V_nom)
        save[f"{nm}_rstar"] = rstar
        save[f"{nm}_rstar_rank"] = order
        print(f"  V_nominal = {V_nom}  r* (eps={EPSILON:.0%}) = {np.round(rstar, 3)}  "
              f"tightest tiers {list(map(int, order))}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    main()
