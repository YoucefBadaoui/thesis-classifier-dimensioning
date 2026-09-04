"""Bootstrap confidence intervals for the CESNET dimensioning conclusions.

C-only redraws each row of the raw count matrix as Multinomial(n_i, C_i.) with the offered-load prior and the nominal capacity held fixed. Joint also redraws the 23 category supports as Multinomial(N, p), moving the load, V and r*; only that design gives r* an interval. Mean holding times stay at point estimates, since the archive keeps no within-category dispersion.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.scenarios import compute_wbd
from src.analytical.kaufman_roberts import blocking_deviation, capacity_overhead
from src.cesnet.tiers import (
    B_TARGET,
    CONDS,
    GROUP_REC,
    TIER_AU,
    agg,
    grouping_arrays,
    pick_matrix,
    tier_loads,
)
from src.analytical.recall_thresholds import per_class_recall_search

# line-buffered so a piped log follows a long run
sys.stdout.reconfigure(line_buffering=True)

NPZ = ROOT / "data" / "processed" / "cesnet_definitive.npz"
OUT = ROOT / "data" / "processed" / "cesnet_bootstrap_ci.npz"
EPSILON = 0.05

_G = {}


def _init(ct, tiers, t, sup, holdm, cms):
    _G.update(ct=ct, tiers=tiers, t=t, sup=sup, holdm=holdm, cms=cms)
    # the C-only design leaves the offered loads and capacities fixed
    _G["fixed"] = _loads_and_capacities(sup, holdm, ct, tiers, t)


def _resample_rows(cm, rng):
    out = np.zeros_like(cm)
    for i in range(cm.shape[0]):
        n = int(round(cm[i].sum()))
        if n <= 0:
            continue
        out[i] = rng.multinomial(n, cm[i] / n)
    return out


def _loads_and_capacities(sup, holdm, ct, tiers, t):
    """Tier loads under both priors and their nominal capacities."""
    a_c, a_e = tier_loads(sup, holdm, ct, tiers)
    Vc = capacity_overhead(a_c, t, B_TARGET, V_start=1)
    Ve = capacity_overhead(a_e, t, B_TARGET, V_start=1)
    return a_c, a_e, Vc, Ve


def _evaluate(rng, a_c, a_e, Vc, Ve):
    """Resample every condition's matrix and dimension it under both priors."""
    ct, tiers, t = _G["ct"], _G["tiers"], _G["t"]
    row = []
    for cm in _G["cms"]:
        C = agg(_resample_rows(cm, rng), ct, tiers)
        out = []
        for a, V in ((a_c, Vc), (a_e, Ve)):
            dev = blocking_deviation(V, a, C, t)
            Vp = capacity_overhead(dev["a_hat"], t, B_TARGET, V_start=V)
            out += [(Vp - V) / V, compute_wbd(C, a, t)]
        row.append(out)
    return np.array(row)


def _rep_conly(seed):
    rng = np.random.default_rng(seed)
    return _evaluate(rng, *_G["fixed"])


def _rep_joint(seed):
    rng = np.random.default_rng(seed)
    ct, tiers, t = _G["ct"], _G["tiers"], _G["t"]
    sup, holdm = _G["sup"], _G["holdm"]
    N = int(round(sup.sum()))
    sup_b = rng.multinomial(N, sup / sup.sum()).astype(float)
    a_c, a_e, Vc, Ve = _loads_and_capacities(sup_b, holdm, ct, tiers, t)
    rstar = per_class_recall_search(a_e, t, Ve, B_TARGET, EPSILON)
    return _evaluate(rng, a_c, a_e, Vc, Ve), np.array([Vc, Ve], dtype=float), rstar


def qs(x):
    """Median and the 95% percentile interval over the replicate axis."""
    return (np.median(x, axis=0),
            np.percentile(x, 2.5, axis=0),
            np.percentile(x, 97.5, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=min(24, mp.cpu_count()))
    a = ap.parse_args()
    B = a.replicates

    z = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in z["category_names"]]
    ct, tiers, t = grouping_arrays(names, GROUP_REC)
    sup = z["train_support"].astype(float)
    holdm = z["hold_mean"]

    cms, conds = [], []
    for cond in CONDS:
        cm, _ = pick_matrix(z, cond)
        if cm is None:
            continue
        cms.append(cm.astype(float))
        conds.append(cond)
    print(f"[boot] {B} replicates, {len(conds)} conditions, "
          f"evaluation row supports {int(cms[0].sum())} flows total")

    ini = (ct, tiers, t, sup, holdm, cms)
    with mp.Pool(a.workers, initializer=_init, initargs=ini) as pool:
        res_c = np.array(pool.map(_rep_conly, range(10_000, 10_000 + B)))
        joint = pool.map(_rep_joint, range(50_000, 50_000 + B))
    res_j = np.array([r[0] for r in joint])
    V_j = np.array([r[1] for r in joint])
    rstar_j = np.array([r[2] for r in joint])

    # res_* axes: (replicate, condition, [dVc, WBDc, dVe, WBDe])
    labels = ["dVoverV_count", "wbd_count", "dVoverV_erlang", "wbd_erlang"]
    save = {"conds": np.array(conds), "replicates": np.array(B),
            "tier_au": np.array(TIER_AU), "epsilon": np.array(EPSILON)}

    for tag, res in (("conly", res_c), ("joint", res_j)):
        med, lo, hi = qs(res)
        print(f"\n===== {tag} bootstrap ({B} replicates) =====")
        for ci, cond in enumerate(conds):
            print(f"  {cond:13s} "
                  f"dV/V(erl) {med[ci,2]*100:+6.3f}% [{lo[ci,2]*100:+6.3f}, {hi[ci,2]*100:+6.3f}]   "
                  f"WBD(erl) {med[ci,3]:6.2f} [{lo[ci,3]:6.2f}, {hi[ci,3]:6.2f}]")
        for li, lab in enumerate(labels):
            save[f"{tag}_{lab}_median"] = med[:, li]
            save[f"{tag}_{lab}_lo"] = lo[:, li]
            save[f"{tag}_{lab}_hi"] = hi[:, li]

    for tag, (m, lo, hi) in (("V", qs(V_j)), ("rstar", qs(rstar_j))):
        save[f"joint_{tag}_median"] = m
        save[f"joint_{tag}_lo"] = lo
        save[f"joint_{tag}_hi"] = hi
        print(f"  joint {tag}: median {np.round(m, 3)} "
              f"CI [{np.round(lo, 3)}, {np.round(hi, 3)}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    main()
