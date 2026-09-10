"""Dimensioning under the flow-count matrix versus the duration-weighted matrix.

Reads data/processed/cesnet_duration_weighted.npz and reports, per classifier condition and per offered-load prior, the load shift Delta L = sum_j a_hat_j t_j - sum_k a_k t_k, the aggregate allocation-unit demand created (positive) or released (negative) by misclassification; the capacity overhead Delta V / V; the weighted bandwidth deficit; and the per-class minimum-recall thresholds r*. Whether these agree across the two matrices is what decides if the conditional independence behind a_hat = C^T a matters in practice. Under uniform spillover r* depends on the load and demand geometry alone, so it is reported once per prior rather than per condition.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.scenarios import compute_wbd
from src.analytical.kaufman_roberts import blocking_deviation, capacity_overhead, row_normalise
from src.cesnet.tiers import B_TARGET, GROUP_REC, agg, grouping_arrays, tier_loads
from src.analytical.recall_thresholds import per_class_recall_search

DEF = ROOT / "data" / "processed" / "cesnet_definitive.npz"
DW = ROOT / "data" / "processed" / "cesnet_duration_weighted.npz"
OUT = ROOT / "data" / "processed" / "cesnet_duration_compare.npz"
EPSILON = 0.05


def dimension_with_load_shift(C, a, t):
    V = capacity_overhead(a, t, B_TARGET, V_start=1)
    dev = blocking_deviation(V, a, C, t)
    Vp = capacity_overhead(dev["a_hat"], t, B_TARGET, V_start=V)
    a_hat = dev["a_hat"]
    dL = float(np.dot(a_hat, t) - np.dot(a, t))
    return V, (Vp - V) / V, compute_wbd(C, a, t), dL


def main():
    zd = np.load(DEF, allow_pickle=True)
    z = np.load(DW, allow_pickle=True)
    names = [str(x) for x in z["category_names"]]
    assert names == [str(x) for x in zd["category_names"]], "category order mismatch"
    ct, tiers, t = grouping_arrays(names, GROUP_REC)

    sup = zd["train_support"].astype(float)
    holdm = zd["hold_mean"]
    a_count, a_erl = tier_loads(sup, holdm, ct, tiers)

    # "<cond>_count" is the raw count matrix, "<cond>_C_count" its row-normalised twin, not a separate condition
    conds = sorted({k[: -len("_count")] for k in z.files
                    if k.endswith("_count") and not k.endswith("_C_count")})
    print(f"conditions: {conds}")
    print(f"a_count  = {np.round(a_count, 2)}")
    print(f"a_erlang = {np.round(a_erl, 2)}   t = {list(map(int, t))}\n")

    save = {"conds": np.array(conds), "a_count": a_count, "a_erlang": a_erl,
            "t": t, "epsilon": np.array(EPSILON)}
    rows = {k: [] for k in ("bacc_count", "bacc_dur", "ratio_min", "ratio_med",
                            "ratio_max", "cell_absdev_max", "cell_absdev_mean")}
    for prior in ("count", "erlang"):
        for m in ("cnt", "dur"):
            for q in ("V", "dVoverV", "wbd", "dL", "dir"):
                rows[f"{prior}_{m}_{q}"] = []

    for cond in conds:
        cnt = z[f"{cond}_count"]
        dsum = z[f"{cond}_dursum"]
        ratio = z[f"{cond}_ratio"]
        C_cnt = agg(cnt, ct, tiers)
        C_dur = agg(dsum, ct, tiers)
        finite = np.isfinite(ratio) & (cnt > 0)
        # ratio summary over cells with real mass, so a one-flow cell cannot dominate
        mass = row_normalise(cnt)
        sig = finite & (mass >= 0.01)
        rows["bacc_count"].append(float(np.mean(np.diag(C_cnt))))
        rows["bacc_dur"].append(float(np.mean(np.diag(C_dur))))
        rmin, rmed, rmax = ((float(np.min(ratio[sig])), float(np.median(ratio[sig])),
                             float(np.max(ratio[sig]))) if sig.any()
                            else (np.nan, np.nan, np.nan))
        rows["ratio_min"].append(rmin)
        rows["ratio_med"].append(rmed)
        rows["ratio_max"].append(rmax)
        d = np.abs(C_dur - C_cnt)
        rows["cell_absdev_max"].append(float(d.max()))
        rows["cell_absdev_mean"].append(float(d.mean()))

        print(f"===== {cond} =====")
        print(f"  six-tier diagonal mean: count {np.mean(np.diag(C_cnt)):.4f}   "
              f"duration {np.mean(np.diag(C_dur)):.4f}")
        print(f"  E[D|i,j]/E[D|i] over cells with >=1% row mass: "
              f"min {rmin:.3f}  median {rmed:.3f}  max {rmax:.3f}")
        print(f"  six-tier |C_dur - C_count|: max {d.max():.4f}  mean {d.mean():.5f}")
        for prior, a in (("count", a_count), ("erlang", a_erl)):
            got = {}
            for tag, label, C in (("cnt", "count", C_cnt), ("dur", "dur", C_dur)):
                V, dV, w, dL = dimension_with_load_shift(C, a, t)
                direction = "lo->hi" if dL > 0 else "hi->lo"
                got[tag] = (label, V, dV, w, dL, direction)
                for q, v in zip(("V", "dVoverV", "wbd", "dL", "dir"),
                                (V, dV, w, dL, direction)):
                    rows[f"{prior}_{tag}_{q}"].append(v)
            flip = "  (sign flip)" if got["cnt"][5] != got["dur"][5] else ""
            for tag in ("cnt", "dur"):
                label, V, dV, w, dL, direction = got[tag]
                print(f"  {prior:7s} {label:5s} : V={V:4d} dV/V={dV*100:+7.3f}% "
                      f"WBD={w:7.2f} dL={dL:+8.3f} {direction}"
                      f"{flip if tag == 'dur' else ''}")
        print()

    for k, v in rows.items():
        save[k] = np.array(v)

    # r* depends on (a, t, V) only; report it per prior.
    for prior, a in (("count", a_count), ("erlang", a_erl)):
        V = capacity_overhead(a, t, B_TARGET, V_start=1)
        rs = per_class_recall_search(a, t, V, B_TARGET, EPSILON)
        save[f"rstar_{prior}"] = rs
        save[f"rstar_{prior}_rank"] = np.argsort(-rs)
        save[f"V_{prior}"] = np.array(V)
        print(f"r* ({prior} prior, V={V}) = {np.round(rs, 3)}  "
              f"ranking tightest first {list(map(int, np.argsort(-rs)))}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    main()
