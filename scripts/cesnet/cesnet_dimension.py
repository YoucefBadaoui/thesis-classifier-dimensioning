"""Dimensioning, re-tiering and stability on the CESNET ensemble matrices.

Reads data/processed/cesnet_definitive.npz and cesnet_degraded.npz. For the Recommended OTT grouping, which drops the machine-to-machine background categories, and for an all-inclusive whole-backbone control grouping, reports the flow-count and Erlang-corrected (count times mean holding time) offered-load priors with cov(a,t) and the null-normalised tier direction bias, Kaufman-Roberts dimensioning per classifier condition, multi-seed stability, and an AU-ladder perturbation table. The 6-tier object here is the K=6 H1 scenario, separate from the K=23 Finding-F1 anchor in cesnet_highk_real.npz.
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
    GROUP_ALL,
    GROUP_REC,
    SEEDS,
    TIER_AU,
    agg,
    dimension,
    grouping_arrays,
    pick_matrix,
    tier_load,
    tier_loads,
    tier_vec,
    updown_flows,
)

NPZ = ROOT / "data" / "processed" / "cesnet_definitive.npz"
DEGRADED_NPZ = ROOT / "data" / "processed" / "cesnet_degraded.npz"
OUT = ROOT / "data" / "processed" / "cesnet_dimension.npz"

# tier-bias verdict bands, Ch.4 sec:cesnet-auperturb
BIAS_HI, BIAS_LO = 1.25, 0.8

# rows collected per classifier condition, and their archive key suffixes
ROW_COLS = ("cond", "bacc", "Vc", "dVc", "Wc", "dc", "Ve", "dVe", "We", "de")
SAVE_AS = {"cond": "conds", "bacc": "bacc", "Vc": "V_count", "dVc": "dVoverV_count",
           "Wc": "wbd_count", "dc": "dir_count", "Ve": "V_erlang",
           "dVe": "dVoverV_erlang", "We": "wbd_erlang", "de": "dir_erlang"}


def null_bias(cm, ct, tiers, t):
    """Null-normalized tier-level down/up: observed vs uniform-spillover null."""
    Cn = cm.copy().astype(float)
    rs = Cn.sum(1, keepdims=True)
    Cn = Cn / np.where(rs == 0, 1, rs)
    Null = np.zeros_like(Cn)
    n = Cn.shape[0]
    for i in range(n):
        off = 1.0 - Cn[i, i]
        Null[i, :] = off / (n - 1)
        Null[i, i] = Cn[i, i]
    Ct_obs = agg(cm, ct, tiers)
    Ct_null = agg(Null * rs, ct, tiers)
    sup = tier_vec(cm.sum(1), ct, tiers)
    a = sup / sup.sum() * A_TOTAL
    up_o, dn_o = updown_flows(Ct_obs, a, t)
    up_n, dn_n = updown_flows(Ct_null, a, t)
    r_obs = dn_o / up_o if up_o else float("inf")
    r_null = dn_n / up_n if up_n else float("inf")
    return r_obs, r_null, (r_obs / r_null if r_null else float("inf"))


def run_grouping(z, names, mapping, label, save, prefix):
    ct, tiers, t = grouping_arrays(names, mapping)
    sup = z["train_support"].astype(float)
    holdm = z["hold_mean"]
    a_count, a_erl = tier_loads(sup, holdm, ct, tiers)
    cov = lambda a: population_covariance(a, t)
    excl = int((ct < 0).sum())
    exclmass = sup[ct < 0].sum() / sup.sum() * 100
    print(f"\n===== {label}  (K={len(t)}, AU={list(t)}, excluded={excl} cats = {exclmass:.1f}% mass) =====")
    print(f"  a_count  = {np.round(a_count,2)}  cov={cov(a_count):+.3f}")
    print(f"  a_erlang = {np.round(a_erl,2)}  cov={cov(a_erl):+.3f}")
    cm0, _ = pick_matrix(z, "xgb_clean")
    r_o, r_n, bias = null_bias(cm0, ct, tiers, t)
    verdict = ("manufactures hi->lo" if bias > BIAS_HI
               else "preserves lo->hi" if bias < BIAS_LO else "binning-neutral")
    print(f"  null-norm tier bias (xgb_clean) obs/null = {r_o:.2f}/{r_n:.2f} -> {bias:.2f} [{verdict}]")
    print(f"  {'cond':12s} {'bacc':>6s} | {'count dV/V':>10s} {'WBD':>6s} {'dir':>6s} | {'erlang dV/V':>11s} {'WBD':>6s} {'dir':>6s}")
    rows = {k: [] for k in ROW_COLS}
    cms = []
    for cond in CONDS:
        cm, _ = pick_matrix(z, cond)
        if cm is None:
            continue
        C = agg(cm, ct, tiers); bacc = float(np.mean(np.diag(C)))
        Vc, dVc, wc, dc = dimension(C, a_count, t)
        Ve, dVe, we, de = dimension(C, a_erl, t)
        print(f"  {cond:12s} {bacc:6.3f} | V={Vc} {dVc*100:+6.2f}% {wc:6.1f} {dc:>6s} | "
              f"V={Ve} {dVe*100:+6.2f}% {we:6.1f} {de:>6s}")
        for k, v in zip(ROW_COLS, (cond, bacc, Vc, dVc, wc, dc, Ve, dVe, we, de)):
            rows[k].append(v)
        cms.append(C)
    save[f"{prefix}_cov_count"] = np.array(cov(a_count))
    save[f"{prefix}_cov_erlang"] = np.array(cov(a_erl))
    save[f"{prefix}_a_count"] = a_count
    save[f"{prefix}_a_erlang"] = a_erl
    save[f"{prefix}_t"] = t
    save[f"{prefix}_excl_mass"] = np.array(exclmass)
    save[f"{prefix}_tier_bias"] = np.array(bias)
    for k, name in SAVE_AS.items():
        save[f"{prefix}_{name}"] = np.array(rows[k])
    save[f"{prefix}_tier_cm"] = np.array(cms)
    return ct, tiers, t, a_count, a_erl


def stability(z, names):
    ct, tiers, t = grouping_arrays(names, GROUP_REC)
    sup = z["train_support"].astype(float)
    a = tier_load(sup, ct, tiers)
    print("\n===== multi-seed stability (Recommended, count prior) =====")
    for cond in ["xgb_clean", "mlp_clean"]:
        dv_pct, wbd = [], []
        for s in SEEDS:
            key = f"{cond}_s{s}"
            if key not in z.files:
                continue
            C = agg(z[key], ct, tiers)
            V, dV, w, d = dimension(C, a, t)
            dv_pct.append(dV * 100)
            wbd.append(w)
        dv, wb = np.array(dv_pct), np.array(wbd)
        print(f"  {cond}: dV/V {dv.mean():+.2f}% (range {dv.min():+.2f}..{dv.max():+.2f}), "
              f"WBD {wb.mean():.1f} (range {wb.min():.1f}..{wb.max():.1f}) over {len(dv)} seeds")


def add_degraded(save):
    if not DEGRADED_NPZ.exists():
        return
    zd = np.load(DEGRADED_NPZ, allow_pickle=True)
    dnames = [str(x) for x in zd["category_names"]]
    ct, tiers, t = grouping_arrays(dnames, GROUP_REC)
    print("\n===== degraded classifiers (tier-6 aggregation) =====")
    suffix = "_cat_counts"
    for v in sorted(k[:-len(suffix)] for k in zd.files if k.endswith(suffix)):
        C = agg(zd[v + suffix], ct, tiers)
        tier6 = float(np.mean(np.diag(C)))
        cat = float(zd["bacc_" + v])
        save[v + "_tier6_bacc"] = np.array(tier6)
        save[v + "_cat_bacc"] = np.array(cat)
        print(f"  {v:20s} tier-6 bacc={tier6:.4f}  23-class bacc={cat:.4f}")


def au_perturbation(z, names, save):
    """cov(a,t) and V_nominal under +/-1 AU moves of each tier, count prior."""
    ct, tiers, t = grouping_arrays(names, GROUP_REC)
    sup = z["train_support"].astype(float)
    a = tier_load(sup, ct, tiers)
    rows = []
    for k in range(len(t)):
        for delta in (-1, 1):
            tp = t.astype(int).copy()
            tp[k] = max(1, int(tp[k]) + delta)
            covp = population_covariance(a, tp)
            Vp = int(capacity_overhead(a, tp, B_TARGET, V_start=1))
            rows.append([k, delta, int(tp[k]), covp, Vp])
    arr = np.array(rows, float)
    save["auperturb"] = arr
    save["auperturb_cols"] = np.array(["tier_idx", "delta", "new_au", "cov_count", "V_count"])
    covs = arr[:, 3]
    flips = "all stay negative" if covs.max() < 0 else "SIGN FLIPS under a +/-1 move"
    print(f"\nAU perturbation cov range {covs.min():+.3f} to {covs.max():+.3f} [{flips}]")


def main():
    z = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in z["category_names"]]
    save = {
        "size": z["size"], "A_total": np.array(A_TOTAL), "B_target": np.array(B_TARGET),
        "tier_au": np.array(TIER_AU),
    }
    print(f"size={str(z['size'])}  categories={len(names)}  A_total={A_TOTAL}")
    run_grouping(z, names, GROUP_REC, "Recommended OTT (6-tier, 8 excl)", save, "rec")
    run_grouping(z, names, GROUP_ALL, "All-inclusive whole-backbone control (0 excl)", save, "allk")
    stability(z, names)
    add_degraded(save)
    au_perturbation(z, names, save)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **save)
    print(f"\n[saved] {OUT}  ({len(save)} keys)")


if __name__ == "__main__":
    main()
