"""Recompute the counterexamples and reduction conventions quoted in Chapters 3, 5 and 7 (appendix S16, Table A.10) from the frozen archives and the analytical kernel, and write data/processed/review_counterexamples.json. Exit 1 if a quoted number fails to reproduce.

Keys: ott_r080 and fg_r080 (high-target family against its second-target variant at r = 0.80), uniform_082 / uniform_087 / uniform_02894 (uniform spillover at those recalls), fg_rstar_eps5 (5G threshold of both at eps = 5 percent), cesnet_excluded_predictions (share of retained-flow predictions dropped by the six-tier reduction, the unconstrained minimum capacity under that convention, and the outcome when those predictions get a fixed reservation instead), cesnet_tier_weighting (pooled evaluation counts against prior-consistent weights inside a tier under both offered-load priors, with the residual between the category-level and the tier-level composition), cesnet_duration_mixing (the same comparison for the duration control, whose tier rows are pooled by flow count for the count matrix and by holding-time mass for the duration matrix), ott_rank_test_k5 (exact permutation p-values of the OTT rank tests with tied ranks).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

from src.analytical.constants import A_5G, A_OTT, B_TARGET_DEFAULT, T_5G, T_OTT, V_NOMINAL_5G, V_NOMINAL_OTT
from src.analytical.kaufman_roberts import capacity_overhead, row_normalise, uniform_spillover_cm
from src.analytical.scenarios import make_5g_cm
from src.cesnet.tiers import A_TOTAL, GROUP_REC, TIER_AU, agg, dimension, grouping_arrays, tier_loads

SELECTED = {"xgb_clean": 123, "xgb_drift": 42, "lgbm_clean": 7, "lgbm_drift": 42, "mlp_clean": 42, "mlp_drift": 42}
# family V' and variant V' quoted in the text
EXPECTED = {"ott_r080": (706, 715), "fg_r080": (291, 293), "uniform_082": 530, "uniform_087": 522, "uniform_02894": 620}
FALLBACK_AU = (1, 2, 4, 6, 10, 15)


def target_family(a, t, r):
    """All error mass to the highest-t class; that class spreads its own error mass uniformly."""
    K, tgt, off = len(t), int(np.argmax(t)), 1.0 - r
    C = np.zeros((K, K))
    for i in range(K):
        C[i, i] = r
        if i == tgt:
            C[i, [j for j in range(K) if j != i]] = off / (K - 1)
        else:
            C[i, tgt] = off
    return C


def v_prime(C, a, t, V):
    return int(capacity_overhead(C.T @ a, t, B_TARGET_DEFAULT, V_start=V))


def second_target_variant(C, t, r):
    """Same family, but the highest-t row keeps its error mass on the second-highest-t class."""
    order = np.argsort(t)
    hi, second = int(order[-1]), int(order[-2])
    D = C.copy()
    D[hi, :] = 0.0
    D[hi, hi], D[hi, second] = r, 1.0 - r
    return D


def agg_weighted(cm, w, ct, tiers):
    """Tier matrix with each retained category weighted by w inside its tier, after dropping predictions outside the map."""
    C = row_normalise(cm)
    pos = {t: i for i, t in enumerate(tiers)}
    K = len(tiers)
    o, aw = np.zeros((K, K)), np.zeros(K)
    for i in range(cm.shape[0]):
        if ct[i] < 0:
            continue
        inmap = sum(C[i, j] for j in range(cm.shape[1]) if ct[j] >= 0)
        for j in range(cm.shape[1]):
            if ct[j] >= 0:
                o[pos[ct[i]], pos[ct[j]]] += w[i] * C[i, j] / inmap
        aw[pos[ct[i]]] += w[i]
    return o / aw[:, None]


def category_composition(cm, w, ct, tiers):
    """Apparent tier loads from the category-level composition under the prior w, with predictions outside the map dropped."""
    keep = ct >= 0
    a = np.where(keep, w, 0.0)
    a = a / a.sum() * A_TOTAL
    sub = cm[np.ix_(keep, keep)]
    ah = (sub / sub.sum(1, keepdims=True)).T @ a[keep]
    pos = {t: i for i, t in enumerate(tiers)}
    out = np.zeros(len(tiers))
    for c, v in zip(np.where(keep)[0], ah):
        out[pos[ct[c]]] += v
    return out


def stats(C, a, t):
    V, dV, wbd, direction = dimension(C, a, t)
    V_min = int(capacity_overhead(C.T @ a, t, B_TARGET_DEFAULT, V_start=1))
    dL = float(sum(a[i] * C[i, j] * (t[j] - t[i]) for i in range(len(t)) for j in range(len(t))))
    return {"V": int(V), "V_min": V_min, "overhead_pct": round(float(dV) * 100, 4), "wbd": round(float(wbd), 4), "dL": round(dL, 4), "direction": direction}


def exact_perm_p(x, y):
    """Spearman rho with average ranks, its exact two-sided permutation p over every ordering of y, and the largest |rho| those ties allow with its p."""
    r0 = float(spearmanr(x, y).statistic)
    rs = np.array([abs(spearmanr(x, y[list(q)]).statistic) for q in itertools.permutations(range(len(y)))])
    return r0, float(np.mean(rs >= abs(r0) - 1e-12)), float(rs.max()), float(np.mean(rs >= rs.max() - 1e-12))


def rstar_5g(build, eps=0.05):
    """Smallest recall on a 0.005 lattice at which the family built by build() keeps the 5G overhead within eps."""
    for r in np.arange(0.50, 1.0001, 0.005):
        C = build(round(float(r), 3))
        Vp = capacity_overhead(C.T @ A_5G, T_5G, B_TARGET_DEFAULT, V_start=V_NOMINAL_5G)
        if (Vp - V_NOMINAL_5G) / V_NOMINAL_5G <= eps:
            return round(float(r), 3)
    return None


def main() -> int:
    out, failures = {}, 0
    for key, (a, t, V) in {"ott_r080": (A_OTT, T_OTT, V_NOMINAL_OTT), "fg_r080": (A_5G, T_5G, V_NOMINAL_5G)}.items():
        C = target_family(a, t, 0.80)
        fam, alt = v_prime(C, a, t, V), v_prime(second_target_variant(C, t, 0.80), a, t, V)
        out[key] = {"V_nominal": V, "family_V": fam, "family_overhead_pct": round((fam - V) / V * 100, 3),
                    "alternative_V": alt, "alternative_overhead_pct": round((alt - V) / V * 100, 3)}
        failures += (fam, alt) != EXPECTED[key]
    for key, r in {"uniform_082": 0.82, "uniform_087": 0.87, "uniform_02894": 0.2894}.items():
        Vp = v_prime(uniform_spillover_cm(5, r), A_OTT, T_OTT, V_NOMINAL_OTT)
        out[key] = {"recall": r, "V": Vp, "overhead_pct": round((Vp - V_NOMINAL_OTT) / V_NOMINAL_OTT * 100, 3)}
        failures += Vp != EXPECTED[key]

    z = np.load(PROC / "cesnet_definitive.npz", allow_pickle=True)
    names, sup, hm = list(z["category_names"]), z["train_support"].astype(float), z["hold_mean"].astype(float)
    ct, tiers, t_tier = grouping_arrays(names, GROUP_REC)
    keep = ct >= 0
    t_cat = np.array([TIER_AU[ct[j]] if ct[j] >= 0 else 1 for j in range(len(names))], dtype=float)
    a_cat = np.where(keep, sup, 0.0)
    a_cat = a_cat / a_cat.sum() * A_TOTAL
    cesnet, weighting = {}, {}
    a6, a6e = tier_loads(sup, hm, ct, tiers)
    for cond, seed in SELECTED.items():
        cm = z[f"{cond}_s{seed}"].astype(float)
        retained, discarded = cm[keep].sum(), cm[keep][:, ~keep].sum()
        C6 = agg(cm, ct, tiers)
        V, dV, wbd, direction = dimension(C6, a6, t_tier)
        V_min = int(capacity_overhead(C6.T @ a6, t_tier, B_TARGET_DEFAULT, V_start=1))
        published = {"V": int(V), "V_min": V_min, "overhead_pct": round(float(dV) * 100, 3), "wbd": round(float(wbd), 3)}
        Cc = row_normalise(cm)
        ah = Cc.T @ a_cat
        sweep = {}
        for tb in FALLBACK_AU:
            t_b = np.where(keep, t_cat, float(tb))
            aus_b = sorted(set(t_b.tolist()))
            Vn_b = int(capacity_overhead(np.array([a_cat[t_b == u].sum() for u in aus_b]), np.array(aus_b), B_TARGET_DEFAULT, V_start=1))
            Vm_b = int(capacity_overhead(np.array([ah[t_b == u].sum() for u in aus_b]), np.array(aus_b), B_TARGET_DEFAULT, V_start=1))
            wbd_b = float(sum(a_cat[i] * Cc[i, j] * (t_b[i] - t_b[j]) for i in range(len(names)) for j in range(len(names)) if t_b[i] > t_b[j]))
            sweep[str(tb)] = {"V_nominal": Vn_b, "V_min": Vm_b, "overhead_pct": round(max(Vm_b - Vn_b, 0) / Vn_b * 100, 4), "wbd": round(wbd_b, 4)}
        cesnet[cond] = {"seed": seed, "retained_true_flows": int(retained), "discarded_predictions": int(discarded),
                        "discarded_share_pct": round(discarded / retained * 100, 3),
                        "published": published, "fallback_1au": sweep["1"], "fallback_sweep": sweep}
        Cw, Ce = agg_weighted(cm, sup, ct, tiers), agg_weighted(cm, sup * hm, ct, tiers)
        tw, pe, we = stats(Cw, a6, t_tier), stats(C6, a6e, t_tier), stats(Ce, a6e, t_tier)
        weighting[cond] = {"eval_pooled": dict(published, direction=direction), "train_weighted": tw,
                           "wbd_rel_change_pct": round((tw["wbd"] - wbd) / wbd * 100, 2),
                           "identity_residual_count": float(np.abs(category_composition(cm, sup, ct, tiers) - Cw.T @ a6).max()),
                           "erlang": {"count_pooled": pe, "prior_weighted": we,
                                      "wbd_rel_change_pct": round((we["wbd"] - pe["wbd"]) / pe["wbd"] * 100, 2),
                                      "identity_residual_erl": float(np.abs(category_composition(cm, sup * hm, ct, tiers) - Ce.T @ a6e).max())}}
    out["cesnet_excluded_predictions"] = cesnet
    zdw = np.load(PROC / "cesnet_duration_weighted.npz", allow_pickle=True)
    assert [str(x) for x in zdw["category_names"]] == names
    mixing, changed, wbd_moves = {}, [], []
    for cond in sorted({k[:-6] for k in zdw.files if k.endswith("_count") and not k.endswith("_C_count")}):
        mixing[cond] = {}
        for mat, key in (("count_matrix", "count"), ("duration_matrix", "dursum")):
            M = zdw[f"{cond}_{key}"].astype(float)
            mixing[cond][mat] = {}
            for prior, w, a in (("count", sup, a6), ("erlang", sup * hm, a6e)):
                pooled, mixed = stats(agg(M, ct, tiers), a, t_tier), stats(agg_weighted(M, w, ct, tiers), a, t_tier)
                mixing[cond][mat][prior] = {"pooled": pooled, "prior_weighted": mixed}
                wbd_moves.append(abs(mixed["wbd"] - pooled["wbd"]) / pooled["wbd"] * 100)
                if pooled["overhead_pct"] != mixed["overhead_pct"] or pooled["direction"] != mixed["direction"]:
                    changed.append((cond, mat, prior))
    mixing["largest_wbd_rel_change_pct"] = round(max(wbd_moves), 2)
    out["cesnet_duration_mixing"] = mixing
    failures += set(changed) != {("degraded_dur1", "count_matrix", "count"), ("degraded_dur1", "count_matrix", "erlang"), ("mlp_clean", "count_matrix", "erlang"), ("xgb_reduced", "duration_matrix", "count")}
    xr = mixing["xgb_reduced"]["duration_matrix"]["count"]
    failures += not (xr["pooled"]["overhead_pct"] == 0.0 and abs(xr["prior_weighted"]["overhead_pct"] - 0.1473) < 0.001 and xr["pooled"]["dL"] < 0 < xr["prior_weighted"]["dL"])
    za = np.load(PROC / "analytical_results.npz", allow_pickle=True)
    rstar = za["rstar_ott_eps5"].astype(float)
    rank = {}
    for key, x in (("load_demand_product", A_OTT * T_OTT), ("bandwidth_gap", T_OTT.max() - T_OTT)):
        rho, p, rmax, pmax = exact_perm_p(np.asarray(x, float), rstar)
        rank[key] = {"rho": round(rho, 3), "p_exact": round(p, 3), "largest_abs_rho_with_ties": round(rmax, 3), "p_of_largest": round(pmax, 3)}
    out["ott_rank_test_k5"] = rank
    failures += not (abs(rank["load_demand_product"]["rho"] + 0.632) < 0.01 and abs(rank["load_demand_product"]["p_exact"] - 0.333) < 0.01)
    val = z["val_support"].astype(float)
    ratio = (val[keep] / val[keep].sum()) / (sup[keep] / sup[keep].sum())
    out["cesnet_tier_weighting"] = {"val_over_train_share_ratio": [round(float(ratio.min()), 4), round(float(ratio.max()), 4)], "conditions": weighting}
    out["fg_rstar_eps5"] = {"family": rstar_5g(lambda r: make_5g_cm(r, "worst")), "second_target_variant": rstar_5g(lambda r: second_target_variant(make_5g_cm(r, "worst"), T_5G, r))}
    failures += out["fg_rstar_eps5"] != {"family": 0.81, "second_target_variant": 0.83}
    failures += not (ratio.max() < 1.0005 and ratio.min() > 0.9995)
    mlp = cesnet["mlp_clean"]
    failures += not (abs(mlp["discarded_share_pct"] - 5.764) < 0.01 and mlp["fallback_1au"]["overhead_pct"] == 0.0)
    failures += not abs(cesnet["xgb_clean"]["discarded_share_pct"] - 0.805) < 0.01
    failures += cesnet["xgb_clean"]["published"]["V_min"] != 678
    failures += any(max(w["identity_residual_count"], w["erlang"]["identity_residual_erl"]) > 1e-9 for w in weighting.values())
    e = weighting["mlp_clean"]["erlang"]
    failures += not (abs(e["count_pooled"]["overhead_pct"] - 0.608) < 0.01 and e["prior_weighted"]["overhead_pct"] == 0.0 and e["prior_weighted"]["direction"] == "hi->lo")
    for cond in ("xgb_clean", "xgb_drift", "lgbm_clean", "lgbm_drift"):
        e = weighting[cond]["erlang"]
        failures += any(e["count_pooled"][k] != e["prior_weighted"][k] for k in ("V", "overhead_pct", "direction"))

    (PROC / "review_counterexamples.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("FAIL" if failures else "OK", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
