"""Multi-link EFPA cascade over a tandem of full-availability groups.

Topology from src/analytical/constants.py: CASCADE_LINKS links in series. The five OTT/IPTV classes traverse every link and carry the distortion a_hat = C^T a; each link also carries a local copy of the mix scaled by CASCADE_BG_FACTOR, which decorrelates the links. A call is lost if any link on its route is full.
Computed: reduced-load EFPA against an independent-links baseline, EFPA error against Monte Carlo as the background share rises, and the end-to-end deviation of a_hat from a versus path length L under a per-link and a fixed end-to-end grade of service. Monte Carlo: M = 30, seeds 1 to 30, 5e6 arrivals, Student-t 95%, gate 10,000 blocked calls or 1% relative half-width.
"""

import functools
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.efpa import efpa_fixed_point, efpa_independent
from src.analytical.kaufman_roberts import (
    bridge_equation,
    capacity_overhead,
    row_normalise,
    uniform_spillover_cm,
)
from src.analytical.constants import (
    A_OTT, T_OTT, B_TARGET_DEFAULT, CLASS_ORDER_OTT,
    CASCADE_LINKS, CASCADE_LINK_NAMES, CASCADE_BG_FACTOR,
)
from src.monte_carlo.multilink import run_replications_multilink

PROCESSED = ROOT / "data" / "processed"
K = len(A_OTT)
M_REPS = 30
N_ARRIVALS = 5_000_000
MC_SEED = 1
N_MIN_BLOCKS = 10_000       # Chapter 6 rare-event floor
REL_HW_MAX = 0.01           # Chapter 6 relative half-width criterion
WEIGHT = A_OTT * T_OTT      # AU*Erl weighting (thesis system-blocking convention)
UNIFORM_R = 0.85            # uniform-spillover recall of the stage-C reference matrix
E2E_AT = float(np.sum(A_OTT * T_OTT))   # monitored offered load per link, AU*Erl


def build_streams(n_links: int, a_e2e: np.ndarray, bg_factor: np.ndarray):
    """Streams for an L-link cascade: K end-to-end plus K local per link.

    Background uses the undistorted mix A_OTT; only a_e2e carries the distortion.
    """
    routes = [list(range(n_links))] * K
    offered = [np.asarray(a_e2e, dtype=float)]
    demands = [T_OTT]
    for l in range(n_links):
        routes += [[l]] * K
        offered.append(bg_factor[l] * A_OTT)
        demands.append(T_OTT)
    return np.concatenate(offered), np.concatenate(demands), routes


def per_link_capacities(n_links: int, bg_factor: np.ndarray, b_target: float) -> np.ndarray:
    dem = np.concatenate([T_OTT, T_OTT])
    V = np.zeros(n_links, dtype=int)
    for l in range(n_links):
        loads = np.concatenate([A_OTT, bg_factor[l] * A_OTT])
        V[l] = capacity_overhead(loads, dem, b_target, V_start=1)
    return V


def sublink_target(n_links: int, b_e2e: float = B_TARGET_DEFAULT) -> float:
    """Per-link sub-target giving end-to-end GoS b_e2e under link independence."""
    return 1.0 - (1.0 - b_e2e) ** (1.0 / n_links)


def efpa_e2e(V, a_e2e, bg_factor, n_links) -> np.ndarray:
    off, dem, routes = build_streams(n_links, a_e2e, bg_factor)
    res = efpa_fixed_point(V, off, dem, routes)
    assert res["converged"], f"EFPA did not converge (residual={res['residual']:.3g})"
    return res["B_e2e"][:K]


def indep_e2e(V, a_e2e, bg_factor, n_links) -> np.ndarray:
    off, dem, routes = build_streams(n_links, a_e2e, bg_factor)
    return efpa_independent(V, off, dem, routes)["B_e2e"][:K]


def mc_e2e(V, a_e2e, bg_factor, n_links, seed=MC_SEED) -> dict:
    off, dem, routes = build_streams(n_links, a_e2e, bg_factor)
    all_B, _, all_blk = run_replications_multilink(
        V, off, dem, routes, M=M_REPS, n_arrivals=N_ARRIVALS, base_seed=seed)
    be = all_B[:, :K]
    mean = be.mean(axis=0)
    half = stats.t.ppf(0.975, M_REPS - 1) * be.std(axis=0, ddof=1) / np.sqrt(M_REPS)
    blocked = all_blk[:, :K].sum(axis=0)
    rel_hw = half / np.maximum(mean, 1e-12)
    converged = (blocked >= N_MIN_BLOCKS) | (rel_hw <= REL_HW_MAX)
    return {"mean": mean, "half": half, "blocked": blocked,
            "rel_hw": rel_hw, "converged": converged}


def wmean(x) -> float:
    return float(np.sum(WEIGHT * np.asarray(x)) / np.sum(WEIGHT))


@functools.cache
def load_cm(name: str) -> np.ndarray:
    z = np.load(PROCESSED / "confusion_matrices.npz", allow_pickle=True)
    return row_normalise(np.asarray(z[name], dtype=float))


def main() -> None:
    # suppress governs scientific notation on the printed deviation arrays
    np.set_printoptions(precision=5, suppress=True)
    classes = list(CLASS_ORDER_OTT)
    bgf = CASCADE_BG_FACTOR
    L = CASCADE_LINKS
    save = {
        "class_order": np.array(classes),
        "link_names": np.array(list(CASCADE_LINK_NAMES)),
        "cascade_bg_factor": bgf, "weight": WEIGHT,
        "m_reps": np.int64(M_REPS), "n_arrivals": np.int64(N_ARRIVALS),
    }

    # stage A: inter-link coupling
    print("[A] 3-link tandem, OTT e2e + local OTT-mix background")
    V3 = per_link_capacities(L, bgf, B_TARGET_DEFAULT)
    print(f"  per-link-1% capacities V_l = {V3} ({CASCADE_LINK_NAMES})")
    ef = efpa_e2e(V3, A_OTT, bgf, L)
    ind = indep_e2e(V3, A_OTT, bgf, L)
    mc = mc_e2e(V3, A_OTT, bgf, L)
    print("\n[A] Coupling at the design point (true load)")
    for i, cl in enumerate(classes):
        coup = (ind[i] - ef[i]) / ind[i]
        acc = (ef[i] - mc["mean"][i]) / mc["mean"][i]
        flag = "" if mc["converged"][i] else "  [MC not converged]"
        print(f"  {cl:12s} indep={ind[i]:.4f} EFPA={ef[i]:.4f} "
              f"MC={mc['mean'][i]:.4f}+/-{mc['half'][i]:.4f}  "
              f"coupling={coup:+.1%} EFPAvsMC={acc:+.1%}{flag}")
    # analytical coupling vs link load, no Monte Carlo
    scales = np.array([1.00, 1.05, 1.10, 1.15, 1.20])
    util, ef_mean, ind_mean, coup_rel = [], [], [], []
    for sc in scales:
        e = efpa_e2e(V3, A_OTT * sc, bgf * sc, L)
        n = indep_e2e(V3, A_OTT * sc, bgf * sc, L)
        u = max((E2E_AT * sc + bgf[l] * E2E_AT * sc) / V3[l] for l in range(L))
        util.append(u)
        ef_mean.append(wmean(e))
        ind_mean.append(wmean(n))
        coup_rel.append(float(np.max((n - e) / np.maximum(n, 1e-12))))
    print("\n[A] Coupling vs utilisation (analytical)")
    for i, sc in enumerate(scales):
        print(f"  scale={sc:.2f} maxUtil={util[i]:.2f} "
              f"wB_EFPA={ef_mean[i]:.4f} wB_indep={ind_mean[i]:.4f} "
              f"maxCoupling={coup_rel[i]:.1%}")
    save.update({"V_design": V3.astype(np.int64),
                 "A_efpa": ef, "A_indep": ind,
                 "A_mc_mean": mc["mean"], "A_mc_half": mc["half"],
                 "A_mc_converged": mc["converged"], "A_mc_blocked": mc["blocked"],
                 "A_util": np.array(util), "A_util_efpa": np.array(ef_mean),
                 "A_util_indep": np.array(ind_mean), "A_util_coupling": np.array(coup_rel)})

    # stage B: EFPA accuracy vs decorrelation
    print("\n[B] EFPA accuracy vs background share (per-link-1%, true load)")
    gs = [0.25, 0.5, 1.0, 2.0]
    bshare, berr, bef, bmc, bconv = [], [], [], [], []
    for g in gs:
        f = bgf * g
        Vg = per_link_capacities(L, f, B_TARGET_DEFAULT)
        e = efpa_e2e(Vg, A_OTT, f, L)
        m = mc_e2e(Vg, A_OTT, f, L)
        share = g * float(np.sum(bgf)) / (L + g * float(np.sum(bgf)))
        err = wmean((e - m["mean"]) / np.maximum(m["mean"], 1e-12))
        bshare.append(share); berr.append(err)
        bef.append(wmean(e)); bmc.append(wmean(m["mean"]))
        bconv.append(m["converged"])
        nconv = int(np.sum(~m["converged"]))
        print(f"  g={g:.2f} bgShare={share:.2f}  wEFPA={wmean(e):.4f} "
              f"wMC={wmean(m['mean']):.4f}  EFPAvsMC(weighted)={err:+.1%} "
              f"({nconv} classes not converged)")
    save.update({"B_g": np.array(gs), "B_share": np.array(bshare),
                 "B_err_weighted": np.array(berr),
                 "B_efpa_weighted": np.array(bef), "B_mc_weighted": np.array(bmc),
                 "B_converged": np.array(bconv)})

    # stage C: classifier impact vs L, two conventions
    print("\n[C] Classifier-impact end-to-end deviation vs link count L")
    cms = [("xgb_clean", load_cm("xgb_clean")),
           (f"uniform{int(UNIFORM_R * 100)}", uniform_spillover_cm(K, UNIFORM_R))]
    conventions = [("per_link", lambda n: B_TARGET_DEFAULT),
                   ("e2e_gos", sublink_target)]
    Ls = [1, 2, 3]
    for cname, btfn in conventions:
        for cmname, C in cms:
            a_hat = bridge_equation(C, A_OTT, normalise=True)
            d_mc, d_ef, d_ind, conv = [], [], [], []
            for n in Ls:
                Vn = per_link_capacities(n, bgf, btfn(n))
                rows = {tag: (mc_e2e(Vn, loads, bgf, n), efpa_e2e(Vn, loads, bgf, n),
                              indep_e2e(Vn, loads, bgf, n))
                        for tag, loads in (("t", A_OTT), ("s", a_hat))}
                (t, t_ef, t_ind), (s, s_ef, s_ind) = rows["t"], rows["s"]
                d_mc.append(wmean(s["mean"] - t["mean"]))
                d_ef.append(wmean(s_ef - t_ef))
                d_ind.append(wmean(s_ind - t_ind))
                conv.append(t["converged"] & s["converged"])
            d_mc, d_ef, d_ind = map(np.array, (d_mc, d_ef, d_ind))
            r_mc = d_mc[-1] / d_mc[0] if d_mc[0] else float("nan")
            nbad = int(np.sum(~np.array(conv)))
            print(f"  [{cname:8s} {cmname:9s}] L=1,2,3 weighted dDelta  "
                  f"MC={np.round(d_mc, 4)} EFPA={np.round(d_ef, 4)} "
                  f"INDEP={np.round(d_ind, 4)}  ratio L3/L1 MC={r_mc:.2f}"
                  f"  ({nbad} class-runs not converged)")
            save[f"C_{cname}_{cmname}_mc"] = d_mc
            save[f"C_{cname}_{cmname}_efpa"] = d_ef
            save[f"C_{cname}_{cmname}_indep"] = d_ind
            save[f"C_{cname}_{cmname}_conv"] = np.array(conv)
    save["C_link_counts"] = np.array(Ls)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(PROCESSED / "cascade_results.npz", **save)
    print(f"\nSaved cascade_results.npz  Keys: {len(save)}")


if __name__ == "__main__":
    main()
