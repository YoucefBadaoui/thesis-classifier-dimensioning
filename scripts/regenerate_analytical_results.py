"""Regenerate analytical_results.npz from the current confusion_matrices.npz.

Headless counterpart of notebooks/03_blocking_deviation.ipynb that writes only the archive, no figures or tables. Run it after retrain_mlp_weighted.py refreshes the MLP slots in confusion_matrices.npz. It also recomputes the +/-10% operating-point perturbation envelope for the OTT scenario from the frozen scenario constants, and prints the two-scenario |S|_l2h / |S|_h2l ratio and the Spearman rank correlation used in the H1 and H3 discussions.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.kaufman_roberts import (
    blocking_deviation,
    bridge_equation,
    capacity_overhead,
    fix_zero_rows,
    kaufman_roberts,
    minimum_recall_search,
    sensitivity_analysis,
    sensitivity_analysis_projected,
)
from src.analytical.scenarios import (
    compute_wbd,
    empirical_5g_results,
    flowpic_vpn_4class_results,
    make_5g_cm,
)
from src.analytical.recall_thresholds import per_class_recall_search
from src.analytical.constants import (
    A_5G,
    A_OTT,
    B_TARGET_DEFAULT,
    CLASS_ORDER_5G,
    CLASS_ORDER_OTT,
    T_5G,
    T_OTT,
    V_SEARCH_MAX,
)

PROCESSED = ROOT / "data" / "processed"
B_TARGET = B_TARGET_DEFAULT


EMPIRICAL_CM_NAMES = [
    "flowpic_nonvpn", "flowpic_vpn", "flowpic_tor",
    "xgb_clean", "mlp_clean",
    "xgb_vpn_shift", "mlp_vpn_shift",
    "xgb_reduced_feat", "mlp_reduced_feat",
]


def l2h_h2l_ratio(S: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    """Mean |S| for low-to-high and high-to-low off-diagonal entries; return ratio."""
    t = np.asarray(t)
    up = t[None, :] > t[:, None]
    dn = t[None, :] < t[:, None]
    A = np.abs(S)
    l2h_mean = float(A[:, up].mean()) if up.any() else 0.0
    h2l_mean = float(A[:, dn].mean()) if dn.any() else 0.0
    return l2h_mean, h2l_mean, l2h_mean / max(h2l_mean, 1e-12)


def _sweep(cm_of_r, recalls, a, t, V_nom):
    """Capacity overhead in percent and WBD across the recall lattice."""
    oh, wbd = [], []
    for r in recalls:
        C = cm_of_r(r)
        a_hat = bridge_equation(C, a)
        try:
            V_p = capacity_overhead(a_hat, t, B_TARGET,
                                    V_start=V_nom, V_max=V_SEARCH_MAX)
            oh.append((V_p - V_nom) / V_nom * 100)
        except ValueError:
            oh.append(float("nan"))
        wbd.append(compute_wbd(C, a, t))
    return np.array(oh), np.array(wbd)


def main() -> None:
    data = np.load(PROCESSED / "confusion_matrices.npz", allow_pickle=True)

    # OTT nominal capacity
    V_nominal = capacity_overhead(A_OTT, T_OTT, B_TARGET, V_start=1, V_max=500)
    _, B_baseline_ott = kaufman_roberts(V_nominal, A_OTT, T_OTT)
    print(f"[OTT] V_nominal = {V_nominal}, B_baseline = {B_baseline_ott}")

    # 5G nominal capacity
    V_5g_nominal = capacity_overhead(A_5G, T_5G, B_TARGET, V_start=1, V_max=500)
    _, B_baseline_5g = kaufman_roberts(V_5g_nominal, A_5G, T_5G)
    print(f"[5G]  V_nominal = {V_5g_nominal}, B_baseline = {B_baseline_5g}")

    # Delta B and the per-CM capacity overhead in one pass over the matrices
    all_delta_B, per_cm = {}, {}
    _, B_true_ott = kaufman_roberts(V_nominal, A_OTT, T_OTT)
    for name in EMPIRICAL_CM_NAMES:
        C = fix_zero_rows(data[name])
        # the FlowPic VPN matrix has an all-zero Browsing row: the VPN partition holds no Browsing flows. Evaluated as published, so that load stays lost and the distorted-load sum is 63, not 88.
        if name == "flowpic_vpn":
            # C.T @ a directly, since the zero row cannot pass the row-stochasticity guard in bridge_equation
            a_hat_vpn = data[name].T @ A_OTT
            _, B_dist_vpn = kaufman_roberts(V_nominal, a_hat_vpn, T_OTT)
            all_delta_B[name] = B_dist_vpn - B_true_ott
        else:
            all_delta_B[name] = blocking_deviation(
                V_nominal, A_OTT, C, T_OTT)["delta_B"]
        # the overhead column uses the zero-row-fixed matrix for every name, so the VPN row carries its repaired self-classification here
        try:
            V_p = capacity_overhead(bridge_equation(C, A_OTT), T_OTT, B_TARGET,
                                    V_start=V_nominal, V_max=1000)
            oh = (V_p - V_nominal) / V_nominal * 100
        except ValueError:
            oh = float("inf")
        # balanced accuracy over rows carrying mass; fix_zero_rows has already repaired an all-zero row, so the mask only bites on a numerically empty row that survived it
        active = C.sum(axis=1) > 1e-10
        per_cm[name] = (oh, compute_wbd(C, A_OTT, T_OTT),
                        float(np.mean(np.diag(C)[active])) if active.sum() else 0.0)

    # sensitivity tensors, diagonal to off-diagonal swap
    S_xgb = sensitivity_analysis(V_nominal, A_OTT, fix_zero_rows(data["xgb_clean"]),
                                 T_OTT)
    S_tor = sensitivity_analysis(V_nominal, A_OTT, fix_zero_rows(data["flowpic_tor"]),
                                 T_OTT)
    # 5G uniform at r=0.90
    C_5g_u = make_5g_cm(0.90, "uniform")
    S_5g = sensitivity_analysis(V_5g_nominal, A_5G, C_5g_u, T_5G)

    # projected-gradient sensitivity on the simplex tangent
    proj_xgb = sensitivity_analysis_projected(
        V_nominal, A_OTT, fix_zero_rows(data["xgb_clean"]), T_OTT)
    proj_tor = sensitivity_analysis_projected(
        V_nominal, A_OTT, fix_zero_rows(data["flowpic_tor"]), T_OTT)
    proj_5g = sensitivity_analysis_projected(
        V_5g_nominal, A_5G, C_5g_u, T_5G)
    print("\nProjected-gradient sensitivity (max-row L2, worst-row capacity-planning):")
    for label, proj in (("OTT XGB clean", proj_xgb), ("OTT FlowPic Tor", proj_tor),
                        ("5G  uniform r=0.90", proj_5g)):
        print(f"  {label:18s}: {proj['max_row_l2']:.6e}")
    print(f"  Cross-scenario ratio OTT(XGB)/5G = "
          f"{proj_xgb['max_row_l2']/proj_5g['max_row_l2']:.3f}x")
    print(f"  Cross-scenario ratio OTT(Tor)/5G = "
          f"{proj_tor['max_row_l2']/proj_5g['max_row_l2']:.3f}x")

    # l2h/h2l ratios
    for label, S, t_s in (("OTT XGB clean", S_xgb, T_OTT),
                          ("OTT FlowPic Tor", S_tor, T_OTT),
                          ("5G  uniform r=0.90", S_5g, T_5G)):
        l2h, h2l, ratio = l2h_h2l_ratio(S, t_s)
        print(f"{label:18s}: |S|_l2h/|S|_h2l = {ratio:.3f}x  "
              f"(l2h={l2h:.6f}, h2l={h2l:.6f})")

    # r_k* tables
    epsilons = [0.05, 0.10, 0.15]
    rstar_ott = {eps: per_class_recall_search(A_OTT, T_OTT, V_nominal, B_TARGET, eps)
                 for eps in epsilons}
    rstar_5g = {eps: per_class_recall_search(A_5G, T_5G, V_5g_nominal, B_TARGET, eps)
                for eps in epsilons}

    at_ott = A_OTT * T_OTT
    at_5g = A_5G * T_5G
    print("\nH3 Spearman correlations (descriptive, n small):")
    for eps in epsilons:
        rho_o, _ = stats.spearmanr(at_ott, rstar_ott[eps])
        rho_5, _ = stats.spearmanr(at_5g, rstar_5g[eps])
        print(f"  eps={int(eps*100)}%: OTT rho={rho_o:+.4f},  5G rho={rho_5:+.4f}")

    # +/-10% operating-point perturbation envelope (OTT). The per-class sweep runs at the perturbed loads and their recomputed capacity; the uniform-recall sweep holds V at the unperturbed nominal, so it isolates load sensitivity of the epsilon budget and reaches r* = 1.0 at +10%. Both sweep a 0.01 recall lattice, 51 steps over [0.50, 1.00], and thresholds are stored rounded to that resolution.
    PERTURBATION = 0.10
    EPS_SYSTEM = 0.05
    R_STEPS_ENVELOPE = 51

    a_plus = A_OTT * (1.0 + PERTURBATION)
    a_minus = A_OTT * (1.0 - PERTURBATION)

    V_plus10 = capacity_overhead(a_plus, T_OTT, B_TARGET,
                                 V_start=1, V_max=V_SEARCH_MAX)
    V_minus10 = capacity_overhead(a_minus, T_OTT, B_TARGET,
                                  V_start=1, V_max=V_SEARCH_MAX)
    print(f"\n[OTT] +-10% envelope: V(+10% loads) = {V_plus10}, "
          f"V(-10% loads) = {V_minus10}")

    rstar_eps5_plus10 = per_class_recall_search(
        a_plus, T_OTT, V_plus10, B_TARGET, EPS_SYSTEM,
        r_steps=R_STEPS_ENVELOPE)
    rstar_eps5_minus10 = per_class_recall_search(
        a_minus, T_OTT, V_minus10, B_TARGET, EPS_SYSTEM,
        r_steps=R_STEPS_ENVELOPE)

    def _q2(x: float) -> np.float64:
        return np.float64(round(float(x), 2))

    envelope: dict[str, np.ndarray] = {
        "V_nominal_ott_plus10": np.int64(V_plus10),
        "V_nominal_ott_minus10": np.int64(V_minus10),
        "rstar_ott_eps5_plus10": np.round(rstar_eps5_plus10, 2),
        "rstar_ott_eps5_minus10": np.round(rstar_eps5_minus10, 2),
    }
    # the uniform-recall sweeps hold V at the unperturbed nominal
    for tag, a_env in (("nominal", A_OTT), ("plus10_fixedV", a_plus),
                       ("minus10_fixedV", a_minus)):
        envelope[f"rstar_uniform_ott_eps5_{tag}"] = _q2(minimum_recall_search(
            a_env, T_OTT, V_nominal, B_TARGET,
            epsilon=EPS_SYSTEM, r_steps=R_STEPS_ENVELOPE))
    print(f"[OTT] +-10% envelope: uniform r* nominal="
          f"{envelope['rstar_uniform_ott_eps5_nominal']}, "
          f"minus10_fixedV={envelope['rstar_uniform_ott_eps5_minus10_fixedV']}, "
          f"plus10_fixedV={envelope['rstar_uniform_ott_eps5_plus10_fixedV']}")

    # Recall sweeps for capacity overhead + WBD
    recalls = np.arange(0.50, 1.00, 0.01)
    SPILLS = ("uniform", "worst", "best")
    ott = {p: _sweep(lambda r, p=p: fix_zero_rows(data[f"{p}_r{r:.2f}"]),
                     recalls, A_OTT, T_OTT, V_nominal) for p in SPILLS}
    g5 = {p: _sweep(lambda r, p=p: make_5g_cm(r, p),
                    recalls, A_5G, T_5G, V_5g_nominal) for p in SPILLS}

    save_dict: dict[str, np.ndarray] = {
        "V_nominal_ott": np.int64(V_nominal),
        "V_nominal_5g": np.int64(V_5g_nominal),
        "a_ott": A_OTT,
        "t_ott": T_OTT,
        "a_5g": A_5G,
        "t_5g": T_5G,
        "class_order_ott": np.array(list(CLASS_ORDER_OTT)),
        "class_order_5g": np.array(list(CLASS_ORDER_5G)),
        "B_baseline_ott": B_baseline_ott,
        "B_baseline_5g": B_baseline_5g,
        "empirical_cm_names": np.array(EMPIRICAL_CM_NAMES),
        "S_xgb_clean": S_xgb,
        "S_flowpic_tor": S_tor,
        "S_5g_uniform": S_5g,
        "S_proj_xgb_clean": proj_xgb["S_proj"],
        "S_proj_flowpic_tor": proj_tor["S_proj"],
        "S_proj_5g_uniform": proj_5g["S_proj"],
        "S_sys_proj_xgb_clean": proj_xgb["S_sys_proj"],
        "S_sys_proj_flowpic_tor": proj_tor["S_sys_proj"],
        "S_sys_proj_5g_uniform": proj_5g["S_sys_proj"],
        "sens_scalar_maxrow_xgb_clean": np.float64(proj_xgb["max_row_l2"]),
        "sens_scalar_maxrow_flowpic_tor": np.float64(proj_tor["max_row_l2"]),
        "sens_scalar_maxrow_5g_uniform": np.float64(proj_5g["max_row_l2"]),
        "sens_scalar_meanrow_xgb_clean": np.float64(proj_xgb["mean_row_l2"]),
        "sens_scalar_meanrow_flowpic_tor": np.float64(proj_tor["mean_row_l2"]),
        "sens_scalar_meanrow_5g_uniform": np.float64(proj_5g["mean_row_l2"]),
        "sens_scalar_frob_xgb_clean": np.float64(proj_xgb["frobenius"]),
        "sens_scalar_frob_flowpic_tor": np.float64(proj_tor["frobenius"]),
        "sens_scalar_frob_5g_uniform": np.float64(proj_5g["frobenius"]),
        "recalls_sweep": recalls,
        "overhead_uniform": ott["uniform"][0],
        "overhead_worst": ott["worst"][0],
        "overhead_best": ott["best"][0],
        "wbd_uniform": ott["uniform"][1],
        "wbd_worst": ott["worst"][1],
        "wbd_best": ott["best"][1],
        "recalls_5g_sweep": recalls,
        "oh_5g_uniform": g5["uniform"][0],
        "oh_5g_worst": g5["worst"][0],
        "oh_5g_best": g5["best"][0],
        "wbd_5g_uniform": g5["uniform"][1],
        "wbd_5g_worst": g5["worst"][1],
        "wbd_5g_best": g5["best"][1],
    }
    for name in EMPIRICAL_CM_NAMES:
        save_dict[f"delta_B_{name}"] = all_delta_B[name]
    for eps in epsilons:
        eps_key = str(int(eps * 100))
        save_dict[f"rstar_ott_eps{eps_key}"] = rstar_ott[eps]
        save_dict[f"rstar_5g_eps{eps_key}"] = rstar_5g[eps]

    # FP-VPN 4-class subspace (Browsing dropped per Shapira 2021, Table I). See src/analytical/scenarios.py for the perm = [3, 2, 1, 0] construction and the explicit Streaming<->Video label remap.
    r4 = flowpic_vpn_4class_results()
    for k, v in r4.items():
        save_dict[f"fp_vpn_4x4_{k}"] = v

    # 5G empirical CMs (3 Malkoc + 9 Islam, all 3-class eMBB/mMTC/URLLC).
    r5g = empirical_5g_results()
    save_dict["empirical_5g_cm_names"] = np.array(list(r5g.keys()))
    for name, entry in r5g.items():
        for k, v in entry.items():
            save_dict[f"5g_{name}_{k}"] = v

    # the envelope depends only on the frozen scenario constants (A_OTT, T_OTT, B_TARGET_DEFAULT), so drift against the existing archive means constants.py moved; the archive is then left untouched
    ar_path = PROCESSED / "analytical_results.npz"
    if ar_path.exists():
        prev = np.load(ar_path)
        drifted = []
        for key, fresh in envelope.items():
            if key not in prev.files or not np.array_equal(fresh, prev[key]):
                drifted.append(key)
        if drifted:
            print("\nENVELOPE MISMATCH against existing analytical_results.npz:")
            for key in drifted:
                old_val = prev[key] if key in prev.files else "<absent>"
                print(f"  {key}: fresh={envelope[key]}  archived={old_val}")
            print("Archive left untouched; resolve the scenario-constant "
                  "drift before regenerating.")
            sys.exit(1)

    save_dict.update(envelope)

    np.savez(PROCESSED / "analytical_results.npz", **save_dict)
    print(f"\nSaved: {PROCESSED / 'analytical_results.npz'}")
    print(f"Keys: {len(save_dict)}")

    r80 = int(np.argmin(np.abs(recalls - 0.80)))
    print()
    print("Ch.5 table values")
    print(f"OTT overhead @ r=0.80: uniform={ott['uniform'][0][r80]:+.2f}%,  "
          f"best={ott['best'][0][r80]:+.2f}%,  worst={ott['worst'][0][r80]:+.2f}%")
    print(f"OTT WBD      @ r=0.80: uniform={ott['uniform'][1][r80]:.2f},  "
          f"best={ott['best'][1][r80]:.2f},  worst={ott['worst'][1][r80]:.2f}  (AU-erlangs)")
    print(f"5G  overhead @ r=0.80: uniform={g5['uniform'][0][r80]:+.2f}%,  "
          f"best={g5['best'][0][r80]:+.2f}%,  worst={g5['worst'][0][r80]:+.2f}%")
    print(f"5G  WBD      @ r=0.80: uniform={g5['uniform'][1][r80]:.2f},  "
          f"best={g5['best'][1][r80]:.2f},  worst={g5['worst'][1][r80]:.2f}  (AU-erlangs)")
    print()
    print("Delta B per CM (OTT, max abs value):")
    for name in EMPIRICAL_CM_NAMES:
        v = all_delta_B[name]
        sum_sign = "+" if np.sum(v) > 0 else "-"
        print(f"  {name:22s}: max|dB|={np.max(np.abs(v)):.6f}  sum_sign={sum_sign}")
    print()
    print("Per-CM capacity overhead (search in OTT):")
    for name in EMPIRICAL_CM_NAMES:
        oh, wbd, bacc = per_cm[name]
        print(f"  {name:22s}: overhead={oh:+.2f}%  WBD={wbd:.2f}  bacc={bacc:.1%}")


if __name__ == "__main__":
    main()
