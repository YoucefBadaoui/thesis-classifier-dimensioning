#!/usr/bin/env python3
"""Statistical power of the r*_k rank correlations at larger class counts.

Repeats the isolated-class-k uniform-spillover r*_k protocol on random synthetic scenarios for K = 5 to 15 and tests two predictors of r*_k:

  H3: a_k * t_k            (load-demand product)
  F1: max_j(t_j) - t_k     (upward bandwidth gap)

The thesis scenarios run at n = 5 and n = 3. At n = 5 only a perfect ordering reaches significance, since the smallest two-sided exact p-value is 2/5! = 0.017. At n = 3 no ordering can reach significance, since the floor is 2/3! = 0.333. All p-values are two-sided permutation p-values, exact enumeration for n <= 7 and Monte Carlo otherwise.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.analytical.kaufman_roberts import capacity_overhead
from src.analytical.recall_thresholds import (
    predictors,
    rstar_per_class,
    spearman_perm,
)
from src.analytical.constants import (
    A_OTT, T_OTT, A_5G, T_5G, B_TARGET_DEFAULT, V_NOMINAL_5G, V_NOMINAL_OTT,
)

PROCESSED = ROOT / "data" / "processed"
B_TARGET = B_TARGET_DEFAULT

SEED = 20260529
ALPHA = 0.05
EPS_POWER = 0.01           # tolerance at which r*_k stays informative for all K
EPS_THESIS = 0.05          # thesis headline tolerance, used to show high-K degeneracy
EPS_ROBUST = [0.01, 0.02, 0.03, 0.05]
K_GRID = [5, 7, 9, 11, 13, 15]
K_REPRESENTATIVE = 12      # canonical scenario and eps-robustness class count
M_SCENARIOS = 300          # random scenarios per K
A_LOW, A_HIGH = 3.0, 25.0  # offered-load draw range (Erl); keeps max(a*t) < ~1000
T_MIN, T_MAX = 1, 15       # demand draw range (AU); integer

SWEEP_COLS = ("frac_sig_h3", "frac_sig_f1", "med_rho_h3", "med_rho_f1",
              "frac_neg_h3", "frac_pos_f1")
ROBUST_COLS = ("frac_degen", "frac_pos_f1", "frac_sig_f1", "med_rho_f1")


def draw_scenario(rng, K):
    """One random (a, t, V_nominal) scenario, or None when the draw is rejected.

    Rejects a flat demand vector, for which the F1 gap predictor is constant, and a capacity search that does not terminate below V_max.
    """
    t = rng.integers(T_MIN, T_MAX + 1, size=K).astype(float)
    a = rng.uniform(A_LOW, A_HIGH, size=K)
    if t.max() == t.min():
        return None
    try:
        V = capacity_overhead(a, t, B_TARGET, V_start=1, V_max=8000)
    except ValueError:
        # unreachable for the stated draw range (max a*t about 375 Erl at 15 AU), kept so the range can be widened without a silent crash
        return None
    return a, t, V


def sanity_check():
    """Gate: the protocol reproduces the OTT and 5G thesis values before the sweep runs."""
    expected = {
        "OTT": (V_NOMINAL_OTT, [0.817, 0.754, 0.50, 0.50, 0.817], -0.63),
        "5G": (V_NOMINAL_5G, [0.50, 0.773, 0.742], -1.00),
    }
    # tolerances are rounding margins at the precision the expected values above are quoted to: r* to three decimals, rho to two
    R_TOL, RHO_TOL = 0.005, 0.02
    print("SANITY CHECK against the thesis values")
    for name, a, t in [("OTT", np.asarray(A_OTT, float), np.asarray(T_OTT, float)),
                       ("5G", np.asarray(A_5G, float), np.asarray(T_5G, float))]:
        V_nom = capacity_overhead(a, t, B_TARGET, V_start=1, V_max=600)
        rstar = rstar_per_class(a, t, V_nom, EPS_THESIS)
        at, gap = predictors(a, t)
        rho_h3, _ = stats.spearmanr(at, rstar)
        V_exp, r_exp, rho_exp = expected[name]
        ok = (V_nom == V_exp and np.all(np.abs(rstar - np.array(r_exp)) <= R_TOL)
              and abs(rho_h3 - rho_exp) <= RHO_TOL)
        print(f"[{name}] V_nominal={V_nom}  r*_k(eps=5%)={np.round(rstar,4).tolist()}  "
              f"H3 rho={rho_h3:+.4f}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(
                f"sanity check failed for {name}: expected V={V_exp}, r*={r_exp}, "
                f"rho={rho_exp}; got V={V_nom}, r*={np.round(rstar, 4).tolist()}, "
                f"rho={rho_h3:+.4f}")
    print()


def run_power_sweep():
    rng = np.random.default_rng(SEED)
    prng = np.random.default_rng(SEED + 10)   # permutation draws, kept apart from the scenario draws
    cols = {k: np.zeros(len(K_GRID)) for k in SWEEP_COLS}
    n_valid = np.zeros(len(K_GRID), dtype=int)

    print(f"MONTE CARLO POWER SWEEP  (M={M_SCENARIOS}/K, eps={EPS_POWER*100:.0f}%, "
          f"alpha={ALPHA}, seed={SEED})")
    for ki, K in enumerate(K_GRID):
        stats_rows = []
        for _ in range(M_SCENARIOS):
            drawn = draw_scenario(rng, K)
            if drawn is None:
                continue
            a, t, V_nom = drawn
            rstar = rstar_per_class(a, t, V_nom, EPS_POWER)
            if np.ptp(rstar) == 0:
                continue  # degenerate: every class floored, correlation undefined
            at, gap = predictors(a, t)
            rho_h3, p_h3 = spearman_perm(at, rstar, prng)
            rho_f1, p_f1 = spearman_perm(gap, rstar, prng)
            if np.isnan(rho_h3) or np.isnan(rho_f1):
                continue  # constant predictor, excluded by the flat-demand guard
            stats_rows.append((rho_h3, p_h3, rho_f1, p_f1))
        arr = np.array(stats_rows).reshape(-1, 4)
        rho_h3_arr, p_h3_arr, rho_f1_arr, p_f1_arr = arr.T
        nv = len(arr)
        n_valid[ki] = nv
        for name, v in (("frac_sig_h3", np.mean(p_h3_arr < ALPHA) if nv else float("nan")),
                        ("frac_sig_f1", np.mean(p_f1_arr < ALPHA) if nv else float("nan")),
                        ("med_rho_h3", np.median(rho_h3_arr) if nv else float("nan")),
                        ("med_rho_f1", np.median(rho_f1_arr) if nv else float("nan")),
                        ("frac_neg_h3", np.mean(rho_h3_arr < 0) if nv else float("nan")),
                        ("frac_pos_f1", np.mean(rho_f1_arr > 0) if nv else float("nan"))):
            cols[name][ki] = v
        print(f"K={K:2d}  n_valid={nv:4d}  "
              f"H3: med_rho={cols['med_rho_h3'][ki]:+.3f} frac_neg={cols['frac_neg_h3'][ki]:.2f} "
              f"frac_sig={cols['frac_sig_h3'][ki]:.2f} | "
              f"F1: med_rho={cols['med_rho_f1'][ki]:+.3f} frac_pos={cols['frac_pos_f1'][ki]:.2f} "
              f"frac_sig={cols['frac_sig_f1'][ki]:.2f}")
    return {"K_grid": np.array(K_GRID), "n_valid": n_valid, **cols}


def run_canonical():
    """One reproducible K=12 scenario, the first seeded draw, reported in full with permutation significance for both predictors."""
    rng = np.random.default_rng(SEED + 1)
    K = K_REPRESENTATIVE
    t = rng.integers(T_MIN, T_MAX + 1, size=K).astype(float)
    a = rng.uniform(A_LOW, A_HIGH, size=K)
    V_nom = capacity_overhead(a, t, B_TARGET, V_start=1, V_max=8000)
    rstar = rstar_per_class(a, t, V_nom, EPS_POWER)
    at, gap = predictors(a, t)
    # the two predictors get independent permutation streams
    prng_h3 = np.random.default_rng(SEED + 2)
    prng_f1 = np.random.default_rng(SEED + 3)
    rho_h3, p_h3 = spearman_perm(at, rstar, prng_h3, n_perm=19999)
    rho_f1, p_f1 = spearman_perm(gap, rstar, prng_f1, n_perm=19999)
    print(f"REPRESENTATIVE SCENARIO  K={K}, V_nominal={V_nom}, eps={EPS_POWER*100:.0f}%")
    order = np.argsort(t)
    for label, vec, nd in ((" t_k", t, 0), (" a_k", a, 1), (" a*t", a * t, 1),
                           (" gap", gap, 0), (" r*_k", rstar, 4)):
        v = np.round(vec[order], nd)
        print(f"{label} :", (v.astype(int) if nd == 0 else v).tolist())
    print(f" H3  Spearman(a*t, r*)  rho={rho_h3:+.4f}  perm-p={p_h3:.4f}  (n={K})")
    print(f" F1  Spearman(gap, r*)  rho={rho_f1:+.4f}  perm-p={p_f1:.4f}  (n={K})")
    return dict(
        canon_K=np.int64(K), canon_V=np.int64(V_nom),
        canon_a=a, canon_t=t, canon_rstar=rstar, canon_at=at, canon_gap=gap,
        canon_rho_h3=np.float64(rho_h3), canon_p_h3=np.float64(p_h3),
        canon_rho_f1=np.float64(rho_f1), canon_p_f1=np.float64(p_f1),
    )


def run_eps_robustness():
    """Vary the overhead tolerance epsilon at fixed K.

    Reports how often the isolated-class-k r*_k stays non-degenerate and how the F1 bandwidth-gap correlation behaves: the F1 sign holds across tolerances while the headline eps=5% metric degenerates at high K.
    """
    K = K_REPRESENTATIVE
    rng = np.random.default_rng(SEED + 100)
    prng = np.random.default_rng(SEED + 110)
    cols = {k: np.zeros(len(EPS_ROBUST)) for k in ROBUST_COLS}
    print(f"EPS ROBUSTNESS / DEGENERACY at fixed K={K}  (M={M_SCENARIOS}, seed={SEED+100})")
    # pre-draw scenarios so every epsilon sees the same set
    scenarios = []
    while len(scenarios) < M_SCENARIOS:
        drawn = draw_scenario(rng, K)
        if drawn is not None:
            scenarios.append(drawn)
    for ei, eps in enumerate(EPS_ROBUST):
        degen = 0
        rho_list, p_list = [], []
        for a, t, V in scenarios:
            rstar = rstar_per_class(a, t, V, eps)
            if np.ptp(rstar) == 0:
                degen += 1
                continue
            _, gap = predictors(a, t)
            rho, p = spearman_perm(gap, rstar, prng)
            if np.isnan(rho):
                continue  # constant predictor, not a floored r*; same rule as the sweep
            rho_list.append(rho); p_list.append(p)
        nv = len(rho_list)
        for name, v in (("frac_degen", degen / len(scenarios)),
                        ("frac_pos_f1", np.mean(np.array(rho_list) > 0) if nv else float("nan")),
                        ("frac_sig_f1", np.mean(np.array(p_list) < ALPHA) if nv else float("nan")),
                        ("med_rho_f1", np.median(rho_list) if nv else float("nan"))):
            cols[name][ei] = v
        print(f"eps={eps*100:4.0f}%  degenerate={cols['frac_degen'][ei]:.2f}  "
              f"F1 med_rho={cols['med_rho_f1'][ei]:+.3f}  frac_pos={cols['frac_pos_f1'][ei]:.2f}  "
              f"frac_sig={cols['frac_sig_f1'][ei]:.2f}  (n_valid={nv})")
    return {"eps_grid": np.array(EPS_ROBUST), **cols, "robust_K": np.int64(K)}


def main():
    sanity_check()
    sweep = run_power_sweep()
    robust = run_eps_robustness()
    canon = run_canonical()
    save = {f"sweep_{k}": v for k, v in sweep.items()}
    save.update({f"robust_{k}": v for k, v in robust.items()})
    save.update(canon)
    save["seed"] = np.int64(SEED)
    save["alpha"] = np.float64(ALPHA)
    save["eps_power"] = np.float64(EPS_POWER)
    save["eps_thesis"] = np.float64(EPS_THESIS)
    save["m_scenarios"] = np.int64(M_SCENARIOS)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(PROCESSED / "highk_power.npz", **save)
    print()
    print(f"Saved: {PROCESSED / 'highk_power.npz'}  ({len(save)} keys)")


if __name__ == "__main__":
    main()
