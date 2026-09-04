"""Check the ch5 covariance-sign claims against the frozen archives.

Verifies the three quantitative statements of sec:cesnet-auperturb:

1. the semantic six-tier flow-count covariance reproduces the archived cov(a, t) = -6.23;
2. the AU-demand versus measured p99 downstream-rate coupling across the fifteen retained categories is only weakly monotone (Spearman about +0.30, Pearson about +0.67);
3. re-binning the fifteen categories into the six tiers by p99 downstream-rate rank, same tier occupancies with rank order replacing service semantics, flips the covariance positive.

Exit code 1 on any failed check.
"""
import sys
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
PROC = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT))
from src.cesnet.tiers import A_TOTAL, GROUP_REC, TIER_AU, grouping_arrays, tier_vec
from src.analytical.kaufman_roberts import population_covariance

# expected couplings of check 2, quoted in the docstring above
RHO_S_EXPECTED, RHO_P_EXPECTED, RHO_TOL = 0.30, 0.67, 0.02


def main() -> int:
    failures = 0
    zdef = np.load(PROC / "cesnet_definitive.npz", allow_pickle=True)
    zdim = np.load(PROC / "cesnet_dimension.npz", allow_pickle=True)
    names = [str(x) for x in zdef["category_names"]]
    support = np.asarray(zdef["train_support"], float)

    ct, tiers, t_au = grouping_arrays(names, GROUP_REC)
    retained = [c for c in names if GROUP_REC.get(c, -1) >= 0]
    tier_of = {c: GROUP_REC[c] for c in retained}
    sup_of = dict(zip(names, support))

    # 1. Semantic covariance reproduces the archive.
    a_tier = tier_vec(support, ct, tiers)
    a_tier = a_tier / a_tier.sum() * A_TOTAL
    cov_sem = population_covariance(a_tier, t_au)
    cov_arch = float(zdim["rec_cov_count"])
    ok = abs(cov_sem - cov_arch) < 1e-6
    failures += not ok
    print(f"1. semantic cov(a,t) = {cov_sem:+.4f} "
          f"(archive {cov_arch:+.4f}) {'OK' if ok else 'FAIL'}")

    # 2. Weak AU-vs-p99 monotonicity across the retained categories.
    tp = json.load(open(PROC / "cesnet_category_throughput.json"))
    pc = tp["per_category"]
    au_cat = np.array([TIER_AU[tier_of[c]] for c in retained], float)
    p99 = np.array([pc[c]["down_mbps_p99"] for c in retained], float)
    rho_s = float(stats.spearmanr(au_cat, p99).statistic)
    rho_p = float(stats.pearsonr(au_cat, p99).statistic)
    ok = (abs(rho_s - RHO_S_EXPECTED) < RHO_TOL
          and abs(rho_p - RHO_P_EXPECTED) < RHO_TOL)
    failures += not ok
    print(f"2. AU vs p99 coupling: Spearman {rho_s:+.3f}, "
          f"Pearson {rho_p:+.3f} "
          f"(prose {RHO_S_EXPECTED:+.2f} / {RHO_P_EXPECTED:+.2f}) "
          f"{'OK' if ok else 'FAIL'}")

    # 3. Rank re-binning flips the covariance positive.
    occupancy = [sum(1 for c in retained if tier_of[c] == i) for i in tiers]
    order = np.argsort(p99, kind="stable")
    a_rebin = np.zeros(len(tiers))
    pos = 0
    for ti, count in enumerate(occupancy):
        for j in order[pos:pos + count]:
            a_rebin[ti] += sup_of[retained[int(j)]]
        pos += count
    a_rebin = a_rebin / a_rebin.sum() * A_TOTAL
    cov_rebin = population_covariance(a_rebin, t_au)
    ok = cov_rebin > 0
    failures += not ok
    print(f"3. p99-rank re-binned cov(a,t) = {cov_rebin:+.4f} "
          f"(sign flip expected) {'OK' if ok else 'FAIL'}")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} "
          f"({failures} failing checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
