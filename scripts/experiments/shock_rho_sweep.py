"""Correlation sweep under the exchangeable common-shock error process.

The shock Z_i is drawn once per class and replication, so within a replication the error indicators are exchangeable with no finite correlation time, and the replication ensemble is a random-effect mixture over shock regimes rather than one ergodic process. Chapter 6 reports this beside the ergodic reading that scripts/experiments/markov_rho_sweep.py drives. Seeds run 1 to M and the per-replication arrival budget matches the archives, so a rerun reproduces them.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.monte_carlo.rho_sweep import run_replications_with_rho
from src.monte_carlo.sweep_common import (
    RHOS,
    base_archive,
    load_scenario,
    parse_args,
    print_drift_table,
    summarise,
)

# line-buffered so a piped log follows a long run
sys.stdout.reconfigure(line_buffering=True)


def main():
    # argparse description is the docstring's first line, so keep it a one-liner
    a = parse_args(__doc__.splitlines()[0])
    for name in a.scenarios.split(","):
        spec, a_true, t, C, a_hat, B_an, p_err = load_scenario(name)
        V = spec["V"]
        print(f"\n===== {name}: V={V} cm={spec['cm_key']} K={len(a_true)} "
              f"M={a.M} arrivals={a.arrivals:,} =====")
        print(f"  p_err = {np.round(p_err, 4)}")

        allB, allA, allBl = [], [], []
        for rho in RHOS:
            B, arr, blk = run_replications_with_rho(
                V, a_true, C, rho, t, M=a.M, n_arrivals=a.arrivals,
                n_workers=a.workers)
            allB.append(B); allA.append(arr); allBl.append(blk)
            print(f"  rho={rho}: B_mean={np.array2string(B.mean(0), precision=6)}")

        allB = np.array(allB); allA = np.array(allA); allBl = np.array(allBl)
        M, B_mean, bp_std, ci_half = summarise(allB)
        print_drift_table(spec["order"], B_mean)

        out = a.out / f"monte_carlo_rho_sweep_M{M}_{name}.npz"
        np.savez(
            out,
            **base_archive(name, spec, a, allB, allA, allBl, B_mean, bp_std, ci_half,
                           a_true, t, C, a_hat, B_an),
            design=np.array("per_rep_independent_z"),
            notes=np.array(
                "Exchangeable common-shock design. Z_i drawn once per (class, "
                "replication); per-arrival activator I and per-flow draw U drawn "
                "fresh inside the kernel, with w = sqrt(rho) so the realised "
                "pairwise correlation of the error indicators equals rho. The "
                "replication ensemble is a random-effect mixture over the two "
                "shock regimes and has no finite correlation time. Seeds 1..M. "
                "ci_half uses Student-t(0.975, M-1)."),
        )
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
