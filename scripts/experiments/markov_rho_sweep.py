"""Correlation sweep under a two-state Markov error process.

A per-class two-state chain is advanced on each arrival of that class. It preserves the marginal confusion matrix exactly and gives an ergodic error process with lag-1 autocorrelation rho and geometric error clusters. The companion sweep in scripts/experiments/shock_rho_sweep.py instead draws a common shock once per class and replication, which has no finite correlation time. Alongside the blocking estimates this records the measured marginal error rate, lag-1 autocorrelation and mean cluster length, so the realised error structure is reported rather than assumed.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))

from src.monte_carlo.markov_error import (
    mean_cluster_length,
    run_replications_markov,
)
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

        allB, allA, allBl, diagsets = [], [], [], []
        for rho in RHOS:
            B, arr, blk, diags = run_replications_markov(
                V, a_true, C, rho, t, M=a.M, n_arrivals=a.arrivals,
                n_workers=a.workers)
            allB.append(B); allA.append(arr); allBl.append(blk)
            # pool the per-replication diagnostics by summing counts, so the statistics cover the whole ensemble
            tc = np.sum([d["true_count"] for d in diags], axis=0)
            ec = np.sum([d["err_count"] for d in diags], axis=0)
            rc = np.sum([d["run_count"] for d in diags], axis=0)
            rho_meas = np.mean([d["rho_measured"] for d in diags], axis=0)
            diagsets.append(dict(
                p_measured=ec / np.maximum(tc, 1),
                rho_measured=rho_meas,
                cluster_measured=ec / np.maximum(rc, 1),
                cluster_target=mean_cluster_length(p_err, rho),
            ))
            d = diagsets[-1]
            print(f"  rho={rho}: B_mean={np.array2string(B.mean(0), precision=6)}\n"
                  f"           p_meas   {np.round(d['p_measured'], 4)}\n"
                  f"           rho_meas {np.round(d['rho_measured'], 4)}\n"
                  f"           cluster  {np.round(d['cluster_measured'], 3)} "
                  f"(target {np.round(d['cluster_target'], 3)})")

        allB = np.array(allB); allA = np.array(allA); allBl = np.array(allBl)
        M, B_mean, bp_std, ci_half = summarise(allB)
        print_drift_table(spec["order"], B_mean)

        out = a.out / f"markov_rho_sweep_{name}.npz"
        np.savez(
            out,
            **base_archive(name, spec, a, allB, allA, allBl, B_mean, bp_std, ci_half,
                           a_true, t, C, a_hat, B_an),
            p_err=p_err,
            p_measured=np.array([d["p_measured"] for d in diagsets]),
            rho_measured=np.array([d["rho_measured"] for d in diagsets]),
            cluster_measured=np.array([d["cluster_measured"] for d in diagsets]),
            cluster_target=np.array([d["cluster_target"] for d in diagsets]),
            design=np.array("markov_two_state"),
            notes=np.array(
                "Two-state Markov-modulated error process, one chain per true "
                "class advanced on each arrival of that class. Stationary "
                "marginal P(error)=1-C_ii for every rho, so the marginal "
                "confusion matrix is preserved exactly; lag-1 autocorrelation "
                "equals rho and the autocorrelation function is rho^k, giving a "
                "finite correlation time. Error clusters are geometric with mean "
                "1/((1-p)(1-rho)) class arrivals. Complements the common-shock "
                "design of monte_carlo_rho_sweep_*.npz, in which Z was drawn "
                "once per (class, replication) and the ensemble was therefore a "
                "random-effect mixture over shock regimes rather than one "
                "ergodic process. ci_half uses Student-t(0.975, M-1)."),
        )
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
