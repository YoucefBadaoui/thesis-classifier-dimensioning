"""Correlated-error designs compared: common shock versus two-state Markov.

``06_error_process_designs`` for the Assumption A3 correlation sweep. Panel (a) is mean per-class blocking drift against the rho = 0 reference under both designs, OTT/IPTV. Panel (b) is the measured mean error-run length: the common shock has no finite error-run length, the Markov design clusters geometrically.

Inputs in ``data/processed``: ``monte_carlo_rho_sweep_M300_ott.npz`` (common shock) and ``markov_rho_sweep_ott.npz`` (Markov).
"""

import matplotlib.pyplot as plt
import numpy as np

from src.figures import style

def _drift(B_mean: np.ndarray) -> np.ndarray:
    base = B_mean[0]
    return (B_mean - base) / base * 100.0


def plot_error_process_designs() -> plt.Figure:
    shock = np.load(style.PROCESSED / "monte_carlo_rho_sweep_M300_ott.npz",
                    allow_pickle=True)
    mark = np.load(style.PROCESSED / "markov_rho_sweep_ott.npz",
                   allow_pickle=True)

    classes = [str(c) for c in shock["class_order"]]
    assert [str(c) for c in mark["class_order"]] == classes, "class order mismatch"
    rhos = np.asarray(shock["rho_values"], float)
    assert np.allclose(rhos, np.asarray(mark["rho_values"], float)), "rho grid mismatch"

    d_shock = _drift(np.asarray(shock["B_mean"], float))
    d_mark = _drift(np.asarray(mark["B_mean"], float))

    fig, (ax_d, ax_c) = plt.subplots(
        1, 2, figsize=style.size("wide", style.width_in("wide") / 1.85))

    x = np.arange(len(classes), dtype=float)
    w = 0.20
    series = [
        (arr[ri], col, alpha, f"{name}, $\\rho={rhos[ri]:g}$")
        for arr, name, col in ((d_shock, "common shock", style.PALETTE[1]),
                               (d_mark, "Markov", style.PALETTE[0]))
        for ri, alpha in ((1, 0.55), (2, 1.00))
    ]
    for i, (vals, col, alpha, lab) in enumerate(series):
        ax_d.bar(x + (i - 1.5) * w, vals, w, color=col, alpha=alpha, label=lab,
                 edgecolor="none")
    ax_d.axhline(0.0, color=style.PALETTE[3], lw=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([style.class_short(c) for c in classes],
                         fontsize=style.FS_ANNOT)
    ax_d.set_ylabel(f"drift {style.VERSUS} $\\rho=0$ (%)", fontsize=style.FS_ANNOT)
    ax_d.set_title("(a) drift by error-process design")
    top = max(d_shock[2].max(), d_mark[2].max())
    ax_d.set_ylim(0.0, top * 1.08)
    ax_d.set_yticks(np.arange(0.0, top + 20.0, 20.0))

    cl = np.asarray(mark["cluster_measured"], float)
    for ri, rho in enumerate(rhos):
        ax_c.plot(x, cl[ri], marker="o", ms=4, lw=1.2,
                  color=style.PALETTE[[3, 0, 1][ri]],
                  label=f"$\\rho={rho:g}$")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([style.class_short(c) for c in classes])
    ax_c.set_ylabel("mean error-run length (arrivals)", fontsize=style.FS_ANNOT)
    ax_c.set_title("(b) Markov error-run length")
    # Headroom so the legend clears the rho = 0.6 curve.
    ax_c.set_ylim(top=cl.max() * 1.34)
    ax_c.legend(fontsize=style.FS_ANNOT, frameon=False, loc="upper left", ncol=3,
                columnspacing=1.0, handlelength=1.4)

    # the four drift labels are wider than panel (a), so the legend sits above both panels at figure level
    handles, labels = ax_d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=style.FS_LEGEND,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    return fig


def main() -> None:
    style.apply()
    fig = plot_error_process_designs()
    style.savefig(fig, "06_error_process_designs", width="wide")


if __name__ == "__main__":
    main()
