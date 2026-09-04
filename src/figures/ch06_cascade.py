"""Cascade (multi-link EFPA) figures.

Reads data/processed/cascade_results.npz (scripts/experiments/cascade_analysis.py) and renders:
  06_cascade_coupling    inter-link coupling: independent, EFPA, simulation
  06_cascade_accuracy    EFPA over-estimate against background share
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import src.figures.style as style

style.apply()

_NPZ = style.PROCESSED / "cascade_results.npz"

# The utilisation sweep is analytical only; the path-based simulation is evaluated at the design point alone.
LABEL_INDEP = "independent links"
LABEL_EFPA = "EFPA (reduced load)"
LABEL_SIM = "simulation"

# same tick precision in both panels
_FMT3 = FuncFormatter(lambda v, _pos: f"{v:.3f}")


def _load() -> dict:
    npz = style.load_npz(
        _NPZ, "python scripts/experiments/cascade_analysis.py",
        allow_pickle=True,
    )
    return {k: npz[k] for k in npz.files}


def plot_coupling(d: dict) -> plt.Figure:
    classes = [style.class_short(str(c)) for c in d["class_order"]]
    fig, (axc, axb) = plt.subplots(1, 2, figsize=style.size("wide", 3.5))

    # The y-axis starts at 0 so the independent-to-EFPA gap is not exaggerated by a truncated baseline.
    u = d["A_util"]
    axc.plot(u, d["A_util_indep"], marker="s", linestyle="dashed", color=style.COLOR_WORST,
             label=LABEL_INDEP, markersize=4)
    axc.plot(u, d["A_util_efpa"], "o-", color=style.COLOR_ANALYTICAL,
             label=LABEL_EFPA, markersize=4)
    axc.set_xlabel("max link utilisation")
    axc.set_ylabel("weighted end-to-end blocking")
    axc.set_title("(a) utilisation sweep", fontsize=style.FS_TITLE)
    # the line styles are not keyed by the bar legend above
    axc.legend(fontsize=style.FS_ANNOT, frameon=False, loc="upper left",
               handlelength=2.2, handletextpad=0.5)
    axc.set_ylim(bottom=0)
    axc.yaxis.set_major_formatter(_FMT3)
    axc.tick_params(labelsize=style.FS_TICK)

    x = np.arange(len(classes))
    bw = 0.27
    axb.bar(x - bw, d["A_indep"], bw, color=style.COLOR_WORST,
            edgecolor="white", linewidth=0.6, label=LABEL_INDEP)
    axb.bar(x, d["A_efpa"], bw, color=style.COLOR_ANALYTICAL,
            edgecolor="white", linewidth=0.6, label=LABEL_EFPA)
    axb.bar(x + bw, d["A_mc_mean"], bw, yerr=d["A_mc_half"],
            color=style.COLOR_SIMULATION, edgecolor="white", linewidth=0.6,
            error_kw=dict(elinewidth=1.0, capsize=3, ecolor=style.GREY_ANNOT),
            label=LABEL_SIM)
    axb.set_xticks(x)
    axb.set_xticklabels(classes, fontsize=style.FS_TICK)
    axb.set_ylabel("per-class end-to-end blocking $B_k$")
    axb.set_title("(b) design point", fontsize=style.FS_TITLE)
    axb.yaxis.set_major_formatter(_FMT3)
    axb.tick_params(axis="y", labelsize=style.FS_TICK)

    # handles come from the right panel, the only one carrying all three series
    handles, labels = axb.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=style.FS_LEGEND,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def plot_accuracy(d: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=style.size("med", 3.1))
    share = d["B_share"]
    err = d["B_err_weighted"] * 100.0
    ax.plot(share, err, "o-", color=style.COLOR_ANALYTICAL, markersize=5)
    ax.axhline(0.0, color=style.GREY_RULE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("background share of link load")
    ax.set_ylabel("EFPA over-estimate (%)")

    # headroom and right-edge slack so the first and last data labels clear the spines
    ax.set_xlim(float(share.min()) - 0.03, float(share.max()) + 0.07)
    ax.set_ylim(bottom=min(0.0, float(err.min()) - 3.0),
                top=float(err.max()) * 1.16)
    for xi, yi in zip(share, err):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(7, 5), ha="left", va="bottom",
                    fontsize=style.FS_ANNOT, color=style.GREY_ANNOT)
    fig.tight_layout()
    return fig


def main() -> None:
    d = _load()
    for builder, stem, width in (
        (plot_coupling, "06_cascade_coupling", "wide"),
        (plot_accuracy, "06_cascade_accuracy", "med"),
    ):
        fig = builder(d)
        style.savefig(fig, stem, width=width)
        plt.close(fig)


if __name__ == "__main__":
    main()
