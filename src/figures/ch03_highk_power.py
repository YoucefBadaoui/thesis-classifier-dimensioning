"""High-K statistical-power figure for Chapter 5.

Panel (a): fraction of random synthetic scenarios in which the Spearman correlation between a predictor and the per-class minimum recall r*_k reaches p < 0.05, against class count K, for the F1 bandwidth-gap and H3 load-demand predictors. Panel (b): the median Spearman correlation over the same scenarios, with the fraction of scenarios in which each predictor keeps its predicted sign.

Source: ``data/processed/highk_power.npz`` (``scripts/experiments/highk_power_analysis.py``).
"""

import numpy as np
import matplotlib.pyplot as plt

import src.figures.style as style

style.apply()

_NPZ = style.PROCESSED / "highk_power.npz"


def _load() -> dict:
    npz = np.load(_NPZ, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def plot_highk(d: dict) -> plt.Figure:
    K = np.asarray(d["sweep_K_grid"], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=style.size("wide", 3.30))

    ax = axes[0]
    # {\max}_j braced: bare \max_j takes display-style limits in mathtext and drops the subscript onto a second line
    ax.plot(K, d["sweep_frac_sig_f1"], marker="o", color=style.COLOR_UNIFORM,
            label=r"F1: ${\max}_j(t_j-t_k)$")
    ax.plot(K, d["sweep_frac_sig_h3"], marker="s", color=style.COLOR_WORST,
            label=r"H3: $a_k t_k$")
    # solid so it stands apart from the dotted grid
    ax.axhline(0.80, color=style.GREY_RULE, linewidth=0.9, linestyle="-",
               alpha=0.9)
    ax.axvline(5, color=style.GREY_RULE, linewidth=0.8, linestyle="dashed",
               alpha=0.6)
    ax.set_xticks(K)
    ax.annotate("empirical\n$K=5$", xy=(5.3, 1.00),
                fontsize=style.FS_ANNOT, ha="left", va="top",
                color=style.GREY_ANNOT)
    ax.set_xlabel(r"number of traffic classes $K$")
    ax.set_ylabel(r"fraction with $p<0.05$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"(a) power {style.VERSUS} class count")
    # the free band lies between the two curves at K >= 9
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.63),
              fontsize=style.FS_LEGEND, frameon=False, handlelength=1.2)

    ax = axes[1]
    ax.axhline(0.0, color=style.GREY_RULE, linewidth=0.8, alpha=0.9)
    ax.plot(K, d["sweep_med_rho_f1"], marker="o", color=style.COLOR_UNIFORM)
    ax.plot(K, d["sweep_med_rho_h3"], marker="s", color=style.COLOR_WORST)
    pos_f1 = float(np.min(d["sweep_frac_pos_f1"]))
    neg_h3 = d["sweep_frac_neg_h3"]
    ax.annotate(
        f"positive in {pos_f1:.0%} of scenarios",
        xy=(K[-1], d["sweep_med_rho_f1"][-1]), xytext=(0, 9),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=style.FS_ANNOT, color=style.GREY_ANNOT)
    ax.annotate(
        f"negative in {np.min(neg_h3):.0%} to {np.max(neg_h3):.0%}",
        xy=(K[-1], d["sweep_med_rho_h3"][-1]), xytext=(0, -9),
        textcoords="offset points", ha="right", va="top",
        fontsize=style.FS_ANNOT, color=style.GREY_ANNOT)
    ax.set_xticks(K)
    ax.set_xlabel(r"number of traffic classes $K$")
    ax.set_ylabel(r"median Spearman $\rho_s$ with $r^*_k$")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"(b) effect size {style.VERSUS} class count")

    fig.tight_layout()
    return fig


def main() -> None:
    d = _load()
    fig = plot_highk(d)
    style.savefig(fig, "03_highk_power", width="wide")
    plt.close(fig)


if __name__ == "__main__":
    main()
