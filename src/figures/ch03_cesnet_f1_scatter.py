"""Chapter 5, CESNET Finding-F1 anchor: per-category minimum recall by allocation-unit tier.

Reads ``data/processed/cesnet_highk_real.npz`` (``category_names``, ``t``, ``gap``, ``rstar``, ``rho_f1``, ``p_f1``). The bandwidth gap is a function of the tier, so the six tiers are evenly spaced on x and each category is one marker: filled and named when its threshold binds above the search floor, hollow on the floor. The 6-tier dimensioning object in ``cesnet_dimension.npz`` is never read here. The annotated p is a permutation p; an n=23 bootstrap CI would be fragile.
"""

import numpy as np
import matplotlib.pyplot as plt

import src.figures.style as style

style.apply()

_NPZ = style.PROCESSED / "cesnet_highk_real.npz"
_DODGE = 0.13        # x spacing between hollow markers stacked on the floor
_LABEL_GAP = 0.03    # minimum vertical spacing between name labels, in r* units
_LABEL_DX = 0.17     # horizontal offset of a name from its marker, in tier units


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Label positions for values sorted descending, pushed down so none overlap."""
    out: list[float] = []
    for v in values:
        out.append(v if not out else min(v, out[-1] - min_gap))
    return out


def main() -> None:
    npz = np.load(_NPZ, allow_pickle=True)
    names = [str(n) for n in npz["category_names"]]
    t = npz["t"].astype(float)
    gap = npz["gap"].astype(float)
    rstar = npz["rstar"].astype(float)
    rho_f1 = float(npz["rho_f1"])
    p_f1 = float(npz["p_f1"])

    tiers = sorted(set(t.tolist()), reverse=True)   # gap 0 on the left, gap 14 on the right
    xpos = {tk: float(i) for i, tk in enumerate(tiers)}
    on_floor = np.isclose(rstar, style.R_FLOOR)

    fig, ax = plt.subplots(figsize=style.size("wide", 3.40))
    ax.axhline(style.R_FLOOR, color=style.GREY_RULE, linewidth=0.8,
               linestyle="dashed", alpha=0.7, zorder=2)

    x = np.array([xpos[tk] for tk in t])
    for tk in tiers:
        idx = np.where((t == tk) & on_floor)[0]
        x[idx] = xpos[tk] + (np.arange(len(idx)) - (len(idx) - 1) / 2) * _DODGE
    style.rstar_markers(ax, x, rstar)

    # names on the binding markers: the rightmost tier labels to the right, the others to the left; a leader line where a label had to move
    for tk in tiers:
        idx = sorted(np.where((t == tk) & ~on_floor)[0], key=lambda k: -rstar[k])
        if not idx:
            continue
        side = 1 if tk == tiers[-1] else -1
        for k, yl in zip(idx, _spread([rstar[k] for k in idx], _LABEL_GAP)):
            moved = abs(yl - rstar[k]) > 1e-9
            ax.annotate(
                names[k], xy=(xpos[tk], rstar[k]),
                xytext=(xpos[tk] + side * _LABEL_DX, yl), textcoords="data",
                ha="left" if side > 0 else "right", va="center",
                fontsize=style.FS_ANNOT, color=style.GREY_ANNOT, zorder=5,
                arrowprops=({"arrowstyle": "-", "color": style.GREY_RULE,
                             "linewidth": 0.5, "shrinkA": 0, "shrinkB": 4}
                            if moved else None),
            )

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f"{int(gap[t == tk][0])}\n$t_k={int(tk)}$" for tk in tiers])
    ax.set_xlim(-0.6, len(tiers) - 1 + 2.4)
    ax.set_ylim(0.44, 0.94)
    ax.set_xlabel(r"bandwidth gap ${\max}_j(t_j-t_k)$ (AU), one column per allocation-unit tier")
    ax.set_ylabel(r"$r^*_k$")
    ax.set_title(rf"$\rho_s = +{rho_f1:.3f}$, permutation $p = {p_f1:.3f}$, $K = 23$ categories")
    ax.legend(loc="upper left", fontsize=style.FS_LEGEND, frameon=False,
              handlelength=1.6, handletextpad=0.4)

    fig.tight_layout()
    style.savefig(fig, "03_cesnet_f1_gap_scatter", width="wide")
    plt.close(fig)


if __name__ == "__main__":
    main()
