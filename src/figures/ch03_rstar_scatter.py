"""Per-class minimum recall against the load-demand product, Chapter 5.

Two-panel scatter, one panel per tolerance (5% and 10%), of the per-class minimum recall threshold r*_k against a_k.t_k for the five OTT/IPTV classes. Predictor on x, r*_k on y, as in ``ch03_highk_power``. The eps=15% panel is omitted because every r*_k clips to the 0.50 lower bound at that tolerance.

Source: ``data/processed/analytical_results.npz``.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import src.figures.style as style

style.apply()

_NPZ = style.PROCESSED / "analytical_results.npz"

_EPS_VALUES = [5, 10]

# Per-panel, per-class label offsets (dx, dy in data units).
# VoIP and Chat share an x-band, so they are separated in y. At eps=10 Chat also sits on the lower bound, so its label is routed below the line.
_LABEL_OFFSETS_BY_EPS: dict[int, dict[str, tuple[float, float]]] = {
    5: {
        "VoIP":         (  6,  0.045),
        "Browsing":     ( 10, -0.035),
        "Chat":         (  6, -0.055),
        "Streaming":    (-12,  0.080),  # up-left, taller leader to clear FT
        "FileTransfer": (  4,  0.030),
    },
    10: {
        "VoIP":         (  6,  0.045),
        "Browsing":     ( 10, -0.035),
        "Chat":         (  6, -0.050),
        "Streaming":    (  4, -0.050),  # routed under the r=0.50 line so it
                                        # clears the Browsing marker band
        "FileTransfer": (  4,  0.045),
    },
}


def _load() -> dict:
    npz = np.load(_NPZ)
    return {k: npz[k] for k in npz.files}


def plot_rstar(d: dict) -> plt.Figure:
    """Two-panel r*_k against a_k.t_k scatter (epsilon = 5% and 10%)."""
    classes = list(d["class_order_ott"])
    a = d["a_ott"]
    t = d["t_ott"]
    atk = a * t

    rstar_by_eps = {
        5:  np.round(d["rstar_ott_eps5"],  2),
        10: np.round(d["rstar_ott_eps10"], 2),
    }

    colors = style.CLASS_COLORS[:len(classes)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=style.size("wide", 3.50),
        sharey=True,
    )

    lb_handle = None
    for panel_idx, eps in enumerate(_EPS_VALUES):
        ax = axes[panel_idx]
        rstar = rstar_by_eps[eps]

        rho, _ = stats.spearmanr(atk, rstar)

        # drawn before the markers so they paint over the dash where they intersect it
        lb_handle = ax.axhline(
            0.50, color=style.GREY_RULE, linewidth=0.8, linestyle="dashed",
            alpha=0.7, zorder=2, label="lower bound $r=0.50$",
        )

        offsets = _LABEL_OFFSETS_BY_EPS[eps]
        for i, cls in enumerate(classes):
            ax.scatter(
                atk[i], rstar[i], color=colors[i], s=style.MARKER_SIZE,
                edgecolor="white", linewidth=0.6, zorder=4,
            )
            dx, dy = offsets[cls]
            ax.annotate(
                style.class_short(cls), xy=(atk[i], rstar[i]),
                xytext=(atk[i] + dx, rstar[i] + dy),
                fontsize=style.FS_ANNOT,
                ha="left" if dx > 0 else "right",
                va="center", color=style.GREY_ANNOT,
                arrowprops={"arrowstyle": "-", "color": style.GREY_RULE,
                            "lw": 0.4},
            )

        ax.set_xlim(-5, 205)
        ax.set_ylim(0.42, 0.94)
        ax.set_title(
            f"({chr(97 + panel_idx)}) $\\varepsilon={eps}\\%$,"
            f" $\\rho_s={rho:.2f}$"
        )
        ax.set_xlabel(r"$a_k \cdot t_k$" + " (" + style.UNIT_AU_ERL_LATEX + ")")
        if panel_idx == 0:
            ax.set_ylabel(r"$r^*_k$")

    fig.legend(
        [lb_handle], ["lower bound $r=0.50$"],
        loc="lower center", ncol=1, fontsize=style.FS_LEGEND,
        frameon=False, handlelength=1.6, handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return fig


def main() -> None:
    d = _load()
    fig = plot_rstar(d)
    style.savefig(fig, "03_rstar_vs_atk", width="wide")
    plt.close(fig)


if __name__ == "__main__":
    main()
