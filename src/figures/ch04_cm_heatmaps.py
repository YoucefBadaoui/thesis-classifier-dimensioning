"""Chapter 4, confusion-matrix heatmaps for MLP and XGBoost.

Writes ``04_mlp_cm_clean``, ``04_xgb_cm_clean`` and ``04_xgb_cm_vpn`` from ``data/processed/confusion_matrices.npz`` keys ``mlp_clean``, ``xgb_clean`` and ``xgb_vpn_shift`` (5x5, row-stochastic, alphabetical class order).

Also owns ``draw_cm`` and ``add_cm_colourbar``, the shared CM template that ``ch04_cesnet_tier_cm`` and ``ch02_flowpic_cms`` import.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.figures import style

style.apply()

_NPZ_PATH = style.PROCESSED / "confusion_matrices.npz"

_CLASSES = style.CLASS_ORDER_OTT
_SHORT   = [style.class_short(c) for c in _CLASSES]

_SPECS = [
    ("mlp_clean",     "04_mlp_cm_clean"),
    ("xgb_clean",     "04_xgb_cm_clean"),
    ("xgb_vpn_shift", "04_xgb_cm_vpn"),
]

# Cell-value text switches from dark to white at this row proportion: the contrast crossover of the Blues ramp, at which the minimum contrast anywhere on the ramp is 4.6:1.
CM_TEXT_SWITCH = 0.715

# X tick rotation shared by every CM figure.
CM_TICK_ROTATION = 35


def draw_cm(
    ax,
    cm: np.ndarray,
    labels: list[str],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    n_slots: int | None = None,
):
    """Render one row-stochastic confusion matrix on ``ax``.

    ``n_slots`` pads the axes limits to that many cells while leaving the matrix at its native size, so one cell is the same physical size in every panel of a figure. Returns the image handle for the colourbar.
    """
    n = cm.shape[0]
    im = ax.imshow(cm, cmap=style.SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="equal")

    for r in range(n):
        for c in range(n):
            val = cm[r, c]
            ax.text(
                c, r, f"{val * 100:.1f}",
                ha="center", va="center",
                fontsize=style.FS_ANNOT,
                color="white" if val > CM_TEXT_SWITCH else "black",
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=style.FS_TICK,
                       rotation=CM_TICK_ROTATION, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=style.FS_TICK)

    # fraction of the axes box covered by the matrix; titles and axis labels centre on the matrix, not on the padded box
    frac = 1.0
    if n_slots is not None and n_slots > n:
        ax.set_xlim(-0.5, n_slots - 0.5)
        ax.set_ylim(n_slots - 0.5, -0.5)
        ax.set_anchor("NW")
        frac = n / n_slots

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=style.FS_AXIS, x=frac / 2.0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=style.FS_AXIS, y=1.0 - frac / 2.0)
    if title is not None:
        ax.set_title(title, fontsize=style.FS_TITLE, pad=6, x=frac / 2.0)

    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    return im


def add_cm_colourbar(fig, im, *, ax=None, cax=None, orientation="vertical"):
    """Attach the shared CM colourbar.

    The grid is forced off: ``axes.grid`` is on globally and a colourbar is an Axes, so the y-grid would otherwise cross the colour ramp.
    """
    kw = {"cax": cax} if cax is not None else {"ax": ax, "fraction": 0.046, "pad": 0.04}
    cbar = fig.colorbar(im, orientation=orientation, **kw)
    axis = cbar.ax.yaxis if orientation == "vertical" else cbar.ax.xaxis
    axis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.set_label(style.CM_CBAR_LABEL, fontsize=style.FS_AXIS)
    cbar.ax.tick_params(labelsize=style.FS_TICK)
    cbar.ax.grid(False)
    cbar.ax.minorticks_off()
    return cbar


def _make_cm_figure(cm: np.ndarray, stem: str) -> None:
    """Render one CM heatmap and save it under ``stem``."""
    fig, ax = plt.subplots(figsize=style.size("med", 3.95))

    im = draw_cm(
        ax, cm, _SHORT,
        xlabel=style.CM_AXIS_PRED,
        ylabel=style.CM_AXIS_TRUE,
    )
    add_cm_colourbar(fig, im, ax=ax)

    fig.tight_layout(pad=0.6)
    style.savefig(fig, stem, width="med")
    plt.close(fig)


def main() -> None:
    npz = np.load(_NPZ_PATH)
    for npz_key, stem in _SPECS:
        cm = npz[npz_key].astype(float)
        _make_cm_figure(cm, stem)


if __name__ == "__main__":
    main()
