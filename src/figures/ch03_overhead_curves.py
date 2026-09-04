"""Capacity overhead curves for Chapter 3.

Two figures on one layout: ``03_capacity_overhead_curves`` (OTT/IPTV) and ``03_5g_overhead_curves`` (5G slicing). Each is one row of two panels, (a) capacity overhead with the best-case to worst-case feasible region shaded and (b) weighted bandwidth deficit, under a shared legend.

Source: ``data/processed/analytical_results.npz``.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import src.figures.style as style

style.apply()

_NPZ = style.PROCESSED / "analytical_results.npz"

# Overhead tolerance levels marked by horizontal guide lines.
_EPS_LEVELS = [5, 10, 15]

# One dash pattern per spillover model, so series identity survives where two curves coincide: in the 5G scenario the uniform and best-case overheads are both identically zero.
_LS_UNIFORM = "solid"
_LS_WORST = (0, (5, 2))
_LS_BEST = (0, (3.5, 1.4, 1, 1.4))

_SERIES_LABELS = ["uniform", "worst-case", "best-case"]
_BAND_LABEL = "feasible region"


def _load() -> dict:
    npz = np.load(_NPZ)
    return {k: npz[k] for k in npz.files}


def _overhead_top(*series: np.ndarray) -> float:
    """Where the guides bind, the limit is the guide value exactly, so the 15 percent line sits on the upper spine."""
    data_top = max(float(np.max(s)) for s in series)
    eps_top = float(max(_EPS_LEVELS))
    return eps_top if eps_top >= data_top else data_top * 1.04


def _add_eps_lines(ax: plt.Axes) -> None:
    """Draw the tolerance guides and label them at the right edge.

    Guides are labelled individually when they are at least one line of text apart on the current y scale; closer guides stay unlabelled, which only a scenario whose curves force a taller y range would reach.
    """
    for eps in _EPS_LEVELS:
        ax.axhline(eps, color=style.GREY_RULE, linewidth=0.75,
                   linestyle=(0, (3, 2)), zorder=1.5)

    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo
    axes_h_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    # one line of 8 pt text plus leading, as a fraction of the y range
    label_h = (style.FS_ANNOT * 1.5 / 72.0) / max(axes_h_in, 1e-9) * span

    gaps = [b - a for a, b in zip(_EPS_LEVELS, _EPS_LEVELS[1:])]
    resolved = all(g >= label_h for g in gaps)

    common = {
        "ha": "right", "fontsize": style.FS_ANNOT,
        "color": style.GREY_ANNOT, "zorder": 5,
    }
    if resolved:
        for eps in _EPS_LEVELS:
            # A guide sitting on the upper spine has no room above it.
            above = eps + label_h <= y_hi
            ax.text(0.985, eps, f"{eps}%",
                    transform=ax.get_yaxis_transform(),
                    va="bottom" if above else "top", **common)


def _plot_three_lines(
    ax: plt.Axes,
    recalls: np.ndarray,
    y_u: np.ndarray,
    y_w: np.ndarray,
    y_b: np.ndarray,
) -> list:
    """Draw the uniform / worst-case / best-case triplet and return handles."""
    h_u, = ax.plot(recalls, y_u, color=style.COLOR_UNIFORM,
                   linestyle=_LS_UNIFORM, zorder=3.0)
    h_w, = ax.plot(recalls, y_w, color=style.COLOR_WORST,
                   linestyle=_LS_WORST, zorder=3.1)
    h_b, = ax.plot(recalls, y_b, color=style.COLOR_BEST,
                   linestyle=_LS_BEST, zorder=3.2)
    ax.axhline(0, color=style.GREY_RULE, linewidth=0.6, zorder=1.2)
    ax.set_xlim(recalls[0], recalls[-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True)
    )
    return [h_u, h_w, h_b]


def _build_overhead_figure(
    recalls: np.ndarray,
    oh_u: np.ndarray, oh_w: np.ndarray, oh_b: np.ndarray,
    wbd_u: np.ndarray, wbd_w: np.ndarray, wbd_b: np.ndarray,
    wbd_ylabel: str = r"WBD (AU$\cdot$Erl)",
) -> plt.Figure:
    """Render the two-panel layout shared by both scenarios."""
    fig = plt.figure(figsize=style.size("wide", 3.30))
    gs = gridspec.GridSpec(
        1, 2,
        wspace=0.34,
        left=0.105, right=0.985,
        top=0.905, bottom=0.235,
    )
    ax_oh = fig.add_subplot(gs[0, 0])
    ax_wbd = fig.add_subplot(gs[0, 1])

    band = ax_oh.fill_between(
        recalls, oh_b, oh_w,
        color=style.COLOR_UNIFORM, alpha=0.18, linewidth=0, zorder=1.0,
    )
    handles = _plot_three_lines(ax_oh, recalls, oh_u, oh_w, oh_b)
    ax_oh.set_xlabel("recall threshold $r$")
    ax_oh.set_ylabel("capacity overhead (%)")
    ax_oh.set_title("(a) overhead envelope")
    top = _overhead_top(oh_u, oh_w, oh_b)
    ax_oh.set_ylim(-0.035 * top, top)
    # drawn after the limits settle so the highest guide can sit on the upper spine without reopening autoscaling
    _add_eps_lines(ax_oh)
    if np.allclose(oh_u, oh_b):
        ax_oh.text(
            0.035, 0.045, "uniform and best-case\ncoincide at zero",
            transform=ax_oh.transAxes, ha="left", va="bottom",
            fontsize=style.FS_ANNOT, color=style.GREY_ANNOT, zorder=5,
        )

    _plot_three_lines(ax_wbd, recalls, wbd_u, wbd_w, wbd_b)
    ax_wbd.set_xlabel("recall threshold $r$")
    ax_wbd.set_ylabel(wbd_ylabel)
    ax_wbd.set_title("(b) bandwidth deficit")

    fig.legend(
        handles + [band], _SERIES_LABELS + [_BAND_LABEL],
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=False, fontsize=style.FS_LEGEND,
        columnspacing=1.5, handletextpad=0.6,
    )
    return fig


# Figure stem to (recall key, overhead key prefix, deficit key prefix).
_SCENARIOS: dict[str, tuple[str, str, str]] = {
    "03_capacity_overhead_curves": ("recalls_sweep", "overhead_", "wbd_"),
    "03_5g_overhead_curves": ("recalls_5g_sweep", "oh_5g_", "wbd_5g_"),
}

_MODELS = ("uniform", "worst", "best")


def main() -> None:
    d = _load()

    for stem, (recall_key, oh_prefix, wbd_prefix) in _SCENARIOS.items():
        fig = _build_overhead_figure(
            d[recall_key],
            *(d[oh_prefix + m] for m in _MODELS),
            *(d[wbd_prefix + m] for m in _MODELS),
        )
        style.savefig(fig, stem, width="wide")
        plt.close(fig)


if __name__ == "__main__":
    main()
