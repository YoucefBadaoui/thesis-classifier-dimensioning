"""Chapter 6, Monte Carlo validation figures.

Three stems from ``data/processed/monte_carlo_results.npz``: ``06_convergence_ott`` (per-class convergence trace, 3x2 grid, M=30), ``06_mc_vs_analytical_bar`` (two-panel grouped bar with CI whiskers) and ``06_mc_distorted_ott`` (three-panel distorted-load comparison). Blocking probabilities are displayed in units of 1e-3; the NPZ values are unscaled.
"""

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from scipy import stats

import src.figures.style as style

style.apply()

_NPZ_PATH = style.PROCESSED / "monte_carlo_results.npz"

# Display names for distorted-load CM identifiers.
_DISTORTED_DISPLAY: dict[str, str] = {
    "xgb_clean": "XGBoost\nclean",
    "flowpic_nonvpn": "FlowPic\nnon-VPN",
    "xgb_vpn_shift": "XGBoost\nVPN shift",
}
_DISTORTED_KEYS = ["xgb_clean", "flowpic_nonvpn", "xgb_vpn_shift"]

# Display scale: blocking probabilities are plotted in units of 1e-3.
MILLI = 1.0e3
UNIT_SUFFIX = r" ($\times 10^{-3}$)"

# One name per series, shared by all three figures in the chapter.
LABEL_ANALYTICAL = "analytical"
LABEL_SIMULATED = "simulated"

# round tick steps, so a fixed decimal format never spaces ticks unevenly
_ROUND_STEPS = [1, 2, 2.5, 5, 10]


def _fmt(decimals: int) -> FuncFormatter:
    """Fixed-decimal tick formatter, uniform across a figure's panels."""
    return FuncFormatter(lambda v, _pos: f"{v:.{decimals}f}")


def _figure_convergence(npz: np.lib.npyio.NpzFile) -> None:
    classes = list(npz["class_order_ott"])
    # rows are replications, columns are classes
    all_blocking = npz["ott_all_blocking"]
    B_analytical = npz["ott_B_analytical"]          # (K,)
    M = all_blocking.shape[0]

    cum_mean = np.cumsum(all_blocking, axis=0) / np.arange(1, M + 1)[:, None]
    cum_ci = np.full_like(cum_mean, np.nan, dtype=float)
    for m in range(2, M + 1):
        sample = all_blocking[:m, :]
        sem = sample.std(axis=0, ddof=1) / np.sqrt(m)
        cum_ci[m - 1, :] = stats.t.ppf(0.975, df=m - 1) * sem

    colors = [style.CLASS_COLORS[i] for i in range(len(classes))]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=style.size("wide", 6.20),
        sharex=True,
        constrained_layout=True,
    )
    reps = np.arange(1, M + 1)

    for idx, (cls, color, B_ref) in enumerate(zip(classes, colors, B_analytical)):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        lower = (cum_mean[:, idx] - cum_ci[:, idx]) * MILLI
        upper = (cum_mean[:, idx] + cum_ci[:, idx]) * MILLI
        ax.fill_between(
            reps,
            lower,
            upper,
            where=np.isfinite(lower) & np.isfinite(upper),
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(reps, cum_mean[:, idx] * MILLI, color=color, linewidth=1.7,
                label=cls, zorder=2)

        # analytical reference line, value stated in the same units as the axis ticks
        ax.axhline(B_ref * MILLI, color=style.COLOR_ANALYTICAL, linestyle="dashed",
                   linewidth=1.2, zorder=3)
        ax.text(
            0.97,
            0.06,
            f"K-R = {B_ref * MILLI:.2f}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=style.FS_ANNOT,
            color=style.COLOR_ANALYTICAL,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
        )

        display_name = f"({chr(97 + idx)}) {style.class_short(cls)}"
        ax.set_title(display_name, fontsize=style.FS_TITLE, pad=3)
        ax.set_xlim(1, M)
        ax.set_xticks([1, 10, 20, M])
        # nbins is 5 because a tighter budget collapses the Streaming panel, which has the widest early CI band, to a two-tick axis
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=_ROUND_STEPS))
        ax.yaxis.set_major_formatter(_fmt(2))
        ax.tick_params(labelsize=style.FS_TICK)

    # The legend takes the empty sixth cell below panel (d), so sharex hides panel (d)'s tick labels although it is the bottom-most drawn axes in its column. Re-enable labels on the bottom-most axes of each column.
    for ax in (axes[1, 1], axes[2, 0]):
        ax.tick_params(axis="x", labelbottom=True, labelsize=style.FS_TICK)

    ax_legend = axes[2, 1]
    ax_legend.axis("off")

    # one handle per class, drawn side by side, because each panel uses its own class colour and a single swatch would be wrong for four of five
    mean_handles = tuple(
        Line2D([], [], color=c, linewidth=1.7) for c in colors
    )
    ci_handles = tuple(
        Patch(facecolor=c, edgecolor="none", alpha=0.16) for c in colors
    )
    ref_handle = Line2D([], [], color=style.COLOR_ANALYTICAL, linestyle="dashed",
                        linewidth=1.2)
    ax_legend.legend(
        handles=[mean_handles, ci_handles, ref_handle],
        labels=["cumulative mean", "95% Student-$t$ CI", "K-R analytical $B_k$"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="center",
        fontsize=style.FS_LEGEND,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.6,
    )

    fig.supxlabel("replication index $m$", fontsize=style.FS_AXIS)
    fig.supylabel("cumulative mean blocking probability $B_k$" + UNIT_SUFFIX,
                  fontsize=style.FS_AXIS)
    style.savefig(fig, "06_convergence_ott", width="wide")
    plt.close(fig)


def _figure_mc_vs_analytical_bar(npz: np.lib.npyio.NpzFile) -> None:
    ott_classes = list(npz["class_order_ott"])
    fg5_classes = list(npz["class_order_5g"])

    # Panel widths track the class counts so a bar is the same physical width in both panels.
    fig, axes = plt.subplots(
        1, 2,
        figsize=style.size("med", 3.30),
        sharey=True,
        gridspec_kw={"width_ratios": [len(ott_classes), len(fg5_classes)]},
    )

    _draw_bar_panel(
        axes[0],
        ott_classes,
        npz["ott_B_analytical"],
        npz["ott_B_simulated"],
        npz["ott_ci_half"],
        title=f"(a) {style.SCENARIO_OTT} scenario",
    )
    _draw_bar_panel(
        axes[1],
        fg5_classes,
        npz["5g_B_analytical"],
        npz["5g_B_simulated"],
        npz["5g_ci_half"],
        title=f"(b) {style.SCENARIO_5G} scenario",
    )

    axes[0].set_ylabel("blocking probability $B_k$" + UNIT_SUFFIX,
                       fontsize=style.FS_AXIS)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        fontsize=style.FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.4, handletextpad=0.4, columnspacing=1.4,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    style.savefig(fig, "06_mc_vs_analytical_bar", width="med")
    plt.close(fig)


def _draw_bar_panel(
    ax: plt.Axes,
    classes: list[str],
    B_analytical: np.ndarray,
    B_simulated: np.ndarray,
    ci_half: np.ndarray,
    title: str,
    *,
    rotation: float = 30,
    nbins: int = 5,
    title_pad: float | None = None,
) -> None:
    K = len(classes)
    x = np.arange(K)
    w = 0.35

    ax.bar(
        x - w / 2, B_analytical * MILLI, w,
        label=LABEL_ANALYTICAL,
        color=style.COLOR_ANALYTICAL,
        edgecolor="white", linewidth=0.6,
    )
    ax.bar(
        x + w / 2, B_simulated * MILLI, w,
        label=LABEL_SIMULATED,
        color=style.COLOR_SIMULATION,
        edgecolor="white", linewidth=0.6,
        yerr=ci_half * MILLI,
        error_kw=dict(elinewidth=1.0, capsize=3, ecolor=style.GREY_ANNOT),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([style.class_short(c) for c in classes],
                       rotation=rotation, ha="right", fontsize=style.FS_TICK)
    ax.set_title(title, fontsize=style.FS_TITLE, pad=title_pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=_ROUND_STEPS))
    ax.yaxis.set_major_formatter(_fmt(1))
    ax.tick_params(axis="y", labelsize=style.FS_TICK)


def _figure_mc_distorted_ott(npz: np.lib.npyio.NpzFile) -> None:
    classes = list(npz["class_order_ott"])

    fig, axes = plt.subplots(1, 3, figsize=style.size("wide", 3.10))

    # the three panels cover different ranges and cannot share a y-axis; a round locator plus one decimal keeps the precision uniform
    for idx, (ax, key) in enumerate(zip(axes, _DISTORTED_KEYS)):
        _draw_bar_panel(
            ax,
            classes,
            npz[f"ott_dist_{key}_B_analytical"],
            npz[f"ott_dist_{key}_B_simulated"],
            npz[f"ott_dist_{key}_ci_half"],
            title=f"({chr(97 + idx)}) {_DISTORTED_DISPLAY[key]}",
            rotation=45,
            nbins=6,
            title_pad=3,
        )

    fig.supylabel("blocking probability $B_k$" + UNIT_SUFFIX, fontsize=style.FS_AXIS)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        fontsize=style.FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.4, handletextpad=0.4, columnspacing=1.4,
    )
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.88))
    style.savefig(fig, "06_mc_distorted_ott", width="wide")
    plt.close(fig)


def main() -> None:
    npz = style.load_npz(
        _NPZ_PATH, "notebooks/04_monte_carlo.ipynb", allow_pickle=True,
    )

    _figure_convergence(npz)
    _figure_mc_vs_analytical_bar(npz)
    _figure_mc_distorted_ott(npz)


if __name__ == "__main__":
    main()
