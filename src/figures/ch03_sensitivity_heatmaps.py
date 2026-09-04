"""Chapter 3, sensitivity heatmaps and cross-scenario projected bar.

Three figures from ``data/processed/analytical_results.npz``: ``03_sensitivity_xgb_clean`` and ``03_sensitivity_flowpic_tor``, each a ``(K=5, K, K)`` swap-sensitivity tensor drawn as five panels plus a shared colourbar; ``03_sensitivity_heavy_cross_scenario``, the row L2 norms of the projected tensors for both scenarios as a two-panel bar chart.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.figures import style

style.apply()

NPZ_PATH = style.PROCESSED / "analytical_results.npz"


def _format_cell(val: float, vmax: float, scale: float = 1.0) -> str:
    """Annotation text for a heatmap cell.

    Three decimals when the panel range is below 0.05, two otherwise, and two in scaled units where a power of ten is factored into the colourbar label. Values that would round to zero render as a dot; exact zeros on the structural diagonal render as "0".
    """
    if val == 0.0:
        return "0"
    if scale != 1.0:
        scaled = val / scale
        return r"$\cdot$" if abs(scaled) < 0.005 else f"{scaled:+.2f}"
    # threshold below which the rounded annotation would be "+0.00"
    threshold = 0.005 if vmax >= 0.05 else 0.0005
    if abs(val) < threshold:
        return r"$\cdot$"
    return f"{val:+.3f}" if vmax < 0.05 else f"{val:+.2f}"


# Authored height of the three-row K=5 panel grid, inside MAX_HEIGHT_IN.
_GRID_HEIGHT_IN = 6.20


def _draw_panel_grid(
    S: np.ndarray,
    class_labels: list[str],
    title_format: str,
    vmax_floor: float = 0.0,
) -> plt.Figure:
    """Draw a 3-row 2-col grid of K=5 panels, one heatmap per class.

    Panels 0-1 on row 0, 2-3 on row 1, 4 on row 2 with the shared colourbar in the sixth cell. Cells use ``aspect="auto"``: square cells wide enough for a five-character annotation would overrun the one-page float budget.

    ``title_format`` is a mathtext string with one ``{cls}`` placeholder. ``vmax_floor`` raises the symmetric colour range to at least ``+/-vmax_floor``. Cell annotations are factored by a shared power of ten when the panel range is below 1e-3; the colourbar always factors one.
    """
    K = S.shape[0]
    data_vmax = float(np.abs(S).max())
    vmax = max(data_vmax, float(vmax_floor))
    vmin = -vmax
    uses_sci = data_vmax < 1e-3

    # Shared scale exponent for tiny-magnitude panels (cell annotations).
    if uses_sci:
        scale_exp = int(np.floor(np.log10(data_vmax)))
        scale = 10.0 ** scale_exp
    else:
        scale_exp = 0
        scale = 1.0

    # Colourbar exponent, factored from the displayed range so tick mantissas stay O(1).
    cbar_exp = int(np.floor(np.log10(vmax)))

    fig = plt.figure(figsize=style.size("full", _GRID_HEIGHT_IN))
    gs = fig.add_gridspec(
        3, 2,
        wspace=0.22,
        hspace=0.42,
        left=0.120, right=0.985,
        top=0.945, bottom=0.075,
    )

    axes: list[plt.Axes] = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
    ]

    # Bottom-most drawn panel of each column carries the x-axis label; column 1 ends at panel 3 because the colourbar takes its last cell.
    xlabel_panels = {3, 4}

    for k in range(K):
        ax = axes[k]
        ax.grid(False)
        mat = S[k]
        im = ax.imshow(
            mat, cmap=style.DIVERGING, vmin=vmin, vmax=vmax, aspect="auto",
        )
        # Cell boundaries: an explicit minor-tick grid, so separators stay visible on the near-white cells around zero.
        ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
        ax.grid(which="minor", color=style.GREY_RULE, linewidth=0.6,
                alpha=0.7)
        ax.tick_params(which="minor", length=0)

        ax.set_title(
            f"({chr(97 + k)}) " + title_format.format(cls=class_labels[k]),
            fontsize=style.FS_TITLE, pad=4)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(class_labels, fontsize=style.FS_TICK, rotation=0)
        ax.set_yticklabels(class_labels, fontsize=style.FS_TICK)
        ax.tick_params(axis="x", pad=1.5)
        ax.tick_params(axis="y", pad=1.5)
        if k in xlabel_panels:
            ax.set_xlabel(r"predicted $j$", fontsize=style.FS_AXIS,
                          labelpad=2)
        if k in (0, 2, 4):
            ax.set_ylabel(r"true $i$", fontsize=style.FS_AXIS, labelpad=2)

        text_color_threshold = 0.55 * vmax
        for i in range(K):
            for j in range(K):
                val = mat[i, j]
                ax.text(
                    j, i, _format_cell(val, vmax, scale),
                    ha="center", va="center",
                    fontsize=style.FS_ANNOT,
                    color=("black" if abs(val) < text_color_threshold
                           else "white"),
                )

    # shared colourbar in the empty bottom-right cell
    cbar_host = fig.add_subplot(gs[2, 1])
    cbar_host.set_axis_off()
    cbar_ax = cbar_host.inset_axes((0.06, 0.46, 0.88, 0.16))
    cbar_ax.grid(False)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        r"$\partial B_k / \partial C_{ij}|_\mathrm{swap}$"
        fr"  ($\times 10^{{{cbar_exp}}}$)",
        fontsize=style.FS_AXIS,
    )
    cbar.ax.tick_params(labelsize=style.FS_TICK)
    cbar.ax.grid(False)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [f"{(t / 10.0 ** cbar_exp):+.2f}" for t in ticks]
    )
    return fig


def _draw_cross_scenario_bars(
    row_l2_ott: np.ndarray,
    row_l2_5g: np.ndarray,
    classes_ott: list[str],
    classes_5g: list[str],
) -> plt.Figure:
    """Paired bar chart of per-class row L2 projected sensitivity.

    Left panel is ``S_sys_proj_xgb_clean`` (5 bars), right is ``S_sys_proj_5g_uniform`` (3 bars). The worst row in each panel is highlighted; both panels share one y range.
    """
    fig, (ax_ott, ax_5g) = plt.subplots(
        1, 2, figsize=style.size("wide", 3.00),
    )

    # Shared y-axis upper bound.
    ymax_raw = max(row_l2_ott.max(), row_l2_5g.max())
    ymax = ymax_raw * 1.22

    short_ott = [style.class_short(c) for c in classes_ott]
    short_5g = [style.class_short(c) for c in classes_5g]

    def _draw_panel(ax, labels, values, panel_title, show_ylabel):
        worst_idx = int(np.argmax(values))
        colours = [
            style.COLOR_WORST if i == worst_idx else style.COLOR_UNIFORM
            for i in range(len(values))
        ]
        bars = ax.bar(
            range(len(values)), values,
            color=colours,
            edgecolor=style.GREY_ANNOT, linewidth=0.5,
            width=0.72,
        )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=style.FS_TICK, rotation=0)
        if show_ylabel:
            ax.set_ylabel(r"row-$L_2$ projected sensitivity",
                          fontsize=style.FS_AXIS)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_ylim(0, ymax)
        ax.set_title(panel_title, fontsize=style.FS_TITLE)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)

        # bar-top annotations; the worst row is bolded to match the colour
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + ymax * 0.02,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=style.FS_ANNOT,
                fontweight="bold" if i == worst_idx else "normal",
                color=style.GREY_ANNOT,
            )

    _draw_panel(
        ax_ott, short_ott, row_l2_ott,
        "(a) OTT/IPTV (XGBoost clean)", True,
    )
    _draw_panel(
        ax_5g, short_5g, row_l2_5g,
        r"(b) 5G slicing (uniform $r{=}0.90$)", False,
    )

    # Explicit placement gives both panels the same physical bar width; a 5:3 gridspec split does not, because each cell also carries its own axis furniture. The right bound keeps slack because panel (b)'s title overhangs its axes and the symmetric tight crop would otherwise eat the y-axis label on the opposite side.
    n_ott, n_5g = len(row_l2_ott), len(row_l2_5g)
    left, right, bottom, top, gap = 0.110, 0.935, 0.145, 0.895, 0.075
    span = (right - left) - gap
    w_ott = span * n_ott / (n_ott + n_5g)
    w_5g = span * n_5g / (n_ott + n_5g)
    ax_ott.set_position([left, bottom, w_ott, top - bottom])
    ax_5g.set_position([left + w_ott + gap, bottom, w_5g, top - bottom])
    return fig


def main() -> None:
    d = style.load_npz(NPZ_PATH, "scripts/regenerate_analytical_results.py")

    classes_ott_full = list(d["class_order_ott"])
    classes_5g_full = list(d["class_order_5g"])
    ott_short = [style.class_short(c) for c in classes_ott_full]

    # vmax_floor=0.15 keeps the +0.13/+0.11 cells off the colour maximum.
    fig1 = _draw_panel_grid(
        d["S_xgb_clean"],
        ott_short,
        title_format=r"$B_\mathrm{{{cls}}}$",
        vmax_floor=0.15,
    )
    style.savefig(fig1, "03_sensitivity_xgb_clean", width="full")
    plt.close(fig1)

    fig2 = _draw_panel_grid(
        d["S_flowpic_tor"],
        ott_short,
        title_format=r"$B_\mathrm{{{cls}}}$",
    )
    style.savefig(fig2, "03_sensitivity_flowpic_tor", width="full")
    plt.close(fig2)

    S_sys_ott = d["S_sys_proj_xgb_clean"]
    S_sys_5g = d["S_sys_proj_5g_uniform"]
    row_l2_ott = np.linalg.norm(S_sys_ott, axis=1)
    row_l2_5g = np.linalg.norm(S_sys_5g, axis=1)

    fig3 = _draw_cross_scenario_bars(
        row_l2_ott, row_l2_5g,
        classes_ott_full, classes_5g_full,
    )
    style.savefig(fig3, "03_sensitivity_heavy_cross_scenario", width="wide")
    plt.close(fig3)


if __name__ == "__main__":
    main()
