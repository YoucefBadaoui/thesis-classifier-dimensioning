"""Chapter 3, grouped bar chart of per-class delta_B.

Per-class delta_B across the nine confusion matrices of the OTT/IPTV ensemble, in two stacked panels: three published FlowPic anchors above six own-classifier matrices.

Source: ``data/processed/analytical_results.npz``.
"""

import numpy as np
import matplotlib.pyplot as plt

import src.figures.style as style

style.apply()

_NPZ = style.PROCESSED / "analytical_results.npz"

# Nine series in one grouped bar exceed the seven non-black hues of the palette, so each cohort gets its own panel.
_PUBLISHED: list[tuple[str, str]] = [
    ("FP-NonVPN", "delta_B_flowpic_nonvpn"),
    ("FP-Tor", "delta_B_flowpic_tor"),
    ("FP-VPN (4-class)", "fp_vpn_4x4_delta_B"),
]

_OWN: list[tuple[str, str]] = [
    ("XGB-clean", "delta_B_xgb_clean"),
    ("MLP-clean", "delta_B_mlp_clean"),
    ("XGB-VPN", "delta_B_xgb_vpn_shift"),
    ("MLP-V", "delta_B_mlp_vpn_shift"),
    ("XGB red. feat.", "delta_B_xgb_reduced_feat"),
    ("MLP red. feat.", "delta_B_mlp_reduced_feat"),
]

_COLORS: dict[str, str] = {
    "FP-NonVPN": style.CLASS_COLORS[0],
    "FP-Tor": style.CLASS_COLORS[1],
    "FP-VPN (4-class)": style.GREY_MUTED,  # 4-class structural outlier
    # two hues recur across the cohorts; each panel carries its own legend
    "XGB-clean": style.CLASS_COLORS[2],
    "MLP-clean": style.CLASS_COLORS[3],
    "XGB-VPN": style.CLASS_COLORS[5],
    "MLP-V": style.COLOR_MLPV,  # reddish purple, failure mode
    "XGB red. feat.": style.CLASS_COLORS[0],
    "MLP red. feat.": style.CLASS_COLORS[1],
}

_HATCH: dict[str, str] = {"MLP-V": "///"}

_CLASSES_5 = style.CLASS_ORDER_OTT

_LINTHRESH = 1e-4
_LINSCALE = 0.5


def _load() -> dict:
    npz = np.load(_NPZ, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def _series(d: dict, key: str) -> np.ndarray:
    """Per-class delta_B on the 5-class axis, NaN where not evaluated.

    The FlowPic VPN matrix is native 4-class: the VPN partition of the ISCX corpus has no Browsing flows. That slot carries NaN, not zero, so an empty bar is not read as a genuine zero deviation.
    """
    if key != "fp_vpn_4x4_delta_B":
        return np.asarray(d[key], dtype=float)

    out = np.full(len(_CLASSES_5), np.nan)
    for cls, val in zip([str(c) for c in d["fp_vpn_4x4_classes"]],
                        np.asarray(d[key], dtype=float)):
        out[_CLASSES_5.index(cls)] = val
    return out


def _draw_panel(ax, d: dict, cohort: list[tuple[str, str]], title: str) -> None:
    n = len(cohort)
    x = np.arange(len(_CLASSES_5))
    total_width = 0.80
    bar_w = total_width / n
    offsets = (np.arange(n) - (n - 1) / 2) * bar_w

    for (label, key), off in zip(cohort, offsets):
        vals = _series(d, key)
        ax.bar(x + off, vals, width=bar_w,
               color=_COLORS[label], hatch=_HATCH.get(label),
               edgecolor="white", linewidth=0.4, label=label)

        for xi in np.flatnonzero(np.isnan(vals)):
            ax.annotate("n/a", (xi + off, 0.0), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom",
                        fontsize=style.FS_ANNOT, color=style.GREY_ANNOT)

    ax.axhline(0, color=style.GREY_RULE, linewidth=0.8, zorder=3)
    ax.set_yscale("symlog", linthresh=_LINTHRESH, linscale=_LINSCALE)
    ax.set_title(title, fontsize=style.FS_TITLE, loc="left")
    ax.tick_params(axis="y", labelsize=style.FS_TICK)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1,
              frameon=False, fontsize=style.FS_LEGEND,
              handlelength=1.2, handletextpad=0.5, labelspacing=0.35)


def plot_delta_b(d: dict) -> plt.Figure:
    """Per-class delta_B, published anchors above own-classifier matrices."""
    fig, (ax_pub, ax_own) = plt.subplots(
        2, 1, sharex=True, sharey=True, figsize=style.size("wide", 4.60)
    )

    _draw_panel(ax_pub, d, _PUBLISHED, "(a) published FlowPic anchors")
    _draw_panel(ax_own, d, _OWN, "(b) own-classifier matrices")

    ax_own.set_xticks(np.arange(len(_CLASSES_5)))
    ax_own.set_xticklabels(style.CLASS_ORDER_OTT_SHORT,
                           fontsize=style.FS_TICK)
    ax_own.set_xlabel("traffic class")

    fig.supylabel(r"$\Delta B_k$  (symlog, linear within $\pm 10^{-4}$)",
                  fontsize=style.FS_AXIS, x=0.005)

    fig.tight_layout()
    return fig


def main() -> None:
    d = _load()
    fig = plot_delta_b(d)
    style.savefig(fig, "03_delta_B_grouped_bar", width="wide")
    plt.close(fig)


if __name__ == "__main__":
    main()
