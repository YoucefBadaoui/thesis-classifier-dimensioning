"""Chapter 4, CESNET-TLS-Year22 per-category sample distribution.

Per-category flow counts from ``data/processed/cesnet_definitive.npz``, keys ``category_names`` (23,), ``train_support``, ``val_support``, ``test_support``. Only the 23-category object is plotted. Bars take their colour from ``style.CESNET_CATEGORY_TO_TIER``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import src.figures.style as style

style.apply()

_NPZ_PATH = style.PROCESSED / "cesnet_definitive.npz"

# Counts are plotted in units of 1e5, with the unit stated on the axis.
_COUNT_SCALE = 1e5

# Orange is skipped because it and vermillion read alike at bar size.
_TIER_COLOR_IDX = [0, 1, 2, 3, 6, 4]

# 23 categories share the axis, so tick labels are set vertical; a shallower rotation would collide at this slot width.
_TICK_ROTATION = 90


def main() -> None:
    npz = np.load(_NPZ_PATH, allow_pickle=True)
    names = [str(n) for n in npz["category_names"]]
    train = npz["train_support"].astype(int)
    val = npz["val_support"].astype(int)
    test = npz["test_support"].astype(int)
    total_per_cat = train + val + test

    order = np.argsort(total_per_cat)[::-1]
    names_o = [names[i] for i in order]
    train_o = train[order] / _COUNT_SCALE
    val_o = val[order] / _COUNT_SCALE
    test_o = test[order] / _COUNT_SCALE

    tier_of = style.CESNET_CATEGORY_TO_TIER
    tier_palette = {i: style.CLASS_COLORS[_TIER_COLOR_IDX[i]] for i in range(6)}
    tier_palette[-1] = style.GREY_MUTED
    bar_colors = [tier_palette[tier_of[n]] for n in names_o]

    fig, ax = plt.subplots(figsize=style.size("full", 3.95))
    x = np.arange(len(names_o))

    # bar height is total per-category support
    ax.bar(x, train_o, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.bar(x, val_o, bottom=train_o, color=bar_colors, alpha=0.72,
           edgecolor="white", linewidth=0.4)
    ax.bar(x, test_o, bottom=train_o + val_o, color=bar_colors, alpha=0.45,
           edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [style.cesnet_class_short(n) for n in names_o],
        rotation=_TICK_ROTATION, ha="center", fontsize=style.FS_TICK,
    )
    ax.set_xlim(-0.7, len(names_o) - 0.3)
    ax.set_ylabel(r"flow count ($\times 10^{5}$)", fontsize=style.FS_AXIS)
    ax.set_xlabel("service category", fontsize=style.FS_AXIS)

    legend_handles = [
        Patch(facecolor=tier_palette[i], edgecolor="white",
              label=f"tier {i + 1} (AU {style.CESNET_TIER_AU[i]})")
        for i in range(6)
    ] + [Patch(facecolor=style.GREY_MUTED, edgecolor="white",
               label="background (excluded)")]
    ax.legend(
        handles=legend_handles, fontsize=style.FS_LEGEND, ncol=2,
        frameon=False, loc="upper right", handlelength=1.2,
        handletextpad=0.4, columnspacing=1.0,
    )

    fig.tight_layout(pad=0.4)
    style.savefig(fig, "04_cesnet_class_distribution", width="full")
    plt.close(fig)


if __name__ == "__main__":
    main()
