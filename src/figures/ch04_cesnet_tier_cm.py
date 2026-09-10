"""Chapter 4, CESNET-TLS-Year22 6-tier confusion-matrix heatmap.

The 6x6 row-stochastic tier-aggregated confusion of the XGBoost clean classifier, from ``cesnet_dimension.npz['rec_tier_cm']`` (6, 6, 6) at the ``rec_conds`` index of ``xgb_clean``. The K=23 per-category object lives in ``cesnet_highk_real.npz`` and is never read here.
"""

import numpy as np
import matplotlib.pyplot as plt

import src.figures.style as style
from src.figures.ch04_cm_heatmaps import add_cm_colourbar, draw_cm

style.apply()

_DIM_NPZ = style.PROCESSED / "cesnet_dimension.npz"

# Tick labels carry the tier index only; the AU ladder is stated once beneath the axis rather than on twelve ticks.
_TIER_TICKS = [f"T{i}" for i in range(1, len(style.CESNET_TIER_AU) + 1)]
_TIER_LADDER = "tiers T1 to T6 carry {} AU".format(
    ", ".join(str(au) for au in style.CESNET_TIER_AU)
)


def main() -> None:
    dim = np.load(_DIM_NPZ, allow_pickle=True)
    conds = [str(c) for c in dim["rec_conds"]]
    xgb_idx = conds.index("xgb_clean")
    cm = dim["rec_tier_cm"][xgb_idx].astype(float)

    fig, ax = plt.subplots(figsize=style.size("med", 3.95))

    im = draw_cm(
        ax, cm, _TIER_TICKS,
        xlabel=style.CM_AXIS_PRED_TIER,
        ylabel=style.CM_AXIS_TRUE_TIER,
    )
    add_cm_colourbar(fig, im, ax=ax)

    ax.annotate(
        _TIER_LADDER,
        xy=(0.5, -0.24), xycoords="axes fraction",
        ha="center", va="top",
        fontsize=style.FS_ANNOT, color=style.GREY_ANNOT,
    )

    fig.tight_layout(pad=0.6)
    style.savefig(fig, "04_cesnet_tier_cm", width="med")
    plt.close(fig)


if __name__ == "__main__":
    main()
