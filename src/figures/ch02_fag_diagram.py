"""Chapter 2, Full-Availability Group schematic.

Symbolic FAG diagram: K=5 classes with a_k and t_k on the left, shared capacity V in the centre, B_k on the right.
"""

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.figures import style

style.apply()

# Classes are numbered in ascending AU demand, the subscript convention of Chapter 3 (t_1 is VoIP at 1 AU, t_5 is Streaming at 15 AU), not the alphabetical storage order of the empirical confusion matrices.
K = 5
_CLASSES = ["VoIP", "Chat", "Browsing", "File Transfer", "Streaming"]

# Horizontal layout on a 0..10 axis.
_LEFT_X = 0.15
_ARROW_IN_0, _ARROW_IN_1 = 2.75, 3.60
_BOX_L, _BOX_R = 3.70, 7.60
_ARROW_OUT_0, _ARROW_OUT_1 = 7.70, 8.50
_RIGHT_X = 8.70

# Vertical layout: K rows on a unit pitch.
_ROW_PITCH = 1.0
_ROW_TOP = 4.4
_SUBLABEL_DROP = 0.30


def _draw_arrow(ax, x0, x1, y):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=style.GREY_ANNOT, lw=1.1),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=style.size("med", 2.75))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.35, 5.05)
    ax.axis("off")
    # apply() turns the grid on globally; a schematic has no data axes.
    ax.grid(False)

    row_ys = [_ROW_TOP - i * _ROW_PITCH for i in range(K)]

    box_bot = row_ys[-1] - 0.45
    box_top = row_ys[0] + 0.45
    server_rect = mpatches.FancyBboxPatch(
        (_BOX_L, box_bot),
        _BOX_R - _BOX_L,
        box_top - box_bot,
        boxstyle="round,pad=0.06",
        linewidth=1.1,
        edgecolor=style.GREY_ANNOT,
        # alpha is baked into the face colour; a patch-level alpha would fade the edge too and leave the block without an outline
        facecolor=mcolors.to_rgba(style.PALETTE[0], 0.10),
        zorder=2,
    )
    ax.add_patch(server_rect)

    box_cx = (_BOX_L + _BOX_R) / 2
    box_cy = (box_top + box_bot) / 2
    ax.text(
        box_cx, box_cy + 0.42,
        "Full-Availability\nGroup",
        ha="center", va="center",
        fontsize=style.FS_TITLE, zorder=3,
    )
    ax.text(
        box_cx, box_cy - 0.62,
        r"capacity $V$ AU",
        ha="center", va="center",
        fontsize=style.FS_AXIS, zorder=3,
    )

    for i, cls in enumerate(_CLASSES):
        y = row_ys[i]

        ax.text(
            _LEFT_X, y,
            f"{i + 1}. {cls}",
            ha="left", va="center",
            fontsize=style.FS_AXIS,
        )
        ax.text(
            _LEFT_X, y - _SUBLABEL_DROP,
            rf"$a_{i + 1}$ Erl, $t_{i + 1}$ AU",
            ha="left", va="center",
            fontsize=style.FS_ANNOT, color=style.GREY_ANNOT,
        )

        _draw_arrow(ax, _ARROW_IN_0, _ARROW_IN_1, y)
        _draw_arrow(ax, _ARROW_OUT_0, _ARROW_OUT_1, y)

        ax.text(
            _RIGHT_X, y,
            rf"$B_{i + 1}$",
            ha="left", va="center",
            fontsize=style.FS_AXIS,
        )

    fig.tight_layout(pad=0.3)
    style.savefig(fig, "02_fag_diagram", width="med")
    plt.close(fig)


if __name__ == "__main__":
    main()
