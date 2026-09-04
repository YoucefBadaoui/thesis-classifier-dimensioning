"""Chapter 2, FlowPic confusion matrices (Shapira & Shavitt 2021).

Three published CMs: non-VPN (5 classes), VPN (4 classes, no Browsing), Tor (5 classes). Source ``data/processed/confusion_matrices.npz``, keys ``flowpic_nonvpn``, ``flowpic_vpn`` (5x5, row 0 zero), ``flowpic_tor``. Class order is alphabetical (style.CLASS_ORDER_OTT).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.figures import style
from src.figures.ch04_cm_heatmaps import add_cm_colourbar, draw_cm

style.apply()

_NPZ_PATH = style.PROCESSED / "confusion_matrices.npz"

_CLASSES_5 = style.CLASS_ORDER_OTT
_SHORT_5   = [style.class_short(c) for c in _CLASSES_5]

_VPN_INDICES = [1, 2, 3, 4]   # Chat, FileTransfer, Streaming, VoIP
_CLASSES_VPN = [_CLASSES_5[i] for i in _VPN_INDICES]
_SHORT_VPN   = [style.class_short(c) for c in _CLASSES_VPN]

# cell pitch is held to the widest matrix so one cell is the same size in all three panels
_N_SLOTS = len(_CLASSES_5)

_PANEL_TITLES = [
    "(a) non-VPN (5-class)",
    "(b) VPN (4-class)",
    "(c) Tor (5-class)",
]


def main() -> None:
    npz = np.load(_NPZ_PATH)

    raw_nonvpn = npz["flowpic_nonvpn"].astype(float)
    raw_vpn    = npz["flowpic_vpn"].astype(float)       # (5,5), row 0 = 0
    raw_tor    = npz["flowpic_tor"].astype(float)

    cm_vpn = raw_vpn[np.ix_(_VPN_INDICES, _VPN_INDICES)]

    panels = [
        (raw_nonvpn, _SHORT_5,   _PANEL_TITLES[0]),
        (cm_vpn,     _SHORT_VPN, _PANEL_TITLES[1]),
        (raw_tor,    _SHORT_5,   _PANEL_TITLES[2]),
    ]

    fig = plt.figure(figsize=style.size("full", 5.35))
    gs = GridSpec(
        2, 2,
        figure=fig,
        hspace=0.62,
        wspace=0.28,
        left=0.09,
        right=0.98,
        top=0.93,
        bottom=0.09,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
    ]

    for ax, (cm, labels, title) in zip(axes, panels):
        im = draw_cm(
            ax, cm, labels,
            title=title,
            xlabel=style.CM_AXIS_PRED,
            ylabel=style.CM_AXIS_TRUE,
            n_slots=_N_SLOTS,
        )

    # the fourth quadrant carries the shared colourbar
    cbar_host = fig.add_subplot(gs[1, 1])
    cbar_host.axis("off")
    cbar_ax = cbar_host.inset_axes((0.08, 0.46, 0.84, 0.09))
    add_cm_colourbar(fig, im, cax=cbar_ax, orientation="horizontal")

    style.savefig(fig, "02_flowpic_cms", width="full")
    plt.close(fig)


if __name__ == "__main__":
    main()
