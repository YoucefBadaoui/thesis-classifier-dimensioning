"""Canonical figure style: colours, point sizes, geometry, savefig paths.

Every figure script and notebook calls ``apply()`` and writes through ``savefig()``, so output is consistent across entry points. PALETTE indices carry fixed roles across figures. Unit notation is ``AU·Erl`` (U+00B7) in matplotlib text. Figures are written to the figure directory only: figures/ in the code repository, PUT-MSc-Thesis/figures/ when the manuscript tree is present.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.analytical.constants import CLASS_ORDER_OTT as _CLASS_ORDER_OTT
from src.cesnet.tiers import GROUP_REC, TIER_AU

# Okabe-Ito eight-colour categorical palette, CVD-safe (Okabe & Ito 2002).
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#000000",  # black
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#F0E442",  # yellow
]

# Role-based aliases (used inside the rendering modules).
COLOR_UNIFORM = PALETTE[0]
COLOR_WORST = PALETTE[1]
COLOR_BEST = PALETTE[2]
COLOR_ANALYTICAL = PALETTE[3]
COLOR_SIMULATION = PALETTE[4]
COLOR_MLPV = PALETTE[5]

# Per-class palette. Excludes black, which is reserved for the analytical reference line drawn on the same axes. Seven hues cover the five OTT/IPTV classes, the three 5G slices, and the six CESNET tiers.
CLASS_COLORS: list[str] = [
    PALETTE[0],
    PALETTE[1],
    PALETTE[2],
    PALETTE[4],
    PALETTE[5],
    PALETTE[6],
    PALETTE[7],
]

# Named greys used by every figure module.
GREY_ANNOT = "#444444"  # annotation and data-label text
GREY_RULE = "#888888"   # reference rules, leader lines, spans
GREY_MUTED = "#BBBBBB"  # de-emphasised series, excluded categories

# Diverging colormap for signed sensitivity tensors; positive inflates blocking, negative deflates it.
DIVERGING = "BrBG"

# Sequential colormap for row-stochastic CMs.
SEQUENTIAL = "Blues"

# Canonical class orderings.
CLASS_ORDER_OTT: list[str] = list(_CLASS_ORDER_OTT)


def class_short(name: str) -> str:
    """Map a full class name to its short label used in heatmap ticks."""
    return {"Browsing": "Brws", "FileTransfer": "FT",
            "Streaming": "Strm"}.get(name, name)


CLASS_ORDER_OTT_SHORT: list[str] = [class_short(c) for c in CLASS_ORDER_OTT]

# Allocation units per CESNET tier, index i is tier i. Matches the leading axis of ``cesnet_dimension.npz['rec_tier_cm']`` and ``rec_t``.
CESNET_TIER_AU: list[int] = TIER_AU

# Category to tier index. The 15 foreground categories map to tiers 0..5; the 8 background categories (38.9% of flow mass) map to -1 and are excluded from the dimensioning object; the mapping is GROUP_REC of src/cesnet/tiers.py.
CESNET_CATEGORY_TO_TIER: dict[str, int] = dict(GROUP_REC)

# Canonical unit notation.
UNIT_AU_ERL_LATEX = r"AU$\cdot$Erl"  # for matplotlib mathtext

# Geometry: \textwidth is 426.0pt in ppfcmthesis at a4paper. Figures are authored at their on-page width so \includegraphics never rescales them and a nominal 10pt label renders at 10pt.
TEXTWIDTH_PT = 426.0
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.0  # 5.917 in

# Canonical widths. The key is the \includegraphics fraction the .tex source must use.
WIDTH_FRACTIONS: dict[str, float] = {
    "full": 1.00,
    "wide": 0.95,
    "med": 0.85,
    "narrow": 0.70,
}

# Tallest authored height that leaves room for a caption on one page.
MAX_HEIGHT_IN = 7.60


def width_in(token: str) -> float:
    """Authored width in inches for a canonical width token."""
    return TEXTWIDTH_IN * WIDTH_FRACTIONS[token]


def size(token: str, height_in: float) -> tuple[float, float]:
    """Return ``(width, height)`` in inches for a canonical width token."""
    return (width_in(token), height_in)


# Lexicon: one spelling per cross-figure wording choice.
VERSUS = "versus"

CM_AXIS_TRUE = "true class"
CM_AXIS_PRED = "predicted class"
CM_AXIS_TRUE_TIER = "true tier"
CM_AXIS_PRED_TIER = "predicted tier"

# A row-stochastic CM encodes P(predicted | true); only the diagonal is recall.
CM_CBAR_LABEL = "row-normalised rate (%)"

SCENARIO_OTT = "OTT/IPTV"
SCENARIO_5G = "5G slicing"

# Permitted point sizes.
FS_TITLE = 11.0
FS_AXIS = 10.0
FS_TICK = 9.0
FS_LEGEND = 9.0
FS_ANNOT = 8.0

# Repository roots.
REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_FIG_DIR = (REPO_ROOT / "PUT-MSc-Thesis" / "figures"
                  if (REPO_ROOT / "PUT-MSc-Thesis").is_dir()
                  else REPO_ROOT / "figures")
PROCESSED = REPO_ROOT / "data" / "processed"

# Marker area shared by the r*-family scatter figures.
MARKER_SIZE = 55


def apply() -> None:
    """Install the canonical matplotlib rcParams."""
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "font.family": "sans-serif",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": ":",
        "grid.alpha": 0.45,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.6,
        "patch.linewidth": 0.6,
    })


def savefig(fig: plt.Figure, stem: str, width: str) -> Path:
    """Write ``<stem>.png`` and ``<stem>.pdf`` into the figure directory.

    Returns the PDF path. The tight bounding box is widened symmetrically to the canonical ``width``, so the written width does not depend on label lengths; vertical extent stays tight.
    """
    from matplotlib.transforms import Bbox

    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = THESIS_FIG_DIR / f"{stem}.png"
    pdf_path = THESIS_FIG_DIR / f"{stem}.pdf"

    target = width_in(width)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    # match the pad bbox="tight" would apply, so vertical framing is unchanged from the default path
    pad = mpl.rcParams.get("savefig.pad_inches", 0.1)
    x0, y0, x1, y1 = tight.x0, tight.y0 - pad, tight.x1, tight.y1 + pad
    grow = (target - (x1 - x0)) / 2.0
    bbox = Bbox.from_extents(x0 - grow, y0, x1 + grow, y1)

    fig.savefig(png_path, bbox_inches=bbox)
    fig.savefig(pdf_path, bbox_inches=bbox)
    print(f"Saved: {pdf_path}")
    return pdf_path


def load_npz(path: Path, hint: str, *, allow_pickle: bool = False):
    """Open an NPZ archive, naming the script that regenerates it if absent."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run {hint}")
    return np.load(path, allow_pickle=allow_pickle)


R_FLOOR = 0.50  # lower end of the recall search; a class at the floor never binds


def rstar_markers(ax, x, rstar) -> None:
    """Filled markers for binding thresholds, hollow markers on the search floor; r* on y."""
    x = np.asarray(x, dtype=float)
    r = np.asarray(rstar, dtype=float)
    on_floor = np.isclose(r, R_FLOOR)
    ax.scatter(x[~on_floor], r[~on_floor], s=MARKER_SIZE, color=CLASS_COLORS[0],
               edgecolor="white", linewidth=0.6, zorder=4,
               label="binding threshold")
    ax.scatter(x[on_floor], r[on_floor], s=MARKER_SIZE, facecolor="white",
               edgecolor=CLASS_COLORS[0], linewidth=1.2, zorder=4,
               label=f"at the lower bound $r={R_FLOOR:.2f}$")



# Short tick labels for the 23 CESNET categories; full names overflow a 23-tick axis at textwidth. Order follows ``cesnet_definitive.npz['category_names']``.
_CESNET_SHORT: dict[str, str] = {
    "Advertising": "Advert", "Analytics & Telemetry": "Analytics",
    "Antivirus": "Antivir", "Authentication": "Auth",
    "File sharing": "FileShr", "Games": "Games",
    "Information systems": "InfoSys", "Instant messaging": "IM",
    "Internet banking": "Banking", "Location": "Locat",
    "Mail": "Mail", "Media": "Media", "Music": "Music",
    "Notifications": "Notif", "Other APIs": "OthAPI",
    "Other services": "OthSvc", "Remote desktop": "RemDesk",
    "Search": "Search", "Social": "Social",
    "Software updates": "SwUpd", "Videoconferencing": "VidConf",
    "Virtual assistant": "VirtAsst", "Weather": "Weather",
}


def cesnet_class_short(name: str) -> str:
    """Map a full CESNET category name to its short heatmap tick label."""
    return _CESNET_SHORT.get(name, name)
