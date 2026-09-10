"""Single entry point for regenerating every thesis figure.

Renders every thesis figure into the figure directory (figures/ in the code repository, PUT-MSc-Thesis/figures/ beside the manuscript) through the modules under src/figures/, as PNG at 300 DPI and PDF. Notebooks call the same modules, so the rendered artefact does not depend on the entry point.
"""

import importlib
import sys
from pathlib import Path

# put the project root on the import path when run as a script
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.figures import style

# one list, in render order; the module name is also the import name
_NAMES = [
    "ch02_fag_diagram",
    "ch02_flowpic_cms",
    "ch03_overhead_curves",
    "ch03_delta_b_bar",
    "ch03_rstar_scatter",
    "ch03_highk_power",
    "ch03_sensitivity_heatmaps",
    "ch04_cm_heatmaps",
    "ch04_cesnet_class_distribution",
    "ch04_cesnet_tier_cm",
    "ch03_cesnet_f1_scatter",
    "ch06_monte_carlo",
    "ch06_cascade",
    "ch06_error_process",
]


def main() -> None:
    failures: list[tuple[str, Exception]] = []
    for name in _NAMES:
        print(f"[{name}]")
        try:
            importlib.import_module(f"src.figures.{name}").main()
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAILED: {exc}")
    print()
    if failures:
        print(f"FAILED: {len(failures)} module(s): "
              + ", ".join(n for n, _ in failures))
        sys.exit(1)
    print(f"DONE: {len(_NAMES)} modules rendered to {style.THESIS_FIG_DIR}")


if __name__ == "__main__":
    main()
