"""Create the seven-panel UpSet plot for Figure 6.

Required input
--------------
Run the real-data analysis for APV, ATV, IDV, LPV, NFV, RTV, and SQV first.
For each drug, this script expects:

    results/real_data/<drug>/algorithm_rejections.txt

Each line in that file must have the form:

    METHOD_NAME: index_1 index_2 ...

where METHOD_NAME is one of the internal names listed in ``METHOD_DISPLAY_NAMES``.

Before running
--------------
1. Set ``RESULTS_DIR`` if your real-data output directory differs.
2. Confirm that every drug folder contains ``algorithm_rejections.txt``.
3. Run from the repository root:

       python figures/Figure6.py
"""

from pathlib import Path
import re
from types import MethodType

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from upsetplot import UpSet, from_contents

RESULTS_DIR = Path("results/real_data")
OUTPUT_FILE = Path("figures/Figure6.pdf")
SHOW_FIGURE = False

DATASETS = ("APV", "ATV", "IDV", "LPV", "NFV", "RTV", "SQV")

# Internal output names from main_realdata.py and their paper display names.
METHOD_DISPLAY_NAMES = {
    "GM": "GM",
    "ATZ_P_1": "ATZ-P(0.5)",
    "ATZ_P_2": "ATZ-P(1)",
    "ATZ_P_3": "ATZ-P(2)",
    "AGM_P_2": "AGM-P(1)",
    "HATZ_P_2": "HATZ-P(1)",
    "ATZ_A_2": "ATZ-A(1)",
    "SEC_S": "SEC-S",
    "ODGM": "ODGM",
}

# Backward-compatible names in older real-data result files.
METHOD_NAME_ALIASES = {
    "SEC2": "SEC_S",
    "EDGM": "ODGM",
    "new": "ODGM",
}

METHOD_ORDER = tuple(METHOD_DISPLAY_NAMES)

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def normalise_method_name(name: str) -> str:
    """Map legacy method names to the current internal names."""
    return METHOD_NAME_ALIASES.get(name, name)


def read_algorithm_rejections(dataset: str) -> dict[str, set[int]]:
    """Read final selected feature indices for one HIV-1 drug data set."""
    input_file = RESULTS_DIR / dataset / "algorithm_rejections.txt"
    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing {input_file}. "
            "Create this final-selection file with main_realdata.py before plotting."
        )

    selected_sets: dict[str, set[int]] = {}
    with input_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            raw_name, raw_indices = line.split(":", 1)
            method_name = normalise_method_name(raw_name.strip())
            if method_name not in METHOD_DISPLAY_NAMES:
                continue

            indices = {int(value) for value in re.findall(r"-?\d+", raw_indices)}
            selected_sets[method_name] = indices

    missing_methods = [name for name in METHOD_ORDER if name not in selected_sets]
    if missing_methods:
        raise ValueError(
            f"{input_file} is missing final selections for: {missing_methods}."
        )

    return selected_sets


def prepare_upset_data(selected_sets: dict[str, set[int]]):
    """Convert final selected feature sets to the format used by upsetplot."""
    contents = {
        METHOD_DISPLAY_NAMES[method]: selected_sets[method] for method in METHOD_ORDER
    }
    return from_contents(contents)


def make_subfigure_compatible(subfigure, figure, n_columns: int = 4, n_rows: int = 2):
    """Provide size methods required by upsetplot on older Matplotlib versions."""

    def get_width(_self):
        width = subfigure.bbox.width / figure.dpi
        return width if width > 0 else figure.get_figwidth() / n_columns

    def get_height(_self):
        height = subfigure.bbox.height / figure.dpi
        return height if height > 0 else figure.get_figheight() / n_rows

    if not hasattr(subfigure, "get_figwidth"):
        subfigure.get_figwidth = MethodType(get_width, subfigure)
    if not hasattr(subfigure, "get_figheight"):
        subfigure.get_figheight = MethodType(get_height, subfigure)
    if not hasattr(subfigure, "get_size_inches"):
        subfigure.get_size_inches = lambda: np.array(
            [subfigure.get_figwidth(), subfigure.get_figheight()]
        )
    if not hasattr(subfigure, "set_figwidth"):
        subfigure.set_figwidth = lambda *args, **kwargs: None
    if not hasattr(subfigure, "set_figheight"):
        subfigure.set_figheight = lambda *args, **kwargs: None
    if not hasattr(subfigure, "set_size_inches"):
        subfigure.set_size_inches = lambda *args, **kwargs: None


def style_upset_axes(axes: dict) -> None:
    """Apply a compact, consistent style to one UpSet plot."""
    if "totals" in axes:
        for patch in axes["totals"].patches:
            patch.set_facecolor("#8B1A1A")
            patch.set_edgecolor("#8B1A1A")

    for axis in axes.values():
        axis.tick_params(labelsize=5)
        axis.xaxis.label.set_size(5)
        axis.yaxis.label.set_size(5)
        axis.xaxis.get_offset_text().set_fontsize(5)
        axis.yaxis.get_offset_text().set_fontsize(5)
        for text in axis.texts:
            text.set_fontsize(5)


def add_dataset_label(figure, subfigure, dataset: str, y_offset: float = 0.018) -> None:
    """Place the drug name beneath one UpSet subfigure."""
    figure.canvas.draw()
    bbox = subfigure.bbox.transformed(figure.transFigure.inverted())
    figure.text(
        (bbox.x0 + bbox.x1) / 2,
        bbox.y0 - y_offset,
        dataset,
        ha="center",
        va="top",
        fontsize=16,
        family="serif",
    )


def create_figure() -> None:
    """Create and save the combined Figure 6 UpSet plot."""
    figure = plt.figure(figsize=(20, 10))
    grid = figure.add_gridspec(
        2,
        4,
        left=0.025,
        right=0.995,
        bottom=0.085,
        top=0.990,
        wspace=0.12,
        hspace=0.42,
    )
    subfigures = [
        figure.add_subfigure(grid[row, column])
        for row in range(2)
        for column in range(4)
    ]

    figure.canvas.draw()

    for subfigure, dataset in zip(subfigures, DATASETS):
        make_subfigure_compatible(subfigure, figure)
        upset = UpSet(
            prepare_upset_data(read_algorithm_rejections(dataset)),
            subset_size="count",
            show_counts=True,
            sort_by=None,
            sort_categories_by=None,
        )
        axes = upset.plot(fig=subfigure)
        style_upset_axes(axes)
        add_dataset_label(figure, subfigure, dataset)

    subfigures[-1].set_visible(False)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FILE, format="pdf", bbox_inches="tight")
    print(f"Saved Figure 6 to: {OUTPUT_FILE}")

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    create_figure()
