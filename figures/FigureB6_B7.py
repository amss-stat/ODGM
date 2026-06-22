"""Create the k1-sensitivity boxplots for Figure B6 or Figure B7.

The script combines:
1. the original main_simu.py summary, which provides k1 = 2; and
2. the change_k1_new.py summary, which provides k1 = 0.5 and k1 = 1
   using the same saved data and injected-noise realisation.

It plots the six A- and P-calibrated procedures across k1 = 0.5, 1, and 2.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class FigureConfig:
    """Input and output settings for one k1-sensitivity figure."""

    base_results_file: Path
    sensitivity_results_file: Path
    output_file: Path
    show_figure: bool = False


# Select which supplementary figure to create: "FigureB6" or "FigureB7".
FIGURE_TO_CREATE = "FigureB6"

FIGURE_CONFIGS = {
    "FigureB6": FigureConfig(
        base_results_file=Path(
            "results/n800_p200_p150_A0.5_rho0_k12/Alltrials_FDP_Power.txt"
        ),
        sensitivity_results_file=Path(
            "results/k1_sensitivity_n800_p200_p150_A0.5_rho0/" "Alltrials_FDP_Power.txt"
        ),
        output_file=Path("figures/supplementary/FigureB6.pdf"),
    ),
    "FigureB7": FigureConfig(
        base_results_file=Path(
            "results/n800_p200_p150_A0.5_rho0.5_k12/Alltrials_FDP_Power.txt"
        ),
        sensitivity_results_file=Path(
            "results/k1_sensitivity_n800_p200_p150_A0.5_rho0.5/"
            "Alltrials_FDP_Power.txt"
        ),
        output_file=Path("figures/supplementary/FigureB7.pdf"),
    ),
}

ALGORITHMS = [
    "AGM_A",
    "ATZ_A",
    "HATZ_A",
    "AGM_P",
    "ATZ_P",
    "HATZ_P",
]

DISPLAY_NAMES = {
    "AGM_A": r"$\mathrm{AGM\!-\!A}$",
    "ATZ_A": r"$\mathrm{ATZ\!-\!A}$",
    "HATZ_A": r"$\mathrm{HATZ\!-\!A}$",
    "AGM_P": r"$\mathrm{AGM\!-\!P}$",
    "ATZ_P": r"$\mathrm{ATZ\!-\!P}$",
    "HATZ_P": r"$\mathrm{HATZ\!-\!P}$",
}

METHOD_COLORS = [
    "#b46d94",
    "#d995b9",
    "#fad6e9",
    "#ade3db",
    "#66c2ba",
    "#248D84E4",
]

K1_LABELS = ["0.5", "1", "2"]


def parse_metric_line(line: str) -> dict[str, float]:
    """Parse one logged dictionary of method-specific values."""
    return {
        method.strip(): float(value)
        for method, value in re.findall(
            r"([^,{]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line,
        )
    }


def read_trial_metrics(file_path: Path, metric_name: str) -> list[dict[str, float]]:
    """Read all trial-level records for one metric from one result file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Simulation result file not found: {file_path}")

    records = []
    current_metric = None

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "FDP:":
                current_metric = "FDP"
                continue
            if line == "Power:":
                current_metric = "Power"
                continue
            if current_metric == metric_name and line.startswith("{"):
                records.append(parse_metric_line(line))
                current_metric = None

    if not records:
        raise ValueError(f"No {metric_name} records were found in {file_path}.")
    return records


def extract_method_values(
    records: list[dict[str, float]],
    method_names: list[str],
    file_path: Path,
    metric_name: str,
) -> list[list[float]]:
    """Extract one list of trial-level values for each requested method name."""
    values_by_method = [[] for _ in method_names]

    for trial_index, record in enumerate(records, start=1):
        for index, method_name in enumerate(method_names):
            if method_name not in record:
                raise ValueError(
                    f"Method '{method_name}' is missing from the {metric_name} "
                    f"record for Trial {trial_index} in {file_path}."
                )
            values_by_method[index].append(record[method_name])

    return values_by_method


def collect_metric_data(
    config: FigureConfig,
    metric_name: str,
) -> dict[str, list[list[float]]]:
    """Collect values for k1 = 0.5, 1, and 2 for all six procedures."""
    sensitivity_records = read_trial_metrics(
        config.sensitivity_results_file,
        metric_name,
    )
    base_records = read_trial_metrics(config.base_results_file, metric_name)

    metric_data = {}
    for algorithm in ALGORITHMS:
        k1_half, k1_one = extract_method_values(
            sensitivity_records,
            [f"{algorithm}_2_1", f"{algorithm}_2_2"],
            config.sensitivity_results_file,
            metric_name,
        )
        (k1_two,) = extract_method_values(
            base_records,
            [f"{algorithm}_2"],
            config.base_results_file,
            metric_name,
        )
        metric_data[algorithm] = [k1_half, k1_one, k1_two]

    return metric_data


def draw_grouped_boxplots(
    ax,
    metric_data: dict[str, list[list[float]]],
    y_label: str,
) -> None:
    """Draw grouped boxplots for k1 = 0.5, 1, and 2."""
    box_width = 0.12
    group_spacing = 1.5
    positions = []
    data_to_plot = []

    for k1_index in range(len(K1_LABELS)):
        for method_index, method in enumerate(ALGORITHMS):
            positions.append(
                k1_index * len(ALGORITHMS) * box_width * group_spacing
                + method_index * box_width
            )
            data_to_plot.append(metric_data[method][k1_index])

    boxplot = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    )

    for index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(METHOD_COLORS[index % len(ALGORITHMS)])
        box.set_edgecolor("black")
        box.set_linewidth(0.8)

    for median in boxplot["medians"]:
        median.set_color("red")
        median.set_linewidth(1.0)

    for whisker in boxplot["whiskers"]:
        whisker.set_linewidth(0.8)
    for cap in boxplot["caps"]:
        cap.set_linewidth(0.8)

    centers = [
        k1_index * len(ALGORITHMS) * box_width * group_spacing
        + (len(ALGORITHMS) - 1) * box_width / 2
        for k1_index in range(len(K1_LABELS))
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels(K1_LABELS, fontsize=12)
    ax.set_xlabel(r"$k_1$", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="y", labelsize=10)


def create_figure(config: FigureConfig) -> None:
    """Create and save one k1-sensitivity boxplot figure."""
    fdp_data = collect_metric_data(config, "FDP")
    power_data = collect_metric_data(config, "Power")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    draw_grouped_boxplots(axes[0], fdp_data, "FDP")
    draw_grouped_boxplots(axes[1], power_data, "Proportion of true rejections")

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=DISPLAY_NAMES[method])
        for method, color in zip(ALGORITHMS, METHOD_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=12,
        handlelength=1.4,
        handletextpad=0.55,
        labelspacing=0.7,
    )

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.16, top=0.96, wspace=0.28)
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_file, format="pdf", bbox_inches="tight")
    print(f"Loaded k1=2 results from: {config.base_results_file}")
    print(f"Loaded k1=0.5 and 1 results from: {config.sensitivity_results_file}")
    print(f"Saved figure to: {config.output_file}")

    if config.show_figure:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if FIGURE_TO_CREATE not in FIGURE_CONFIGS:
        raise ValueError(
            f"FIGURE_TO_CREATE must be one of {list(FIGURE_CONFIGS)}, "
            f"but received '{FIGURE_TO_CREATE}'."
        )

    create_figure(FIGURE_CONFIGS[FIGURE_TO_CREATE])
