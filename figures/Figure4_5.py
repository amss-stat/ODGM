"""Create the simulation boxplots for Figure 4 or Figure 5.

The script reads trial-level FDP and power values from
``results/n{n}_p{p}_p1{p1}_A{a}_rho{rho}_k1{par}/Alltrials_FDP_Power.txt``.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureConfig:
    """Inputs and display settings for one simulation boxplot figure."""

    parameter_values: tuple[float, ...]
    fixed_a: float
    fixed_rho: float
    x_label: str
    output_file: Path
    fdp_upper_limit: float
    n: int = 800
    p: int = 200
    p1: int = 50
    par: int = 2
    result_root: Path = Path("results")
    box_width: float = 0.15
    show_figure: bool = False


# Select which paper figure to create: "Figure4" or "Figure5".
FIGURE_TO_CREATE = "Figure4"

FIGURE_CONFIGS = {
    "Figure4": FigureConfig(
        parameter_values=(0.3, 0.4, 0.5, 0.6, 0.7),
        fixed_a=0.5,
        fixed_rho=0.5,
        x_label="signal strength",
        output_file=Path("figures/Figure4.pdf"),
        fdp_upper_limit=0.51,
    ),
    "Figure5": FigureConfig(
        parameter_values=(0.0, 0.2, 0.4, 0.6, 0.8),
        fixed_a=0.5,
        fixed_rho=0.5,
        x_label="correlation coefficient",
        output_file=Path("figures/Figure5.pdf"),
        fdp_upper_limit=0.55,
    ),
}

# Methods displayed in the paper figures.
# ATZ-A(1) is retained for continuity with the reported results.
ALGORITHMS_TO_PLOT = [
    "GM",
    "ATZ_P_1",
    "ATZ_P_2",
    "ATZ_P_3",
    "AGM_P_2",
    "HATZ_P_2",
    "ATZ_A_2",
    "SEC_S",
    "ODGM",
]

DISPLAY_NAMES = {
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

METHOD_COLORS = [
    "#f87069",
    "#f49679",
    "#f8ba7c",
    "#f6d886",
    "#f5f5ae",
    "#d0f0f9",
    "#a5def1",
    "#74add1",
    "#4575b4",
]


def format_value(value: float | int) -> str:
    """Format a parameter value exactly as used in simulation folder names."""
    return f"{value:g}"


def get_result_folder(config: FigureConfig, parameter_value: float) -> Path:
    """Return the simulation result folder for one displayed parameter value."""
    if config.x_label == "signal strength":
        a = parameter_value
        rho = config.fixed_rho
    elif config.x_label == "correlation coefficient":
        a = config.fixed_a
        rho = parameter_value
    else:
        raise ValueError(f"Unsupported x-axis label: {config.x_label}")

    folder_name = (
        f"n{config.n}_p{config.p}_p1{config.p1}_"
        f"A{format_value(a)}_rho{format_value(rho)}_k1{format_value(config.par)}"
    )
    return config.result_root / folder_name


def parse_metric_line(line: str) -> dict[str, float]:
    """Parse one logged dictionary of method-specific values."""
    return {
        method.strip(): float(value)
        for method, value in re.findall(
            r"([^,{]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line
        )
    }


def read_trial_metrics(
    file_path: Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Read trial-level FDP and power values from one simulation result file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Simulation result file not found: {file_path}")

    fdp_data = {method: [] for method in ALGORITHMS_TO_PLOT}
    power_data = {method: [] for method in ALGORITHMS_TO_PLOT}
    current_metric = None

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "FDP:":
                current_metric = "fdp"
                continue
            if line == "Power:":
                current_metric = "power"
                continue
            if current_metric is None or not line.startswith("{"):
                continue

            values = parse_metric_line(line)
            target = fdp_data if current_metric == "fdp" else power_data
            for method in ALGORITHMS_TO_PLOT:
                if method in values:
                    target[method].append(values[method])
            current_metric = None

    for method in ALGORITHMS_TO_PLOT:
        if not fdp_data[method] or not power_data[method]:
            raise ValueError(
                f"No complete trial-level FDP/power values were found for "
                f"'{method}' in {file_path}."
            )

    return fdp_data, power_data


def load_figure_data(
    config: FigureConfig,
) -> tuple[dict[float, dict[str, list[float]]], dict[float, dict[str, list[float]]]]:
    """Load results across all parameter values for the selected figure."""
    fdp_data = {}
    power_data = {}

    for parameter_value in config.parameter_values:
        result_file = (
            get_result_folder(config, parameter_value) / "Alltrials_FDP_Power.txt"
        )
        parameter_fdp, parameter_power = read_trial_metrics(result_file)
        fdp_data[parameter_value] = parameter_fdp
        power_data[parameter_value] = parameter_power
        print(f"Loaded {config.x_label}={parameter_value} from {result_file}")

    return fdp_data, power_data


def draw_boxplots(
    ax,
    data: dict[float, dict[str, list[float]]],
    y_label: str,
    config: FigureConfig,
    apply_fdp_limits: bool = False,
) -> None:
    """Draw grouped boxplots for one performance metric."""
    parameters = list(config.parameter_values)
    data_to_plot = []
    positions = []
    group_width = len(ALGORITHMS_TO_PLOT) * config.box_width * 1.5

    for group_index, parameter_value in enumerate(parameters):
        for method_index, method in enumerate(ALGORITHMS_TO_PLOT):
            positions.append(
                group_index * group_width + method_index * config.box_width
            )
            data_to_plot.append(data[parameter_value][method])

    boxplot = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=config.box_width,
        patch_artist=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    )

    for index, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(METHOD_COLORS[index % len(ALGORITHMS_TO_PLOT)])
        box.set_linewidth(0.8)

    for median in boxplot["medians"]:
        median.set_color("red")
        median.set_linewidth(1.0)

    centers = [
        group_index * group_width + (len(ALGORITHMS_TO_PLOT) - 1) * config.box_width / 2
        for group_index in range(len(parameters))
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels([format_value(value) for value in parameters], fontsize=12)
    ax.set_xlabel(config.x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    if apply_fdp_limits:
        ax.set_ylim(-0.01, config.fdp_upper_limit)


def create_figure(config: FigureConfig) -> None:
    """Create and save the selected two-panel simulation boxplot figure."""
    fdp_data, power_data = load_figure_data(config)

    fig, (fdp_ax, power_ax) = plt.subplots(1, 2, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.15)

    draw_boxplots(fdp_ax, fdp_data, "FDP", config, apply_fdp_limits=True)
    draw_boxplots(power_ax, power_data, "Proportion of true rejections", config)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[index],
            linewidth=4,
            label=DISPLAY_NAMES[method],
        )
        for index, method in enumerate(ALGORITHMS_TO_PLOT)
    ]
    power_ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=12,
        frameon=False,
    )

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_file, format="pdf", bbox_inches="tight")
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
