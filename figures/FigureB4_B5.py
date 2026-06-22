"""Create empirical-convergence plots for Figure B4 or Figure B5.

The script reads trial-level FDP and power values from
``results/n{n}_p{p}_p1{p1}_A{a}_rho{rho}_k1{par}/Alltrials_FDP_Power.txt``.
It plots their cumulative averages across simulation trials and does not
recompute selections, thresholds, or ODGM voting.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FigureConfig:
    """Inputs and display settings for one empirical-convergence figure."""

    rho: float
    output_file: Path
    fdr_upper_limit: float | None = None
    n: int = 800
    p: int = 200
    p1: int = 50
    a: float = 0.5
    par: int = 2
    result_root: Path = Path("results")
    show_figure: bool = False


# Select which supplementary figure to create: "FigureB4" or "FigureB5".
FIGURE_TO_CREATE = "FigureB4"

FIGURE_CONFIGS = {
    "FigureB4": FigureConfig(
        rho=0.0,
        output_file=Path("figures/supplementary/FigureB4.pdf"),
    ),
    "FigureB5": FigureConfig(
        rho=0.5,
        output_file=Path("figures/supplementary/FigureB5.pdf"),
    ),
}

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


def get_result_file(config: FigureConfig) -> Path:
    """Return the simulation log file for the selected correlation setting."""
    scenario_folder = (
        f"n{config.n}_p{config.p}_p1{config.p1}_"
        f"A{format_value(config.a)}_rho{format_value(config.rho)}_"
        f"k1{format_value(config.par)}"
    )
    return config.result_root / scenario_folder / "Alltrials_FDP_Power.txt"


def normalise_method_name(name: str) -> str:
    """Return method name as-is (no legacy mapping needed)."""
    return name


def parse_metric_line(line: str) -> dict[str, float]:
    """Parse one logged dictionary of method-specific performance values."""
    values = {}
    for method, value in re.findall(
        r"([^,{]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line
    ):
        values[normalise_method_name(method.strip())] = float(value)
    return values


def read_trial_metrics(
    file_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read trial-level FDP and power values for all displayed methods."""
    if not file_path.exists():
        raise FileNotFoundError(f"Simulation result file not found: {file_path}")

    fdp_values = {method: [] for method in ALGORITHMS_TO_PLOT}
    power_values = {method: [] for method in ALGORITHMS_TO_PLOT}
    trial_numbers = []
    current_metric = None
    current_trial = None

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            trial_match = re.fullmatch(r"Trial\s+(\d+):", line)
            if trial_match:
                current_trial = int(trial_match.group(1))
                trial_numbers.append(current_trial)
                current_metric = None
                continue

            if line == "FDP:":
                current_metric = "fdp"
                continue
            if line == "Power:":
                current_metric = "power"
                continue
            if (
                current_trial is None
                or current_metric is None
                or not line.startswith("{")
            ):
                continue

            values = parse_metric_line(line)
            target = fdp_values if current_metric == "fdp" else power_values
            for method in ALGORITHMS_TO_PLOT:
                if method not in values:
                    raise ValueError(
                        f"Method '{method}' is missing from the {current_metric.upper()} "
                        f"record for Trial {current_trial} in {file_path}."
                    )
                target[method].append(values[method])
            current_metric = None

    if not trial_numbers:
        raise ValueError(f"No trial-level records were found in {file_path}")

    expected_count = len(trial_numbers)
    for method in ALGORITHMS_TO_PLOT:
        if (
            len(fdp_values[method]) != expected_count
            or len(power_values[method]) != expected_count
        ):
            raise ValueError(
                f"Incomplete results for '{method}' in {file_path}: expected "
                f"{expected_count} FDP and power values, but found "
                f"{len(fdp_values[method])} and {len(power_values[method])}."
            )

    return (
        np.asarray(trial_numbers),
        {method: np.asarray(values) for method, values in fdp_values.items()},
        {method: np.asarray(values) for method, values in power_values.items()},
    )


def cumulative_means(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute the cumulative mean of each method's trial-level values."""
    return {
        method: np.cumsum(method_values) / np.arange(1, len(method_values) + 1)
        for method, method_values in values.items()
    }


def draw_metric_panel(
    ax,
    trial_numbers: np.ndarray,
    cumulative_values: dict[str, np.ndarray],
    y_label: str,
    config: FigureConfig,
    apply_fdr_limit: bool = False,
) -> None:
    """Draw one cumulative-mean performance panel."""
    for method, color in zip(ALGORITHMS_TO_PLOT, METHOD_COLORS):
        ax.plot(
            trial_numbers,
            cumulative_values[method],
            linewidth=1.7,
            color=color,
            label=DISPLAY_NAMES[method],
        )

    ax.set_xlabel("Number of simulation replications")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=10)

    if apply_fdr_limit:
        ax.set_ylim(0.0, 0.2)
    elif y_label == "Power":
        ax.set_ylim(0.5, 1.0)


def create_figure(config: FigureConfig) -> None:
    """Create and save the empirical-convergence figure for one rho setting."""
    result_file = get_result_file(config)
    trial_numbers, fdp_values, power_values = read_trial_metrics(result_file)

    cumulative_fdp = cumulative_means(fdp_values)
    cumulative_power = cumulative_means(power_values)

    fig, (fdp_ax, power_ax) = plt.subplots(1, 2, figsize=(12.0, 4.5))
    fig.subplots_adjust(wspace=0.28, right=0.79)

    draw_metric_panel(
        fdp_ax,
        trial_numbers,
        cumulative_fdp,
        "FDR",
        config,
        apply_fdr_limit=True,
    )
    draw_metric_panel(
        power_ax,
        trial_numbers,
        cumulative_power,
        "Power",
        config,
    )

    legend_handles, legend_labels = power_ax.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.81, 0.5),
        frameon=False,
        fontsize=9.5,
    )

    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_file, format="pdf", bbox_inches="tight")
    print(f"Loaded trial-level results from: {result_file}")
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
