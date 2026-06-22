"""Create the derandomisation stability plots for Figure B2 or Figure B3.

The script reads method-selection matrices from
``results/<scenario>/<trial>/result_matrix_iter.txt``. It reconstructs the
ODGM majority-vote selection at each perturbation realisation and plots the
cumulative averages of the number of rejections and the false discovery
proportion.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Some macOS environments may require it because of an OpenMP runtime conflict.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass(frozen=True)
class FigureConfig:
    """Input locations and display settings for one derandomisation figure."""

    scenario_folder: str
    output_file: Path
    trial_folder: str = "1_trial"
    result_root: Path = Path("results")
    show_figure: bool = False


# Select the supplementary figure to create: "FigureB2" for rho = 0 or
# "FigureB3" for rho = 0.5.
FIGURE_TO_CREATE = "FigureB2"

FIGURE_CONFIGS = {
    "FigureB2": FigureConfig(
        scenario_folder="n800_p200_p150_A0.5_rho0_k12",
        output_file=Path("figures/supplementary/FigureB2.pdf"),
    ),
    "FigureB3": FigureConfig(
        scenario_folder="n800_p200_p150_A0.5_rho0.5_k12",
        output_file=Path("figures/supplementary/FigureB3.pdf"),
    ),
}

# Methods included in the ODGM majority vote.
VOTING_ALGORITHM_NAMES = [
    "AGM_A_1",
    "AGM_A_2",
    "AGM_A_3",
    "AGM_P_1",
    "AGM_P_2",
    "AGM_P_3",
    "ATZ_A_1",
    "ATZ_A_2",
    "ATZ_A_3",
    "ATZ_P_1",
    "ATZ_P_2",
    "ATZ_P_3",
    "HATZ_A_1",
    "HATZ_A_2",
    "HATZ_A_3",
    "HATZ_P_1",
    "HATZ_P_2",
    "HATZ_P_3",
    "SEC_S",
]

# Support result files created with earlier method labels.
METHOD_NAME_ALIASES = {
    "SEC_S": "SEC_S",
    "ODGM": "ODGM",
}


def normalise_method_name(name: str) -> str:
    """Map legacy method names to the current repository names."""
    return METHOD_NAME_ALIASES.get(name, name)


def get_input_paths(config: FigureConfig) -> tuple[Path, Path]:
    """Return the selection-matrix and null-index file paths."""
    scenario_path = config.result_root / config.scenario_folder
    trial_path = scenario_path / config.trial_folder
    return trial_path / "result_matrix_iter.txt", scenario_path / "S0.pt"


def read_selection_matrices(file_path: Path) -> tuple[list[str], list[np.ndarray]]:
    """Read all saved method-selection matrices from one trial."""
    if not file_path.exists():
        raise FileNotFoundError(f"Selection-matrix file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError(f"Selection-matrix file is empty: {file_path}")

    header_names = [normalise_method_name(name) for name in lines[0].split("\t")]
    algorithm_names = ["GM", *header_names]

    matrices = []
    current_rows: list[list[float]] = []

    for line in lines[1:]:
        if line.startswith("Iteration") or line.startswith("第"):
            if current_rows:
                matrices.append(np.asarray(current_rows, dtype=float))
                current_rows = []
            continue

        # Older result files may include these lines after the method header.
        if line.startswith("S0:") or line.startswith("S1:"):
            continue

        current_rows.append([float(value) for value in line.split()])

    if current_rows:
        matrices.append(np.asarray(current_rows, dtype=float))

    if not matrices:
        raise ValueError(f"No selection matrices were found in {file_path}")

    expected_rows = len(algorithm_names)
    for iteration, matrix in enumerate(matrices, start=1):
        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"Iteration {iteration} has {matrix.shape[0]} rows, but the header "
                f"defines {expected_rows} methods in {file_path}."
            )

    return algorithm_names, matrices


def reconstruct_odgm_selections(
    algorithm_names: list[str],
    matrices: list[np.ndarray],
) -> np.ndarray:
    """Apply ODGM majority voting to every saved selection matrix."""
    voting_indices = []
    for method_name in VOTING_ALGORITHM_NAMES:
        if method_name not in algorithm_names:
            raise ValueError(
                f"Voting method '{method_name}' is absent from the selection-matrix "
                "header. Check that the selected scenario uses the ODGM library "
                "specified in this script."
            )
        voting_indices.append(algorithm_names.index(method_name))

    vote_threshold = len(voting_indices) / 2
    selections = [
        matrix[voting_indices].sum(axis=0) > vote_threshold for matrix in matrices
    ]
    return np.asarray(selections, dtype=bool)


def compute_cumulative_metrics(
    odgm_selections: np.ndarray,
    s0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative average rejection size and FDP across iterations."""
    if odgm_selections.ndim != 2:
        raise ValueError("ODGM selections must be a two-dimensional Boolean array.")

    s0_mask = np.zeros(odgm_selections.shape[1], dtype=bool)
    s0_mask[np.asarray(s0, dtype=int)] = True

    rejections_per_iteration = odgm_selections.sum(axis=1)
    false_discoveries = (odgm_selections & s0_mask).sum(axis=1)
    fdp_per_iteration = np.divide(
        false_discoveries,
        rejections_per_iteration,
        out=np.zeros_like(false_discoveries, dtype=float),
        where=rejections_per_iteration > 0,
    )

    iteration_numbers = np.arange(1, odgm_selections.shape[0] + 1)
    cumulative_rejections = np.cumsum(rejections_per_iteration) / iteration_numbers
    cumulative_fdp = np.cumsum(fdp_per_iteration) / iteration_numbers
    return iteration_numbers, cumulative_rejections, cumulative_fdp


def create_figure(config: FigureConfig) -> None:
    """Create and save one derandomisation stability figure."""
    matrix_file, s0_file = get_input_paths(config)
    algorithm_names, matrices = read_selection_matrices(matrix_file)

    if not s0_file.exists():
        raise FileNotFoundError(
            f"Null-index file not found: {s0_file}. "
            "The simulation script should save S0.pt in the scenario folder."
        )
    s0 = torch.load(s0_file, map_location="cpu", weights_only=True).numpy()

    odgm_selections = reconstruct_odgm_selections(algorithm_names, matrices)
    iterations, cumulative_rejections, cumulative_fdp = compute_cumulative_metrics(
        odgm_selections,
        s0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    axes[0].plot(
        iterations,
        cumulative_rejections,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="ODGM",
    )
    axes[0].set_xlabel("Number of perturbation realisations")
    axes[0].set_ylabel("Average number of rejections")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(frameon=False)

    axes[1].plot(
        iterations,
        cumulative_fdp,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="ODGM",
    )
    axes[1].set_xlabel("Number of perturbation realisations")
    axes[1].set_ylabel("Average FDP")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(frameon=False)

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
