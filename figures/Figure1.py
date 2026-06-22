"""Create Figure 1: geometric interpretation of the Gaussian mirror mechanism."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


@dataclass(frozen=True)
class Figure1Config:
    """Parameters controlling the Figure 1 illustration."""

    grid_limit: float = 5.0
    grid_size: int = 100
    signal_shift: float = 2.5
    cutpoint: float = 1.2
    figure_size: tuple[float, float] = (8.0, 8.0)
    output_path: str = "figures/Figure1.pdf"
    show_figure: bool = False


def main(config: Figure1Config) -> None:
    """Draw and save the Gaussian mirror geometry illustration."""
    grid = np.linspace(-config.grid_limit, config.grid_limit, config.grid_size)
    X, Y = np.meshgrid(grid, grid)
    position = np.dstack((X, Y))

    covariance = np.eye(2)
    null_density = multivariate_normal([0.0, 0.0], covariance).pdf(position)
    left_density = multivariate_normal([-config.signal_shift, 0.0], covariance).pdf(
        position
    )
    right_density = multivariate_normal([config.signal_shift, 0.0], covariance).pdf(
        position
    )

    fig, ax = plt.subplots(figsize=config.figure_size)

    rejection_region = np.abs(X) - np.abs(Y) > config.cutpoint
    calibration_region = np.abs(X) - np.abs(Y) < -config.cutpoint

    ax.contourf(
        X,
        Y,
        rejection_region.astype(float),
        levels=[0.5, 1.5],
        colors="lightblue",
        alpha=0.8,
    )
    ax.contourf(
        X,
        Y,
        calibration_region.astype(float),
        levels=[0.5, 1.5],
        colors="peachpuff",
        alpha=0.8,
    )

    ax.contour(X, Y, null_density, colors="blue", levels=4)
    ax.contour(X, Y, left_density, colors="black", levels=4)
    ax.contour(X, Y, right_density, colors="black", levels=4)

    ax.plot(
        [-config.grid_limit, config.grid_limit],
        [-config.grid_limit, config.grid_limit],
        "k--",
        alpha=0.5,
    )
    ax.plot(
        [-config.grid_limit, config.grid_limit],
        [config.grid_limit, -config.grid_limit],
        "k--",
        alpha=0.5,
    )

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-config.grid_limit, config.grid_limit)
    ax.set_ylim(-config.grid_limit, config.grid_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\hat{\gamma}_j$", loc="right", labelpad=3, fontsize=14)
    ax.text(
        0.45,
        0.97,
        r"$\hat{\eta}_j$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")

    if config.show_figure:
        plt.show()
    plt.close(fig)

    print(f"Saved Figure 1 to: {output_path}")


if __name__ == "__main__":
    main(Figure1Config())
