"""Plot the rejection and calibration regions."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Figure2Config:
    """Configuration for Figure 2."""

    k1: float = 2.0
    output_path: str = "figures/Figure2.pdf"
    rejection_xlim: tuple[float, float] = (-4.0, 4.0)
    rejection_ylim: tuple[float, float] = (-4.0, 4.0)
    calibration_xlim: tuple[float, float] = (-4.0, 4.0)
    calibration_ylim: tuple[float, float] = (-4.0, 4.0)
    grid_size: int = 600
    figure_size: tuple[float, float] = (10.5, 4.8)


def curve_agm(x, y, k1):
    """Return the AGM rejection region."""
    return k1 * np.abs(x) - np.abs(y) > 1.5


def curve_atz(x, y, k1):
    """Return the ATZ rejection region."""
    return (
        np.sign(k1 * np.abs(x) - np.abs(y)) * np.maximum(k1 * np.abs(x), np.abs(y))
        > 2.8
    )


def curve_hatz(x, y, k1):
    """Return the HATZ rejection region."""
    return np.sign(k1 * np.abs(x) - np.abs(y)) * (k1 * np.abs(x) + np.abs(y)) > 3.8


def curve_sec(x, y, k1):
    """Return the sectoral rejection region."""
    return np.sign(k1 * np.abs(x) - np.abs(y)) * (x**2 + y**2) > 4.0


def region_angular(x, y):
    """Return the angular calibration region."""
    return np.abs(y) - 2.0 * np.abs(x) > 0.5


def region_parabola(x, y):
    """Return the parabolic calibration region."""
    return np.abs(y) - x**2 > 1.0


def region_sector(x, y):
    """Return the sectoral calibration region."""
    return np.sign(np.abs(y) - np.abs(x)) * (x**2 + y**2) > 5.0


def style_axes(ax, xlim, ylim):
    """Apply a common coordinate-axis style."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    ax.text(
        0.98,
        0.51,
        r"$\tilde{\gamma}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
    )
    ax.text(
        0.53,
        0.98,
        r"$\tilde{\eta}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )


def add_panel_label(ax, label):
    """Add a panel label in the upper-left corner."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )


def plot_figure(config: Figure2Config):
    """Create Figure 2."""
    x1 = np.linspace(*config.rejection_xlim, config.grid_size)
    y1 = np.linspace(*config.rejection_ylim, config.grid_size)
    X1, Y1 = np.meshgrid(x1, y1)

    x2 = np.linspace(*config.calibration_xlim, config.grid_size)
    y2 = np.linspace(*config.calibration_ylim, config.grid_size)
    X2, Y2 = np.meshgrid(x2, y2)

    fig, axes = plt.subplots(1, 2, figsize=config.figure_size, constrained_layout=True)

    ax = axes[0]
    ax.plot(x1, config.k1 * x1, "--", color="grey", linewidth=1.0)
    ax.plot(x1, -config.k1 * x1, "--", color="grey", linewidth=1.0)

    ax.contour(X1, Y1, curve_agm(X1, Y1, config.k1), levels=[0.5], colors="red")
    ax.contour(X1, Y1, curve_atz(X1, Y1, config.k1), levels=[0.5], colors="green")
    ax.contour(X1, Y1, curve_hatz(X1, Y1, config.k1), levels=[0.5], colors="blue")
    ax.contour(
        X1,
        Y1,
        curve_sec(X1, Y1, config.k1),
        levels=[0.5],
        colors="orange",
        alpha=0.6,
    )
    style_axes(ax, config.rejection_xlim, config.rejection_ylim)
    add_panel_label(ax, "A")
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", lw=2, label="AGM"),
            Line2D([0], [0], color="green", lw=2, label="ATZ"),
            Line2D([0], [0], color="blue", lw=2, label="HATZ"),
            Line2D([0], [0], color="orange", lw=2, label="SEC"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=10,
    )

    ax = axes[1]
    ax.contour(
        X2,
        Y2,
        region_angular(X2, Y2),
        levels=[0.5],
        colors="darkorange",
    )
    ax.contour(
        X2,
        Y2,
        region_parabola(X2, Y2),
        levels=[0.5],
        colors="green",
    )
    ax.contour(
        X2,
        Y2,
        region_sector(X2, Y2),
        levels=[0.5],
        colors="blue",
        alpha=0.6,
    )
    style_axes(ax, config.calibration_xlim, config.calibration_ylim)
    add_panel_label(ax, "B")
    ax.legend(
        handles=[
            Line2D([0], [0], color="darkorange", lw=2, label="Angular"),
            Line2D([0], [0], color="green", lw=2, label="Parabola"),
            Line2D([0], [0], color="blue", lw=2, label="Sector"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=10,
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 to: {output_path}")


if __name__ == "__main__":
    plot_figure(Figure2Config())
