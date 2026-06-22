"""Create Figure B1: standardised estimator pairs under two covariance settings."""

from dataclasses import dataclass
from pathlib import Path

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from nonlinshrink import shrink_cov
from torch.distributions.multivariate_normal import MultivariateNormal

# Some macOS environments may require it because of an OpenMP runtime conflict.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass(frozen=True)
class FigureB1Config:
    """Parameters controlling the Figure B1 simulation and output."""

    n: int = 1000
    p: int = 400
    p1: int = 50
    a: float = 0.5
    rhos: tuple[float, float] = (0.0, 0.5)
    sigma_e: float = 10.0
    figure_size: tuple[float, float] = (10.5, 4.8)
    output_path: str = "figures/supplementary/FigureB1.pdf"
    show_figure: bool = False


def generate_beta(p: int, p1: int, a: float, device: torch.device):
    """Generate a sparse coefficient vector with randomly signed signals."""
    beta_sample = torch.zeros(p, device=device)
    all_indices = torch.randperm(p, device=device)
    S1 = all_indices[:p1]
    S0_mask = torch.ones(p, dtype=torch.bool, device=device)
    S0_mask[S1] = False
    S0 = torch.arange(p, device=device)[S0_mask]

    signs = torch.where(torch.rand(p1, device=device) < 0.5, -1.0, 1.0)
    beta_sample[S1] = a * signs
    return beta_sample, S0, S1


def generate_design_matrix(p: int, n: int, rho: float, device: torch.device):
    """Generate and standardise a Gaussian design matrix."""
    if rho == 0:
        x_sample = torch.randn(n, p, device=device)
    else:
        indices = torch.arange(p, device=device)
        difference = torch.abs(indices[:, None] - indices[None, :])
        covariance = torch.pow(torch.tensor(rho, device=device), difference)
        distribution = MultivariateNormal(
            torch.zeros(p, device=device),
            covariance_matrix=covariance,
        )
        x_sample = distribution.rsample(sample_shape=(n,))

    x_sample = (x_sample - x_sample.mean(dim=0)) / x_sample.std(dim=0)
    return x_sample


def generate_response(
    x_sample: torch.Tensor,
    beta_sample: torch.Tensor,
    sigma_e: float,
    device: torch.device,
):
    """Generate a Gaussian linear-model response."""
    noise = torch.normal(
        mean=0.0,
        std=float(np.sqrt(sigma_e)),
        size=(x_sample.shape[0], 1),
        device=device,
    )
    return x_sample @ beta_sample.unsqueeze(1) + noise


def compute_standardised_pairs(
    x_sample: torch.Tensor,
    y_sample: torch.Tensor,
    sigma_e: float,
    device: torch.device,
):
    """Compute one standardised Gaussian mirror estimator pair per feature."""
    n, p = x_sample.shape
    kappa = p / n
    gamma_nor = torch.empty(p, device=device)
    eta_nor = torch.empty(p, device=device)

    covariance_estimate = shrink_cov(x_sample.cpu().numpy())
    precision_estimate = np.linalg.pinv(covariance_estimate)
    diagonal_identity = torch.eye(n, device=device)
    sqrt_n = torch.sqrt(torch.tensor(float(n), device=device))
    epsilon = 1e-10

    for i in range(p):
        xi = x_sample[:, i].unsqueeze(1)
        x_i = x_sample[:, [j for j in range(p) if j != i]]
        zi = torch.randn(n, 1, device=device)

        U_xi, singular_values, _ = torch.linalg.svd(x_i, full_matrices=False)
        U_xi = U_xi[:, singular_values > epsilon]
        projection = U_xi @ U_xi.T

        residual_projection = diagonal_identity - projection
        a_term = xi.T @ residual_projection @ residual_projection @ xi
        denominator = zi.T @ residual_projection @ zi
        ci = torch.sqrt(a_term / denominator)

        ci_zi = ci * zi
        x_new = torch.cat((xi + ci_zi, xi - ci_zi, x_i), dim=1)
        beta_i = torch.linalg.pinv(x_new) @ y_sample

        sigma_star = torch.sqrt(
            torch.tensor(
                (1 - kappa) / (precision_estimate[i, i] * sigma_e),
                dtype=torch.float32,
                device=device,
            )
        )
        gamma_nor[i] = sqrt_n * (beta_i[0] + beta_i[1]) * sigma_star
        eta_nor[i] = sqrt_n * (beta_i[0] - beta_i[1]) * sigma_star

    return gamma_nor.cpu().numpy(), eta_nor.cpu().numpy()


def plot_setting(ax, gamma_values, eta_values, S0, S1, title: str) -> None:
    """Draw one covariance setting."""
    S0 = S0.cpu().numpy()
    S1 = S1.cpu().numpy()

    ax.scatter(
        gamma_values[S0],
        eta_values[S0],
        s=10,
        c="red",
        alpha=0.7,
        edgecolors="none",
    )
    ax.scatter(
        gamma_values[S1],
        eta_values[S1],
        s=10,
        c="blue",
        alpha=0.7,
        edgecolors="none",
    )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\tilde{\gamma}$", fontsize=12)
    ax.set_ylabel(r"$\tilde{\eta}$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))


def main(config: FigureB1Config) -> None:
    """Draw and save Figure B1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    beta_sample, S0, S1 = generate_beta(config.p, config.p1, config.a, device)

    results = []
    for rho in config.rhos:
        print(f"Generating estimator pairs for rho={rho}.")
        x_sample = generate_design_matrix(config.p, config.n, rho, device)
        y_sample = generate_response(x_sample, beta_sample, config.sigma_e, device)
        gamma_values, eta_values = compute_standardised_pairs(
            x_sample,
            y_sample,
            config.sigma_e,
            device,
        )
        results.append((rho, gamma_values, eta_values))

    fig, axes = plt.subplots(1, len(config.rhos), figsize=config.figure_size)

    for ax, (rho, gamma_values, eta_values) in zip(axes, results):
        title = (
            "Independent setting" if rho == 0 else rf"Correlated setting ($\rho={rho}$)"
        )
        plot_setting(ax, gamma_values, eta_values, S0, S1, title)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=5,
            label="Null features",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="blue",
            markeredgecolor="blue",
            markersize=5,
            label="Non-null features",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.22)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")

    if config.show_figure:
        plt.show()
    plt.close(fig)

    print(f"Saved Figure B1 to: {output_path}")


if __name__ == "__main__":
    main(FigureB1Config())
