"""Data generation for the ODGM simulation study."""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def generate_signal_configuration(p, p1, a, device):
    """Generate the coefficient vector and the null and signal index sets."""
    beta_sample = torch.zeros(p, device=device)
    all_indices = torch.randperm(p)
    S1 = all_indices[:p1]
    S0_mask = torch.ones(p, dtype=torch.bool, device=device)
    S0_mask[S1] = False
    S0 = torch.arange(p, device=device)[S0_mask]

    signs = torch.where(
        torch.rand(p1, device=device) < 0.5,
        torch.tensor(-1.0, device=device),
        torch.tensor(1.0, device=device),
    )
    beta_sample[S1] = a * signs
    return beta_sample, S0, S1


def generate_data(p, n, rho, sigma_e, beta_sample, device):
    """Generate one standardised design matrix and response vector."""
    if rho == 0:
        x_sample = torch.randn(n, p, device=device)
    else:
        indices = torch.arange(p, device=device)
        diff = torch.abs(indices.view(-1, 1) - indices.view(1, -1))
        cov_matrix = torch.pow(rho, diff)
        mean = torch.zeros(p, device=device)
        mvn = MultivariateNormal(mean, covariance_matrix=cov_matrix)
        x_sample = mvn.rsample(sample_shape=(n,))

    std_dev = torch.sqrt(torch.tensor(sigma_e, dtype=torch.float32, device=device))
    noise = torch.normal(mean=0, std=std_dev, size=(n, 1), device=device)
    y_sample = torch.matmul(x_sample, beta_sample.unsqueeze(1)) + noise

    x_sample_mean = x_sample.mean(dim=0)
    x_sample_std = x_sample.std(dim=0)
    x_sample_std[x_sample_std == 0] = 1.0
    x_sample = (x_sample - x_sample_mean) / x_sample_std
    return x_sample, y_sample
