"""Core statistics, calibration thresholds, and voting for ODGM analyses."""

import torch


def GM(gamma, eta, par):
    """Gaussian mirror statistic with slope parameter ``par``."""
    return par * torch.abs(gamma) - torch.abs(eta)


def TZ(gamma, eta, par):
    """Adapted trapezoid statistic."""
    return torch.sign(par * torch.abs(gamma) - torch.abs(eta)) * torch.maximum(
        par * torch.abs(gamma), torch.abs(eta)
    )


def HATZ(gamma, eta, par):
    """Holey adapted trapezoid statistic."""
    return torch.sign(par * torch.abs(gamma) - torch.abs(eta)) * (
        par * torch.abs(gamma) + torch.abs(eta)
    )


def SEC(gamma, eta, par):
    """Sector statistic."""
    return torch.sign(par * torch.abs(gamma) - torch.abs(eta)) * (gamma**2 + eta**2)


method_functions = {
    "GM": GM,
    "TZ": TZ,
    "HATZ": HATZ,
    "SEC": SEC,
}


def tq_mirror(M, q, device):
    """Return the Gaussian-mirror threshold at nominal FDR level ``q``."""
    M_sort = torch.sort(M).values
    numerator = (M[:, None] < -M_sort[None, :]).sum(dim=0)
    denominator = torch.maximum(
        (M[:, None] > M_sort[None, :]).sum(dim=0),
        torch.tensor(1, device=device),
    )
    fdp_estimate = numerator.float() / denominator.float()
    valid_indices = torch.where(fdp_estimate <= q)[0]

    if len(valid_indices) > 0:
        return M_sort[valid_indices[0]].item()
    return M_sort[-1].item() + 10


def tq_parabola(method_name, M, gamma, eta, par, par_pb, q, device, mc_samples):
    """Return the threshold using a parabolic calibration region."""
    func = method_functions[method_name]
    M_sort = torch.sort(M).values

    generate_m = func(mc_samples[:, 0], mc_samples[:, 1], par)
    generate_pb = torch.abs(mc_samples[:, 1]) - par_pb * (mc_samples[:, 0] ** 2)
    sorted_generate_pb = torch.sort(generate_pb, descending=True).values

    count_ge_tq = (generate_m[:, None] > M_sort[None, :]).sum(dim=0)
    t = torch.full((M.numel(),), float("-inf"), device=device)
    valid_mask = count_ge_tq > 0
    t[valid_mask] = sorted_generate_pb[count_ge_tq[valid_mask] - 1]

    numerator_condition = torch.abs(eta) - par_pb * (gamma**2)
    numerator = (numerator_condition[:, None] > t[None, :]).sum(dim=0)
    numerator[count_ge_tq == 0] = 0

    denominator = torch.maximum(
        (M[:, None] > M_sort[None, :]).sum(dim=0),
        torch.tensor(1, device=device),
    )
    fdp_estimate = numerator.float() / denominator.float()
    valid_indices = torch.where(fdp_estimate <= q)[0]

    if len(valid_indices) > 0:
        return M_sort[valid_indices[0]].item()
    return M_sort[-1].item() + 10


def tq_sector(M, gamma, eta, par, par_ag, q, device):
    """Return the threshold using the sectoral calibration region."""
    M_sort = torch.sort(M).values
    M_fdp = SEC(eta, gamma, par_ag)

    alpha1 = torch.atan(torch.tensor(par, device=device))
    alpha2 = torch.atan(torch.tensor(par_ag, device=device))
    numerator = (M_fdp[:, None] > M_sort[None, :]).sum(dim=0) * (alpha1 / alpha2).item()
    denominator = torch.maximum(
        (M[:, None] > M_sort[None, :]).sum(dim=0),
        torch.tensor(1, device=device),
    )
    fdp_estimate = numerator.float() / denominator.float()
    valid_indices = torch.where(fdp_estimate <= q)[0]

    if len(valid_indices) > 0:
        return M_sort[valid_indices[0]].item()
    return M_sort[-1].item() + 10


def tq_angle(method_name, M, gamma, eta, par, par_pb, q, device, mc_samples):
    """Return the threshold using an angular calibration region."""
    func = method_functions[method_name]
    M_sort = torch.sort(M).values

    generate_m = func(mc_samples[:, 0], mc_samples[:, 1], par)
    generate_pb = torch.abs(mc_samples[:, 1]) - par_pb * torch.abs(mc_samples[:, 0])
    sorted_generate_pb = torch.sort(generate_pb, descending=True).values

    count_ge_tq = (generate_m[:, None] > M_sort[None, :]).sum(dim=0)
    t = torch.full((M.numel(),), float("-inf"), device=device)
    valid_mask = count_ge_tq > 0
    t[valid_mask] = sorted_generate_pb[count_ge_tq[valid_mask] - 1]

    numerator_condition = torch.abs(eta) - par_pb * torch.abs(gamma)
    numerator = (numerator_condition[:, None] > t[None, :]).sum(dim=0)
    numerator[count_ge_tq == 0] = 0

    denominator = torch.maximum(
        (M[:, None] > M_sort[None, :]).sum(dim=0),
        torch.tensor(1, device=device),
    )
    fdp_estimate = numerator.float() / denominator.float()
    valid_indices = torch.where(fdp_estimate <= q)[0]

    if len(valid_indices) > 0:
        return M_sort[valid_indices[0]].item()
    return M_sort[-1].item() + 10


def odgm_voting(all_algorithms, algorithms_to_vote_names, p, device):
    """Apply majority voting to the selected methods in the ODGM library."""
    algo_name_to_index = {algo[0]: idx for idx, algo in enumerate(all_algorithms)}
    full_vote_matrix = torch.zeros(
        (len(all_algorithms), p), dtype=torch.bool, device=device
    )

    for i, (_, M_algo, tq_algo) in enumerate(all_algorithms):
        full_vote_matrix[i] = M_algo > tq_algo

    algorithms_to_vote = [
        (name, M, tq)
        for name, M, tq in all_algorithms
        if name in algorithms_to_vote_names
    ]
    vote_matrix = torch.zeros(
        (len(algorithms_to_vote), p), dtype=torch.bool, device=device
    )

    for i, (algo_name, _, _) in enumerate(algorithms_to_vote):
        vote_matrix[i] = full_vote_matrix[algo_name_to_index[algo_name]]

    rejection_counts = torch.sum(vote_matrix, dim=0).float()
    M_odgm = (rejection_counts > len(algorithms_to_vote) / 2).float()
    tq_odgm = 0.5
    return M_odgm, tq_odgm, vote_matrix, full_vote_matrix
