"""Simulation study for ODGM.

Edit the ``SimulationConfig`` object in ``main()`` to run a different
simulation scenario. The script writes one directory under ``results/`` for
each configuration.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

from data_generation import generate_data, generate_signal_configuration
from methods import (
    GM,
    HATZ,
    SEC,
    TZ,
    odgm_voting,
    tq_angle,
    tq_mirror,
    tq_parabola,
    tq_sector,
)

# Some macOS environments may require it because of an OpenMP runtime conflict.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass
class SimulationConfig:
    """Parameters for one ODGM simulation scenario."""

    n: int = 800
    p: int = 200
    p1: int = 50
    a: float = 0.5
    rho: float = 0.0
    sigma_e: float = 10.0
    par: float = 2.0
    q: float = 0.1
    simu_trial: int = 1
    simu_derandom: int = 100
    num_mc: int = 100000
    results_dir: str = "results"


def get_algorithm_names():
    """Return the candidate procedures and the ODGM voting library."""
    algorithms = [
        "GM",
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
        "ODGM",
    ]

    voting_algo_names = [
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
    return algorithms, voting_algo_names


def fdp_func(M, t, S0, device):
    """Compute the realised false discovery proportion for one trial."""
    if len(S0) == 0:
        return 0.0

    numerator = torch.sum(M[S0] > t).item()
    denominator = torch.max(torch.sum(M > t), torch.tensor(1, device=device)).item()
    return numerator / denominator


def power_func(M, t, S1):
    """Compute the realised power for one trial."""
    if len(S1) == 0:
        return 0.0

    numerator = torch.sum(M[S1] > t).item()
    return numerator / len(S1)


def run_simulation(config: SimulationConfig):
    """Run the ODGM simulation study under ``config``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n = config.n
    p = config.p
    p1 = config.p1
    a = config.a
    rho = config.rho
    sigma_e = config.sigma_e
    par = config.par
    q = config.q
    simu_trial = config.simu_trial
    simu_derandom = config.simu_derandom
    num_mc = config.num_mc
    kappa = p / n

    data_folder = os.path.join(
        config.results_dir,
        f"example_n{n}_p{p}_p1{p1}_A{a}_rho{rho}_k1{par}",
    )
    os.makedirs(data_folder, exist_ok=False)

    alltrials_fdp_power_filename = os.path.join(data_folder, "Alltrials_FDP_Power.txt")

    beta_sample, S0, S1 = generate_signal_configuration(p, p1, a, device)
    torch.save(S0.cpu(), os.path.join(data_folder, "S0.pt"))
    torch.save(S1.cpu(), os.path.join(data_folder, "S1.pt"))

    algorithms, voting_algo_names = get_algorithm_names()

    with open(alltrials_fdp_power_filename, "w", encoding="utf-8") as f:
        f.write(
            f"n={n}, p={p}, p1={p1}, A={a}, rho={rho}, "
            f"sigma_e={sigma_e}, k1={par}, q={q}\n"
        )
        f.write(
            f"simu_trial={simu_trial}, simu_derandom={simu_derandom}, "
            f"num_mc={num_mc}\n"
        )
        f.write(f"participating algorithms: {algorithms}\n")
        f.write(f"voting-participating algorithms: {voting_algo_names}\n")

    alltrials_fdp = {algo: [] for algo in algorithms}
    alltrials_power = {algo: [] for algo in algorithms}

    sigma_star = torch.zeros(p, device=device)
    for i in range(p):
        if rho == 0:
            sigma_star[i] = torch.sqrt(
                torch.tensor((1 - kappa) / sigma_e, dtype=torch.float32, device=device)
            )
        elif i == 0 or i == p - 1:
            sigma_star[i] = torch.sqrt(
                torch.tensor(
                    (1 - kappa) * (1 - rho**2) / sigma_e,
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            sigma_star[i] = torch.sqrt(
                torch.tensor(
                    (1 - kappa) * (1 - rho**2) / (sigma_e * (1 + rho**2)),
                    dtype=torch.float32,
                    device=device,
                )
            )

    mc_samples = torch.randn(num_mc, 2, device=device)

    for trial in range(simu_trial):
        x_sample, y_sample = generate_data(p, n, rho, sigma_e, beta_sample, device)

        trial_filename = f"{trial + 1}_trial"
        trial_folder = os.path.join(data_folder, trial_filename)
        os.makedirs(trial_folder, exist_ok=True)

        result_matrix_iter_filename = os.path.join(
            trial_folder, "result_matrix_iter.txt"
        )
        with open(result_matrix_iter_filename, "w", encoding="utf-8") as f:
            f.write("\t".join(algorithms[1:-1]) + "\n")

        all_gamma_nor = torch.zeros(simu_derandom, p, device=device)
        all_eta_nor = torch.zeros(simu_derandom, p, device=device)
        odgm_rejection_matrix = torch.zeros(
            simu_derandom, p, dtype=torch.bool, device=device
        )

        fdp_results_z = {algo: [] for algo in algorithms}
        power_results_z = {algo: [] for algo in algorithms}

        final_rejections_filename = os.path.join(
            trial_folder, f"final_rejection_results.txt"
        )
        with open(final_rejections_filename, "w", encoding="utf-8") as f:
            f.write(f"participating algorithms: {algorithms}\n")
            f.write(f"voting-participating algorithms: {voting_algo_names}\n")

        sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=device))
        diag_n = torch.eye(n, device=device)
        epsilon = 1e-10

        x_i_list = []
        x_i_without_i_list = []
        Pi_list = []
        a_list = []

        for i in range(p):
            xi = x_sample[:, i].unsqueeze(1)
            x_i_cols = list(range(p))
            x_i_cols.pop(i)
            x_i = x_sample[:, x_i_cols]

            U_xi, D_xi, _ = torch.linalg.svd(x_i, full_matrices=False)
            U_xi_filtered = U_xi[:, D_xi > epsilon]
            Pi = torch.matmul(U_xi_filtered, U_xi_filtered.T)

            term1 = diag_n - Pi
            a_i = torch.matmul(xi.T, torch.matmul(term1, torch.matmul(term1, xi)))

            x_i_list.append(xi)
            x_i_without_i_list.append(x_i)
            Pi_list.append(Pi)
            a_list.append(a_i)

        for k in range(simu_derandom):
            gamma_ = torch.empty(p, device=device)
            eta_ = torch.empty(p, device=device)
            gamma_nor = torch.empty(p, device=device)
            eta_nor = torch.empty(p, device=device)

            for i in range(p):
                xi = x_i_list[i]
                x_i = x_i_without_i_list[i]
                Pi = Pi_list[i]
                a_i = a_list[i]

                zi = torch.randn(n, 1, device=device)
                denominator_ci = torch.matmul(zi.T, torch.matmul(diag_n - Pi, zi))
                ci = torch.sqrt(a_i / denominator_ci)
                ci_zi = ci * zi

                x_new = torch.cat((xi + ci_zi, xi - ci_zi, x_i), dim=1)
                beta_i = torch.matmul(torch.linalg.pinv(x_new), y_sample)

                gamma_[i] = beta_i[0] + beta_i[1]
                eta_[i] = beta_i[0] - beta_i[1]
                gamma_nor[i] = sqrt_n * (beta_i[0] + beta_i[1]) * sigma_star[i]
                eta_nor[i] = sqrt_n * (beta_i[0] - beta_i[1]) * sigma_star[i]

            all_gamma_nor[k, :] = gamma_nor
            all_eta_nor[k, :] = eta_nor

            # GM is evaluated once using the first injected-noise realisation.
            # The resulting selection is retained for the remaining ODGM iterations.
            if k == 0:
                M_GM = GM(gamma_, eta_, 1)
                tq_GM = tq_mirror(M_GM, q, device)

            M_ATZ = TZ(gamma_nor, eta_nor, par)
            M_AGM = GM(gamma_nor, eta_nor, par)
            M_HATZ = HATZ(gamma_nor, eta_nor, par)
            M_SEC = SEC(gamma_nor, eta_nor, par)

            tq_AGM_A_1 = tq_angle(
                "GM", M_AGM, gamma_nor, eta_nor, par, 0.5, q, device, mc_samples
            )
            tq_AGM_A_2 = tq_angle(
                "GM", M_AGM, gamma_nor, eta_nor, par, 1, q, device, mc_samples
            )
            tq_AGM_A_3 = tq_angle(
                "GM", M_AGM, gamma_nor, eta_nor, par, 2, q, device, mc_samples
            )
            tq_AGM_P_1 = tq_parabola(
                "GM", M_AGM, gamma_nor, eta_nor, par, 0.5, q, device, mc_samples
            )
            tq_AGM_P_2 = tq_parabola(
                "GM", M_AGM, gamma_nor, eta_nor, par, 1, q, device, mc_samples
            )
            tq_AGM_P_3 = tq_parabola(
                "GM", M_AGM, gamma_nor, eta_nor, par, 2, q, device, mc_samples
            )

            tq_ATZ_A_1 = tq_angle(
                "TZ", M_ATZ, gamma_nor, eta_nor, par, 0.5, q, device, mc_samples
            )
            tq_ATZ_A_2 = tq_angle(
                "TZ", M_ATZ, gamma_nor, eta_nor, par, 1, q, device, mc_samples
            )
            tq_ATZ_A_3 = tq_angle(
                "TZ", M_ATZ, gamma_nor, eta_nor, par, 2, q, device, mc_samples
            )
            tq_ATZ_P_1 = tq_parabola(
                "TZ", M_ATZ, gamma_nor, eta_nor, par, 0.5, q, device, mc_samples
            )
            tq_ATZ_P_2 = tq_parabola(
                "TZ", M_ATZ, gamma_nor, eta_nor, par, 1, q, device, mc_samples
            )
            tq_ATZ_P_3 = tq_parabola(
                "TZ", M_ATZ, gamma_nor, eta_nor, par, 2, q, device, mc_samples
            )

            tq_HATZ_A_1 = tq_angle(
                "HATZ", M_HATZ, gamma_nor, eta_nor, par, 0.5, q, device, mc_samples
            )
            tq_HATZ_A_2 = tq_angle(
                "HATZ", M_HATZ, gamma_nor, eta_nor, par, 1, q, device, mc_samples
            )
            tq_HATZ_A_3 = tq_angle(
                "HATZ", M_HATZ, gamma_nor, eta_nor, par, 2, q, device, mc_samples
            )
            tq_HATZ_P_1 = tq_parabola(
                "HATZ", M_HATZ, gamma_nor, eta_nor, par, 0.5, q, device, mc_samples
            )
            tq_HATZ_P_2 = tq_parabola(
                "HATZ", M_HATZ, gamma_nor, eta_nor, par, 1, q, device, mc_samples
            )
            tq_HATZ_P_3 = tq_parabola(
                "HATZ", M_HATZ, gamma_nor, eta_nor, par, 2, q, device, mc_samples
            )

            tq_SEC_S = tq_sector(M_SEC, gamma_nor, eta_nor, par, 1 / par, q, device)

            all_algorithms = [
                ("GM", M_GM, tq_GM),
                ("AGM_A_1", M_AGM, tq_AGM_A_1),
                ("AGM_A_2", M_AGM, tq_AGM_A_2),
                ("AGM_A_3", M_AGM, tq_AGM_A_3),
                ("AGM_P_1", M_AGM, tq_AGM_P_1),
                ("AGM_P_2", M_AGM, tq_AGM_P_2),
                ("AGM_P_3", M_AGM, tq_AGM_P_3),
                ("ATZ_A_1", M_ATZ, tq_ATZ_A_1),
                ("ATZ_A_2", M_ATZ, tq_ATZ_A_2),
                ("ATZ_A_3", M_ATZ, tq_ATZ_A_3),
                ("ATZ_P_1", M_ATZ, tq_ATZ_P_1),
                ("ATZ_P_2", M_ATZ, tq_ATZ_P_2),
                ("ATZ_P_3", M_ATZ, tq_ATZ_P_3),
                ("HATZ_A_1", M_HATZ, tq_HATZ_A_1),
                ("HATZ_A_2", M_HATZ, tq_HATZ_A_2),
                ("HATZ_A_3", M_HATZ, tq_HATZ_A_3),
                ("HATZ_P_1", M_HATZ, tq_HATZ_P_1),
                ("HATZ_P_2", M_HATZ, tq_HATZ_P_2),
                ("HATZ_P_3", M_HATZ, tq_HATZ_P_3),
                ("SEC_S", M_SEC, tq_SEC_S),
            ]

            M_odgm, tq_odgm, _, full_vote_matrix = odgm_voting(
                all_algorithms, voting_algo_names, p, device
            )
            odgm_rejection_matrix[k, :] = M_odgm > tq_odgm

            with open(result_matrix_iter_filename, "a", encoding="utf-8") as f:
                f.write(f"Iteration {k + 1} selection matrix\n")
                np.savetxt(f, full_vote_matrix.float().cpu().numpy(), fmt="%.1f")

            for algo_name, M, tq in all_algorithms:
                fdp_val = fdp_func(M, tq, S0, device)
                power_val = power_func(M, tq, S1)
                fdp_results_z[algo_name].append(fdp_val)
                power_results_z[algo_name].append(power_val)

            fdp_results_z["ODGM"].append(fdp_func(M_odgm, tq_odgm, S0, device))
            power_results_z["ODGM"].append(power_func(M_odgm, tq_odgm, S1))

            if k == 0:
                with open(final_rejections_filename, "w", encoding="utf-8") as f:
                    for algo_name, M, tq in all_algorithms:
                        rejections = (M > tq).cpu().numpy()
                        R = np.sum(rejections)
                        fdp_val = fdp_results_z[algo_name][-1]
                        power_val = power_results_z[algo_name][-1]
                        rejected_indices = np.where(rejections)[0]
                        f.write(
                            f"{algo_name}: R={R}, fdp={fdp_val:.6f}, "
                            f"power={power_val:.6f}\n"
                        )
                        f.write(
                            f"feature subset:{', '.join(map(str, rejected_indices))}\n"
                        )

            if k == simu_derandom - 1:
                total_rejections = torch.sum(odgm_rejection_matrix).item()
                R = int(np.floor(total_rejections / simu_derandom))
                odgm_selection_frequency = (
                    torch.sum(odgm_rejection_matrix, dim=0).float() / simu_derandom
                )
                sorted_indices = torch.argsort(
                    odgm_selection_frequency, descending=True
                )
                final_odgm_rejected_indices = sorted_indices[:R].cpu().numpy()
                final_odgm_rejections = np.zeros(p, dtype=bool)
                final_odgm_rejections[final_odgm_rejected_indices] = True

                odgm_fdp_z = fdp_func(
                    torch.tensor(final_odgm_rejections, device=device), 0.5, S0, device
                )
                odgm_power_z = power_func(
                    torch.tensor(final_odgm_rejections, device=device), 0.5, S1
                )

                with open(final_rejections_filename, "a", encoding="utf-8") as f:
                    f.write(
                        f"ODGM: R={R}, fdp={odgm_fdp_z:.6f}, "
                        f"power={odgm_power_z:.6f}\n"
                    )
                    f.write(
                        f"feature subset:{', '.join(map(str, final_odgm_rejected_indices))}\n"
                    )

                for algo_name, _, _ in all_algorithms:
                    alltrials_fdp[algo_name].append(fdp_results_z[algo_name][0])
                    alltrials_power[algo_name].append(power_results_z[algo_name][0])
                alltrials_fdp["ODGM"].append(odgm_fdp_z)
                alltrials_power["ODGM"].append(odgm_power_z)

        with open(alltrials_fdp_power_filename, "a", encoding="utf-8") as f:
            f.write(f"Trial {trial + 1}:\n")
            f.write("FDP:\n")
            f.write(
                f" {{{', '.join([f'{algo}: {alltrials_fdp[algo][-1]:.6f}' for algo in algorithms])}}}\n"
            )
            f.write("Power:\n")
            f.write(
                f"{{{', '.join([f'{algo}: {alltrials_power[algo][-1]:.6f}' for algo in algorithms])}}}\n"
            )

        print(f"After {trial + 1} trials:")
        print({algo: np.mean(alltrials_fdp[algo]) for algo in algorithms})
        print({algo: np.mean(alltrials_power[algo]) for algo in algorithms})

        torch.save(all_gamma_nor.cpu(), os.path.join(trial_folder, "gamma_nor.pt"))
        torch.save(all_eta_nor.cpu(), os.path.join(trial_folder, "eta_nor.pt"))

        del x_i_list, x_i_without_i_list, Pi_list, a_list, all_gamma_nor, all_eta_nor
        del odgm_rejection_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Configure and run the default simulation scenario."""
    config = SimulationConfig()
    run_simulation(config)


if __name__ == "__main__":
    main()
