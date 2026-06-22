"""HIV-1 drug-resistance analysis using GM, deformable GM procedures, and ODGM.

Edit the ``RealDataConfig`` object in ``main()`` to analyse a different drug
data set or to change the ODGM settings.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from nonlinshrink import shrink_cov
from sklearn.linear_model import LinearRegression

from data_loading import load_hiv_data
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
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@dataclass
class RealDataConfig:
    """Parameters for one HIV-1 drug-resistance analysis."""

    dataname: str = "ATV"
    data_folder: str = os.path.join("HIVdata")
    results_dir: str = os.path.join("results", "real_data")
    simu: int = 100
    par: float = 2.0
    q: float = 0.1
    num_mc: int = 100000


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


def run_realdata_analysis(config: RealDataConfig):
    """Run ODGM for one HIV-1 protease inhibitor data set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataname = config.dataname
    simu = config.simu
    par = config.par
    q = config.q
    num_mc = config.num_mc

    results_folder = os.path.join(config.results_dir, dataname)
    os.makedirs(results_folder, exist_ok=False)

    log_filename = os.path.join(results_folder, "log_rejections.txt")
    algorithm_rejections_filename = os.path.join(
        results_folder, "algorithm_rejections.txt"
    )

    x_sample, y_sample = load_hiv_data(config.data_folder, dataname, device)
    p = x_sample.shape[1]
    n = x_sample.shape[0]
    kappa = p / n
    print(f"n={n}, p={p}")

    cov_est = shrink_cov(x_sample.cpu().numpy())
    precision_est = np.linalg.pinv(cov_est)

    model = LinearRegression(fit_intercept=False)
    model.fit(x_sample.cpu().numpy(), y_sample.cpu().numpy())
    resi = y_sample - torch.tensor(
        model.predict(x_sample.cpu().numpy()), dtype=torch.float32, device=device
    )
    sigma_e = torch.sum(resi**2) / (n - p)

    algorithms, voting_algo_names = get_algorithm_names()

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Data set: {dataname}\n")
        f.write(f"n={n}, p={p}, sigma_e_hat={sigma_e.item():.6f}\n")
        f.write(f"simu={simu}, k1={par}, q={q}, num_mc={num_mc}\n")
        f.write(f"Participating algorithms: {algorithms}\n")
        f.write(f"Voting algorithms: {voting_algo_names}\n")

    mc_samples = torch.randn(num_mc, 2, device=device)
    selection_counts = torch.zeros(len(algorithms), p, dtype=torch.long, device=device)

    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=device))
    diag_n = torch.eye(n, device=device)
    epsilon = 1e-10

    x_i_list = []
    x_i_without_i_list = []
    Pi_list = []
    a_list = []
    sigma_star = torch.zeros(p, device=device)

    for i in range(p):
        sigma_star[i] = torch.sqrt(
            torch.tensor(
                (1 - kappa) / (precision_est[i, i] * sigma_e.item()),
                dtype=torch.float32,
                device=device,
            )
        )

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

    for k in range(simu):
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

        M_odgm, tq_odgm, _, _ = odgm_voting(
            all_algorithms, voting_algo_names, p, device
        )

        for algorithm_index, (_, statistic, threshold) in enumerate(all_algorithms):
            selection_counts[algorithm_index] += (statistic > threshold).long()
        selection_counts[-1] += (M_odgm > tq_odgm).long()

    average_rejections = selection_counts.sum(dim=1).float() / simu
    final_selections = {}

    for algorithm_index, algorithm_name in enumerate(algorithms):
        selection_size = int(torch.floor(average_rejections[algorithm_index]).item())
        ranked_features = torch.argsort(
            selection_counts[algorithm_index],
            descending=True,
        )
        final_selections[algorithm_name] = ranked_features[:selection_size] + 1

    with open(algorithm_rejections_filename, "w", encoding="utf-8") as f:
        f.write(
            "# Final feature selections after derandomisation. "
            "Feature indices are one-based.\n"
        )
        f.write(
            "# Each method retains the top floor(mean number of rejections) "
            "features ranked by selection frequency.\n"
        )
        for algorithm_name in algorithms:
            selected_indices = final_selections[algorithm_name].cpu().tolist()
            f.write(f"{algorithm_name}: " + " ".join(map(str, selected_indices)) + "\n")

    print(f"Saved final algorithm selections to: {algorithm_rejections_filename}")


def main():
    """Configure and run the default HIV-1 analysis."""
    config = RealDataConfig()
    run_realdata_analysis(config)


if __name__ == "__main__":
    main()
