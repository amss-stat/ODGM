"""Run the k1-sensitivity experiment used for Figure B6 and Figure B7.

The script reuses the saved standardised estimator pairs from one completed
main_simu.py run. Thus, every compared k1 value uses the same simulated data
and the same injected-noise realisation within each trial.

Only k1 = 0.5 and k1 = 1 are evaluated. The output is a compact
Alltrials_FDP_Power.txt file for the Figure B6/B7 plotting script.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from methods import GM, HATZ, TZ, tq_angle, tq_parabola


@dataclass(frozen=True)
class K1SensitivityConfig:
    """Parameters for one fixed-data k1-sensitivity experiment."""

    n: int = 800
    p: int = 200
    p1: int = 50
    a: float = 0.5
    rho: float = 0.0
    sigma_e: float = 10.0
    q: float = 0.1
    simu_trial: int = 1
    num_mc: int = 100000
    noise_index: int = 0
    source_results_dir: Path = Path("results")
    output_results_dir: Path = Path("results")


def format_value(value: float | int) -> str:
    """Format a parameter value exactly as used in result-folder names."""
    return f"{value:g}"


def get_source_folder(config: K1SensitivityConfig) -> Path:
    """Return the completed main-simulation folder reused by this experiment."""
    return config.source_results_dir / (
        f"n{config.n}_p{config.p}_p1{config.p1}_"
        f"A{format_value(config.a)}_rho{format_value(config.rho)}_k1{format_value(2)}"
    )


def get_output_folder(config: K1SensitivityConfig) -> Path:
    """Return the folder for the compact k1-sensitivity summaries."""
    return config.output_results_dir / (
        f"k1_sensitivity_n{config.n}_p{config.p}_p1{config.p1}_"
        f"A{format_value(config.a)}_rho{format_value(config.rho)}"
    )


K1_VALUES = (0.5, 1.0)
METHOD_SPECS = (
    ("AGM", "GM"),
    ("ATZ", "TZ"),
    ("HATZ", "HATZ"),
)
CALIBRATION_SPECS = (
    ("A", tq_angle),
    ("P", tq_parabola),
)


def fdp_func(statistic: torch.Tensor, threshold: float, s0: torch.Tensor) -> float:
    """Compute the realised false discovery proportion."""
    rejected = statistic > threshold
    denominator = max(int(rejected.sum().item()), 1)
    return float(rejected[s0].sum().item() / denominator)


def power_func(statistic: torch.Tensor, threshold: float, s1: torch.Tensor) -> float:
    """Compute the realised power."""
    if len(s1) == 0:
        return 0.0
    return float((statistic[s1] > threshold).sum().item() / len(s1))


def get_sigma_star(config: K1SensitivityConfig, device: torch.device) -> torch.Tensor:
    """Compute the standardisation constants used in main_simu.py."""
    kappa = config.p / config.n
    sigma_star = torch.zeros(config.p, device=device)

    for index in range(config.p):
        if config.rho == 0:
            value = (1 - kappa) / config.sigma_e
        elif index in (0, config.p - 1):
            value = (1 - kappa) * (1 - config.rho**2) / config.sigma_e
        else:
            value = (
                (1 - kappa)
                * (1 - config.rho**2)
                / (config.sigma_e * (1 + config.rho**2))
            )
        sigma_star[index] = torch.sqrt(torch.tensor(value, device=device))

    return sigma_star


def build_algorithm_results(
    gamma_nor: torch.Tensor,
    eta_nor: torch.Tensor,
    config: K1SensitivityConfig,
    mc_samples: torch.Tensor,
    device: torch.device,
) -> list[tuple[str, torch.Tensor, float]]:
    """Evaluate A- and P-calibrated methods for all requested k1 values."""
    results = []

    for method_prefix, method_name in METHOD_SPECS:
        statistic_function = {"GM": GM, "TZ": TZ, "HATZ": HATZ}[method_name]

        for calibration_suffix, threshold_function in CALIBRATION_SPECS:
            for k1_index, k1 in enumerate(K1_VALUES, start=1):
                statistic = statistic_function(gamma_nor, eta_nor, k1)
                threshold = threshold_function(
                    method_name,
                    statistic,
                    gamma_nor,
                    eta_nor,
                    k1,
                    1.0,
                    config.q,
                    device,
                    mc_samples,
                )
                algorithm_name = f"{method_prefix}_{calibration_suffix}_2_{k1_index}"
                results.append((algorithm_name, statistic, threshold))

    return results


def run_k1_sensitivity(config: K1SensitivityConfig) -> None:
    """Evaluate k1 values using fixed saved data and injected-noise draws."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_folder = get_source_folder(config)
    output_folder = get_output_folder(config)
    output_folder.mkdir(parents=True, exist_ok=False)

    s0_path = source_folder / "S0.pt"
    s1_path = source_folder / "S1.pt"
    if not s0_path.exists() or not s1_path.exists():
        raise FileNotFoundError(
            f"Expected S0.pt and S1.pt in {source_folder}. Run main_simu.py first."
        )

    s0 = torch.load(s0_path, map_location=device, weights_only=True)
    s1 = torch.load(s1_path, map_location=device, weights_only=True)
    sigma_star = get_sigma_star(config, device)
    sqrt_n = torch.sqrt(torch.tensor(config.n, dtype=torch.float32, device=device))
    mc_samples = torch.randn(config.num_mc, 2, device=device)

    algorithms = [
        f"{method}_{calibration}_2_{k1_index}"
        for method, _ in METHOD_SPECS
        for calibration, _ in CALIBRATION_SPECS
        for k1_index in range(1, len(K1_VALUES) + 1)
    ]
    all_fdp = {algorithm: [] for algorithm in algorithms}
    all_power = {algorithm: [] for algorithm in algorithms}

    output_file = output_folder / "Alltrials_FDP_Power.txt"
    with output_file.open("w", encoding="utf-8") as file:
        file.write(
            f"source_results={source_folder}\n"
            f"k1_values={list(K1_VALUES)}, noise_index={config.noise_index}\n"
            f"participating algorithms: {algorithms}\n"
        )

    for trial in range(1, config.simu_trial + 1):
        trial_folder = source_folder / f"{trial}_trial"
        gamma_path = trial_folder / "gamma_nor.pt"
        eta_path = trial_folder / "eta_nor.pt"

        if not gamma_path.exists() or not eta_path.exists():
            raise FileNotFoundError(
                f"Missing gamma_nor.pt or eta_nor.pt in {trial_folder}."
            )

        gamma_all = torch.load(gamma_path, map_location=device, weights_only=True)
        eta_all = torch.load(eta_path, map_location=device, weights_only=True)

        if config.noise_index >= gamma_all.shape[0]:
            raise IndexError(
                f"noise_index={config.noise_index} is unavailable in {trial_folder}."
            )

        gamma_nor = gamma_all[config.noise_index]
        eta_nor = eta_all[config.noise_index]

        algorithm_results = build_algorithm_results(
            gamma_nor,
            eta_nor,
            config,
            mc_samples,
            device,
        )

        trial_fdp = {}
        trial_power = {}
        for algorithm_name, statistic, threshold in algorithm_results:
            trial_fdp[algorithm_name] = fdp_func(statistic, threshold, s0)
            trial_power[algorithm_name] = power_func(statistic, threshold, s1)
            all_fdp[algorithm_name].append(trial_fdp[algorithm_name])
            all_power[algorithm_name].append(trial_power[algorithm_name])

        with output_file.open("a", encoding="utf-8") as file:
            file.write(f"Trial {trial}:\n")
            file.write("FDP:\n")
            file.write(
                " {"
                + ", ".join(f"{name}: {trial_fdp[name]:.6f}" for name in algorithms)
                + "}\n"
            )
            file.write("Power:\n")
            file.write(
                "{"
                + ", ".join(f"{name}: {trial_power[name]:.6f}" for name in algorithms)
                + "}\n"
            )

        print(f"Completed trial {trial}/{config.simu_trial}.")

    print(f"Saved k1-sensitivity results to: {output_file}")


def main() -> None:
    """Configure and run one k1-sensitivity experiment."""
    config = K1SensitivityConfig()
    run_k1_sensitivity(config)


if __name__ == "__main__":
    main()
