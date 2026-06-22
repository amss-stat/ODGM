# ODGM: Feature Selection via Deformable Gaussian Mirror with FDR Control

Reproducibility code for the paper **“ODGM: Feature Selection via Deformable Gaussian Mirror with FDR Control.”**

ODGM is a p-value-free framework for feature selection in proportional high-dimensional linear regression. It extends Gaussian Mirror by allowing flexible rejection and calibration regions, aggregates a library of deformable Gaussian mirror procedures through majority voting, and stabilises the final selection through derandomisation.

## Installation

Python 3.10 or later is recommended. The Python code automatically uses CUDA when an appropriate PyTorch installation and compatible GPU are available. A CPU installation also works, but may be substantially slower. Install the required packages:

```bash
pip install -r requirements.txt
```

Figure 7 is produced in R. Install the required R packages:

```r
source("requirements.R")
```

## Simulation study

The simulation scenario is controlled by the `SimulationConfig` object in `main_simu.py`. For example:

```python
config = SimulationConfig(
    n=800,
    p=200,
    p1=50,
    a=0.5,
    rho=0.5,
    sigma_e=10.0,
    par=2.0,
    q=0.1,
    simu_trial=100,
    simu_derandom=100,
    num_mc=100000,
)
```

### Data generation

For each trial, the response follows the linear model

$$
y = X \beta + \epsilon,
$$

where:

- the rows of $X$ are generated from $\mathcal{N}(0_p,\Sigma)$;
- $\Sigma=I_p$ when `rho = 0`, and otherwise $\Sigma_{jk}=\rho^{|j-k|}$;
- the columns of $X$ are standardised after generation;
- $p_1$ signal coordinates are selected uniformly at random;
- nonzero coefficients are independently assigned values in $\{-A,A\}$ with equal probability;
- $\epsilon\sim \mathcal{N}(0_n,\sigma_\epsilon^2 I_n)$, with default $\sigma_\epsilon^2=10$.

The default paper settings use $p_1=50$, target FDR level $q=0.1$, $100$ simulation trials, $100$ ODGM derandomisation repetitions, and $100000$ Monte Carlo draws for calibration.

### Methods

The main simulation evaluates GM, the 19-method ODGM library, and ODGM itself:

- AGM-A, AGM-P, ATZ-A, ATZ-P, HATZ-A, and HATZ-P with $k_1=2$ and $k_2\in\{0.5,1,2\}$;
- SEC-S with $k_1=2$ and $k_2=0.5$;
- ODGM, formed by majority voting over the 19 deformable procedures and then derandomised over repeated injected-noise realisations.

### Running the simulation

Before running the script, edit the `SimulationConfig` object in `main_simu.py` to specify the desired scenario. For example, the following configuration runs the default independent-design setting with $100$ trials and $100$ ODGM derandomisation repetitions:

```python
config = SimulationConfig(
    n=800,
    p=200,
    p1=50,
    a=0.5,
    rho=0.0,
    sigma_e=10.0,
    par=2.0,
    q=0.1,
    simu_trial=100,
    simu_derandom=100,
    num_mc=100_000,
)
```

Run the main simulation from the `simulation/` directory:

```bash
cd simulation
python data_generation.py
python methods.py
python main_simu.py
```

### Simulation outputs

The script creates one scenario-specific directory:

```text
results/
└── n{n}_p{p}_p1{p1}_A{a}_rho{rho}_k1{par}/
```

For the configuration above, this becomes:

```text
results/n800_p200_p150_A0.5_rho0.5_k12/
```

Each scenario directory contains:

```text
n.../
├── Alltrials_FDP_Power.txt
├── S0.pt
├── S1.pt
├── 1_trial/
│   ├── gamma_nor.pt
│   ├── eta_nor.pt
│   ├── result_matrix_iter.txt
│   └── final_rejection_results.txt
├── 2_trial/
│   └── ...
└── ...
```

| File | Purpose |
|---|---|
| `Alltrials_FDP_Power.txt` | Trial-level FDP and power for the methods reported in the main simulation. |
| `S0.pt`, `S1.pt` | True null and non-null feature indices; used to evaluate FDP and power. |
| `gamma_nor.pt`, `eta_nor.pt` | Standardised estimator pairs across injected-noise realisations; used for Figure B6/B7. |
| `result_matrix_iter.txt` | Method-selection matrices at each ODGM derandomisation iteration; used for Figure B2/B3. |
| `final_rejection_results.txt` | Selection summaries from the first realisation and the final derandomised ODGM result. |

### k1 sensitivity analysis

`simulation/simu_k1.py` reuses `S0.pt`, `S1.pt`, `gamma_nor.pt`, and `eta_nor.pt` from a completed `main_simu.py` run. It therefore compares $k_1=0.5$ and $k_1=1$ on the same saved simulation data and injected-noise realisation. The $k_1=2$ results are read directly from the corresponding `main_simu.py` output when producing Figure B6/B7.

`simulation/simu_k1.py` produces:

```text
results/k1_sensitivity_n{n}_p{p}_p1{p1}_A{a}_rho{rho}/Alltrials_FDP_Power.txt
```

## An example

The `example/` directory contains the saved output for one trial from the default independent-design setting:

```text
n = 800, p = 200, p1 = 50, A = 0.5, rho = 0, k1 = 2.
```

It is included to illustrate the expected result-file layout without requiring users to run the full simulation first.

The example contains:

| File | Purpose |
|---|---|
| `S0.pt`, `S1.pt` | Indices of the true null and non-null features. |
| `gamma_nor.pt`, `eta_nor.pt` | Standardised estimator pairs over injected-noise realisations. |
| `result_matrix_iter.txt` | Candidate-method selection matrices at each ODGM iteration. |
| `final_rejection_results.txt` | Selection summaries from the first realisation and the final derandomised ODGM result. |

The example is intended for inspecting the output format and for testing figure scripts that use one-trial intermediate results. It is not a substitute for the full simulation study.

## Real-data analysis

### Data loading

The real-data analysis uses the HIV-1 protease inhibitor drug-resistance data of Rhee et al. (2006), as described in the paper. The analysis considers seven protease inhibitor drugs:

```text
APV, ATV, IDV, LPV, NFV, RTV, SQV
```

For each drug, the design matrix records the presence or absence of protease mutations in HIV-1 isolates, and the response is the corresponding phenotypic fold change in drug susceptibility. The analysis removes mutations with no variation, standardises the mutation design matrix, log-transforms and standardises the response, and applies ODGM with $q=0.1$.

Prepare the source files separately, then place them under:

```text
HIVdata/
├── PIs.txt
├── XAPV.csv
├── XATV.csv
├── XIDV.csv
├── XLPV.csv
├── XNFV.csv
├── XRTV.csv
├── XSQV.csv
├── yAPV.RData
├── yATV.RData
├── yIDV.RData
├── yLPV.RData
├── yNFV.RData
├── yRTV.RData
└── ySQV.RData
```

`PIs.txt` is used only by the TSM-position figure. Each `X<drug>.csv` file provides mutation names and mutation features; each `y<drug>.RData` file provides the response vector.

### Running the real-data analysis

Change `dataname` in the `RealDataConfig` object in `main_realdata.py` to the drug to be analysed, for example:

```python
config = RealDataConfig(
    dataname="APV",
)
```

Valid choices are `"APV"`, `"ATV"`, `"IDV"`, `"LPV"`, `"NFV"`, `"RTV"`, and `"SQV"`. Then run the main analysis from the `realdata/` directory:

```bash
cd realdata
python data_loading.py
python methods.py
python main_realdata.py
```

Run the script once for each drug data set. The default configuration uses 100 injected-noise repetitions, $k_1=2$, $q=0.1$, and 100000 Monte Carlo samples for calibration.

### Real-data outputs

For each drug, `main_realdata.py` writes:

```text
results/real_data/<drug>/
├── log_rejections.txt
└── algorithm_rejections.txt
```

| File | Purpose |
|---|---|
| `log_rejections.txt` | Analysis settings, sample size, feature dimension, estimated residual variance, and the list of participating and voting methods. |
| `algorithm_rejections.txt` | Final one-based mutation indices selected by all 21 procedures after derandomisation. This file is used to generate the two real-data figures.|

## Method names used in the code

Python and R identifiers cannot contain characters such as hyphens and parentheses. The code therefore uses internal names, while figures and the paper use the display names below.

| Internal name | Display name |
|---|---|
| `GM` | GM |
| `AGM_A_1`, `AGM_A_2`, `AGM_A_3` | AGM-A(0.5), AGM-A(1), AGM-A(2) |
| `AGM_P_1`, `AGM_P_2`, `AGM_P_3` | AGM-P(0.5), AGM-P(1), AGM-P(2) |
| `ATZ_A_1`, `ATZ_A_2`, `ATZ_A_3` | ATZ-A(0.5), ATZ-A(1), ATZ-A(2) |
| `ATZ_P_1`, `ATZ_P_2`, `ATZ_P_3` | ATZ-P(0.5), ATZ-P(1), ATZ-P(2) |
| `HATZ_A_1`, `HATZ_A_2`, `HATZ_A_3` | HATZ-A(0.5), HATZ-A(1), HATZ-A(2) |
| `HATZ_P_1`, `HATZ_P_2`, `HATZ_P_3` | HATZ-P(0.5), HATZ-P(1), HATZ-P(2) |
| `SEC_S` | SEC-S |
| `ODGM` | ODGM |

Here, the suffixes `_1`, `_2`, and `_3` denote $k_2=0.5$, $1$, and $2$, respectively. In the main ODGM library, the rejection-region parameter is fixed at $k_1=2$.

## Figures

The `figures/` directory contains scripts for both the main text and the supplementary material. `Figure1.py`, `Figure2.py`, `Figure4_5.py`, and `Figure6.py` generate figures in the main text, with `Figure4_5.py` generating both Figures 4 and 5. Files beginning with `FigureB` generate supplementary figures. `FigureB1.py` generates Figure B1, `FigureB2_B3.py` generates Figures B2 and B3, `FigureB4_B5.py` generates Figures B4 and B5, and `FigureB6_B7.py` generates Figures B6 and B7.

Check the configuration block near the top of each script before reproducing all figures.

For scripts that generate paired figures, select the target figure in the configuration block. For example:

```python
FIGURE_TO_CREATE = "Figure4"
```

or:

```python
FIGURE_TO_CREATE = "Figure5"
```

Generated figure files are saved under:

```text
figures/
├── Figure1.pdf
├── Figure2.pdf
├── Figure4.pdf
├── Figure5.pdf
├── Figure6.pdf
├── Figure7.pdf
└── supplementary/
    ├── FigureB1.pdf
    ├── FigureB2.pdf
    ├── FigureB3.pdf
    ├── FigureB4.pdf
    ├── FigureB5.pdf
    ├── FigureB6.pdf
    └── FigureB7.pdf
```

## License

This repository is released under the MIT License. See `LICENSE` for details.

**References**

Rhee, S.-Y., Taylor, J., Fessel, W. J., Kaufman, D., Towner, W., Troia, P., Ruane, P., Hellinger, J., Shirvani, V., Zolopa, A., and Shafer, R. W. (2006). Genotypic predictors of human immunodeficiency virus type 1 drug resistance. *Proceedings of the National Academy of Sciences*, 103(46), 17355–17360. https://doi.org/10.1073/pnas.0607274103

Azaïs, J. M., & De Castro, Y. (2022). Multiple testing and variable selection along the path of the least angle regression. Information and Inference: A Journal of the IMA, 11(4), 1329-1388.
GitHub repository: https://github.com/ydecastro/lar_testing/
