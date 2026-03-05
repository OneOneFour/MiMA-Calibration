# MiMA Calibration Workflow

This directory contains a modular workflow for running Bayesian calibration experiments using the **MiMA** toolkit. The workflow is centralized around the `main.py` Command Line Interface (CLI).

## Workflow Overview

The typical workflow involves:
1.  **Running HMC Sampling**: Calibrate the model using MCMC (Mici or BlackJax backends).
2.  **Analysis & Plotting**: Generate diagnostic plots, predictive checks, and posterior analysis.

## Automated Pipeline: `run_analysis.sh`

The `run_analysis.sh` script automates the entire process: running MCMC, archiving model files, and generating all plots and analysis.

**Usage:**

```bash
./run_analysis_uv.sh <backend> <model_dir> <warmup> <main_iter> <sample_step> <n_chain>
```

or

```bash
./run_analysis_conda.sh <env> <backend> <model_dir> <warmup> <main_iter> <sample_step> <n_chain>
```

Note that `<sample_step>` is not about the MCMC, it refers to plotting every n'th simulation run in the plotting scripts. For example, with 15 simulations and using a value of 3 inplace of `<sample_step>` yields figures showing 5 simulation runs to demonstrate the spread of the simulations.

`./run_analysis_conda.sh` requires the name of the conda environment of your choosing.

**Example:**

```bash
./run_analysis_uv.sh mici models/T21_land 100 100 3 2
```

or

```bash
./run_analysis_conda.sh your_conda_env mici models/T21_land 100 100 3 2
```

## unified CLI: `main.py`

All main operations are performed using the `main.py` script.

### 1. Running MCMC (`run`)

Run Hamiltonian Monte Carlo sampling for a specific model.

```bash
python main.py run <model_dir> -W <warmup> -N <main_iter> --n_chain <chains> --backend <backend>
```

**Arguments:**
*   `model_dir`: Path to the model directory (e.g., `models/T21`).
*   `-W`, `--warmup`: Number of warmup iterations (default: 100).
*   `-N`, `--main`: Number of main sampling iterations (default: 100).
*   `--n_chain`: Number of chains (default: 1).
*   `--backend`: MCMC backend to use, either `mici` (default) or `blackjax`.

**Example:**
```bash
python main.py run models/T21 -W 100 -N 100 --backend mici
```

Output is saved to `experiments/<ModelName>/<Timestamp>_W<W>_N<N>/`.

### 2. Plotting Results (`plot`)

Generate various plots from the experiment results.

**Diagnostic Plots (Trace, Autocorr, ESS):**
```bash
python main.py plot trace --model_dir <experiment_run_dir>/model --output_dir <experiment_run_dir>
```

**Posterior Predictive Check (PPC):**
```bash
python main.py plot ppc --model_dir <experiment_run_dir>/model --file_path <experiment_run_dir>/<chain_file>.nc --output_dir <experiment_run_dir>/ppc
```

**Simulation Samples:**
```bash
python main.py plot sim_sample --model_dir <experiment_run_dir>/model --output_dir <experiment_run_dir>
```

**Predictions (Experimental):**
```bash
python main.py plot predictions --model_dir <experiment_run_dir>/model --output_dir <experiment_run_dir>
```

### 3. Analyzing Posterior (`analyze`)

Calculate statistics, find local optima, and export results to JSON.

```bash
python main.py analyze <experiment_run_dir>
```

**Example:**
```bash
python main.py analyze experiments/T21/20251212_120000_W100_N100
```

## Directory Structure

*   `main.py`: Main CLI entry point.
*   `runners/`: Contains MCMC runner implementations (`mici.py`, `blackjax.py`).
*   `analysis/`: Contains analysis and plotting scripts (`posterior.py`, `diagnostics.py`, etc.).
*   `models/`: Directory containing model definitions (each model has its own folder with `model.py` and `config.py`).
*   `experiments/`: Output directory where experiment results, plots, and analysis are saved.
