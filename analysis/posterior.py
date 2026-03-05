import json
import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arviz as az
import jax
import numpy as np
import scipy.signal
from jax import config as jax_config

from utils import load_config_from_model_dir, load_model_from_model_dir

# Import utils from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

jax_config.update("jax_enable_x64", True)


def find_peaks(grid, pdf):
    """
    Find local maxima in the PDF.
    Returns list of dicts: {'x': val, 'density': val} sorted by density descending.
    """
    # Find indices of peaks
    peaks_idx, properties = scipy.signal.find_peaks(pdf)

    peaks = []
    for idx in peaks_idx:
        peaks.append({"x": float(grid[idx]), "density": float(pdf[idx])})

    # Sort by density descending
    peaks.sort(key=lambda p: p["density"], reverse=True)
    return peaks


def run(experiment_dir: str):
    """
    Analyze the posterior distribution from MCMC results.
    """
    if not experiment_dir:
        print("Error: experiment_dir must be provided.")
        return

    experiment_dir = os.path.abspath(experiment_dir)
    if not os.path.isdir(experiment_dir):
        print(f"Error: Experiment directory does not exist: {experiment_dir}")
        return

    print(f"Analyzing Experiment Directory: {experiment_dir}")

    # 1. Find NetCDF file
    nc_files = [f for f in os.listdir(experiment_dir) if f.endswith(".nc")]
    if not nc_files:
        print(f"Error: No NetCDF (.nc) file found in {experiment_dir}")
        return
    if len(nc_files) > 1:
        print(
            f"Error: Multiple NetCDF files found in {experiment_dir}. Please ensure only one exists."
        )
        return

    netcdf_path = os.path.join(experiment_dir, nc_files[0])
    print(f"Found NetCDF file: {netcdf_path}")

    # 2. Load Config and Model from 'model' subdirectory
    model_dir = os.path.join(experiment_dir, "model")
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        print("Expected structure: <experiment_dir>/model/ with config.py and model.py")
        return

    try:
        config_module = load_config_from_model_dir(model_dir)
        experiment_config = config_module.experiment_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Identify calibration parameters
    param_names = [p.name for p in experiment_config.parameters]

    # 3. Load Data
    idata = az.from_netcdf(netcdf_path)
    posterior = idata.posterior

    # 4. Calculate Prior Means
    try:
        Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)

        # Reconstruct tminmax from config
        tminmax = {p.name: tuple(p.range) for p in experiment_config.parameters}

        prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
        prior_leaves, _ = jax.tree.flatten(prior_dict)

        prior_means_physical = {}
        for p in prior_leaves:
            try:
                # If distribution mean is available
                mean_unconstrained = p.distribution.mean
                # Convert to physical
                mean_physical = p.forward(mean_unconstrained)
                # Handle JAX scalar
                mean_physical = float(np.array(mean_physical))
                prior_means_physical[p.name] = mean_physical
            except Exception as e:
                print(f"Could not calculate prior mean for {p.name}: {e}")
                prior_means_physical[p.name] = None
    except Exception as e:
        print(f"Error calculating priors: {e}")
        prior_means_physical = {}

    # 5. Analyze Posterior
    results = {
        "experiment_dir": experiment_dir,
        "netcdf_file": netcdf_path,
        "parameters": {},
    }

    # Iterate over relevant parameters
    for var_name in param_names:
        if var_name not in posterior.data_vars:
            print(f"Warning: {var_name} not found in posterior.")
            continue

        var_data = posterior[var_name]  # (chain, draw)
        n_chains = var_data.shape[0]

        chain_stats = []
        all_peaks = []

        # Per chain analysis
        for chain_idx in range(n_chains):
            chain_samples = var_data[chain_idx].values

            # Mean
            mean_val = float(np.mean(chain_samples))

            # KDE
            try:
                grid, pdf = az.kde(chain_samples)
                peaks = find_peaks(grid, pdf)
            except Exception as e:
                print(f"KDE failed for {var_name} chain {chain_idx}: {e}")
                peaks = []

            chain_stats.append({"chain": chain_idx, "mean": mean_val, "peaks": peaks})
            all_peaks.append(peaks)

        # Grand Mean
        grand_mean = float(np.mean(var_data.values))

        # Average Peaks
        num_peaks_per_chain = [len(p) for p in all_peaks]
        if num_peaks_per_chain:
            N = min(num_peaks_per_chain)
        else:
            N = 0

        avg_peaks = []
        if N > 0:
            for i in range(N):
                # Valid peak index i
                x_vals = [p[i]["x"] for p in all_peaks]
                dens_vals = [p[i]["density"] for p in all_peaks]

                avg_peaks.append(
                    {
                        "rank": i + 1,
                        "avg_x": float(np.mean(x_vals)),
                        "avg_density": float(np.mean(dens_vals)),
                    }
                )

        results["parameters"][var_name] = {
            "prior_mean": prior_means_physical.get(var_name),
            "grand_mean": grand_mean,
            "chains": chain_stats,
            "average_peaks": avg_peaks,
        }

    # 6. Save JSON
    output_path = os.path.join(experiment_dir, "posterior_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Analysis saved to {output_path}")
