import glob
import os
from datetime import datetime
from typing import Tuple

import arviz
import gpjax as gpx
import jax.numpy as jnp
import kohgpjax as kgx
import numpy as np
import pandas as pd
import xarray as xr
from jax import config

try:
    from config_schema import ExperimentConfig
except ImportError:
    from .config_schema import ExperimentConfig

config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX


def apply_thinning(ds: xr.Dataset, thinning: dict) -> xr.Dataset:
    """Thin the dataset along specified dimensions.

    Args:
        ds (xr.Dataset): Dataset to thin.
        thinning (dict): Dictionary mapping dimension names to thinning factors.
            Factor n means keep 1/n of data (indices [0, n, 2n...]) approximately.
            For fractional n (e.g. 1.5), we pick indices floor(i * n).

    Returns:
        xr.Dataset: Thinned dataset.
    """
    if not thinning:
        return ds

    for dim, factor in thinning.items():
        if dim in ds.dims:
            size = ds.sizes[dim]
            if factor > 1:
                # Calculate indices for fractional thinning
                # E.g. factor=1.5 -> indices 0, 1, 3, 4, ...
                num_points = int(np.ceil(size / factor))
                indices = [int(i * factor) for i in range(num_points)]
                # Ensure valid indices
                indices = [i for i in indices if i < size]

                print(
                    f"[DataHandler] Thinning dimension '{dim}' by factor {factor} (size {size} -> {len(indices)})"
                )
                ds = ds.isel({dim: indices})
    return ds


def load(
    experiment_config: ExperimentConfig, data_root: str
) -> Tuple[kgx.KOHDataset, dict, float]:
    """Load the simulation and observation data from CSV files and prepare them for modeling.
    Args:
        experiment_config (ExperimentConfig): Configuration object.
        data_root (str): Root directory where the data files are located.
    Returns:
        Tuple[kgx.KOHDataset, Dict[str, Tuple[float, float]], float]:
            - kohdataset: A KOHDataset containing the field and component datasets.
            - tminmax: A dictionary with the min and max values for each calibration parameter.
            - ycmean: The mean of the centered output data.
    """
    # Construct paths using config
    base_dir = os.path.join(data_root, experiment_config.data.experiment_name)

    obs_path = os.path.join(base_dir, experiment_config.data.obs_data_path)

    sim_variable_name = experiment_config.data.sim_variable_name
    obs_variable_name = experiment_config.data.obs_variable_name

    # Branch: single NetCDF file containing all runs
    if getattr(experiment_config.data, "sim_data_path", None):
        sim_data_path = os.path.join(base_dir, experiment_config.data.sim_data_path)
        ds = xr.open_dataset(sim_data_path, decode_times=False)

        # Determine the run dimension and parameter coordinates
        dims = list(ds.dims)

        sim_coord_names = experiment_config.data.sim_coord_names

        # We need to identify the run dimension.
        # It's the dimension that is NOT a spatial/time coordinate and NOT the variable itself (though variable is not a dim).
        # We assume sim_coord_names (e.g. ['lat']) cover the spatial dims.

        run_dims = [
            d
            for d in dims
            if d not in sim_coord_names
            and d != sim_variable_name
            and d != "phalf"
            and d != "pfull"
            and d != "time"
            and d != "lon"
        ]

        # Heuristic: if 'N' is present, it's the run dim. Else if 'rh' is present, it's the run dim.
        if "N" in dims:
            run_dim = "N"
        elif "rh" in dims:
            run_dim = "rh"
        elif "t" in dims:
            run_dim = "t"
        else:
            # Fallback: take the first non-lat dim
            run_dim = run_dims[0] if run_dims else "rh"

        da = ds[sim_variable_name]

        # Ensure we have a dataset with the variable
        sim_data = xr.Dataset({sim_variable_name: da})

        # Apply thinning
        if getattr(experiment_config, "thinning", None):
            sim_data = apply_thinning(sim_data, experiment_config.thinning)

        # If parameters are not in sim_data coords (e.g. if loaded from separate file?), but here we loaded from NC.
        # We assume rh/tau are in coords if they define the run.

        # Define stack dimensions: (run_dim, 'coord1', 'coord2', ...)
        # This ensures we get Size = N_runs * N_spatial_points
        stack_dims = [run_dim] + sim_coord_names
        # Verify these dims exist
        stack_dims = [d for d in stack_dims if d in sim_data.dims]

        sim_data_flat = sim_data.stack(sample=stack_dims)

    else:
        # --- Load simulation parameters ---
        sim_params_path = os.path.join(base_dir, experiment_config.data.sim_params_path)
        t_sim_df = pd.read_csv(sim_params_path, header=None)

        # --- Load data files ---
        sim_pattern = os.path.join(base_dir, experiment_config.data.sim_data_pattern)
        sim_files = sorted(glob.glob(sim_pattern))

        def select_vars(ds):
            return ds[[sim_variable_name]]

        sim_data = xr.open_mfdataset(
            sim_files,
            combine="nested",
            concat_dim="t",
            preprocess=select_vars,
        )

        # Apply thinning
        if getattr(experiment_config, "thinning", None):
            sim_data = apply_thinning(sim_data, experiment_config.thinning)

        # Assign coords from t_sim_df
        # If t_sim_df has 1 col, assign to 't'.
        # If multiple, assign appropriately.
        # For legacy compatibility, assume 't' is the run index/dim.
        sim_data = sim_data.assign_coords(t=np.arange(len(t_sim_df)))

        # Add parameter values as coordinates
        for i in range(t_sim_df.shape[1]):
            # Name them param_0, param_1 etc or try to match config?
            # Let's assign them to the 't' dimension
            sim_data = sim_data.assign_coords({f"param_{i}": ("t", t_sim_df[i])})

        stack_dims = ["t"] + experiment_config.data.sim_coord_names
        sim_data_flat = sim_data.stack(sample=stack_dims)

    # --- Prepare field data ---
    obs_data = xr.open_dataset(obs_path)

    # Apply thinning to observations
    if getattr(experiment_config, "thinning", None):
        obs_data = apply_thinning(obs_data, experiment_config.thinning)

    # Extract Observation Coordinates
    # We support multiple coords by stacking them column-wise
    # But we check for dims_present first.
    pass

    # Ensure consistent flattening between coordinates and variable
    # Stack observation dims in same order as sim coord list if possible, or just flatten.
    # But sim and obs might have different names (lat vs latitude).
    # We need to flatten obs_data[{obs_variable_name}] and extract corresponding coordinates.

    # Let's stack obs_data using obs_coord_names
    # If there are multiple, result is 1D array of value, and Xf is (N_obs, N_coords)

    # Check if obs_data has the coordinate names as dimensions
    dims_present = [
        d for d in experiment_config.data.obs_coord_names if d in obs_data.dims
    ]

    if len(dims_present) > 0:
        # Stack
        # We need to be careful: if we stack, we get a MultiIndex.
        # .data will give values.
        # .coords will give coordinate values.

        # We want to flatten the variable first
        # But if we just .values.flatten(), the order depends on dimension order.
        # We should explicitly stack or transpose to ensure order matches our expectation (and matches Sim data order logic).
        # The sim data logic was: stack_dims = [run_dim] + sim_coord_names.
        # So it iterates run_dim first (slowest?), then coord1, then coord2...
        # Wait, xarray stack is last-dim-fastest?
        # Actually stack implementation typically does outer product.

        # For Observation data, we just want the spatial/time points.
        # Let's transpose to matching order of names provided in config, then flatten.

        # IMPORTANT: sim_data_flat was created with [run_dim] + sim_coords.
        # So within each run, the order is determined by sim_coords order.
        # We must ensure obs_data follows obs_coords order.

        # E.g. if sim_coords=["lat"], obs_coords=["latitude"].
        # We want obs to be ordered by latitude.

        # Transpose
        obs_da = obs_data[obs_variable_name]

        # Only transpose if dimensions match
        # obs_da might have dims (time, lat) etc.
        # We only care about dims in obs_coord_names.
        # If there are extra dims not in obs_coord_names, what do we do?
        # The original code assumed obs was (lat,).

        # Safe approach: stack using obs_coord_names
        obs_da_stacked = obs_da.stack(sample=experiment_config.data.obs_coord_names)

        yf = obs_da_stacked.to_numpy().reshape(-1, 1).astype(jnp.float64)

        # Now extract Xf columns
        xf_cols = []
        for coord in experiment_config.data.obs_coord_names:
            # After stacking, coordinates are accessible in the MultiIndex or as coords of stacked array
            # xarray handles this well.
            vals = obs_da_stacked[coord].to_numpy()
            xf_cols.append(vals.reshape(-1, 1))

        xf = jnp.hstack(xf_cols).astype(jnp.float64)

    else:
        # Fallback if no dims found (maybe scalar or error)
        # Original code did:
        # xf = jnp.array(obs_data["lat"]).reshape(-1, 1).astype(jnp.float64)
        # yf = jnp.array(obs_data[variable_name]).reshape(-1, 1).astype(jnp.float64)
        raise ValueError(
            f"Could not find observation coordinates {experiment_config.data.obs_coord_names} in {obs_path}"
        )

    # --- Build Xc matrix ---
    # We need to construct Xc = [sim_coords, param1, param2, ...]

    # Extract sim coordinates
    # sim_data_flat is already stacked by [run_dim] + sim_coord_names
    # So we just extract sim_coord_names

    xc_coords_list = []
    for coord in experiment_config.data.sim_coord_names:
        vals = sim_data_flat[coord].to_numpy().reshape(-1, 1)
        xc_coords_list.append(vals)

    xc_coords = jnp.hstack(xc_coords_list).astype(jnp.float64)

    tminmax = {p.name: tuple(p.range) for p in experiment_config.parameters}
    t_normalized_list = []

    # Heuristic mapping
    # Check if calib_coord_names is provided
    calib_map = experiment_config.data.calib_coord_names

    for i, param in enumerate(experiment_config.parameters):
        vals = None

        # Name to look for:
        # 1. Map from calib_coord_names if available
        # 2. param.name

        search_names = []
        if calib_map and i < len(calib_map):
            search_names.append(calib_map[i])
        search_names.append(param.name)

        # Other heuristics (legacy) can be added or we rely on robust config now.
        if param.name == "theta_0":
            search_names.append("rh")
        if param.name == "theta_1":
            search_names.append("tau")

        for name in search_names:
            if name in sim_data_flat.coords:
                vals = sim_data_flat[name].to_numpy()
                break
            # Check for param_i fallback
            if f"param_{i}" in sim_data_flat.coords:
                vals = sim_data_flat[f"param_{i}"].to_numpy()
                break

        # Fallback for T21 legacy: if 't' contains the parameter value
        if (
            vals is None
            and "t" in sim_data_flat.coords
            and i == 0
            and len(experiment_config.parameters) == 1
        ):
            vals = sim_data_flat["t"].to_numpy()

        if vals is None:
            raise ValueError(
                f"Could not locate simulation values for parameter {param.name}"
            )

        vals = vals.reshape(-1, 1)

        # Normalize
        pmin, pmax = param.range
        p_norm = (vals - pmin) / (pmax - pmin)
        t_normalized_list.append(p_norm)

    # Compose Xc: [sim_coords, norm_param1, norm_param2, ...]
    Xc = jnp.hstack([xc_coords] + t_normalized_list)

    # Extract simulation outputs to match Xc
    yc = sim_data_flat[sim_variable_name].to_numpy().reshape(-1, 1).astype(jnp.float64)

    # --- Center the outputs ---
    yc_mean = jnp.mean(yc, axis=0)
    yc_centered = yc - yc_mean  # Center the outputs
    yf_centered = yf - yc_mean  # Center the outputs

    # --- Create KOHDataset ---
    field_dataset = gpx.Dataset(X=xf, y=yf_centered)
    comp_dataset = gpx.Dataset(X=Xc, y=yc_centered)
    koh_dataset = kgx.KOHDataset(field_dataset, comp_dataset)

    # --- Apply Filters if defined in config ---
    if getattr(experiment_config, "filters", None):
        koh_dataset = filter_dataset(
            koh_dataset, experiment_config.filters, experiment_config, tminmax
        )

    return koh_dataset, tminmax, yc_mean


def transform_chains(traces, model_parameters, prior_dict, tminmax):
    """Transforms the MCMC chains."""
    traces_transformed = {}
    for var, trace in traces.items():
        if var == "hamiltonian":
            continue

        # Find the prior corresponding to this variable
        # model_parameters.priors_flat is a list of ParameterPrior objects
        # We need to find the one with name == var
        prior = next((p for p in model_parameters.priors_flat if p.name == var), None)

        if prior:
            # Transform from unconstrained to constrained space (e.g. log to linear)
            traces_transformed[var] = prior.forward(np.array(trace))
        else:
            # If not found in priors (shouldn't happen for sampled vars), keep as is
            traces_transformed[var] = np.array(trace)

        # If it's a calibration parameter (theta), un-normalize it
        # The prior_dict structure is typically {'thetas': {'theta_0': ...}, ...}
        if "thetas" in prior_dict and var in prior_dict["thetas"]:
            trace_val = traces_transformed[var]
            if var in tminmax:
                tmin, tmax = tminmax[var]
                # Scale from [0, 1] back to [tmin, tmax]
                traces_transformed[var] = (np.array(trace_val) * (tmax - tmin)) + tmin

    return traces_transformed


def thin_runs_by_div(data: np.ndarray, div: int, x_dim: int = 1) -> np.ndarray:
    """Thin the runs by a specified divisor.
    Args:
        data (np.ndarray): The data to be thinned.
        div (int): The divisor for thinning.
        x_dim (int): The number of control/regression variables.
    Returns:
        np.ndarray: The thinned data.
    """
    if div <= 1:
        return data

    unique_params = np.unique(data[:, (x_dim + 1) :], axis=0)

    thinned_data = np.empty((0, data.shape[1]), dtype=data.dtype)
    for params in unique_params:
        mask = np.all(data[:, (x_dim + 1) :] == params, axis=1)
        thinned_subset = data[mask][::div]
        thinned_data = np.vstack((thinned_data, thinned_subset))
    return thinned_data


def save_chains_to_netcdf(
    raw_traces,
    transformed_traces,
    file_name: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    n_sim: int,
    ycmean: float,
    inference_library_name: str,
    output_dir: str = None,
) -> None:
    """
    Save the MCMC traces to a NetCDF file in Arviz InferenceData format.

    Args:
        raw_traces: The raw MCMC traces.
        transformed_traces: The transformed MCMC traces.
        file_name: The base name for the output file (used if output_dir is None).
        n_warm_up_iter: Number of warm-up iterations.
        n_main_iter: Number of main iterations.
        n_sim: Number of simulation output points.
        ycmean: The mean of the centered output data.
        inference_library_name: The name of the inference library used.
        output_dir: The directory to save the output file.
    """
    # Create the directory for the experiment if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join("chains", file_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference_data = arviz.from_dict(
        posterior=transformed_traces,
        unconstrained_posterior=raw_traces,
    )
    inference_data.attrs["ycmean"] = str(ycmean)
    inference_data.attrs["inference_library"] = inference_library_name
    inference_data.attrs["created_at"] = datetime.now().isoformat()

    # Use a fixed name "posterior.nc" if saving to a specific run directory,
    # otherwise keep the descriptive name for backward compatibility or if output_dir is generic.
    # But to be safe and consistent with the new plan, let's use the descriptive name inside the run folder.
    # Or better: just use "posterior.nc" inside the run folder to make it easy to find.

    # Let's stick to the descriptive name for now to avoid confusion, but place it in the run folder.
    filename = f"W{n_warm_up_iter}-N{n_main_iter}-Nsim{n_sim}.nc"
    inference_data.to_netcdf(os.path.join(output_dir, filename))
    print(f"Saved chains to {os.path.join(output_dir, filename)}")


def filter_dataset(
    kohdataset: kgx.KOHDataset,
    filters: dict,
    experiment_config: ExperimentConfig,
    tminmax: dict,
) -> kgx.KOHDataset:
    """
    Filter the KOHDataset based on variable ranges.

    Args:
        kohdataset: The dataset to filter.
        filters: Dictionary defining filters. Keys can be coordinate names or parameter names.
                 Values are tuples (min, max). Use None for no bound.
                 Example: {'lat': [-70, 70], 'theta_0': [0.35, None]}
        experiment_config: Experiment configuration.
        tminmax: Dictionary of parameter ranges (min, max) for denormalization.

    Returns:
        Filtered KOHDataset.
    """
    import numpy as np

    print(f"\n[Filter Dataset] Applying filters: {filters}")
    print(f"Original Xc: {kohdataset.Xc.shape}, y: {kohdataset.y.shape}")
    print(f"Original Xf: {kohdataset.Xf.shape}, z: {kohdataset.z.shape}")

    # --- Filter Simulation Data (Xc, y) ---
    # Start with all true
    keep_sim = np.ones(kohdataset.Xc.shape[0], dtype=bool)

    # 1. Filter by Sim Coordinates
    # Sim coords are the first N columns of Xc, where N = len(sim_coord_names)
    sim_coord_names = experiment_config.data.sim_coord_names

    for k, coord_name in enumerate(sim_coord_names):
        if coord_name in filters:
            # Column index k
            coord_vals = kohdataset.Xc[:, k]
            fmin, fmax = filters[coord_name]
            if fmin is not None:
                keep_sim &= coord_vals >= fmin
            if fmax is not None:
                keep_sim &= coord_vals <= fmax

    # 2. Filter by Parameters (Columns 1..N of Xc)
    # Map parameter names to column indices
    # Xc columns: [Lat, Param1_Norm, Param2_Norm, ...]
    # Params are in order of experiment_config.parameters
    for i, param in enumerate(experiment_config.parameters):
        if param.name in filters:
            # Column index is i + len(sim_coord_names)
            col_idx = i + len(sim_coord_names)
            norm_vals = kohdataset.Xc[:, col_idx]

            # Denormalize
            if param.name in tminmax:
                pmin, pmax = tminmax[param.name]
                phys_vals = norm_vals * (pmax - pmin) + pmin
            else:
                # Should strict fail or warn? tminmax should have it.
                phys_vals = norm_vals  # fallback

            fmin, fmax_val = filters[param.name]
            if fmin is not None:
                keep_sim &= phys_vals >= fmin
            if fmax_val is not None:
                keep_sim &= phys_vals <= fmax_val

    # Apply simulation mask
    Xc_filtered = kohdataset.Xc[keep_sim]
    y_filtered = kohdataset.y[keep_sim]

    # --- Filter Field Data (Xf, z) ---
    keep_field = np.ones(kohdataset.Xf.shape[0], dtype=bool)

    # 1. Filter by Obs Coordinates
    # Obs coords are columns of Xf
    # NB: We assume obs_coord_names map one-to-one to Xf columns in order.
    # But filters might use the *simulation* coordinate name (e.g. "lat") but technically they should imply the same physical dimension.
    # Usually users filter by "lat". If obs has "latitude", we might need to map?
    # For now, we assume users use the keys exactly as they appear in filters dict.
    # AND we assume filters keys match `sim_coord_names` mostly.
    # If `obs_coord_names` are different (e.g. "latitude"), we should check if filter key matches obs_coord_name.

    obs_coord_names = experiment_config.data.obs_coord_names
    for k, coord_name in enumerate(obs_coord_names):
        # We also check if "lat" is in filters and current coord is "latitude" (if we want to be smart).
        # But per instruction "Coordinate agnostic", we should respect config.

        # However, if sim uses "lat" and obs uses "latitude", user probably puts "lat" in filters.
        # We need a way to map filter keys (which likely match sim coords) to obs coords.

        # Simple heuristic: if len(sim) == len(obs), map by index.
        # If user filters on sim_coord_names[k], apply to obs_coord_names[k].

        # Check if filter exists for this dimension
        key_to_check = coord_name

        # Map sim name to obs name if possible
        if k < len(sim_coord_names):
            if sim_coord_names[k] in filters:
                key_to_check = sim_coord_names[
                    k
                ]  # Check the key corresponding to sim name

        if key_to_check in filters:
            fmin, fmax = filters[key_to_check]
            coord_vals = kohdataset.Xf[:, k]
            if fmin is not None:
                keep_field &= coord_vals >= fmin
            if fmax is not None:
                keep_field &= coord_vals <= fmax

        # Also check direct name
        if coord_name != key_to_check and coord_name in filters:
            fmin, fmax = filters[coord_name]
            coord_vals = kohdataset.Xf[:, k]
            if fmin is not None:
                keep_field &= coord_vals >= fmin
            if fmax is not None:
                keep_field &= coord_vals <= fmax

    # Apply field mask
    Xf_filtered = kohdataset.Xf[keep_field]
    z_filtered = kohdataset.z[keep_field]

    # Create new KOHDataset
    field_dataset_filtered = gpx.Dataset(X=Xf_filtered, y=z_filtered)
    comp_dataset_filtered = gpx.Dataset(X=Xc_filtered, y=y_filtered)
    new_dataset = kgx.KOHDataset(field_dataset_filtered, comp_dataset_filtered)

    print(f"Filtered Xc: {new_dataset.Xc.shape}, y: {new_dataset.y.shape}")
    print(f"Filtered Xf: {new_dataset.Xf.shape}, z: {new_dataset.z.shape}\n")

    return new_dataset
