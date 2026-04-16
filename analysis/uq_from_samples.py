import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple, cast

# Add parent directory to path to allow importing utils and datahandler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datahandler import load
from kohgpjax.parameters import ModelParameters
from utils import load_config_from_model_dir, load_model_from_model_dir


def _parse_space_dims(space_dims_arg: str | None) -> List[str] | None:
    if not space_dims_arg:
        return None
    dims = [d.strip() for d in space_dims_arg.split(",") if d.strip()]
    return dims if dims else None


def _load_manifest(manifest_path: str) -> pd.DataFrame:
    required_cols = ["run_name", "sample_index", "chain", "draw"]
    manifest = pd.read_csv(manifest_path)

    missing = [c for c in required_cols if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    manifest = manifest.copy()
    manifest["sample_index"] = manifest["sample_index"].astype(int)
    manifest["chain"] = manifest["chain"].astype(int)
    manifest["draw"] = manifest["draw"].astype(int)

    if manifest["sample_index"].duplicated().any():
        raise ValueError("Manifest sample_index values must be unique.")

    manifest = manifest.sort_values("sample_index").reset_index(drop=True)

    expected = np.arange(len(manifest))
    if not np.array_equal(manifest["sample_index"].to_numpy(), expected):
        raise ValueError("Manifest sample_index must be contiguous and start at 0.")

    if manifest[["chain", "draw"]].duplicated().any():
        raise ValueError("Manifest (chain, draw) pairs must be unique.")

    return manifest


def _infer_sample_dim(
    g_da: xr.DataArray,
    manifest: pd.DataFrame,
    sample_dim_arg: str | None,
) -> str:
    if sample_dim_arg:
        if sample_dim_arg not in g_da.dims:
            raise ValueError(
                f"Provided sample dimension '{sample_dim_arg}' not found in g variable dims {g_da.dims}."
            )
        return sample_dim_arg

    n_samples = len(manifest)
    preferred = ["sample_index", "sample", "run", "draw"]

    for dim in preferred:
        if dim in g_da.dims and g_da.sizes[dim] == n_samples:
            return dim

    candidates = [d for d in g_da.dims if g_da.sizes[d] == n_samples]
    if len(candidates) == 1:
        return str(candidates[0])

    if not candidates:
        raise ValueError(
            "Could not infer sample dimension: no dimension length matches manifest row count. "
            "Provide --sample_dim explicitly."
        )

    raise ValueError(
        f"Could not infer sample dimension unambiguously. Candidates: {candidates}. "
        "Provide --sample_dim explicitly."
    )


def _align_g_samples_with_manifest(
    g_da: xr.DataArray,
    sample_dim: str,
    manifest: pd.DataFrame,
) -> xr.DataArray:
    n_samples = len(manifest)
    if g_da.sizes[sample_dim] != n_samples:
        raise ValueError(
            f"G sample dimension '{sample_dim}' has size {g_da.sizes[sample_dim]}, "
            f"but manifest has {n_samples} rows."
        )

    coord_values = None
    if sample_dim in g_da.coords:
        coord_values = np.asarray(g_da.coords[sample_dim].values)

    if coord_values is not None:
        sample_indices = manifest["sample_index"].to_numpy()
        run_names = manifest["run_name"].astype(str).to_numpy()

        # Try sample_index labels first
        if np.issubdtype(coord_values.dtype, np.number):
            if np.isin(sample_indices, coord_values).all():
                return g_da.sel({sample_dim: sample_indices})

        # Then try run_name labels
        coord_as_str = coord_values.astype(str)
        if np.isin(run_names, coord_as_str).all():
            return g_da.sel({sample_dim: run_names})

    # Fall back to positional assumption; safe because sample_index is contiguous and size matched.
    return g_da


def _prepare_space_grid(
    g_da: xr.DataArray,
    sample_dim: str,
    space_dims: List[str] | None,
    expected_num_variable_params: int,
) -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray]:
    resolved_space_dims: List[str]
    if space_dims is None:
        resolved_space_dims = [str(d) for d in g_da.dims if str(d) != sample_dim]
    else:
        resolved_space_dims = [str(d) for d in space_dims]

    missing = [d for d in resolved_space_dims if d not in g_da.dims]
    if missing:
        raise ValueError(f"Requested space dims missing from g variable: {missing}")

    if len(resolved_space_dims) != expected_num_variable_params:
        raise ValueError(
            f"Number of space dims ({len(resolved_space_dims)}) must match model variable input dimensions "
            f"({expected_num_variable_params}). Got space dims {resolved_space_dims}."
        )

    coords: Dict[str, np.ndarray] = {}
    for d in resolved_space_dims:
        if d in g_da.coords:
            coords[d] = np.asarray(g_da.coords[d].values)
        else:
            coords[d] = np.arange(g_da.sizes[d])

    mesh = np.meshgrid(*[coords[d] for d in resolved_space_dims], indexing="ij")
    x_points = np.column_stack([m.reshape(-1) for m in mesh])

    return resolved_space_dims, coords, x_points


def _build_chain_draw_index_map(
    posterior_ds: xr.Dataset,
    manifest: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("Posterior NetCDF group must include 'chain' and 'draw' dimensions.")

    chain_coords = np.asarray(posterior_ds["chain"].values)
    draw_coords = np.asarray(posterior_ds["draw"].values)

    chain_map = {int(v): i for i, v in enumerate(chain_coords)}
    draw_map = {int(v): i for i, v in enumerate(draw_coords)}

    chains = manifest["chain"].to_numpy()
    draws = manifest["draw"].to_numpy()

    missing_chain = sorted(set(int(c) for c in chains if int(c) not in chain_map))
    missing_draw = sorted(set(int(d) for d in draws if int(d) not in draw_map))

    if missing_chain:
        raise ValueError(f"Manifest references chain values not present in posterior: {missing_chain}")
    if missing_draw:
        raise ValueError(f"Manifest references draw values not present in posterior: {missing_draw}")

    chain_idx = np.array([chain_map[int(c)] for c in chains], dtype=int)
    draw_idx = np.array([draw_map[int(d)] for d in draws], dtype=int)
    return chain_idx, draw_idx


def _extract_posterior_samples(
    posterior_ds: xr.Dataset,
    manifest: pd.DataFrame,
    required_vars: List[str],
) -> Dict[str, np.ndarray]:
    missing = [v for v in required_vars if v not in posterior_ds.data_vars]
    if missing:
        raise ValueError(f"Posterior NetCDF missing required variables: {missing}")

    chain_idx, draw_idx = _build_chain_draw_index_map(posterior_ds, manifest)

    selected: Dict[str, np.ndarray] = {}
    for v in required_vars:
        arr = np.asarray(posterior_ds[v].values)
        if arr.ndim != 2:
            raise ValueError(f"Posterior variable '{v}' must be 2D (chain, draw).")
        vals = arr[chain_idx, draw_idx]
        if not np.isfinite(vals).all():
            raise ValueError(f"Posterior variable '{v}' has non-finite values at referenced chain/draw indices.")
        selected[v] = vals

    return selected


def _theta_norm_from_physical(
    theta_physical: Dict[str, float],
    tminmax: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    out = dict(theta_physical)
    for name, value in theta_physical.items():
        if name not in tminmax:
            continue
        tmin, tmax = tminmax[name]
        norm = (value - tmin) / (tmax - tmin)
        out[name] = float(norm)
    return out


def _compute_summary_fields(values: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "p05": np.quantile(values, 0.05, axis=0),
        "p50": np.quantile(values, 0.50, axis=0),
        "p95": np.quantile(values, 0.95, axis=0),
    }


def _scalar_metrics(target_values: np.ndarray, truth: np.ndarray | None) -> Dict[str, float]:
    out = {
        "global_mean": float(np.mean(target_values)),
        "global_std": float(np.std(target_values)),
    }

    if truth is not None:
        target_mean = np.mean(target_values, axis=0)
        diff = target_mean - truth
        out["mean_bias"] = float(np.mean(diff))
        out["rmse_of_mean"] = float(np.sqrt(np.mean(diff**2)))
        out["ensemble_rmse"] = float(np.sqrt(np.mean((target_values - truth[None, :]) ** 2)))

        q05 = np.quantile(target_values, 0.05, axis=0)
        q95 = np.quantile(target_values, 0.95, axis=0)
        out["coverage_90"] = float(np.mean((truth >= q05) & (truth <= q95)))

    return out


def _plot_distribution_histograms(
    output_dir: str,
    targets: Dict[str, np.ndarray],
    truth: np.ndarray | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 40

    ax.hist(targets["g_only"].ravel(), bins=bins, alpha=0.35, label="G only", density=True)
    ax.hist(
        targets["g_plus_delta"].ravel(),
        bins=bins,
        alpha=0.35,
        label="G + delta",
        density=True,
    )
    ax.hist(
        targets["g_plus_delta_plus_epsilon"].ravel(),
        bins=bins,
        alpha=0.35,
        label="G + delta + epsilon",
        density=True,
    )

    if truth is not None:
        ax.hist(truth.ravel(), bins=bins, alpha=0.35, label="Truth", density=True)

    ax.set_title("Distribution Comparison")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distribution_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_profile_if_1d(
    output_dir: str,
    x: np.ndarray,
    targets: Dict[str, np.ndarray],
    truth: np.ndarray | None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    style = [
        ("g_only", "G only", "tab:blue"),
        ("g_plus_delta", "G + delta", "tab:green"),
        ("g_plus_delta_plus_epsilon", "G + delta + epsilon", "tab:red"),
    ]

    for key, label, color in style:
        arr = targets[key]
        mean = np.mean(arr, axis=0)
        p05 = np.quantile(arr, 0.05, axis=0)
        p95 = np.quantile(arr, 0.95, axis=0)

        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, p05, p95, color=color, alpha=0.20)

    if truth is not None:
        ax.plot(x, truth, color="black", linewidth=1.4, linestyle="--", label="Truth")

    ax.set_title("Profile Comparison")
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "profile_comparison.png"), dpi=150)
    plt.close(fig)


def run(
    model_dir: str,
    manifest_path: str,
    g_samples_path: str,
    chain_path: str,
    output_dir: str,
    g_var: str = "precip",
    sample_dim: str | None = None,
    space_dims: List[str] | None = None,
    truth_path: str | None = None,
    truth_var: str | None = None,
    composition_mode: str = "deterministic",
    random_seed: int = 0,
) -> None:
    if composition_mode not in {"deterministic", "stochastic"}:
        raise ValueError("composition_mode must be either 'deterministic' or 'stochastic'.")

    os.makedirs(output_dir, exist_ok=True)

    # Load model config, data, and priors
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config
    ModelClass, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)

    kohdataset, tminmax, yc_mean = load(experiment_config=experiment_config, data_root="data")
    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)
    model = cast(Any, ModelClass)(
        model_parameters=model_parameters,
        kohdataset=kohdataset,
    )

    # Load and validate manifest
    manifest = _load_manifest(manifest_path)

    # Load G samples and align with manifest
    g_ds = xr.open_dataset(g_samples_path, decode_times=False)
    if g_var not in g_ds:
        raise ValueError(f"Variable '{g_var}' not found in {g_samples_path}.")

    g_da = g_ds[g_var]
    inferred_sample_dim = _infer_sample_dim(g_da, manifest, sample_dim)
    g_da = _align_g_samples_with_manifest(g_da, inferred_sample_dim, manifest)

    if len(manifest) != g_da.sizes[inferred_sample_dim]:
        raise ValueError("Manifest row count does not match aligned G sample count.")

    space_dims, space_coords, x_points = _prepare_space_grid(
        g_da=g_da,
        sample_dim=inferred_sample_dim,
        space_dims=space_dims,
        expected_num_variable_params=kohdataset.num_variable_params,
    )

    g_arr = g_da.transpose(inferred_sample_dim, *space_dims).to_numpy()
    n_samples = g_arr.shape[0]
    space_shape = g_arr.shape[1:]
    n_space = int(np.prod(space_shape))
    g_flat = g_arr.reshape(n_samples, n_space)

    if not np.isfinite(g_flat).all():
        raise ValueError("G samples contain non-finite values.")

    # Load posterior chain metadata and posterior group samples from NetCDF
    posterior_root_ds = xr.open_dataset(chain_path, decode_times=False)
    if "ycmean" not in posterior_root_ds.attrs:
        raise ValueError("Posterior NetCDF is missing required global attribute 'ycmean'.")

    posterior_ds = xr.open_dataset(chain_path, group="posterior", decode_times=False)

    required_vars = [p.name for p in model_parameters.priors_flat]
    posterior_samples = _extract_posterior_samples(posterior_ds, manifest, required_vars)

    # Allocate output arrays
    target_g_only = np.zeros((n_samples, n_space), dtype=float)
    target_g_plus_delta = np.zeros((n_samples, n_space), dtype=float)
    target_g_plus_delta_plus_epsilon = np.zeros((n_samples, n_space), dtype=float)

    delta_mean_all = np.zeros((n_samples, n_space), dtype=float)
    delta_var_all = np.zeros((n_samples, n_space), dtype=float)
    target_b_var_all = np.zeros((n_samples, n_space), dtype=float)
    target_c_var_all = np.zeros((n_samples, n_space), dtype=float)
    epsilon_var_all = np.zeros(n_samples, dtype=float)

    # Keep posterior theta in physical units for traceability
    theta_names = [p.name for p in experiment_config.parameters]
    theta_phys = {name: np.zeros(n_samples, dtype=float) for name in theta_names}
    theta_norm = {name: np.zeros(n_samples, dtype=float) for name in theta_names}

    x_points_jnp = jnp.asarray(x_points)
    key = jax.random.PRNGKey(random_seed)

    for i in range(n_samples):
        sample_param_values = {name: float(posterior_samples[name][i]) for name in required_vars}

        for name in theta_names:
            if name in sample_param_values:
                theta_phys[name][i] = sample_param_values[name]

        sample_param_values_norm = _theta_norm_from_physical(sample_param_values, tminmax)

        for name in theta_names:
            if name in sample_param_values_norm:
                theta_norm[name][i] = sample_param_values_norm[name]

        flat_values = []
        for prior in model_parameters.priors_flat:
            if prior.name not in sample_param_values_norm:
                raise ValueError(f"Missing parameter '{prior.name}' for sample index {i}.")
            flat_values.append(np.asarray(sample_param_values_norm[prior.name]))

        params_tree = model_parameters.unflatten_sample(flat_values)

        theta_vector = np.array([sample_param_values_norm[name] for name in theta_names], dtype=float)
        theta_jnp = jnp.asarray(theta_vector).reshape(1, -1)

        train_data = kohdataset.get_dataset(theta_jnp)
        theta_tiled = jnp.tile(theta_jnp, (n_space, 1))
        test_inputs = jnp.hstack((x_points_jnp, theta_tiled))

        posterior_gp = model.GP_posterior(params_tree)
        delta_dist = posterior_gp.predict_delta(test_inputs, train_data=train_data)

        delta_mean = np.asarray(delta_dist.mean).reshape(-1)
        delta_var = np.asarray(delta_dist.variance).reshape(-1)
        delta_var = np.clip(delta_var, a_min=0.0, a_max=None)

        eps_var = float(sample_param_values["epsilon_variance"])

        g_i = g_flat[i]

        target_a = g_i
        if composition_mode == "deterministic":
            target_b = g_i + delta_mean
            target_c = target_b.copy()
        else:
            key, k1, k2 = jax.random.split(key, 3)
            try:
                delta_sample = np.asarray(delta_dist.sample(k1, sample_shape=(1,)))[0]
            except (TypeError, ValueError, RuntimeError):
                delta_sample = delta_mean + np.asarray(
                    jax.random.normal(k1, shape=(n_space,))
                ) * np.sqrt(delta_var)

            epsilon_sample = np.asarray(jax.random.normal(k2, shape=(n_space,))) * np.sqrt(
                max(eps_var, 0.0)
            )
            target_b = g_i + delta_sample
            target_c = target_b + epsilon_sample

        target_g_only[i] = target_a
        target_g_plus_delta[i] = target_b
        target_g_plus_delta_plus_epsilon[i] = target_c

        delta_mean_all[i] = delta_mean
        delta_var_all[i] = delta_var
        target_b_var_all[i] = delta_var
        target_c_var_all[i] = delta_var + eps_var
        epsilon_var_all[i] = eps_var

    # Optional truth data
    truth_flat = None
    if truth_path is not None:
        truth_ds = xr.open_dataset(truth_path, decode_times=False)
        truth_name = truth_var if truth_var else g_var
        if truth_name not in truth_ds:
            raise ValueError(f"Truth variable '{truth_name}' not found in {truth_path}.")

        truth_da = truth_ds[truth_name]
        if inferred_sample_dim in truth_da.dims:
            if truth_da.sizes[inferred_sample_dim] == 1:
                truth_da = truth_da.isel({inferred_sample_dim: 0})
            else:
                raise ValueError(
                    "Truth variable includes sample dimension with size > 1. "
                    "Provide a single truth field over space dimensions only."
                )

        missing_truth_dims = [d for d in space_dims if d not in truth_da.dims]
        if missing_truth_dims:
            raise ValueError(
                f"Truth variable missing required space dims: {missing_truth_dims}."
            )

        truth_flat = truth_da.transpose(*space_dims).to_numpy().reshape(-1)
        if truth_flat.shape[0] != n_space:
            raise ValueError("Truth field shape does not match G sample spatial shape.")

    # Assemble per-sample output dataset
    def _reshape(values: np.ndarray) -> np.ndarray:
        return values.reshape((n_samples, *space_shape))

    coords = {"sample_index": manifest["sample_index"].to_numpy()}
    for d in space_dims:
        coords[d] = space_coords[d]

    uq_ds = xr.Dataset(
        data_vars={
            "g_input": (("sample_index", *space_dims), _reshape(g_flat)),
            "target_g_only": (("sample_index", *space_dims), _reshape(target_g_only)),
            "target_g_plus_delta": (
                ("sample_index", *space_dims),
                _reshape(target_g_plus_delta),
            ),
            "target_g_plus_delta_plus_epsilon": (
                ("sample_index", *space_dims),
                _reshape(target_g_plus_delta_plus_epsilon),
            ),
            "delta_mean": (("sample_index", *space_dims), _reshape(delta_mean_all)),
            "delta_variance": (("sample_index", *space_dims), _reshape(delta_var_all)),
            "target_b_variance": (("sample_index", *space_dims), _reshape(target_b_var_all)),
            "target_c_variance": (("sample_index", *space_dims), _reshape(target_c_var_all)),
            "epsilon_variance": ("sample_index", epsilon_var_all),
            "posterior_chain": ("sample_index", manifest["chain"].to_numpy()),
            "posterior_draw": ("sample_index", manifest["draw"].to_numpy()),
        },
        coords=coords,
        attrs={
            "model_dir": model_dir,
            "chain_path": chain_path,
            "manifest_path": manifest_path,
            "g_samples_path": g_samples_path,
            "composition_mode": composition_mode,
            "ycmean": str(yc_mean),
        },
    )

    for name in theta_names:
        uq_ds[f"posterior_{name}_physical"] = ("sample_index", theta_phys[name])
        uq_ds[f"posterior_{name}_normalized"] = ("sample_index", theta_norm[name])

    # Carry through all manifest columns as sample-indexed metadata variables.
    for col in manifest.columns:
        if col == "sample_index":
            continue
        uq_ds[f"meta_{col}"] = ("sample_index", manifest[col].to_numpy())

    if truth_flat is not None:
        uq_ds["truth"] = (space_dims, truth_flat.reshape(space_shape))

    uq_samples_path = os.path.join(output_dir, "uq_samples.nc")
    uq_ds.to_netcdf(uq_samples_path)

    targets = {
        "g_only": target_g_only,
        "g_plus_delta": target_g_plus_delta,
        "g_plus_delta_plus_epsilon": target_g_plus_delta_plus_epsilon,
    }

    # Summary dataset over samples
    summary_fields = {}
    for key, values in targets.items():
        fields = _compute_summary_fields(values)
        for stat_name, stat_values in fields.items():
            summary_fields[f"{key}_{stat_name}"] = (space_dims, stat_values.reshape(space_shape))

    summary_fields["delta_variance_mean"] = (
        space_dims,
        np.mean(delta_var_all, axis=0).reshape(space_shape),
    )
    summary_fields["epsilon_variance_mean"] = (
        space_dims,
        np.mean(target_c_var_all - target_b_var_all, axis=0).reshape(space_shape),
    )

    if truth_flat is not None:
        summary_fields["truth"] = (space_dims, truth_flat.reshape(space_shape))
        summary_fields["g_only_bias_of_mean"] = (
            space_dims,
            (np.mean(target_g_only, axis=0) - truth_flat).reshape(space_shape),
        )
        summary_fields["g_plus_delta_bias_of_mean"] = (
            space_dims,
            (np.mean(target_g_plus_delta, axis=0) - truth_flat).reshape(space_shape),
        )
        summary_fields["g_plus_delta_plus_epsilon_bias_of_mean"] = (
            space_dims,
            (np.mean(target_g_plus_delta_plus_epsilon, axis=0) - truth_flat).reshape(space_shape),
        )

    summary_ds = xr.Dataset(data_vars=summary_fields, coords={d: space_coords[d] for d in space_dims})
    summary_ds.attrs["composition_mode"] = composition_mode

    uq_summary_path = os.path.join(output_dir, "uq_summary.nc")
    summary_ds.to_netcdf(uq_summary_path)

    # Scalar JSON summary
    scalar_summary = {
        "inputs": {
            "model_dir": model_dir,
            "manifest_path": manifest_path,
            "g_samples_path": g_samples_path,
            "chain_path": chain_path,
            "n_samples": int(n_samples),
            "space_dims": space_dims,
            "space_shape": [int(x) for x in space_shape],
            "composition_mode": composition_mode,
        },
        "targets": {
            "g_only": _scalar_metrics(target_g_only, truth_flat),
            "g_plus_delta": _scalar_metrics(target_g_plus_delta, truth_flat),
            "g_plus_delta_plus_epsilon": _scalar_metrics(
                target_g_plus_delta_plus_epsilon, truth_flat
            ),
        },
    }

    json_path = os.path.join(output_dir, "uq_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scalar_summary, f, indent=2)

    # Plots
    _plot_distribution_histograms(output_dir=output_dir, targets=targets, truth=truth_flat)
    if len(space_dims) == 1:
        x = np.asarray(space_coords[space_dims[0]])
        _plot_profile_if_1d(output_dir=output_dir, x=x, targets=targets, truth=truth_flat)

    print(f"Saved per-sample outputs: {uq_samples_path}")
    print(f"Saved summary outputs:   {uq_summary_path}")
    print(f"Saved scalar summary:    {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Input-agnostic UQ from provided G samples + posterior discrepancy/noise."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--g_samples_path", type=str, required=True)
    parser.add_argument("--chain_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--g_var", type=str, default="precip")
    parser.add_argument("--sample_dim", type=str, default=None)
    parser.add_argument(
        "--space_dims",
        type=str,
        default=None,
        help="Comma-separated space dimensions, e.g. 'lat' or 'level,latitude'.",
    )

    parser.add_argument("--truth_path", type=str, default=None)
    parser.add_argument("--truth_var", type=str, default=None)

    parser.add_argument(
        "--composition_mode",
        type=str,
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    run(
        model_dir=args.model_dir,
        manifest_path=args.manifest_path,
        g_samples_path=args.g_samples_path,
        chain_path=args.chain_path,
        output_dir=args.output_dir,
        g_var=args.g_var,
        sample_dim=args.sample_dim,
        space_dims=_parse_space_dims(args.space_dims),
        truth_path=args.truth_path,
        truth_var=args.truth_var,
        composition_mode=args.composition_mode,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
