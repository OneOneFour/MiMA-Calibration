import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datahandler import load
from kohgpjax.parameters import ModelParameters
from utils import load_config_from_model_dir, load_model_from_model_dir


def run(model_dir, file_path, dpi=300, output_dir=None):
    # Load config
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config

    # Load data
    kohdataset, tminmax, ycmean = load(experiment_config, "data")

    # Load chains
    chains = az.from_netcdf(file_path)
    posterior_means = chains.posterior.mean()

    # Prepare GP parameters
    Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)
    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)
    model_parameters = ModelParameters(prior_dict=prior_dict)

    model = Model(model_parameters=model_parameters, kohdataset=kohdataset)

    # Flatten priors to map posterior means to GP params
    prior_leaves, prior_tree = jax.tree.flatten(prior_dict)
    GP_params_flat = {p.name: posterior_means[p.name].values for p in prior_leaves}

    # Normalize theta parameters for GP
    for i in range(experiment_config.n_calib_params):
        # Assuming parameters are named theta_0, theta_1 etc in the model priors
        # But config has names. We need to map.
        # In T21.py, config param name is "theta_0".
        # In get_ModelParameterPriorDict, it uses "theta_0".
        # So names should match.

        # However, experiment_config.parameters[i].name might be "theta_0".
        # tminmax keys are "theta_0".

        # We iterate over config parameters
        p_name = experiment_config.parameters[i].name

        # Check if this parameter is in GP_params_flat
        if p_name in GP_params_flat:
            tmin, tmax = tminmax[p_name]
            val = GP_params_flat[p_name]
            # Normalize to [0, 1]
            GP_params_flat[p_name] = (val - tmin) / (tmax - tmin)

    GP_params = jax.tree.unflatten(prior_tree, GP_params_flat.values())

    # Build GP posterior
    GP_posterior = model.GP_posterior(GP_params)

    # Generate test points (use observation locations)
    # Xf is (N, 1) for T21 (latitudes)
    X_test = kohdataset.Xf

    # --- Prior Predictive Check ---
    print("Generating prior predictions...")
    n_prior_samples = 50
    prior_preds = []
    key = jax.random.PRNGKey(42)

    for _ in range(n_prior_samples):
        key, subkey = jax.random.split(key)

        # Sample all parameters from prior
        prior_sample_flat = {}
        for p in model_parameters.priors_flat:
            subkey, sk = jax.random.split(subkey)
            # Sample from the distribution (constrained space)
            val = p.distribution.sample(sk)
            prior_sample_flat[p.name] = val

        # Reconstruct GP_params
        # Note: Thetas are already in [0, 1] if sampled from Beta.
        GP_params_prior = jax.tree.unflatten(prior_tree, prior_sample_flat.values())

        # Build GP
        GP_posterior_prior = model.GP_posterior(GP_params_prior)

        # Prepare test inputs with sampled theta
        theta_vals_prior = []
        for i in range(experiment_config.n_calib_params):
            p_name = experiment_config.parameters[i].name
            # Handle potential name mismatch if any
            if p_name in prior_sample_flat:
                theta_vals_prior.append(prior_sample_flat[p_name])
            else:
                theta_vals_prior.append(prior_sample_flat[f"theta_{i}"])

        theta_arr_prior = jnp.array(theta_vals_prior).reshape(1, -1)
        theta_tiled_prior = jnp.tile(theta_arr_prior, (X_test.shape[0], 1))
        test_GP_prior = jnp.hstack([X_test, theta_tiled_prior])

        dataset_prior = kohdataset.get_dataset(theta_arr_prior)

        # Predict
        pred = GP_posterior_prior.predict_obs(test_GP_prior, dataset_prior)

        # Sample a realization from the prediction
        subkey, sk = jax.random.split(subkey)
        # pred.mean is (N,), pred.variance is (N,) (if diagonal) or covariance (N, N)
        # Assuming we want marginal variance for plotting
        mean_vec = pred.mean
        std_vec = jnp.sqrt(pred.variance)

        # Sample from independent Gaussians for visualization (approximation)
        # or just store the mean +/- 2sigma?
        # Let's store the mean for now to show the spread of the process mean
        prior_preds.append(mean_vec + ycmean)

    prior_preds = jnp.array(prior_preds)
    prior_mean = jnp.mean(prior_preds, axis=0)
    prior_lower = jnp.percentile(prior_preds, 2.5, axis=0)
    prior_upper = jnp.percentile(prior_preds, 97.5, axis=0)

    # --- Posterior Predictive Check ---
    # We need to augment X_test with theta for the emulator prediction
    # We use the posterior mean of theta
    theta_gp_vals = []
    for i in range(experiment_config.n_calib_params):
        p_name = experiment_config.parameters[i].name
        if p_name in GP_params_flat:
            theta_gp_vals.append(GP_params_flat[p_name])
        else:
            # Fallback if name mismatch, though unlikely if config is consistent
            # Try theta_{i}
            theta_gp_vals.append(GP_params_flat[f"theta_{i}"])

    theta_gp_arr = jnp.array(theta_gp_vals).reshape(1, -1)
    # Tile for all X points
    theta_tiled = jnp.tile(theta_gp_arr, (X_test.shape[0], 1))

    test_GP = jnp.hstack([X_test, theta_tiled])

    # Dataset for prediction context
    dataset = kohdataset.get_dataset(theta_gp_arr)

    # Predict
    print("Generating posterior predictions...")
    # predict_obs gives f_zeta + epsilon (observation process)
    obs_pred = GP_posterior.predict_obs(test_GP, dataset)

    # Extract mean and variance
    mean = obs_pred.mean + ycmean
    std = jnp.sqrt(obs_pred.variance)

    # Plot
    if output_dir:
        save_dir = output_dir
    else:
        save_dir = os.path.join(os.path.dirname(file_path), "ppc")

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort for plotting
    sort_idx = np.argsort(X_test[:, 0])
    x_sorted = X_test[sort_idx, 0]

    # Plot Prior Predictive
    ax.fill_between(
        x_sorted,
        prior_lower[sort_idx],
        prior_upper[sort_idx],
        color="gray",
        alpha=0.3,
        label="Prior Predictive (95% CI)",
    )

    # Plot observations
    # kohdataset.z is centered, add ycmean
    obs_y = kohdataset.z + ycmean
    ax.scatter(X_test[:, 0], obs_y, color="black", label="Observations", s=10, zorder=5)

    # Plot GP prediction (mean + 95% CI)
    mean_sorted = mean[sort_idx]
    std_sorted = std[sort_idx]

    ax.plot(x_sorted, mean_sorted, color="blue", label="Posterior Mean Prediction")
    ax.fill_between(
        x_sorted,
        mean_sorted - 1.96 * std_sorted,
        mean_sorted + 1.96 * std_sorted,
        color="blue",
        alpha=0.2,
        label="Posterior Predictive (95% CI)",
    )

    ax.set_xlabel(experiment_config.data.obs_coord_names[0])
    ax.set_ylabel(experiment_config.data.obs_variable_name)
    ax.legend()
    ax.set_title(f"Posterior Predictive Check - {experiment_config.name}")

    plt.savefig(os.path.join(save_dir, "ppc.png"), dpi=dpi)
    plt.close()
    print(f"Saved PPC plot to {save_dir}/ppc.png")
