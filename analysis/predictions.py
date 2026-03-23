import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datahandler import load
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ModelParameterPriorDict, ModelParameters
from utils import load_config_from_model_dir, load_model_from_model_dir


def load_params_from_chain(chain_path: str):
    """
    Load posterior means from an ArviZ InferenceData NetCDF and return a dict of parameter means.
    If the file does not exist or loading fails, return None.
    """
    try:
        import arviz as az

        idata = az.from_netcdf(chain_path)
        summary = az.summary(idata, kind="stats")
        means = summary["mean"].to_dict()
        return means
    except Exception:
        return None


def build_params_constrained(
    model_parameters: ModelParameters,
    prior_dict: ModelParameterPriorDict,
    means_dict=None,
):
    """
    Build constrained parameter tree from either posterior means (if provided) or prior means.
    """
    # Start from prior means in unconstrained space
    prior_leaves, _ = jax.tree.flatten(prior_dict)
    prior_means_unconstrained = jax.tree.map(
        lambda x: x.inverse(x.distribution.mean), prior_leaves
    )

    # Map back to constrained
    constrained_flat = []
    for p in model_parameters.priors_flat:
        # Use posterior mean if available; otherwise use prior mean
        if means_dict is not None and p.name in means_dict:
            val = np.array(means_dict[p.name])
            constrained_flat.append(val)
        else:
            constrained_flat.append(
                p.forward(
                    np.array(prior_means_unconstrained[p.index])
                    if hasattr(p, "index")
                    else np.array(
                        prior_means_unconstrained[model_parameters.priors_flat.index(p)]
                    )
                )
            )

    params_constrained = model_parameters.unflatten_sample(constrained_flat)
    return params_constrained


def run(
    model_dir: str,
    output_dir: str,
    chain_path: str,
    plot_observations: bool = False,
):
    config_module = load_config_from_model_dir(model_dir)
    experiment_config = config_module.experiment_config
    file_name = experiment_config.name
    Model, get_ModelParameterPriorDict = load_model_from_model_dir(model_dir)

    kohdataset, tminmax, yc_mean = load(
        experiment_config=experiment_config,
        data_root="data",
    )

    prior_dict = get_ModelParameterPriorDict(config_module, tminmax)

    model_parameters = ModelParameters(prior_dict=prior_dict)

    # Optionally load posterior means
    means = None

    if not chain_path or not os.path.exists(chain_path):
        raise ValueError(f"Chain file {chain_path} does not exist.")

    means = load_params_from_chain(chain_path)

    # Pass Model class into build_params_constrained via calling scope
    params_constrained = build_params_constrained(model_parameters, prior_dict, means)

    model: KOHModel = Model(model_parameters=model_parameters, kohdataset=kohdataset)
    posterior_GP = model.GP_posterior(params_constrained)

    # extract each theta parameter (unknown number) and stack them into a single array
    theta = [params_constrained["thetas"][key] for key in params_constrained["thetas"]]
    theta = jnp.array(theta)
    theta = theta.reshape(1, -1)
    q = theta.shape[1]

    # Determine dimensionality
    num_sim_dims = kohdataset.Xc.shape[1] - q

    # Plot directory
    os.makedirs(output_dir, exist_ok=True)

    def plot_GP(ax, X, GP, yc_mean, color, linestyle, label, alpha=0.15):
        ax.plot(X, GP.mean + yc_mean, color=color, linestyle=linestyle, label=label)
        stddev = GP.stddev()
        ax.fill_between(
            X.flatten(),
            GP.mean + yc_mean - 2 * stddev,
            GP.mean + yc_mean + 2 * stddev,
            color=color,
            alpha=alpha,
        )

    if num_sim_dims == 2:
        print("Detected 2D inputs, generating 3D prediction plots...")

        # Grid generation for 2D
        # Use Simulation Bounds (Xc) to avoid extrapolation
        # columns 0 and 1 of Xc are the spatial coordinates
        x0_min, x0_max = (
            float(kohdataset.Xc[:, 0].min()),
            float(kohdataset.Xc[:, 0].max()),
        )
        x1_min, x1_max = (
            float(kohdataset.Xc[:, 1].min()),
            float(kohdataset.Xc[:, 1].max()),
        )

        # Create grid
        grid_res = 30
        x0_grid = jnp.linspace(x0_min, x0_max, grid_res)
        x1_grid = jnp.linspace(x1_min, x1_max, grid_res)
        X0, X1 = jnp.meshgrid(x0_grid, x1_grid)
        X_pred_2d = jnp.column_stack([X0.ravel(), X1.ravel()])

        # Append theta
        theta_tiled = jnp.tile(theta, (X_pred_2d.shape[0], 1))
        test_GP_2d = jnp.hstack([X_pred_2d, theta_tiled])

        # Predict on 2D grid
        train_data = kohdataset.get_dataset(theta)
        eta_pred_2d = posterior_GP.predict_eta(test_GP_2d, train_data=train_data)
        zeta_pred_2d = posterior_GP.predict_zeta(test_GP_2d, train_data=train_data)
        _obs_pred_2d = posterior_GP.predict_obs(test_GP_2d, train_data=train_data)
        delta_pred_2d = posterior_GP.predict_delta(test_GP_2d, train_data=train_data)

        # Plotting
        fig = plt.figure(figsize=(18, 6))

        # Subplot 1: f_eta (Simulator)
        ax1 = fig.add_subplot(131, projection="3d")
        _surf1 = ax1.plot_trisurf(
            X_pred_2d[:, 0],
            X_pred_2d[:, 1],
            eta_pred_2d.mean + yc_mean,
            cmap="viridis",
            alpha=0.8,
        )
        ax1.set_title(r"$f_\eta(x, \theta)$")
        ax1.set_xlabel("x0")
        ax1.set_ylabel("x1")

        # Subplot 2: f_zeta (Reality Process) + Obs
        ax2 = fig.add_subplot(132, projection="3d")
        _surf2 = ax2.plot_trisurf(
            X_pred_2d[:, 0],
            X_pred_2d[:, 1],
            zeta_pred_2d.mean + yc_mean,
            cmap="viridis",
            alpha=0.6,
        )

        if plot_observations and kohdataset.Xf.shape[1] >= 2:
            # Aggregate obs for plotting
            # datahandler.py filtering ensures Xf is within bounds now.
            obs_y = kohdataset.z + yc_mean
            df_obs = pd.DataFrame(kohdataset.Xf[:, :2], columns=["x0", "x1"])
            df_obs["y"] = obs_y

            # Group by unique locations and mean
            df_agg = df_obs.groupby(["x0", "x1"]).mean().reset_index()

            ax2.scatter(
                df_agg["x0"],
                df_agg["x1"],
                df_agg["y"],
                color="black",
                marker="o",
                s=40,
                label="Obs (Mean)",
                zorder=10,
            )

        ax2.set_title(r"$f_\zeta(x)$ + Obs")
        ax2.set_xlabel("x0")
        ax2.set_ylabel("x1")

        # Subplot 3: f_delta (Discrepancy)
        ax3 = fig.add_subplot(133, projection="3d")
        _surf3 = ax3.plot_trisurf(
            X_pred_2d[:, 0],
            X_pred_2d[:, 1],
            delta_pred_2d.mean,
            cmap="coolwarm",
            alpha=0.8,
        )
        ax3.set_title(r"$f_\delta(x)$")
        ax3.set_xlabel("x0")
        ax3.set_ylabel("x1")

        plt.suptitle(f"Predictions for {file_name} (3D)")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "predictions_3d.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")

    else:
        # Predict on a grid covering the data range (1D assumption)
        if kohdataset.Xf.shape[0] > 0:
            x_min = float(kohdataset.Xf.min())
            x_max = float(kohdataset.Xf.max())
        else:
            x_min, x_max = 0.0, 1.0

        X_pred = jnp.linspace(x_min, x_max, 200).reshape(-1, 1)
        # X_pred must have shape (N, p+q) where p is the number of input features and q is the number of theta parameters
        X_pred = jnp.hstack((X_pred, jnp.ones((X_pred.shape[0], q)) * theta.squeeze()))

        train_data = kohdataset.get_dataset(theta)
        eta_pred = posterior_GP.predict_eta(X_pred, train_data=train_data)
        zeta_pred = posterior_GP.predict_zeta(X_pred, train_data=train_data)
        obs_pred = posterior_GP.predict_obs(X_pred, train_data=train_data)
        delta_pred = posterior_GP.predict_delta(X_pred, train_data=train_data)

        # Standard 1D plotting
        fig, (ax,ax2) = plt.subplots(1,2,figsize=(13, 6))
        
        dashed = (0, (5, 5))

        theta_str = "".join([f"{val:.2f}, " for val in theta.flatten()])[:-2]

        plot_GP(
            ax,
            X_pred[:, 0],
            eta_pred,
            yc_mean,
            "blue",
            dashed,
            rf"$f_\eta(x, {theta_str})$",
        )
        plot_GP(
            ax,
            X_pred[:, 0],
            obs_pred,
            yc_mean,
            "red",
            dashed,
            r"$f_{\zeta+\epsilon}(x)$",
        )
        if plot_observations:
            print("Plotting observations")
            ax.scatter(
                kohdataset.Xf,
                kohdataset.z + yc_mean,
                color="red",
                marker="x",
                label="Observations",
            )
        plot_GP(
            ax,
            X_pred[:, 0],
            zeta_pred,
            yc_mean,
            "green",
            dashed,
            r"$f_\zeta(x)$",
            alpha=0.22,
        )
        plot_GP(
            ax2,
            X_pred[:, 0],
            delta_pred,
            0,
            "black",
            dashed,
            r"$f_\delta(x)$",
            alpha=0.1,
        )

        # See link below for combining legends from two axes
        # https://stackoverflow.com/a/10129461
        lines, labels = ax.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines , labels , loc=0)

        if len(experiment_config.data.obs_coord_names) > 0:
            ax.set_xlabel(experiment_config.data.obs_coord_names[0])
        else:
            ax.set_xlabel("x")
        ax2.set_ylabel("Discrepancy")
        ax.set_ylabel(experiment_config.data.obs_variable_name)

        title = f"Predictions for {file_name}"
        ax.set_title(title)

        fig.tight_layout()
        out_path = os.path.join(output_dir, "predictions.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
