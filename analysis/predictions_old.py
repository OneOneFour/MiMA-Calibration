import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
    W=None,
    N=None,
    chain_path=None,
    plot_observations=False,
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
    from kohgpjax.parameters import ModelParameters

    model_parameters = ModelParameters(prior_dict=prior_dict)

    # Optionally load posterior means
    means = None
    if chain_path is None and W is not None and N is not None:
        # Try to find latest run dir with W/N
        # Assumes experiments dir is in root
        exp_dir = os.path.join("experiments", file_name)
        if os.path.exists(exp_dir):
            candidates = sorted(
                [d for d in os.listdir(exp_dir) if d.endswith(f"_W{W}_N{N}")]
            )
            if candidates:
                latest = os.path.join(exp_dir, candidates[-1])
                # pick any .nc file
                nc_files = [f for f in os.listdir(latest) if f.endswith(".nc")]
                if nc_files:
                    chain_path = os.path.join(latest, nc_files[0])

    if chain_path is not None and os.path.exists(chain_path):
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

    # Predict on a grid covering the data range
    if kohdataset.Xf.shape[0] > 0:
        x_min = float(kohdataset.Xf.min())
        x_max = float(kohdataset.Xf.max())
    else:
        x_min, x_max = 0.0, 1.0

    X_pred = jnp.linspace(x_min, x_max, 200).reshape(-1, 1)
    # X_pred must have shape (N, p+q) where p is the number of input features and q is the number of theta parameters
    # This should be written generically for any number of input features and theta parameters
    X_pred = jnp.hstack((X_pred, jnp.ones((X_pred.shape[0], q)) * theta.T))

    train_data = kohdataset.get_dataset(theta)
    eta_pred = posterior_GP.predict_eta(X_pred, train_data=train_data)
    zeta_pred = posterior_GP.predict_zeta(X_pred, train_data=train_data)
    obs_pred = posterior_GP.predict_obs(X_pred, train_data=train_data)
    delta_pred = posterior_GP.predict_delta(X_pred, train_data=train_data)

    # Plot
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

    # Example from Brynjarsdottir & O'Hagan (2014)
    # def eta(x, t):
    #     """Simulation model: eta(x, t) = t * x"""
    #     return t * x

    # def zeta(x):
    #     """Reality model: zeta(x) = (theta * x) / (1 + x/a)"""
    #     THETA_TRUE = 0.65
    #     A = 20.0
    #     return (THETA_TRUE * x) / (1 + x / A)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
    dashed = (0, (5, 5))
    # ax.plot(
    #     X_pred[:, 0],
    #     eta(X_pred[:, 0], theta).squeeze(),
    #     color="y",
    #     linestyle="solid",
    #     label=rf"$\eta(x, {theta.squeeze()})$",
    # )
    # ax.plot(
    #     X_pred[:, 0],
    #     zeta(X_pred[:, 0]),
    #     color="m",
    #     linestyle="solid",
    #     label=r"$\zeta(x)$",
    # )
    plot_GP(
        axes[0],
        X_pred[:, 0],
        eta_pred,
        yc_mean,
        "blue",
        dashed,
        rf"$f_\eta(x, {theta.squeeze():.3f})$",
    )
    plot_GP(
        axes[0],
        X_pred[:, 0],
        obs_pred,
        yc_mean,
        "red",
        dashed,
        r"$f_{\zeta+\epsilon}(x)$",
    )
    if plot_observations:
        print("Plotting observations")
        axes[0].scatter(
            kohdataset.Xf,
            kohdataset.z + yc_mean,
            color="red",
            marker="x",
            label="Observations",
        )
    plot_GP(
        axes[0],
        X_pred[:, 0],
        zeta_pred,
        yc_mean,
        "green",
        dashed,
        r"$f_\zeta(x)$",
        alpha=0.22,
    )
    # axt = ax.twinx()
    plot_GP(
        axes[1],
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
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = axt.get_legend_handles_labels()
    # ax.legend(lines + lines2, labels + labels2, loc=0)

    if len(experiment_config.data.obs_coord_names) > 0:
        axes[1].set_xlabel(experiment_config.data.obs_coord_names[0])
    else:
        axes[1].set_xlabel("x")
    # axes[0].set_ylabel(experiment_config.data.obs_variable_name)
    axes[0].set_ylabel("Precipitation [mm/day]")
    axes[1].set_ylabel("Discrepancy")

    title = f"Predictions for {file_name}"
    axes[0].set_title(title)

    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(output_dir, "predictions.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
