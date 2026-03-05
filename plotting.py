from typing import Dict, Tuple

import arviz
import cola
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from gpjax.distributions import GaussianDistribution
from kohgpjax import KOHDataset

plot_style = {
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "axes.labelpad": 2.0,
    "figure.dpi": 150,
}


def plot_sim_sample(
    kohdataset: KOHDataset,
    tminmax: Dict[str, Tuple[float, float]],
    ycmean: float,
    num_samples: int = None,
    alpha: float = 0.9,
    sample_step: int = 1,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a sample of the simulation data.
    Args:
        kohdataset: KOHDataset object containing the simulation data.
        tminmax: dictionary containing the minimum and maximum values for each calibration parameter.
        ycmean: mean value of the y-coordinates for the simulation data.
        num_samples: number of samples to plot. If None, plot all.
        alpha: transparency of the simulation lines.
    Returns:
        Tuple containing the figure and axes of the plot.
    """
    xf = np.array(kohdataset.Xf)
    yf = np.array(kohdataset.z + ycmean)
    xc = np.array(kohdataset.Xc[:, 0])
    yc = np.array(kohdataset.y + ycmean)
    tc = np.array(kohdataset.Xc[:, 1:])
    # scale tc from [0, 1] to the original range
    for i, (_, (tmin, tmax)) in enumerate(tminmax.items()):
        tc[:, i] = tc[:, i] * (tmax - tmin) + tmin

    fig, ax = plt.subplots(1, 1)

    ax.scatter(xf, yf, label="Observations", color="black", zorder=10)

    unique_ts = np.unique(tc, axis=0)

    # Thin the simulation runs by sample_step
    ts_thinned = unique_ts[::sample_step]

    if num_samples is not None and num_samples < len(ts_thinned):
        rng = np.random.default_rng()
        ts = rng.permutation(ts_thinned)[:num_samples]
    else:
        ts = ts_thinned

    # Use a color cycle for each simulation
    color_cycle = plt.cm.get_cmap("tab10", len(ts))

    for i, t in enumerate(ts):
        rows = np.all(tc == t, axis=1)
        x_rows = xc[rows]
        y_rows = yc[rows]
        sort_idx = np.argsort(x_rows)

        label_vals = [f"{ti:.2f}" for ti in t]
        label = f"Sim {i + 1}: $t$=({', '.join(label_vals)})"

        ax.plot(
            x_rows[sort_idx],
            y_rows[sort_idx],
            "--",
            label=label,
            alpha=alpha,
            color=color_cycle(i),
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    return fig, ax


def plot_pairwise_samples(
    traces,
    var_names: list[str],
    figsize=(6, 6),
    plot_style=plot_style,
) -> plt.Axes:
    with plt.style.context(plot_style):
        axes = arviz.plot_pair(
            traces,
            var_names=var_names,
            figsize=figsize,
        )
        axes[0, 0].figure.tight_layout()

    return axes


def plot_posterior_chains_with_priors(
    traces,
    config,
    model_parameters,
    tminmax: Dict[str, Tuple[float, float]],
    tracer_index_dict: dict[str, int],
    figsize=(9, 14),
    plot_style=plot_style,
    **kwargs,
) -> plt.Axes:
    # Plot the prior, posterior, true values (if known) and chains
    with plt.style.context(plot_style):
        axes = arviz.plot_trace(
            traces,
            figsize=figsize,
            legend=True,
            compact=False,
            # lines=lines,
            **kwargs,
        )

    if model_parameters is not None and tracer_index_dict is not None:
        for i in range(axes.shape[0]):
            title = axes[i, 0].get_title()

            left, right = axes[i, 0].get_xlim()
            left, right = left * 0.9, right * 1.1
            x = np.linspace(left, right, 1000)
            x_pdf = x

            # Transform the x-axis to a range suitable for the theta prior distributions
            jacobian = 1.0
            if title in tminmax:
                tmin, tmax = tminmax[title]
                x_pdf = (x_pdf - tmin) / (tmax - tmin)
                jacobian = 1.0 / (tmax - tmin)

            prior_dist = model_parameters.priors_flat[
                tracer_index_dict[title]
            ].distribution
            pdf = np.exp(prior_dist.log_prob(x_pdf)) * jacobian

            # Scaling factor to make prior visible against the posterior KDE
            # Arviz KDEs are densities, so this should ideally be 1.0 if we want to compare densities.
            # However, keeping a scaling factor might be useful if the posterior is very peaked.
            # Removing the hardcoded 1000 factor to try and match density scale.
            axes[i, 0].plot(x, pdf, color="red", linestyle="--", label="Prior")
            axes[i, 0].legend()

    # Fix overlap
    if axes.size > 0:
        axes.ravel()[0].figure.tight_layout()

    return axes


def gen_x_test_data(
    xmin: float = 0.0,
    xmax: float = 4.0,
    num_points: int = 1000,
) -> np.ndarray:
    """Generate the test data for the GP model.
    Args:
        xmin: The minimum value of x.
        xmax: The maximum value of x.
        num_points: The number of points to generate.

    Returns:
        x_test: The input values for GP predictions. shape (num_points, 2)
    """
    x_test_0 = np.linspace(xmin, xmax, num_points)
    x_test_1 = np.zeros_like(x_test_0)
    x_test = np.column_stack((x_test_0, x_test_1))
    return x_test


def gen_GP_test_data(
    theta_vec: np.ndarray,
    num_calib_params: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the test data for the GP model.
    Args:
        theta_vec: The array of theta values for the GP model.
        num_calib_params: The number of calibration parameters.
    Returns:
        GP_test: The input values for GP predictions.
    """
    x_test = gen_x_test_data()[:, 0].reshape(-1, 1)  # take only the first column
    theta_arr = np.tile(
        theta_vec[:num_calib_params],  # Take only the first num_calib_params elements
        (x_test.shape[0], 1),
    )
    GP_test = np.hstack((x_test, theta_arr))
    return GP_test


def gen_plot_test_data(
    theta_vec: np.ndarray,
    num_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the test data for plotting.
    Args:
        theta_vec: The array of theta values for the GP model.
    Returns:
        plot_test_x: The input values for true funcs.
        plot_test_theta: The theta values for true funcs.
    """
    x_test = gen_x_test_data(num_points=num_points)
    theta_arr = np.tile(theta_vec, (x_test.shape[0], 1))
    return x_test, theta_arr


def plot_GP(
    ax: plt.Axes,
    x: np.ndarray,
    distribution: GaussianDistribution,
    y_translation: float = 0.0,
    label: str = "GP",
    alpha: float = 0.3,
    **kwargs,
) -> plt.Axes:
    """
    Plot the GP distribution on the given axes.

    Args:
        ax: The matplotlib Axes to plot on.
        x: The input values for the GP.
        distribution: The GaussianDistribution object containing the GP mean and covariance.

    Returns:
        The updated Axes with the GP plot.
    """
    mean = distribution.mean
    cov = distribution.variance
    sd = jnp.sqrt(cov)

    ax.plot(x, mean + y_translation, label=label, **kwargs)
    ax.fill_between(
        x,
        y_translation + mean - 1.96 * sd,
        y_translation + mean + 1.96 * sd,
        alpha=alpha,
        **kwargs,
    )

    return ax


def plot_f_eta(
    config,
    x_test: np.ndarray,
    test_GP: np.ndarray,
    thetas_test: np.ndarray,
    GP_eta: GaussianDistribution,
    y_translation: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    x0_plot = x_test[:, 0]
    true_params = {p["name"]: p["true_value"] for p in config.PARAMETERS}
    thetas_true = np.tile(np.array(list(true_params.values())), (x0_plot.shape[0], 1))

    theta_label_components = [
        f"{theta:.2f}" for theta in thetas_test[0, : config.N_CALIB_PARAMS]
    ]
    theta_label = ", ".join(theta_label_components)

    ax.plot(
        x0_plot,
        config.eta(x_test, thetas_true),
        color="black",
        linestyle="--",
        label="True process",
    )
    ax.plot(
        x0_plot, config.eta(x_test, thetas_test), label=rf"$\eta(x, {theta_label})$"
    )
    plot_GP(
        ax,
        test_GP[:, 0],
        GP_eta,
        y_translation=y_translation,
        label=rf"$f_\eta(x, {theta_label})$",
        color="orange",
    )

    ax.legend()
    fig.suptitle("Computer simulation and GP model of simulator")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax


def plot_f_zeta(
    config,
    x_test: np.ndarray,
    test_GP: np.ndarray,
    thetas_test: np.ndarray,
    GP_zeta: GaussianDistribution,
    GP_zeta_epsilon: GaussianDistribution,
    scatter_xf: np.ndarray,
    scatter_yf: np.ndarray,
    y_translation: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    x0_plot = x_test[:, 0]

    theta_label_components = [
        f"{theta:.2f}" for theta in thetas_test[0, : config.N_CALIB_PARAMS]
    ]
    theta_label = ", ".join(theta_label_components)

    ax = plot_GP(
        ax,
        test_GP[:, 0],
        GP_zeta_epsilon,
        y_translation=y_translation,
        label=rf"$f_{{\zeta+\epsilon}}(x, {theta_label})$ GP reconstruction",
        color="orange",
    )

    ax = plot_GP(
        ax,
        test_GP[:, 0],
        GP_zeta,
        y_translation=y_translation,
        label=rf"$f_\zeta(x, {theta_label})$ GP reconstruction",
        color="green",
    )

    ax.plot(
        x0_plot,
        config.zeta(x_test),
        color="k",
        linestyle="--",
        label=r"$\zeta(x)$ True process",
    )
    ax.scatter(scatter_xf, scatter_yf + y_translation, label="Observations")

    ax.legend()
    fig.suptitle("True process, observations and GP reconstruction of $\zeta$")
    ax.set_xlabel("x")
    ax.set_ylabel("Z")
    return fig, ax


def plot_f_delta(
    config,
    x_test: np.ndarray,
    test_GP: np.ndarray,
    thetas_test: np.ndarray,
    delta_gp_mean: np.ndarray,
    delta_gp_cov: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(nrows=1, ncols=1)

    x0_plot = x_test[:, 0]

    theta_label_components = [
        f"{theta:.2f}" for theta in thetas_test[0, : config.N_CALIB_PARAMS]
    ]
    theta_label = ", ".join(theta_label_components)

    ax.plot(
        x0_plot,
        config.zeta(x_test) - config.eta(x_test, thetas_test),
        label=rf"$\delta(x) = \zeta(x) - \eta(x, {theta_label})$",
    )

    plot_GP(
        ax,
        test_GP[:, 0],
        GaussianDistribution(
            loc=delta_gp_mean,
            scale=cola.ops.Dense(delta_gp_cov),
        ),
        label=r"$f_\delta(x)$",
        color="orange",
    )

    ax.plot(
        x0_plot,
        config.discrepancy(x_test),
        color="black",
        linestyle="--",
        label="True discrepancy",
    )

    ax.legend()
    fig.suptitle("GP model of discrepancy function $\delta(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"\delta")

    return fig, ax
