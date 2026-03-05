from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_time_vs_sim_output_single(
    fig: plt.Figure,
    axes: List[plt.Axes],  # of 3 axes for real, user, and sys time
    path_to_csv: str,
    label: str,
    marker: str = "o",
    # color: str = "blue",
    linestyle: str = "-",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot MCMC time against number of simulator outputs for simulation.

    Args:
        fig: The matplotlib Figure to plot on.
        ax: The matplotlib Axes to plot on.
        path_to_csv: Path to the CSV file containing the data.
        label: Label for the plot.
        marker: Marker style for the plot.
        color: Color for the plot.
        linestyle: Line style for the plot.

    Returns:
        The updated Figure and Axes with the plot.
    """
    data = np.genfromtxt(path_to_csv, delimiter=",", skip_header=1)
    divs = data[:, 3]
    x = np.array([int(300 / d) for d in divs])
    y_real = data[:, 4]  # Real time in seconds
    y_user = data[:, 5]  # User time in seconds
    y_sys = data[:, 6]  # System time in seconds

    axes[0].plot(x, y_real, marker=marker, linestyle=linestyle, label=label)
    axes[1].plot(x, y_user, marker=marker, linestyle=linestyle, label=label)
    axes[2].plot(x, y_sys, marker=marker, linestyle=linestyle, label=label)

    return fig, axes


def plot_time_vs_sim_output() -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot MCMC time against number of simulator outputs for simulation.

    Args:"""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    fig.suptitle("MCMC time vs number of simulator outputs")
    axes[0].set_ylim(top=2000)
    axes[1].set_ylim(top=500)
    axes[2].set_ylim(top=80)
    axes[0].set_title('"Real"')
    axes[1].set_title('"User"')
    axes[2].set_title('"Sys"')
    axes[0].set_xlabel("Number of simulator outputs")
    axes[1].set_xlabel("Number of simulator outputs")
    axes[2].set_xlabel("Number of simulator outputs")
    axes[0].set_ylabel("Time (s)")
    axes[1].set_ylabel("Time (s)")
    axes[2].set_ylabel("Time (s)")

    # num_files_found = 0

    for i, file in enumerate(["a_cpu", "a_gpu", "ab_gpu", "abc_gpu", "abcd_gpu"]):
        file = f"timing_results_sin-{file}.csv"
        if not file.endswith(".csv"):
            continue
        model = file.split("_")[2]
        mode = file.split("_")[-1].split(".")[0]
        num_calib_params = len(model.split("-")[-1])

        if mode == "gpu":
            linestyle = "--"
        else:
            linestyle = "-"

        marker = ["o", "o", "D", "^", "*"][i]

        file_path = f"timings/{file}"
        fig, axes = plot_time_vs_sim_output_single(
            fig,
            axes,
            path_to_csv=file_path,
            label=f"{num_calib_params} params using {mode}",
            marker=marker,
            linestyle=linestyle,
        )

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.tight_layout()

    return fig, axes


def run():
    fig, axes = plot_time_vs_sim_output()
    plt.savefig("figures/time-vs-sim-output.png", dpi=300)
    plt.close(fig)
