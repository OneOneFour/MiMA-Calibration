import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main(indices: list[int]):
    assert len(indices) == 3, "Must provide list of 3 indices"
    assert min(indices) >= 0, "Indices must be non-negative"
    assert max(indices) < 30, "Indices must be less than 30"

    # Load Simulation Data
    ds_sim = xr.open_dataset(
        "data/streamfunction/hadley_cell_all.nc", decode_times=False
    )
    # Load Reference Data
    ds_ref = xr.open_dataset("data/streamfunction/era5_ref_streamfunction.nc")

    # Extract coordinate values (Sim)
    rh_vals = ds_sim["rh"].values
    tau_vals = ds_sim["tau"].values
    sim_streamfunction = ds_sim["streamfunction_mean"]

    # Extract Reference Data
    ref_streamfunction = ds_ref["streamfunction_avg"]

    # Determine global min/max for consistent plotting levels
    # Use .min().item() to get a python scalar, avoiding xarray broadcasting issues in linspace
    vmin = min(sim_streamfunction.min().item(), ref_streamfunction.min().item())
    vmax = max(sim_streamfunction.max().item(), ref_streamfunction.max().item())

    levels = 20
    # Create fixed levels for consistent comparison
    level_values = np.linspace(vmin, vmax, levels)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes_flat = axes.flatten()

    # --- Plot 1: Reference (Top Left) ---
    ax_ref = axes_flat[0]

    # Check coords for Ref
    # era5 usually: level, latitude
    x_ref = (
        ref_streamfunction.coords["latitude"]
        if "latitude" in ref_streamfunction.coords
        else ref_streamfunction.coords["lat"]
    )
    y_ref = (
        ref_streamfunction.coords["level"]
        if "level" in ref_streamfunction.coords
        else ref_streamfunction.coords["pfull"]
    )

    ax_ref.contour(
        x_ref,
        y_ref,
        ref_streamfunction,
        levels=level_values,
        cmap="RdBu_r",
        linewidths=1.5,
    )

    ax_ref.invert_yaxis()
    ax_ref.set_ylabel("Pressure (hPa)")
    ax_ref.set_xlabel("Latitude")
    ax_ref.set_title("Reference (ERA5)")

    # --- Plot 2, 3, 4: Simulations ---
    # We use the 3 indices provided for the remaining 3 subplots
    sim_indices_to_plot = indices

    for i, idx_val in enumerate(sim_indices_to_plot):
        ax = axes_flat[i + 1]  # Slots 1, 2, 3

        # Select data by N dimension
        data_slice = sim_streamfunction.isel(N=idx_val)

        # Sim Coords: lat, pfull
        ax.contour(
            data_slice.coords["lat"],
            data_slice.coords["pfull"],
            data_slice,
            levels=level_values,  # Share levels
            cmap="RdBu_r",
            linewidths=1.5,
        )

        # Invert Y axis for pressure (pfull)
        ax.invert_yaxis()
        ax.set_ylabel("Pressure (hPa)")
        ax.set_xlabel("Latitude")

        # Set Title
        rh_v = rh_vals[idx_val]
        tau_v = tau_vals[idx_val]
        ax.set_title(f"Run {idx_val + 1}: Rh={rh_v:.3f}, Tau={tau_v:.0f}")

    # Add shared colorbar
    # User requested a "blocky" (discrete) colorbar to match contour lines.
    # We use BoundaryNorm with the defined levels.
    import matplotlib.colors as mcolors

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.BoundaryNorm(level_values, cmap.N)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(
        sm, cax=cbar_ax, label="Streamfunction (kg/s)", ticks=level_values[::2]
    )  # Optional: thin ticks

    # fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for colorbar
    # Using subplots_adjust is often safer with add_axes

    fig.suptitle("Overturning Streamfunction")
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("streamfunction.png")
    print("Saved streamfunction.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", type=int, nargs=3)
    args = parser.parse_args()
    main(args.indices)
