import xarray as xr
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# Correct path to the file based on user info
filepath = os.path.join(script_dir, "data/T21_land_2D/moist_variables_all.nc")

print(f"Loading {filepath}...")
try:
    ds = xr.open_dataset(filepath, decode_times=False)
except FileNotFoundError:
    print("File not found.")
    exit(1)

print("\n--- Coordinates ---")
print(ds.coords)

print("\n--- Coordinate Ranges ---")
for coord in ds.coords:
    vals = ds[coord].values
    if np.issubdtype(vals.dtype, np.number):
        print(f"{coord}: min={vals.min()}, max={vals.max()}, len={len(vals)}")
    else:
        print(f"{coord}: len={len(vals)}")

# Attempt to identify Rh and tau
# Config suggested theta_0 and theta_1. Datahandler renames rh to t if only 1 param, but there are 2.
# Let's check if 'rh' and 'tau' are in coords.

lat_name = "lat"
rh_name = None
tau_name = None

for c in ds.coords:
    if c.lower() == "rh":
        rh_name = c
    if c.lower() == "tau":
        tau_name = c

print(f"\nIdentified Rh: {rh_name}")
print(f"Identified tau: {tau_name}")

if rh_name and tau_name:
    # Check filtering
    # User condition: Cut out where Rh < 0.3 OR tau < 2000.
    # Keep: Rh >= 0.3 AND tau >= 2000.
    # Also Lat in [-75, 75].

    rh_vals = ds[rh_name].values
    tau_vals = ds[tau_name].values
    lat_vals = ds[lat_name].values

    # Assuming grid structure (stacking dims)
    # Total points = product of lengths if they are orthogonal dimensions
    # Or strict product?

    # Check if dimensions are orthogonal
    print("\n--- Dimensions ---")
    print(ds.dims)

    # Calculate total planned points vs detailed points
    # If they are dimensions:
    n_rh = len(rh_vals)
    n_tau = len(tau_vals)
    n_lat = len(lat_vals)

    total_original = n_rh * n_tau * n_lat
    print(f"\nTotal potential grid points: {total_original}")

    # Filter
    valid_rh = rh_vals[rh_vals >= 0.3]
    valid_tau = tau_vals[tau_vals >= 200]
    valid_lat = lat_vals[(lat_vals >= -75) & (lat_vals <= 75)]

    n_rh_new = len(valid_rh)
    n_tau_new = len(valid_tau)
    n_lat_new = len(valid_lat)

    total_new = n_rh_new * n_tau_new * n_lat_new
    print(f"Filtered Rh (>= 0.3): count={n_rh_new}")
    print(f"Filtered tau (>= 200): count={n_tau_new}")
    print(f"Filtered Lat ([-75, 75]): count={n_lat_new}")

    print(f"\nTotal points after filter: {total_new}")
    print(f"Reduction ratio: {total_new / total_original:.2%}")

    # Memory Size Estimation
    # Original: 28800
    # User had 28800. Let's see if 28800 matches n_rh * n_tau * n_lat.

else:
    print("Could not find 'rh' or 'tau' parameters to estimate filtering.")
