import os
import sys

# Add parent directory to path to allow importing config_schema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_schema import DataConfig, ExperimentConfig, Parameter

experiment_config = ExperimentConfig(
    name="streamfunction",
    parameters=[
        Parameter(name="theta_0", range=(0.0, 1.0)),
        Parameter(name="theta_1", range=(0.0, 14400.0)),
    ],
    data=DataConfig(
        experiment_name="streamfunction",
        obs_data_path="era5_ref_streamfunction.nc",
        sim_data_path="hadley_cell_all.nc",
        sim_variable_name="streamfunction_mean",
        obs_variable_name="overturning_avg",
        sim_coord_names=["pfull", "lat"],
        obs_coord_names=["level", "latitude"],
        calib_coord_names=["rh", "tau"],
    ),
    n_calib_params=2,
    n_control_params=2,
    output_dims=1,
    # n_simulation_runs=30,
    # n_simulation_points=32,
    # n_observation_points=72,
    filters={
        "lat": (-55, 55),
        "theta_0": (0.35, None),
        "theta_1": (1300, None),
    },
    thinning={
        "pfull": 2.5,
    },
)

FILE_NAME = experiment_config.name

# --- General Settings ---
# Number of calibration parameters to use in the simulation.
N_CALIB_PARAMS = experiment_config.n_calib_params
N_CONTROL_PARAMS = experiment_config.n_control_params
OUTPUT_DIMS = experiment_config.output_dims


# --- Parameter Definitions ---
# Using a list of dictionaries to define parameters generically.
PARAMETERS = [
    {"name": p.name, "true_value": p.true_value, "range": list(p.range)}
    for p in experiment_config.parameters
]

# # --- Control Parameter Definitions ---
# CONTROL_PARAMETERS = [
#     {"name": "x0", "range": [0.02, 3.98]},
# ]

# --- Data Generation Settings ---
N_SIMULATION_RUNS = experiment_config.n_simulation_runs
N_SIMULATION_POINTS = experiment_config.n_simulation_points
N_OBSERVATION_POINTS = experiment_config.n_observation_points

# --- Observation Noise ---
# Standard deviation of the observation noise.
# OBS_NOISE_STD = [0.5]
