import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from datahandler import load
from plotting import plot_sim_sample
from utils import load_config_from_model_dir


def run(
    config_path: str = None,
    model_dir: str = None,
    dpi: int = 300,
    num_samples: int = None,
    alpha: float = 0.3,
    output_dir: str = None,
    sample_step: int = 1,
):
    if model_dir is not None:
        config_module = load_config_from_model_dir(model_dir)
    else:
        raise ValueError("model_dir must be provided.")
    experiment_config = config_module.experiment_config
    experiment_name = experiment_config.name

    kohdataset, tminmax, ycmean = load(
        experiment_config=experiment_config, data_root="data"
    )

    # 6 Plotting
    # 6.0 Create the directory for figures if it doesn't exist
    if output_dir:
        save_dir = output_dir
    else:
        print("Creating figures directory...")
        save_dir = os.path.join("figures", experiment_name)

    os.makedirs(save_dir, exist_ok=True)

    print("Plotting...")

    # 6.1 Plot sample of the data
    fig, ax = plot_sim_sample(
        kohdataset=kohdataset,
        tminmax=tminmax,
        ycmean=ycmean,
        num_samples=num_samples,
        alpha=alpha,
        sample_step=sample_step,
    )
    plt.savefig(os.path.join(save_dir, "obs-and-sim-sample.png"), dpi=dpi)
    plt.close()
