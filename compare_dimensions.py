import importlib.util
import os

from datahandler import load


def load_config(model_name):
    # Construct absolute path to the config file
    config_path = os.path.abspath(f"models/{model_name}/config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    # Execute the module
    spec.loader.exec_module(module)
    return module.experiment_config


def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"


def main():
    experiments = ["T21_land", "T21_land_2D", "streamfunction"]
    data_root = os.path.abspath("data")

    print("Comparing Matrix Dimensions and RAM Usage\n")

    for exp in experiments:
        print(f"--- Experiment: {exp} ---")
        try:
            config = load_config(exp)
            # datahandler.load expects the DataConfig object to describe paths relative to data/{experiment_name}
            # The 'data_root' argument is just 'data' folder.

            koh_dataset, _, _ = load(config, data_root)

            # Simulation Data
            # Xc: Simulation Inputs [Coords, ValidParams...]
            # yc: Simulation Outputs (centered)
            Xc = koh_dataset.Xc
            yc = koh_dataset.y

            # Observation Data
            # Xf: Observation Inputs [Coords]
            # zf: Observation Outputs (centered) - datahandler uses 'z' for yf in filters?
            # datahandler.py load returns: koh_dataset = kgx.KOHDataset(field_dataset, comp_dataset)
            # kohgpjax KOHDataset: field_dataset is observed, comp_dataset is simulation
            # field_dataset.X -> Xf, field_dataset.y -> z

            Xf = koh_dataset.Xf
            zf = koh_dataset.z  # z is the observed output yf

            # Combine Data using kohdataset.X(theta)
            import jax.numpy as jnp

            n_calib = config.n_calib_params
            theta = jnp.zeros((1, n_calib))

            X_full = koh_dataset.X(theta)

            print("Combined Input Matrix X = kohdataset.X(0):")
            print(f"  X shape: {X_full.shape}, RAM: {format_bytes(X_full.nbytes)}")

            # X^T X
            # shape (D, D)
            feat_dim = X_full.shape[1]
            xtx_shape = (feat_dim, feat_dim)
            xtx_bytes = feat_dim * feat_dim * X_full.dtype.itemsize
            print(f"  X^T X shape: {xtx_shape}, RAM: {format_bytes(xtx_bytes)}")

            # X X^T (Gram Matrix / Kernel Matrix dimension)
            # shape (N, N)
            num_samples = X_full.shape[0]
            xxt_shape = (num_samples, num_samples)
            xxt_bytes = num_samples * num_samples * X_full.dtype.itemsize
            print(
                f"  X X^T (Gram Matrix) shape: {xxt_shape}, RAM: {format_bytes(xxt_bytes)}"
            )
            # RAM required for inversion is typically the size of the matrix itself (if in-place) or more.
            print(f"  RAM required for Inversion (NxN): {format_bytes(xxt_bytes)}")

            print("Simulation Data (Xc, yc):")
            print(f"  Xc shape: {Xc.shape}, RAM: {format_bytes(Xc.nbytes)}")
            print(f"  yc shape: {yc.shape}, RAM: {format_bytes(yc.nbytes)}")

            print("Observation Data (Xf, yf):")
            print(f"  Xf shape: {Xf.shape}, RAM: {format_bytes(Xf.nbytes)}")
            print(f"  yf shape: {zf.shape}, RAM: {format_bytes(zf.nbytes)}")

        except Exception as e:
            print(f"Failed to analyze {exp}: {e}")
            # Comment out traceback to keep output clean unless debugging
            # import traceback
            # traceback.print_exc()
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
