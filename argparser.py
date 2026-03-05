import argparse


def get_base_parser():
    """
    Creates and returns a base argparse.ArgumentParser with common MCMC arguments.
    """
    parser = argparse.ArgumentParser(description="Run the MCMC posterior sampling.")
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the model directory (e.g., models/T21)",
    )
    parser.add_argument("W", type=int, help="Number of warm-up iterations.")
    parser.add_argument("N", type=int, help="Number of main iterations.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--n_chain", type=int, default=2, help="Number of MCMC chains.")
    return parser
