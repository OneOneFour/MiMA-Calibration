import argparse

from analysis import diagnostics, posterior, ppc, predictions, sim_sample, timings


def main():
    parser = argparse.ArgumentParser(description="MiMA Calibration Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- RUN Command ---
    run_parser = subparsers.add_parser("run", help="Run MCMC Sampling")
    # Add base arguments (model_dir, W, N, seed, n_chain)
    run_parser.add_argument("model_dir", type=str, help="Path to model directory")
    run_parser.add_argument(
        "-W", "--warmup", dest="W", type=int, default=100, help="Warmup iterations"
    )
    run_parser.add_argument(
        "-N", "--main", dest="N", type=int, default=100, help="Main iterations"
    )
    run_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    run_parser.add_argument("--n_chain", type=int, default=1, help="Number of chains")

    # Specific arguments
    run_parser.add_argument(
        "--backend", choices=["mici", "blackjax"], default="mici", help="MCMC backend"
    )
    run_parser.add_argument(
        "--n_processes", type=int, default=1, help="Number of processes (mici only)"
    )
    run_parser.add_argument(
        "--max_tree_depth", type=int, default=5, help="Max tree depth (mici only)"
    )
    run_parser.add_argument(
        "--max_num_doublings",
        type=int,
        default=5,
        help="Max num doublings (blackjax only)",
    )

    # --- PLOT Command ---
    plot_parser = subparsers.add_parser("plot", help="Generate Plots")
    plot_subparsers = plot_parser.add_subparsers(dest="plot_type", required=True)

    # plot trace -> diagnostics
    trace_parser = plot_subparsers.add_parser(
        "trace", help="Plot Trace and Diagnostics"
    )
    trace_parser.add_argument("--model_dir", type=str, required=True)
    trace_parser.add_argument("--output_dir", type=str)

    # plot ppc -> ppc
    ppc_parser = plot_subparsers.add_parser(
        "ppc", help="Plot Posterior Predictive Check"
    )
    ppc_parser.add_argument("--model_dir", type=str, required=True)
    ppc_parser.add_argument(
        "--file_path", type=str, required=True, help="Path to NetCDF"
    )
    ppc_parser.add_argument("--output_dir", type=str)

    # plot predictions
    pred_parser = plot_subparsers.add_parser(
        "predictions", help="Plot Model Predictions"
    )
    pred_parser.add_argument("--model_dir", type=str, required=True)
    pred_parser.add_argument("--output_dir", type=str, required=True)
    pred_parser.add_argument("--chain_path", type=str, required=True)
    pred_parser.add_argument("--plot_observations", type=bool, default=False)

    # plot sim_sample
    sim_parser = plot_subparsers.add_parser(
        "sim_sample", help="Plot Simulation Samples"
    )
    sim_parser.add_argument("--model_dir", type=str, required=True)
    sim_parser.add_argument("--output_dir", type=str)
    sim_parser.add_argument("--num_samples", type=int)
    sim_parser.add_argument("--sample_step", type=int, default=1)
    sim_parser.add_argument("--alpha", type=float, default=0.3)

    # plot timings
    # time_parser = plot_subparsers.add_parser("timings", help="Plot Timings")

    # --- ANALYZE Command ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze Posterior")
    analyze_parser.add_argument(
        "experiment_dir", type=str, help="Path to experiment directory"
    )

    args = parser.parse_args()

    if args.command == "run":
        if args.backend == "mici":
            from runners import mici

            mici.run(
                model_dir=args.model_dir,
                n_warm_up_iter=args.W,
                n_main_iter=args.N,
                seed=args.seed,
                n_chain=args.n_chain,
                n_processes=args.n_processes,
                max_tree_depth=args.max_tree_depth,
            )
        elif args.backend == "blackjax":
            from runners import blackjax

            blackjax.run(
                model_dir=args.model_dir,
                n_warm_up_iter=args.W,
                n_main_iter=args.N,
                seed=args.seed,
                n_chain=args.n_chain,
                max_num_doublings=args.max_num_doublings,
            )

    elif args.command == "plot":
        if args.plot_type == "trace":
            diagnostics.run(model_dir=args.model_dir, output_dir=args.output_dir)
        elif args.plot_type == "ppc":
            ppc.run(
                model_dir=args.model_dir,
                file_path=args.file_path,
                output_dir=args.output_dir,
            )
        elif args.plot_type == "predictions":
            predictions.run(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                chain_path=args.chain_path,
                plot_observations=args.plot_observations,
            )
        elif args.plot_type == "sim_sample":
            sim_sample.run(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                sample_step=args.sample_step,
                alpha=args.alpha,
            )
        elif args.plot_type == "timings":
            timings.run()

    elif args.command == "analyze":
        posterior.run(experiment_dir=args.experiment_dir)


if __name__ == "__main__":
    main()
