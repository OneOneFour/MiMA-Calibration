from typing import Any, Dict

import gpjax as gpx
import jax
import kohgpjax as kgx
import mici
import numpy as np
from jax import config

from freethreading import process_workers
from runners.base import BaseRunner

config.update("jax_enable_x64", True)


class MiciRunner(BaseRunner):
    def __init__(
        self,
        model_dir: str,
        n_warm_up_iter: int,
        n_main_iter: int,
        seed: int,
        n_chain: int,
        n_processes: int,
        max_tree_depth: int,
    ):
        super().__init__(model_dir, n_warm_up_iter, n_main_iter, seed, n_chain)
        self.n_processes = n_processes
        self.max_tree_depth = max_tree_depth

        print("GPJax version:", gpx.__version__)
        print("KOHGPJax version:", kgx.__version__)
        print("JAX Device:", jax.devices())

    def _run(self) -> Dict[str, Any]:
        # Mici setup
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=self.model.get_KOH_neg_log_pos_dens_func(),
            backend="jax",
        )
        integrator = mici.integrators.LeapfrogIntegrator(system)

        prior_leaves, _ = jax.tree.flatten(self.prior_dict)
        prior_means = jax.tree.map(
            lambda x: x.inverse(x.distribution.mean), prior_leaves
        )
        init_states = np.array(prior_means)

        print(f"Initial states: {init_states}")
        f = self.model.get_KOH_neg_log_pos_dens_func()
        print(f(init_states))

        tracer_index_dict = {}
        for i, prior in enumerate(self.model_parameters.priors_flat):
            tracer_index_dict[prior.name] = i

        rng = np.random.default_rng(self.seed)
        sampler = mici.samplers.DynamicMultinomialHMC(
            system, integrator, rng, max_tree_depth=self.max_tree_depth
        )
        adapters = [
            mici.adapters.DualAveragingStepSizeAdapter(0.8),
            mici.adapters.OnlineCovarianceMetricAdapter(),
        ]

        def trace_func(state):
            trace = {key: state.pos[index] for key, index in tracer_index_dict.items()}
            trace["hamiltonian"] = system.h(state)
            return trace

        # Run MCMC
        final_states, traces, stats = sampler.sample_chains(
            self.n_warm_up_iter,
            self.n_main_iter,
            [init_states] * self.n_chain,
            adapters=adapters,
            **process_workers(self.n_processes),
            trace_funcs=[trace_func],
            monitor_stats=("n_step", "accept_stat", "step_size", "diverging"),
        )

        # Prepare metadata
        algorithm_parameters = {
            "sampler_class": sampler.__class__.__name__,
            "max_tree_depth": self.max_tree_depth,
            "adapters": [a.__class__.__name__ for a in adapters],
            "adapter_settings": [
                self._filter_json_serializable(getattr(a, "__dict__", {}))
                for a in adapters
            ],
        }

        extra_mcmc_params = {
            "n_processes": self.n_processes,
        }

        return {
            "traces": traces,
            "backend_name": "mici",
            "algorithm_name": "DynamicMultinomialHMC",
            "algorithm_parameters": algorithm_parameters,
            "extra_mcmc_params": extra_mcmc_params,
        }


def run(
    model_dir: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    seed: int,
    n_chain: int,
    n_processes: int,
    max_tree_depth: int,
):
    runner = MiciRunner(
        model_dir=model_dir,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        seed=seed,
        n_chain=n_chain,
        n_processes=n_processes,
        max_tree_depth=max_tree_depth,
    )
    runner.run()
