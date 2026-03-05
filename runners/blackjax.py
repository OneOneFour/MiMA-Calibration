from typing import Any, Dict

import blackjax
import gpjax as gpx
import jax
import jax.numpy as jnp
import kohgpjax as kgx
from jax import config

from runners.base import BaseRunner

config.update("jax_enable_x64", True)


print("GPJax version:", gpx.__version__)
print("KOHGPJax version:", kgx.__version__)
print("JAX Device:", jax.devices())


class BlackjaxRunner(BaseRunner):
    def __init__(
        self,
        model_dir: str,
        n_warm_up_iter: int,
        n_main_iter: int,
        seed: int,
        n_chain: int,
        max_num_doublings: int,
    ):
        super().__init__(model_dir, n_warm_up_iter, n_main_iter, seed, n_chain)
        self.max_num_doublings = max_num_doublings

    def _run(self) -> Dict[str, Any]:
        # Negative log posterior density function (expects flat unconstrained vector)
        neg_log_post = self.model.get_KOH_neg_log_pos_dens_func()

        # --- Define the log probability function ---
        @jax.jit
        def log_prob(params_vec: jnp.ndarray) -> jax.Array:
            """Return log-posterior given flat unconstrained params vector."""
            return -neg_log_post(params_vec)

        # --- Initialize the MCMC sampler ---
        key = jax.random.PRNGKey(self.seed)
        key, subkey = jax.random.split(key)
        prior_leaves, _ = jax.tree.flatten(self.prior_dict)
        prior_means = jax.tree.map(
            lambda x: x.inverse(x.distribution.mean), prior_leaves
        )
        initial_position = jnp.array(prior_means)

        # --- Window adaptation for HMC ---
        warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)

        # Split keys
        key, warmup_key, sample_key = jax.random.split(key, 3)

        # Initial positions for multiple chains: (n_chain, n_params)
        initial_positions = jnp.tile(initial_position, (self.n_chain, 1))

        # Warmup keys: (n_chain,)
        warmup_keys = jax.random.split(warmup_key, self.n_chain)

        print(
            f"Running warmup for {self.n_chain} chains with {self.n_warm_up_iter} iterations..."
        )

        # Run warmup vectorized over chains, explicitly JIT-ed
        @jax.jit
        def run_warmup(keys, initial_positions):
            return jax.vmap(warmup.run, in_axes=(0, 0, None))(
                keys, initial_positions, self.n_warm_up_iter
            )

        (state, parameters), _ = run_warmup(warmup_keys, initial_positions)

        print(
            f"Adapted parameters (first chain): {jax.tree.map(lambda x: x[0], parameters)}",
            "\n",
        )

        # --- Sampling ---
        print(
            f"Running sampling for {self.n_chain} chains with {self.n_main_iter} iterations..."
        )

        @jax.jit
        def run_inference(rng_key, initial_state, parameters):
            def step_fn(key, state, params):
                kernel = blackjax.nuts(
                    log_prob, max_num_doublings=self.max_num_doublings, **params
                ).step
                return kernel(key, state)

            # vmap over chains: key(0), state(0), params(0)
            step_fn_vmap = jax.vmap(step_fn)

            def one_step(state, key):
                new_state, info = step_fn_vmap(key, state, parameters)
                return new_state, new_state

            # Generate keys: (num_samples, n_chain)
            keys = jax.random.split(rng_key, self.n_main_iter)
            keys = jax.vmap(lambda k: jax.random.split(k, self.n_chain))(keys)

            _, states = jax.lax.scan(one_step, initial_state, keys)
            return states

        # Run sampling
        states = run_inference(sample_key, state, parameters)

        # states.position is (n_main_iter, n_chain, n_params)
        # We want (n_chain, n_main_iter, n_params)
        positions = jnp.swapaxes(states.position, 0, 1)

        # Build raw traces dict: each param -> [n_chain, n_main_iter]
        tracer_index_dict = {
            p.name: i for i, p in enumerate(self.model_parameters.priors_flat)
        }
        raw_traces = {
            name: jnp.array(positions[:, :, idx])
            for name, idx in tracer_index_dict.items()
        }

        # Prepare metadata for saving
        algorithm_parameters = {
            "max_num_doublings": self.max_num_doublings,
        }
        # Add adapted parameters (step_size, inverse_mass_matrix, etc.)
        for k, v in parameters.items():
            if hasattr(v, "tolist"):
                algorithm_parameters[k] = v.tolist()
            else:
                algorithm_parameters[k] = (
                    float(v) if isinstance(v, (float, int, jnp.ndarray)) else str(v)
                )

        return {
            "traces": raw_traces,
            "backend_name": "blackjax",
            "algorithm_name": "nuts",
            "algorithm_parameters": algorithm_parameters,
        }


def run(
    model_dir: str,
    n_warm_up_iter: int,
    n_main_iter: int,
    seed: int,
    n_chain: int,
    max_num_doublings: int,
):
    runner = BlackjaxRunner(
        model_dir=model_dir,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        seed=seed,
        n_chain=n_chain,
        max_num_doublings=max_num_doublings,
    )
    runner.run()
