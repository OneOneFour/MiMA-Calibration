import importlib.metadata
import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import arviz
import numpy as np
from kohgpjax.parameters import ModelParameters

from datahandler import load, save_chains_to_netcdf, transform_chains
from utils import load_config_from_model_dir, load_model_from_model_dir


class BaseRunner(ABC):
    def __init__(
        self,
        model_dir: str,
        n_warm_up_iter: int,
        n_main_iter: int,
        seed: int,
        n_chain: int,
    ):
        self.model_dir = model_dir
        self.n_warm_up_iter = n_warm_up_iter
        self.n_main_iter = n_main_iter
        self.seed = seed
        self.n_chain = n_chain

        self.config_module = load_config_from_model_dir(model_dir)
        self.experiment_config = self.config_module.experiment_config
        self.file_name = self.experiment_config.name
        self.Model, self.get_ModelParameterPriorDict = load_model_from_model_dir(
            model_dir
        )

        # Determine data root
        # Assuming current script is in runners/base.py, go up one level to root
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_root = os.path.join(self.script_dir, "data")

        print(f"Loading dataset from {self.file_name}...")
        self.kohdataset, self.tminmax, self.yc_mean = load(
            experiment_config=self.experiment_config,
            data_root=self.data_root,
        )
        self.n_sim = self.kohdataset.num_sim_obs

        self.prior_dict = self.get_ModelParameterPriorDict(
            self.config_module, self.tminmax
        )
        self.model_parameters = ModelParameters(prior_dict=self.prior_dict)

        self.model = self.Model(
            model_parameters=self.model_parameters,
            kohdataset=self.kohdataset,
        )

    def get_library_versions(self) -> Dict[str, str]:
        versions = {}
        for lib in ["mici", "blackjax", "jax", "numpy", "gpjax", "kohgpjax"]:
            try:
                versions[lib] = importlib.metadata.version(lib)
            except Exception:
                versions[lib] = None
        return versions

    def setup_output_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_W{self.n_warm_up_iter}_N{self.n_main_iter}"
        output_dir = os.path.join(
            self.script_dir, "experiments", self.file_name, run_id
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _filter_json_serializable(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to ensure dict is JSON serializable."""
        out = {}
        for k, v in d.items():
            if callable(v):
                continue
            try:
                import jax.numpy as jnp

                if isinstance(v, (np.ndarray, jnp.ndarray)):
                    v = v.tolist()
                elif isinstance(v, (np.generic, jnp.generic)):
                    v = float(v)
            except Exception:
                pass

            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                out[k] = v
        return out

    def save_results(
        self,
        traces: Dict[str, Any],
        backend_name: str,
        algorithm_name: str,
        algorithm_parameters: Dict[str, Any],
        extra_mcmc_params: Dict[str, Any] = None,
    ):
        # Transform chains
        traces_transformed = transform_chains(
            traces, self.model_parameters, self.prior_dict, self.tminmax
        )

        print(arviz.summary(traces_transformed))

        output_dir = self.setup_output_dir()
        print(f"Saving results to {output_dir}")

        # Prepare settings
        mcmc_params = {
            "n_chain": self.n_chain,
            "n_warm_up_iter": self.n_warm_up_iter,
            "n_main_iter": self.n_main_iter,
            "seed": self.seed,
            "model_dir": self.model_dir,
        }
        if extra_mcmc_params:
            mcmc_params.update(extra_mcmc_params)

        settings = {
            "backend": backend_name,
            "algorithm": algorithm_name,
            "algorithm_parameters": algorithm_parameters,
            "mcmc_parameters": mcmc_params,
            "library_versions": self.get_library_versions(),
            "cli_command": " ".join(sys.argv),
        }

        with open(os.path.join(output_dir, "mcmc_settings.json"), "w") as f:
            json.dump(settings, f, indent=2)

        save_chains_to_netcdf(
            raw_traces=traces,
            transformed_traces=traces_transformed,
            file_name=self.file_name,
            n_warm_up_iter=self.n_warm_up_iter,
            n_main_iter=self.n_main_iter,
            n_sim=self.n_sim,
            ycmean=self.yc_mean,
            inference_library_name=backend_name,
            output_dir=output_dir,
        )

    @abstractmethod
    def _run(self) -> Dict[str, Any]:
        """
        Run the MCMC sampler and return the raw traces and metadata.

        Returns:
            A dictionary containing:
            - traces: Dict of raw MCMC traces (param_name -> array)
            - backend_name: str
            - algorithm_name: str
            - algorithm_parameters: Dict[str, Any]
            - extra_mcmc_params: Optional[Dict[str, Any]]
        """
        pass

    def run(self):
        """
        Execute the MCMC run pipeline: _run() -> save_results().
        """
        result = self._run()

        self.save_results(
            traces=result["traces"],
            backend_name=result["backend_name"],
            algorithm_name=result["algorithm_name"],
            algorithm_parameters=result["algorithm_parameters"],
            extra_mcmc_params=result.get("extra_mcmc_params"),
        )
