from typing import Dict, Tuple

import gpjax as gpx
import jax.numpy as jnp
import numpyro.distributions as dist
from kohgpjax.kohmodel import KOHModel
from kohgpjax.parameters import ModelParameterPriorDict, ParameterPrior


class Model(KOHModel):
    def k_eta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        params = params_constrained["eta"]
        kernels = [
            gpx.kernels.RBF(
                active_dims=[0],
                lengthscale=jnp.array(params["lengthscales"]["x_0"]),
                variance=jnp.array(params["variances"]["variance"]),
            ),
            gpx.kernels.RBF(
                active_dims=[1],
                lengthscale=jnp.array(params["lengthscales"]["theta_0"]),
            ),
        ]
        return gpx.kernels.ProductKernel(kernels=kernels)

    def k_delta(self, params_constrained) -> gpx.kernels.AbstractKernel:
        params = params_constrained["delta"]
        return gpx.kernels.RBF(
            active_dims=[0],
            lengthscale=jnp.array(params["lengthscales"]["x_0"]),
            variance=jnp.array(params["variances"]["variance"]),
        )

    def k_epsilon(self, params_constrained) -> gpx.kernels.AbstractKernel:
        params = params_constrained["epsilon"]
        return gpx.kernels.White(
            active_dims=[0],
            variance=jnp.array(params["variances"]["variance"]),
        )


def get_ModelParameterPriorDict(
    config,
    tminmax: Dict[str, Tuple[float, float]] = None,
) -> ModelParameterPriorDict:
    print(f"Creating ModelParameterPriorDict for configs/{config.FILE_NAME} model...")

    # param = config.PARAMETERS[0]
    # assert param["name"] == "theta_0", (
    #     f"Expected first parameter to be 'theta_0', got '{param['name']}'"
    # )
    # prior_range = param["range"]

    # # account for the scaling onto [0, 1]
    # tmm = tminmax["theta_0"]
    # A = (prior_range[0] - tmm[0]) / (tmm[1] - tmm[0])
    # B = (prior_range[1] - tmm[0]) / (tmm[1] - tmm[0])

    prior_dict: ModelParameterPriorDict = {
        "thetas": {
            "theta_0": ParameterPrior(
                dist.Beta(
                    concentration1=4.0, concentration0=2.0
                ),  # often around 0.7, probably >0.5, beta dist good, p(0)=0, p(1)!=0
                name="theta_0",
            ),
        },
        "eta": {
            "variances": {
                "variance": ParameterPrior(
                    dist.Gamma(concentration=6.0, rate=1.2),  # E(X)=5.0, SD(X)=2.04
                    name="eta_variance",
                ),
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    # dist.Gamma(concentration=5.0, rate=1 / 0.25),
                    dist.Gamma(concentration=5.0, rate=1 / 3.0),
                    name="eta_lengthscale_x_0",
                ),
                "theta_0": ParameterPrior(
                    # dist.Gamma(concentration=2.0, rate=1 / 3.0),
                    dist.Gamma(concentration=2.0, rate=2.0),
                    name="eta_lengthscale_theta_0",
                ),
            },
        },
        "delta": {
            "variances": {
                "variance": ParameterPrior(
                    # dist.Gamma(concentration=1.5, rate=1 / 4.0),
                    dist.Gamma(concentration=3.0, rate=3.0),  # E(X)=1.0, SD(X)=0.577
                    name="delta_variance",
                ),
            },
            "lengthscales": {
                "x_0": ParameterPrior(
                    # dist.Gamma(
                    #     concentration=7.0, rate=1 / 0.1
                    # ),
                    dist.Gamma(
                        concentration=5.0, rate=1 / 5
                    ),  # encourage long value => Very smooth. Don't infer too much where we are naturally uncertain.
                    name="delta_lengthscale_x_0",
                ),
            },
        },
        "epsilon": {  # This is required despite not appearing in the model
            "variances": {
                "variance": ParameterPrior(
                    # dist.Gamma(concentration=3, rate=1 / 0.3),  # E(X)=0.9, SD(X)=0.5196
                    dist.Gamma(concentration=1.5, rate=10),  # E(X)=0.15, SD(X)=0.1225
                    name="epsilon_variance",
                ),
            },
        },
    }

    return prior_dict
