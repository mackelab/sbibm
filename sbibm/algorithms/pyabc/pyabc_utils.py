from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np

import pyabc
import torch


class PyAbcSimulator:
    """Wrapper from sbibm task to pyABC. 

    pyABC defines its own priors and they are sampled without batch dimension. This
    wrapper defines a call method that takes a single parameter set from a pyABC prior
    and uses the sbibm task simulator to generate the corresponding data and to return
    it in pyABC format. 
    """
    def __init__(self, task):
        self.simulator = task.get_simulator()
        self.dim_parameters = task.dim_parameters
        self.name = task.name

    def __call__(self, pyabc_parameter) -> Dict:
        parameters = torch.tensor([[pyabc_parameter[f"param{dim+1}"] for dim in range(self.dim_parameters)]], dtype=torch.float32)
        data = self.simulator(parameters).numpy().squeeze()
        return dict(data=data)

    @property
    def __name__(self) -> str:
        return self.name



def wrap_prior(task):
    """Returns a pyABC.Distribution prior given a prior defined on a sbibm task.

    Note: works only for a specific set of priors: Uniform, LogNormal, Normal.
    """
    log = logging.getLogger(__name__)
    log.warn("Will discard any correlations in prior")

    bounds = {}

    prior_cls = str(task.prior_dist)
    if prior_cls == "Independent()":
        prior_cls = str(task.prior_dist.base_dist)

    prior_params = {}
    if "MultivariateNormal" in prior_cls:
        prior_params["m"] = task.prior_params["loc"].numpy()
        if "precision_matrix" in prior_cls:
            prior_params["C"] = np.linalg.inv(
                task.prior_params["precision_matrix"].numpy()
            )
        if "covariance_matrix" in prior_cls:
            prior_params["C"] = task.prior_params["covariance_matrix"].numpy()

        prior_dict = {}
        for dim in range(task.dim_parameters):
            loc = prior_params["m"][dim]
            scale = np.sqrt(prior_params["C"][dim, dim])

            prior_dict[f"param{dim+1}"] = pyabc.RV("norm", loc, scale)
        prior = pyabc.Distribution(**prior_dict)
    
    elif "LogNormal" in prior_cls:
        # Note the difference in parameterisation between pytorch LogNormal and scipy
        # lognorm: 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
        prior_params["s"] = task.prior_params["scale"].numpy()
        prior_params["scale"] = np.exp(task.prior_params["loc"].numpy())

        prior_dict = {}
        for dim in range(task.dim_parameters):
            prior_dict[f"param{dim+1}"] = pyabc.RV("lognorm", 
                                                    s=prior_params["s"][dim],
                                                    scale=prior_params["scale"][dim])

        prior = pyabc.Distribution(**prior_dict)

    elif "Uniform" in prior_cls:
        prior_params["low"] = task.prior_params["low"].numpy()
        prior_params["high"] = task.prior_params["high"].numpy()

        prior_dict = {}
        for dim in range(task.dim_parameters):
            loc = prior_params["low"][dim]
            scale = prior_params["high"][dim] - loc

            prior_dict[f"param{dim+1}"] = pyabc.RV("uniform", loc, scale)

        prior = pyabc.Distribution(**prior_dict)

    else:
        log.info("No support for prior yet")
        raise NotImplementedError

    return prior


def get_distance(distance: str) -> Callable:
    """Return distance function for pyabc."""

    if distance == "l1":

        def distance_fun(x, y):
            abs_diff = abs(x["data"] - y["data"])
            return np.atleast_1d(abs_diff).mean(axis=-1)

    elif distance == "mse":

        def distance_fun(x, y):
            return np.mean((x["data"] - y["data"]) ** 2, axis=-1)
    
    elif distance == "l2":

        def distance_fun(x, y):
            sq_diff = (x["data"] - y["data"])**2
            return np.atleast_1d(sq_diff).mean(axis=-1)

    else:
        raise NotImplementedError(f"Distance '{distance}' not implemented.")

    return distance_fun


def clip_int(value, minimum, maximum):
    value = int(value)
    minimum = int(minimum)
    maximum = int(maximum)
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
