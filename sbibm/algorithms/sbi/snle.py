import logging
import math
from typing import Dict, Optional

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 1,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 100,
    automatic_transforms_enabled: bool = False,
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = {},
    z_score_x: bool = True,
    z_score_theta: bool = True,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Runs (S)NLE from `sbi`

    Args:
        task: Task instance
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of maf / mdn / made / nsf
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NLE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNLE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    prior = wrap_prior_dist(prior, transforms)
    simulator = wrap_simulator_fn(simulator, transforms)

    density_estimator = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference_method = inference.SNLE_A(
        density_estimator=density_estimator,
        simulator=simulator,
        prior=prior,
        mcmc_method=mcmc_method,
        mcmc_parameters=mcmc_parameters,
        simulation_batch_size=simulation_batch_size,
        show_round_summary=True,
    )

    posteriors = []
    proposal = None
    for r in range(num_rounds):
        posterior = inference_method(
            num_simulations=num_simulations_per_round,
            proposal=proposal,
            training_batch_size=training_batch_size,
        )
        posterior.set_default_x(observation)
        mcmc_parameters["warmup_steps"] = 25
        if r > 1:
            mcmc_parameters["init_strategy"] = "latest_sample"
            posterior.set_mcmc_parameters(mcmc_parameters)
        proposal = posterior
        posteriors.append(posterior)

    posterior = wrap_posterior(posterior, transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    return samples, simulator.num_simulations, None
