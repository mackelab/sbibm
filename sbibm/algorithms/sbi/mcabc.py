from typing import Optional

import numpy as np
import torch
from sbi.inference import MCABC

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.io import save_tensor_to_csv
from sbibm.utils.kde import get_kde


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_top_samples: Optional[int] = None,
    quantile: Optional[float] = None,
    eps: Optional[float] = None,
    distance: str = "l2",
    batch_size: int = 1000,
    save_distances: bool = False,
    kde_bandwidth: Optional[str] = None,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Runs MC-ABC from `sbi`

    Choose one of `num_top_samples`, `quantile`, `eps`.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_top_samples: If given, will use `top=True` with num_top_samples
        quantile: Quantile to use
        eps: Epsilon threshold to use
        distance: Distance to use
        batch_size: Batch size for simulator
        save_distances: If True, stores distances of samples to disk
        kde_bandwidth: If not None, will resample using KDE when necessary

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    assert not (num_top_samples is None and quantile is None and eps is None)
    # SBI takes only quantile or eps. Derive quantile from num_top_samples if needed.
    if num_top_samples is not None and quantile is None:
        quantile = num_top_samples / num_simulations

    log = sbibm.get_logger(__name__)
    log.info(f"Running MC-ABC")

    prior = task.get_prior_dist()
    simulator = task.get_simulator(max_calls=num_simulations)
    if observation is None:
        observation = task.get_observation(num_observation)

    inference_method = MCABC(
        simulator=simulator,
        prior=prior,
        simulation_batch_size=batch_size,
        distance=distance,
        show_progress_bars=True,
    )
    posterior, distances = inference_method(
        x_o=observation,
        num_simulations=num_simulations,
        eps=eps,
        quantile=quantile,
        return_distances=True,
    )

    if save_distances:
        save_tensor_to_csv("distances.csv", distances)

    assert simulator.num_simulations == num_simulations

    if kde_bandwidth is not None:
        samples = posterior._samples

        log.info(
            f"KDE on {samples.shape[0]} samples with bandwidth option {kde_bandwidth}"
        )
        kde = get_kde(samples, bandwidth=kde_bandwidth)

        return (
            kde.sample(num_samples),
            simulator.num_simulations,
            None,
        )

    samples = posterior.sample((num_samples,)).detach()

    if num_observation is not None:
        true_parameters = task.get_true_parameters(num_observation=num_observation)
        log_prob_true_parameters = posterior.log_prob(true_parameters)
        return samples, simulator.num_simulations, log_prob_true_parameters
    else:
        return samples, simulator.num_simulations, None
