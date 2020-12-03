import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import pyabc
import time
import torch

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.torch import sample_with_weights

from sbibm.utils.kde import get_kde
from sbibm.algorithms.sbi.utils import run_lra, get_sass_transform

from .pyabc_utils import (
    PyAbcSimulator,
    get_distance,
    wrap_prior,
    run_pyabc,
)


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    population_size: Optional[int] = None,
    distance: Optional[str] = "l2",
    epsilon_decay: Optional[float] = 0.5,
    kernel: Optional[str] = "gaussian",
    kernel_variance_scale: Optional[float] = 0.5,
    population_strategy: Optional[str] = "constant",
    use_last_pop_samples: bool = False,
    num_workers: int = 1,
    sass: bool = False,
    sass_sample_weights: bool = False,
    sass_feature_expansion_degree: int = 1,
    sass_fraction: float = 0.5,
    lra: bool = False,
    lra_sample_weights: bool = True,
    kde_bandwidth: Optional[str] = None,
    kde_sample_weights: bool = False,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """ABC-SMC using pyabc toolbox

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        population_size: If None, uses heuristic: 1000 if `num_simulations` is greater
            than 10k, else 100
        distance: Distance function, options = {l1, l2, mse}
        epsilon_decay: Decay for epsilon, quantile based.
        kernel: Kernel distribution used to perturb the particles.
        kernel_variance_scale: Scaling factor for kernel variance.
        sass: If True, summary statistics are learned as in
            Fearnhead & Prangle 2012.
        sass_sample_weights: Whether to weigh SASS samples
        sass_feature_expansion_degree: Degree of polynomial expansion of the summary
            statistics.
        sass_fraction: Fraction of simulation budget to use for sass.
        lra: If True, posterior samples are adjusted with
            linear regression as in Beaumont et al. 2002.
        lra_sample_weights: Whether to weigh LRA samples
        kde_bandwidth: If not None, will resample using KDE when necessary, set
            e.g. to "cv" for cross-validated bandwidth selection
        kde_sample_weights: Whether to weigh KDE samples
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    log = sbibm.get_logger(__name__)
    time_flag = time.time()

    # Wrap sbibm prior and simulator for pyABC
    prior = wrap_prior(task)
    simulator = PyAbcSimulator(task)

    distance = get_distance(distance)
    if observation is None:
        observation = task.get_observation(num_observation)
    observation = np.atleast_1d(np.array(observation, dtype=float).squeeze())

    # Epsilon schedule
    epsilon = pyabc.epsilon.QuantileEpsilon(
        initial_epsilon="from_sample", alpha=epsilon_decay
    )

    # Perturbation kernel
    transition = pyabc.transition.MultivariateNormalTransition(
        scaling=kernel_variance_scale
    )

    # Population size strategy
    if population_size is None:
        population_size = 100
        if num_simulations > 10_000:
            population_size = 1000

    population_size = min(population_size, num_simulations)

    if population_strategy == "constant":
        population_size_strategy = population_size
    elif population_strategy == "adaptive":
        raise NotImplementedError("Not implemented atm.")
        population_size_strategy = pyabc.populationstrategy.AdaptivePopulationSize(
            start_nr_particles=population_size,
            max_population_size=int(10 * population_size),
            min_population_size=int(0.1 * population_size),
        )

    # Multiprocessing
    if num_workers > 1:
        sampler = pyabc.sampler.MulticoreParticleParallelSampler(n_procs=num_workers)
    else:
        sampler = pyabc.sampler.SingleCoreSampler()

    # Collect kwargs
    kwargs = dict(
        parameter_priors=[prior],
        distance_function=distance,
        population_size=population_size_strategy,
        transitions=[transition],
        eps=epsilon,
        sampler=sampler,
    )

    # Semi-automatic summary statistics.
    if sass:
        num_pilot_simulations = int(sass_fraction * num_simulations)
        log.info(f"SASS pilot run with {num_pilot_simulations} simulations.")
        pilot_db = "sqlite:///" + os.path.join(
            tempfile.gettempdir(), f"sassrun_{time_flag}.db"
        )
        kwargs["models"] = [simulator]

        # Run pyabc with fixed budget.
        pilot_theta, pilot_weights = run_pyabc(
            pilot_db,
            num_pilot_simulations,
            observation,
            pyabc_kwargs=kwargs,
            prior=task.get_prior(),
        )

        # Regression
        # TODO: Posterior does not return xs, which we would need for
        # regression adjustment. So we will resimulate, which is
        # unneccessary. Should ideally change `inference_method` to return xs
        # if requested instead. This step thus does not count towards budget
        pilot_x = task.get_simulator(max_calls=None)(pilot_theta)

        # Run SASS.
        sumstats_transform = get_sass_transform(
            theta=pilot_theta,
            x=pilot_x,
            expansion_degree=sass_feature_expansion_degree,
            sample_weight=pilot_weights if sass_sample_weights else None,
        )

        # Update simulator to use sass summary stats.
        def sumstats_simulator(theta):
            # Pyabc simulator returns dict.
            x = simulator(theta)["data"].reshape(1, -1)
            # Transform return Tensor.
            sx = sumstats_transform(x)
            return dict(data=sx.numpy().squeeze())

        observation = sumstats_transform(observation.reshape(1, -1))
        observation = np.atleast_1d(np.array(observation, dtype=float).squeeze())
        log.info(f"Finished learning summary statistics.")
    else:
        sumstats_simulator = simulator
        num_pilot_simulations = 0
        population_size = min(population_size, num_simulations)

    log.info(
        f"""Running ABC-SMC-pyabc with {num_simulations - num_pilot_simulations}
    simulations"""
    )
    db = "sqlite:///" + os.path.join(tempfile.gettempdir(), f"pyabc_{time_flag}.db")
    kwargs["models"] = [sumstats_simulator]

    # Run pyabc with fixed budget.
    particles, weights = run_pyabc(
        db,
        num_simulations=num_simulations - num_pilot_simulations,
        observation=observation,
        pyabc_kwargs=kwargs,
        prior=task.get_prior(),
    )

    if lra:
        log.info(f"Running linear regression adjustment.")
        # TODO: Posterior does not return xs, which we would need for
        # regression adjustment. So we will resimulate, which is
        # unneccessary. Should ideally change `inference_method` to return xs
        # if requested instead.
        xs = task.get_simulator(max_calls=None)(particles)

        # NOTE: If posterior is bounded we should do the regression in
        # unbounded space, as described in https://arxiv.org/abs/1707.01254
        transform_to_unbounded = True
        transforms = task._get_transforms(transform_to_unbounded)["parameters"]

        # Update the particles with LRA.
        particles = run_lra(
            theta=particles,
            x=xs,
            observation=torch.tensor(observation, dtype=torch.float32).unsqueeze(0),
            sample_weight=weights if lra_sample_weights else None,
            transforms=transforms,
        )

        # TODO: Maybe set weights uniform because they can't be updated?
        # weights = torch.ones(particles.shape[0]) / particles.shape[0]

    if kde_bandwidth is not None:
        samples = particles

        log.info(
            f"KDE on {samples.shape[0]} samples with bandwidth option {kde_bandwidth}"
        )
        kde = get_kde(
            samples,
            bandwidth=kde_bandwidth,
            sample_weight=weights if kde_sample_weights else None,
        )
        samples = kde.sample(num_samples)
    else:
        log.info(f"Sampling {num_samples} samples from trace")
        samples = sample_with_weights(particles, weights, num_samples=num_samples)

    log.info(f"Unique samples: {torch.unique(samples, dim=0).shape[0]}")

    return samples, simulator.simulator.num_simulations, None
