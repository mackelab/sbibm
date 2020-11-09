from __future__ import annotations

import os
import tempfile
import warnings
from typing import Optional

import numpy as np
import pyabc
import torch

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.torch import sample_with_weights

from .pyabc_utils import PyAbcSimulator, clip_int, get_distance, wrap_prior


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    population_size: Optional[int] = None,
    distance: Optional[str] = "l1",
    epsilon_quantile: Optional[float] = 0.5,
    verbose: bool = True,
    kernel: Optional[str] = "gaussian",
    kernel_variance_scale: Optional[float] = 1.0,
    population_strategy: Optional[str] = "constant",
    num_workers: int = 1,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """ABC-SMC using pyabc toolbox.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        population_size: If None, uses heuristic
        distance: Distance function, options = {l1, l2, mse}
        epsilon_decay: Decay for epsilon
        ess_min: Minimum ESS
        initial_round_factor: Used to determine initial round size
        verbose: Verbosity flag
        kernel: Kernel distribution used to perturb the particles.
        kernel_variance_scale: Scaling factor for kernel variance.
        use_last_pop_samples: If True, samples of a population that was quit due to
            budget are used by filling up missing particles from the previous
            population.

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = sbibm.get_logger(__name__)

    # Wrap sbibm prior and simulator for pyABC.
    prior = wrap_prior(task)
    simulator = PyAbcSimulator(task)

    # import pdb; pdb.set_trace()
    # prior, simulator = get_prior_and_simulator(task)

    distance = get_distance(distance)
    if observation is None:
        observation = task.get_observation(num_observation)
    observation = np.atleast_1d(np.array(observation, dtype=float).squeeze())

    # Epsilon schedule.
    epsilon = pyabc.epsilon.QuantileEpsilon(
        initial_epsilon="from_sample", alpha=epsilon_quantile
    )

    # Perturbation kernel.
    transition = pyabc.transition.MultivariateNormalTransition(
        scaling=kernel_variance_scale
    )

    # Population size strategy.
    if population_size is None:
        population_size = clip_int(
            value=0.1 * num_simulations, minimum=500, maximum=num_samples / 2,
        )

    population_size = min(population_size, num_simulations)

    if population_strategy == "constant":
        population_size_strategy = population_size
    elif population_strategy == "adaptive":
        population_size_strategy = pyabc.populationstrategy.AdaptivePopulationSize(
            start_nr_particles=population_size,
            max_population_size=int(10 * population_size),
            min_population_size=int(0.1 * population_size),
        )

    # Multiprocessing.
    if num_workers > 1:
        sampler = pyabc.sampler.MulticoreParticleParallelSampler(n_procs=num_workers)
    else:
        sampler = pyabc.sampler.SingleCoreSampler()

    # Collect kwargs
    kwargs = dict(
        models=[simulator],
        parameter_priors=[prior],
        distance_function=distance,
        population_size=population_size_strategy,
        transitions=[transition],
        eps=epsilon,
        sampler=sampler,
    )

    log.info(f"Starting to run ABC-SMC-pyabc")
    db = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")

    # initial run
    abc = pyabc.ABCSMC(**kwargs)
    abc.new(db, {"data": observation})
    history = abc.run(max_nr_populations=1)
    run_id = history.id
    num_calls = history.total_nr_simulations

    # Subtract one population size from total budget to avoid starting new population
    # just before the budget is over.
    budget = num_simulations
    budget_left = budget > num_calls

    if budget_left:
        while budget_left:

            abc_continued = pyabc.ABCSMC(**kwargs)

            abc_continued.load(db, run_id)

            history = abc_continued.run(max_nr_populations=1)
            run_id = history.id

            num_calls = history.total_nr_simulations

            budget_left = budget > num_calls

        # We allow 10% over the budget.
        if num_calls > budget * 1.1:
            log.info(
                f"pyabc exceeded budget by more than 10 percent, returning previous population."
            )
            # Return previous population.
            (particles_df, weights) = history.get_distribution(t=history.max_t - 1)
        else:
            # Return current population.
            (particles_df, weights) = history.get_distribution(t=history.max_t)

        particles = torch.as_tensor(particles_df.values, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)

        log.info(f"Sampling {num_samples} samples from trace")
        samples = sample_with_weights(particles, weights, num_samples=num_samples)
    # This happens when the initial population already used up the budget.
    # Then we just return the prior samples.
    else:
        log.info(f"pyabc exceeded budget in initial run. Returning prior samples!")
        log.info(f"Sampling {num_samples} samples from prior")
        samples = task.get_prior()(num_samples=num_samples)

    log.info(f"Unique samples: {torch.unique(samples, dim=0).shape[0]}")

    if num_calls > num_simulations:
        warnings.warn(f"Simualtion budget exceeded: {num_simulations} << {num_calls}")

    return samples, simulator.simulator.num_simulations, None
