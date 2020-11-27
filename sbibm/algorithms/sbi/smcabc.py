from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sbi.inference import SMCABC
from sklearn.linear_model import LinearRegression

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.kde import get_kde

from .utils import clip_int


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    population_size: Optional[int] = None,
    distance: str = "l2",
    epsilon_quantile: float = 0.5,
    distance_based_decay: bool = True,
    ess_min: float = 0.5,
    initial_round_factor: int = 5,
    batch_size: int = 1000,
    kernel: str = "gaussian",
    kernel_variance_scale: float = 0.5,
    use_last_pop_samples: bool = False,
    algorithm_variant: str = "C",
    save_summary: bool = False,
    learn_summary_statistics: bool = False,
    linear_regression_adjustment: bool = False,
    kde_bandwidth: Optional[str] = None,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs SMC-ABC from `sbi`

    SMC-ABC supports two different ways of scheduling epsilon:
    1) Exponential decay: eps_t+1 = epsilon_decay * eps_t
    2) Distance based decay: the new eps is determined from the "epsilon_decay" 
        quantile of the distances of the accepted simulations in the previous population. This is used if `distance_based_decay` is set to True.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        population_size: If None, uses heuristic: 1000 if `num_simulations` is greater
            than 10k, else 100
        distance: Distance function, options = {l1, l2, mse}
        epsilon_quantile: Decay for epsilon
        distance_based_decay: Whether to determine new epsilon from quantile of
            distances of the previous population.
        ess_min: Threshold for resampling a population if effective sampling size is too
            small.
        initial_round_factor: Used to determine initial round size
        batch_size: Batch size for the simulator
        kernel: Kernel distribution used to perturb the particles.
        kernel_variance_scale: Scaling factor for kernel variance.
        use_last_pop_samples: If True, samples of a population that was quit due to
            budget are used by filling up missing particles from the previous
            population.
        algorithm_variant: There are three SMCABC variants implemented: A, B, and C.
            See doctstrings in SBI package for more details.
        save_summary: Whether to save a summary containing all populations, distances,
            etc. to file.
        learn_summary_statistics: If True, summary statistics are learned as in
            Fearnhead & Prangle 2012.
        linear_regression_adjustment: If True, posterior samples are adjusted with
            linear regression as in Beaumont et al. 2002.
        kde_bandwidth: If not None, will resample using KDE when necessary, set
            e.g. to "cv" for cross-validated bandwidth selection

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = sbibm.get_logger(__name__)
    smc_papers = dict(A="Toni 2010", B="Sisson et al. 2007", C="Beaumont et al. 2009")
    log.info(f"Running SMC-ABC as in {smc_papers[algorithm_variant]}.")

    prior = task.get_prior_dist()
    simulator = task.get_simulator(max_calls=num_simulations)
    if observation is None:
        observation = task.get_observation(num_observation)

    if population_size is None:
        population_size = 100
        if num_simulations > 10_000:
            population_size = 1000

    if learn_summary_statistics:
        # Pilot run
        log.info(f"Pilot run for semi-automatic summary stats.")

        num_pilot_simulations = int(num_simulations / 2)

        population_size = min(population_size, num_pilot_simulations)

        initial_round_size = clip_int(
            value=initial_round_factor * population_size,
            minimum=population_size,
            maximum=max(0.5 * num_pilot_simulations, population_size),
        )

        inference_method = SMCABC(
            simulator=simulator,
            prior=prior,
            simulation_batch_size=batch_size,
            distance=distance,
            show_progress_bars=True,
            kernel=kernel,
            algorithm_variant=algorithm_variant,
        )
        pilot_posterior = inference_method(
            x_o=observation,
            num_particles=population_size,
            num_initial_pop=initial_round_size,
            num_simulations=num_pilot_simulations,
            epsilon_decay=epsilon_quantile,
            distance_based_decay=distance_based_decay,
            ess_min=ess_min,
            kernel_variance_scale=kernel_variance_scale,
            use_last_pop_samples=use_last_pop_samples,
            return_summary=False,
        )

        # Regression
        pilot_theta = pilot_posterior._samples
        # TODO: Posterior does not return xs, which we would need for
        # regression adjustment. So we will resimulate, which is
        # unneccessary. Should ideally change `inference_method` to return xs
        # if requested instead. This step thus does not count towards budget
        pilot_x = task.get_simulator(max_calls=None)(pilot_theta)
        sumstats_map = np.zeros((task.dim_data, task.dim_parameters))

        for parameter_idx in range(task.dim_parameters):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=pilot_x,
                y=pilot_theta[:, parameter_idx],
                sample_weight=pilot_posterior._log_weights.exp(),
            )
            sumstats_map[:, parameter_idx] = regression_model.coef_
        sumstats_map = torch.tensor(sumstats_map, dtype=torch.float32)

        def sumstats_transform(x):
            return x.mm(sumstats_map)

        sumstats_simulator = lambda theta: sumstats_transform(simulator(theta))
        observation = sumstats_transform(observation)
        log.info(f"Finished learning summary statistics.")
    else:
        sumstats_simulator = simulator
        num_pilot_simulations = 0

        population_size = min(population_size, num_simulations)

        initial_round_size = clip_int(
            value=initial_round_factor * population_size,
            minimum=population_size,
            maximum=max(0.5 * num_simulations, population_size),
        )

    inference_method = SMCABC(
        simulator=sumstats_simulator,
        prior=prior,
        simulation_batch_size=batch_size,
        distance=distance,
        show_progress_bars=True,
        kernel=kernel,
        algorithm_variant=algorithm_variant,
    )
    posterior, summary = inference_method(
        x_o=observation,
        num_particles=population_size,
        num_initial_pop=initial_round_size,
        num_simulations=num_simulations - num_pilot_simulations,
        epsilon_decay=epsilon_quantile,
        distance_based_decay=distance_based_decay,
        ess_min=ess_min,
        kernel_variance_scale=kernel_variance_scale,
        use_last_pop_samples=use_last_pop_samples,
        return_summary=True,
    )

    if save_summary:
        log.info("Saving smcabc summary to csv.")
        pd.DataFrame.from_dict(summary,).to_csv("summary.csv", index=False)

    assert simulator.num_simulations == num_simulations

    if linear_regression_adjustment:
        log.info(f"Running linear regression adjustment.")
        samples = posterior._samples

        # TODO: Posterior does not return xs, which we would need for
        # regression adjustment. So we will resimulate, which is
        # unneccessary. Should ideally change `inference_method` to return xs
        # if requested instead.
        xs = task.get_simulator(max_calls=None)(samples)

        # NOTE: If posterior is bounded we should do the regression in
        # unbounded space, as described in https://arxiv.org/abs/1707.01254
        transform_to_unbounded = True
        transforms = task._get_transforms(transform_to_unbounded)["parameters"]

        samples_adjusted = transforms(samples)
        for parameter_idx in range(task.dim_parameters):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=xs,
                y=samples[:, parameter_idx],
                sample_weight=posterior._log_weights.exp(),
            )
            samples_adjusted[:, parameter_idx] += regression_model.predict(observation)
            samples_adjusted[:, parameter_idx] -= regression_model.predict(xs)

        posterior._samples = transforms.inv(samples_adjusted)

    samples = posterior.sample((num_samples,)).detach()

    if kde_bandwidth is not None:
        log.info(
            f"KDE on {samples.shape[0]} samples with bandwidth option {kde_bandwidth}"
        )
        kde = get_kde(samples, bandwidth=kde_bandwidth)

        samples = kde.sample(num_samples)

    if num_observation is not None:
        true_parameters = task.get_true_parameters(num_observation=num_observation)
        log_prob_true_parameters = posterior.log_prob(true_parameters)
        return samples, simulator.num_simulations, log_prob_true_parameters
    else:
        return samples, simulator.num_simulations, None
