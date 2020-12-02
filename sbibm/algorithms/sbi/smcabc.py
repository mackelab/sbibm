from typing import Optional, Tuple

import pandas as pd
import torch
from sbi.inference import SMCABC
from sklearn.linear_model import LinearRegression

import sbibm
from sbibm.tasks.task import Task
from sbibm.utils.kde import get_kde

from .utils import clip_int, get_sass_transform, run_lra


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    population_size: Optional[int] = None,
    distance: str = "l2",
    epsilon_decay: float = 0.5,
    distance_based_decay: bool = True,
    ess_min: float = 0.5,
    initial_round_factor: int = 5,
    batch_size: int = 1000,
    kernel: str = "gaussian",
    kernel_variance_scale: float = 0.5,
    use_last_pop_samples: bool = False,
    algorithm_variant: str = "C",
    save_summary: bool = False,
    sass: bool = False,
    sass_sample_weights: bool = False,
    sass_fraction: float = 0.5,
    sass_feature_expansion_degree: int = 1,
    lra: bool = False,
    lra_sample_weights: bool = True,
    kde_bandwidth: Optional[str] = None,
    kde_sample_weights: bool = False,
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
        epsilon_decay: Decay for epsilon; treated as quantile in case of distance based decay.
        distance_based_decay: Whether to determine new epsilon from quantile of
            distances of the previous population.
        ess_min: Threshold for resampling a population if effective sampling size is 
            too small.
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
        sass: If True, summary statistics are learned as in
            Fearnhead & Prangle 2012.
        sass_sample_weights: Whether to weigh SASS samples
        sass_fraction: Fraction of simulation budget to use for sass.
        sass_feature_expansion_degree: Degree of polynomial expansion of the summary
            statistics.
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

    if sass:
        # Pilot run
        log.info(f"Pilot run for semi-automatic summary stats.")

        num_pilot_simulations = int(sass_fraction * num_simulations)

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
            epsilon_decay=epsilon_decay,
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

        # Run SASS with weights.
        sumstats_transform = get_sass_transform(
            theta=pilot_theta,
            x=pilot_x,
            expansion_degree=sass_feature_expansion_degree,
            sample_weight=pilot_posterior._log_weights.exp()
            if sass_sample_weights
            else None,
        )

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
        epsilon_decay=epsilon_decay,
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

    if lra:
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

        # Adjust with LRA.
        lra_samples = run_lra(
            samples,
            xs,
            observation,
            sample_weight=posterior._log_weights.exp() if lra_sample_weights else None,
            transforms=transforms,
        )

        # Update SMC weights with LRA adjusted weights
        new_log_weights = inference_method._calculate_new_log_weights(
            new_particles=lra_samples,
            old_particles=samples,
            old_log_weights=posterior._log_weights,
        )

        # Update posterior
        posterior._samples = lra_samples
        posterior._log_weights = new_log_weights

    if kde_bandwidth is not None:
        samples = posterior._samples

        log.info(
            f"KDE on {samples.shape[0]} samples with bandwidth option {kde_bandwidth}"
        )

        kde = get_kde(
            samples,
            bandwidth=kde_bandwidth,
            sample_weight=posterior._log_weights.exp() if kde_sample_weights else None,
        )
        samples = kde.sample(num_samples)
    else:
        samples = posterior.sample((num_samples,)).detach()

    if num_observation is not None:
        true_parameters = task.get_true_parameters(num_observation=num_observation)
        log_prob_true_parameters = posterior.log_prob(true_parameters)
        return samples, simulator.num_simulations, log_prob_true_parameters
    else:
        return samples, simulator.num_simulations, None
