from typing import Optional, Tuple

import numpy as np
import torch
from pyro.distributions.empirical import Empirical
from sbi.inference import MCABC
from sklearn.linear_model import LinearRegression

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
    num_top_samples: Optional[int] = 100,
    quantile: Optional[float] = None,
    eps: Optional[float] = None,
    distance: str = "l2",
    batch_size: int = 1000,
    save_distances: bool = False,
    kde_bandwidth: Optional[str] = None,
    learn_summary_statistics: bool = False,
    linear_regression_adjustment: bool = False,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs REJ-ABC from `sbi`

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
        kde_bandwidth: If not None, will resample using KDE when necessary, set
            e.g. to "cv" for cross-validated bandwidth selection
        learn_summary_statistics: If True, summary statistics are learned as in
            Fearnhead & Prangle 2012.
        linear_regression_adjustment: If True, posterior samples are adjusted with
            linear regression as in Beaumont et al. 2002.
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    assert not (num_top_samples is None and quantile is None and eps is None)

    log = sbibm.get_logger(__name__)
    log.info(f"Running REJ-ABC")

    prior = task.get_prior_dist()
    simulator = task.get_simulator(max_calls=num_simulations)
    if observation is None:
        observation = task.get_observation(num_observation)

    if learn_summary_statistics:
        # Pilot run
        log.info(f"Pilot run for semi-automatic summary stats.")
        num_pilot_simulations = int(num_simulations / 2)
        if num_top_samples is not None and quantile is None:
            quantile = num_top_samples / num_pilot_simulations

        inference_method = MCABC(
            simulator=simulator,
            prior=prior,
            simulation_batch_size=batch_size,
            distance=distance,
            show_progress_bars=True,
        )
        pilot_posterior = inference_method(
            x_o=observation,
            num_simulations=num_pilot_simulations,
            eps=None,
            quantile=quantile,
            return_distances=False,
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
            regression_model.fit(X=pilot_x, y=pilot_theta[:, parameter_idx])
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

        # SBI takes only quantile or eps. Derive quantile from num_top_samples if needed.
        if num_top_samples is not None and quantile is None:
            quantile = num_top_samples / num_simulations

    inference_method = MCABC(
        simulator=sumstats_simulator,
        prior=prior,
        simulation_batch_size=batch_size,
        distance=distance,
        show_progress_bars=True,
    )
    posterior, distances = inference_method(
        x_o=observation,
        num_simulations=num_simulations - num_pilot_simulations,
        eps=eps,
        quantile=quantile,
        return_distances=True,
    )

    assert simulator.num_simulations == num_simulations

    if save_distances:
        save_tensor_to_csv("distances.csv", distances)

    if linear_regression_adjustment:
        samples = posterior._samples

        # TODO: Posterior does not return xs, which we would need for
        # regression adjustment. So we will resimulate, which is
        # unneccessary. Should ideally change `inference_method` to return xs
        # if requested instead. This step thus does not count towards budget
        xs = task.get_simulator(max_calls=None)(samples)

        # NOTE: If posterior is bounded we should do the regression in
        # unbounded space, as described in https://arxiv.org/abs/1707.01254
        transform_to_unbounded = True
        transforms = task._get_transforms(transform_to_unbounded)["parameters"]

        samples_adjusted = transforms(samples)
        for parameter_idx in range(task.dim_parameters):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(X=xs, y=samples[:, parameter_idx])
            samples_adjusted[:, parameter_idx] += regression_model.predict(observation)
            samples_adjusted[:, parameter_idx] -= regression_model.predict(xs)

        posterior._samples = transforms.inv(samples_adjusted)

    if kde_bandwidth is not None:
        samples = posterior._samples

        log.info(
            f"KDE on {samples.shape[0]} samples with bandwidth option {kde_bandwidth}"
        )
        kde = get_kde(samples, bandwidth=kde_bandwidth)

        samples = kde.sample(num_samples)
    else:
        samples = posterior.sample((num_samples,)).detach()

    if num_observation is not None:
        true_parameters = task.get_true_parameters(num_observation=num_observation)
        log_prob_true_parameters = posterior.log_prob(true_parameters)
        return samples, simulator.num_simulations, log_prob_true_parameters
    else:
        return samples, simulator.num_simulations, None
