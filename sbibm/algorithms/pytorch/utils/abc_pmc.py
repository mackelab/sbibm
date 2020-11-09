from __future__ import annotations

import logging
import sys
from typing import Callable, Optional, Tuple

import torch
from pyro.distributions import Uniform
from torch.distributions import Distribution, Multinomial, MultivariateNormal

from sbibm.utils.exceptions import SimulationBudgetExceeded

from .distances import (
    l1_distance_over_batch,
    l2_distance_over_batch,
    mean_squared_distance_over_batch,
)


class ABCPMC:
    """ABC PMC as in Beaumont 2002 and 2009."""

    def __init__(
        self,
        prior: Distribution,
        simulate: Callable,
        observation: torch.Tensor,
        distance: Optional[str] = "mse",
        kernel: Optional[str] = "gaussian",
    ):

        self.prior = prior
        self.simulation_fun = simulate
        assert observation.ndim == 2, "observation needs a batch dim of 1"
        assert kernel in ("gaussian", "uniform"), f"Kernel '{kernel}' not supported."
        self.x0 = observation

        if distance == "mse":
            distance_fun = mean_squared_distance_over_batch
        elif distance == "l1":
            distance_fun = l1_distance_over_batch
        elif distance == "l2":
            distance_fun = l2_distance_over_batch
        else:
            raise ValueError(f"Distance type '{distance}' is not supported.")

        self.distance_to_x0 = lambda x: distance_fun(observation, x)
        self.num_simulations = 0
        self.num_simulation_budget = 0
        self.kernel = kernel
        self.logger = logging.getLogger(__name__)

    def run_abcpmc(
        self,
        num_particles: int,
        num_initial_pop: int,
        epsilon_decay: float,
        num_simulation_budget: int,
        ess_min: Optional[float] = None,
        batch_size: int = 100,
        qt_decay: bool = False,
        kernel_variance_scale: Optional[float] = 1.0,
        use_last_pop_samples: Optional[bool] = True,
    ):
        pop_idx = 0
        self.num_simulation_budget = num_simulation_budget

        # run initial population
        particles, epsilon, distances = self._sample_initial_population(
            num_particles, num_initial_pop, batch_size=batch_size
        )
        log_weights = torch.log(1 / num_particles * torch.ones(num_particles))

        self.logger.info(
            f"population={pop_idx}, eps={epsilon}, ess={1.0}, num_sims={num_initial_pop}\n"
        )

        all_particles = [particles]
        all_log_weights = [log_weights]
        all_distances = [distances]
        all_epsilons = [epsilon]
        all_qts = [epsilon_decay]

        while self.num_simulations < num_simulation_budget:

            pop_idx += 1
            if qt_decay:
                epsilon = self._get_next_epsilon(
                    all_distances[pop_idx - 1], all_qts[pop_idx - 1]
                )
            else:
                epsilon *= epsilon_decay

            # Get kernel variance from previous pop.
            self.kernel_variance = self.get_kernel_variance(
                all_particles[pop_idx - 1],
                torch.exp(all_log_weights[pop_idx - 1]),
                num_samples=1000,
                kernel_variance_scale=kernel_variance_scale,
            )
            num_simulations_pre = self.num_simulations
            try:
                particles, log_weights, distances = self._sample_next_population(
                    particles=all_particles[pop_idx - 1],
                    log_weights=all_log_weights[pop_idx - 1],
                    distances=all_distances[pop_idx - 1],
                    epsilon=epsilon,
                    batch_size=batch_size,
                    use_last_pop_samples=use_last_pop_samples,
                )
            except SimulationBudgetExceeded:
                self.logger.info("Simulation budget exceeded, quit simulation loop.")
                break

            # resample if ess too low
            if ess_min is not None:
                ess = (
                    1 / torch.sum(torch.exp(2.0 * log_weights), dim=0)
                ) / num_particles
                if ess < ess_min:
                    self.logger.info(
                        f"ESS={ess:.2f} too low, resampling pop {pop_idx}..."
                    )

                    particles = self.sample_from_population_with_weights(
                        particles, torch.exp(log_weights), num_samples=num_particles
                    )
                    log_weights = torch.log(
                        1 / num_particles * torch.ones(num_particles)
                    )

            # calculate new qt from acceptance rate
            acc = num_particles / (self.num_simulations - num_simulations_pre)
            qt = 1 - acc

            self.logger.info(
                f"population={pop_idx} done: eps={epsilon:.6f}, ess={ess:.2f}, num_sims={self.num_simulations}, acc={acc:.4f}\n"
            )

            # collect results
            all_particles.append(particles)
            all_log_weights.append(log_weights)
            all_distances.append(distances)
            all_epsilons.append(epsilon)
            all_qts.append(qt)

        return all_particles, all_log_weights, all_epsilons, all_distances

    def _sample_initial_population(
        self,
        num_particles: int,
        num_initial_pop: int,
        logger=sys.stdout,
        batch_size: int = 100,
    ) -> Tuple[torch.Tensor, float]:

        assert (
            num_initial_pop <= num_initial_pop
        ), "number of initial round simulations must be greater than population size"

        parameters = self.prior.sample((num_initial_pop,))
        data = self._run_simulations(parameters, batch_size=batch_size)
        distances = self.distance_to_x0(data)
        sortidx = torch.argsort(distances)
        particles = parameters[sortidx][:num_particles]
        initial_epsilon = distances[sortidx][num_particles - 1]

        if not torch.isfinite(initial_epsilon):
            initial_epsilon = 1e8

        return particles, initial_epsilon, distances[sortidx][:num_particles]

    def _sample_next_population(
        self,
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        distances: torch.Tensor,
        epsilon: float,
        batch_size: int = 100,
        use_last_pop_samples: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # new_particles = torch.zeros_like(particles)
        # new_log_weights = torch.zeros_like(log_weights)
        new_particles = []
        new_log_weights = []
        new_distances = []

        num_accepted_particles = 0
        num_particles = particles.shape[0]

        while num_accepted_particles < num_particles:

            # make sure batch size doesn't exceed budged or population
            num_batch = min(
                batch_size,
                num_particles - num_accepted_particles,
                self.num_simulation_budget - self.num_simulations,
            )

            # sample and perturb from previous population
            particle_candidates = self._sample_and_perturb(
                particles, torch.exp(log_weights), num_samples=num_batch
            )
            # simulate and select
            data = self._run_simulations(particle_candidates, batch_size=num_batch)
            dists = self.distance_to_x0(data)
            is_accepted = dists <= epsilon
            num_accepted_batch = is_accepted.sum().item()

            if num_accepted_batch > 0:
                # set new particles
                new_particles.append(particle_candidates[is_accepted])
                # calculate and set new weights
                new_log_weights.append(
                    self._calculate_new_log_weights(
                        particle_candidates[is_accepted], particles, log_weights,
                    )
                )
                new_distances.append(dists[is_accepted])
                num_accepted_particles += num_accepted_batch

            if self.num_simulations >= self.num_simulation_budget:
                if use_last_pop_samples:
                    num_remaining = num_particles - num_accepted_particles
                    self.logger.info(
                        f"""Simulation Budget exceeded, filling up with {num_remaining}
                        samples from last population."""
                    )
                    # Some new particles have been accepted already, therefore
                    # fill up the remaining once with old particles and weights.
                    new_particles.append(particles[:num_remaining, :])
                    # Recalculate weights with new particles.
                    new_log_weights = [
                        self._calculate_new_log_weights(
                            torch.cat(new_particles), particles, log_weights,
                        )
                    ]
                    new_distances.append(distances[:num_remaining])
                else:
                    self.logger.info(
                        f"Simulation Budget exceeded, returning previous population."
                    )
                    new_particles = [particles]
                    new_log_weights = [log_weights]
                    new_distances = [distances]

                break

        # collect lists of tensors into tensors
        new_particles = torch.cat(new_particles)
        new_log_weights = torch.cat(new_log_weights)
        new_distances = torch.cat(new_distances)

        # normalize the new weights
        new_log_weights -= torch.logsumexp(new_log_weights, dim=0)

        return new_particles, new_log_weights, new_distances

    def _get_next_epsilon(self, distances: torch.Tensor, quantile: float) -> float:
        """Return epsilon for next round based on quantile of this round's distances.

        Note: distances are made unique to avoid repeated distances from simulations 
        that result in the same observation. 

        Arguments:
            distances  -- The distances accepted in this round. 
            quantile -- quantile in the distance distribution to determine new epsilon
        
        Returns:
            epsilon -- epsilon for the next population.
        """
        # make distances unique to skip simulations with same outcome
        distances = torch.unique(distances)
        # sort distances
        distances = distances[torch.argsort(distances)]
        # get cumsum as cdf proxy
        distances_cdf = torch.cumsum(distances, dim=0) / distances.sum()
        # take the q quantile of distances
        try:
            qidx = torch.where(distances_cdf >= quantile)[0][0]
        except IndexError as err:
            self.logger.warning(
                f"Accepted unique distances={distances} dont match quantile={quantile:.2f}. Selecting last distance."
            )
            qidx = -1

        # the new epsilon is given by that distance
        return distances[qidx].item()

    def _calculate_new_log_weights(
        self,
        new_particles: torch.Tensor,
        old_particles: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> torch.Tensor:

        # prior can be batched across new particles
        prior_log_probs = self.prior.log_prob(new_particles)

        # contstruct function to get kernel log prob for given new particle
        # the kernel is centered on each new particle
        kernel_log_prob = lambda newpar: self.get_new_kernel(newpar).log_prob(
            old_particles
        )

        # but we have to loop over particles here because
        # the kernel log probs are already batched across old particles
        log_weighted_sum = torch.tensor(
            [
                torch.logsumexp(log_weights + kernel_log_prob(newpar), dim=0)
                for newpar in new_particles
            ]
        )
        # new weights are prior probs over weighted sum:
        return prior_log_probs - log_weighted_sum

    def _run_simulations(
        self, parameters: torch.Tensor, batch_size: int = 100
    ) -> torch.Tensor:

        parameter_batches = torch.chunk(
            parameters, max(1, int(parameters.shape[0] / batch_size)), dim=0
        )
        data = []
        for batch in parameter_batches:
            ds = self.simulation_fun(batch)
            data.append(ds)
            self.num_simulations += batch.shape[0]

        # return list of tensors collected into single tensor
        return torch.cat(data)

    @staticmethod
    def sample_from_population_with_weights(
        particles: torch.Tensor, weights: torch.Tensor, num_samples: int = 1
    ):
        # define multinomial with weights as probs
        multi = Multinomial(probs=weights)
        # sample num samples, with replacement
        samples = multi.sample(sample_shape=(num_samples,))
        # get indices of success trials
        indices = torch.where(samples)[1]
        # return those indices from trace
        return particles[indices]

    def _sample_and_perturb(
        self, particles: torch.Tensor, weights: torch.Tensor, num_samples: int = 1
    ):
        """Sample and perturb batch of new parameters from trace. 

        Reject sampled and perturbed parameters outside of prior. 
        
        Arguments:
            trace {torch.Tensor} -- [description]
            weights {torch.Tensor} -- [description]
        
        Keyword Arguments:
            num_samples {int} -- [description] (default: {1})
        """

        num_accepted = 0
        parameters = []
        while num_accepted < num_samples:
            # sample
            parms = self.sample_from_population_with_weights(
                particles, weights, num_samples - num_accepted
            )

            # create kernel on params and perturb
            parms_perturbed = self.get_new_kernel(parms).sample()

            # accept if in prior
            is_within_prior = torch.isfinite(self.prior.log_prob(parms_perturbed))
            num_accepted += is_within_prior.sum().item()

            if num_accepted > 0:
                parameters.append(parms_perturbed[is_within_prior])

        return torch.cat(parameters)

    def get_kernel_variance(
        self,
        particles: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int = 1000,
        kernel_variance_scale: Optional[float] = 1.0,
    ) -> torch.Tensor:

        # get weighted samples
        samples = self.sample_from_population_with_weights(
            particles, weights, num_samples=num_samples
        )

        if self.kernel == "gaussian":
            mean = torch.mean(samples, dim=0).unsqueeze(0)

            # take double the weighted sample cov as proposed in Beaumont 2009
            population_cov = torch.matmul(samples.T, samples) / (
                num_samples - 1
            ) - torch.matmul(mean.T, mean)

            return kernel_variance_scale * population_cov

        elif self.kernel == "uniform":
            # Variance spans the range of parameters for every dimension.
            return kernel_variance_scale * torch.tensor(
                [max(theta_column) - min(theta_column) for theta_column in samples.T]
            )
        else:
            raise ValueError(f"Kernel, '{self.kernel}' not supported.")

    def get_new_kernel(self, thetas: torch.Tensor) -> Distribution:
        """Return new kernel distribution for a given set of paramters."""

        if self.kernel == "gaussian":
            return MultivariateNormal(
                loc=thetas, covariance_matrix=self.kernel_variance
            )

        elif self.kernel == "uniform":
            low = thetas - self.kernel_variance
            high = thetas + self.kernel_variance
            # Move batch shape to event shape to get Uniform that is multivariate in
            # parameter dimension.
            return Uniform(low=low, high=high).to_event(1)
        else:
            raise ValueError(f"Kernel, '{self.kernel}' not supported.")
