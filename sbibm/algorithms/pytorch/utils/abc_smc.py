from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch
from pyro.distributions import Uniform
from torch import distributions as dist
from tqdm import tqdm

from .distances import (
    l1_distance_over_batch,
    l2_distance_over_batch,
    mean_squared_distance_over_batch,
)


class ABCSMC:
    """
    Performs ABC SMC as described in Tina Toni's PhD thesis, 2010.
    """

    def __init__(
        self,
        observation: torch.Tensor,
        model: Callable,
        prior: dist.Distribution,
        n_initial_round: int,
        population_size: int,
        simulation_budget: int,
        distance: Optional[str] = "mse",
        verbose: Optional[bool] = True,
        eps_decay_base: Optional[float] = 0.9,
        kernel: Optional[str] = "gaussian",
        kernel_variance_scale: Optional[float] = 0.5,
        ess_min: Optional[float] = None,
        eps0: Optional[float] = None,
        batch_size: Optional[int] = 100,
    ):

        self.model = model
        self.observation = observation
        assert (
            self.observation.dim() == 2
        ), f"observation must be 2D, first D \
            is batch dimension, is {observation.dim}"
        assert (
            population_size <= n_initial_round
        ), "n initial round should be larger than population size"
        assert kernel in ("gaussian", "uniform"), f"Kernel '{kernel}' not supported."

        self.prior = prior
        self.dim_params = prior.sample().size()[0]

        self.eps_decay_base = eps_decay_base
        self.verbose = verbose

        self.population_size = population_size
        self.n_initial_round = n_initial_round
        self.quantile = population_size / n_initial_round
        self.simulation_budget = simulation_budget
        self.batch_size = batch_size

        self.kernel = kernel
        self.population_covariance = None
        self.n_simulations = 0
        self.ess_min = ess_min
        self.kernel_variance_scale = kernel_variance_scale

        if distance == "mse":
            self.distance_fun = mean_squared_distance_over_batch
        elif distance == "l1":
            self.distance_fun = l1_distance_over_batch
        elif distance == "l2":
            self.distance_fun = l2_distance_over_batch
        else:
            raise ValueError(f"Distance type '{distance}' is not supported.")

        self.log = logging.getLogger(__name__)

    def simulate_and_check_distance(
        self, parameters: torch.Tensor, eps: float
    ) -> Tuple(torch.Tensor, torch.Tensor):
        """Simulate a batch of parameters and calculate distance to observed data. 
        
        Arguments:
            parameters {torch.Tensor} -- batch of parameters
            eps {float} -- rejection distance threshold
        
        Returns:
            [(torch.Tensor, torch.Tensor)] -- mask of accepted samples, simulated data
        """
        assert parameters.ndim == 2, "parameters need batch dimension"

        # simulate batch of parameters
        simulated_data = self.model(parameters)

        # check distance, count accepted sims
        acceptance_mask = self.distance_fun(self.observation, simulated_data) <= eps

        # update budget
        self.n_simulations += parameters.shape[0]
        self.pbar.update(parameters.shape[0])

        return acceptance_mask, simulated_data

    def run_abcsmc(self):

        self.log.info(
            f"Running with popsize {self.population_size}, initial round size {self.n_initial_round}, quantile q={self.quantile}"
        )

        # setup global pbar
        self.pbar = tqdm(total=self.simulation_budget, disable=not self.verbose)

        with self.pbar as pbar:

            # run initial round for determining epsilon
            thetas, weights, eps0 = self.run_initial_population()
            self.eps = eps0
            self.log.info(f"Initial round quantile-based eps0={eps0:.2f}")

            n_accepted = self.population_size
            self.n_simulations += self.n_initial_round

            self.log.info(
                f"ABCSMC initial population: {self.population_size} / {self.n_simulations} accepted"
            )

            # loop over populations
            population_idx = 1
            while self.n_simulations < self.simulation_budget:

                # epsilon decay
                self.eps *= self.eps_decay_base
                # update population variance for perturbation kernel
                self.population_variance = self._get_population_variance(thetas)

                thetas_new, weights_new = self.sample_new_population(thetas, weights)

                self.log.info(
                    f"population {population_idx} done, budget: {100 * self.n_simulations / self.simulation_budget:.2f} %"
                )

                assert torch.isclose(
                    torch.sum(weights_new), torch.tensor([1.0])
                ), f"weights must sum to 1, sum={torch.sum(weights_new)}"

                if self.ess_min is not None:
                    ess = (
                        1 / torch.sum(weights_new ** 2, dim=0)
                    ) / self.population_size
                    if ess < self.ess_min:
                        self.log.info(
                            f"ESS={ess:.2f} too low, resampling pop {population_idx}..."
                        )

                        thetas_new = self.sample_from_trace(
                            thetas_new, weights_new, num_samples=self.population_size
                        )
                        weights_new = (
                            torch.ones(self.population_size) / self.population_size
                        )

                # update weights and thetas
                thetas = thetas_new
                weights = weights_new
                population_idx += 1

        return thetas, weights

    def sample_new_population(self, thetas, weights):
        n_accepted = 0
        thetas_new = torch.zeros_like(thetas)
        weights_new = torch.zeros_like(weights)

        # loop over simulations
        while n_accepted < self.population_size:

            # get new parameters from trace
            # make sure batch size doesn't exceed budged or population
            n_batch = min(
                self.batch_size,
                self.population_size - n_accepted,
                self.simulation_budget - self.n_simulations,
            )
            theta_candidates = self.sample_from_trace_and_perturb(
                thetas, weights, num_samples=n_batch,
            )

            # simulate with current eps
            accepted_mask, _ = self.simulate_and_check_distance(
                theta_candidates, self.eps
            )
            n_accepted_batch = accepted_mask.sum().item()

            # compare and reject: if at least 1 one was accepted
            if n_accepted_batch > 0:
                # get accepted thetas
                thetas_accepted = theta_candidates[accepted_mask]
                # calculate batch of weights
                weights_accepted = torch.tensor(
                    [
                        self.calculate_weight(th, weights, thetas)
                        for th in thetas_accepted
                    ]
                )
                # update thetas and weights
                thetas_new[
                    n_accepted : (n_accepted + n_accepted_batch)
                ] = thetas_accepted
                weights_new[
                    n_accepted : (n_accepted + n_accepted_batch)
                ] = weights_accepted
                n_accepted += n_accepted_batch

            # if budget is reached, break, return previous round
            if self.n_simulations >= self.simulation_budget:
                thetas_new = thetas
                weights_new = weights

                self.log.info(
                    f"Simulation budget depleted, returning samples of previous population"
                )
                break

        # normalize the weights
        weights_new /= torch.sum(weights_new)

        return thetas_new, weights_new

    def sample_from_trace_and_perturb(
        self, thetas: torch.Tensor, weights: torch.Tensor, num_samples: int = 1
    ):
        """Sample and perturb batch of new parameters from trace. 

        Reject sampled and perturbed parameters outside of prior. 
        
        Arguments:
            trace {torch.Tensor} -- [description]
            weights {torch.Tensor} -- [description]
        
        Keyword Arguments:
            num_samples {int} -- [description] (default: {1})
        """

        n_accepted = 0
        parameters = []
        while n_accepted < num_samples:
            parms = self.sample_from_trace(thetas, weights, num_samples - n_accepted)
            parms_perturbed = self._get_new_kernel(parms).sample()

            # accept if in prior
            in_prior_mask = torch.isfinite(
                self.prior.log_prob(
                    parms_perturbed
                )  # sum across second dim to cover Uniform prior bug
            )
            n_accepted += in_prior_mask.sum().item()

            if in_prior_mask.sum().item() > 0:
                parameters.append(parms_perturbed[in_prior_mask])

        return torch.cat(parameters)

    def calculate_weight(self, theta, weights_pre, thetas_pre):
        # construct multidimensional kernel with thetas as batch dim
        k = self._get_new_kernel(theta)
        # evaluate new theta under kernel
        log_prob_kernel = k.log_prob(thetas_pre)
        log_prob_prior = self.prior.log_prob(theta)

        # weight with previous weights, multiply in log space
        log_weighted_sum = torch.logsumexp(
            log_prob_kernel + torch.log(weights_pre), dim=0
        )

        # calculate log of formula (ignoring b for repetitions)
        log_weight = log_prob_prior - log_weighted_sum

        weight = torch.exp(log_weight)
        assert torch.isfinite(weight), f"weight not finite: {weight}"

        return weight

    def run_initial_population(self):

        # sample, simulate, take distance for initial population
        ths = self.prior.sample((self.n_initial_round,))
        distances = self.distance_fun(self.observation, self.model(ths))

        sortidx = torch.argsort(distances)
        # the quantile is chosen such that it covers the population size
        # quantile idx == population_size
        quantile_idx = int(self.quantile * self.n_initial_round)
        eps0 = self.select_initial_epsilon(distances[sortidx], quantile_idx)

        # recycle pilot run simulations
        thetas = ths[sortidx][:quantile_idx]
        weights = torch.ones(thetas.shape[0]) / thetas.shape[0]

        return thetas, weights, eps0

    @staticmethod
    def sample_from_trace(trace, weights, num_samples=1):
        # define multinomial with weights as probs
        multi = dist.Multinomial(probs=weights)
        # sample num samples, with replacement
        samples = multi.sample(sample_shape=(num_samples,))
        # get indices of success trials
        indices = torch.where(samples)[1]
        # return those indices from trace
        return trace[indices]

    def select_initial_epsilon(self, sorted_distances, quantile_idx):

        eps0 = sorted_distances[quantile_idx - 1]

        if eps0 == 0:
            self.log.warning("Initial epsilon is zero, adding small eps")
            eps0 += 1e-10
        if not torch.isfinite(eps0):
            self.log.warning("Initial epsilon is inf, choosing largest finite one.")
            quantile_distances = sorted_distances[:quantile_idx]
            eps0 = quantile_distances[torch.isfinite(quantile_distances)][-1]

        return eps0

    def _get_population_variance(self, thetas: torch.Tensor) -> torch.Tensor:
        # from Toni PhD Thesis pp 30 equ 3.11
        # Get parameter ranges component wise.
        return torch.max(thetas, dim=0).values - torch.min(thetas, dim=0).values

    def _get_new_kernel(self, theta: torch.Tensor) -> Distribution:
        """Return new kernel centered on given thetas and scaled by population
        variance."""

        if self.kernel == "gaussian":
            return dist.MultivariateNormal(
                loc=theta,
                covariance_matrix=self.kernel_variance_scale
                * torch.diag(self.population_variance),
            )
        elif self.kernel == "uniform":
            low = theta - self.kernel_variance_scale * self.population_variance
            high = theta + self.kernel_variance_scale * self.population_variance
            # Move batch shape to event shape to get Uniform that is multivariate in
            # parameter dim.
            return Uniform(low=low, high=high).to_event(1)
