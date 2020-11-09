from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from tqdm.auto import tqdm

from .distances import (
    l1_distance_over_batch,
    l2_distance_over_batch,
    mean_squared_distance_over_batch,
)

log = logging.getLogger(__name__)


class ABCRejection:
    def __init__(
        self,
        simulator: Callable,
        prior,
        observation: torch.Tensor,
        distance: str = "l2",
        batch_size: int = 1000,
    ):
        """Rejection ABC

        Args:
            simulator: Simulator
            prior: Prior distribution
            observation: Observed data
            distance: Distance to use
            batch_size: Batch size for simulations
        """
        self.simulator = simulator
        self.prior = prior
        self.observation = observation

        if distance == "l2":
            self.distance_fun = l2_distance_over_batch
        elif distance == "l1":
            self.distance_fun = l1_distance_over_batch
        elif distance == "mse":
            self.distance_fun = mean_squared_distance_over_batch
        else:
            raise NotImplementedError

        self.batch_size = batch_size

    def run(
        self,
        num_simulations: int,
        eps: Optional[float] = None,
        quantile: Optional[float] = None,
        num_top_samples: Optional[int] = None,
        top_method: Optional[str] = "naive",
    ) -> (torch.Tensor, torch.Tensor):
        """Runs rejection ABC

        Specify one of eps, quantile, or num_top_samples.

        Args:
            num_simulations: Number of similation
            eps: Epsilon threshold
            quantile: Quantile of total simulations to return
            num_top_samples: Number of top samples to return
        
        Returns:
            Accepted parameters and associated distances
        """
        if num_simulations < self.batch_size:
            self.batch_size = num_simulations
            log.info("Reduced batch_size to num_simulations")

        if eps is not None:
            assert quantile is None and num_top_samples is None
            parameters_accepted, distances = self.simulate_eps(num_simulations, eps)
            return parameters_accepted, distances

        if quantile is not None:
            assert num_top_samples is None
            num_top_samples = int(quantile * num_simulations)

        assert eps is None
        assert top_method in [
            "naive",
            "efficient",
        ], f"Top method '{top_method}' not defined."

        if top_method == "naive":
            parameters_accepted, distances = self.simulate_top_naive(
                num_simulations, num_top_samples
            )
        else:
            parameters_accepted, distances = self.simulate_top_efficient(
                num_simulations, num_top_samples
            )

        return parameters_accepted, distances

    def simulate_eps(self, num_simulations: int, eps: float):
        """Simulate with batches, return only accepted params and data based on epsilon
        """
        parameters = []
        distances = []
        num_accepted = 0

        for idx in tqdm(range(int(num_simulations / self.batch_size))):
            parameter_batch = self.prior(self.batch_size)
            distance_batch = self.distance_fun(
                self.observation, self.simulator(parameter_batch)
            )
            is_accepted = distance_batch <= eps
            num_accepted_batch = is_accepted.sum().item()

            if num_accepted_batch > 0:
                parameters.append(parameter_batch[is_accepted])
                distances.append(distance_batch[is_accepted])
                num_accepted += num_accepted_batch

        assert num_accepted > 0, f"No parameters accepted, eps={eps} too small"

        return torch.cat(parameters), torch.cat(distances)

    def simulate_top_efficient(self, num_simulations: int, num_top_samples: int):
        """Simulate in batches and retain num_top_samples
        """
        if num_simulations < num_top_samples:
            num_top_samples = num_simulations
            log.info("Reduced num_top_samples to num_simulations")

        dim_parameters = self.prior(1).shape[1]
        parameters_top = float("nan") * torch.ones((num_top_samples, dim_parameters))
        distances_top = float("inf") * torch.ones((num_top_samples,))

        for _ in tqdm(range(int(num_simulations / self.batch_size))):
            parameters = self.prior(self.batch_size)

            distances = self.distance_fun(self.observation, self.simulator(parameters))

            distances_sort_idx = torch.argsort(distances)
            distances_sorted = distances[distances_sort_idx]
            parameters_sorted = parameters[distances_sort_idx]

            for sdis, sparam in zip(distances_sorted, parameters_sorted):

                # Catch duplicate distances.
                if sdis in distances_top:
                    # get duplicate distances
                    idx = (sdis == distances_top).nonzero()[-1]
                    try:
                        # replace the next index in the indices of top indices.
                        distances_top[idx + 1] = sdis
                        # do the same for the corresponding parameter.
                        parameters_top[idx + 1] = sparam
                    # Catch corner case when idx+1 out of array.
                    except IndexError:
                        pass
                else:
                    try:
                        # get last index that is smaller than any in current top
                        # distances.
                        idx = (sdis < distances_top).nonzero()[-1]
                        # replace this index in the indices of top indices.
                        distances_top[idx] = sdis
                        # do the same for the corresponding parameter.
                        parameters_top[idx] = sparam
                    # IndexError because sdis is larger than any in distances_top.
                    # Because we are iterating over sorted distances and all the
                    # remaining sdis will be larger than the current one, we are
                    # done here and can break.
                    except IndexError:  # noqa
                        break

        sorted_idx = torch.argsort(distances_top)
        distances_top = distances_top[sorted_idx]
        parameters_top = parameters_top[sorted_idx]

        # NOTE: if distances are inf they will not replace the initial inf and nan
        # entries in the prelocated tensors. This is checked for here:
        if bool(torch.isnan(parameters_top).any()):
            # checking only first column is OK because entire row is NaN
            is_finite = torch.isnan(parameters_top[:, 0])
            num_finite = is_finite.sum().item()

            parameters_top = parameters_top[is_finite]
            distances_top = distances_top[is_finite]
            log.warning(
                f"Unable to collect all top {num_top_samples} samples. This can happen "
                " if distances to observed data are infinite. Returning top {num_finite} samples."
            )

        return parameters_top, distances_top

    def simulate_top_naive(self, num_simulations: int, num_top_samples: int):
        """Simulate in batches and retain num_top_samples
        """
        if num_simulations < num_top_samples:
            num_top_samples = num_simulations
            log.info("Reduced num_top_samples to num_simulations")

        all_distances = []
        all_parameters = []

        for idx in tqdm(range(int(num_simulations / self.batch_size))):
            parameters = self.prior(self.batch_size)

            all_parameters.append(parameters)
            all_distances.append(
                self.distance_fun(self.observation, self.simulator(parameters))
            )

        all_distances = torch.cat(all_distances, dim=0)
        all_parameters = torch.cat(all_parameters, dim=0)
        sort_idx = torch.argsort(all_distances)

        return (
            all_parameters[sort_idx][:num_top_samples],
            all_distances[sort_idx][:num_top_samples],
        )
