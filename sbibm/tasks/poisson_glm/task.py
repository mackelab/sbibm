from __future__ import annotations

import pickle
from os.path import join
from pathlib import Path
from typing import Callable, Optional, Tuple

import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
import numpy as np


class PoissonGLM(Task):
    def __init__(self, upper_rate_bound=100000):
        """Poisson GLM

        A Poisson GLM model for generation of synapse counts given structural features
        of rat sensory cortex encoded in a design matrix.
        
        The structural features are given by a design matrix. The dimensions of the
        design matrix are determined by the dimensionality of the parameters (number
        of structural features), and the dimensionality of the data (number of neuron-
        neuron-voxel pairs for which synapse counts are observed). Currently, the
        design matrix supports up to 11 features, and up to 100 rows. The number of
        rows can easily scaled to up to 130 million.
        
        Args: 
            upper_rate_bound: Upper bound for Poisson rates. For parameters that result
            in larger rates the simulator will return NaN data.
        """
        # Observation seeds to use when generating ground truth. These seeds where
        # selected such that the resulting observation is biologically plausible,
        # i.e., the resulting synapse counts lie in the range 0-50k
        # NOTE: These seeds are selected for the low parameter_dim case, i.e.,
        # dim_parameters=3. They are valid for different dim_data, but not for
        # differentdim_parameters.

        observation_seeds = [
            1000003,
            1000007,
            1000015,
            1000022,
            1000023,
            1000025,
            1000030,
            1000054,
            1000059,
            1000066,
        ]

        super().__init__(
            dim_parameters=3,
            dim_data=10,
            name=Path(__file__).parent.name,
            name_display="Poisson GLM",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        self.upper_rate_bound = upper_rate_bound
        self.design_matrix = self._load_structural_features()
        assert self.design_matrix.shape == torch.Size(
            (self.dim_data, self.dim_parameters)
        ), "Design matrix must have task shape."

        low, high = self._get_prior_bounds()

        assert self.dim_parameters == 3, "This is hard coded for parameter dim=3."
        self.prior_params = {
            "low": low,
            "high": high,
        }

        self.prior_dist = pdist.Uniform(
            low=self.prior_params["low"], high=self.prior_params["high"]
        ).to_event(1)

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            rate = torch.exp(self.design_matrix.mm(parameters.T)).T
            data = pyro.sample(
                "data", pdist.Poisson(rate=rate.clamp(0, self.upper_rate_bound))
            )
            return data

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _load_structural_features(self) -> torch.Tensor:
        """Load design matrix containing structural features from file.

        The design matrix containes a very small subset (N=100) of neuron-neuron-voxel
        entries from rat sensory cortex.
        For each row entry there are eleven structural features:
            pre-synaptic bouton count,
            post-synaptic target count,
            post synaptic target counts,
            post-synaptic pia_soma_distance,
            post-synaptic bb_dz_apical,
            subc_length_axon,
            subc_bifurcations_apical,
            subc_bifurcations_basal,
            subc_bifurcations_axon,
            subc_distance_dendrites_center_of_mass,
            subc_distance_primary_bifurcation,
                
        Returns:
            Design matrix with dimensions (self.dim_data, self.dim_parameters)
        """
        with open(
            join(
                Path(__file__).parent, "files/design_matrix_dso_sorted_100_by_11.pickle"
            ),
            "rb",
        ) as fh:
            design_matrix = pickle.load(fh)["design_matrix"]

        with open(
            join(
                Path(__file__).parent,
                "files/features_selection_D2_highestOverlap_seed10_dim10.npz",
            ),
            "rb",
        ) as fh:
            old_design_matrix = torch.as_tensor(
                pickle.load(fh)["features"], dtype=torch.float32
            )

        num_rows, num_features = design_matrix.shape
        assert (
            self.dim_parameters <= num_features
        ), f"The current design matrix has up to {num_features} features only."
        assert (
            self.dim_data <= num_rows
        ), f"The current design matrix has up to {num_rows} rows only."

        # Select features based on number of parameters.
        design_matrix = torch.as_tensor(
            design_matrix[:, : self.dim_parameters], dtype=torch.float32
        )

        # Select rows based on dimension in x and on features already present in old
        # design matrix.
        combined_design_matrix = [old_design_matrix]
        num_accepted_rows = 10
        row_idx = 0
        while num_accepted_rows < self.dim_data and row_idx < num_rows:

            if not design_matrix[row_idx, 0] in torch.cat(combined_design_matrix)[:, 0]:
                combined_design_matrix.append(design_matrix[row_idx, :].reshape(1, -1))
                num_accepted_rows += 1

            row_idx += 1

        return torch.cat(combined_design_matrix)

    def _get_prior_bounds(self,) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return prior bounds based on dimensionality of theta."""

        low = torch.tensor(
            [0.01, 0.01, -3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        )[: self.dim_parameters]
        high = torch.tensor([3.0, 3.0, -0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[
            : self.dim_parameters
        ]

        return low, high

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_observation: Observation number
            num_samples: Number of samples to generate
            observation: Observed data, if None, will be loaded using `num_observation`
            kwargs: Passed to run_mcmc

        Returns:
            Samples from reference posterior
        """
        assert observation is None

        from sbibm.algorithms.pyro.mcmc import run as run_mcmc
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        proposal_samples = run_mcmc(
            task=self,
            kernel="nuts",
            num_observation=num_observation,
            observation=observation,
            num_samples=10000,
            initial_params=self.get_true_parameters(num_observation=num_observation),
            num_chains=1,
            jit_compile=False,
            num_warmup=10000,
            thinning=1,
        )

        proposal_dist = get_proposal(
            task=self,
            samples=proposal_samples,
            prior_weight=0.1,
            bounded=False,
            density_estimator="flow",
            flow_model="maf",
        )

        return run_rejection(
            task=self,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            batch_size=1_000,
            num_batches_without_new_max=2_000,
            multiplier_M=1.3,
            proposal_dist=proposal_dist,
        )


if __name__ == "__main__":
    task = PoissonGLM()
    task._setup(create_reference=True)
