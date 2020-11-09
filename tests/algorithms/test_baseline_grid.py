import pytest
import torch

import sbibm
from sbibm.algorithms.pytorch.baseline_grid import run as run_grid
from sbibm.metrics.c2st import c2st


@pytest.mark.parametrize(
    "task_name",
    [(task_name) for task_name in ["gaussian_linear_uniform"]],
)
def test_grid(
    task_name, num_observation=1, num_samples=1000, num_simulations=100_000_000
):
    task = sbibm.get_task(task_name)

    samples = run_grid(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        num_simulations=num_simulations,
    )

    reference_samples = task.get_reference_posterior_samples(
        num_observation=num_observation
    )

    acc = c2st(samples, reference_samples[:num_samples, :])

    assert torch.abs(acc - 0.5) < 0.025
