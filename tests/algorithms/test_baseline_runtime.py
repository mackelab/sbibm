import pytest
import torch

import sbibm
from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
from sbibm.metrics.c2st import c2st


@pytest.mark.parametrize(
    "task_name",
    [(task_name) for task_name in ["gaussian_linear"]],
)
def test_runtime(
    task_name,
    num_observation=1,
    num_samples=1000,
):
    task = sbibm.get_task(task_name)

    samples = run_runtime(
        task=task,
        num_observation=num_observation,
        num_samples=num_samples,
        num_simulations=1000,
    )

    assert len(samples) == num_samples
    assert torch.isnan(samples.sum())
