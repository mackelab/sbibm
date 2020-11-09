from __future__ import annotations

import torch


def mean_squared_distance_over_batch(
    observation: torch.Tensor, simulated_data: torch.Tensor
) -> torch.Tensor:
    """Take mean squared distance over batch dimension

    Args:
        observation: observed data, could be 1D
        simulated_data: batch of simulated data, has batch dimension

    Returns:
        Torch tensor with batch of distances
    """
    assert simulated_data.ndim == 2, "simulated data needs batch dimension"

    return torch.mean((observation - simulated_data) ** 2, dim=-1)


def l2_distance_over_batch(
    observation: torch.Tensor, simulated_data: torch.Tensor
) -> torch.Tensor:
    """Take L2 distance over batch dimension

    Args:
        observation: observed data, could be 1D
        simulated_data: batch of simulated data, has batch dimension

    Returns:
        Torch tensor with batch of distances
    """
    assert (
        simulated_data.ndim == 2
    ), f"Simulated data needs batch dimension, is {simulated_data.shape}."

    return torch.norm((observation - simulated_data), dim=-1)


def l1_distance_over_batch(
    observation: torch.Tensor, simulated_data: torch.Tensor
) -> torch.Tensor:
    """Take mean absolute distance over batch dimension

    Args:
        observation: observed data, could be 1D
        simulated_data: batch of simulated data, has batch dimension

    Returns:
        Torch tensor with batch of distances
    """
    assert (
        simulated_data.ndim == 2
    ), f"Simulated data needs batch dimension, is {simulated_data.shape}."

    return torch.mean(abs(observation - simulated_data), dim=-1)
