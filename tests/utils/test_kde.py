import torch

import sbibm
from sbibm.utils.kde import get_kde
from sbi.utils.plot import pairplot


def test_kde_transform_log_prob_visual(plt):
    task = sbibm.get_task("gaussian_linear_uniform", dim=1)
    observation = torch.tensor([0.9]).reshape(1, -1)
    posterior_samples = task._sample_reference_posterior(
        observation=observation, num_samples=1000
    )

    params = torch.linspace(0.0, +0.99, 50).reshape(-1, 1)

    kde = get_kde(posterior_samples)
    log_probs = kde.log_prob(params)
    plt.plot(params, torch.exp(log_probs), "-b")

    transform = task._get_transforms()["parameters"]
    kde_2 = get_kde(posterior_samples, transform=transform)
    log_probs_2 = kde_2.log_prob(params)
    plt.plot(params, torch.exp(log_probs_2), "-r")


def test_kde_transform_samples_visual(plt):
    task = sbibm.get_task("slcp")
    posterior_samples = task.get_reference_posterior_samples(num_observation=1)[
        :1000, :
    ]

    transform = task._get_transforms()["parameters"]
    kde = get_kde(posterior_samples, transform=transform)

    kde_samples = kde.sample((1000,))

    pairplot([posterior_samples.numpy(), kde_samples.numpy()])
