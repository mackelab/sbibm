from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import torch

from sbibm.third_party.igms.main import ExpQuadKernel as tp_ExpQuadKernel
from sbibm.third_party.igms.main import mmd2_unbiased as tp_mmd2_unbiased
from sbibm.third_party.torch_two_sample.main import MMDStatistic as tp_MMDStatistic
from sbibm.utils.torch import get_default_device

log = logging.getLogger(__name__)


def mmd_gaussian_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    implementation: str = "tp_sutherland",
    z_score: bool = True,
    bandwidth: str = "X",
) -> torch.Tensor:
    """Estimate MMD^2 statistic with Gaussian kernel


    Currently different implementations are available, in order to validate accuracy and compare speeds.

    The widely used median heuristic for bandwidth-selection of the Gaussian kernel is used.

    Note that there are different options to choose the heuristic:

        1. only use median distance between samples in X (Papamakarios et al., Greenberg et al.)
        2. use median distances between samples in Z = [X,Y] (Gretton 2012)
        3. use median distances between samples of X and Y

    We use 1. here, in line with Papamakarios and APT. Gretton 2012 uses option 2.
    TODO: Decide on scheme.

    Alternatives: Cross-validation, lambda scheme proposed in On the Decreasing Power of Kernel and Distance Based Nonparametric Hypothesis Tests in High Dimensions, mixtures of kernels.
    """
    if torch.isnan(X).any() or torch.isnan(Y).any():
        return torch.tensor(float("nan"))

    tic = time.time()  # noqa
    # log.info(f"MMD implementation: {implementation}")

    if z_score:
        X_mean = torch.mean(X, axis=0)
        X_std = torch.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    n_1 = X.shape[0]
    n_2 = Y.shape[0]

    # Bandwidth
    if bandwidth == "X":
        sigma_tensor = torch.median(torch.pdist(X))
    elif bandwidth == "XY":
        sigma_tensor = torch.median(torch.pdist(torch.cat([X, Y])))
    else:
        raise NotImplementedError

    if False:
        # Computes alternatives heuristics in numpy and torch
        xs = X.cpu().numpy()
        ys = Y.cpu().numpy()
        xx_sq_dists = np.sum(
            np.array([x1 - x2 for x1, x2 in itertools.combinations(xs, 2)]) ** 2, axis=1
        )
        yy_sq_dists = np.sum(
            np.array([y1 - y2 for y1, y2 in itertools.combinations(ys, 2)]) ** 2, axis=1
        )
        xy_sq_dists = np.sum(
            np.array([x1 - y2 for x1, y2 in itertools.product(xs, ys)]) ** 2, axis=1
        )
        scale_1 = np.median(np.sqrt(xx_sq_dists))
        scale_2 = np.median(
            np.sqrt(np.concatenate([xx_sq_dists, yy_sq_dists, xy_sq_dists]))
        )
        scale_3 = np.median(np.sqrt(np.concatenate([xy_sq_dists])))

        scale_1_torch = torch.median(torch.pdist(X))
        scale_2_torch = torch.median(torch.pdist(torch.cat([X, Y])))
        scale_3_torch = tts_pdist(X, Y).median()

    # Compute MMD
    if implementation == "tp_sutherland":
        K = tp_ExpQuadKernel(X, Y, sigma=sigma_tensor)
        statistic = tp_mmd2_unbiased(K)

    elif implementation == "tp_djolonga":
        alpha = 1 / (2 * sigma_tensor ** 2)
        test = tp_MMDStatistic(n_1, n_2)
        statistic = test(X, Y, [alpha])

    else:
        raise NotImplementedError

    toc = time.time()  # noqa
    # log.info(f"Took {toc-tic:.3f}sec")

    return statistic
