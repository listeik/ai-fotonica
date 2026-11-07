"""Differentiable reconstruction objectives shared by learning and refinement."""

import torch
import torch.nn.functional as F
from torch import Tensor


def normalize_amplitude(amplitude: Tensor) -> Tensor:
    """Scale each plane independently by its peak, keeping zero planes zero."""
    return amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def reconstruction_loss(amplitude: Tensor, target: Tensor) -> Tensor:
    """Combine amplitude MSE, PCC loss and local SSIM loss across planes."""
    if amplitude.shape != target.shape:
        raise ValueError("Amplitude and target shapes must match")
    x = normalize_amplitude(amplitude).flatten(0, -3).unsqueeze(1)
    y = target.flatten(0, -3).unsqueeze(1)
    xc = x - x.mean(dim=(-2, -1), keepdim=True)
    yc = y - y.mean(dim=(-2, -1), keepdim=True)
    pcc = (xc * yc).sum(dim=(-2, -1)) / (
        xc.square().sum(dim=(-2, -1)) * yc.square().sum(dim=(-2, -1))
    ).clamp_min(1e-12).sqrt()
    window = min(7, *x.shape[-2:])
    # Subsample windows during optimization. Official SSIM is evaluated at stride 1.
    mx, my, exx, eyy, exy = F.avg_pool2d(
        torch.cat((x, y, x.square(), y.square(), x * y), dim=1), window, stride=2
    ).unbind(1)
    vx, vy = (exx - mx.square()).clamp_min(0), (eyy - my.square()).clamp_min(0)
    cov = exy - mx * my
    ssim = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    )
    return (x - y).square().mean() + 0.4 * (1 - pcc.mean()) + 0.2 * (1 - ssim.mean())
