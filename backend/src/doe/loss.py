"""Differentiable reconstruction objectives shared by learning and refinement."""

from torch import Tensor


def normalize_amplitude(amplitude: Tensor) -> Tensor:
    """Scale each plane independently by its peak, keeping zero planes zero."""
    return amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def global_ssim(x: Tensor, y: Tensor) -> Tensor:
    """SSIM per image using population moments over both spatial dimensions."""
    mean_x = x.mean(dim=(-2, -1), keepdim=True)
    mean_y = y.mean(dim=(-2, -1), keepdim=True)
    xc, yc = x - mean_x, y - mean_y
    var_x = xc.square().mean(dim=(-2, -1), keepdim=True)
    var_y = yc.square().mean(dim=(-2, -1), keepdim=True)
    covariance = (xc * yc).mean(dim=(-2, -1), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    )
    return score.squeeze(-1).squeeze(-1)


def reconstruction_loss(
    amplitude: Tensor,
    target: Tensor,
    *,
    pcc_weight: float = 0.4,
    ssim_weight: float = 0.2,
    worst_pcc_weight: float = 0.0,
) -> Tensor:
    """Combine amplitude MSE, PCC loss and global SSIM loss across planes."""
    if amplitude.shape != target.shape:
        raise ValueError("Amplitude and target shapes must match")
    x = normalize_amplitude(amplitude).flatten(0, -3).unsqueeze(1)
    y = target.flatten(0, -3).unsqueeze(1)
    xc = x - x.mean(dim=(-2, -1), keepdim=True)
    yc = y - y.mean(dim=(-2, -1), keepdim=True)
    pcc = (xc * yc).sum(dim=(-2, -1)) / (
        xc.square().sum(dim=(-2, -1)) * yc.square().sum(dim=(-2, -1))
    ).clamp_min(1e-12).sqrt()
    ssim = global_ssim(x, y)
    loss = (x - y).square().mean() + pcc_weight * (1 - pcc.mean()) + ssim_weight * (1 - ssim.mean())
    if worst_pcc_weight:
        per_sample_worst = pcc.reshape(target.shape[:-2]).amin(dim=-1)
        loss = loss + worst_pcc_weight * (1 - per_sample_worst.mean())
    return loss
