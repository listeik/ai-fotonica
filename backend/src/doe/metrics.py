"""Common image metrics for the exported phase and reconstruction PNGs."""

import numpy as np
from torch import Tensor

from doe.physics import DISTANCES_M


def evaluate(
    reconstruction: Tensor, target: Tensor, distances_m: tuple[float, ...] = DISTANCES_M
) -> dict:
    if (
        reconstruction.shape != target.shape
        or reconstruction.ndim != 3
        or reconstruction.shape[0] != len(distances_m)
    ):
        raise ValueError("Expected matching (planes, height, width) tensors")
    actual, expected = [
        image.detach().cpu().numpy().astype(np.float64) for image in (reconstruction, target)
    ]
    planes = []
    for z, x, y in zip(distances_m, actual, expected, strict=True):
        mean_x, mean_y = x.mean(), y.mean()
        xc, yc = (x - mean_x).ravel(), (y - mean_y).ravel()
        denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
        pcc = np.clip(np.sum(xc * yc) / denom, -1, 1) if np.ptp(x) and np.ptp(y) and denom else 0.0
        var_x, var_y = np.var(x), np.var(y)
        cov_xy = np.mean(xc * yc)
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / (
            (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
        )
        planes.append({"distance_mm": z * 1e3, "pcc": float(pcc), "ssim": float(ssim)})
    return {
        "planes": planes,
        **{f"mean_{key}": float(np.mean([p[key] for p in planes])) for key in ("pcc", "ssim")},
        **{f"worst_{key}": min(p[key] for p in planes) for key in ("pcc", "ssim")},
    }
