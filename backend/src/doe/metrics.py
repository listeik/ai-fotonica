"""Common image metrics for the exported phase and reconstruction PNGs."""

import numpy as np
from skimage.metrics import structural_similarity
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
        xc, yc = (x - x.mean()).ravel(), (y - y.mean()).ravel()
        denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
        pcc = np.clip(np.sum(xc * yc) / denom, -1, 1) if np.ptp(x) and np.ptp(y) and denom else 0.0
        window = min(7, *x.shape)
        window -= 1 - window % 2
        ssim = structural_similarity(x, y, data_range=1.0, win_size=window)
        planes.append({"distance_mm": z * 1e3, "pcc": float(pcc), "ssim": float(ssim)})
    return {
        "planes": planes,
        **{f"mean_{key}": float(np.mean([p[key] for p in planes])) for key in ("pcc", "ssim")},
        **{f"worst_{key}": min(p[key] for p in planes) for key in ("pcc", "ssim")},
    }
