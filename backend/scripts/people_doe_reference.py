"""Explicit global SSIM convention for the photo dataset, independent of app metrics."""

import numpy as np
import torch
from doe.physics import gray_to_png, phase_to_png, png_to_gray, png_to_phase


def normalize_amplitude(amplitude):
    return amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


class PhotoObjective:
    def __init__(self, target):
        self.target = target
        self.centered = target - target.mean(dim=(-2, -1), keepdim=True)
        self.energy = self.centered.square().sum(dim=(-2, -1))

    def scores(self, pixels):
        centered = pixels - pixels.mean(dim=(-2, -1), keepdim=True)
        pcc = (centered * self.centered).sum(dim=(-2, -1)) / (
            centered.square().sum(dim=(-2, -1)) * self.energy
        ).clamp_min(1e-12).sqrt()
        pcc = torch.where(pixels.amax((-2, -1)) > pixels.amin((-2, -1)), pcc, 0)
        x, y = pixels, self.target
        mx, my = [v.mean(dim=(-2, -1), keepdim=True) for v in (x, y)]
        xc, yc = x - mx, y - my
        vx, vy = [v.square().mean(dim=(-2, -1), keepdim=True) for v in (xc, yc)]
        cov = (xc * yc).mean(dim=(-2, -1), keepdim=True)
        score = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
            (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
        )
        return pcc, score.squeeze(-1).squeeze(-1)


def global_metrics(reconstruction, target, distances):
    actual, expected = [
        image.detach().cpu().numpy().astype(np.float64) for image in (reconstruction, target)
    ]
    planes = []
    for z, x, y in zip(distances, actual, expected, strict=True):
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


def export_photo_phase(phase, target, propagator):
    phase_png = phase_to_png(phase)
    decoded = png_to_phase(phase_png, propagator.size).to(propagator.input_axis_m.device)
    amplitude = normalize_amplitude(propagator.amplitude(decoded)).cpu()
    images = tuple(gray_to_png(plane) for plane in amplitude)
    pixels = torch.stack([png_to_gray(png, propagator.size) for png in images])
    return phase_png, images, global_metrics(pixels, target, propagator.distances_m)
