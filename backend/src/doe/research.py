"""Image descriptors and diagnostics shared by the API and smart experiments."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from doe.loss import normalize_amplitude
from doe.models import load_model
from doe.physics import FresnelPropagator, gray_to_png, png_to_phase
from doe.synthesis import synthesize
from doe.training.dataset import synthetic_targets

REFERENCE_SETS = 32
REFERENCE_SEED = 2026
HIGH_FREQUENCY = 0.20
EDGE_THRESHOLD = 0.15
SUPPORT_THRESHOLD = 2 / 255
POWER_DB_RANGE = (-50, 30)
CHECKPOINTS = {
    "unet": ("Projection U-Net", "doe-unet.pt"),
    "fourier": ("Residual Fourier U-Net", "doe-residual-fourier.pt"),
    "carrier": ("Carrier U-Net", "doe-carrier-physics-teacher-v1.pt"),
}


def descriptors(image: np.ndarray) -> dict[str, float]:
    bins = np.bincount(np.rint(image * 255).astype(np.uint8).ravel(), minlength=256)
    probabilities = bins[bins > 0] / image.size
    padded = np.pad(image, 1, mode="edge")
    # Apply the separable 3×3 Sobel kernels with reflected one-pixel borders.
    horizontal = padded[:, 2:] - padded[:, :-2]
    vertical = padded[2:, :] - padded[:-2, :]
    gradient_x = (horizontal[:-2] + 2 * horizontal[1:-1] + horizontal[2:]) / 4
    gradient_y = (vertical[:, :-2] + 2 * vertical[:, 1:-1] + vertical[:, 2:]) / 4
    gradient = np.hypot(gradient_x, gradient_y)
    frequencies = np.hypot(
        np.fft.fftfreq(image.shape[0])[:, None], np.fft.rfftfreq(image.shape[1])[None, :]
    )
    energy = np.abs(np.fft.rfft2(image - image.mean())) ** 2
    # Restore the conjugate bins omitted by rFFT, excluding DC and Nyquist columns.
    energy[:, 1:-1] *= 2
    energy[0, 0] = 0
    total = energy.sum()
    return {
        "contrast": float(image.std()),
        "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        "edge_fraction": float(np.mean(gradient >= EDGE_THRESHOLD)),
        "spectral_centroid": float((energy * frequencies).sum() / total) if total else 0.0,
        "high_frequency_fraction": float(energy[frequencies >= HIGH_FREQUENCY].sum() / total)
        if total
        else 0.0,
    }


@lru_cache(maxsize=1)
def reference_bands() -> dict:
    # Quantify normal variation between four-plane sets, using their per-set means.
    sets = [
        [
            descriptors(image)
            for image in synthetic_targets(index, split="train", seed=REFERENCE_SEED).numpy()
        ]
        for index in range(REFERENCE_SETS)
    ]
    return {
        name: np.quantile(
            [np.mean([plane[name] for plane in planes]) for planes in sets], [0.1, 0.5, 0.9]
        ).tolist()
        for name in sets[0][0]
    }


def analyze_targets(targets: Tensor) -> dict:
    return {
        "planes": [descriptors(plane) for plane in targets.numpy()],
        "reference": {"sets": REFERENCE_SETS, "seed": REFERENCE_SEED, "bands": reference_bands()},
        "edge_threshold": EDGE_THRESHOLD,
        "high_frequency_cutoff": HIGH_FREQUENCY,
    }


def research_options(root: Path) -> list[dict]:
    return [
        {"id": identifier, "name": name}
        for identifier, (name, filename) in CHECKPOINTS.items()
        if (root / filename).is_file()
    ]


def compare_checkpoint(identifier: str, root: Path, targets: Tensor, propagator: FresnelPropagator):
    _, filename = CHECKPOINTS[identifier]
    model, version = load_model(root / filename, propagator.input_axis_m.device)
    return synthesize(targets, "neural", propagator, model=model), {
        "id": Path(filename).stem,
        "version": version,
        "architecture": model.architecture,
    }


@torch.no_grad()
def diagnose_phase(phase_png: bytes, targets: Tensor, propagator: FresnelPropagator) -> dict:
    phase = png_to_phase(phase_png, propagator.size).to(propagator.input_axis_m.device)
    amplitude = propagator.amplitude(phase)
    # The error uses the same eight-bit values exported and scored during synthesis.
    absolute_error = (
        normalize_amplitude(amplitude).cpu().mul(255).round().div(255) - targets
    ).abs()
    intensity = amplitude.cpu().square()
    support = targets > SUPPORT_THRESHOLD
    low, high = POWER_DB_RANGE
    maps = (10 * intensity.clamp_min(10 ** (low / 10)).log10() - low) / (high - low)
    return {
        "power_db_range": POWER_DB_RANGE,
        "support_threshold": SUPPORT_THRESHOLD,
        "planes": [
            {
                "distance_mm": round(distance * 1000, 4),
                "output_pitch_um": round(pitch * 1e6, 4),
                "mae": float(error.mean()),
                "power_in_target": float((power[mask].sum() / power.sum()).clamp(0, 1)),
                "error_png": gray_to_png(error),
                "intensity_png": gray_to_png(power_map),
            }
            for distance, pitch, error, power, mask, power_map in zip(
                propagator.distances_m,
                propagator.output_pitch_m.tolist(),
                absolute_error,
                intensity,
                support,
                maps,
                strict=True,
            )
        ],
    }
