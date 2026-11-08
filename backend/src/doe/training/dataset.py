"""Deterministic 8-bit procedural targets for four-plane synthesis."""

from typing import Literal

import numpy as np
import torch
from torch import Tensor

Split = Literal["train", "val", "test", "demo"]
SIZE = 128
DEMO_NAMES = ("Геометрия", "Цифры и символы", "Контуры", "Градиенты")
_SPLITS = {"train": 0, "val": 1, "test": 2, "demo": 3}
_GRID = np.linspace(-1.0, 1.0, SIZE, dtype=np.float32)
_X, _Y = np.meshgrid(_GRID, _GRID)
_PIXEL = 2.0 / (SIZE - 1)

# Segment centers and half-width/half-height in glyph coordinates.
_SEGMENTS = [
    (0.0, -0.8, 0.44, 0.12),  # top
    (0.47, -0.41, 0.12, 0.32),  # upper right
    (0.47, 0.41, 0.12, 0.32),  # lower right
    (0.0, 0.8, 0.44, 0.12),  # bottom
    (-0.47, 0.41, 0.12, 0.32),  # lower left
    (-0.47, -0.41, 0.12, 0.32),  # upper left
    (0.0, 0.0, 0.44, 0.12),  # center
]
_GLYPHS = (
    "012345",
    "12",
    "01346",
    "01236",
    "1256",
    "02356",
    "023456",
    "012",
    "0123456",
    "012356",
    "012456",
    "0345",
    "03456",
    "0456",
    "12456",
    "01456",
    "12345",  # 0–9, A, C, E, F, H, P, U
)


def _grid(rng: np.random.Generator, *, glyph: bool) -> tuple[np.ndarray, np.ndarray, float]:
    center_x, center_y = rng.uniform(-0.33, 0.33, size=2)
    scale_x, scale_y = rng.uniform(0.32, 0.58, size=2)
    if glyph:
        scale_x, scale_y = rng.uniform(0.49, 0.75, size=2)
    angle = rng.uniform(-0.40, 0.40) if glyph else rng.uniform(-np.pi, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    x, y = _X - center_x, _Y - center_y
    return (c * x + s * y) / scale_x, (-s * x + c * y) / scale_y, min(scale_x, scale_y)


def _shape(x: np.ndarray, y: np.ndarray, kind: int) -> np.ndarray:
    """Signed level sets, with negative values inside each primitive."""
    if kind == 0:  # disk or ellipse
        return np.hypot(x, y) - 0.8
    if kind == 1:  # rectangle
        return np.maximum(np.abs(x), np.abs(y)) - 0.8
    if kind == 2:  # diamond
        return np.abs(x) + np.abs(y) - 1.02
    if kind == 3:  # triangle
        return np.maximum(0.5 * y + np.abs(x), -y - 0.2) - 0.44
    if kind == 4:  # cross
        major, minor = np.maximum(np.abs(x), np.abs(y)), np.minimum(np.abs(x), np.abs(y))
        return np.maximum(major - 0.94, minor - 0.28)
    return np.abs(np.hypot(x, y) - 0.65) - 0.19  # annulus


def _glyph(x: np.ndarray, y: np.ndarray, segments: str) -> np.ndarray:
    distances = []
    for digit in segments:
        cx, cy, sx, sy = _SEGMENTS[int(digit)]
        distances.append(np.maximum(np.abs(x - cx) - sx, np.abs(y - cy) - sy))
    return np.minimum.reduce(distances)


def _plane(rng: np.random.Generator, family: int) -> np.ndarray:
    is_glyph = family == 1
    x, y, scale = _grid(rng, glyph=is_glyph)
    if is_glyph:
        distances = _glyph(x, y, _GLYPHS[int(rng.integers(len(_GLYPHS)))])
    else:
        distances = _shape(x, y, int(rng.integers(6)))
    if family == 0:  # binary geometry
        return (distances <= 0).astype(np.uint8) * 255

    if family == 2:  # sparse contours
        distances = np.abs(distances) - rng.uniform(0.045, 0.13)
    coverage = np.clip(0.5 - distances * scale / _PIXEL, 0, 1)
    if family == 3:  # gray ramps and sinusoidal texture
        angle = rng.uniform(0, 2 * np.pi)
        coordinate = np.cos(angle) * x + np.sin(angle) * y
        frequency = rng.uniform(2.0, 6.5)
        offset = rng.uniform(-np.pi, np.pi)
        ramp = np.clip(0.62 + 0.30 * coordinate, 0.28, 0.95)
        coverage *= ramp * (0.82 + 0.18 * np.cos(frequency * coordinate + offset))
    return np.rint(255 * coverage).astype(np.uint8)


def synthetic_targets(index: int, *, split: Split = "test", seed: int = 2026) -> Tensor:
    """Return reproducible [4, 128, 128] float32 grayscale levels in [0, 1].

    The split is part of the random stream ID. Every consecutive group of four
    indices covers geometry, glyphs, contours and textured grayscale targets.
    """
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if index < 0 or seed < 0:
        raise ValueError("index and seed must be non-negative")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _SPLITS[split], index]))
    planes = [_plane(rng, index % 4) for _ in range(4)]
    return torch.from_numpy(np.stack(planes)).to(torch.float32).div_(255)
