"""One-FFT scalar Fresnel diffraction and the challenge PNG formats."""

from io import BytesIO
from math import isfinite, pi

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn

SIZE = 128
WAVELENGTH_M = 532e-9
PITCH_M = 10e-6
DISTANCES_M = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)


class FresnelPropagator(nn.Module):
    """Propagate an (..., H, W) field to (..., Z, H, W), all lengths in metres."""

    def __init__(
        self,
        size: int = SIZE,
        wavelength_m: float = WAVELENGTH_M,
        pitch_m: float = PITCH_M,
        distances_m: tuple[float, ...] = DISTANCES_M,
    ) -> None:
        super().__init__()
        if (
            size < 2
            or not distances_m
            or not all(isfinite(v) and v > 0 for v in (wavelength_m, pitch_m, *distances_m))
        ):
            raise ValueError("Grid size must be ≥ 2 and all physical lengths must be positive")
        self.size = size
        self.pitch_m = pitch_m
        self.wavelength_m = wavelength_m
        self.distances_m = tuple(distances_m)
        z = torch.tensor(distances_m, dtype=torch.float64)
        axis = torch.arange(size, dtype=torch.float64) - size // 2
        output_pitch = wavelength_m * z / (size * pitch_m)
        self.register_buffer("input_axis_m", axis * pitch_m, persistent=False)
        self.register_buffer("output_pitch_m", output_pitch, persistent=False)
        self.register_buffer("output_axes_m", output_pitch[:, None] * axis, persistent=False)
        r_sq = axis.square()[:, None] + axis.square()[None, :]
        lz = (wavelength_m * z)[:, None, None]
        pre = torch.exp(1j * pi * r_sq * pitch_m**2 / lz)
        post = torch.exp(
            1j * (2 * pi * z[:, None, None] / wavelength_m)
            + 1j * pi * r_sq * output_pitch[:, None, None].square() / lz
        ) * (pitch_m**2 / (1j * lz))
        self.register_buffer("_pre", pre, persistent=False)
        self.register_buffer("_post", post, persistent=False)
        self.register_buffer("_gain", pitch_m**2 / lz, persistent=False)

    def _spectrum(self, field: Tensor) -> Tensor:
        if field.shape[-2:] != (self.size, self.size) or field.dtype not in (
            torch.complex64,
            torch.complex128,
        ):
            raise ValueError("Expected a square complex64 or complex128 field on the input grid")
        dims = (-2, -1)
        chirped = field.unsqueeze(-3) * self._pre.to(dtype=field.dtype)
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(chirped, dim=dims)), dim=dims)

    def forward(self, field: Tensor) -> Tensor:
        return self._spectrum(field) * self._post.to(dtype=field.dtype)

    def inverse_planes(self, fields: Tensor) -> Tensor:
        """Invert each plane independently on its native Fresnel grid."""
        if fields.shape[-3:] != (len(self.distances_m), self.size, self.size) or (
            fields.dtype not in (torch.complex64, torch.complex128)
        ):
            raise ValueError("Expected (..., Z, H, W) complex fields")
        dims = (-2, -1)
        spectrum = fields / self._post.to(dtype=fields.dtype)
        return (
            torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=dims)), dim=dims)
            * self._pre.to(dtype=fields.dtype).conj()
        )

    def amplitude(self, phase: Tensor) -> Tensor:
        """Return |U| (not intensity) for a phase-only DOE and plane wave."""
        if phase.dtype not in (torch.float32, torch.float64):
            raise ValueError("Phase must use float32 or float64 radians")
        field = torch.polar(torch.ones_like(phase), phase)
        # The output chirp and global propagation phase have unit magnitude.
        return self._spectrum(field).abs() * self._gain.to(dtype=phase.dtype)


def wrap_phase(phase: Tensor) -> Tensor:
    """Map arbitrary radians to [-π, π] while preserving both in-range endpoints."""
    return torch.where(
        (phase >= -pi) & (phase <= pi), phase, torch.remainder(phase + pi, 2 * pi) - pi
    )


def _phase_codes(phase: Tensor) -> Tensor:
    return ((wrap_phase(phase) + pi) * (255 / (2 * pi))).round().clamp(0, 255)


def quantize_phase(phase: Tensor) -> Tensor:
    """Round to the exported phase levels and decode back to radians."""
    return _phase_codes(phase) * (2 * pi / 255) - pi


def gray_to_png(gray: Tensor) -> bytes:
    """Encode [0, 1] as an eight-bit grayscale PNG."""
    if gray.ndim != 2 or not torch.isfinite(gray).all():
        raise ValueError("Expected one finite grayscale plane")
    array = gray.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    out = BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def png_to_gray(data: bytes, size: int = SIZE) -> Tensor:
    """Validate and decode a single-frame, eight-bit grayscale PNG."""
    if (
        data[:8] != b"\x89PNG\r\n\x1a\n"
        or data[8:16] != b"\x00\x00\x00\x0dIHDR"
        or data[24:26] != b"\x08\x00"
        or int.from_bytes(data[16:20], "big") != size
        or int.from_bytes(data[20:24], "big") != size
    ):
        raise ValueError(f"Expected a single {size}×{size} 8-bit grayscale PNG")
    try:
        with Image.open(BytesIO(data), formats=("PNG",)) as image:
            if (
                image.size != (size, size)
                or image.mode != "L"
                or getattr(image, "n_frames", 1) != 1
                or "transparency" in image.info
            ):
                raise ValueError(f"Expected a single {size}×{size} 8-bit grayscale PNG")
            image.verify()
        with Image.open(BytesIO(data), formats=("PNG",)) as image:
            return torch.from_numpy(np.array(image)).to(torch.float32) / 255
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or corrupt PNG") from exc


def phase_to_png(phase: Tensor) -> bytes:
    return gray_to_png(_phase_codes(phase) / 255)


def png_to_phase(data: bytes, size: int = SIZE) -> Tensor:
    return (png_to_gray(data, size) * 255) * (2 * pi / 255) - pi
