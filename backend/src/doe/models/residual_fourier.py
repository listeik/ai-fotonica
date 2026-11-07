"""Phase-only DOE synthesis with a fixed Fresnel residual and a Fourier U-Net."""

import math

import torch
from torch import Tensor, nn

from doe.physics import FresnelPropagator

from .unet import DEFAULT_AMPLITUDE_FLOOR, _block, validate_amplitude_floor

ARCHITECTURE = "physics-informed-multiplane-residual-fourier-unet-v1"


def _unit(field: Tensor) -> Tensor:
    magnitude = field.abs()
    return torch.where(magnitude > 1e-6, field / magnitude.clamp_min(1e-6), 1 + 0j)


class FourierResidual(nn.Module):
    """Learn a channelwise global filter in a band of low spatial frequencies."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(0.02 * torch.randn(channels, 2 * modes + 1, modes, 2))
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        height, width = features.shape[-2:]
        spectrum = torch.fft.rfft2(features.float(), norm="ortho")
        modes = self.modes
        # Exclude DC and Nyquist columns so irfft2 can supply their omitted conjugates
        # without requiring constraints on the learned complex weights.
        rows = torch.cat((spectrum[..., -modes:, :], spectrum[..., : modes + 1, :]), dim=-2)
        band = rows[..., 1 : modes + 1] * torch.view_as_complex(self.weights)
        filtered = torch.zeros_like(spectrum)
        filtered[..., -modes:, 1 : modes + 1] = band[..., :modes, :]
        filtered[..., : modes + 1, 1 : modes + 1] = band[..., modes:, :]
        spatial = torch.fft.irfft2(filtered, s=(height, width), norm="ortho")
        return features + self.mix(spatial)


class MultiPlaneResidualFourierUNet(nn.Module):
    """Read four target amplitudes and emit one 128×128 phase map in radians."""

    architecture = ARCHITECTURE

    def __init__(
        self,
        width: int = 12,
        modes: int = 5,
        *,
        amplitude_floor: float = DEFAULT_AMPLITUDE_FLOOR,
    ) -> None:
        super().__init__()
        self.propagator = FresnelPropagator()
        size = self.propagator.size
        if not 4 <= width <= 64 or not 1 <= modes < size // 16:
            raise ValueError("Invalid U-Net width or Fourier mode count")
        validate_amplitude_floor(amplitude_floor)
        self.amplitude_floor = amplitude_floor
        self.encode1 = _block(14, width)
        self.encode2 = _block(width, 2 * width)
        self.encode3 = _block(2 * width, 4 * width)
        self.bridge = _block(4 * width, 8 * width)
        self.fourier = FourierResidual(8 * width, modes)
        self.decode3 = _block(8 * width, 4 * width)
        self.decode2 = _block(4 * width, 2 * width)
        self.decode1 = _block(2 * width, width)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8 * width, 4 * width, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        # Use the same deterministic plane carrier as the comparison model.
        rng = torch.Generator(device="cpu").manual_seed(1597334677)
        phase_code = (2 * torch.rand(4, size, size, generator=rng) - 1) * math.pi
        self.register_buffer("phase_code", phase_code)

    def forward(self, target: Tensor) -> Tensor:
        amplitude = target.float() + self.amplitude_floor
        initial = self.propagator.inverse_planes(torch.polar(amplitude, self.phase_code))
        probe = _unit(initial.sum(dim=1))
        fields = self.propagator(probe)
        magnitude = fields.abs()
        dims = (-2, -1)
        scale = magnitude.norm(dim=dims, keepdim=True) / amplitude.norm(dim=dims, keepdim=True)
        mismatch = (scale * amplitude - magnitude) * _unit(fields)
        residual = self.propagator.inverse_planes(mismatch)
        reference = _unit(probe + residual.mean(dim=1))

        features = torch.cat(
            (
                target,
                reference.real[:, None],
                reference.imag[:, None],
                residual.real,
                residual.imag,
            ),
            dim=1,
        )
        enc1 = self.encode1(features)
        enc2 = self.encode2(self.pool(enc1))
        enc3 = self.encode3(self.pool(enc2))
        bridge = self.fourier(self.bridge(self.pool(enc3)))
        dec3 = self.decode3(torch.cat((self.up3(bridge), enc3), dim=1))
        dec2 = self.decode2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.decode1(torch.cat((self.up1(dec2), enc1), dim=1))
        delta = math.pi * self.head(dec1).float().tanh()
        return torch.angle(reference[:, None] * torch.polar(torch.ones_like(delta), delta))
