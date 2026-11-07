"""A U-Net with a differentiable Fresnel readout."""

import math

import torch
from torch import Tensor, nn

from doe.physics import FresnelPropagator

ARCHITECTURE = "fresnel-unet-v1"
PROJECTED_ARCHITECTURE = "fresnel-unet-projection-v2"
DEFAULT_AMPLITUDE_FLOOR = 0.03


def validate_amplitude_floor(value: float) -> None:
    if not math.isfinite(value) or not 1e-4 <= value <= 1:
        raise ValueError("amplitude_floor must be in [0.0001, 1]")


def _block(inputs: int, outputs: int) -> nn.Sequential:
    groups = math.gcd(8, outputs)
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(groups, outputs),
        nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(groups, outputs),
        nn.SiLU(),
    )


class FresnelUNet(nn.Module):
    """Predict plane fields and map their superposition to a phase-only DOE."""

    def __init__(
        self,
        width: int = 16,
        projection_steps: int = 0,
        *,
        amplitude_floor: float = DEFAULT_AMPLITUDE_FLOOR,
    ) -> None:
        super().__init__()
        if not 4 <= width <= 64:
            raise ValueError("U-Net width must be between 4 and 64")
        validate_amplitude_floor(amplitude_floor)
        self.width = width
        self.amplitude_floor = amplitude_floor
        if type(projection_steps) is not int or not 0 <= projection_steps <= 8:
            raise ValueError("Expected an integer projection count in [0, 8]")
        self.projection_steps = projection_steps
        self.architecture = PROJECTED_ARCHITECTURE if projection_steps else ARCHITECTURE
        if projection_steps:
            self.plane_logits = nn.Parameter(torch.zeros(projection_steps, 4))
            self.relaxation_logits = nn.Parameter(torch.ones(projection_steps) * 2)
        self.propagator = FresnelPropagator()
        self.encode1 = _block(4, width)
        self.encode2 = _block(width, 2 * width)
        self.encode3 = _block(2 * width, 4 * width)
        self.bridge = _block(4 * width, 8 * width)
        self.decode3 = _block(8 * width, 4 * width)
        self.decode2 = _block(4 * width, 2 * width)
        self.decode1 = _block(2 * width, width)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8 * width, 4 * width, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.head = nn.Conv2d(width, 8, 1)
        # Start close to a phase-coded optical superposition. Small nonzero
        # head weights allow gradients to reach the encoder on the first step.
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        rng = torch.Generator(device="cpu").manual_seed(1597334677)
        size = self.propagator.size
        phase_code = (2 * torch.rand(4, size, size, generator=rng) - 1) * math.pi
        self.register_buffer("phase_code", phase_code)

    def forward(self, target: Tensor) -> Tensor:
        enc1 = self.encode1(target)
        enc2 = self.encode2(self.pool(enc1))
        enc3 = self.encode3(self.pool(enc2))
        bridge = self.bridge(self.pool(enc3))
        dec3 = self.decode3(torch.cat([self.up3(bridge), enc3], dim=1))
        dec2 = self.decode2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.decode1(torch.cat([self.up1(dec2), enc1], dim=1))
        phase_delta, gain = self.head(dec1).float().tanh().chunk(2, dim=1)
        phase = self.phase_code + math.pi * phase_delta
        # A weak floor allows the model to shape the field outside each target.
        amplitude = (target.float() + self.amplitude_floor) * torch.exp(0.8 * gain)
        wavefront = self.propagator.inverse_planes(torch.polar(amplitude, phase)).sum(
            dim=1, keepdim=True
        )
        if self.projection_steps:
            # Fixed-depth learned projection layers. Each layer adopts the predicted
            # target magnitude at the current plane phase, then enforces a phase-only DOE.
            energy = amplitude.square().sum(dim=(-2, -1), keepdim=True)
            for logits, relaxation in zip(self.plane_logits, self.relaxation_logits, strict=True):
                doe = self._unit(wavefront)
                fields = self.propagator(doe.squeeze(1))
                magnitude = fields.abs()
                scale = (magnitude.square().sum(dim=(-2, -1), keepdim=True) / energy).sqrt()
                projected = self.propagator.inverse_planes(scale * amplitude * self._unit(fields))
                weight = logits.softmax(dim=0)[None, :, None, None]
                combined = (projected * weight).sum(dim=1, keepdim=True)
                fraction = relaxation.sigmoid()
                wavefront = (1 - fraction) * doe + fraction * combined
        # atan2's derivative is undefined at zero. Mask those inputs first.
        safe = torch.where(wavefront.abs() < 1e-6, 1e-6 + 0j, wavefront)
        return torch.angle(safe)

    @staticmethod
    def _unit(field: Tensor) -> Tensor:
        magnitude = field.abs()
        return torch.where(magnitude > 1e-6, field / magnitude.clamp_min(1e-6), 1 + 0j)
