"""A U-Net with a differentiable Fresnel readout."""

import math

import torch
from torch import Tensor, nn

from doe.physics import FresnelPropagator

ARCHITECTURE = "fresnel-unet-v1"


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

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        if not 4 <= width <= 64:
            raise ValueError("U-Net width must be between 4 and 64")
        self.width = width
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
        amplitude = (target.float() + 0.03) * torch.exp(0.8 * gain)
        wavefront = self.propagator.inverse_planes(torch.polar(amplitude, phase)).sum(
            dim=1, keepdim=True
        )
        # atan2's derivative is undefined at zero. Mask those inputs first.
        safe = torch.where(wavefront.abs() < 1e-6, 1e-6 + 0j, wavefront)
        return torch.angle(safe)
