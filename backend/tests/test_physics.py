from io import BytesIO
from math import pi

import numpy as np
import pytest
import torch
from PIL import Image

from doe.physics import (
    DISTANCES_M,
    FresnelPropagator,
    gray_to_png,
    phase_to_png,
    png_to_gray,
    png_to_phase,
    quantize_phase,
)


@pytest.mark.parametrize("size", [7, 8])
def test_complex_field_matches_direct_fresnel_integral(size):
    # The reference sums the spatial-domain kernel with no FFT or chirp factorization.
    rng = np.random.default_rng(7)
    u = np.exp(1j * rng.normal(size=(size, size)))
    p = FresnelPropagator(size)
    actual = p(torch.from_numpy(u)).numpy()
    d, wavelength = 10e-6, 532e-9
    index = np.arange(size) - size // 2
    y, x = np.meshgrid(index * d, index * d, indexing="ij")
    for plane, z in enumerate(DISTANCES_M):
        yp, xp = np.meshgrid(
            index * wavelength * z / (size * d),
            index * wavelength * z / (size * d),
            indexing="ij",
        )
        r_sq = (xp[..., None, None] - x) ** 2 + (yp[..., None, None] - y) ** 2
        kernel = np.exp(1j * pi * r_sq / (wavelength * z))
        expected = (kernel * u).sum(axis=(-2, -1))
        expected *= d**2 * np.exp(2j * pi * z / wavelength) / (1j * wavelength * z)
        np.testing.assert_allclose(actual[plane], expected, rtol=2e-10, atol=2e-10)


def test_focal_spot_units_power_and_batch():
    p = FresnelPropagator()
    z = DISTANCES_M[0]
    assert p.output_pitch_m[0].item() == pytest.approx(10.0165625e-6)
    assert p.output_axes_m[0, 64].item() == 0
    x = p.input_axis_m
    lens = -pi * (x.square()[:, None] + x.square()[None, :]) / (532e-9 * z)
    phase = lens.float().expand(2, -1, -1)
    field = p(torch.exp(1j * phase))
    assert field.shape == (2, 4, 128, 128)
    assert field.dtype == torch.complex64
    assert field[0, 0].abs().argmax().item() == 64 * 128 + 64
    assert torch.isfinite(field).all()
    energy = field[0].abs().square().sum(dim=(-2, -1)) * p.output_pitch_m.square()
    torch.testing.assert_close(energy, torch.full_like(energy, 128**2 * 1e-10))
    torch.testing.assert_close(p.amplitude(phase), field.abs())


def test_amplitude_gradient_matches_finite_difference():
    p = FresnelPropagator(5, distances_m=(DISTANCES_M[1],))
    phase = torch.linspace(-2.5, 2.5, 25, dtype=torch.float64).reshape(5, 5)
    phase.requires_grad_()
    assert torch.autograd.gradcheck(p.amplitude, (phase,))


def test_inverse_recovers_input_from_each_plane():
    p = FresnelPropagator(13)
    field = torch.exp(1j * torch.randn(2, 13, 13, dtype=torch.float64))
    expected = field.unsqueeze(1).expand(-1, 4, -1, -1)
    torch.testing.assert_close(p.inverse_planes(p(field)), expected)


def test_exported_phase_matches_quantizer_and_reconstruction():
    p = FresnelPropagator(16)
    phase = torch.linspace(-pi, pi, 16**2).reshape(16, 16)
    png = phase_to_png(phase)
    gray = png_to_gray(png, 16)
    assert gray[0, 0].item() == 0
    assert gray[-1, -1].item() == 1
    decoded = png_to_phase(png, 16)
    torch.testing.assert_close(decoded, quantize_phase(phase))
    torch.testing.assert_close(p.amplitude(decoded), p.amplitude(quantize_phase(phase)))
    off_grid = phase + 0.01
    delta = png_to_phase(phase_to_png(off_grid), 16) - off_grid
    assert torch.atan2(delta.sin(), delta.cos()).abs().max().item() <= pi / 255 + 1e-6


def test_png_validation():
    for mode, size in [
        ("RGB", (128, 128)),
        ("I;16", (128, 128)),
        ("1", (128, 128)),
        ("L", (64, 64)),
    ]:
        buf = BytesIO()
        Image.new(mode, size).save(buf, format="PNG")
        with pytest.raises(ValueError):
            png_to_gray(buf.getvalue())
    with pytest.raises(ValueError):
        png_to_gray(gray_to_png(torch.zeros(128, 128))[:40])
