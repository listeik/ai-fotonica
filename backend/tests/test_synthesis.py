from math import pi

import pytest
import torch
from torch import nn

from doe.loss import normalize_amplitude, reconstruction_loss
from doe.metrics import evaluate
from doe.physics import FresnelPropagator, png_to_gray, png_to_phase, quantize_phase
from doe.synthesis import optimize_phase, synthesize


def test_adam_reduces_quantized_objective_with_repeatable_restarts():
    p = FresnelPropagator(16)
    rng = torch.Generator().manual_seed(123)
    true_phase = quantize_phase(torch.rand((16, 16), generator=rng) * (2 * pi) - pi)
    target = normalize_amplitude(p.amplitude(true_phase))
    initial = torch.zeros_like(true_phase)
    result = optimize_phase(target, p, iterations=25, initial_phase=initial, restarts=2, seed=8)
    repeat = optimize_phase(target, p, iterations=25, initial_phase=initial, restarts=2, seed=8)
    assert result.history == repeat.history
    assert result.loss < 0.8 * result.history[0][0]
    assert result.loss == pytest.approx(
        reconstruction_loss(p.amplitude(result.phase), target).item()
    )


def test_exported_pngs_match_physics_and_metrics():
    p = FresnelPropagator(16)
    rng = torch.Generator().manual_seed(9)
    target = torch.rand((4, 16, 16), generator=rng)
    result = synthesize(target, "classical", p, iterations=4)
    amplitude = normalize_amplitude(p.amplitude(png_to_phase(result.phase_png, 16)))
    reconstructed = torch.stack([png_to_gray(item, 16) for item in result.reconstruction_pngs])
    torch.testing.assert_close(reconstructed, (amplitude * 255).round() / 255)
    assert result.metrics == evaluate(reconstructed, target)


def test_hybrid_keeps_the_best_quantized_candidate():
    class FixedModel(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x[:, :1])

    p = FresnelPropagator(16)
    target = torch.rand((4, 16, 16), generator=torch.Generator().manual_seed(8))
    neural = synthesize(target, "neural", p, model=FixedModel())
    hybrid = synthesize(target, "hybrid", p, model=FixedModel(), iterations=8)

    def loss(result):
        return reconstruction_loss(p.amplitude(png_to_phase(result.phase_png, 16)), target).item()

    assert loss(hybrid) < loss(neural)
    with pytest.raises(ValueError, match="trained model"):
        synthesize(target, "neural", p)


def test_metrics_identity_and_undefined_pcc():
    image = torch.linspace(0, 1, 64).reshape(8, 8).expand(4, -1, -1).clone()
    image[0] = 0.5
    scores = evaluate(image, image)
    assert scores["mean_pcc"] == pytest.approx(0.75)
    assert scores["worst_pcc"] == 0
    assert scores["mean_ssim"] == pytest.approx(1)
    assert evaluate(1 - image, image)["worst_pcc"] == pytest.approx(-1)
