import pytest
import torch
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.models.unet import FOURIER_UNET_ARCHITECTURE
from doe.training.dataset import synthetic_targets


def spectral_checkpoint(model):
    return {
        "format_version": CHECKPOINT_FORMAT,
        "architecture": model.architecture,
        "width": model.width,
        "projection_steps": model.projection_steps,
        "amplitude_floor": model.amplitude_floor,
        "fourier_bottleneck": {"modes": model.fourier_modes},
        "step": 1,
        "model": model.state_dict(),
    }


@pytest.mark.parametrize("width,projections", [(16, 3), (24, 6), (32, 9)])
def test_study_widths_and_projection_counts_produce_finite_phase(width, projections):
    torch.set_num_threads(2)
    model = FresnelUNet(width, projections, carrier_conditioning=True)
    target = synthetic_targets(0, split="train").unsqueeze(0)
    with torch.inference_mode():
        phase = model(target)
    assert phase.shape == (1, 1, 128, 128)
    assert torch.isfinite(phase).all()
    assert model.plane_logits.shape == (projections, 4)


def test_zero_initialized_fourier_mix_preserves_seeded_base_and_trains():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    original = FresnelUNet(4, 1, carrier_conditioning=True)
    torch.manual_seed(71)
    spectral = FresnelUNet(4, 1, carrier_conditioning=True, fourier_modes=3)
    assert spectral.architecture == FOURIER_UNET_ARCHITECTURE
    assert torch.count_nonzero(spectral.fourier.mix.weight) == 0
    original_state = original.state_dict()
    for key, value in original_state.items():
        torch.testing.assert_close(spectral.state_dict()[key], value, rtol=0, atol=0)
    target = synthetic_targets(1, split="train").unsqueeze(0)
    with torch.inference_mode():
        torch.testing.assert_close(spectral(target), original(target), rtol=0, atol=0)
    optimizer = torch.optim.Adam(spectral.parameters(), lr=0.001)
    spectral(target).square().mean().backward()
    assert spectral.fourier.mix.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in spectral.parameters() if p.grad is not None)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    spectral(target).square().mean().backward()
    assert spectral.fourier.weights.grad.norm() > 0


@pytest.mark.parametrize("carrier", [False, True])
def test_fourier_unet_checkpoint_round_trip_with_optional_carrier(tmp_path, carrier):
    torch.set_num_threads(2)
    model = FresnelUNet(4, 9, carrier_conditioning=carrier, fourier_modes=3)
    state = spectral_checkpoint(model)
    if carrier:
        state["carrier_conditioning"] = {"scale": 1}
    path = tmp_path / "variant.pt"
    torch.save(state, path)
    restored, version = load_model(path)
    assert version.startswith(f"{FOURIER_UNET_ARCHITECTURE}:")
    assert restored.projection_steps == 9
    assert restored.fourier_modes == 3
    assert restored.carrier_conditioning == carrier
    target = synthetic_targets(0, split="train").unsqueeze(0)
    with torch.inference_mode():
        torch.testing.assert_close(restored(target), model.eval()(target), rtol=0, atol=0)


@pytest.mark.parametrize(
    "metadata", [None, {}, {"modes": 0}, {"modes": 8}, {"modes": True}, {"modes": 3, "unknown": 1}]
)
def test_fourier_unet_rejects_invalid_bottleneck_metadata(tmp_path, metadata):
    model = FresnelUNet(4, 1, fourier_modes=3)
    state = spectral_checkpoint(model)
    state["fourier_bottleneck"] = metadata
    path = tmp_path / "invalid.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_fourier_unet_requires_bottleneck_metadata(tmp_path):
    model = FresnelUNet(4, 1, fourier_modes=3)
    state = spectral_checkpoint(model)
    del state["fourier_bottleneck"]
    path = tmp_path / "missing.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_non_spectral_checkpoint_rejects_bottleneck_metadata(tmp_path):
    model = FresnelUNet(4, 1)
    path = tmp_path / "mislabeled.pt"
    torch.save(spectral_checkpoint(model), path)
    with pytest.raises(ValueError):
        load_model(path)


@pytest.mark.parametrize("modes", [-1, 8, True, 1.0])
def test_fourier_mode_count_requires_supported_integer(modes):
    with pytest.raises(ValueError):
        FresnelUNet(4, 1, fourier_modes=modes)
