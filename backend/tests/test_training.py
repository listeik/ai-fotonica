import math
from dataclasses import replace

import pytest
import torch

from doe.loss import background_power_fraction, reconstruction_loss
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.models.residual_fourier import COMPLEX_ARCHITECTURE, MultiPlaneResidualFourierUNet
from doe.training.dataset import SyntheticTargets, synthetic_targets
from doe.training.train import TrainingConfig, train


def test_dataset_splits_and_png_levels():
    train_set = SyntheticTargets("train", count=4)
    train_targets = [train_set[i] for i in range(len(train_set))]
    for index, sample in enumerate(train_targets):
        assert sample.shape == (4, 128, 128)
        assert sample.dtype == torch.float32
        assert torch.equal(sample, synthetic_targets(index, split="train"))
        torch.testing.assert_close(sample * 255, (sample * 255).round())
        assert (sample.amax(dim=(-2, -1)) > 0).all()
        assert (sample.amin(dim=(-2, -1)) == 0).all()
        split_samples = [synthetic_targets(index, split=split) for split in ("val", "test", "demo")]
        distinct = {item.numpy().tobytes() for item in [sample, *split_samples]}
        assert len(distinct) == 4

    assert torch.unique(train_targets[0]).numel() == 2
    assert torch.unique(train_targets[3]).numel() > 100


@pytest.mark.parametrize("projection_steps", [0, 3])
def test_zero_targets_have_finite_phase_gradient(projection_steps):
    torch.set_num_threads(2)
    model = FresnelUNet(width=4, projection_steps=projection_steps)
    target = torch.zeros(2, 4, 128, 128)
    phase = model(target)
    assert phase.shape == (2, 1, 128, 128)
    assert torch.isfinite(phase).all()
    assert phase.abs().amax() <= math.pi
    phase.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_complex_fourier_checkpoint_reloads_the_trained_head(tmp_path):
    torch.set_num_threads(2)
    model = MultiPlaneResidualFourierUNet(width=4, modes=3, complex_correction=True)
    target = synthetic_targets(0, split="train").unsqueeze(0)
    assert torch.count_nonzero(model.head.weight) == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(target).square().mean().backward()
    optimizer.step()
    assert torch.count_nonzero(model.head.weight) > 0

    checkpoint = tmp_path / "fourier-v2.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT,
            "architecture": model.architecture,
            "width": 4,
            "config": {"fourier_modes": 3},
            "step": 1,
            "model": model.state_dict(),
        },
        checkpoint,
    )
    restored, version = load_model(checkpoint)
    assert version.startswith(f"{COMPLEX_ARCHITECTURE}:")
    with torch.inference_mode():
        phase = restored(target)
        assert phase.shape == (1, 1, 128, 128)
        assert torch.isfinite(phase).all()
        torch.testing.assert_close(phase, model.eval()(target), rtol=0, atol=0)


def test_cpu_checkpoint_resume_matches_continuous_training(tmp_path):
    config = TrainingConfig(
        epochs=2,
        train_count=2,
        val_count=1,
        width=4,
        batch_size=1,
        threads=2,
        patience=0,
    )
    device = torch.device("cpu")
    resumed_path = tmp_path / "resumed"
    continuous_path = tmp_path / "continuous"
    history = train(config, continuous_path, device)
    train(replace(config, epochs=1), resumed_path, device)
    resumed_history = train(config, resumed_path, device, resume=True)

    assert history == resumed_history
    assert history[-1]["step"] == 4
    continuous, _ = load_model(continuous_path / "best.pt", device)
    resumed, model_id = load_model(resumed_path / "best.pt", device)
    assert model_id.startswith("fresnel-unet-v1:")
    assert not resumed.training
    target = synthetic_targets(0, split="val").unsqueeze(0)
    with torch.inference_mode():
        torch.testing.assert_close(continuous(target), resumed(target), rtol=0, atol=0)


def test_background_penalty_uses_power_per_plane():
    amplitude = torch.tensor([[[[2.0, 1.0]], [[0.0, 0.0]]]], requires_grad=True)
    target = torch.tensor([[[[2 / 255, 3 / 255]], [[0.0, 0.0]]]])
    fractions = background_power_fraction(amplitude, target)
    torch.testing.assert_close(fractions, torch.tensor([[0.8, 0.0]]))
    fractions.sum().backward()
    assert amplitude.grad[0, 0, 0, 0] > 0
    assert amplitude.grad[0, 0, 0, 1] < 0
    assert torch.isfinite(amplitude.grad).all()


def test_worst_pcc_penalty_balances_samples():
    target = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).expand(2, 4, 8, 8)
    amplitude = target.clone()
    amplitude[0, 0] = 1 - target[0, 0]
    baseline = reconstruction_loss(amplitude, target)
    penalized = reconstruction_loss(amplitude, target, worst_pcc_weight=0.35)
    assert (penalized - baseline).item() == pytest.approx(0.35)


def test_finetune_loads_legacy_weights_and_keeps_floor_and_lineage_on_resume(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "finetune"
    model = FresnelUNet(width=4, projection_steps=1)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT,
            "architecture": model.architecture,
            "width": model.width,
            "projection_steps": model.projection_steps,
            "step": 1,
            "model": model.state_dict(),
        },
        source,
    )
    initial, version = load_model(source)
    assert initial.amplitude_floor == 0.03

    config = TrainingConfig(
        epochs=1,
        train_count=1,
        val_count=1,
        width=4,
        projection_steps=1,
        batch_size=1,
        threads=2,
        amplitude_floor=0.005,
        background_weight=0.25,
        selection="pcc-ssim",
    )
    device = torch.device("cpu")
    history = train(config, output, device, init_checkpoint=source)
    assert history[0]["initialized_from"] == version
    assert "reference_validation" in history[0]
    resumed = train(replace(config, epochs=2), output, device, resume=True)
    assert resumed[:2] == history
    tuned, _ = load_model(output / "last.pt")
    assert tuned.amplitude_floor == config.amplitude_floor
    assert resumed[-1]["step"] == 2
    assert torch.load(output / "last.pt", weights_only=True)["initialized_from"] == version
