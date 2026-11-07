import math
from dataclasses import replace

import torch

from doe.models import FresnelUNet, load_model
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


def test_zero_targets_have_finite_phase_gradient():
    torch.set_num_threads(2)
    model = FresnelUNet(width=4)
    target = torch.zeros(2, 4, 128, 128)
    phase = model(target)
    assert phase.shape == (2, 1, 128, 128)
    assert torch.isfinite(phase).all()
    assert phase.abs().amax() <= math.pi
    phase.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


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
