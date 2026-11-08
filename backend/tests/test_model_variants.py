import pytest
import torch

from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.training.dataset import synthetic_targets


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


def test_non_spectral_checkpoint_rejects_bottleneck_metadata(tmp_path):
    model = FresnelUNet(4, 1)
    path = tmp_path / "mislabeled.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT,
            "architecture": model.architecture,
            "width": model.width,
            "projection_steps": model.projection_steps,
            "fourier_bottleneck": {"modes": 3},
            "step": 1,
            "model": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError):
        load_model(path)
