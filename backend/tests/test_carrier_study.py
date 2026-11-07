import numpy as np
import pytest
import torch
from doe.training.carrier_study import StudyConfig, train, windowed_ssim
from skimage.metrics import structural_similarity


@pytest.mark.parametrize("window", [7, 11])
def test_training_windowed_ssim_matches_export_metric_and_has_gradients(window):
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(91)
    x = torch.rand(2, 4, 24, 24, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.rand(2, 4, 24, 24, generator=generator, dtype=torch.float64)
    expected = np.array(
        [
            [
                structural_similarity(a.detach().numpy(), b.numpy(), win_size=window, data_range=1)
                for a, b in zip(first, second, strict=True)
            ]
            for first, second in zip(x, y, strict=True)
        ]
    )
    actual = windowed_ssim(x, y, window)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=0, atol=1e-10)
    actual.mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


@pytest.mark.parametrize("window", [7, 11])
def test_windowed_ssim_constant_targets_stays_finite(window):
    x = torch.zeros(1, 4, 24, 24, requires_grad=True)
    result = windowed_ssim(x, torch.zeros_like(x), window)
    torch.testing.assert_close(result, torch.ones_like(result))
    result.mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": 6},
        {"lr": 0},
        {"lr": float("nan")},
        {"profile": "unknown"},
        {"window_size": 10},
        {"supervised_epochs": -1},
        {"supervised_epochs": 1},
    ],
)
def test_invalid_study_configuration_fails_before_loading_data(tmp_path, options):
    with pytest.raises(ValueError):
        train(StudyConfig(name="invalid", **options), tmp_path / "missing-data", tmp_path / "out")
