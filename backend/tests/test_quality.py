import numpy as np
import pytest
import torch

from doe.loss import global_ssim, reconstruction_loss
from doe.metrics import evaluate
from doe.physics import FresnelPropagator, png_to_gray
from doe.quality import QualityObjective, search
from doe.synthesis import export_phase


def test_global_ssim_matches_population_formula_and_loss():
    rng = torch.Generator().manual_seed(31)
    target = torch.rand((2, 4, 9, 11), generator=rng, dtype=torch.float64)
    pixels = torch.rand(target.shape, generator=rng, dtype=torch.float64)
    pixels[..., 0, 0] = 1
    target[0, 0] = 0
    pixels[0, 0] = 0
    target[0, 1] = 0.25
    pixels[0, 1] = 1
    pixels[1, 0] = 1 - target[1, 0]
    pixels[1, 0, 0, 0] = 1
    expected = []
    for actual_sample, target_sample in zip(pixels.numpy(), target.numpy(), strict=True):
        planes = []
        for x, y in zip(actual_sample, target_sample, strict=True):
            mean_x, mean_y = np.mean(x), np.mean(y)
            covariance = np.mean((x - mean_x) * (y - mean_y))
            numerator = (2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
            denominator = (mean_x**2 + mean_y**2 + 0.01**2) * (np.var(x) + np.var(y) + 0.03**2)
            planes.append(numerator / denominator)
        expected.append(planes)
        official = evaluate(torch.from_numpy(actual_sample), torch.from_numpy(target_sample))
        assert [plane["ssim"] for plane in official["planes"]] == pytest.approx(planes)
    assert global_ssim(pixels, target).numpy() == pytest.approx(np.array(expected))
    baseline = reconstruction_loss(pixels, target, ssim_weight=0)
    with_ssim = reconstruction_loss(pixels, target, ssim_weight=1)
    assert (with_ssim - baseline).item() == pytest.approx(1 - np.mean(expected))
    pixels.requires_grad_()
    global_ssim(pixels, target).sum().backward()
    assert torch.isfinite(pixels.grad).all()


def test_differentiable_scores_match_official_metrics():
    rng = torch.Generator().manual_seed(3)
    target, pixels = torch.randint(256, (2, 4, 16, 16), generator=rng).float() / 255
    target[0] = 0
    pixels[1] = 0.25
    pcc, ssim = QualityObjective(target).scores(pixels)
    official = evaluate(pixels, target)
    assert pcc.tolist() == pytest.approx([plane["pcc"] for plane in official["planes"]], abs=1e-6)
    assert ssim.tolist() == pytest.approx([plane["ssim"] for plane in official["planes"]], abs=1e-6)


def test_search_keeps_baseline_eligible_and_ranks_exported_pixels():
    rng = torch.Generator().manual_seed(9)
    optics = FresnelPropagator(12)
    reference = torch.rand((12, 12), generator=rng)
    other = torch.rand((12, 12), generator=rng)
    _, pngs, _ = export_phase(reference, torch.zeros((4, 12, 12)), optics)
    target = torch.stack([png_to_gray(png, 12) for png in pngs])

    winners, trials = search(
        target,
        optics,
        [("reference", reference), ("other", other)],
        steps=2,
        check_every=1,
        ssim_weights=(1.0,),
        pcc_reference="reference",
    )

    assert winners["balanced"].score == pytest.approx(2.0)
    assert winners["balanced"].score == max(trial.get("best", trial)["score"] for trial in trials)
    assert winners["pcc-preserved"].pcc == pytest.approx((1.0, 1.0))
    for winner in winners.values():
        assert winner.metrics == export_phase(winner.phase, target, optics)[2]
