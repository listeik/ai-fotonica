import pytest
import torch

from doe.metrics import evaluate
from doe.physics import FresnelPropagator, png_to_gray
from doe.quality import QualityObjective, search
from doe.synthesis import export_phase


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
