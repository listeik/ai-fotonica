"""Evaluate a checkpoint with the shared quantized reconstruction protocol."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from doe.loss import normalize_amplitude, reconstruction_loss
from doe.metrics import evaluate
from doe.models import load_model
from doe.physics import FresnelPropagator, phase_to_png, png_to_phase, quantize_phase

from .dataset import SyntheticTargets


@torch.inference_mode()
def validation_metrics(
    model, propagator, batches, device: torch.device, *, png_round_trip: bool = False
) -> dict:
    model.eval()
    count = 0
    loss_sum = 0.0
    pcc_sum = 0.0
    ssim_sum = 0.0
    worst_pcc = float("inf")
    worst_ssim = float("inf")
    for target in batches:
        target = target.to(device)
        phase = model(target).squeeze(1)
        if png_round_trip:
            phase = torch.stack([png_to_phase(phase_to_png(item)) for item in phase.cpu()])
            phase = phase.to(device)
        else:
            phase = quantize_phase(phase)
        amplitude = propagator.amplitude(phase)
        loss_sum += float(reconstruction_loss(amplitude, target)) * len(target)
        normalized = (255 * normalize_amplitude(amplitude)).round() / 255
        for reconstructed, expected in zip(normalized.cpu(), target.cpu(), strict=True):
            values = evaluate(reconstructed, expected, distances_m=propagator.distances_m)
            pcc_sum += values["mean_pcc"]
            ssim_sum += values["mean_ssim"]
            worst_pcc = min(worst_pcc, values["worst_pcc"])
            worst_ssim = min(worst_ssim, values["worst_ssim"])
            count += 1
    return {
        "loss": loss_sum / count,
        "mean_pcc": pcc_sum / count,
        "mean_ssim": ssim_sum / count,
        "worst_pcc": worst_pcc,
        "worst_ssim": worst_ssim,
        "samples": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, version = load_model(args.checkpoint, device)
    propagator = FresnelPropagator().to(device)
    batches = DataLoader(
        SyntheticTargets(args.split, args.count, args.seed), batch_size=args.batch_size
    )
    result = validation_metrics(model, propagator, batches, device, png_round_trip=True)
    print(
        json.dumps({"model": version, "split": args.split, "seed": args.seed, **result}, indent=2)
    )


if __name__ == "__main__":
    main()
