"""Evaluate a checkpoint with the shared quantized reconstruction protocol."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from doe.loss import background_power_fraction, normalize_amplitude, reconstruction_loss
from doe.metrics import evaluate
from doe.models import load_model
from doe.physics import FresnelPropagator, phase_to_png, png_to_gray, png_to_phase, quantize_phase

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
    background_sum = 0.0
    max_peak_power = 0.0
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
        background_sum += background_power_fraction(amplitude, target).mean(dim=-1).sum().item()
        intensity = amplitude.square()
        peak = intensity.amax(dim=(-2, -1)) / intensity.sum(dim=(-2, -1)).clamp_min(1e-8)
        max_peak_power = max(max_peak_power, peak.max().item())
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
        "mean_background_power_fraction": background_sum / count,
        "max_peak_power_fraction": max_peak_power,
        "samples": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=("val", "test"), default="val")
    source.add_argument(
        "--target-dir",
        type=Path,
        action="append",
        help="directory with four PNGs named in distance order (repeat for more sets)",
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, version = load_model(args.checkpoint, device)
    propagator = FresnelPropagator().to(device)
    if args.target_dir:
        samples = []
        try:
            for folder in args.target_dir:
                paths = sorted(folder.glob("*.png"))
                if len(paths) != 4:
                    raise ValueError(f"{folder} must contain four target PNGs in distance order")
                samples.append(torch.stack([png_to_gray(path.read_bytes()) for path in paths]))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        batches = DataLoader(samples, batch_size=args.batch_size)
        metadata = {"split": "external", "target_dirs": [str(path) for path in args.target_dir]}
    else:
        batches = DataLoader(
            SyntheticTargets(args.split, args.count, args.seed), batch_size=args.batch_size
        )
        metadata = {"split": args.split, "seed": args.seed}
    result = validation_metrics(model, propagator, batches, device, png_round_trip=True)
    print(json.dumps({"model": version, **metadata, **result}, indent=2))


if __name__ == "__main__":
    main()
