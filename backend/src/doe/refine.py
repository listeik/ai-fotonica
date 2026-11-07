"""Refine an existing 8-bit DOE against four targets and measure optical leakage."""

import argparse
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import Tensor

from doe.loss import reconstruction_loss
from doe.physics import FresnelPropagator, png_to_gray, png_to_phase
from doe.synthesis import LEARNING_RATE, export_phase, optimize_phase


def leakage_masks(target: Tensor, optics: FresnelPropagator) -> tuple[Tensor, Tensor]:
    """Build a dark mask and a soft adjacent-plane silhouette mask on each native grid."""
    dark = target <= 2 / 255
    guard = F.max_pool2d((~dark).float().unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
    support = (target >= 0.08).float()
    size = optics.size
    axis = torch.arange(size, device=target.device, dtype=target.dtype) - size // 2
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1)
    pitches = optics.output_pitch_m.to(target)
    ghost_masks = []
    for index, pitch in enumerate(pitches):
        union = torch.zeros_like(target[index])
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(pitches):
                mapped = coordinates * (pitch / pitches[neighbor]) + size // 2
                grid = (2 * mapped / (size - 1) - 1).unsqueeze(0)
                mask = F.grid_sample(support[neighbor][None, None], grid, align_corners=True)
                union = torch.maximum(union, mask[0, 0])
        ghost_masks.append(union * (1 - guard[index]))
    return dark, torch.stack(ghost_masks)


def power_fractions(amplitude: Tensor, dark: Tensor, ghost: Tensor) -> tuple[Tensor, Tensor]:
    intensity = amplitude.square()
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-8)
    dark_power = (intensity * dark).sum(dim=(-2, -1)) / total
    ghost_power = (intensity * ghost).sum(dim=(-2, -1)) / total
    return dark_power, ghost_power


@torch.no_grad()
def _record(
    phase: Tensor, target: Tensor, optics: FresnelPropagator, dark: Tensor, ghost: Tensor
) -> tuple[bytes, tuple[bytes, ...], dict]:
    phase_png, pngs, metrics = export_phase(phase, target, optics)
    amplitude = optics.amplitude(png_to_phase(phase_png, optics.size).to(target.device))
    background, silhouettes = power_fractions(amplitude, dark, ghost)
    for plane, bg, xt in zip(
        metrics["planes"], background.tolist(), silhouettes.tolist(), strict=True
    ):
        plane["background_power_fraction"] = bg
        plane["ghost_region_power_fraction"] = xt
    metrics["mean_background_power_fraction"] = background.mean().item()
    metrics["mean_ghost_region_power_fraction"] = silhouettes.mean().item()
    return phase_png, pngs, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs=4, type=Path, help="PNGs ordered by increasing distance")
    parser.add_argument("--phase", required=True, type=Path, help="Input DOE PNG")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the results")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--background-weight", type=float, default=0.08)
    parser.add_argument("--ghost-weight", type=float, default=0.04)
    parser.add_argument("--pcc-weight", type=float, default=0.40)
    parser.add_argument("--shape-weight", type=float, default=0.70)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.output.exists():
        parser.error("Output directory already exists")
    weights = {
        "background": args.background_weight,
        "ghost": args.ghost_weight,
        "pcc": args.pcc_weight,
        "shape": args.shape_weight,
    }
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        parser.error("Loss weights must be finite and nonnegative")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    optics = FresnelPropagator().to(device)
    try:
        target = torch.stack([png_to_gray(path.read_bytes()) for path in args.targets]).to(device)
        initial = png_to_phase(args.phase.read_bytes()).to(device)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    dark, ghost = leakage_masks(target, optics)

    def objective(amplitude: Tensor, expected: Tensor) -> Tensor:
        background, silhouettes = power_fractions(amplitude, dark, ghost)
        shape = 1 - F.cosine_similarity(amplitude.flatten(1), expected.flatten(1), dim=1).mean()
        return (
            reconstruction_loss(amplitude, expected, pcc_weight=args.pcc_weight)
            + args.background_weight * background.mean()
            + args.ghost_weight * silhouettes.mean()
            + args.shape_weight * shape
        )

    _, _, before = _record(initial, target, optics, dark, ghost)
    started = perf_counter()
    try:
        result = optimize_phase(
            target,
            optics,
            iterations=args.steps,
            lr=args.lr,
            initial_phase=initial,
            objective=objective,
        )
    except ValueError as exc:
        parser.error(str(exc))
    seconds = perf_counter() - started
    phase_png, reconstructions, after = _record(result.phase, target, optics, dark, ghost)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "doe.png").write_bytes(phase_png)
    for index, png in enumerate(reconstructions, 1):
        (args.output / f"reconstruction-{index}.png").write_bytes(png)
    report = {
        "steps": args.steps,
        "learning_rate": args.lr,
        "weights": weights,
        "device": args.device,
        "seconds": seconds,
        "before": before,
        "after": after,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved refined DOE, four reconstructions and metrics to {args.output}")


if __name__ == "__main__":
    main()
