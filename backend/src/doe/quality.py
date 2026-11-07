"""Search multiple starts for a high quality phase-only DOE and export the winners."""

import argparse
import json
from dataclasses import dataclass
from math import isfinite, pi
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import Tensor

from doe.loss import normalize_amplitude
from doe.metrics import evaluate
from doe.physics import FresnelPropagator, png_to_gray, png_to_phase, quantize_phase
from doe.synthesis import export_phase


class QualityObjective:
    """Differentiable approximation of the official PCC and SSIM on 8-bit images."""

    def __init__(self, target: Tensor) -> None:
        self.target = target
        self.dark = target <= 2 / 255
        self.centered = target - target.mean(dim=(-2, -1), keepdim=True)
        self.energy = self.centered.square().sum(dim=(-2, -1))
        window = min(7, *target.shape[-2:])
        self.window = window - (1 - window % 2)
        n = self.window**2
        self.correction = n / (n - 1)
        self.mean, square = self.pool(torch.stack((target, target.square()), dim=1)).unbind(1)
        self.variance = (square - self.mean.square()).clamp_min(0) * self.correction

    def pool(self, image: Tensor) -> Tensor:
        rows = F.avg_pool2d(image, (1, self.window), stride=1)
        return F.avg_pool2d(rows, (self.window, 1), stride=1)

    def scores(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        centered = pixels - pixels.mean(dim=(-2, -1), keepdim=True)
        pcc = (centered * self.centered).sum(dim=(-2, -1)) / (
            centered.square().sum(dim=(-2, -1)) * self.energy
        ).clamp_min(1e-12).sqrt()
        # Match the official convention for constant input images.
        pcc = torch.where(pixels.amax((-2, -1)) > pixels.amin((-2, -1)), pcc, 0)
        mean, square, product = self.pool(
            torch.stack((pixels, pixels.square(), pixels * self.target), dim=1)
        ).unbind(1)
        variance = (square - mean.square()).clamp_min(0) * self.correction
        covariance = (product - mean * self.mean) * self.correction
        ssim = ((2 * mean * self.mean + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mean.square() + self.mean.square() + 0.01**2) * (variance + self.variance + 0.03**2)
        )
        return pcc, ssim.mean(dim=(-2, -1))

    def __call__(self, amplitude: Tensor, pixels: Tensor, ssim_weight: float) -> Tensor:
        pcc, ssim = self.scores(pixels)
        background, peak = power(amplitude, self.dark)
        # A bright spike can raise peak-normalized SSIM by dimming the rest of the image.
        # Penalize it and light in the dark regions in unnormalized optical power.
        peak_penalty = (peak - 0.01).clamp_min(0).mean()
        return (
            -(0.8 * pcc.mean() + 0.2 * pcc.min())
            - ssim_weight * (0.8 * ssim.mean() + 0.2 * ssim.min())
            + 0.12 * background.mean()
            + 4.0 * peak_penalty
        )


def power(amplitude: Tensor, dark: Tensor) -> tuple[Tensor, Tensor]:
    intensity = amplitude.square()
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-8)
    return (intensity * dark).sum(dim=(-2, -1)) / total, intensity.amax((-2, -1)) / total


@dataclass(frozen=True)
class Candidate:
    phase: Tensor
    metrics: dict
    start: str
    ssim_weight: float | None
    step: int
    mean_background: float
    max_peak: float

    @property
    def score(self) -> float:
        return 0.8 * (self.metrics["mean_pcc"] + self.metrics["mean_ssim"]) + 0.2 * (
            self.metrics["worst_pcc"] + self.metrics["worst_ssim"]
        )

    @property
    def pcc(self) -> tuple[float, float]:
        return self.metrics["mean_pcc"], self.metrics["worst_pcc"]

    def report(self) -> dict:
        return {
            "start": self.start,
            "ssim_weight": self.ssim_weight,
            "step": self.step,
            "score": self.score,
            "mean_background_power_fraction": self.mean_background,
            "max_peak_power_fraction": self.max_peak,
            "metrics": self.metrics,
        }


def search(
    target: Tensor,
    optics: FresnelPropagator,
    starts: list[tuple[str, Tensor]],
    *,
    steps: int,
    check_every: int,
    ssim_weights: tuple[float, ...],
    pcc_reference: str | None = None,
) -> tuple[dict[str, Candidate], list[dict]]:
    """Select from initial phases and checkpoints scored on exact export pixels."""
    objective = QualityObjective(target)
    winners: dict[str, Candidate] = {}
    report = []

    @torch.no_grad()
    def candidate(phase, amplitude, pixels, name, weight, step):
        background, peak = power(amplitude, objective.dark)
        return Candidate(
            phase,
            evaluate(pixels, target, optics.distances_m),
            name,
            weight,
            step,
            background.mean().item(),
            peak.max().item(),
        )

    initial = []
    for name, phase in starts:
        quantized = quantize_phase(phase)
        amplitude = optics.amplitude(quantized)
        pixels = normalize_amplitude(amplitude).mul(255).round().div(255)
        initial.append(candidate(quantized, amplitude, pixels, name, None, 0))
    reference = next((item for item in initial if item.start == pcc_reference), None)

    def consider(item: Candidate) -> None:
        if "balanced" not in winners or item.score > winners["balanced"].score:
            winners["balanced"] = item
        if reference is not None and all(a >= b for a, b in zip(item.pcc, reference.pcc)):
            if "pcc-preserved" not in winners or item.score > winners["pcc-preserved"].score:
                winners["pcc-preserved"] = item

    for item in initial:
        consider(item)
        report.append(item.report())

    for item in initial:
        for weight in ssim_weights:
            phase = item.phase.clone().requires_grad_()
            optimizer = torch.optim.Adam([phase], lr=0.04)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.002)
            best = item
            for step in range(steps + 1):
                quantized = quantize_phase(phase.detach())
                amplitude = optics.amplitude(quantized + (phase - phase.detach()))
                image = normalize_amplitude(amplitude)
                pixels = image.detach().mul(255).round().div(255)
                if step and (step % check_every == 0 or step == steps):
                    current = candidate(quantized, amplitude, pixels, item.start, weight, step)
                    consider(current)
                    if current.score > best.score:
                        best = current
                if step < steps:
                    loss = objective(amplitude, pixels + (image - image.detach()), weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            report.append({"start": item.start, "ssim_weight": weight, "best": best.report()})
            print(f"{item.start}, SSIM weight {weight:g}: score {best.score:.4f}", flush=True)
    return winners, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs=4, type=Path, help="PNGs in ascending distance order")
    parser.add_argument("--phase", action="append", type=Path, default=[], help="Starting DOE PNG")
    parser.add_argument("--pcc-reference", type=Path, help="Extra DOE that sets both PCC floors")
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500, help="Adam steps per start and weight")
    parser.add_argument("--check-every", type=int, default=10, help="Interval for exact scoring")
    parser.add_argument("--ssim-weights", type=float, nargs="+", default=(0.5, 2.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.random_starts < 0 or args.steps < 1 or args.check_every < 1:
        parser.error("Expected nonnegative random starts and positive steps and check interval")
    if any(not isfinite(weight) or weight < 0 for weight in args.ssim_weights):
        parser.error("SSIM weights must be finite and nonnegative")
    if args.output.exists():
        parser.error("Output directory already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    torch.set_num_threads(2)
    device = torch.device(args.device)
    optics = FresnelPropagator().to(device)
    reference = args.pcc_reference.resolve() if args.pcc_reference is not None else None
    phases = dict.fromkeys(path.resolve() for path in args.phase)
    if reference is not None:
        phases[reference] = None
    try:
        target = torch.stack([png_to_gray(path.read_bytes()) for path in args.targets]).to(device)
        starts = [(str(path), png_to_phase(path.read_bytes()).to(device)) for path in phases]
        rng = torch.Generator(device=device).manual_seed(args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    for index in range(args.random_starts):
        phase = torch.rand((optics.size, optics.size), generator=rng, device=device) * (2 * pi) - pi
        starts.append((f"random-{index + 1}", phase))
    if not starts:
        parser.error("Provide a DOE PNG or use at least one random start")

    started = perf_counter()
    winners, trials = search(
        target,
        optics,
        starts,
        steps=args.steps,
        check_every=args.check_every,
        ssim_weights=tuple(args.ssim_weights),
        pcc_reference=str(reference) if reference else None,
    )
    seconds = perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=False)
    for name, item in winners.items():
        folder = args.output / name
        folder.mkdir()
        phase_png, reconstructions, metrics = export_phase(item.phase, target, optics)
        (folder / "doe.png").write_bytes(phase_png)
        for index, png in enumerate(reconstructions, 1):
            (folder / f"reconstruction-{index}.png").write_bytes(png)
        (folder / "metrics.json").write_text(
            json.dumps({**item.report(), "metrics": metrics}, indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "steps_per_trial": args.steps,
        "total_steps": args.steps * len(starts) * len(args.ssim_weights),
        "check_every": args.check_every,
        "seed": args.seed,
        "device": args.device,
        "seconds": seconds,
        "targets": [str(path.resolve()) for path in args.targets],
        "pcc_reference": str(reference) if reference else None,
        "trials": trials,
        "winners": {name: item.report() for name, item in winners.items()},
    }
    (args.output / "search.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
