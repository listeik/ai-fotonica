"""Synthesize a DOE and export the official PNGs without running a server."""

import argparse
import json
from pathlib import Path

import torch

from doe.models import load_model
from doe.physics import FresnelPropagator, png_to_gray
from doe.synthesis import synthesize
from doe.training.dataset import DEMO_NAMES, synthetic_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="PNGs in ascending distance order")
    parser.add_argument("--demo", type=int, choices=range(len(DEMO_NAMES)), metavar="0–3")
    parser.add_argument("--method", choices=("classical", "neural", "hybrid"), default="hybrid")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/doe-unet.pt"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.demo is None and len(args.images) != 4) or (args.demo is not None and args.images):
        parser.error("provide four images or select --demo")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    propagator = FresnelPropagator().to(device)
    try:
        targets = (
            synthetic_targets(args.demo, split="demo")
            if args.demo is not None
            else torch.stack([png_to_gray(image.read_bytes()) for image in args.images])
        )
        model, version = (
            load_model(args.checkpoint, device) if args.method != "classical" else (None, None)
        )
        result = synthesize(
            targets,
            args.method,
            propagator,
            model=model,
            iterations=args.iterations,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "doe.png").write_bytes(result.phase_png)
    for index, png in enumerate(result.reconstruction_pngs, 1):
        (args.output / f"reconstruction-{index}.png").write_bytes(png)
    metadata = {
        "method": args.method,
        "model_version": version,
        "seconds": result.seconds,
        "iterations": result.iterations,
        "seed": args.seed,
        "device": str(device),
        "metrics": result.metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved DOE and four reconstructions to {args.output}")


if __name__ == "__main__":
    main()
