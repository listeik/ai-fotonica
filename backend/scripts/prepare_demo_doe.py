"""Optimize teacher DOEs for the exact four presets served by the application."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from doe.loss import normalize_amplitude
from doe.models import DEFAULT_CHECKPOINT, load_model
from doe.physics import FresnelPropagator, gray_to_png, png_to_gray, png_to_phase, quantize_phase
from doe.quality import QualityObjective
from doe.synthesis import export_phase
from doe.training.dataset import DATASET_VERSION, DEMO_NAMES, synthetic_targets


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rank(metrics):
    return (
        min(metrics["worst_pcc"], metrics["worst_ssim"]),
        metrics["mean_pcc"] + metrics["mean_ssim"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--random-starts", type=int, default=3)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--lbfgs-rounds", type=int, default=15)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--continue-from", type=Path, help="previous output for the same presets")
    args = parser.parse_args()
    if args.steps < 1 or args.random_starts < 0 or args.check_every < 1 or args.lbfgs_rounds < 0:
        parser.error(
            "steps and check interval must be positive, start and round counts nonnegative"
        )
    if args.output.exists():
        parser.error("choose a new output directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    optics = FresnelPropagator()
    target = torch.stack([synthetic_targets(i, split="demo") for i in range(len(DEMO_NAMES))])
    args.output.mkdir(parents=True)
    records = []
    phases, samples, names = [], [], []

    @torch.no_grad()
    def add_initial(phase, index, name):
        phases.append(quantize_phase(phase).detach())
        samples.append(index)
        names.append(name)

    model, version = load_model(args.checkpoint)
    with torch.no_grad():
        neural = model(target).squeeze(1)
    for index, name in enumerate(DEMO_NAMES):
        folder = args.output / f"demo-{index}"
        folder.mkdir()
        hashes = {}
        for plane, image in enumerate(target[index], 1):
            data = gray_to_png(image)
            filename = f"target-{plane}.png"
            (folder / filename).write_bytes(data)
            hashes[filename] = hashlib.sha256(data).hexdigest()
        records.append(
            {"id": index, "name": name, "directory": folder.name, "target_sha256": hashes}
        )
        add_initial(neural[index], index, "neural")
        if args.continue_from:
            previous = args.continue_from / folder.name
            if any(
                (previous / file).read_bytes() != (folder / file).read_bytes() for file in hashes
            ):
                raise ValueError("previous run uses different targets")
            add_initial(png_to_phase((previous / "doe.png").read_bytes()), index, "previous-best")
        rng = torch.Generator().manual_seed(args.seed + index)
        for start in range(args.random_starts + 1):
            phase = torch.rand((optics.size, optics.size), generator=rng) * (2 * math.pi) - math.pi
            if start == 0:
                with torch.no_grad():
                    desired = target[index] + 0.005
                    desired = desired / desired.square().sum((-2, -1), keepdim=True).sqrt()
                    desired = desired * optics._gain.float() * optics.size**2
                    for _ in range(100):
                        fields = optics(torch.polar(torch.ones_like(phase), phase))
                        phase = (
                            optics.inverse_planes(torch.polar(desired, fields.angle()))
                            .mean(0)
                            .angle()
                        )
            add_initial(phase, index, "projection-100" if start == 0 else f"random-{start}")

    phase = torch.stack(phases).requires_grad_()
    indices = torch.tensor(samples)
    targets = target[indices]
    objective = QualityObjective(targets)
    optimizer = torch.optim.Adam([phase], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=0.0005)
    best = [None] * len(DEMO_NAMES)
    history = []
    started = perf_counter()
    for step in range(args.steps + 1):
        quantized = quantize_phase(phase.detach())
        amplitude = optics.amplitude(quantized + (phase - phase.detach()))
        image = normalize_amplitude(amplitude)
        pixels = image.detach().mul(255).round().div(255)
        pcc, ssim = objective.scores(pixels + (image - image.detach()))
        if step % args.check_every == 0 or step == args.steps:
            with torch.no_grad():
                scores = torch.cat((pcc, ssim), dim=-1)
                minimum, mean = scores.amin(-1), scores.mean(-1)
                for index in range(len(DEMO_NAMES)):
                    eligible = [i for i, sample in enumerate(samples) if sample == index]
                    winner = max(eligible, key=lambda i: (minimum[i].item(), mean[i].item()))
                    phase_png, pngs, metrics = export_phase(
                        quantized[winner], target[index], optics
                    )
                    if best[index] is None or rank(metrics) > rank(best[index]["metrics"]):
                        row = {
                            "metrics": metrics,
                            "step": step,
                            "start": names[winner],
                            "qualified_above_0_9": rank(metrics)[0] > 0.9,
                        }
                        best[index] = row
                        folder = args.output / records[index]["directory"]
                        (folder / "doe.png").write_bytes(phase_png)
                        for plane, data in enumerate(pngs, 1):
                            (folder / f"reconstruction-{plane}.png").write_bytes(data)
                        write_json(folder / "metrics.json", row)
                row = {
                    "step": step,
                    "seconds": perf_counter() - started,
                    "worst_of_eight": [rank(item["metrics"])[0] for item in best],
                }
                history.append(row)
                write_json(args.output / "history.json", history)
                print(json.dumps(row), flush=True)
        if step == args.steps:
            break
        scores = torch.cat((pcc, ssim), dim=-1)
        soft_worst = -0.03 * torch.logsumexp(-scores / 0.03, dim=-1)
        loss = (-0.5 * scores.mean(-1) - 0.5 * soft_worst).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    lbfgs_evaluations = 0
    for index in range(len(DEMO_NAMES)):
        folder = args.output / records[index]["directory"]
        phase = png_to_phase((folder / "doe.png").read_bytes()).requires_grad_()
        objective = QualityObjective(target[index])
        optimizer = torch.optim.LBFGS(
            [phase], lr=0.8, max_iter=20, history_size=30, line_search_fn="strong_wolfe"
        )

        def closure():
            nonlocal lbfgs_evaluations
            optimizer.zero_grad(set_to_none=True)
            image = normalize_amplitude(optics.amplitude(phase))
            pcc, ssim = objective.scores(image)
            scores = torch.cat((pcc, ssim))
            soft_worst = -0.03 * torch.logsumexp(-scores / 0.03, dim=-1)
            loss = -0.5 * scores.mean() - 0.5 * soft_worst
            loss.backward()
            lbfgs_evaluations += 1
            return loss

        for round_index in range(args.lbfgs_rounds):
            optimizer.step(closure)
            phase_png, pngs, metrics = export_phase(phase, target[index], optics)
            if rank(metrics) > rank(best[index]["metrics"]):
                best[index] = {
                    "metrics": metrics,
                    "step": round_index + 1,
                    "start": "LBFGS-from-best",
                    "qualified_above_0_9": rank(metrics)[0] > 0.9,
                }
                (folder / "doe.png").write_bytes(phase_png)
                for plane, data in enumerate(pngs, 1):
                    (folder / f"reconstruction-{plane}.png").write_bytes(data)
                write_json(folder / "metrics.json", best[index])
        print(
            json.dumps({"demo": index, "stage": "LBFGS", "best": rank(best[index]["metrics"])}),
            flush=True,
        )

    for record, result in zip(records, best, strict=True):
        record.update(result)
    manifest = {
        "dataset": DATASET_VERSION,
        "split": "demo",
        "seed": args.seed,
        "usage": "training-only teacher examples; these presets must not be validation/test data",
        "optics": {
            "size": optics.size,
            "wavelength_m": optics.wavelength_m,
            "pitch_m": optics.pitch_m,
            "distances_m": optics.distances_m,
        },
        "metrics": "PCC and global SSIM, population moments, c1=0.01**2, c2=0.03**2; 8-bit export",
        "initial_model": version,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "total_updates": args.steps * len(samples),
        "lbfgs_closure_evaluations": lbfgs_evaluations,
        "seconds": perf_counter() - started,
        "qualified_sets": sum(record["qualified_above_0_9"] for record in records),
        "records": records,
    }
    write_json(args.output / "manifest.json", manifest)
    torch.save(
        {
            "format": "demo-doe-teachers-v1",
            "targets_uint8": target.mul(255).round().to(torch.uint8),
            "phase_uint8": torch.stack(
                [
                    png_to_gray((args.output / record["directory"] / "doe.png").read_bytes())
                    for record in records
                ]
            )
            .mul(255)
            .round()
            .to(torch.uint8),
            "qualified_above_0_9": torch.tensor(
                [record["qualified_above_0_9"] for record in records]
            ),
            "metrics": [record["metrics"] for record in records],
            "demo_ids": torch.arange(len(records)),
            "split": "training-only",
        },
        args.output / "training-pairs.pt",
    )


if __name__ == "__main__":
    main()
