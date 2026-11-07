"""Prepare training-only natural-image DOE teachers with a bounded phase optimizer."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from doe.loss import global_ssim, normalize_amplitude
from doe.models import load_model
from doe.physics import FresnelPropagator, phase_to_png, png_to_phase, quantize_phase
from doe.training.carrier_study import optical_objective, pcc, reconstruction, windowed_ssim
from doe.training.multiplane_study import digest, environment, write_json


def pixel_hash(image):
    return hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()


def targets_from_training(data_path, count, seed):
    manifest_path = data_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path, test_path = data_path / "train-val.pt", data_path / "test.pt"
    if digest(train_path) != manifest["train_val_sha256"]:
        raise ValueError("Training cache SHA-256 mismatch")
    if digest(test_path) != manifest["test_sha256"]:
        raise ValueError("Held-out cache SHA-256 mismatch")
    training = torch.load(train_path, map_location="cpu", weights_only=True)
    bank = training["natural_bank"]
    origins = manifest["natural_bank"]["records"]
    if bank.dtype != torch.uint8 or tuple(bank.shape[1:]) != (128, 128):
        raise ValueError("Expected uint8 natural training images of size 128x128")
    if len(bank) != len(origins) or len(bank) < 4:
        raise ValueError("Training bank and source records disagree")
    forbidden = set()
    for stacks in training["validation"].values():
        forbidden.update(pixel_hash(plane) for stack in stacks for plane in stack)
    heldout = torch.load(test_path, map_location="cpu", weights_only=True)
    for stacks in heldout["test"].values():
        forbidden.update(pixel_hash(plane) for stack in stacks for plane in stack)
    del heldout
    for image, record in zip(bank, origins, strict=True):
        checksum = pixel_hash(image)
        if checksum != record["pixel_sha256"] or checksum in forbidden:
            raise ValueError("Natural train image changed or overlaps a held-out image")
    existing = {pixel_hash(stack) for stack in training["teacher_targets"]}
    rng = np.random.default_rng(seed)
    targets, records = [], []
    attempts = 0
    while len(targets) < count:
        attempts += 1
        if attempts > 100 * count:
            raise ValueError("Could not sample enough distinct training quartets")
        indices = rng.choice(len(bank), size=4, replace=False)
        sources = [origins[int(index)] for index in indices]
        if len({(row["source"], str(row["origin"])) for row in sources}) != 4:
            continue
        target = bank[indices].clone()
        checksum = pixel_hash(target)
        if checksum in existing:
            continue
        existing.add(checksum)
        targets.append(target)
        records.append(
            {
                "split": "train",
                "bank_indices": indices.tolist(),
                "sources": sources,
                "target_sha256": checksum,
            }
        )
    return (
        torch.stack(targets),
        records,
        {
            "prepared_manifest_sha256": digest(manifest_path),
            "train_val_sha256": digest(train_path),
            "hash_only_heldout_sha256": digest(test_path),
            "validation_or_test_pixel_overlap": 0,
            "source_selection": "Four distinct source origins from verified natural training bank",
        },
    )


@torch.no_grad()
def metrics(phase, targets, optics):
    _, _, pixels = reconstruction(phase, targets, optics, gradient=False)
    correlations, global_scores = pcc(pixels, targets), global_ssim(pixels, targets)
    local_scores = windowed_ssim(pixels, targets)
    values = torch.stack((correlations, global_scores, local_scores), -1)
    score = (
        0.35 * correlations.mean(1)
        + 0.2 * global_scores.mean(1)
        + 0.2 * local_scores.mean(1)
        + 0.15 * correlations.amin(1)
        + 0.1 * global_scores.amin(1)
    )
    return score, values


def optimize(initial, targets, optics, steps, interval, lr):
    phase = initial.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([phase], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, steps), eta_min=lr / 100
    )
    best_score, _ = metrics(initial, targets, optics)
    best_phase = quantize_phase(initial.detach())
    best_steps = torch.zeros(len(targets), dtype=torch.int64, device=targets.device)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        amplitude, image, pixels = reconstruction(phase, targets, optics, gradient=True)
        loss, _ = optical_objective(amplitude, image, pixels, targets, "legacy")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite phase optimization objective")
        loss.backward()
        if phase.grad is None or not torch.isfinite(phase.grad).all():
            raise FloatingPointError("Non-finite phase optimization gradient")
        optimizer.step()
        schedule.step()
        if step % interval == 0 or step == steps:
            score, _ = metrics(phase, targets, optics)
            improved = score > best_score
            best_phase = torch.where(
                improved[:, None, None], quantize_phase(phase.detach()), best_phase
            )
            best_steps = torch.where(improved, step, best_steps)
            best_score = torch.maximum(best_score, score)
    return best_phase.cpu(), best_steps.cpu()


@torch.no_grad()
def canonical(phase, targets, optics):
    pngs = [phase_to_png(item.cpu()) for item in phase]
    decoded = torch.stack([png_to_phase(png) for png in pngs])
    amplitude = optics.amplitude(decoded)
    images = normalize_amplitude(amplitude).mul(255).round().to(torch.uint8)
    score, values = metrics(decoded, targets, optics)
    return decoded, images, score, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.count, args.batch, args.check_every) < 1 or args.steps < 0:
        parser.error("Counts and check interval must be positive, steps must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be positive and finite")
    if args.output.exists():
        parser.error("Choose a new teacher output directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    checkpoint_sha256 = digest(args.checkpoint)
    model, version = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    targets, records, provenance = targets_from_training(args.data, args.count, args.seed)
    if digest(args.checkpoint) != checkpoint_sha256:
        raise ValueError("Initialization checkpoint changed during preparation")
    optics = FresnelPropagator().to(device)
    cpu_optics = FresnelPropagator()
    args.output.mkdir(parents=True)
    chunks = {
        key: []
        for key in (
            "phases",
            "reconstructions",
            "initial_phases",
            "initial_metrics",
            "best_metrics",
            "initial_scores",
            "best_scores",
            "best_steps",
        )
    }
    started = perf_counter()
    for start in range(0, len(targets), args.batch):
        target_cpu = targets[start : start + args.batch].float().div(255)
        target = target_cpu.to(device)
        with torch.no_grad():
            initial = model(target).squeeze(1).detach()
        candidate, best_steps = optimize(
            initial, target, optics, args.steps, args.check_every, args.lr
        )
        before = canonical(initial.cpu(), target_cpu, cpu_optics)
        after = canonical(candidate, target_cpu, cpu_optics)
        improved = after[2] > before[2]
        retained = torch.where(improved[:, None, None], after[0], before[0])
        best = canonical(retained, target_cpu, cpu_optics)
        if not torch.all(best[2] >= before[2]):
            raise ValueError("Retained teacher is worse than its initialization")
        chunks["phases"].append(best[0])
        chunks["reconstructions"].append(best[1])
        chunks["initial_phases"].append(before[0])
        chunks["initial_metrics"].append(before[3])
        chunks["best_metrics"].append(best[3])
        chunks["initial_scores"].append(before[2])
        chunks["best_scores"].append(best[2])
        chunks["best_steps"].append(torch.where(improved, best_steps, 0))
        print(
            f"teachers={start + len(target)}/{len(targets)} improved={int(improved.sum())}/{len(target)} "
            f"score={before[2].mean():.6f}->{best[2].mean():.6f} seconds={perf_counter() - started:.1f}",
            flush=True,
        )
    payload = {"targets": targets, **{key: torch.cat(parts) for key, parts in chunks.items()}}
    path = args.output / "teachers.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    if not all(torch.equal(value, restored[key]) for key, value in payload.items()):
        raise ValueError("Teacher tensor round-trip failed")
    for start in range(0, len(targets), args.batch):
        raw = restored["targets"][start : start + args.batch]
        actual = canonical(
            restored["phases"][start : start + args.batch], raw.float() / 255, cpu_optics
        )
        if not torch.equal(actual[0], restored["phases"][start : start + args.batch]):
            raise ValueError("Saved phases are not exact exported PNG levels")
        if not torch.equal(actual[1], restored["reconstructions"][start : start + args.batch]):
            raise ValueError("Saved phases and reconstructions disagree")
    write_json(
        args.output / "manifest.json",
        {
            "version": "carrier-natural-teachers-v1",
            "teachers_sha256": digest(path),
            "checkpoint_sha256": checkpoint_sha256,
            "initialized_from": version,
            "train_val_sha256": provenance["train_val_sha256"],
            "provenance": provenance,
            "records": records,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "metric_axis": ["PCC", "global SSIM", "windowed SSIM 7x7"],
            "selection": "0.35 mean PCC + 0.20 mean global SSIM + 0.20 mean windowed SSIM + 0.15 worst PCC + 0.10 worst global SSIM",
            "objective": "carrier_study.optical_objective legacy, phase-only Adam, exact phase/pixel STE",
            "improved_count": int((restored["best_scores"] > restored["initial_scores"]).sum()),
            "initial_mean_score": float(restored["initial_scores"].mean()),
            "best_mean_score": float(restored["best_scores"].mean()),
            "environment": environment(device),
            "seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
