"""Train a separate GPU DOE checkpoint, then compare fixed held-out image sets."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.physics import FresnelPropagator, quantize_phase
from people_doe_reference import PhotoObjective, normalize_amplitude
from PIL import Image, ImageDraw
from prepare_people_doe import measure, require, save_json, sha256
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]


def load_png(path, checksum, *, pixel_checksum=False):
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as image:
        require(image.mode == "L" and image.size == (128, 128), f"invalid target: {path}")
        pixels = np.array(image)
    require(
        hashlib.sha256(pixels.tobytes() if pixel_checksum else data).hexdigest() == checksum,
        f"changed target: {path}",
    )
    return pixels


def prepare_data(broad, people, output, config):
    """Verify selected target bytes once and cache uint8 arrays, never DOE labels."""
    cache = output / "cache"
    cache.mkdir()
    rng = np.random.default_rng(config["seed"])
    inventory = {split: {} for split in ("train", "validation", "test", "ood")}
    with gzip.open(broad / "manifest.csv.gz", "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            inventory[row["split"]].setdefault(int(row["set"]), []).append(row)
    chosen = {}
    arrays = {}
    all_hashes = set()
    for split, count in (
        ("train", config["broad_train_count"]),
        ("validation", config["broad_validation_count"]),
        ("test", config["broad_test_count"]),
        ("ood", config["ood_count"]),
    ):
        indices = sorted(rng.choice(sorted(inventory[split]), count, replace=False).tolist())
        chosen[split] = indices
        path = cache / f"broad-{split}.npy"
        array = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.uint8, shape=(count, 4, 128, 128)
        )
        for item, index in enumerate(indices):
            rows = sorted(inventory[split][index], key=lambda row: int(row["plane"]))
            require(len(rows) == 4, "expected four target planes")
            for plane, row in enumerate(rows):
                require(row["sha256"] not in all_hashes, "duplicate broad target")
                all_hashes.add(row["sha256"])
                filename = f"{index:05d}_{plane + 1}.png"
                array[item, plane] = load_png(
                    broad / split / f"{index // 100:03d}" / filename,
                    row["sha256"],
                    pixel_checksum=True,
                )
            if (item + 1) % 2000 == 0:
                print(json.dumps({"cache": split, "sets": item + 1}), flush=True)
        array.flush()
        arrays[f"broad_{split}"] = np.load(path, mmap_mode="r")
    manifest = json.loads((people / "manifest.json").read_text(encoding="utf-8"))
    subjects = {}
    by_subject = {}
    face_paths = {}
    for split in ("train", "validation", "test"):
        records = [row for row in manifest["records"] if row["split"] == split]
        faces = np.zeros((len(records), 4, 128, 128), dtype=np.uint8)
        face_paths[split] = [row["directory"] for row in records]
        for index, record in enumerate(records):
            for plane, source in enumerate(record["sources"]):
                subject = source["subject_id"]
                require(subjects.setdefault(subject, split) == split, "portrait subject leakage")
                image = load_png(
                    people / record["directory"] / source["file"], source["target_sha256"]
                )
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                require(pixel_hash not in all_hashes, "repeated photo target pixels")
                all_hashes.add(pixel_hash)
                faces[index, plane] = image
                if split == "train":
                    by_subject.setdefault(subject, {})[source["view_id"]] = image
        arrays[f"people_{split}"] = faces
    require(
        len(by_subject) == 32 and all(len(views) == 10 for views in by_subject.values()),
        "portrait bank incomplete",
    )
    arrays["face_bank"] = np.stack(
        [
            np.stack([views[view] for view in sorted(views)])
            for _, views in sorted(by_subject.items())
        ]
    )
    provenance = {
        "broad_root": str(broad.resolve()),
        "broad_manifest_sha256": sha256(broad / "manifest.csv.gz"),
        "people_root": str(people.resolve()),
        "people_manifest_sha256": sha256(people / "manifest.json"),
        "broad_indices": chosen,
        "portrait_directories": face_paths,
        "portrait_subjects_by_split": {
            split: sorted(s for s, part in subjects.items() if part == split)
            for split in ("train", "validation", "test")
        },
        "portrait_training_augmentation": "four distinct training subjects, one random view each, random focal-plane assignment; 25% diverse training replay",
        "label_usage": "target images only; no iterative phase labels used",
        "unique_selected_target_pixels": len(all_hashes),
        "duplicate_selected_targets": 0,
    }
    save_json(output / "data.json", provenance)
    return arrays, provenance


class TrainingImages(Dataset):
    def __init__(self, arrays, config, stage, epoch):
        self.arrays, self.config, self.stage, self.epoch = arrays, config, stage, epoch

    def __len__(self):
        return (
            len(self.arrays["broad_train"])
            if self.stage == "pretrain"
            else self.config["finetune_samples"]
        )

    def __getitem__(self, index):
        rng = np.random.default_rng(self.config["seed"] + 100003 * self.epoch + index)
        broad = self.arrays["broad_train"]
        if self.stage == "pretrain":
            image = broad[index][rng.permutation(4)]
        elif index % 4 == 0:
            image = broad[rng.integers(len(broad))][rng.permutation(4)]
        else:
            bank = self.arrays["face_bank"]
            subjects = rng.choice(len(bank), 4, replace=False)
            image = bank[subjects, rng.integers(10, size=4)]
        return torch.from_numpy(np.array(image, copy=True))


def optical_loss(phase, target, physics):
    phase = quantize_phase(phase.detach()) + (phase - phase.detach())
    amplitude = physics.amplitude(phase)
    image = normalize_amplitude(amplitude)
    pixels = image.detach().mul(255).round().div(255) + (image - image.detach())
    pcc, ssim = PhotoObjective(target).scores(pixels)
    loss = 1 - pcc.mean() + 1 - ssim.mean()
    loss += 0.25 * (2 - pcc.amin(-1).mean() - ssim.amin(-1).mean())
    loss += 0.1 * (image - target).square().mean()
    power = amplitude.square()
    leakage = (power * (target <= 2 / 255)).sum((-2, -1)) / power.sum((-2, -1)).clamp_min(1e-8)
    return loss + 0.05 * leakage.mean()


@torch.inference_mode()
def validate(model, arrays, physics, device, batch_size):
    model.eval()
    result = {}
    for name in ("broad_validation", "people_validation"):
        totals = []
        for start in range(0, len(arrays[name]), batch_size):
            target = (
                torch.from_numpy(np.array(arrays[name][start : start + batch_size], copy=True))
                .to(device)
                .float()
                / 255
            )
            phase = quantize_phase(model(target).squeeze(1))
            pixels = normalize_amplitude(physics.amplitude(phase)).mul(255).round().div(255)
            pcc, ssim = PhotoObjective(target).scores(pixels)
            totals.append(
                torch.stack((pcc.mean(-1), ssim.mean(-1), pcc.amin(-1), ssim.amin(-1)), -1).cpu()
            )
        values = torch.cat(totals).mean(0).tolist()
        result[name] = dict(
            zip(
                ("mean_pcc", "mean_global_ssim", "mean_worst_pcc", "mean_worst_global_ssim"),
                values,
                strict=True,
            )
        )
    # Same fixed criterion for both stages; hold out test/OOD until training ends.
    result["selection"] = sum(
        weight * (result[name]["mean_pcc"] + result[name]["mean_global_ssim"])
        for name, weight in (("broad_validation", 0.3), ("people_validation", 0.7))
    )
    return result


def save_checkpoint(model, output, config, epoch, step, validation, stage):
    state = {
        "format_version": CHECKPOINT_FORMAT,
        "architecture": model.architecture,
        "width": config["width"],
        "projection_steps": config["projection_steps"],
        "amplitude_floor": config["amplitude_floor"],
        "config": config,
        "epoch": epoch,
        "step": step,
        "stage": stage,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "validation": validation,
        "dataset": "doe-dataset-v1 + people-doe-100-v1, disjoint source splits",
    }
    temporary = output.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(output)


def train_stage(model, arrays, config, stage, output, device, initial_step):
    physics = FresnelPropagator().to(device)
    epochs = config[f"{stage}_epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config[f"{stage}_lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=config[f"{stage}_lr"] * 0.1
    )
    best_score, stale, step = -math.inf, 0, initial_step
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        dataset = TrainingImages(arrays, config, stage, epoch)
        order = torch.Generator().manual_seed(config["seed"] + epoch)
        batches = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, generator=order
        )
        total = torch.zeros((), device=device)
        for index, batch in enumerate(batches):
            target = batch.to(device).float() / 255
            optimizer.zero_grad(set_to_none=True)
            loss = optical_loss(model(target).squeeze(1), target, physics)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(batch)
            step += 1
            if (index + 1) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "epoch": epoch,
                            "batch": index + 1,
                            "batches": len(batches),
                            "loss": loss.item(),
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        validation = validate(model, arrays, physics, device, config["batch_size"])
        record = {
            "stage": stage,
            "epoch": epoch,
            "step": step,
            "train_loss": total.item() / len(dataset),
            "seconds": time.perf_counter() - started,
            "validation": validation,
        }
        history.append(record)
        if validation["selection"] > best_score:
            best_score, stale = validation["selection"], 0
            save_checkpoint(
                model, output / f"{stage}-best.pt", config, epoch, step, validation, stage
            )
        else:
            stale += 1
        save_checkpoint(model, output / f"{stage}-last.pt", config, epoch, step, validation, stage)
        save_json(output / f"{stage}-history.json", history)
        print(json.dumps(record), flush=True)
        if stage == "finetune" and stale >= config["patience"]:
            break
    model, _ = load_model(output / f"{stage}-best.pt", device)
    return model, step


@torch.inference_mode()
def evaluate_models(output, arrays, config, provenance, device):
    paths = {
        "new": output / "best.pt",
        "previous_fourier": ROOT / "artifacts/doe-residual-fourier.pt",
        "previous_projection": ROOT / "artifacts/doe-unet.pt",
    }
    physics = FresnelPropagator()
    report = {
        "protocol": "all models infer on CUDA; phase and reconstructions round-trip through uint8 PNG, then metrics on CPU; SSIM is global plus separate 7x7 local SSIM",
        "device": str(device),
        "models": {},
    }
    for name, path in paths.items():
        model, version = load_model(path, device)
        results = {}
        for split in ("people_test", "broad_test", "broad_ood"):
            records = []
            for start in range(0, len(arrays[split]), config["batch_size"]):
                target_cpu = (
                    torch.from_numpy(
                        np.array(arrays[split][start : start + config["batch_size"]], copy=True)
                    ).float()
                    / 255
                )
                phases = model(target_cpu.to(device)).squeeze(1).cpu()
                for offset, (phase, target) in enumerate(zip(phases, target_cpu, strict=True)):
                    phase_png, images, metrics = measure(phase, target, physics)
                    index = start + offset
                    records.append({"index": index, "metrics": metrics})
                    if split == "people_test":
                        folder = output / "predictions" / name / f"{index:03d}"
                        folder.mkdir(parents=True)
                        (folder / "doe.png").write_bytes(phase_png)
                        for plane, image in enumerate(images, 1):
                            (folder / f"reconstruction-{plane}.png").write_bytes(image)
                        save_json(folder / "metrics.json", metrics)
            keys = (
                "mean_pcc",
                "mean_ssim",
                "worst_pcc",
                "worst_ssim",
                "mean_local_ssim",
                "worst_local_ssim",
            )
            summary = {
                key: float(np.mean([row["metrics"][key] for row in records])) for key in keys
            }
            summary["samples"] = len(records)
            summary["all_four_planes_pcc_and_global_ssim_above_0_9"] = sum(
                min(row["metrics"]["worst_pcc"], row["metrics"]["worst_ssim"]) > 0.9
                for row in records
            )
            results[split] = {"summary": summary, "records": records}
            print(json.dumps({"evaluation": name, "split": split, **summary}), flush=True)
        report["models"][name] = {
            "checkpoint": str(path.resolve()),
            "sha256": sha256(path),
            "version": version,
            "results": results,
        }
        del model
    report["portrait_test_directories"] = provenance["portrait_directories"]["test"]
    save_json(output / "evaluation.json", report)
    make_preview(output, provenance)
    return report


def make_preview(output, provenance):
    people = Path(provenance["people_root"])
    sheet = Image.new("RGB", (16 * 128, 3 * 154), "#18212b")
    draw = ImageDraw.Draw(sheet)
    for row in range(3):
        groups = [people / provenance["portrait_directories"]["test"][row]]
        groups += [
            output / "predictions" / name / f"{row:03d}"
            for name in ("previous_fourier", "previous_projection", "new")
        ]
        for group, folder in enumerate(groups):
            for plane in range(1, 5):
                filename = f"{'target' if group == 0 else 'reconstruction'}-{plane}.png"
                with Image.open(folder / filename) as image:
                    sheet.paste(image, ((group * 4 + plane - 1) * 128, row * 154))
        draw.text(
            (8, row * 154 + 132),
            "targets 1-4 | previous Fourier 1-4 | previous Projection 1-4 | new GPU model 1-4",
            fill="white",
        )
    sheet.save(output / "preview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/photo-model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/photo-model-v1")
    parser.add_argument("--broad", type=Path, default=ROOT / "runs/doe-dataset-v1")
    parser.add_argument("--people", type=Path, default=ROOT / "runs/people-doe-100-v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    require(not args.output.exists(), "choose a new output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    started = time.perf_counter()
    environment = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "selection": "0.7 portrait validation PCC+global SSIM + 0.3 broad validation PCC+global SSIM",
    }
    save_json(args.output / "config.json", config)
    save_json(args.output / "environment.json", environment)
    print(json.dumps(environment), flush=True)
    arrays, provenance = prepare_data(args.broad, args.people, args.output, config)
    model = FresnelUNet(
        config["width"], config["projection_steps"], amplitude_floor=config["amplitude_floor"]
    ).to(device)
    initial = validate(model, arrays, FresnelPropagator().to(device), device, config["batch_size"])
    save_json(args.output / "initial-validation.json", initial)
    model, step = train_stage(model, arrays, config, "pretrain", args.output, device, 0)
    model, step = train_stage(model, arrays, config, "finetune", args.output, device, step)
    candidates = [args.output / f"{stage}-best.pt" for stage in ("pretrain", "finetune")]
    selected = max(
        candidates,
        key=lambda path: torch.load(path, map_location="cpu", weights_only=True)["validation"][
            "selection"
        ],
    )
    shutil.copyfile(selected, args.output / "best.pt")
    result = {
        "selected": selected.name,
        "optimizer_steps": step,
        "seconds_including_data_and_validation": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256(args.output / "best.pt"),
        "initial_validation": initial,
        "peak_cuda_memory_MB": torch.cuda.max_memory_allocated() / 1024**2
        if device.type == "cuda"
        else None,
    }
    save_json(args.output / "training.json", result)
    del model
    if not args.skip_evaluation:
        evaluate_models(args.output, arrays, config, provenance, device)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
