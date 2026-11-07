"""Measure a matched DOE study on frozen synthetic and external image stacks."""

import argparse
import csv
import gzip
import json
import struct
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image

from doe.loss import background_power_fraction, normalize_amplitude
from doe.metrics import evaluate
from doe.physics import FresnelPropagator, phase_to_png, png_to_gray, quantize_phase

from .dataset import DATASET_VERSION, SyntheticTargets
from .multiplane_study import (
    BASELINE,
    FOURIER,
    MODELS,
    digest,
    environment,
    load_study_model,
    synchronize,
    write_json,
)

EXTERNAL = {
    "mnist": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    ),
    "fashion": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/"
        "data/fashion/t10k-images-idx3-ubyte.gz",
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    ),
}
METRICS = ("pcc", "ssim", "leakage", "utility")


def external_targets(name: str, cache: Path, count: int, seed: int) -> tuple[torch.Tensor, dict]:
    url, checksum = EXTERNAL[name]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.gz"
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(response.read())
        if digest(temporary) != checksum:
            raise ValueError(f"SHA-256 mismatch for {url}")
        temporary.replace(path)
    if digest(path) != checksum:
        raise ValueError(f"SHA-256 mismatch for {path}")
    with gzip.open(path, "rb") as stream:
        magic, length, height, width = struct.unpack(">4I", stream.read(16))
        if (magic, height, width) != (2051, 28, 28):
            raise ValueError(f"Invalid IDX images in {path}")
        images = np.frombuffer(stream.read(), dtype=np.uint8).reshape(length, height, width)
    indices = np.random.default_rng(seed).choice(length, size=(count, 4), replace=False)
    targets = np.zeros((count, 4, 128, 128), dtype=np.uint8)
    for stack, planes in enumerate(indices):
        for plane, index in enumerate(planes):
            image = Image.fromarray(images[index]).resize((84, 84), Image.Resampling.BILINEAR)
            targets[stack, plane, 22:106, 22:106] = np.asarray(image)
    return torch.from_numpy(targets).float() / 255, {
        "url": url,
        "sha256": checksum,
        "test_indices_by_stack_and_plane": indices.tolist(),
        "preprocessing": "8-bit bilinear 28x28 to 84x84, centered zero padding to 128x128",
    }


@torch.inference_mode()
def measure(model, targets, physics, device, batch_size):
    rows = []
    for batch in targets.split(batch_size):
        target = batch.to(device)
        phase = quantize_phase(model(target).squeeze(1))
        amplitude = physics.amplitude(phase)
        leakage = background_power_fraction(amplitude, target).cpu().tolist()
        normalized = (255 * normalize_amplitude(amplitude)).round().cpu() / 255
        for reconstructed, expected, power in zip(normalized, batch, leakage, strict=True):
            planes = evaluate(reconstructed, expected, physics.distances_m)["planes"]
            for plane, fraction in zip(planes, power, strict=True):
                plane.update(leakage=fraction, utility=plane["pcc"] + plane["ssim"] - fraction)
            rows.append(planes)
    return rows


@torch.inference_mode()
def latencies(models, targets, device, settings):
    rng = np.random.default_rng(9201)
    names = list(models)
    targets = targets.to(device).split(1)
    times = {name: [] for name in names}
    warmup, repeats = settings["latency_warmup"], settings["latency_repeats"]
    for repeat in range(warmup + repeats):
        target = targets[repeat % len(targets)]
        for index in rng.permutation(len(names)):
            name = names[index]
            synchronize(device)
            start = perf_counter()
            quantize_phase(models[name](target).squeeze(1))
            synchronize(device)
            if repeat >= warmup:
                times[name].append(1000 * (perf_counter() - start))
    return times


def summarize(raw, times, settings):
    seeds = settings["seeds"]
    per_run = {
        name: {
            split: {
                metric: float(np.mean([[p[metric] for p in stack] for stack in stacks]))
                for metric in METRICS
            }
            for split, stacks in splits.items()
        }
        for name, splits in raw.items()
    }
    result = {
        "runs": per_run,
        "inference_ms_by_run": {
            key: {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}
            for key, values in times.items()
        },
        "matched": {},
    }
    for name in MODELS:
        summary = {
            split: {
                metric: float(np.mean([per_run[f"{name}-{seed}"][split][metric] for seed in seeds]))
                for metric in METRICS
            }
            for split in ("test", *EXTERNAL)
        }
        summary["external_minus_test"] = {
            split: {metric: summary[split][metric] - summary["test"][metric] for metric in METRICS}
            for split in EXTERNAL
        }
        pooled = [value for seed in seeds for value in times[f"{name}-{seed}"]]
        summary["inference_ms"] = {
            "p50": float(np.median(pooled)),
            "p95": float(np.percentile(pooled, 95)),
        }
        result["matched"][name] = summary
    return result


def paired_differences(raw, settings):
    """Resample training seeds and whole target stacks, keeping all pairs intact."""
    rng = np.random.default_rng(9202)
    seeds = settings["seeds"]
    repeats = settings["bootstrap_repeats"]
    sampled_seeds = rng.integers(len(seeds), size=(repeats, len(seeds)))
    distributions = {}
    result = {}
    for split in ("test", *EXTERNAL):

        def values(name):
            return np.asarray(
                [
                    [
                        [np.mean([p[m] for p in stack]) for m in METRICS]
                        for stack in raw[f"{name}-{seed}"][split]
                    ]
                    for seed in seeds
                ]
            )

        delta = values(FOURIER) - values(BASELINE)
        sampled_stacks = rng.integers(delta.shape[1], size=(repeats, delta.shape[1]))
        samples = delta[sampled_seeds[:, :, None], sampled_stacks[:, None, :]].mean(axis=(1, 2))
        distributions[split] = (delta.mean(axis=1), samples)

    distributions["external_macro"] = tuple(
        np.mean([distributions[split][axis] for split in EXTERNAL], axis=0) for axis in (0, 1)
    )
    for split, (per_seed, samples) in distributions.items():
        result[split] = {
            metric: {
                "mean": float(per_seed[:, i].mean()),
                "by_seed": per_seed[:, i].tolist(),
                "ci95": np.quantile(samples[:, i], (0.025, 0.975)).tolist(),
            }
            for i, metric in enumerate(METRICS)
        }
    primary = result["external_macro"]["utility"]
    low, high = primary["ci95"]
    winner = None
    if low > 0 and min(primary["by_seed"]) > 0:
        winner = FOURIER
    elif high < 0 and max(primary["by_seed"]) < 0:
        winner = BASELINE
    return {"direction": f"{FOURIER} minus {BASELINE}", "differences": result, "winner": winner}


def evaluate_study(study, cache, published, device, *, fourier_study=None):
    settings = json.loads((study / "config.json").read_text(encoding="utf-8"))
    alternative = None
    if fourier_study is not None:
        alternative = json.loads((fourier_study / "config.json").read_text(encoding="utf-8"))
        if {**alternative, "epochs": settings["epochs"]} != settings:
            raise ValueError("The alternative Fourier budget may differ only in epoch count")
    output = fourier_study if fourier_study is not None else study
    torch.set_num_threads(settings["threads"])
    physics = FresnelPropagator().to(device)
    test = torch.stack(
        list(SyntheticTargets("test", settings["test_count"], settings["data_seed"]))
    )
    targets = {"test": test}
    manifest = {
        "test": {
            "generator": DATASET_VERSION,
            "seed": settings["data_seed"],
            "indices": list(range(len(test))),
        }
    }
    for name in EXTERNAL:
        targets[name], manifest[name] = external_targets(
            name, cache, settings["external_count"], settings["external_seed"]
        )
    paths = {
        f"{name}-{seed}": (output if name == FOURIER else study) / f"{name}-{seed}" / "best.pt"
        for seed in settings["seeds"]
        for name in MODELS
    }
    training = {}
    for run, path in paths.items():
        metadata = json.loads((path.parent / "training.json").read_text(encoding="utf-8"))
        checkpoint = path.parent / metadata.get("checkpoint_path", path.name)
        if digest(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {run}")
        paths[run] = checkpoint
        training[run] = metadata
    if published is not None:
        paths["published-projection-v2"] = published
    models = {name: load_study_model(path, device) for name, path in paths.items()}
    times = latencies(models, test, device, settings)
    raw = {
        name: {
            split: measure(model, images, physics, device, settings["batch_size"])
            for split, images in targets.items()
        }
        for name, model in models.items()
    }
    summary = summarize(raw, times, settings)
    summary.update(paired_differences(raw, settings))
    summary["training"] = training
    summary["epochs"] = {
        BASELINE: settings["epochs"],
        FOURIER: (alternative if alternative is not None else settings)["epochs"],
    }
    summary["checkpoint_sha256"] = {key: digest(path) for key, path in paths.items()}
    summary["physics"] = {
        "size": physics.size,
        "wavelength_m": physics.wavelength_m,
        "pitch_m": physics.pitch_m,
        "distances_m": physics.distances_m,
    }
    write_json(output / "datasets.json", manifest)
    write_json(output / "evaluation-environment.json", environment(device))
    write_json(output / "latency-ms.json", times)
    with (output / "planes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("run", "split", "stack", "distance_mm", *METRICS))
        for run, splits in raw.items():
            for split, stacks in splits.items():
                for index, stack in enumerate(stacks):
                    for plane in stack:
                        writer.writerow(
                            (run, split, index, plane["distance_mm"], *(plane[m] for m in METRICS))
                        )
    write_json(output / "results.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_cmd = commands.add_parser("evaluate")
    evaluate_cmd.add_argument("--study", required=True, type=Path)
    evaluate_cmd.add_argument("--cache", required=True, type=Path)
    evaluate_cmd.add_argument("--published", type=Path)
    evaluate_cmd.add_argument("--fourier-study", type=Path, help="use a separate Fourier budget")
    synthesize = commands.add_parser("synthesize")
    synthesize.add_argument("--checkpoint", required=True, type=Path)
    synthesize.add_argument("--targets", required=True, nargs=4, type=Path)
    synthesize.add_argument("--output", required=True, type=Path)
    for command in (evaluate_cmd, synthesize):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.command == "evaluate":
        evaluate_study(
            args.study, args.cache, args.published, device, fourier_study=args.fourier_study
        )
    else:
        torch.set_num_threads(4)
        model = load_study_model(args.checkpoint, device)
        target = torch.stack([png_to_gray(path.read_bytes()) for path in args.targets])
        with torch.inference_mode():
            phase = model(target[None].to(device))[0, 0]
        args.output.write_bytes(phase_to_png(phase))


if __name__ == "__main__":
    main()
