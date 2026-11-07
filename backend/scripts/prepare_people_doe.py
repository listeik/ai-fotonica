"""Prepare 100 fixed photo quartets and optimize an exported phase-only DOE per quartet."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from people_doe_reference import PhotoObjective, export_photo_phase, normalize_amplitude
from PIL import Image, ImageDraw
from scipy.io import loadmat
from skimage.metrics import structural_similarity

from doe.physics import (
    DISTANCES_M,
    PITCH_M,
    SIZE,
    WAVELENGTH_M,
    FresnelPropagator,
    png_to_gray,
    png_to_phase,
    quantize_phase,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs/people-doe-100-v1"
DEFAULT_SOURCE = ROOT / "runs/people-doe-sources/olivettifaces.mat"
SOURCE_URL = "https://ndownloader.figshare.com/files/5976027"
SOURCE_SHA256 = "b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794"
SOURCE_CREDIT = "AT&T Laboratories Cambridge; modified Olivetti faces by Sam Roweis"
SPLITS = {"train": 32, "validation": 4, "test": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_source(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".download")
        with urlopen(SOURCE_URL, timeout=30) as response, partial.open("wb") as stream:
            while chunk := response.read(32768):
                stream.write(chunk)
        require(sha256(partial) == SOURCE_SHA256, "download checksum mismatch")
        partial.replace(path)
    require(sha256(path) == SOURCE_SHA256, "source checksum mismatch")
    faces = loadmat(path)["faces"].T.reshape(400, 64, 64).transpose(0, 2, 1)
    require(faces.dtype == np.uint8, "unexpected source pixel format")
    return faces


def freeze_runtime(output):
    """Keep the exact numerical implementation beside the local data."""
    paths = [
        ROOT / "backend/src/doe" / name
        for name in (
            "__init__.py",
            "physics.py",
            "metrics.py",
            "loss.py",
            "quality.py",
            "synthesis.py",
        )
    ]
    paths += [
        ROOT / "backend/scripts" / name
        for name in ("prepare_people_doe.py", "refine_people_doe.py", "people_doe_reference.py")
    ]
    checksums = {}
    for path in paths:
        relative = Path("reproduce") / (
            Path("doe") / path.name if path.parent.name == "doe" else Path(path.name)
        )
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        checksums[relative.as_posix()] = sha256(target)
    return checksums


def prepare_targets(output, source, seed, steps, restarts):
    faces = load_source(source)
    rng = np.random.default_rng(seed)
    subjects = rng.permutation(40)
    view_order = {int(subject): rng.permutation(10) for subject in subjects}
    start = 0
    records = []
    for split, number in SPLITS.items():
        cohort = subjects[start : start + number]
        start += number
        for round_index in range(10):
            shuffled = rng.permutation(cohort)
            for group in range(number // 4):
                sample = round_index * (number // 4) + group
                folder = output / split / f"{sample:03d}"
                folder.mkdir(parents=True)
                origins = []
                for plane, subject in enumerate(shuffled[group * 4 : group * 4 + 4], 1):
                    view = int(view_order[int(subject)][round_index])
                    source_index = int(subject) * 10 + view
                    image = Image.fromarray(faces[source_index]).resize(
                        (SIZE, SIZE), Image.Resampling.LANCZOS
                    )
                    filename = f"target-{plane}.png"
                    image.save(folder / filename)
                    origins.append(
                        {
                            "plane": plane,
                            "file": filename,
                            "source_index": source_index,
                            "subject_id": int(subject) + 1,
                            "view_id": view + 1,
                            "source_pixel_sha256": hashlib.sha256(
                                faces[source_index].tobytes()
                            ).hexdigest(),
                            "target_sha256": sha256(folder / filename),
                        }
                    )
                records.append(
                    {
                        "split": split,
                        "set": sample,
                        "directory": folder.relative_to(output).as_posix(),
                        "seed": seed + len(records) * 1009,
                        "sources": origins,
                    }
                )
    manifest = {
        "version": "people-doe-100-v1",
        "sets": 100,
        "targets": 400,
        "split_sets": {"train": 80, "validation": 10, "test": 10},
        "source": {
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "credit": SOURCE_CREDIT,
            "description": "400 different captures of 40 subjects, ten views per subject",
            "source_resolution": [64, 64],
            "record": "https://cam-orl.co.uk/facedatabase.html/",
            "distribution_reference": "https://sklearn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html",
            "usage_notice": "Credit AT&T Laboratories Cambridge when using these images",
        },
        "processing": "uint8 pixels preserved, orientation corrected from MATLAB layout, Lanczos resize 64x64 to 128x128",
        "target_origin": "fixed photographs prepared before DOE optimization",
        "target_quantity": "grayscale amplitude target under the project's peak-normalized abs(U) convention",
        "split_rule": "disjoint subject identities; each of the 400 source captures used exactly once",
        "phase_mapping": "phase_radians = png_code * 2*pi/255 - pi",
        "optics": {
            "wavelength_m": WAVELENGTH_M,
            "input_pitch_m": PITCH_M,
            "distances_m": DISTANCES_M,
            "size": SIZE,
        },
        "search": {
            "seed": seed,
            "steps_per_restart": steps,
            "restarts": restarts,
            "check_every": 100,
            "gs_steps_first_start": 160,
            "learning_rate": 0.04,
            "minimum_learning_rate": 0.001,
            "objective": "PCC and global SSIM, 0.7 mean plus 0.3 worst-plane for each",
            "selection": "maximum min of all 8 exported PCC/global SSIM values, mean score breaks ties",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "records": records,
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def read_targets(folder):
    return torch.stack(
        [png_to_gray((folder / f"target-{i}.png").read_bytes()) for i in range(1, 5)]
    )


def rank(metrics):
    return (
        min(metrics["worst_pcc"], metrics["worst_ssim"]),
        metrics["mean_pcc"] + metrics["mean_ssim"],
    )


def measure(phase, targets, optics):
    phase_png, images, metrics = export_photo_phase(phase, targets, optics)
    pixels = np.stack([png_to_gray(image).numpy() for image in images])
    local = [
        float(structural_similarity(y, x, data_range=1))
        for x, y in zip(pixels, targets.numpy(), strict=True)
    ]
    metrics["local_ssim_7x7"] = local
    metrics["worst_local_ssim"] = min(local)
    metrics["mean_local_ssim"] = float(np.mean(local))
    metrics["mse"] = ((pixels - targets.numpy()) ** 2).mean(axis=(-2, -1)).tolist()
    return phase_png, images, metrics


def solve(job):
    output, record, config = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(output) / record["directory"]
    targets = read_targets(folder)
    optics = FresnelPropagator()
    objective = PhotoObjective(targets)
    rng = torch.Generator().manual_seed(record["seed"])
    best = None
    history = []
    started = time.perf_counter()
    steps = config["steps_per_restart"]
    for restart in range(config["restarts"]):
        phase = torch.rand((SIZE, SIZE), generator=rng) * (2 * math.pi) - math.pi
        if restart == 0:
            with torch.no_grad():
                gain = torch.tensor([PITCH_M**2 / (WAVELENGTH_M * z) for z in DISTANCES_M])
                desired = targets / targets.square().sum((-2, -1), keepdim=True).sqrt()
                desired = desired * (SIZE**2 * gain[:, None, None])
                for _ in range(config["gs_steps_first_start"]):
                    fields = optics(torch.polar(torch.ones_like(phase), phase))
                    phase = (
                        optics.inverse_planes(torch.polar(desired, fields.angle())).sum(0).angle()
                    )
        phase = phase.detach().requires_grad_()
        optimizer = torch.optim.Adam([phase], lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, steps, eta_min=config["minimum_learning_rate"]
        )
        for step in range(steps + 1):
            quantized = quantize_phase(phase.detach())
            amplitude = optics.amplitude(quantized + (phase - phase.detach()))
            image = normalize_amplitude(amplitude)
            discrete = image.detach().mul(255).round().div(255)
            if step % config["check_every"] == 0 or step == steps:
                with torch.no_grad():
                    phase_png, reconstructions, metrics = measure(quantized, targets, optics)
                    history.append(
                        {
                            "restart": restart,
                            "step": step,
                            "rank": rank(metrics),
                            "mean_pcc": metrics["mean_pcc"],
                            "mean_ssim": metrics["mean_ssim"],
                        }
                    )
                    if best is None or rank(metrics) > rank(best[2]):
                        best = phase_png, reconstructions, metrics, restart, step
            if step == steps:
                break
            pixels = discrete + (image - image.detach())
            pcc, ssim = objective.scores(pixels)
            loss = -(0.7 * pcc.mean() + 0.3 * pcc.min())
            loss -= 0.7 * ssim.mean() + 0.3 * ssim.min()
            loss += 0.1 * (image - targets).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    phase_png, reconstructions, metrics, restart, step = best
    (folder / "doe.png").write_bytes(phase_png)
    for plane, png in enumerate(reconstructions, 1):
        (folder / f"reconstruction-{plane}.png").write_bytes(png)
    report = {
        "metrics": metrics,
        "best_restart": restart,
        "best_step": step,
        "seconds": time.perf_counter() - started,
        "history": history,
        "files_sha256": {path.name: sha256(path) for path in sorted(folder.glob("*.png"))},
    }
    save_json(folder / "result.json", report)
    return {
        "directory": record["directory"],
        **{key: report[key] for key in report if key != "history"},
    }


def verify(output):
    torch.set_num_threads(1)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if "local_archive" in manifest["source"]:
        require(
            sha256(output / manifest["source"]["local_archive"]) == manifest["source"]["sha256"],
            "local source archive checksum mismatch",
        )
    for name, checksum in manifest.get("reference_code_sha256", {}).items():
        require(sha256(output / name) == checksum, f"reference code checksum mismatch: {name}")
    require(len(manifest["records"]) == 100, "expected 100 sets")
    require(
        Counter(record["split"] for record in manifest["records"])
        == {"train": 80, "validation": 10, "test": 10},
        "split counts differ from 80/10/10",
    )
    require(
        len({record["directory"] for record in manifest["records"]}) == 100,
        "duplicate set directory",
    )
    subjects = {}
    source_indices, pixel_hashes = set(), set()
    optics = FresnelPropagator()
    results = []
    for record in manifest["records"]:
        folder = output / record["directory"]
        require(
            len({source["subject_id"] for source in record["sources"]}) == 4,
            "each quartet must show four different source subjects",
        )
        for source in record["sources"]:
            require(
                source["source_index"] // 10 + 1 == source["subject_id"]
                and source["source_index"] % 10 + 1 == source["view_id"],
                "source identity/view metadata mismatch",
            )
            require(
                subjects.setdefault(source["subject_id"], record["split"]) == record["split"],
                "subject leakage between splits",
            )
            require(source["source_index"] not in source_indices, "repeated source capture")
            source_indices.add(source["source_index"])
            path = folder / source["file"]
            require(sha256(path) == source["target_sha256"], f"target changed: {path}")
            pixels = png_to_gray(path.read_bytes())
            checksum = hashlib.sha256(
                pixels.mul(255).round().to(torch.uint8).numpy().tobytes()
            ).hexdigest()
            require(checksum not in pixel_hashes, "duplicate target pixels")
            pixel_hashes.add(checksum)
        report = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        expected = {"doe.png"} | {
            f"{kind}-{plane}.png" for kind in ("target", "reconstruction") for plane in range(1, 5)
        }
        require(set(report["files_sha256"]) == expected, f"unexpected PNG inventory: {folder}")
        for name, checksum in report["files_sha256"].items():
            require(sha256(folder / name) == checksum, f"checksum mismatch: {folder / name}")
            png_to_gray((folder / name).read_bytes())
        phase = png_to_phase((folder / "doe.png").read_bytes())
        _, reconstructed, metrics = measure(phase, read_targets(folder), optics)
        for plane, png in enumerate(reconstructed, 1):
            require(
                torch.equal(
                    png_to_gray(png),
                    png_to_gray((folder / f"reconstruction-{plane}.png").read_bytes()),
                ),
                f"reconstruction mismatch: {folder}, plane {plane}",
            )
        require(metrics == report["metrics"], f"metric mismatch: {folder}")
        results.append(
            {
                "directory": record["directory"],
                "metrics": metrics,
                "seconds": report["seconds"],
                "best_restart": report["best_restart"],
                "best_step": report["best_step"],
                "files_sha256": report["files_sha256"],
            }
        )
    require(len(source_indices) == 400 and len(subjects) == 40, "source inventory mismatch")
    require(Counter(subjects.values()) == SPLITS, "subject split counts differ from 32/4/4")
    require(len(list(output.rglob("target-*.png"))) == 400, "target count mismatch")
    require(len(list(output.rglob("doe.png"))) == 100, "DOE count mismatch")
    require(len(list(output.rglob("reconstruction-*.png"))) == 400, "reconstruction count mismatch")
    distribution = {}
    for key in (
        "mean_pcc",
        "worst_pcc",
        "mean_ssim",
        "worst_ssim",
        "mean_local_ssim",
        "worst_local_ssim",
    ):
        values = [result["metrics"][key] for result in results]
        distribution[key] = dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                np.percentile(values, [0, 10, 50, 90, 100]).tolist(),
            )
        )
    passed = {
        str(threshold): sum(rank(result["metrics"])[0] >= threshold for result in results)
        for threshold in (0.8, 0.9)
    }
    return {
        "sets": 100,
        "targets": 400,
        "doe": 100,
        "reconstructions": 400,
        "unique_subjects": 40,
        "unique_source_captures": 400,
        "duplicate_targets": 0,
        "subject_leaks": 0,
        "all_four_planes_pcc_and_global_ssim_pass_count": passed,
        "metric_distribution": distribution,
        "results": results,
    }


def preview(output, results):
    chosen = sorted(results, key=lambda row: rank(row["metrics"]))
    selected = [chosen[index] for index in (0, 25, 50, 75, 99)]
    sheet = Image.new("RGB", (8 * SIZE, len(selected) * (SIZE + 28)), "#17202b")
    draw = ImageDraw.Draw(sheet)
    for row, result in enumerate(selected):
        folder = output / result["directory"]
        y = row * (SIZE + 28)
        for column, name in enumerate(
            [f"{kind}-{i}.png" for kind in ("target", "reconstruction") for i in range(1, 5)]
        ):
            with Image.open(folder / name) as image:
                sheet.paste(image, (column * SIZE, y))
        score = result["metrics"]
        draw.text(
            (8, y + SIZE + 5),
            f"{result['directory']} | targets 1-4 / reconstructions 1-4 | worst PCC {score['worst_pcc']:.3f}, SSIM {score['worst_ssim']:.3f}",
            fill="white",
        )
    sheet.save(output / "preview.png")


def build(output, source, seed, steps, restarts, workers):
    require(seed >= 0 and steps > 0 and restarts > 0 and workers > 0, "invalid search settings")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(
            (
                manifest["search"]["seed"],
                manifest["search"]["steps_per_restart"],
                manifest["search"]["restarts"],
            )
            == (seed, steps, restarts),
            "resume settings differ from manifest",
        )
    else:
        require(not any(output.iterdir()), "output contains files without a manifest")
        manifest = prepare_targets(output, source, seed, steps, restarts)
    manifest["reference_code_sha256"] = freeze_runtime(output)
    save_json(output / "manifest.json", manifest)
    pending = [
        record
        for record in manifest["records"]
        if not (output / record["directory"] / "result.json").exists()
    ]
    done = 100 - len(pending)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(solve, (str(output), record, manifest["search"])): record
            for record in pending
        }
        for future in as_completed(jobs):
            result = future.result()
            done += 1
            scores = result["metrics"]
            print(
                f"{done}/100 {result['directory']}: worst PCC={scores['worst_pcc']:.4f}, "
                f"global SSIM={scores['worst_ssim']:.4f}, local SSIM={scores['worst_local_ssim']:.4f}",
                flush=True,
            )
    summary = verify(output)
    summary["current_build_wall_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    preview(output, summary["results"])
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("metric_distribution", "all_four_planes_pcc_and_global_ssim_pass_count")
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "package"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.command == "build":
        build(args.output, args.source, args.seed, args.steps, args.restarts, args.workers)
    else:
        report = verify(args.output)
        print(
            json.dumps(
                {key: report[key] for key in ("sets", "duplicate_targets", "subject_leaks")},
                indent=2,
            )
        )
        if args.command == "package":
            with zipfile.ZipFile(
                args.output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for path in sorted(args.output.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                        archive.write(
                            path, f"{args.output.name}/{path.relative_to(args.output).as_posix()}"
                        )


if __name__ == "__main__":
    main()
