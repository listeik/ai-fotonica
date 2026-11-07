"""Score named DOE checkpoints on a frozen held-out dataset."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend/src"), str(ROOT / "backend/scripts")]

from doe.models import load_model  # noqa: E402
from doe.physics import DISTANCES_M, SIZE, FresnelPropagator, png_to_gray  # noqa: E402
from doe.synthesis import synthesize as synthesize  # noqa: E402
from prepare_people_doe import measure  # noqa: E402
from skimage import __version__ as skimage_version  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

METRICS = ("pcc", "ssim", "local_ssim", "local_ssim_11")


def local_scores(reconstruction_pngs, target, window):
    """Evaluate decoded PNGs in float64 with explicit window conventions."""
    actual = [png_to_gray(data).numpy().astype(np.float64) for data in reconstruction_pngs]
    expected = target.numpy().astype(np.float64)
    return [
        float(structural_similarity(
            reconstructed, desired, data_range=1.0, win_size=window,
            gaussian_weights=False, use_sample_covariance=True,
        ))
        for reconstructed, desired in zip(actual, expected, strict=True)
    ]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_arg(value):
    name, separator, filename = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not filename:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(filename)


def load_test(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("test"), dict):
        raise TypeError("test.pt must contain a 'test' mapping")
    splits = payload["test"]
    if not splits:
        raise ValueError("test.pt has no held-out splits")
    supplied_ids = payload.get("test_ids", {})
    ids = {}
    for split, targets in splits.items():
        if (
            not isinstance(targets, torch.Tensor)
            or targets.dtype != torch.uint8
            or targets.ndim != 4
            or targets.shape[0] == 0
            or targets.shape[1:] != (len(DISTANCES_M), SIZE, SIZE)
        ):
            raise ValueError(f"{split}: expected nonempty uint8 [N,4,128,128] targets")
        sample_ids = supplied_ids.get(split, [f"{split}/{i:03d}" for i in range(len(targets))])
        if (
            len(sample_ids) != len(targets)
            or any(not isinstance(item, str) for item in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)
        ):
            raise ValueError(f"{split}: sample identifiers must be unique strings, one per quartet")
        ids[split] = sample_ids
    face_mask = payload.get("mixed_face_mask")
    if face_mask is not None and (
        "mixed_test" not in splits
        or not isinstance(face_mask, torch.Tensor)
        or face_mask.dtype != torch.bool
        or face_mask.shape != splits["mixed_test"].shape[:2]
    ):
        raise ValueError("mixed_face_mask must be a bool tensor with one flag per mixed_test plane")
    return splits, ids, face_mask


def summarize(rows, face_mask):
    plane_count = len(DISTANCES_M)
    summary = {"samples": len(rows), "planes": len(rows) * plane_count}
    quartet_mins = {}
    for key in METRICS:
        quartet_mean = [fmean(plane[key] for plane in row["planes"]) for row in rows]
        quartet_min = [min(plane[key] for plane in row["planes"]) for row in rows]
        quartet_mins[key] = quartet_min
        summary[f"mean_{key}"] = fmean(quartet_mean)
        summary[f"mean_worst_{key}"] = fmean(quartet_min)
        summary[f"p10_worst_{key}"] = float(np.quantile(quartet_min, 0.1))
        summary[f"min_{key}"] = min(quartet_min)

    joint = [
        min(pcc, ssim) for pcc, ssim in zip(quartet_mins["pcc"], quartet_mins["ssim"], strict=True)
    ]
    summary["mean_worst_joint"] = fmean(joint)
    summary["p10_worst_joint"] = float(np.quantile(joint, 0.1))
    summary["min_joint"] = min(joint)
    summary["quartets_all_pcc_and_global_ssim_above_0_9"] = sum(score > 0.9 for score in joint)
    if face_mask is not None:
        faces = [
            plane
            for row, mask in zip(rows, face_mask.tolist(), strict=True)
            for plane, is_face in zip(row["planes"], mask, strict=True)
            if is_face
        ]
        if faces:
            summary["faces_only"] = {
                "planes": len(faces),
                **{f"mean_{key}": fmean(plane[key] for plane in faces) for key in METRICS},
            }
    return summary


@torch.inference_mode()
def evaluate(path, splits, ids, face_mask, device, batch_size):
    model, version = load_model(path, device)
    optics = FresnelPropagator()
    results = {}
    for split, targets in splits.items():
        rows = []
        for batch in targets.split(batch_size):
            batch_cpu = batch.float().div_(255)
            phases = model(batch_cpu.to(device))
            if phases.shape != (len(batch), 1, SIZE, SIZE) or not torch.isfinite(phases).all():
                raise ValueError(f"{version}: unexpected or nonfinite phase output")
            for phase, target in zip(phases[:, 0].cpu(), batch_cpu, strict=True):
                _, images, scores = measure(phase, target, optics)
                planes = [
                    {**plane, "local_ssim": local7, "local_ssim_11": local11}
                    for plane, local7, local11 in zip(
                        scores["planes"], local_scores(images, target, 7),
                        local_scores(images, target, 11), strict=True,
                    )
                ]
                pixels = target.mul(255).round().to(torch.uint8).contiguous()
                rows.append({
                    "id": ids[split][len(rows)],
                    "target_pixels_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
                    "plane_target_sha256": [
                        hashlib.sha256(plane.numpy().tobytes()).hexdigest() for plane in pixels
                    ],
                    "planes": planes,
                })
        split_mask = face_mask if split == "mixed_test" else None
        results[split] = {"summary": summarize(rows, split_mask), "quartets": rows}
    return {
        "version": version,
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": model.architecture,
        "projection_steps": getattr(model, "projection_steps", None),
        "carrier_conditioning": getattr(model, "carrier_conditioning", None),
        "splits": results,
    }


def deltas(current, baseline):
    differences = {}
    for key, value in current.items():
        if key in ("samples", "planes"):
            continue
        if isinstance(value, dict):
            differences[key] = deltas(value, baseline[key])
        else:
            differences[key] = value - baseline[key]
    return differences


def paired_difference(current, reference):
    """Bootstrap paired quartets, retaining all four planes together."""
    if [row["id"] for row in current] != [row["id"] for row in reference]:
        raise ValueError("Paired comparisons require the same quartet identifiers and order")
    result = {}
    rng = np.random.default_rng(20261107)
    bootstrap = rng.integers(len(current), size=(2000, len(current)))
    for metric in METRICS:
        for prefix, reduction in (("mean", fmean), ("worst", min)):
            differences = np.array(
                [
                    reduction(plane[metric] for plane in lhs["planes"])
                    - reduction(plane[metric] for plane in rhs["planes"])
                    for lhs, rhs in zip(current, reference, strict=True)
                ]
            )
            low, high = np.quantile(differences[bootstrap].mean(axis=1), (0.025, 0.975))
            result[f"{prefix}_{metric}"] = {
                "mean_delta": float(differences.mean()),
                "paired_bootstrap_95": [float(low), float(high)],
                "quartet_win_fraction": float((differences > 0).mean()),
                "quartet_tie_fraction": float((differences == 0).mean()),
                "quartet_deltas": differences.tolist(),
            }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="local_ssim (7x7) is the primary windowed metric; "
               "local_ssim_11 (11x11) is an additional diagnostic on identical PNGs.",
    )
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=checkpoint_arg)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--external", type=Path, nargs=4, metavar="PNG",
                        help="four targets in ascending distance order, evaluated as external_quartet")
    args = parser.parse_args()
    if len(dict(args.model)) != len(args.model):
        parser.error("model names must be unique")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    input_paths = {path.resolve() for _, path in args.model} | {args.test.resolve()}
    if args.output.resolve() in input_paths:
        parser.error("--output must differ from all input paths")

    torch.set_num_threads(args.threads)
    splits, ids, face_mask = load_test(args.test)
    if args.external:
        if "external_quartet" in splits:
            parser.error("external_quartet already exists in test.pt")
        splits["external_quartet"] = torch.stack([
            png_to_gray(path.read_bytes()).mul(255).round().to(torch.uint8)
            for path in args.external
        ])[None]
        ids["external_quartet"] = ["external/" + hashlib.sha256(
            splits["external_quartet"].numpy().tobytes()).hexdigest()]
    test_metadata = {
        split: {
            "samples": len(targets),
            "ids": ids[split],
            "uint8_pixels_sha256": hashlib.sha256(targets.contiguous().numpy()).hexdigest(),
        }
        for split, targets in splits.items()
    }
    if face_mask is not None:
        test_metadata["mixed_test"]["face_mask"] = face_mask.tolist()
    results = {}
    for name, path in args.model:
        started = perf_counter()
        results[name] = evaluate(path, splits, ids, face_mask, args.device, args.batch_size)
        print(
            f"{name}: scored {sum(len(value) for value in splits.values())} quartets "
            f"in {perf_counter() - started:.2f}s",
            flush=True,
        )

    baseline_name = args.model[0][0]
    baseline = results[baseline_name]["splits"]
    report = {
        "protocol": {
            "inference": "one model.forward per input, no optimizer or post-forward refinement; "
            "fixed physical projection depth is recorded per checkpoint",
            "input": "uint8 target amplitude / 255",
            "export": "phase encoded as 8-bit PNG, decoded for CPU Fresnel propagation; "
            "peak-normalized amplitude encoded and decoded as uint8 reconstruction PNG",
            "ssim": "ssim=global population moments; local_ssim=7x7, local_ssim_11=11x11; "
            "both local metrics use uniform valid windows, stride1, float64, data_range1, "
            "sample covariance N/(N-1) with N=49 or121, gaussian_weights=False; "
            "both are evaluated on the same 8-bit PNG reconstruction pixels",
            "worst": "within each quartet, minimum of four planes per metric; joint additionally "
            "takes the minimum of PCC and global SSIM",
            "p10": "linear quantile of the quartet minima",
            "bootstrap": "2000 paired quartet resamples, seed 20261107; descriptive intervals, "
            "not adjusted for repeated model comparisons or shared face identities; portrait quartets "
            "reuse four identities and broad/Mixed may share sources, so split groups are not "
            "independent and these intervals do not establish statistical significance",
            "distances_mm": [round(distance * 1000, 4) for distance in DISTANCES_M],
            "device": args.device,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "torch": str(torch.__version__),
            "python": sys.version,
            "code_sha256": {
                str(path.relative_to(ROOT)): file_hash(path) for path in (
                    Path(__file__).resolve(), ROOT / "backend/src/doe/physics.py",
                    ROOT / "backend/scripts/people_doe_reference.py",
                    ROOT / "backend/scripts/prepare_people_doe.py",
                )
            },
            "scikit_image": skimage_version,
        },
        "test": {
            "path": str(args.test.resolve()),
            "sha256": file_hash(args.test),
            "splits": test_metadata,
        },
        "baseline": baseline_name,
        "models": results,
        "deltas_from_baseline": {
            name: {
                split: deltas(result["summary"], baseline[split]["summary"])
                for split, result in model["splits"].items()
            }
            for name, model in results.items()
            if name != baseline_name
        },
        "paired_differences": {
            name: {
                reference: {
                    split: paired_difference(
                        result["quartets"], results[reference]["splits"][split]["quartets"]
                    )
                    for split, result in model["splits"].items()
                }
                for reference in dict.fromkeys(
                    results
                )
                if reference in results and reference != name
            }
            for name, model in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
