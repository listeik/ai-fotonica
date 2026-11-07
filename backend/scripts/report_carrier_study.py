"""Aggregate carrier-study evidence into a compact portable JSON report.

This command reads existing JSON records and file hashes. It performs no model
inference, training, dataset regeneration or checkpoint selection.
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LARGE_KEYS = {
    "quartets",
    "quartet_deltas",
    "timings_ms",
    "raw_timings",
    "individual_timings",
    "measurements",
    "samples_ms",
    "target_ids",
    "records",
    "ids",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def portable(value):
    if isinstance(value, dict):
        return {str(key).replace("\\", "/"): portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        candidate = value.replace("\\", "/")
        root = ROOT.as_posix()
        if candidate.lower().startswith(root.lower() + "/"):
            return candidate[len(root) + 1 :]
        return candidate
    return value


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key not in LARGE_KEYS}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def artifact(path):
    return {"path": str(path.resolve()), "sha256": digest(path)}


def metrics_summary(metrics):
    if not isinstance(metrics, dict):
        return metrics
    return {name: compact(group) for name, group in metrics.items()}


def compact_evaluation(path):
    payload = read(path)
    result = {
        "artifact": artifact(path),
        "protocol": compact(payload.get("protocol", {})),
        "baseline": payload.get("baseline"),
        "test": compact(payload.get("test", {})),
        "models": {},
    }
    for name, model in payload.get("models", {}).items():
        result["models"][name] = {
            key: compact(value) for key, value in model.items() if key != "splits"
        }
        result["models"][name]["splits"] = {
            split: compact(section["summary"]) for split, section in model.get("splits", {}).items()
        }
    for name in ("deltas_from_baseline", "paired_differences"):
        if name in payload:
            result[name] = compact(payload[name])
    return result


def compact_dataset(path):
    payload = read(path)
    result = {
        key: value for key, value in payload.items() if key not in ("external", "natural_bank")
    }
    natural = payload.get("natural_bank", {})
    source_origins = {}
    source_counts = Counter()
    for record in natural.get("records", []):
        source_counts[record["source"]] += 1
        source_origins.setdefault(record["source"], set()).add(str(record["origin"]))
    result["natural_bank"] = {key: value for key, value in natural.items() if key != "records"}
    result["natural_bank"]["images_by_source"] = dict(source_counts)
    result["natural_bank"]["unique_origins_by_source"] = {
        name: len(origins) for name, origins in source_origins.items()
    }
    external = payload.get("external", {})
    result["external"] = {key: value for key, value in external.items() if key != "records"}
    external_records = external.get("records", [])
    result["external"]["images_by_split"] = dict(Counter(r["split"] for r in external_records))
    special = {"100007.jpg", "146074.jpg", "207038.jpg", "290035.jpg"}
    result["external"]["fixed_quartet_sources"] = [
        record
        for record in external_records
        if record.get("split") == "test" and record.get("source_file") in special
    ]
    result["external"]["source_records_sha256"] = hashlib.sha256(
        json.dumps(external_records, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result["artifact"] = artifact(path)
    return result


def run_record(folder):
    result_path = folder / "result.json"
    result = read(result_path)
    config = read(folder / "config.json")
    provenance = read(folder / "provenance.json")
    history = read(folder / "history.json")
    initial = read(folder / "initial-validation.json")
    supervised = (
        config.get("teacher_weight", 0) > 0
        or config.get("teacher_phase_weight", 0) > 0
        or config.get("supervised_epochs", 0) > 0
    )
    if supervised:
        regime = "teacher_supervision"
    elif provenance.get("teacher_sha256"):
        regime = "matched_teacher_targets_control"
    elif provenance.get("parent_sha256"):
        regime = "warm_start"
    else:
        regime = "scratch"
    curves = []
    for row in history:
        validation = row.get("validation", {})
        curves.append(
            {
                "epoch": row["epoch"],
                "step": row.get("step"),
                "best_epoch_at_time": row.get("best_epoch"),
                "seconds": row.get("seconds"),
                "training": row.get("training", {}),
                "selection": validation.get("selection"),
                "group_scores": {
                    key: value["score"]
                    for key, value in validation.items()
                    if isinstance(value, dict) and "score" in value
                },
            }
        )
    return {
        "name": folder.name,
        "regime": regime,
        "configuration": config,
        "provenance": provenance,
        "artifacts": {
            file: artifact(folder / file)
            for file in (
                "result.json",
                "config.json",
                "initial-validation.json",
                "history.json",
                "provenance.json",
            )
        },
        "checkpoint": {
            "path": str(folder / "best.pt"),
            "sha256": result["sha256"],
            "best_epoch": result["best_epoch"],
            "parameters": result["parameters"],
        },
        "initial_validation": metrics_summary(initial),
        "best_validation": metrics_summary(result["best_validation"]),
        "epochs_completed": len(history),
        "training_updates": history[-1].get("step") if history else 0,
        "elapsed_seconds": result.get("elapsed"),
        "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
        "score_curve": curves,
    }


def teacher_record(folder):
    path = folder / "manifest.json"
    payload = read(path)
    records = payload.get("records", [])
    result = {key: value for key, value in payload.items() if key != "records"}
    result["record_count"] = len(records)
    result["source_record_digest"] = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result["artifact"] = artifact(path)
    tensors = folder / "teachers.pt"
    if tensors.exists():
        import torch

        payload = torch.load(tensors, map_location="cpu", weights_only=True)
        axes = payload.get("metric_axis", result.get("metric_axis", []))
        if not axes:
            axes = ["pcc", "global_ssim", "windowed_ssim"]
        result["quality_metrics"] = {}
        for name in ("initial_metrics", "best_metrics"):
            matrix = payload[name].double()
            result["quality_metrics"][name] = {
                "shape": list(matrix.shape),
                "axis": axes,
                "mean": matrix.mean((0, 1)).tolist(),
                "mean_worst": matrix.amin(1).mean(0).tolist(),
                "minimum": matrix.amin((0, 1)).tolist(),
            }
        result["quality_metrics"]["weakest_joint_score"] = {
            name: {
                "mean": values.double().mean().item(),
                "p10": torch.quantile(values.double(), 0.1).item(),
                "minimum": values.min().item(),
            }
            for name in ("initial_scores", "best_scores")
            for values in (payload[name],)
        }
    quality = folder / "quality.json"
    if quality.exists():
        result["quality"] = compact(read(quality))
        result["quality_artifact"] = artifact(quality)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Study output directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report")
    parser.add_argument("--protocol", type=Path, help="Frozen experiment protocol JSON")
    parser.add_argument("--comparison", type=Path, help="Final held-out comparison JSON")
    parser.add_argument("--promotion", type=Path, help="Promotion-rule decision JSON")
    parser.add_argument("--benchmark", type=Path, help="CPU benchmark JSON")
    parser.add_argument("--export", type=Path, help="Export and round-trip verification JSON")
    args = parser.parse_args()
    study = args.runs.resolve()
    runs = [
        run_record(path.parent)
        for path in sorted(study.glob("*/result.json"))
        if all(
            (path.parent / file).exists()
            for file in (
                "config.json",
                "history.json",
                "provenance.json",
                "initial-validation.json",
            )
        )
    ]
    seeds = sorted({run["configuration"]["seed"] for run in runs})
    report = {
        "format": "carrier-study-report-v1",
        "created_with_python": platform.python_version(),
        "run_directory": str(study),
        "limitations": [
            "Sequential validation screening with one seed per candidate; seed variation was not estimated.",
            "Widths and architectures have explicit update budgets, not a guaranteed equal wall-clock budget.",
            "Warm-start, scratch and teacher-supervised regimes have different parent training histories.",
            "Paired bootstrap intervals are descriptive and not adjusted for repeated model comparisons.",
            "Portrait quartets share identities; distinct source IDs do not prove semantic independence.",
            "Only frozen validation metrics may choose finalists. Held-out results verify the frozen promotion rule.",
        ],
        "seeds_used": seeds,
        "total_recorded_training_seconds": sum(run["elapsed_seconds"] or 0 for run in runs),
        "run_count": len(runs),
        "runs": runs,
        "tracked_code_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted((ROOT / "backend").rglob("*.py"))
            if ".venv" not in path.parts and "__pycache__" not in path.parts
        },
    }
    if args.protocol:
        report["protocol"] = {
            "artifact": artifact(args.protocol),
            "definition": read(args.protocol),
        }
    data_manifest = study / "data/manifest.json"
    if data_manifest.exists():
        report["dataset"] = compact_dataset(data_manifest)
    for filename, name in (
        ("loss-gradient-probe.json", "loss_gradient_probe"),
        ("stage3-selection.json", "stage3_selection"),
        ("continuation-plan.json", "continuation_plan"),
        ("finalists.json", "finalists"),
    ):
        path = study / filename
        if path.exists():
            report[name] = {"artifact": artifact(path), "data": compact(read(path))}
    windows = study / "window-validation.json"
    if windows.exists():
        report["window_validation"] = compact_evaluation(windows)
    teachers = [
        teacher_record(path.parent)
        for path in sorted(study.glob("*/manifest.json"))
        if path.parent.name != "data"
        and read(path).get("version", "").startswith("carrier-natural-teachers")
    ]
    if teachers:
        report["teachers"] = teachers
    for option, key in (
        (args.comparison, "heldout_comparison"),
        (args.promotion, "promotion"),
        (args.benchmark, "benchmark"),
        (args.export, "export_verification"),
    ):
        if option:
            report[key] = (
                compact_evaluation(option)
                if key == "heldout_comparison"
                else {"artifact": artifact(option), "data": compact(read(option))}
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(portable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"Aggregated {len(runs)} runs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
