"""Compare synthesis methods using one held-out target set and evaluator."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class Variant:
    method: str
    steps: int

    @property
    def label(self) -> str:
        return f"{self.method.title()} ({self.steps})" if self.steps else self.method.title()


def _summarize(rows: list[dict[str, Any]], variants: list[Variant]) -> list[dict[str, Any]]:
    summary = []
    for variant in variants:
        runs = [row for row in rows if row["configuration"] == variant.label]
        summary.append(
            {
                "configuration": variant.label,
                "method": variant.method,
                "requested_iterations": variant.steps,
                "mean_iterations": statistics.mean(run["iterations"] for run in runs),
                "mean_pcc": statistics.mean(run["mean_pcc"] for run in runs),
                "mean_ssim": statistics.mean(run["mean_ssim"] for run in runs),
                "worst_pcc": min(run["worst_pcc"] for run in runs),
                "worst_ssim": min(run["worst_ssim"] for run in runs),
                "median_seconds": statistics.median(run["seconds"] for run in runs),
            }
        )
    return summary


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key != "planes"}
    for number, plane in enumerate(row["planes"], start=1):
        for metric in ("distance_mm", "pcc", "ssim"):
            flat[f"plane_{number}_{metric}"] = plane[metric]
    return flat


def _plot(path: Path, summary: list[dict[str, Any]]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    x = list(range(len(summary)))
    labels = [row["configuration"].replace(" (", "\n(") for row in summary]
    colors = {"classical": "#4768A8", "neural": "#A14383", "hybrid": "#007E72"}
    row_colors = [colors[row["method"]] for row in summary]
    for axis, metric in zip(axes[:2], ("pcc", "ssim"), strict=True):
        means = [row[f"mean_{metric}"] for row in summary]
        minima = [row[f"worst_{metric}"] for row in summary]
        axis.bar(x, means, color=row_colors, width=0.64, label="Mean")
        axis.scatter(x, minima, marker="D", s=20, color="#202123", zorder=3, label="Worst")
        axis.set(ylim=(-1.05, 1.05), title=metric.upper())
        axis.axhline(0, color="#808080", linewidth=0.6)
    axes[0].legend(loc="lower right", frameon=False)

    medians = [row["median_seconds"] for row in summary]
    axes[2].scatter(x, medians, c=row_colors, s=42, zorder=3)
    if min(medians) > 0:
        axes[2].set_yscale("log")
    axes[2].set(title="Median synthesis time", ylabel="Seconds")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.22)
    figure.savefig(path, dpi=160)


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = [_csv_row(row) for row in report["samples"]]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = report["metadata"]
    table = [
        "# Сравнение методов синтеза",
        "",
        (
            f"Выборка `{metadata['split']}`: {metadata['sample_count']} случаев, по четыре цели. "
            f"Устройство — {metadata['device_name']}."
        ),
        "",
        (
            "| Метод (лимит шагов) | Среднее число шагов | Средний PCC | Средний SSIM "
            "| Худший PCC | Худший SSIM | Медиана, мс |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"]:
        table.append(
            f"| {row['configuration']} | {row['mean_iterations']:.1f} "
            f"| {row['mean_pcc']:.4f} | {row['mean_ssim']:.4f} "
            f"| {row['worst_pcc']:.4f} | {row['worst_ssim']:.4f} "
            f"| {1000 * row['median_seconds']:.3f} |"
        )
    table.extend(
        [
            "",
            (
                "Средние рассчитаны по всем наборам и плоскостям. "
                "Худшие значения — минимумы среди всех плоскостей и наборов. "
                "Время синтеза не включает загрузку весов, прогрев, экспорт PNG и расчёт метрик."
            ),
            "",
            "![Сравнение PCC, SSIM и времени синтеза](comparison.png)",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(table), encoding="utf-8")
    _plot(output / "comparison.png", report["summary"])


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="trained checkpoint (omit for classical-only runs)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=_positive, default=4, help="PyTorch CPU threads")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--samples", type=_positive, default=16)
    parser.add_argument("--seed", type=_nonnegative, default=2026, help="target generator seed")
    parser.add_argument("--optimizer-seed", type=_nonnegative, default=0)
    parser.add_argument("--classical-steps", type=_positive, nargs="+", default=[20, 200])
    parser.add_argument("--hybrid-steps", type=_positive, nargs="+", default=[20])
    parser.add_argument("--output", type=Path, default=Path("experiments/comparison"))
    args = parser.parse_args(argv)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    try:
        matplotlib_version = version("matplotlib")
    except PackageNotFoundError:
        parser.error("install the backend's benchmark extra to generate reports")

    import torch

    from doe.physics import FresnelPropagator
    from doe.synthesis import synthesize
    from doe.training.dataset import DATASET_VERSION, synthetic_targets

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    variants = [Variant("classical", steps) for steps in dict.fromkeys(args.classical_steps)]
    model = None
    checkpoint = None
    if args.checkpoint is not None:
        from doe.models import load_model

        with args.checkpoint.open("rb") as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        start = perf_counter()
        model, model_version = load_model(args.checkpoint, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        checkpoint = {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "model_version": model_version,
            "load_seconds": perf_counter() - start,
        }
        variants += [Variant("neural", 0)]
        variants += [Variant("hybrid", steps) for steps in dict.fromkeys(args.hybrid_steps)]

    def run(targets: torch.Tensor, variant: Variant, seed: int):
        return synthesize(
            targets,
            variant.method,
            propagator,
            model=model if variant.method != "classical" else None,
            iterations=variant.steps or None,
            seed=seed,
        )

    propagator = FresnelPropagator().to(device)
    warmup = synthetic_targets(0, split="demo", seed=args.seed).to(device)
    start = perf_counter()
    for variant in variants:
        run(warmup, variant, args.optimizer_seed)
    warmup_seconds = perf_counter() - start

    rows = []
    for index in range(args.samples):
        targets = synthetic_targets(index, split=args.split, seed=args.seed)
        target_sha256 = hashlib.sha256(
            targets.mul(255).round().to(torch.uint8).numpy().tobytes()
        ).hexdigest()
        targets = targets.to(device)
        seed = args.optimizer_seed + index
        # Rotate configurations to spread order effects across samples.
        offset = index % len(variants)
        for variant in variants[offset:] + variants[:offset]:
            result = run(targets, variant, seed)
            rows.append(
                {
                    "sample_index": index,
                    "target_sha256": target_sha256,
                    "phase_png_sha256": hashlib.sha256(result.phase_png).hexdigest(),
                    "configuration": variant.label,
                    "method": variant.method,
                    "requested_iterations": variant.steps,
                    "iterations": result.iterations,
                    "optimizer_seed": seed,
                    "seconds": result.seconds,
                    **result.metrics,
                }
            )

    metadata = {
        "split": args.split,
        "sample_count": args.samples,
        "dataset_version": DATASET_VERSION,
        "dataset_seed": args.seed,
        "optimizer_seed_start": args.optimizer_seed,
        "checkpoint": checkpoint,
        "warmup_seconds": warmup_seconds,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine()
        ),
        "cuda_runtime": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            **{name: version(name) for name in ("torch", "numpy", "pillow", "scikit-image")},
            "matplotlib": matplotlib_version,
        },
        "distances_m": [float(distance) for distance in propagator.distances_m],
        "resolution_px": propagator.size,
        "wavelength_m": propagator.wavelength_m,
        "input_pitch_m": propagator.pitch_m,
        "output_pitches_m": propagator.output_pitch_m.tolist(),
    }
    summary = _summarize(rows, variants)
    _write_report(
        args.output,
        {
            "schema_version": 1,
            "metadata": metadata,
            "summary": summary,
            "samples": rows,
        },
    )
    print(f"Saved comparison for {args.samples} {args.split} cases to {args.output}")


if __name__ == "__main__":
    main()
