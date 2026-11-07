"""Refine a fixed photo quartet and retain candidates scored after PNG export."""

import argparse
import json
import time
from pathlib import Path

import torch
from doe.physics import FresnelPropagator, png_to_phase, quantize_phase
from people_doe_reference import PhotoObjective, normalize_amplitude
from prepare_people_doe import measure, rank, read_targets, require, save_json, sha256


def refine(folder, output, method):
    require(not output.exists(), "choose a new output directory")
    require((folder / "doe.png").exists(), "input DOE missing")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    targets = read_targets(folder)
    optics = FresnelPropagator()
    objective = PhotoObjective(targets)
    base = png_to_phase((folder / "doe.png").read_bytes())
    (output / "starting-phase.png").write_bytes((folder / "doe.png").read_bytes())
    inputs = {
        path.name: sha256(path)
        for path in [folder / "doe.png", *sorted(folder.glob("target-*.png"))]
    }
    history, best = [], None
    started = time.perf_counter()

    @torch.no_grad()
    def record(phase, name, step):
        nonlocal best
        png, images, metrics = measure(phase, targets, optics)
        entry = {"method": name, "step": step, "metrics": metrics}
        history.append(entry)
        if best is None or rank(metrics) > rank(best["metrics"]):
            best = entry
            (output / "doe.png").write_bytes(png)
            for plane, image in enumerate(images, 1):
                (output / f"reconstruction-{plane}.png").write_bytes(image)
            save_json(output / "metrics.json", entry)
        print(
            f"{name} {step}: worst PCC={metrics['worst_pcc']:.6f}, "
            f"global SSIM={metrics['worst_ssim']:.6f}",
            flush=True,
        )

    record(base, "baseline", 0)
    phase = base.clone().requires_grad_()
    balanced = method == "balanced"
    steps = 12000 if balanced else 4000
    optimizer = torch.optim.Adam([phase], lr=0.012 if balanced else 0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, steps, eta_min=0.0002 if balanced else 0.0005
    )
    for step in range(steps + 1):
        quantized = quantize_phase(phase.detach())
        amplitude = optics.amplitude(quantized + (phase - phase.detach()))
        if balanced:
            image = normalize_amplitude(amplitude)
            pixels = image.detach().mul(255).round().div(255) + (image - image.detach())
            pcc, ssim = objective.scores(pixels)
            pmin = -0.01 * torch.logsumexp(-pcc / 0.01, dim=0)
            smin = -0.01 * torch.logsumexp(-ssim / 0.01, dim=0)
            loss = -0.5 * (pcc.mean() + ssim.mean()) - 0.5 * (pmin + smin)
        else:
            pcc, _ = objective.scores(amplitude)
            loss = -0.8 * pcc.mean() - 0.2 * pcc.min()
        if step % 500 == 0:
            record(quantized, "balanced_Adam" if balanced else "raw_amplitude_PCC", step)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    if not balanced:
        for source, name in ((base, "baseline_LBFGS"), (phase.detach(), "PCC_then_LBFGS")):
            phase = source.clone().requires_grad_()
            optimizer = torch.optim.LBFGS(
                [phase], lr=0.8, max_iter=20, history_size=40, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                pixels = normalize_amplitude(optics.amplitude(phase))
                pcc, ssim = objective.scores(pixels)
                pmin = -0.02 * torch.logsumexp(-pcc / 0.02, dim=0)
                smin = -0.02 * torch.logsumexp(-ssim / 0.02, dim=0)
                loss = -0.7 * (pcc.mean() + ssim.mean()) - 0.3 * (pmin + smin)
                loss.backward()
                return loss

            for step in range(31):
                record(phase, name, step * 20)
                if step < 30:
                    optimizer.step(closure)
    report = {
        "input_directory": str(folder.resolve()),
        "input_sha256": inputs,
        "method": method,
        "seconds": time.perf_counter() - started,
        "selection": "min of all eight exported PCC/global SSIM values, then sum of means",
        "best": best,
        "history": history,
    }
    save_json(output / "search.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=("balanced", "mixed"), default="balanced")
    parser.add_argument("--apply", action="store_true", help="replace input DOE if export improves")
    args = parser.parse_args()
    refine(args.folder, args.output, args.method)
    if args.apply:
        accept_refinement(args.folder, args.output)


def accept_refinement(folder, output):
    """Accept an improved export with identical targets; refresh root summary using verify."""
    torch.set_num_threads(1)
    search = json.loads((output / "search.json").read_text(encoding="utf-8"))
    require(
        all(sha256(folder / name) == value for name, value in search["input_sha256"].items()),
        "refinement input changed since search",
    )
    report_path = folder / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    phase = png_to_phase((output / "doe.png").read_bytes())
    png, images, metrics = measure(phase, read_targets(folder), FresnelPropagator())
    if rank(metrics) <= rank(report["metrics"]):
        return False
    previous = {
        key: report[key] for key in ("metrics", "best_restart", "best_step", "files_sha256")
    }
    (folder / "doe.png").write_bytes(png)
    for plane, image in enumerate(images, 1):
        (folder / f"reconstruction-{plane}.png").write_bytes(image)
    report["metrics"] = metrics
    report["best_restart"] = None
    report["best_step"] = search["best"]["step"]
    if search["seconds"] is not None:
        report["seconds"] += search["seconds"]
    report.setdefault("refinements", []).append(
        {"method": search["method"], "previous": previous, "search": search}
    )
    report["files_sha256"] = {path.name: sha256(path) for path in sorted(folder.glob("*.png"))}
    save_json(report_path, report)
    return True


if __name__ == "__main__":
    main()
