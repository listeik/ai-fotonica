"""Bounded preprocessing experiments scored against immutable original targets."""

import asyncio
import base64
import hashlib
import logging
import math
import os
from dataclasses import asdict, dataclass
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageFilter
from starlette.concurrency import run_in_threadpool

from doe.metrics import evaluate, windowed_ssim
from doe.physics import gray_to_png, png_to_gray, png_to_phase
from doe.research import analyze_targets, diagnose_phase
from doe.synthesis import Synthesis, synthesize

LOGGER = logging.getLogger(__name__)
TERMINAL = {"completed", "cancelled", "error"}


def encoded(data):
    return base64.b64encode(data).decode("ascii")


@dataclass(frozen=True)
class Policy:
    plane_tolerance: float = 0.002
    mean_tolerance: float = 0.0
    minimum_gain: float = 0.0005

    @classmethod
    def environment(cls):
        values = {
            name: float(os.getenv("DOE_SMART_" + name.upper(), str(default)))
            for name, default in asdict(cls()).items()
        }
        if any(not math.isfinite(value) or not 0 <= value <= 0.05 for value in values.values()):
            raise ValueError("Smart selection tolerances must be finite and between 0 and 0.05")
        return cls(**values)

    def score(self, metrics):
        return sum(metrics[key] for key in ("mean_pcc", "mean_ssim", "worst_pcc", "worst_ssim")) / 4

    def accepts(self, candidate, baseline):
        return (
            all(
                new[key] >= original[key] - self.plane_tolerance
                for new, original in zip(candidate["planes"], baseline["planes"], strict=True)
                for key in ("pcc", "ssim")
            )
            and all(
                candidate[key] >= baseline[key] - self.mean_tolerance
                for key in ("mean_pcc", "mean_ssim")
            )
            and all(
                candidate[key] >= baseline[key] - self.plane_tolerance
                for key in ("worst_pcc", "worst_ssim")
            )
            and self.score(candidate) > self.score(baseline) + self.minimum_gain
        )


def analyze(originals):
    report = analyze_targets(originals)
    for plane, target in zip(report["planes"], originals, strict=True):
        plane.update(
            brightness=float(target.mean()),
            active_fraction=float((target > 2 / 255).float().mean()),
        )
    report["similarity"] = [
        evaluate(originals[[i] * 4], originals[[j] * 4])["mean_pcc"]
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    report["similarity_pairs"] = [[i + 1, j + 1] for i in range(4) for j in range(i + 1, 4)]
    return report


def candidate_configs(report, budget):
    planes = report["planes"]
    noisy = max(p["high_frequency_fraction"] for p in planes) > 0.08
    dim = min(p["brightness"] for p in planes) < 0.3
    low_contrast = min(p["contrast"] for p in planes) < 0.2
    gamma = [0.9, 0.95, 1.05, 1.1] if dim else [1.05, 0.95, 1.1, 0.9]
    variants = [[{"method": "gamma", "value": value}] for value in gamma]
    variants += [[{"method": "brightness", "gain": value}] for value in (1.05, 0.95)]
    contrast = [[{"method": "contrast", "strength": value}] for value in (0.1, 0.2)]
    filters = [[{"method": "denoise", "strength": value}] for value in (0.15, 0.3)] + [
        [{"method": "frequency", "strength": value, "cutoff": 0.3}] for value in (0.1, 0.2)
    ]
    variants = (contrast if low_contrast else []) + (filters if noisy else []) + variants
    variants += ([] if low_contrast else contrast) + ([] if noisy else filters)
    variants += [
        [{"method": "gamma", "value": gamma[0]}, {"method": "denoise", "strength": 0.15}],
        [{"method": "contrast", "strength": 0.1}, {"method": "brightness", "gain": 1.05}],
        [
            {"method": "frequency", "strength": 0.1, "cutoff": 0.3},
            {"method": "gamma", "value": gamma[0]},
        ],
    ]
    return [{"operations": []}] + [{"operations": ops} for ops in variants[: budget - 1]]


def preprocess(originals, config):
    planes = []
    for original in originals.numpy():
        image = original.copy()
        for operation in config["operations"]:
            method = operation["method"]
            if method == "gamma":
                image = image ** operation["value"]
            elif method == "brightness":
                image = image * operation["gain"]
            elif method == "contrast":
                low, high = np.quantile(image, [0.01, 0.99])
                stretched = (image - low) / max(float(high - low), 1 / 255)
                image = image + operation["strength"] * (stretched.clip(0, 1) - image)
            elif method == "denoise":
                source = Image.fromarray(np.rint(image * 255).astype(np.uint8))
                smoothed = np.asarray(source.filter(ImageFilter.MedianFilter(3))) / 255
                image = image + operation["strength"] * (smoothed - image)
            elif method == "frequency":
                radius = np.hypot(np.fft.fftfreq(128)[:, None], np.fft.rfftfreq(128)[None, :])
                transfer = 1 - operation["strength"] * (
                    1 - np.exp(-((radius / operation["cutoff"]) ** 4))
                )
                image = np.fft.irfft2(np.fft.rfft2(image) * transfer, s=image.shape)
            image = image.clip(0, 1)
        # All network inputs follow the same eight-bit PNG protocol.
        planes.append(png_to_gray(gray_to_png(torch.from_numpy(image))))
    return torch.stack(planes)


def assess(result, originals, inputs, physics, response):
    pixels = torch.stack([png_to_gray(png) for png in result.reconstruction_pngs])
    run = response(result)
    run["metrics"] = evaluate(pixels, originals, physics.distances_m)
    diagnostics = diagnose_phase(result.phase_png, originals, physics)
    return {
        "run": run,
        "windowed_ssim": windowed_ssim(pixels, originals),
        "background_power": [1 - plane["power_in_target"] for plane in diagnostics["planes"]],
        "adapted_metrics": evaluate(pixels, inputs, physics.distances_m),
        "input_mae": (inputs - originals).abs().mean((-2, -1)).tolist(),
    }


def baseline_result(phase_png, originals, physics, seconds):
    # Reconstruct only from the exact submitted phase, preserving its PNG bytes.
    from doe.loss import normalize_amplitude

    phase = png_to_phase(phase_png).to(physics.input_axis_m.device)
    with torch.no_grad():
        pixels = normalize_amplitude(physics.amplitude(phase)).cpu()
    images = tuple(gray_to_png(plane) for plane in pixels)
    return Synthesis(phase_png, images, {}, seconds, 0)


class SmartJobs:
    """Single-process jobs with a bounded lifetime and cooperative cancellation."""

    def __init__(self, state, response):
        self.state, self.response = state, response
        self.jobs = {}
        self.tasks = set()
        self.policy = Policy.environment()
        self.budget = int(os.getenv("DOE_SMART_MAX_CANDIDATES", "16"))
        if not 2 <= self.budget <= 20:
            raise ValueError("DOE_SMART_MAX_CANDIDATES must be between 2 and 20")

    def clean(self):
        now = monotonic()
        for identifier, job in list(self.jobs.items()):
            if job["status"] in TERMINAL and now - job["_finished"] > 900:
                del self.jobs[identifier]
        finished = [key for key, job in self.jobs.items() if job["status"] in TERMINAL]
        for key in finished[:-3]:
            del self.jobs[key]

    def start(self, originals, phase_png, seconds, mode):
        self.clean()
        identifier = str(uuid4())
        # Reserve before scheduling the task, sharing the ordinary synthesis lock.
        self.state.limiter.acquire_on_behalf_of_nowait(identifier)
        job = {
            "id": identifier,
            "status": "analysis",
            "mode": mode,
            "checked": 0,
            "total": 0,
            "cancel_requested": False,
            "candidates": [],
            "selection_policy": asdict(self.policy),
            "originals": [encoded(gray_to_png(plane)) for plane in originals],
            "original_sha256": [
                hashlib.sha256(gray_to_png(plane)).hexdigest() for plane in originals
            ],
            "forward_count": 0,
            "total_seconds": 0,
            "best_index": 0,
            "improved": False,
            "error": None,
        }
        self.jobs[identifier] = job
        task = asyncio.create_task(self.run(job, originals.clone(), phase_png, seconds))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return job

    async def run(self, job, originals, phase_png, seconds):
        start = perf_counter()
        try:
            job["analysis"] = await run_in_threadpool(analyze, originals)
            configs = candidate_configs(job["analysis"], self.budget)
            job["total"] = len(configs)
            job["status"] = "search"
            baseline = await run_in_threadpool(
                baseline_result, phase_png, originals, self.state.propagator, seconds
            )
            for index, config in enumerate(configs):
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                    return
                inputs = await run_in_threadpool(preprocess, originals, config)
                result = baseline
                if index:
                    result = await run_in_threadpool(
                        synthesize, inputs, "neural", self.state.propagator, model=self.state.model
                    )
                    job["forward_count"] += 1
                entry = await run_in_threadpool(
                    assess, result, originals, inputs, self.state.propagator, self.response
                )
                entry.update(
                    config=config,
                    index=index,
                    inputs=[encoded(gray_to_png(plane)) for plane in inputs],
                    eligible=False,
                )
                if index:
                    entry["eligible"] = self.policy.accepts(
                        entry["run"]["metrics"], job["candidates"][0]["run"]["metrics"]
                    )
                job["candidates"].append(entry)
                best = job["candidates"][job["best_index"]]
                if entry["eligible"] and self.policy.score(
                    entry["run"]["metrics"]
                ) > self.policy.score(best["run"]["metrics"]):
                    job["best_index"] = index
                job["checked"] += 1
            job["status"] = "cancelled" if job["cancel_requested"] else "completed"
            job["improved"] = job["best_index"] != 0 and job["status"] == "completed"
            LOGGER.info(
                "Smart job %s checked %s candidates, selected %s",
                job["id"],
                job["checked"],
                job["best_index"],
            )
        except Exception:
            LOGGER.exception("Smart optimization failed for %s", job["id"])
            job.update(status="error", error="Не удалось завершить поиск улучшения.")
        finally:
            job["total_seconds"] = perf_counter() - start
            job["_finished"] = monotonic()
            self.state.limiter.release_on_behalf_of(job["id"])
            self.clean()

    def snapshot(self, identifier, full=False):
        self.clean()
        job = self.jobs[identifier]
        if full:
            return {key: value for key, value in job.items() if not key.startswith("_")}
        return {
            key: job.get(key)
            for key in (
                "id",
                "status",
                "mode",
                "checked",
                "total",
                "cancel_requested",
                "forward_count",
                "best_index",
                "improved",
                "error",
                "total_seconds",
            )
        }

    async def close(self):
        for job in self.jobs.values():
            job["cancel_requested"] = True
        if self.tasks:
            await asyncio.gather(*list(self.tasks))
        self.jobs.clear()
