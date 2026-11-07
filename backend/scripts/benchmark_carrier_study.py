"""Benchmark CPU forward and complete Neural export with a frozen input order."""

import argparse
import json
import platform
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from evaluate_carrier_study import (
    FresnelPropagator,
    checkpoint_arg,
    file_hash,
    load_model,
    load_test,
    synthesize,
)


def statistics(values):
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.quantile(values, 0.9)),
            "samples_ms": values}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=checkpoint_arg)
    parser.add_argument("--split", default="people_test")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if len(dict(args.model)) != len(args.model):
        parser.error("model names must be unique")
    if min(args.threads, args.iterations, args.samples) < 1 or args.warmup < 0:
        parser.error("expected positive threads, iterations and samples, nonnegative warmup")
    if args.output.resolve() in {args.test.resolve(), *(path.resolve() for _, path in args.model)}:
        parser.error("output must differ from inputs")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    splits, ids, _ = load_test(args.test)
    if args.split not in splits:
        parser.error("requested benchmark split is absent")
    count = min(args.samples, len(splits[args.split]))
    targets = splits[args.split][:count].float().div(255)
    models = {}
    for name, path in args.model:
        model, version = load_model(path, "cpu")
        models[name] = {"model": model, "version": version, "path": str(path.resolve()),
                        "sha256": file_hash(path),
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "forward": [], "end_to_end": []}
    physics = FresnelPropagator()
    rng = random.Random(20261115)
    for mode in ("forward", "end_to_end"):
        for step in range(args.warmup + args.iterations):
            target = targets[step % count]
            order = list(models)
            rng.shuffle(order)
            for name in order:
                model = models[name]["model"]
                started = perf_counter()
                if mode == "forward":
                    phase = model(target[None])
                else:
                    result = synthesize(target, "neural", physics, model=model)
                duration = (perf_counter() - started) * 1000
                if mode == "forward":
                    if phase.shape != (1, 1, 128, 128) or not torch.isfinite(phase).all():
                        raise ValueError("Invalid forward output during benchmark")
                elif result.iterations != 0 or len(result.reconstruction_pngs) != 4:
                    raise ValueError("Invalid Neural synthesis during benchmark")
                if step >= args.warmup:
                    models[name][mode].append(duration)
    report = {
        "protocol": {"device": "cpu", "threads": args.threads, "interop_threads": 1,
                     "warmup_per_mode_and_model": args.warmup,
                     "measured_iterations_per_mode_and_model": args.iterations,
                     "sample_count": count, "target_ids": ids[args.split][:count],
                     "input_sha256": file_hash(args.test), "model_order": "seeded interleaving",
                     "forward": "model forward only, batch=1; initialization excluded",
                     "end_to_end": "synthesize Neural wall time including forward, phase PNG, "
                                   "CPU propagation, reconstruction PNGs and global metrics; "
                                   "model initialization, file writes and HTTP excluded",
                     "platform": platform.platform(), "torch": str(torch.__version__),
                     "evaluator_sha256": file_hash(Path(__file__).resolve())},
        "models": {
            name: {key: statistics(value) if key in ("forward", "end_to_end") else value
                   for key, value in row.items() if key != "model"}
            for name, row in models.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
