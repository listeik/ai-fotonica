"""Train and evaluate matched single-forward DOE models without changing deployment."""

import argparse
import json
import os
import platform
from hashlib import file_digest
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import __version__ as pillow_version
from skimage import __version__ as skimage_version
from torch.utils.data import DataLoader

from doe.loss import background_power_fraction, reconstruction_loss
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.models.residual_fourier import ARCHITECTURE, MultiPlaneResidualFourierUNet
from doe.physics import FresnelPropagator, quantize_phase

from .dataset import DATASET_VERSION, SyntheticTargets

BASELINE = "projection-v2"
FOURIER = "residual-fourier"
FOURIER_V2 = "residual-fourier-v2"
MODELS = (BASELINE, FOURIER)


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device: torch.device) -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "scikit_image": skimage_version,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "threads": torch.get_num_threads(),
    }


def make_model(name: str, settings: dict) -> torch.nn.Module:
    kwargs = {"width": settings["width"], "amplitude_floor": settings["amplitude_floor"]}
    if name == BASELINE:
        return FresnelUNet(projection_steps=settings["projection_steps"], **kwargs)
    if name in (FOURIER, FOURIER_V2):
        return MultiPlaneResidualFourierUNet(
            modes=settings["fourier_modes"], complex_correction=name == FOURIER_V2, **kwargs
        )
    raise ValueError(f"Unknown study model {name!r}")


def load_study_model(path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["architecture"] != ARCHITECTURE:
        return load_model(path, device)[0]
    if state["format_version"] != CHECKPOINT_FORMAT or state["step"] < 1:
        raise ValueError("Expected a trained Fourier study checkpoint")
    model = make_model(FOURIER, state["config"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def objective(amplitude: torch.Tensor, target: torch.Tensor, settings: dict) -> torch.Tensor:
    return (
        reconstruction_loss(amplitude, target, ssim_weight=settings["ssim_weight"])
        + settings["background_weight"] * background_power_fraction(amplitude, target).mean()
    )


@torch.inference_mode()
def validation_loss(model, physics, batches, settings: dict, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for target in batches:
        target = target.to(device)
        phase = quantize_phase(model(target).squeeze(1))
        total += float(objective(physics.amplitude(phase), target, settings)) * len(target)
    return total / len(batches.dataset)


def train_one(name, seed, settings, output, device, train_targets, val_targets) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = make_model(name, settings).to(device)
    physics = FresnelPropagator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    architecture = model.architecture
    validation = DataLoader(val_targets, batch_size=settings["batch_size"])
    history = [
        {
            "epoch": 0,
            "step": 0,
            "val_loss": validation_loss(model, physics, validation, settings, device),
        }
    ]
    step, best_loss, best_epoch = 0, float("inf"), 0
    elapsed = 0.0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.Generator().manual_seed(seed + epoch)
        batches = DataLoader(
            train_targets, batch_size=settings["batch_size"], shuffle=True, generator=order
        )
        synchronize(device)
        start = perf_counter()
        training_loss = torch.zeros((), device=device)
        for target in batches:
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            phase = model(target).squeeze(1)
            phase = quantize_phase(phase.detach()) + (phase - phase.detach())
            loss = objective(physics.amplitude(phase), target, settings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            training_loss += loss.detach() * len(target)
            step += 1
        synchronize(device)
        seconds = perf_counter() - start
        elapsed += seconds
        score = validation_loss(model, physics, validation, settings, device)
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(training_loss) / len(train_targets),
            "val_loss": score,
            "train_seconds": seconds,
        }
        history.append(record)
        if not np.isfinite(score):
            raise FloatingPointError("Validation objective is nonfinite")
        if score < best_loss:
            best_loss, best_epoch = score, epoch
            temporary = output / "best.tmp"
            torch.save(
                {
                    "format_version": CHECKPOINT_FORMAT,
                    "architecture": architecture,
                    "config": settings,
                    "width": settings["width"],
                    "amplitude_floor": settings["amplitude_floor"],
                    "projection_steps": settings["projection_steps"] if name == BASELINE else 0,
                    "epoch": epoch,
                    "step": step,
                    "seed": seed,
                    "model": {key: value.cpu() for key, value in model.state_dict().items()},
                    "dataset": DATASET_VERSION,
                    "validation": {"loss": score},
                },
                temporary,
            )
            temporary.replace(output / "best.pt")
        write_json(output / "history.json", history)
        print(json.dumps({"model": name, "seed": seed, **record}), flush=True)
    write_json(
        output / "training.json",
        {
            "architecture": architecture,
            "seed": seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch,
            "optimizer_steps": step,
            "samples_seen": settings["train_count"] * settings["epochs"],
            "train_seconds": elapsed,
            "checkpoint_sha256": digest(output / "best.pt"),
        },
    )


def train_all(
    settings: dict, output: Path, device: torch.device, names: tuple[str, ...] = MODELS
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", settings)
    write_json(output / "training-environment.json", environment(device))
    train_targets, val_targets = (
        torch.stack(list(SyntheticTargets(split, settings[key], settings["data_seed"])))
        for split, key in (("train", "train_count"), ("val", "val_count"))
    )
    for index, seed in enumerate(settings["seeds"]):
        order = names if index % 2 == 0 else reversed(names)
        for name in order:
            train_one(
                name, seed, settings, output / f"{name}-{seed}", device, train_targets, val_targets
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--device", choices=("cuda", "cpu"), required=True)
    train.add_argument(
        "--model", choices=(*MODELS, FOURIER_V2), help="train only this architecture"
    )
    train.add_argument("--epochs", type=int, help="override the epoch count")
    args = parser.parse_args()
    device = torch.device(args.device)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        settings["epochs"] = args.epochs
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    train_all(settings, args.output, device, (args.model,) if args.model else MODELS)


if __name__ == "__main__":
    main()
