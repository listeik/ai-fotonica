"""Train a phase-only DOE model through the four-plane Fresnel operator."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from doe.loss import reconstruction_loss
from doe.models import ARCHITECTURE, CHECKPOINT_FORMAT, FresnelUNet
from doe.physics import FresnelPropagator, quantize_phase

from .dataset import DATASET_VERSION, SyntheticTargets
from .validate import validation_metrics


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 24
    train_count: int = 768
    val_count: int = 32
    width: int = 12
    batch_size: int = 8
    learning_rate: float = 0.001
    patience: int = 6
    seed: int = 2026
    threads: int = 4
    overfit: bool = False

    def __post_init__(self) -> None:
        sizes = (self.epochs, self.train_count, self.val_count, self.batch_size, self.threads)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("epochs, sample counts, batch size and threads must be positive integers")
        if type(self.width) is not int or not 4 <= self.width <= 64:
            raise ValueError("width must be an integer between 4 and 64")
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if type(self.overfit) is not bool:
            raise ValueError("overfit must be a boolean")


def _save(data: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def train(
    config: TrainingConfig,
    output: Path,
    device: torch.device,
    *,
    resume: bool = False,
) -> list[dict]:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    log_path = output / "metrics.jsonl"
    if not resume and any(path.exists() for path in (best_path, last_path, log_path)):
        raise FileExistsError(f"{output} contains a previous run. Use --resume or a new --output.")

    model = FresnelUNet(config.width).to(device)
    propagator = FresnelPropagator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    training_data = SyntheticTargets("train", config.train_count, config.seed)
    val_data = (
        training_data if config.overfit else SyntheticTargets("val", config.val_count, config.seed)
    )
    val_batches = DataLoader(val_data, batch_size=config.batch_size)
    val_split = "train" if config.overfit else "val"
    config_dict = asdict(config)
    run_settings = {key: value for key, value in config_dict.items() if key != "epochs"}
    run_settings.update(dataset=DATASET_VERSION, device=device.type)
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if (
            state["format_version"] != CHECKPOINT_FORMAT
            or state["architecture"] != ARCHITECTURE
            or state["settings"] != run_settings
        ):
            raise ValueError("resume requires the same architecture, data, device and settings")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        best_loss, stale, step = state["best_loss"], state["stale"], state["step"]
        if config.epochs < start_epoch:
            raise ValueError("epochs must exceed the last completed epoch when resuming")
    else:
        initial = {"split": val_split, **validation_metrics(model, propagator, val_batches, device)}
        history = [{"epoch": 0, "step": 0, "validation": initial}]
        start_epoch, best_loss, stale, step = 1, float("inf"), 0, 0
        log_path.write_text(json.dumps(history[0]) + "\n", encoding="utf-8")
        print(json.dumps(history[0]), flush=True)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        # The shuffle for each epoch depends only on its number and the seed.
        # Resume can reproduce it without serializing a live DataLoader.
        order = torch.Generator().manual_seed(config.seed + epoch)
        batches = DataLoader(
            training_data, batch_size=config.batch_size, shuffle=True, generator=order
        )
        train_loss = 0.0
        for target in batches:
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            phase = model(target).squeeze(1)
            # Use the hardware levels on the forward pass and the derivative
            # of the unquantized phase on the backward pass.
            phase = quantize_phase(phase.detach()) + (phase - phase.detach())
            loss = reconstruction_loss(propagator.amplitude(phase), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, error_if_nonfinite=True)
            optimizer.step()
            train_loss += float(loss.detach()) * len(target)
            step += 1

        validation = {
            "split": val_split,
            **validation_metrics(model, propagator, val_batches, device),
        }
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss / len(training_data),
            "validation": validation,
        }
        history.append(record)
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT,
            "architecture": ARCHITECTURE,
            "width": config.width,
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "config": config_dict,
            "dataset": DATASET_VERSION,
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": str(torch.__version__),
            "validation": validation,
        }
        if not math.isfinite(validation["loss"]):
            raise FloatingPointError("validation loss is non-finite")
        if validation["loss"] < best_loss:
            best_loss, stale = validation["loss"], 0
            _save(checkpoint, best_path)
        else:
            stale += 1
        _save(
            {
                **checkpoint,
                "optimizer": optimizer.state_dict(),
                "settings": run_settings,
                "best_loss": best_loss,
                "stale": stale,
                "history": history,
            },
            last_path,
        )
        # Keep the log consistent with the last checkpoint even after resume.
        temporary_log = log_path.with_suffix(".tmp")
        temporary_log.write_text(
            "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
        )
        temporary_log.replace(log_path)
        print(json.dumps(record), flush=True)
        if config.patience and stale >= config.patience:
            break
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON file with training hyperparameters")
    parser.add_argument("--output", type=Path, default=Path("runs/doe"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="resume from OUTPUT/last.pt")
    for field in fields(TrainingConfig):
        flag = f"--{field.name.replace('_', '-')}"
        if isinstance(field.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(flag, type=type(field.default))

    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    values.update(
        (field.name, value)
        for field in fields(TrainingConfig)
        if (value := getattr(args, field.name)) is not None
    )
    try:
        config = TrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    print(json.dumps({"device": device, "config": asdict(config), "dataset": DATASET_VERSION}))
    train(config, args.output, torch.device(device), resume=args.resume)


if __name__ == "__main__":
    main()
