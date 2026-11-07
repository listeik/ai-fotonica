"""Train a phase-only DOE model through the four-plane Fresnel operator."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from doe.loss import background_power_fraction, reconstruction_loss
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.models.unet import DEFAULT_AMPLITUDE_FLOOR, validate_amplitude_floor
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
    projection_steps: int = 0
    background_weight: float = 0.0
    ssim_weight: float = 0.2
    worst_pcc_weight: float = 0.0
    selection: str = "loss"
    amplitude_floor: float = DEFAULT_AMPLITUDE_FLOOR

    def __post_init__(self) -> None:
        sizes = (self.epochs, self.train_count, self.val_count, self.batch_size, self.threads)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError(
                "epochs, sample counts, batch size and threads must be positive integers"
            )
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
        if type(self.projection_steps) is not int or not 0 <= self.projection_steps <= 8:
            raise ValueError("projection_steps must be an integer in [0, 8]")
        weights = (self.background_weight, self.ssim_weight, self.worst_pcc_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("loss weights must be finite and nonnegative")
        if self.selection not in ("loss", "pcc-ssim"):
            raise ValueError("selection must be 'loss' or 'pcc-ssim'")
        validate_amplitude_floor(self.amplitude_floor)


def _save(data: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def _selection_loss(validation: dict, selection: str, reference: dict) -> float:
    if selection == "loss":
        return validation["loss"]
    if validation["mean_pcc"] < reference["mean_pcc"]:
        return math.inf
    return -(validation["mean_pcc"] + validation["mean_ssim"])


def train(
    config: TrainingConfig,
    output: Path,
    device: torch.device,
    *,
    resume: bool = False,
    init_checkpoint: Path | None = None,
) -> list[dict]:
    if resume and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
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

    initialized_from = None
    if init_checkpoint is None:
        model = FresnelUNet(
            config.width,
            config.projection_steps,
            amplitude_floor=config.amplitude_floor,
        ).to(device)
    else:
        model, initialized_from = load_model(init_checkpoint, device)
        if (model.width, model.projection_steps) != (config.width, config.projection_steps):
            raise ValueError("init checkpoint must match config width and projection_steps")
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
        defaults = {
            "projection_steps": 0,
            "background_weight": 0.0,
            "ssim_weight": 0.2,
            "worst_pcc_weight": 0.0,
            "selection": "loss",
            "amplitude_floor": DEFAULT_AMPLITUDE_FLOOR,
        }
        if (
            state["format_version"] != CHECKPOINT_FORMAT
            or state["architecture"] != model.architecture
            or {**defaults, **state["settings"]} != run_settings
        ):
            raise ValueError("resume requires the same architecture, data, device and settings")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        initialized_from = state.get("initialized_from", history[0].get("initialized_from"))
        start_epoch = state["epoch"] + 1
        best_loss, stale, step = state["best_loss"], state["stale"], state["step"]
        if config.epochs < start_epoch:
            raise ValueError("epochs must exceed the last completed epoch when resuming")
    else:
        reference = None
        if model.amplitude_floor != config.amplitude_floor:
            reference = validation_metrics(model, propagator, val_batches, device)
            model.amplitude_floor = config.amplitude_floor
        initial = {"split": val_split, **validation_metrics(model, propagator, val_batches, device)}
        history = [{"epoch": 0, "step": 0, "validation": initial}]
        start_epoch, best_loss, stale, step = 1, float("inf"), 0, 0
        if initialized_from is not None:
            history[0]["initialized_from"] = initialized_from
            if reference is not None:
                history[0]["reference_validation"] = {"split": val_split, **reference}
            reference = reference if reference is not None else initial
            best_loss = min(
                _selection_loss(initial, config.selection, reference),
                _selection_loss(reference, config.selection, reference),
            )
        log_path.write_text(json.dumps(history[0]) + "\n", encoding="utf-8")
        print(json.dumps(history[0]), flush=True)

    reference = history[0].get("reference_validation", history[0]["validation"])
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
            amplitude = propagator.amplitude(phase)
            loss = reconstruction_loss(
                amplitude,
                target,
                ssim_weight=config.ssim_weight,
                worst_pcc_weight=config.worst_pcc_weight,
            )
            if config.background_weight:
                # A blank target plane cannot put its power inside a foreground object.
                valid = (target > 2 / 255).any(dim=(-2, -1))
                background = background_power_fraction(amplitude, target)
                loss = loss + config.background_weight * (background * valid).sum() / (
                    valid.sum().clamp_min(1)
                )
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
            "architecture": model.architecture,
            "width": config.width,
            "projection_steps": config.projection_steps,
            "amplitude_floor": config.amplitude_floor,
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "config": config_dict,
            "dataset": DATASET_VERSION,
            "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": str(torch.__version__),
            "validation": validation,
            "initialized_from": initialized_from,
        }
        if not math.isfinite(validation["loss"]):
            raise FloatingPointError("validation loss is non-finite")
        criterion = _selection_loss(validation, config.selection, reference)
        if criterion < best_loss:
            best_loss, stale = criterion, 0
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
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true", help="resume from OUTPUT/last.pt")
    start.add_argument(
        "--init-checkpoint", type=Path, help="fine-tune weights with a fresh optimizer"
    )
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
    train(
        config,
        args.output,
        torch.device(device),
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
