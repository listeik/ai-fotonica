"""Reproducible, single-factor experiments for carrier-conditioned DOE synthesis."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from doe.loss import background_power_fraction, global_ssim, normalize_amplitude
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.physics import FresnelPropagator, quantize_phase

from .multiplane_study import digest, environment, write_json

VALIDATION_WEIGHTS = {
    "broad_val": 0.25,
    "new_broad_val": 0.25,
    "natural_val": 0.30,
    "people_val": 0.15,
    "mixed_val": 0.05,
}
METRICS = ("pcc", "global_ssim", "windowed_ssim")
LOSS_WEIGHTS = {
    "legacy": {
        "pcc": 1.0,
        "global": 1.0,
        "windowed": 0.0,
        "peak_mse": 0.1,
        "fitted_mse": 0.0,
        "background": 0.05,
        "separate_worst": 0.25,
    },
    "windowed": {
        "pcc": 1.0,
        "global": 0.25,
        "windowed": 0.75,
        "peak_mse": 0.1,
        "fitted_mse": 0.0,
        "background": 0.05,
        "mixed_worst": 0.25,
    },
    "balanced": {
        "pcc": 1.5,
        "global": 0.25,
        "windowed": 0.5,
        "peak_mse": 0.03,
        "fitted_mse": 0.0,
        "background": 0.1,
        "soft_worst_fraction": 0.25,
    },
    "amplitude": {
        "pcc": 1.5,
        "global": 0.25,
        "windowed": 0.5,
        "peak_mse": 0.03,
        "fitted_mse": 2.0,
        "background": 0.1,
        "soft_worst_fraction": 0.25,
    },
}


def windowed_ssim(x, y, window_size=7):
    """Uniform valid windows, sample covariance, data_range=1."""
    if type(window_size) is not int or window_size < 3 or window_size % 2 != 1:
        raise ValueError("SSIM window size must be an odd integer >=3")
    shape = x.shape
    x, y = [value.reshape(-1, 1, *shape[-2:]) for value in (x, y)]
    terms = torch.cat((x, y, x.square(), y.square(), x * y), 1)
    values = F.avg_pool2d(
        F.avg_pool2d(terms, (1, window_size), stride=1), (window_size, 1), stride=1
    )
    mx, my, exx, eyy, exy = values.unbind(1)
    correction = window_size**2 / (window_size**2 - 1)
    vx = (exx - mx.square()).clamp_min(0) * correction
    vy = (eyy - my.square()).clamp_min(0) * correction
    covariance = (exy - mx * my) * correction
    score = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    )
    return score.mean((-2, -1)).reshape(shape[:-2])


def pcc(x, y):
    xc, yc = [value - value.mean((-2, -1), keepdim=True) for value in (x, y)]
    denominator = (xc.square().sum((-2, -1)) * yc.square().sum((-2, -1))).clamp_min(1e-12)
    return (xc * yc).sum((-2, -1)) / denominator.sqrt()


def reconstruction(phase, target, optics, *, gradient):
    exact = quantize_phase(phase.detach())
    quantized = exact + (phase - phase.detach()) if gradient else exact
    amplitude = optics.amplitude(quantized)
    image = normalize_amplitude(amplitude)
    exact_pixels = image.detach().mul(255).round().div(255)
    pixels = exact_pixels + (image - image.detach()) if gradient else exact_pixels
    return amplitude, image, pixels


def optical_objective(amplitude, image, pixels, target, profile, window_size=7):
    correlations = pcc(pixels, target)
    global_scores = global_ssim(pixels, target)
    mse = (image - target).square().mean((-2, -1))
    background = background_power_fraction(amplitude, target)
    parts = {
        "amplitude_mse": mse.mean(),
        "pcc_loss": 1 - correlations.mean(),
        "global_ssim_loss": 1 - global_scores.mean(),
        "background": background.mean(),
    }
    if profile == "legacy":
        loss = 2 - correlations.mean() - global_scores.mean()
        loss += 0.25 * (2 - correlations.amin(-1).mean() - global_scores.amin(-1).mean())
        loss += 0.1 * mse.mean() + 0.05 * background.mean()
    else:
        local_scores = windowed_ssim(pixels, target, window_size)
        parts["windowed_ssim_loss"] = 1 - local_scores.mean()
        if profile == "windowed":
            ssim = 0.25 * global_scores + 0.75 * local_scores
            loss = 2 - correlations.mean() - ssim.mean()
            loss += 0.25 * (2 - correlations.amin(-1).mean() - ssim.amin(-1).mean())
            loss += 0.1 * mse.mean() + 0.05 * background.mean()
        elif profile in ("balanced", "amplitude"):
            weights = LOSS_WEIGHTS[profile]
            error = (
                weights["pcc"] * (1 - correlations)
                + weights["global"] * (1 - global_scores)
                + weights["windowed"] * (1 - local_scores)
                + weights["peak_mse"] * mse
                + weights["background"] * background
            )
            if profile == "amplitude":
                # A fitted positive amplitude scale offers a smooth complement to peak normalization.
                scale = (amplitude * target).sum((-2, -1), keepdim=True) / (
                    amplitude.square().sum((-2, -1), keepdim=True).clamp_min(1e-8)
                )
                fitted_mse = (scale * amplitude - target).square().mean((-2, -1))
                parts["fitted_amplitude_mse"] = fitted_mse.mean()
                error = error + weights["fitted_mse"] * fitted_mse
            # Smooth worst-plane objective; subtract log(4) so equal planes retain their scale.
            soft_worst = 0.2 * (torch.logsumexp(error / 0.2, -1) - math.log(4))
            loss = 0.75 * error.mean() + 0.25 * soft_worst.mean()
        else:
            raise ValueError(f"Unknown objective profile: {profile}")
    return loss, {key: value.detach() for key, value in parts.items()}


@dataclass(frozen=True)
class StudyConfig:
    name: str
    width: int = 16
    projections: int = 6
    fourier_modes: int = 0
    carrier_scale: int = 1
    amplitude_floor: float = 0.01
    profile: str = "legacy"
    seed: int = 20261114
    epochs: int = 8
    batches: int = 128
    batch_size: int = 16
    lr: float = 0.001
    teacher_weight: float = 0.0
    teacher_phase_weight: float = 0.0
    supervised_epochs: int = 0
    window_size: int = 7


def batches(data, broad, config, epoch, device, teacher=None):
    rng = np.random.default_rng(config.seed + 100003 * epoch)
    natural = data["natural_bank"]
    bank = data["face_bank"]
    half, quarter = config.batch_size // 2, config.batch_size // 4
    for _ in range(config.batches):
        reference = None
        if teacher is None:
            photos = natural[rng.integers(len(natural), size=(half, 4))]
        else:
            selected = rng.choice(len(teacher["targets"]), half, replace=False)
            photos = teacher["targets"][selected]
            reference = {
                "reconstructions": teacher["reconstructions"][selected].to(device).float() / 255,
                "phases": teacher["phases"][selected].to(device),
            }
        diverse = torch.from_numpy(
            np.stack([broad[i][rng.permutation(4)] for i in rng.integers(len(broad), size=quarter)])
        )
        faces = torch.stack(
            [
                bank[rng.choice(len(bank), 4, replace=False), rng.integers(10, size=4)]
                for _ in range(quarter)
            ]
        )
        yield torch.cat((photos, diverse, faces)).to(device).float() / 255, reference


def summarize(matrix):
    mean = matrix.mean((0, 1))
    worst = matrix.amin(1).mean(0)
    joint = matrix[..., :2].amin((1, 2))
    score = 0.35 * mean[0] + 0.2 * mean[1] + 0.2 * mean[2] + 0.15 * worst[0] + 0.1 * worst[1]
    return {
        "samples": len(matrix),
        "mean": dict(zip(METRICS, mean.tolist(), strict=True)),
        "worst": dict(zip(METRICS, worst.tolist(), strict=True)),
        "plane_means": matrix.mean(0).tolist(),
        "mean_weakest_joint": float(joint.mean()),
        "p10_weakest_joint": float(torch.quantile(joint, 0.1)),
        "score": float(score),
    }


@torch.inference_mode()
def validation(model, optics, data, device):
    model.eval()
    result = {}
    for name, raw in data["validation"].items():
        rows = []
        for batch in raw.split(8):
            target = batch.to(device).float() / 255
            phase = model(target).squeeze(1)
            _, _, pixels = reconstruction(phase, target, optics, gradient=False)
            rows.append(
                torch.stack(
                    (
                        pcc(pixels, target),
                        global_ssim(pixels, target),
                        windowed_ssim(pixels, target),
                    ),
                    -1,
                ).cpu()
            )
        result[name] = summarize(torch.cat(rows))
    if set(VALIDATION_WEIGHTS) - result.keys():
        raise ValueError(f"Missing validation groups: {set(VALIDATION_WEIGHTS) - result.keys()}")
    result["selection"] = sum(
        weight * result[name]["score"] for name, weight in VALIDATION_WEIGHTS.items()
    )
    return result


def checkpoint(model, config, step, epoch, metrics, provenance):
    state = {
        "format_version": CHECKPOINT_FORMAT,
        "architecture": model.architecture,
        "width": model.width,
        "projection_steps": model.projection_steps,
        "amplitude_floor": model.amplitude_floor,
        "config": asdict(config),
        "step": step,
        "epoch": epoch,
        "stage": "carrier-controlled-study-v1",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "validation": metrics,
        "study": provenance,
    }
    if model.carrier_conditioning:
        state["carrier_conditioning"] = {"scale": model.carrier_scale}
    if model.fourier_modes:
        state["fourier_bottleneck"] = {"modes": model.fourier_modes}
    return state


def save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def initialize(config, source, device):
    model = FresnelUNet(
        width=config.width,
        projection_steps=config.projections,
        amplitude_floor=config.amplitude_floor,
        carrier_conditioning=True,
        carrier_scale=config.carrier_scale,
        fourier_modes=config.fourier_modes,
    )
    initial_step = 0
    if source is not None:
        old, _ = load_model(source, "cpu")
        if old.width != config.width:
            raise ValueError(
                "Warm-start width mismatch; use scratch runs for fair width comparison"
            )
        state = old.state_dict()
        # Vary depth while preserving each shared physical layer's learned parameters.
        if old.projection_steps != config.projections:
            if old.projection_steps == 0:
                raise ValueError("Projection transfer requires a projected source model")
            for key in ("plane_logits", "relaxation_logits"):
                indices = torch.arange(config.projections).clamp_max(old.projection_steps - 1)
                state[key] = state[key][indices].clone()
        if old.carrier_conditioning is False:
            first = torch.zeros_like(model.encode1[0].weight)
            first[:, :4] = state["encode1.0.weight"]
            state["encode1.0.weight"] = first
        # New spectral branch starts as exact identity; common learned weights are retained.
        target = model.state_dict()
        for key in target:
            if key in state:
                target[key] = state[key]
            elif not key.startswith("fourier."):
                raise ValueError(f"Missing shared parameter: {key}")
        unexpected = set(state) - target.keys()
        if unexpected and not all(key.startswith("fourier.") for key in unexpected):
            raise ValueError(f"Unexpected shared parameters: {unexpected}")
        model.load_state_dict(target, strict=True)
        initial_step = torch.load(source, map_location="cpu", weights_only=True)["step"]
    return model.to(device), initial_step


def train(config, data_path, output, source=None, teacher_path=None):
    if output.exists():
        raise ValueError("Choose a fresh output directory")
    if config.batch_size < 4 or config.batch_size % 4 or config.epochs < 1 or config.batches < 1:
        raise ValueError("Positive training budget and batch_size divisible by four are required")
    if config.profile not in LOSS_WEIGHTS or not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("A supported loss profile and positive finite learning rate are required")
    if not 0 <= config.supervised_epochs <= config.epochs:
        raise ValueError("Supervised epochs must fit the training budget")
    if config.supervised_epochs and config.teacher_weight <= 0 and config.teacher_phase_weight <= 0:
        raise ValueError("Supervised pretraining requires a positive teacher weight")
    if (
        type(config.window_size) is not int
        or config.window_size < 3
        or config.window_size > 128
        or config.window_size % 2 != 1
    ):
        raise ValueError("SSIM window must be odd and fit the 128x128 grid")
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    data_file = data_path / "train-val.pt"
    data = torch.load(data_file, map_location="cpu", weights_only=True)
    manifest = json.loads((data_path / "manifest.json").read_text(encoding="utf-8"))
    broad_path = Path(manifest["broad_train_npy"])
    if (
        digest(data_file) != manifest["train_val_sha256"]
        or digest(broad_path) != manifest["broad_train_sha256"]
    ):
        raise ValueError("Prepared training inputs do not match their hashes")
    broad = np.load(broad_path, mmap_mode="r")
    teacher = (
        torch.load(teacher_path, map_location="cpu", weights_only=True) if teacher_path else None
    )
    if teacher is not None:
        teacher_manifest = json.loads(
            (teacher_path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        if digest(teacher_path) != teacher_manifest["teachers_sha256"]:
            raise ValueError("Teacher data does not match its manifest")
        if digest(data_file) != teacher_manifest["train_val_sha256"]:
            raise ValueError("Teacher and training dataset provenance differ")
        if source is None or digest(source) != teacher_manifest["checkpoint_sha256"]:
            raise ValueError("Teacher and matched-control initialization must share a checkpoint")
        count = len(teacher["targets"])
        if (
            teacher["targets"].dtype != torch.uint8
            or teacher["reconstructions"].dtype != torch.uint8
            or teacher["phases"].dtype != torch.float32
            or teacher["targets"].shape != (count, 4, 128, 128)
            or teacher["reconstructions"].shape != (count, 4, 128, 128)
            or teacher["phases"].shape != (count, 128, 128)
            or not torch.isfinite(teacher["phases"]).all()
        ):
            raise ValueError("Invalid teacher tensors")
    if (
        config.teacher_weight or config.teacher_phase_weight or config.supervised_epochs
    ) and teacher is None:
        raise ValueError("Teacher data is required for supervised training")
    model, initial_step = initialize(config, source, device)
    optics = FresnelPropagator().to(device)
    initial = validation(model, optics, data, device)
    output.mkdir(parents=True)
    provenance = {
        "parent_sha256": digest(source) if source else None,
        "train_val_sha256": digest(data_file),
        "broad_train_sha256": digest(broad_path),
        "teacher_sha256": digest(teacher_path) if teacher_path else None,
        "validation_weights": VALIDATION_WEIGHTS,
        "loss_weights": LOSS_WEIGHTS[config.profile],
        "window_size_in_loss": config.window_size,
        "code_sha256": {
            "training": digest(Path(__file__)),
            "physics": digest(Path(__file__).resolve().parents[1] / "physics.py"),
        },
        "training_mix": "50% natural quartets,25% broad replay,25% faces",
        "environment": environment(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    write_json(output / "config.json", asdict(config))
    write_json(output / "initial-validation.json", initial)
    write_json(output / "provenance.json", provenance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs, eta_min=config.lr * 0.1
    )
    best, best_epoch, best_metrics = (
        (initial["selection"], 0, initial) if source else (-math.inf, 0, None)
    )
    if source:
        save(output / "best.pt", checkpoint(model, config, initial_step, 0, initial, provenance))
    history = []
    started = perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {}
        for target, reference in batches(data, broad, config, epoch, device, teacher):
            optimizer.zero_grad(set_to_none=True)
            phase = model(target).squeeze(1)
            amplitude, image, pixels = reconstruction(phase, target, optics, gradient=True)
            loss, parts = optical_objective(
                amplitude, image, pixels, target, config.profile, config.window_size
            )
            if reference is not None:
                count = len(reference["phases"])
                teacher_loss = F.smooth_l1_loss(
                    pixels[:count], reference["reconstructions"], beta=0.08
                )
                delta = phase[:count] - reference["phases"]
                piston = torch.atan2(
                    delta.sin().mean((-2, -1), keepdim=True),
                    delta.cos().mean((-2, -1), keepdim=True),
                ).detach()
                phase_loss = (1 - (delta - piston).cos()).mean()
                supervised = (
                    config.teacher_weight * teacher_loss + config.teacher_phase_weight * phase_loss
                )
                if config.supervised_epochs:
                    if epoch <= config.supervised_epochs:
                        loss = supervised
                else:
                    loss = loss + supervised
                parts.update(
                    teacher_reconstruction=teacher_loss.detach(), teacher_phase=phase_loss.detach()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            parts.update(loss=loss.detach(), gradient_norm=norm.detach())
            for name, value in parts.items():
                totals[name] = totals.get(name, 0) + value
        scheduler.step()
        metrics = validation(model, optics, data, device)
        step = initial_step + epoch * config.batches
        state = checkpoint(model, config, step, epoch, metrics, provenance)
        if metrics["selection"] > best:
            best, best_epoch, best_metrics = metrics["selection"], epoch, metrics
            save(output / "best.pt", state)
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        save(output / "last.pt", state)
        row = {
            "epoch": epoch,
            "step": step,
            "validation": metrics,
            "best_epoch": best_epoch,
            "training": {name: float(value / config.batches) for name, value in totals.items()},
            "seconds": perf_counter() - started,
        }
        history.append(row)
        write_json(output / "history.json", history)
        write_json(
            output / "result.json",
            {
                "name": config.name,
                "best_epoch": best_epoch,
                "best_validation": best_metrics,
                "sha256": digest(output / "best.pt"),
                "parameters": provenance["parameters"],
                "elapsed": row["seconds"],
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            },
        )
        print(
            f"{config.name} {epoch:02d}/{config.epochs} val={metrics['selection']:.6f} "
            f"best={best:.6f}@{best_epoch} elapsed={row['seconds']:.1f}s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--teachers", type=Path)
    for field in StudyConfig.__dataclass_fields__.values():
        kind = (
            str if isinstance(field.default, str) or field.name == "name" else type(field.default)
        )
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=kind,
            required=field.name == "name",
            default=None if field.name == "name" else field.default,
        )
    args = parser.parse_args()
    config = StudyConfig(**{name: getattr(args, name) for name in StudyConfig.__dataclass_fields__})
    train(config, args.data, args.output, args.initialize, args.teachers)


if __name__ == "__main__":
    main()
