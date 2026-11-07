"""Classical, neural and hybrid synthesis using one export and scoring path."""

from dataclasses import dataclass
from math import isfinite, pi
from time import perf_counter

import torch
from torch import Tensor, nn

from doe.loss import normalize_amplitude, reconstruction_loss
from doe.metrics import evaluate
from doe.physics import (
    FresnelPropagator,
    gray_to_png,
    phase_to_png,
    png_to_gray,
    png_to_phase,
    quantize_phase,
)

CLASSICAL_STEPS = 200
HYBRID_STEPS = 20
LEARNING_RATE = 0.08


@dataclass(frozen=True)
class Optimization:
    phase: Tensor
    loss: float
    history: tuple[tuple[float, ...], ...]


@torch.enable_grad()
def optimize_phase(
    target: Tensor,
    propagator: FresnelPropagator,
    *,
    iterations: int,
    lr: float = LEARNING_RATE,
    seed: int = 0,
    restarts: int = 1,
    patience: int | None = None,
    initial_phase: Tensor | None = None,
) -> Optimization:
    """Optimize phase with Adam; rank candidates at their exported phase levels."""
    if iterations < 0 or restarts < 1 or not isfinite(lr) or lr <= 0:
        raise ValueError("Expected nonnegative iterations, positive learning rate and restarts")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive")
    if initial_phase is not None and initial_phase.shape != target.shape[-2:]:
        raise ValueError("Initial phase must match the target grid")
    rng = torch.Generator(device=target.device).manual_seed(seed)
    run_candidates, run_losses, history = [], [], []
    for restart in range(restarts):
        initial = (
            initial_phase
            if restart == 0 and initial_phase is not None
            else torch.rand(target.shape[-2:], device=target.device, generator=rng) * (2 * pi) - pi
        )
        phase = initial.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam([phase], lr=lr)
        best_phase = phase.detach()
        best_loss = target.new_tensor(torch.inf)
        losses = []
        last_improvement, previous = 0, float("inf")
        for step in range(iterations + 1):
            # Straight-through derivative of export quantization.
            quantized = quantize_phase(phase.detach())
            candidate = quantized + (phase - phase.detach())
            loss = reconstruction_loss(propagator.amplitude(candidate), target)
            detached = loss.detach()
            losses.append(detached)
            better = detached < best_loss
            best_phase = torch.where(better, quantized, best_phase)
            best_loss = torch.minimum(best_loss, detached)
            if patience is not None:
                current = detached.item()
                if current < previous:
                    last_improvement, previous = step, current
                if step - last_improvement >= patience:
                    break
            if step < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        run_candidates.append(best_phase)
        run_losses.append(best_loss)
        history.append(tuple(torch.stack(losses).tolist()))
    best = min(range(restarts), key=lambda i: run_losses[i].item())
    return Optimization(run_candidates[best], run_losses[best].item(), tuple(history))


@dataclass(frozen=True)
class Synthesis:
    phase_png: bytes
    reconstruction_pngs: tuple[bytes, ...]
    metrics: dict
    seconds: float
    iterations: int


def synthesize(
    targets: Tensor,
    method: str,
    propagator: FresnelPropagator,
    *,
    model: nn.Module | None = None,
    iterations: int | None = None,
    seed: int = 0,
) -> Synthesis:
    """Synthesize a DOE, then reconstruct strictly from the encoded phase PNG."""
    if method not in ("classical", "neural", "hybrid"):
        raise ValueError(f"Unknown synthesis method: {method}")
    if method != "classical" and model is None:
        raise ValueError(f"{method} requires a trained model")
    if targets.shape != (len(propagator.distances_m), propagator.size, propagator.size):
        raise ValueError("Targets must match the number of planes and DOE grid")
    if not torch.isfinite(targets).all() or torch.any((targets < 0) | (targets > 1)):
        raise ValueError("Targets must be finite and lie in [0, 1]")
    device = propagator.input_axis_m.device
    target = targets.detach().to(device=device, dtype=torch.float32)
    if iterations is None:
        iterations = {"classical": CLASSICAL_STEPS, "neural": 0, "hybrid": HYBRID_STEPS}[method]
    if iterations < 0 or (method == "neural" and iterations):
        raise ValueError("Iterations must be nonnegative, and zero for Neural")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    phase = None
    if method != "classical":
        with torch.no_grad():
            phase = model(target[None])[0, 0].float()
    if method != "neural":
        phase = optimize_phase(
            target, propagator, iterations=iterations, seed=seed, initial_phase=phase
        ).phase
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    phase_png = phase_to_png(phase)
    with torch.no_grad():
        decoded = png_to_phase(phase_png, propagator.size).to(device)
        amplitude = normalize_amplitude(propagator.amplitude(decoded)).cpu()
    reconstructions = tuple(gray_to_png(plane) for plane in amplitude)
    pixels = torch.stack([png_to_gray(png, propagator.size) for png in reconstructions])
    metrics = evaluate(pixels, target, propagator.distances_m)
    return Synthesis(phase_png, reconstructions, metrics, seconds, iterations)
