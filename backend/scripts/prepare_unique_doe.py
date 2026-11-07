"""Create and independently verify one physically realizable four-plane DOE reference."""

import argparse
import hashlib
import json
import math
import platform
import zipfile
from pathlib import Path

import numpy as np
import torch
from doe.loss import reconstruction_loss
from doe.metrics import evaluate
from doe.physics import (
    DISTANCES_M,
    PITCH_M,
    SIZE,
    WAVELENGTH_M,
    FresnelPropagator,
    png_to_gray,
    png_to_phase,
)
from doe.synthesis import export_phase, optimize_phase
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

VERSION = "freeform-caustic-reference-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs/unique-doe-gold-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def phase_surface(parameters):
    """Analytic full-aperture lens plus astigmatism, trefoil and two phase gratings."""
    axis = np.arange(SIZE, dtype=np.float64) - SIZE // 2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    u, v = x / (SIZE / 2), y / (SIZE / 2)
    return (
        -np.pi * (x * x + y * y) * PITCH_M**2 / (WAVELENGTH_M * parameters["focus_m"])
        + parameters["trefoil"] * (u**3 - 3 * u * v * v)
        + parameters["astigmatism"] * (u * u - v * v)
        + parameters["grating_sin"] * np.sin(5 * u + 2 * v)
        + parameters["grating_cos"] * np.cos(3 * u - 6 * v)
        + parameters["tilt_x"] * u
        + parameters["tilt_y"] * v
    )


def propagate_numpy(phase, *, direct=False):
    """Use complex128 and either separable direct Fresnel sums or a chirped FFT."""
    axis = np.arange(SIZE, dtype=np.float64) - SIZE // 2
    source = np.exp(1j * phase)
    amplitudes = []
    for z in DISTANCES_M:
        pitch = WAVELENGTH_M * z / (SIZE * PITCH_M)
        if direct:
            kernel = np.exp(
                1j
                * np.pi
                * (axis[:, None] * pitch - axis[None, :] * PITCH_M) ** 2
                / (WAVELENGTH_M * z)
            )
            field = kernel @ source @ kernel.T
        else:
            radius = axis[:, None] ** 2 + axis[None, :] ** 2
            chirp = np.exp(1j * np.pi * radius * PITCH_M**2 / (WAVELENGTH_M * z))
            field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(source * chirp)))
        amplitudes.append(np.abs(field) * PITCH_M**2 / (WAVELENGTH_M * z))
    return np.stack(amplitudes)


def normalized(amplitudes):
    return amplitudes / amplitudes.max(axis=(-2, -1), keepdims=True)


def target_pixels(phase):
    values = normalized(propagate_numpy(phase))
    return np.rint(values * 255).clip(0, 255).astype(np.uint8)


def select_design(seed, candidates):
    """Choose spatially rich, distinct planes before encoding the reference DOE."""
    rng = np.random.default_rng(seed)
    best = None
    trials = []
    for index in range(candidates):
        parameters = {
            "focus_m": float(rng.uniform(0.0248, 0.0270)),
            "trefoil": float(rng.uniform(65, 115)),
            "astigmatism": float(rng.uniform(18, 45)),
            "grating_sin": float(rng.uniform(16, 30)),
            "grating_cos": float(rng.uniform(12, 28)),
            "tilt_x": float(rng.uniform(-10, 10)),
            "tilt_y": float(rng.uniform(-10, 10)),
        }
        phase = phase_surface(parameters)
        pixels = target_pixels(phase)
        correlation = np.corrcoef(pixels.reshape(4, -1))
        max_correlation = float(correlation[np.triu_indices(4, 1)].max())
        contrasts = pixels.std(axis=(-2, -1))
        fill = (pixels > 24).mean(axis=(-2, -1))
        entropy = []
        for image in pixels:
            frequencies = np.bincount(image.ravel(), minlength=256) / image.size
            nonzero = frequencies[frequencies > 0]
            entropy.append(float(-(nonzero * np.log2(nonzero)).sum()))
        valid = bool(contrasts.min() >= 20 and fill.min() >= 0.3)
        score = -max_correlation + 0.02 * min(entropy)
        trial = {
            "index": index,
            "parameters": parameters,
            "eligible": valid,
            "score": score,
            "maximum_pairwise_pcc": max_correlation,
            "entropy_bits": entropy,
            "contrast": contrasts.tolist(),
            "fill_gt_24": fill.tolist(),
        }
        trials.append(trial)
        if valid and (best is None or score > best[0]):
            best = (score, phase, pixels, trial)
    require(best is not None, "no sufficiently diverse candidate found")
    return best[1:], trials


def read_targets(output):
    return torch.stack(
        [png_to_gray((output / f"target-{i}.png").read_bytes()) for i in range(1, 5)]
    )


def additional_metrics(reconstruction, targets):
    local = [
        float(structural_similarity(y, x, data_range=1.0))
        for x, y in zip(reconstruction, targets, strict=True)
    ]
    mse = ((reconstruction - targets) ** 2).mean(axis=(-2, -1))
    psnr = [-10 * math.log10(float(value)) if value else None for value in mse]
    return {"local_ssim_7x7": local, "psnr_db": psnr, "mse": mse.tolist()}


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        require(
            hashlib.sha256((output / name).read_bytes()).hexdigest() == expected,
            f"checksum mismatch: {name}",
        )
    targets = read_targets(output)
    original_targets = target_pixels(phase_surface(manifest["design"]["parameters"]))
    require(
        np.array_equal(original_targets, targets.mul(255).round().numpy().astype(np.uint8)),
        "targets do not match the frozen continuous design",
    )
    phase = png_to_phase((output / "doe.png").read_bytes())
    optics = FresnelPropagator()
    _, reconstructed_pngs, metrics = export_phase(phase, targets, optics)
    reconstructions = torch.stack([png_to_gray(png) for png in reconstructed_pngs])
    for index, png in enumerate(reconstructed_pngs, 1):
        actual = (output / f"reconstruction-{index}.png").read_bytes()
        require(
            torch.equal(png_to_gray(actual), png_to_gray(png)),
            f"saved reconstruction mismatch: plane {index}",
        )
    direct = propagate_numpy(phase.double().numpy(), direct=True)
    fft = propagate_numpy(phase.double().numpy())
    relative_error = float(np.max(np.abs(direct - fft)) / fft.max())
    require(relative_error < 1e-10, "independent direct-sum and FFT Fresnel operators disagree")
    direct_pixels = np.rint(normalized(direct) * 255).clip(0, 255) / 255
    png_delta = float(np.abs(direct_pixels - reconstructions.numpy()).max())
    require(png_delta <= 1 / 255 + 1e-7, "PyTorch and independent NumPy reconstruction differ")
    require(
        metrics["worst_pcc"] >= 0.9 and metrics["worst_ssim"] >= 0.9,
        "requested minimum quality not reached",
    )
    auxiliary = additional_metrics(reconstructions.numpy(), targets.numpy())
    require(min(auxiliary["local_ssim_7x7"]) >= 0.9, "local SSIM below 0.9")
    direct_scores = evaluate(torch.from_numpy(direct_pixels), targets)
    intensity_reconstruction = normalized(direct**2)
    intensity_target = targets.double().numpy() ** 2
    return {
        "project_metrics": metrics,
        **auxiliary,
        "independent_direct_sum_metrics": direct_scores,
        "direct_sum_vs_numpy_fft_relative_amplitude_error": relative_error,
        "direct_sum_vs_pytorch_max_png_difference": png_delta,
        "intensity_metrics_against_squared_amplitude_targets": evaluate(
            torch.from_numpy(intensity_reconstruction), torch.from_numpy(intensity_target)
        ),
        "all_planes_above_0_9_pcc_global_and_local_ssim": True,
    }


def preview(output):
    sheet = Image.new("RGB", (4 * 256, 2 * 282 + 34), "#151a24")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "Fixed targets (top) / reconstruction from 8-bit DOE (bottom)", fill="white")
    for row, kind in enumerate(("target", "reconstruction")):
        for column, z in enumerate(DISTANCES_M):
            with Image.open(output / f"{kind}-{column + 1}.png") as image:
                sheet.paste(
                    image.resize((256, 256), Image.Resampling.NEAREST),
                    (column * 256, 34 + row * 282),
                )
            draw.text(
                (column * 256 + 10, 294 + row * 282),
                f"{kind} {column + 1} | {z * 1000:g} mm",
                fill="white",
            )
    sheet.save(output / "preview.png")


def build(output, seed, candidates, steps):
    require(not output.exists(), f"output already exists: {output}")
    require(seed >= 0 and candidates > 0, "seed must be nonnegative and candidates positive")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (phase, pixels, design), trials = select_design(seed, candidates)
    output.mkdir(parents=True)
    for index, image in enumerate(pixels, 1):
        Image.fromarray(image).save(output / f"target-{index}.png")
    targets = read_targets(output)
    optics = FresnelPropagator()
    phase_png, pngs, before = export_phase(torch.from_numpy(phase), targets, optics)
    initial = png_to_phase(phase_png)

    def objective(amplitude, expected):
        return reconstruction_loss(amplitude, expected, pcc_weight=1.0, ssim_weight=1.0)

    optimized = optimize_phase(
        targets, optics, iterations=steps, lr=0.001, initial_phase=initial, objective=objective
    )
    candidate_png, candidate_images, after = export_phase(optimized.phase, targets, optics)
    baseline_score = min(before["worst_pcc"], before["worst_ssim"])
    refined_score = min(after["worst_pcc"], after["worst_ssim"])
    accepted = refined_score > baseline_score
    if accepted:
        phase_png, pngs = candidate_png, candidate_images
    (output / "doe.png").write_bytes(phase_png)
    for index, png in enumerate(pngs, 1):
        (output / f"reconstruction-{index}.png").write_bytes(png)
    names = ["doe.png"] + [
        f"{kind}-{i}.png" for kind in ("target", "reconstruction") for i in range(1, 5)
    ]
    manifest = {
        "version": VERSION,
        "method": "forward_design_from_continuous_analytic_phase_then_8bit_DOE_export",
        "target_origin": "numpy complex128 Fresnel FFT of the continuous analytic phase",
        "independent_arbitrary_images": False,
        "purpose": "physically realizable supervised reference and end-to-end optics sanity check",
        "sets": 1,
        "planes": 4,
        "shape": [4, SIZE, SIZE],
        "png_mode": "L",
        "bit_depth": 8,
        "split": "standalone_reference_keep_out_of_existing_validation_test_and_OOD",
        "target_quantity": "per_plane_peak_normalized_field_amplitude_abs_U",
        "phase_mapping": "phase_radians = png_code * 2*pi/255 - pi",
        "optics": {
            "wavelength_m": WAVELENGTH_M,
            "input_pitch_m": PITCH_M,
            "distances_m": DISTANCES_M,
            "output_pitch_m": [WAVELENGTH_M * z / (SIZE * PITCH_M) for z in DISTANCES_M],
            "illumination": "uniform amplitude one, normal plane wave",
            "aperture": "128x128 square, no padding",
        },
        "seed": seed,
        "candidates": candidates,
        "design": design,
        "pairwise_target_pcc": np.corrcoef(pixels.reshape(4, -1)).tolist(),
        "phase_label": "best available known analytic design, finite precision export",
        "refinement": {
            "steps": steps,
            "learning_rate": 0.001,
            "accepted": accepted,
            "selection": "maximize minimum of worst-plane exported PCC and global SSIM",
            "before": before,
            "candidate_after": after,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "files_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = verify(output)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "design-search.json").write_text(
        json.dumps(trials, indent=2) + "\n", encoding="utf-8"
    )
    preview(output)
    print(json.dumps(report, indent=2))


def package(output):
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{output.name}/{path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "package"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    if args.command == "build":
        require(args.steps >= 0, "steps must be nonnegative")
        build(args.output, args.seed, args.candidates, args.steps)
    elif args.command == "verify":
        print(json.dumps(verify(args.output), indent=2))
    else:
        verify(args.output)
        package(args.output)


if __name__ == "__main__":
    main()
