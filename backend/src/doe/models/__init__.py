"""Versioned model loading for synthesis and validation."""

from hashlib import file_digest
from pathlib import Path

import torch

from .unet import (
    ARCHITECTURE,
    CARRIER_ARCHITECTURE,
    DEFAULT_AMPLITUDE_FLOOR,
    PROJECTED_ARCHITECTURE,
    FresnelUNet,
)

CHECKPOINT_FORMAT = 1
DEFAULT_CHECKPOINT = Path("artifacts/doe-carrier-physics-teacher-v1.pt")


def load_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[FresnelUNet, str]:
    """Load a trained model and return its content-based version identifier."""
    with Path(path).open("rb") as stream:
        version = file_digest(stream, "sha256").hexdigest()[:12]
        stream.seek(0)
        data = torch.load(stream, map_location="cpu", weights_only=True)
    if (
        not isinstance(data, dict)
        or data.get("format_version") != CHECKPOINT_FORMAT
        or data.get("architecture")
        not in (
            ARCHITECTURE,
            PROJECTED_ARCHITECTURE,
            CARRIER_ARCHITECTURE,
        )
        or not isinstance(data.get("step"), int)
        or data["step"] < 1
    ):
        raise ValueError("checkpoint is not a supported trained DOE model")

    try:
        if "fourier_bottleneck" in data:
            raise ValueError("Fourier bottleneck checkpoints are unsupported")
        conditioned = data["architecture"] == CARRIER_ARCHITECTURE
        scale = 1
        if conditioned:
            conditioning = data["carrier_conditioning"]
            if (
                not isinstance(conditioning, dict)
                or set(conditioning) != {"scale"}
                or type(conditioning["scale"]) is not int
                or conditioning["scale"] not in (0, 1)
            ):
                raise ValueError("checkpoint has invalid carrier conditioning metadata")
            scale = conditioning["scale"]
        elif "carrier_conditioning" in data:
            raise ValueError("carrier conditioning requires the carrier architecture")
        model = FresnelUNet(
            width=data["width"],
            projection_steps=data.get("projection_steps", 0),
            amplitude_floor=data.get("amplitude_floor", DEFAULT_AMPLITUDE_FLOOR),
            carrier_conditioning=conditioned,
            carrier_scale=scale,
        )
        if model.architecture != data["architecture"]:
            raise ValueError("checkpoint architecture and projection count disagree")
        model.load_state_dict(data["model"], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError("checkpoint has invalid model parameters") from exc
    return model.to(device).eval(), f"{model.architecture}:{version}"
