"""Versioned model loading for synthesis and validation."""

from hashlib import file_digest
from pathlib import Path

import torch

from .unet import ARCHITECTURE, FresnelUNet

CHECKPOINT_FORMAT = 1


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[FresnelUNet, str]:
    """Load a trained model and return its content-based version identifier."""
    with Path(path).open("rb") as stream:
        version = file_digest(stream, "sha256").hexdigest()[:12]
        stream.seek(0)
        data = torch.load(stream, map_location="cpu", weights_only=True)
    if (
        not isinstance(data, dict)
        or data.get("format_version") != CHECKPOINT_FORMAT
        or data.get("architecture") != ARCHITECTURE
        or not isinstance(data.get("step"), int)
        or data["step"] < 1
    ):
        raise ValueError("checkpoint is not a supported trained DOE model")

    try:
        model = FresnelUNet(width=data["width"])
        model.load_state_dict(data["model"], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError("checkpoint has invalid model parameters") from exc
    return model.to(device).eval(), f"{ARCHITECTURE}:{version}"
