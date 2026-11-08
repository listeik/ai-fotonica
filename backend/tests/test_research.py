import base64
from io import BytesIO
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from doe.api import create_app
from doe.api.research import descriptors
from doe.models import CHECKPOINT_FORMAT, DEFAULT_CHECKPOINT, FresnelUNet
from doe.physics import png_to_gray


@pytest.fixture
def client(tmp_path, monkeypatch):
    weights = Path(__file__).resolve().parents[2] / DEFAULT_CHECKPOINT
    monkeypatch.chdir(tmp_path)
    DEFAULT_CHECKPOINT.parent.mkdir()
    copyfile(weights, DEFAULT_CHECKPOINT)
    with TestClient(create_app(DEFAULT_CHECKPOINT, device="cpu")) as test_client:
        yield test_client


def encode(image, format="PNG"):
    stream = BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def demo_uploads(client):
    demo = client.get("/api/demo/0").json()["images"]
    return [("images", ("target.png", base64.b64decode(image), "image/png")) for image in demo]


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_preprocessing_preserves_proportions_and_fits_challenge(client, format):
    data = encode(Image.new("RGB", (320, 80), "white"), format)
    result = client.post("/api/images/preprocess", files={"image": ("test", data)})
    assert result.status_code == 200, result.text
    assert result.json()["converted"]
    gray = png_to_gray(base64.b64decode(result.json()["png"]))
    assert gray[64, 64] == 1
    assert gray[0, 0] == 0


def test_preprocessing_preserves_canonical_png_and_rejects_broken_files(client):
    data = encode(Image.new("L", (128, 128), 192))
    result = client.post("/api/images/preprocess", files={"image": ("target.png", data)})
    assert result.status_code == 200
    assert not result.json()["converted"]
    assert base64.b64decode(result.json()["png"]) == data
    assert (
        client.post("/api/images/preprocess", files={"image": ("bad.jpg", b"broken")}).status_code
        == 422
    )


def test_preprocessing_scales_16_bit_grayscale(client):
    data = encode(Image.fromarray(np.full((128, 128), 32768, dtype=np.uint16)))
    result = client.post("/api/images/preprocess", files={"image": ("target.png", data)})
    assert result.status_code == 200
    gray = png_to_gray(base64.b64decode(result.json()["png"]))
    assert torch.allclose(gray, torch.full_like(gray, 128 / 255))


def test_descriptors_distinguish_flat_and_fine_patterns():
    flat = descriptors(np.full((128, 128), 0.5, dtype=np.float32))
    assert set(flat.values()) == {0.0}
    stripes = np.tile(np.arange(128, dtype=np.float32) % 2, (128, 1))
    detail = descriptors(stripes)
    assert detail["contrast"] == 0.5
    assert detail["entropy_bits"] == 1
    assert detail["spectral_centroid"] == 0.5
    assert detail["high_frequency_fraction"] == 1


def test_analysis_reports_reference_bands_per_set(client):
    response = client.post("/api/analysis", files=demo_uploads(client))
    assert response.status_code == 200, response.text
    report = response.json()
    assert len(report["planes"]) == 4
    assert report["reference"]["sets"] == 32
    assert report["planes"][0].keys() == report["reference"]["bands"].keys()
    assert all(low <= median <= high for low, median, high in report["reference"]["bands"].values())


def test_research_comparison_preserves_active_checkpoint_and_explains_exported_phase(client):
    uploads = demo_uploads(client)
    active = client.get("/api/model/info").json()
    assert {"carrier"} == {item["id"] for item in client.get("/api/research/options").json()}
    model = FresnelUNet(width=4, projection_steps=1)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT,
            "architecture": model.architecture,
            "width": model.width,
            "projection_steps": model.projection_steps,
            "step": 1,
            "model": model.state_dict(),
        },
        DEFAULT_CHECKPOINT.with_name("doe-unet.pt"),
    )
    assert {"unet", "carrier"} == {
        item["id"] for item in client.get("/api/research/options").json()
    }
    for selection in (["unet", "unet"], ["unet", "../secret"]):
        invalid = client.post(
            "/api/research/compare", files=uploads, data={"architectures": selection}
        )
        assert invalid.status_code == 422
    comparison = client.post(
        "/api/research/compare", files=uploads, data={"architectures": ["unet", "carrier"]}
    )
    assert comparison.status_code == 200, comparison.text[:500]
    baseline, carrier = comparison.json()
    assert baseline["checkpoint"]["architecture"] != carrier["checkpoint"]["architecture"]
    assert carrier["checkpoint"]["version"] == active["version"]
    assert carrier["parameters"]["learning_rate"] is None
    assert client.get("/api/model/info").json() == active
    assert client.get("/api/models").status_code == 404
    assert client.get("/api/benchmarks").status_code == 404

    phase = ("phase", ("doe.png", base64.b64decode(baseline["phase_png"]), "image/png"))
    response = client.post("/api/research/diagnostics", files=[*uploads, phase])
    assert response.status_code == 200, response.text[:500]
    planes = response.json()["planes"]
    assert [plane["distance_mm"] for plane in planes] == [24.1, 25.3, 26.5, 27.7]
    for plane, reconstruction, (_, target) in zip(planes, baseline["reconstruction_pngs"], uploads):
        residual = (png_to_gray(base64.b64decode(reconstruction)) - png_to_gray(target[1])).abs()
        assert plane["mae"] == pytest.approx(residual.mean().item())
        assert 0 <= plane["power_in_target"] <= 1
        for key in ("error_png", "intensity_png"):
            assert png_to_gray(base64.b64decode(plane[key])).shape == (128, 128)
