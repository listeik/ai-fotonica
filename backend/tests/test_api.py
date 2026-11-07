import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from doe import api
from doe.models.residual_fourier import ARCHITECTURE as FOURIER_ARCHITECTURE


@pytest.fixture
def client(tmp_path):
    with TestClient(api.create_app(tmp_path / "missing.pt", device="cpu")) as test_client:
        yield test_client


def _upload(size=128, mode="L"):
    out = BytesIO()
    Image.new(mode, (size, size), 192).save(out, format="PNG")
    return ("target.png", out.getvalue(), "image/png")


def _uploads(count=4):
    return [("images", _upload()) for _ in range(count)]


def test_default_fourier_checkpoint_synthesizes_with_one_forward(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.delenv("DOE_CHECKPOINT", raising=False)
    app = api.create_app(device="cpu")
    with TestClient(app) as client:
        info = client.get("/api/model/info").json()
        assert info["available"] and info["status"] == "ready"
        assert info["version"] == f"{FOURIER_ARCHITECTURE}:38cf98d0ceb5"
        calls = []
        hook = app.state.model.register_forward_hook(lambda *_: calls.append(1))
        try:
            demo = client.get("/api/demo/0").json()["images"]
            files = [
                ("images", ("demo.png", base64.b64decode(image), "image/png")) for image in demo
            ]
            response = client.post("/api/synthesize", files=files, data={"method": "neural"})
        finally:
            hook.remove()
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["model_version"] == info["version"]
        assert result["iterations"] == 0
        assert calls == [1]
        assert len(result["reconstruction_pngs"]) == 4
        for encoded in (result["phase_png"], *result["reconstruction_pngs"]):
            with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
                assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))


def test_api_synthesizes_an_exportable_phase_and_four_reconstructions(client):
    response = client.post("/api/synthesize", files=_uploads(), data={"iterations": 2})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["method"] == "classical"
    assert result["iterations"] == 2
    assert result["model_version"] is None
    assert result["seconds"] >= 0
    assert result["parameters"]["distances_mm"] == [24.1, 25.3, 26.5, 27.7]
    assert len(result["metrics"]["planes"]) == 4
    for plane in result["metrics"]["planes"]:
        assert -1 <= plane["pcc"] <= 1
        assert -1 <= plane["ssim"] <= 1
    assert len(result["reconstruction_pngs"]) == 4
    for encoded in (result["phase_png"], *result["reconstruction_pngs"]):
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))


def test_demo_images_pass_strict_upload_validation(client):
    demo = client.get("/api/demo/0")
    assert demo.status_code == 200
    images = demo.json()["images"]
    assert len(images) == 4
    files = [("images", ("demo.png", base64.b64decode(encoded), "image/png")) for encoded in images]
    result = client.post("/api/synthesize", files=files, data={"iterations": 1})
    assert result.status_code == 200, result.text
    assert client.get("/api/demo/3").status_code == 200
    assert client.get("/api/demo/4").status_code == 404


def test_upload_count_and_image_format_are_validated(client):
    assert client.post("/api/synthesize", files=_uploads(3)).status_code == 422
    assert client.post("/api/synthesize", files=_uploads(5)).status_code == 422
    files = _uploads()
    files[1] = ("images", _upload(mode="RGB"))
    result = client.post("/api/synthesize", files=files)
    assert result.status_code == 422
    assert result.json()["detail"].startswith("Plane 2:")
    assert (
        client.post(
            "/api/synthesize", files=_uploads(), data={"iterations": api.MAX_ITERATIONS + 1}
        ).status_code
        == 422
    )


def test_missing_checkpoint_disables_neural_and_hybrid(client):
    assert client.get("/api/health").json()["status"] == "ok"
    model = client.get("/api/model/info").json()
    assert model == {
        "available": False,
        "version": None,
        "status": "missing",
        "device": "cpu",
    }
    methods = client.get("/api/methods").json()["methods"]
    assert {method["id"]: method["available"] for method in methods} == {
        "classical": True,
        "neural": False,
        "hybrid": False,
    }
    for method in ("neural", "hybrid"):
        response = client.post("/api/synthesize", files=_uploads(), data={"method": method})
        assert response.status_code == 503
        assert "checkpoint" in response.json()["detail"]


def test_upload_size_is_bounded_per_file_and_across_stream(client):
    files = _uploads()
    files[0] = ("images", ("large.png", b"x" * (api.MAX_FILE_BYTES + 1), "image/png"))
    assert client.post("/api/synthesize", files=files).status_code == 413
    chunk = b"x" * (api.MAX_BODY_BYTES // 2)
    response = client.post("/api/synthesize", content=iter((chunk, chunk, b"more")))
    assert response.status_code == 413


def test_second_synthesis_is_rejected_during_a_run(client, monkeypatch):
    started, finish = Event(), Event()
    actual_synthesize = api.synthesize

    def hold_first(*args, **kwargs):
        started.set()
        if not finish.wait(timeout=10):
            raise RuntimeError("Concurrent request test did not release synthesis")
        return actual_synthesize(*args, **kwargs)

    monkeypatch.setattr(api, "synthesize", hold_first)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post, "/api/synthesize", files=_uploads(), data={"iterations": 1}
        )
        try:
            assert started.wait(timeout=10)
            response = client.post("/api/synthesize", files=_uploads(), data={"iterations": 1})
            assert response.status_code == 429
        finally:
            finish.set()
        assert future.result(timeout=10).status_code == 200
