from pathlib import Path

from fastapi.testclient import TestClient

from doe.api import create_app


def test_static_frontend_does_not_shadow_api_routes(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text('<div id="root">Photonica</div>', encoding="utf-8")
    (tmp_path / "asset.js").write_text("export default 1", encoding="utf-8")
    monkeypatch.setenv("DOE_STATIC_DIR", str(tmp_path))
    with TestClient(create_app(tmp_path / "missing.pt", device="cpu")) as client:
        assert "Photonica" in client.get("/").text
        assert client.get("/asset.js").status_code == 200
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/model/info").json()["available"] is False
        assert client.get("/api/ready").status_code == 503
        assert client.get("/api/nonexistent").status_code == 404
        assert client.get("/docs").status_code == 200


def test_ready_requires_and_identifies_bundled_production_model(monkeypatch):
    monkeypatch.delenv("DOE_STATIC_DIR", raising=False)
    checkpoint = Path(__file__).resolve().parents[2] / "artifacts/doe-carrier-physics-teacher-v1.pt"
    with TestClient(create_app(checkpoint, device="cpu")) as client:
        ready = client.get("/api/ready")
        assert ready.status_code == 200
        assert ready.json()["model_version"] == client.get("/api/model/info").json()["version"]
        assert ready.json()["model_version"].startswith("fresnel-unet-carrier-v1:")
