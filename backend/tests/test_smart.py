import base64
import time
from copy import deepcopy
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from doe import smart
from doe.api import create_app
from doe.metrics import evaluate, windowed_ssim
from doe.models import DEFAULT_CHECKPOINT
from doe.physics import gray_to_png, png_to_gray
from doe.training.dataset import synthetic_targets


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DOE_SMART_MAX_CANDIDATES", "4")
    root = Path(__file__).resolve().parents[2]
    with TestClient(create_app(root / DEFAULT_CHECKPOINT, "cpu")) as connection:
        yield connection


def inputs(client):
    originals = client.get("/api/demo/0").json()["images"]
    files = [("images", ("target.png", base64.b64decode(image))) for image in originals]
    result = client.post("/api/synthesize", files=files, data={"method": "neural"}).json()
    phase = ("phase", ("doe.png", base64.b64decode(result["phase_png"])))
    fields = {"model_version": result["model_version"], "baseline_seconds": result["seconds"]}
    return originals, [*files, phase], fields, result


def wait(client, identifier):
    for _ in range(300):
        status = client.get("/api/smart/" + identifier).json()
        if status["status"] in smart.TERMINAL:
            return client.get("/api/smart/" + identifier + "/result").json()
        time.sleep(0.02)
    raise AssertionError("Smart job did not finish")


def test_selection_guards_each_original_plane_and_baseline():
    targets = synthetic_targets(0)
    metrics = evaluate(targets * 0.9, targets)
    policy = smart.Policy()
    assert not policy.accepts(metrics, metrics)
    candidate = deepcopy(metrics)
    for key in ("mean_pcc", "mean_ssim", "worst_pcc", "worst_ssim"):
        candidate[key] += 0.01
    assert policy.accepts(candidate, metrics)
    candidate["planes"][0]["ssim"] -= 0.01
    assert not policy.accepts(candidate, metrics)


@pytest.mark.parametrize("index", range(4))
def test_preprocessing_is_bounded_reproducible_and_keeps_plane_order(index):
    originals = synthetic_targets(index)
    saved = originals.clone()
    report = smart.analyze(originals)
    configs = smart.candidate_configs(report, 20)
    assert configs[0] == {"operations": []}
    assert len(configs) <= 20
    assert {operation["method"] for config in configs for operation in config["operations"]} == {
        "gamma",
        "brightness",
        "contrast",
        "denoise",
        "frequency",
    }
    for config in configs:
        adapted = smart.preprocess(originals, config)
        assert adapted.shape == originals.shape
        assert torch.isfinite(adapted).all() and adapted.min() >= 0 and adapted.max() <= 1
        assert (adapted - originals).abs().mean() < 0.1
        torch.testing.assert_close(adapted, smart.preprocess(originals, config), rtol=0, atol=0)
    torch.testing.assert_close(saved, originals, rtol=0, atol=0)
    torch.testing.assert_close(smart.preprocess(originals, configs[0]), originals)
    assert len(report["similarity"]) == 6
    assert report["similarity_pairs"][-1] == [3, 4]


def test_windowed_ssim_population_gaussian_protocol():
    targets = synthetic_targets(2)
    assert windowed_ssim(targets, targets) == pytest.approx([1] * 4)
    assert min(windowed_ssim(targets * 0.7, targets)) < 0.99


def test_smart_scores_originals_from_quantized_png_and_limits_forwards(client):
    originals, files, fields, baseline = inputs(client)
    start = client.post("/api/smart", files=files, data=fields)
    assert start.status_code == 202, start.text
    report = wait(client, start.json()["id"])
    assert report["status"] == "completed", report.get("error")
    assert report["checked"] == report["total"] == 4
    assert report["forward_count"] == 3
    assert report["originals"] == originals
    assert report["candidates"][0]["run"]["phase_png"] == baseline["phase_png"]
    target = torch.stack([png_to_gray(base64.b64decode(image)) for image in originals])
    for candidate in report["candidates"]:
        run = candidate["run"]
        pixels = torch.stack(
            [png_to_gray(base64.b64decode(image)) for image in run["reconstruction_pngs"]]
        )
        assert run["metrics"] == evaluate(pixels, target)
        assert run["iterations"] == 0
        assert png_to_gray(base64.b64decode(run["phase_png"])).shape == (128, 128)
        assert len(candidate["windowed_ssim"]) == len(candidate["background_power"]) == 4
    best = report["candidates"][report["best_index"]]["run"]["metrics"]
    first = report["candidates"][0]["run"]["metrics"]
    assert best == first or smart.Policy().accepts(best, first)
    candidate = client.get("/api/smart/" + report["id"] + "/candidates/1")
    assert candidate.json()["inputs"] == report["candidates"][1]["inputs"]


def test_cancel_releases_shared_capacity_and_allows_restart(client, monkeypatch):
    _, files, fields, _ = inputs(client)
    entered, finish = Event(), Event()
    analyze = smart.analyze

    def hold(target):
        entered.set()
        assert finish.wait(10)
        return analyze(target)

    monkeypatch.setattr(smart, "analyze", hold)
    first = client.post("/api/smart", files=files, data=fields)
    identifier = first.json()["id"]
    try:
        assert entered.wait(5)
        assert client.post("/api/smart", files=files, data=fields).status_code == 429
        assert (
            client.post("/api/synthesize", files=files[:4], data={"method": "neural"}).status_code
            == 429
        )
        assert client.get("/api/smart/" + identifier + "/result").status_code == 409
        assert client.post("/api/smart/" + identifier + "/cancel").json()["cancel_requested"]
    finally:
        finish.set()
    assert wait(client, identifier)["status"] == "cancelled"
    second = client.post("/api/smart", files=files, data={**fields, "mode": "adapt"})
    assert second.status_code == 202
    assert wait(client, second.json()["id"])["status"] == "completed"


def test_no_improvement_keeps_exact_baseline_and_errors_are_reported(client, monkeypatch):
    _, files, fields, baseline = inputs(client)
    from doe.synthesis import Synthesis

    fixed = Synthesis(
        base64.b64decode(baseline["phase_png"]),
        tuple(base64.b64decode(image) for image in baseline["reconstruction_pngs"]),
        baseline["metrics"],
        0.01,
        0,
    )
    monkeypatch.setattr(smart, "synthesize", lambda *args, **kwargs: fixed)
    start = client.post("/api/smart", files=files, data=fields)
    report = wait(client, start.json()["id"])
    assert not report["improved"] and report["best_index"] == 0
    assert report["candidates"][0]["run"]["phase_png"] == baseline["phase_png"]
    assert (
        client.post(
            "/api/smart", files=files, data={**fields, "model_version": "other"}
        ).status_code
        == 409
    )

    def fail(*args):
        raise RuntimeError("Unexpected model failure")

    monkeypatch.setattr(smart, "analyze", fail)
    task = client.post("/api/smart", files=files, data=fields).json()
    assert wait(client, task["id"])["status"] == "error"
    assert (
        client.post("/api/synthesize", files=files[:4], data={"method": "neural"}).status_code
        == 200
    )


def test_empty_and_uniform_images_have_finite_preprocessing():
    targets = torch.stack([torch.full((128, 128), level) for level in (0.0, 0.25, 0.5, 1.0)])
    for config in smart.candidate_configs(smart.analyze(targets), 20):
        result = smart.preprocess(targets, config)
        assert np.isfinite(result.numpy()).all()
        for plane in result:
            assert len(gray_to_png(plane)) < 64 * 1024
