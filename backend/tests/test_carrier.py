import pytest
import torch
from fastapi.testclient import TestClient

from doe.api import create_app
from doe.models import CHECKPOINT_FORMAT, FresnelUNet, load_model
from doe.models.unet import CARRIER_ARCHITECTURE
from doe.physics import gray_to_png
from doe.training.dataset import synthetic_targets


def checkpoint(model):
    return {
        "format_version": CHECKPOINT_FORMAT,
        "architecture": model.architecture,
        "width": model.width,
        "projection_steps": model.projection_steps,
        "amplitude_floor": model.amplitude_floor,
        "carrier_conditioning": {"scale": model.carrier_scale},
        "step": 1,
        "model": model.state_dict(),
    }


@pytest.mark.parametrize("scale", [0, 1])
def test_carrier_context_preserves_initial_phase_and_receives_gradients(scale):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    original = FresnelUNet(width=4, projection_steps=1)
    conditioned = FresnelUNet(
        width=4, projection_steps=1, carrier_conditioning=True, carrier_scale=scale
    )
    state = original.state_dict()
    first_weight = torch.zeros_like(conditioned.encode1[0].weight)
    first_weight[:, :4].copy_(state["encode1.0.weight"])
    state["encode1.0.weight"] = first_weight
    conditioned.load_state_dict(state, strict=True)
    target = synthetic_targets(0, split="train").unsqueeze(0)
    with torch.no_grad():
        expected = original(target)
    actual = conditioned(target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    gradient = conditioned.encode1[0].weight.grad[:, 4:]
    assert torch.isfinite(gradient).all()
    assert bool(torch.count_nonzero(gradient)) == bool(scale)
    assert all(parameter.grad is not None for parameter in conditioned.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in conditioned.parameters())


@pytest.mark.parametrize("scale", [0, 1])
def test_carrier_checkpoint_loads_with_standard_loader(tmp_path, scale):
    torch.set_num_threads(2)
    model = FresnelUNet(width=4, projection_steps=1, carrier_conditioning=True, carrier_scale=scale)
    target = synthetic_targets(1, split="train").unsqueeze(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(target).square().mean().backward()
    optimizer.step()
    path = tmp_path / "carrier.pt"
    torch.save(checkpoint(model), path)
    restored, version = load_model(path)
    assert version.startswith(f"{CARRIER_ARCHITECTURE}:")
    assert not restored.training
    assert restored.carrier_conditioning and restored.carrier_scale == scale
    with torch.inference_mode():
        torch.testing.assert_close(restored(target), model.eval()(target), rtol=0, atol=0)


@pytest.mark.parametrize(
    "conditioning", [None, {}, {"scale": True}, {"scale": 2}, {"scale": 1, "unknown": 1}]
)
def test_carrier_checkpoint_rejects_invalid_conditioning(tmp_path, conditioning):
    model = FresnelUNet(width=4, carrier_conditioning=True)
    state = checkpoint(model)
    state["carrier_conditioning"] = conditioning
    path = tmp_path / "invalid.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_carrier_checkpoint_requires_conditioning_metadata(tmp_path):
    model = FresnelUNet(width=4, carrier_conditioning=True)
    state = checkpoint(model)
    del state["carrier_conditioning"]
    path = tmp_path / "missing.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_legacy_architecture_rejects_carrier_metadata(tmp_path):
    model = FresnelUNet(width=4)
    path = tmp_path / "mislabeled.pt"
    torch.save(checkpoint(model), path)
    with pytest.raises(ValueError):
        load_model(path)


def test_carrier_api_synthesis_calls_model_once_without_refinement(tmp_path):
    model = FresnelUNet(width=4, projection_steps=1, carrier_conditioning=True)
    path = tmp_path / "carrier-api.pt"
    torch.save(checkpoint(model), path)
    app = create_app(path, device="cpu")
    target = synthetic_targets(0, split="demo")
    files = [("images", ("target.png", gray_to_png(plane), "image/png")) for plane in target]
    with TestClient(app) as client:
        calls = []
        hook = app.state.model.register_forward_hook(lambda *_: calls.append(1))
        try:
            response = client.post("/api/synthesize", files=files, data={"method": "neural"})
        finally:
            hook.remove()
        assert response.status_code == 200, response.text
        result = response.json()
        assert calls == [1]
        assert result["iterations"] == 0
        assert result["model_version"].startswith(f"{CARRIER_ARCHITECTURE}:")
        assert len(result["reconstruction_pngs"]) == 4


@pytest.mark.parametrize("scale", [-1, 2, True, 1.0])
def test_carrier_scale_must_be_a_supported_integer(scale):
    with pytest.raises(ValueError):
        FresnelUNet(width=4, carrier_conditioning=True, carrier_scale=scale)
